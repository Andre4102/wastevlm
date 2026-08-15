#!/bin/bash
# Submit one stage-2 arm as a resumable single-node chain.
#
#   ./submit_arm_chain.sh <arm> [n_jobs] [hours]
#
# One node (4x A100), global batch 128, ~16.1 s/step at 768px/ps2. A mix bigger
# than a wall-clock therefore runs as a chain:
#   job1            fresh
#   job2..N         afterany:prev  RESUME=auto
#   safety          afternotok:last RESUME=auto  (auto-cancelled if the chain ends clean)
#
# afterANY, not afterok: job1 TIMES OUT by design, which SLURM reports as failure
# and which would cancel an afterok dependant. Each job restores
# optimizer+scheduler+step+sampler_offset from the newest checkpoint, and --epochs 1
# over a fixed mix gives the SAME total_steps in every link, so the cosine LR stays
# one continuous curve across the whole chain rather than restarting its warmup.
#
# The partition allows 24h (MaxTime=1-00:00:00), so default to 24h walls: fewer
# links means fewer restarts, and each restart risks up to SAVE_STEPS of lost work.
set -euo pipefail
cd /leonardo/home/userexternal/adiecidu/scripts/wastevlm

ARM=${1:?arm name, e.g. a1 / pilot / b}
NJOBS=${2:-2}
HOURS=${3:-24}
DATA=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data
MIX="$DATA/train_mixes/mix_${ARM}.jsonl"

if [ ! -s "$MIX" ]; then
  echo "ERROR: $MIX missing/empty — run scripts/build_train_mix.py --arm $ARM first" >&2
  exit 1
fi
N=$(wc -l < "$MIX")
# global batch 128; ~16.1 s/step measured at 768px/ps2 on 4x A100
STEPS=$(( N / 128 ))
EST=$(python3 -c "print(f'{$STEPS*16.1/3600:.1f}')")
echo "[chain] arm=$ARM  records=$N  ~${STEPS} steps  ~${EST}h total  jobs=$NJOBS x ${HOURS}h"

COMMON="ALL,IMG_SIZE=768,PSHUF=2,BS=4,ARM=$ARM,FT_NEXT_JSON=$MIX,FT_NEXT_IMG=$DATA/alignment/normalized,SAVE_STEPS=250"

J=$(sbatch --parsable --time=${HOURS}:00:00 --job-name=vlm_${ARM} \
      --export="$COMMON" slurm_vlm_modern.sh finetune_next)
echo "[chain] job1 (fresh)            = $J"
IDS="$J"
PREV=$J
for i in $(seq 2 "$NJOBS"); do
  J=$(sbatch --parsable --time=${HOURS}:00:00 --job-name=vlm_${ARM} \
        --dependency=afterany:$PREV --export="$COMMON,RESUME=auto" \
        slurm_vlm_modern.sh finetune_next)
  echo "[chain] job$i (afterany:$PREV) = $J"
  IDS="$IDS $J"
  PREV=$J
done
SAFE=$(sbatch --parsable --time=${HOURS}:00:00 --job-name=vlm_${ARM}_safety \
        --dependency=afternotok:$PREV --export="$COMMON,RESUME=auto" \
        slurm_vlm_modern.sh finetune_next)
echo "[chain] safety (afternotok:$PREV) = $SAFE"
echo "[chain] SUBMITTED $ARM: $IDS $SAFE"
echo "[chain] output -> \$WROOT/results/vlm/cradiov4-so_r768ps2_finetune_next_${ARM}/"
