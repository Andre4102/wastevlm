#!/bin/bash
# Submit the scaled stage-2 (finetune_next, ~819K, general-purpose) as a resumable
# chain. One epoch (~6.4k steps) exceeds the 16h wall, so:
#   job1  fresh
#   job2  afterany:job1   RESUME=auto   (job1 TIMES OUT -> non-zero, so afterANY,
#                                        not afterok, or SLURM would cancel job2)
#   job3  afternotok:job2 RESUME=auto   (safety net; auto-cancelled if job2 finishes)
# Each job restores optimizer+scheduler+step from the newest checkpoint, so the
# cosine LR is one continuous curve across the whole chain.
set -euo pipefail
cd /leonardo/home/userexternal/adiecidu/scripts/wastevlm

MIX=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data/alignment/normalized/sft_mix.jsonl
if [ ! -s "$MIX" ]; then
  echo "ERROR: $MIX missing/empty — build the mix first"; exit 1
fi
echo "[chain] sft_mix records: $(wc -l < "$MIX")"

J1=$(sbatch --parsable slurm_vlm_modern.sh finetune_next)
echo "[chain] job1 (fresh)            = $J1"
J2=$(sbatch --parsable --dependency=afterany:$J1 --export=ALL,RESUME=auto \
      slurm_vlm_modern.sh finetune_next)
echo "[chain] job2 (afterany:$J1)  = $J2"
J3=$(sbatch --parsable --dependency=afternotok:$J2 --export=ALL,RESUME=auto \
      slurm_vlm_modern.sh finetune_next)
echo "[chain] job3 (afternotok:$J2) = $J3"
echo "[chain] SUBMITTED chain=$J1,$J2,$J3"
echo "$J1 $J2 $J3" > /leonardo/home/userexternal/adiecidu/.claude/jobs/aa757775/tmp/finetune_next_chain.txt
