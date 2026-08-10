#!/bin/bash
#SBATCH --job-name=vlm_align
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=12:00:00

# Act-2 alignment-budget probe: train the RADIO-frozen connector on ONE matched-
# budget data arm. Connector-only (LLM frozen, stage 1) so the only thing learning
# is the projector -- exactly the LLaVA-1.5 alignment stage. Everything except the
# arm's data (LR, batch, accum, warmup, projector arch, max-steps) is held fixed
# across arms so faithfulness differences are attributable to the data alone.
#
# Arms: density | diversity | spatial | combined  (built by scripts/build_arms.py)
#
# Dry-run (doc step 5, 50 steps, 1 GPU):
#   sbatch --qos=boost_qos_dbg --time=00:30:00 --gres=gpu:1 --cpus-per-task=8 --mem=64G \
#          slurm_vlm_align_arm.sh density radio-l 50
# Full arm:
#   sbatch slurm_vlm_align_arm.sh density radio-l
#   sbatch slurm_vlm_align_arm.sh diversity radio-l
#   sbatch slurm_vlm_align_arm.sh spatial radio-l
#   sbatch slurm_vlm_align_arm.sh combined radio-l
set -euo pipefail

PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
ENV=waste_vlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
ALIGN=$WROOT/data/alignment
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/$ENV/bin
cd "$PROJECT"

ARM=${1:-density}               # density | diversity | spatial | combined
ENCODER=${2:-radio-l}
MAX_STEPS=${3:-2000}            # SAME across arms; dry-run pass e.g. 50
ETAG=${ENCODER//\//_}

export WASTE_VLM_WEIGHTS=$WROOT/weights
export DINOV3_REPO=$WROOT/dinov3_repo
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
# Heterogeneous arms (Vision-Flan / PixMo) mix in near-max-len (2048-tok) samples;
# a batch with several spikes activation memory and OOMs at bs=8 even with grad
# checkpointing. expandable_segments defrags the 6+ GiB of reserved-but-unallocated
# cache that pushed it over on a 64 GB A100.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARM_JSON=$ALIGN/normalized/arm_${ARM}.jsonl
IMG_ROOT=$ALIGN/normalized
[ -f "$ARM_JSON" ] || { echo "missing arm file: $ARM_JSON (run scripts/build_arms.py)"; exit 1; }
# same-length assertion (doc step 4): all arms must be token-matched already
N=$(wc -l < "$ARM_JSON")

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
LAUNCH="$PYBIN/torchrun --standalone --nproc_per_node=$NGPU -m src.vlm_train"

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} start=$(date -Is)"
echo "[slurm] arm=$ARM encoder=$ENCODER ngpu=$NGPU max_steps=$MAX_STEPS n_records=$N"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Connector-only alignment stage; identical hypers across arms, data is the only
# variable. global batch = bs(4) * accum(16) * ngpu(4) = 256 (LLaVA-1.5 pretrain);
# bs lowered 8->4 (accum raised to match) so long-sequence batches fit in 64 GB.
$LAUNCH --stage pretrain --encoder "$ENCODER" \
  --train "$ARM_JSON" --image-root "$IMG_ROOT" \
  --out-dir "$WROOT/results/vlm/align_${ARM}_${ETAG}" \
  --max-steps "$MAX_STEPS" --batch-size 4 --grad-accum 16 \
  --warmup-ratio 0.03 --save-steps 500

echo "[slurm] done=$(date -Is)"
