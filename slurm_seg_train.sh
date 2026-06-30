#!/bin/bash
#SBATCH --job-name=dinoseg_train            # overridable via `sbatch --job-name`
#SBATCH --output=logs/%x_%j.out             # Output file (%x job name, %j job ID)
#SBATCH --error=logs/%x_%j.err              # Error file
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --cpus-per-task=8                   # Request 8 CPU cores
#SBATCH --mem=32G                           # CPU RAM cap (GPU VRAM is the real constraint)
#SBATCH --time=08:00:00                     # Time limit (hh:mm:ss)

set -euo pipefail

PROJECT=/home/ids/diecidue/scripts/waste_vlm
RESULTS=/home/ids/diecidue/results/waste_vlm
PYTHON=/home/ids/diecidue/miniconda3/envs/waste_vlm/bin/python
cd "$PROJECT"

# positional args: BACKBONE_TYPE IMAGE_SIZE EPOCHS BATCH LR OUTNAME [HEAD] [BACKBONE_ID] [MULTI_BLOCK] [MULTILABEL] [FPN_BLOCKS] [FPN_DIM] [FPN_MERGE]
#   MULTI_BLOCK : comma-separated ViT block indices for Phase-2 FPN-lite (e.g. "3,7,11").
#   MULTILABEL  : "1" for per-class sigmoid head; empty/"0" = single-label softmax.
#   FPN_BLOCKS  : comma-separated ViT block indices for FPN head (e.g. "2,5,8,11").
#   FPN_DIM     : FPN pathway width (default 256).
#   FPN_MERGE   : "add" (default) or "concat" (channel-concat before 3×3 conv).
BTYPE=${1:?backbone type}; ISIZE=${2:?image size}; EPOCHS=${3:?epochs}
BATCH=${4:?batch}; LR=${5:?lr}; OUTNAME=${6:?out dir name}
HEAD=${7:-linear}; BACKBONE_ID=${8:-}; MULTI_BLOCK=${9:-}; MULTILABEL=${10:-}
FPN_BLOCKS=${11:-}; FPN_DIM=${12:-}; FPN_MERGE=${13:-}

echo "[slurm] host=$(hostname)  job=${SLURM_JOB_ID:-?}  start=$(date -Is)"
echo "[slurm] btype=$BTYPE id=${BACKBONE_ID:-default} size=$ISIZE epochs=$EPOCHS batch=$BATCH lr=$LR head=$HEAD multi_block=${MULTI_BLOCK:-none} multilabel=${MULTILABEL:-0} fpn_blocks=${FPN_BLOCKS:-default} fpn_dim=${FPN_DIM:-default} fpn_merge=${FPN_MERGE:-add} out=$OUTNAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export HF_HUB_OFFLINE=1   # DINOv3 / RADIO weights are local; no HF round-trip
EXTRA_ARGS=()
if [ -n "$BACKBONE_ID" ]; then EXTRA_ARGS+=(--backbone-id "$BACKBONE_ID"); fi
if [ -n "$MULTI_BLOCK" ]; then EXTRA_ARGS+=(--multi-block "$MULTI_BLOCK"); fi
if [ "$MULTILABEL" = "1" ]; then EXTRA_ARGS+=(--multilabel); fi
if [ -n "$FPN_BLOCKS" ]; then EXTRA_ARGS+=(--fpn-blocks "$FPN_BLOCKS"); fi
if [ -n "$FPN_DIM" ];    then EXTRA_ARGS+=(--fpn-dim "$FPN_DIM"); fi
if [ -n "$FPN_MERGE" ];  then EXTRA_ARGS+=(--fpn-merge "$FPN_MERGE"); fi
"$PYTHON" -m src.seg_train \
  --backbone-type "$BTYPE" \
  "${EXTRA_ARGS[@]}" \
  --epochs "$EPOCHS" --batch-size "$BATCH" --lr "$LR" \
  --image-size "$ISIZE" --head "$HEAD" --bg-weight 0.1 --num-workers 8 \
  --out "$RESULTS/$OUTNAME" --log-every 50

echo "[slurm] done=$(date -Is)"
