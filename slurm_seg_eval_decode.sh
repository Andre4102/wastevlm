#!/bin/bash
#SBATCH --job-name=dinoseg_decode            # overridable via `sbatch --job-name`
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT=/home/ids/diecidue/scripts/waste_vlm
RESULTS=/home/ids/diecidue/results/waste_vlm
PYTHON=/home/ids/diecidue/miniconda3/envs/waste_vlm/bin/python
cd "$PROJECT"

# positional args: OUTNAME BATCH   (checkpoint = $RESULTS/$OUTNAME/best.pt)
OUTNAME=${1:?out dir name}; BATCH=${2:-16}

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} decode=$OUTNAME bs=$BATCH start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export HF_HUB_OFFLINE=1
"$PYTHON" -m src.seg_eval_decode \
  --checkpoint "$RESULTS/$OUTNAME/best.pt" \
  --split test --batch-size "$BATCH" --num-workers 8 \
  --ws-min-distance 5 7 11 15 \
  --out-json "$RESULTS/$OUTNAME/decode_sweep.json"

echo "[slurm] done=$(date -Is)"
