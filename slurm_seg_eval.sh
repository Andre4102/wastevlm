#!/bin/bash
#SBATCH --job-name=dinoseg_eval              # overridable via `sbatch --job-name`
#SBATCH --output=logs/%x_%j.out              # Output file (%x job name, %j job ID)
#SBATCH --error=logs/%x_%j.err               # Error file
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --cpus-per-task=8                    # Request 8 CPU cores
#SBATCH --mem=32G                            # CPU RAM cap
#SBATCH --time=02:00:00                      # Time limit (hh:mm:ss)

set -euo pipefail

PROJECT=/home/ids/diecidue/scripts/waste_vlm
RESULTS=/home/ids/diecidue/results/waste_vlm
PYTHON=/home/ids/diecidue/miniconda3/envs/waste_vlm/bin/python
cd "$PROJECT"

# positional args: OUTNAME BATCH   (checkpoint = $RESULTS/$OUTNAME/best.pt)
OUTNAME=${1:?out dir name}; BATCH=${2:?batch size}

echo "[slurm] host=$(hostname)  job=${SLURM_JOB_ID:-?}  eval=$OUTNAME bs=$BATCH  start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export HF_HUB_OFFLINE=1   # DINOv3 weights are local
"$PYTHON" -m src.seg_eval \
  --checkpoint "$RESULTS/$OUTNAME/best.pt" \
  --split test --batch-size "$BATCH" --num-workers 8 \
  --out-json "$RESULTS/$OUTNAME/test_eval.json"

echo "[slurm] done=$(date -Is)"
