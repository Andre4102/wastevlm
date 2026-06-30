#!/bin/bash
#SBATCH --job-name=dinoseg_tta_eval           # overridable via `sbatch --job-name`
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT=/home/ids/diecidue/scripts/waste_vlm
RESULTS=/home/ids/diecidue/results/waste_vlm
PYTHON=/home/ids/diecidue/miniconda3/envs/waste_vlm/bin/python
cd "$PROJECT"

# positional args: OUTNAME [GRID]
#   OUTNAME : run dir under $RESULTS containing best.pt
#   GRID    : N for N×N non-overlapping crops (default 2)
OUTNAME=${1:?out dir name}
GRID=${2:-2}

echo "[slurm] host=$(hostname)  job=${SLURM_JOB_ID:-?}  tta=${OUTNAME} grid=${GRID}  start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export HF_HUB_OFFLINE=1
"$PYTHON" -m src.seg_eval_tta \
  --checkpoint "$RESULTS/$OUTNAME/best.pt" \
  --tta-grid "$GRID" --split test \
  --out-json "$RESULTS/$OUTNAME/test_eval_tta${GRID}.json"

echo "[slurm] done=$(date -Is)"
