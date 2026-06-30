#!/bin/bash
#SBATCH --job-name=dinoseg_hr1036_eval     # Name of your job
#SBATCH --output=logs/%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=logs/%x_%j.err             # Error file
#SBATCH --partition=L40S
#SBATCH --nodelist=nodemm06
#SBATCH --gres=gpu:1                       # Request 1 GPU
#SBATCH --cpus-per-task=8                  # Request 8 CPU cores
#SBATCH --mem=32G
#SBATCH --time=02:00:00                    # Time limit for the job (hh:mm:ss)

set -euo pipefail

PROJECT=/home/ids/diecidue/scripts/waste_vlm
RESULTS=/home/ids/diecidue/results/waste_vlm
PYTHON=/home/ids/diecidue/miniconda3/envs/waste_vlm/bin/python
cd "$PROJECT"

echo "[slurm] host=$(hostname)  job=${SLURM_JOB_ID:-?}  start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# DinoSeg @ 1036² (HR DINOv2-B) — test-set COCO mAP + mIoU.
"$PYTHON" -m src.seg_eval \
  --checkpoint "$RESULTS/dinoseg_dw_hr1036/best.pt" \
  --split test \
  --batch-size 8 \
  --num-workers 8 \
  --out-json "$RESULTS/dinoseg_dw_hr1036/test_eval.json"

echo "[slurm] done=$(date -Is)"
