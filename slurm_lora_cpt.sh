#!/bin/bash
#SBATCH --job-name=lora_cpt_qwenvl
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1                 # 1 H100 (80GB) is plenty for a 7B bf16 LoRA on a small corpus
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

PROJECT=/home/ids/diecidue/scripts/waste_vlm
RESULTS=/home/ids/diecidue/results/waste_vlm
PYTHON=/home/ids/diecidue/miniconda3/envs/waste_vlm/bin/python
cd "$PROJECT"

# positional args: OUTNAME [MAX_STEPS]   (MAX_STEPS>0 => smoke test)
OUTNAME=${1:?out dir name}; MAX_STEPS=${2:--1}

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} out=$OUTNAME max_steps=$MAX_STEPS start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export HF_HUB_OFFLINE=1   # Qwen2.5-7B-Instruct weights are local
"$PYTHON" -m src.lora_cpt_llm \
  --out "$RESULTS/$OUTNAME" \
  --max-steps "$MAX_STEPS"

echo "[slurm] done=$(date -Is)"
