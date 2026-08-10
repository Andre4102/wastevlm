#!/bin/bash
#SBATCH --job-name=qvl_mask_eval
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Masked-eval a per-Q-head prune checkpoint via the `qwen2_5vl_masked` adapter
# (stock Qwen2.5-VL base + wrapped decoder masks + trained LoRA + mask logits,
# NO materialize). Requires myenv (transformers 4.57 + pruning stack).
# Usage: sbatch slurm_vlm_eval_masked.sh <CKPT_DIR> [DATASET] [PROMPT] [LIMIT]
set -euo pipefail
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
ENV=${ENV:-myenv}
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/$ENV/bin
cd "$PROJECT"

CKPT_DIR=${1:?prune ckpt dir with lora_adapter/ + mask_logits.pt}
DATASET=${2:-dw_paper10}
PROMPT_STYLE=${3:-closed_vocab}
LIMIT=${4:-0}
TAG=$(basename "$CKPT_DIR")
RESULTS=$WROOT/results/vlm_eval/masked_${TAG}_${DATASET}_${PROMPT_STYLE}

export WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} start=$(date -Is)"
echo "[slurm] ckpt=$CKPT_DIR dataset=$DATASET prompt=$PROMPT_STYLE limit=$LIMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
mkdir -p "$RESULTS"
"$PYBIN/python" -m src.vlm_eval \
  --model qwen2_5vl_masked \
  --ckpt "$CKPT_DIR" \
  --task classify \
  --dataset "$DATASET" \
  --prompt-style "$PROMPT_STYLE" \
  --split test --limit "$LIMIT" \
  --out-json "$RESULTS/test_eval.json" \
  --save-raw "$RESULTS/raw_responses.jsonl"
echo "[slurm] done=$(date -Is)"
