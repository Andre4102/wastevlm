#!/bin/bash
#SBATCH --job-name=vlm_pretok
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=lrd_all_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=02:00:00

# CPU-only: pre-tokenize both LLaVA stages into memory-mappable caches so training
# skips the 558K-record JSON parse (per rank) and per-batch tokenization. The cache
# is Qwen-tokenizer-specific, so it is shared by every encoder (radio-l, dinov3-*).
#   sbatch slurm_pretokenize.sh
set -euo pipefail

PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
ENV=waste_vlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
DATA=$WROOT/data
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/$ENV/bin
cd "$PROJECT"

export WASTE_VLM_WEIGHTS=$WROOT/weights
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} start=$(date -Is)"

# Stage 1 (LCS-558K) and stage 2 (LLaVA-Instruct-150K).
"$PYBIN/python" -m src.pretokenize_vlm \
  --train "$DATA/llava_pretrain/blip_laion_cc_sbu_558k.json" \
  --image-root "$DATA/llava_pretrain/images" \
  --out "$DATA/llava_pretrain/token_cache" --workers 4

"$PYBIN/python" -m src.pretokenize_vlm \
  --train "$DATA/llava_instruct/llava_instruct_150k.json" \
  --image-root "$DATA/coco/train2017" \
  --out "$DATA/llava_instruct/token_cache" --workers 4

echo "[slurm] done=$(date -Is)"
