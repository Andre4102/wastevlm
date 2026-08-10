#!/bin/bash
#SBATCH --job-name=vlm_eval_trained
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Waste-benchmark eval of OUR trained VLM (frozen encoder -> projector.pt ->
# merged Qwen2.5). Loads a finetune dir via the `waste_vlm` adapter in
# src.vlm_eval and runs the two-turn open_cot classify eval on DroneWaste.
#
# Usage:  sbatch slurm_vlm_eval_trained.sh <ENCODER> <CKPT_DIR> [DATASET] [LIMIT]
#   ENCODER   radio-l | dinov3-b | ...   (must match how the ckpt was trained)
#   CKPT_DIR  finetune output dir with llm_merged/ + projector.pt
#   DATASET   dw_paper10 (default; AerialWaste m2/m4 not yet available on Leonardo)
#   LIMIT     0 = full test split (default); >0 = smoke on first N images
#
# Smoke:  sbatch --qos=boost_qos_dbg --time=00:20:00 slurm_vlm_eval_trained.sh \
#            dinov3-b <ckpt> dw_paper10 20
set -euo pipefail

PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
# myenv (transformers 4.57 + pruning stack) is REQUIRED to eval a materialized
# custom-qwen2 pruned decoder; waste_vlm (4.49) is fine for stock llm_merged.
# Override with `ENV=myenv sbatch ...`.
ENV=${ENV:-waste_vlm}
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/$ENV/bin
cd "$PROJECT"

ENCODER=${1:?encoder id (radio-l|dinov3-b|...)}
CKPT_DIR=${2:?finetune ckpt dir with llm_merged/ + projector.pt}
DATASET=${3:-dw_paper10}
LIMIT=${4:-0}
PROMPT_STYLE=${5:-open_cot}   # open_cot | closed_vocab | open_caption
# MUST match how the ckpt was trained: the projector's input dim is
# patch_dim * PSHUF^2, so a mismatch fails loudly on load_state_dict.
# Defaults are the legacy 512/no-shuffle arms; the modern recipe is 768/2.
IMG_SIZE=${IMG_SIZE:-512}
PSHUF=${PSHUF:-1}
ETAG=${ENCODER//\//_}
OUTNAME=vlm_${ETAG}_r${IMG_SIZE}ps${PSHUF}_${DATASET}_${PROMPT_STYLE}
RESULTS=$WROOT/results/vlm_eval/$OUTNAME

export WASTE_VLM_WEIGHTS=$WROOT/weights
export DINOV3_REPO=$WROOT/dinov3_repo
export WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} start=$(date -Is)"
echo "[slurm] encoder=$ENCODER ckpt=$CKPT_DIR dataset=$DATASET limit=$LIMIT out=$OUTNAME"
echo "[slurm] img=$IMG_SIZE pshuf=$PSHUF"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

mkdir -p "$RESULTS"
"$PYBIN/python" -m src.vlm_eval \
  --model waste_vlm \
  --ckpt "$CKPT_DIR" \
  --encoder "$ENCODER" \
  --image-size "$IMG_SIZE" \
  --pixel-shuffle "$PSHUF" \
  --task classify \
  --dataset "$DATASET" \
  --prompt-style "$PROMPT_STYLE" \
  --split test --limit "$LIMIT" \
  --out-json "$RESULTS/test_eval.json" \
  --save-raw "$RESULTS/raw_responses.jsonl"

echo "[slurm] done=$(date -Is)"
