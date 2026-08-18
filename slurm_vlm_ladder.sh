#!/bin/bash
#SBATCH --job-name=vlm_ladder
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00

# Scaffolded naming: presence -> appearance -> material name, each rung carried
# into the next, scored both through the keyword parser and as per-category
# Yes/No margins.
#
# Usage:  SPLIT=test sbatch slurm_vlm_ladder.sh <ENCODER> <CKPT_DIR> [DATASET]
set -euo pipefail

PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
ENV=${ENV:-waste_vlm}
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/$ENV/bin
cd "$PROJECT"

ENCODER=${1:?encoder id}
CKPT_DIR=${2:?ckpt dir}
DATASET=${3:-aw_m2}
SPLIT=${SPLIT:-test}
TAG=${TAG:-$SPLIT}
OUT=$WROOT/results/vlm_eval/ladder_$(basename "$CKPT_DIR")_${DATASET}_${TAG}.json

export WASTE_VLM_WEIGHTS=$WROOT/weights
export DINOV3_REPO=$WROOT/dinov3_repo
export WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} ckpt=$CKPT_DIR split=$SPLIT"
"$PYBIN/python" scripts/ladder_probe.py --generate \
  --ckpt "$CKPT_DIR" --encoder "$ENCODER" \
  --image-size 768 --pixel-shuffle 2 \
  --dataset "$DATASET" --split "$SPLIT" --limit "${LIMIT:-0}" \
  --train-limit "${TRAIN_LIMIT:-1200}" --out "$OUT"
echo "[slurm] done=$(date -Is)"
