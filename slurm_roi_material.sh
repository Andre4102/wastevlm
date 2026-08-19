#!/bin/bash
#SBATCH --job-name=roi_mat
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

# Ceiling experiment: perfect localisation handed to the model as a crop.
# Usage:  CKPT=<dir> sbatch slurm_roi_material.sh [aw_m2|aw_m4]
set -euo pipefail
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/${ENV:-waste_vlm}/bin
cd "$PROJECT"
DATASET=${1:-aw_m2}
export WASTE_VLM_WEIGHTS=$WROOT/weights DINOV3_REPO=$WROOT/dinov3_repo
export WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
ARGS=""
[ -n "${CKPT:-}" ] && ARGS="--ckpt $CKPT"
echo "[slurm] $(hostname) job=${SLURM_JOB_ID:-?} dataset=$DATASET ckpt=${CKPT:-none}"
[ -n "${DEGRADE:-}" ] && ARGS="$ARGS --degrade $DEGRADE"
OUT=${OUT:-$WROOT/results/roi_material_${DATASET}.json}
"$PYBIN/python" scripts/roi_material.py --generate --resume --dataset "$DATASET" $ARGS \
  --contexts ${CONTEXTS:-0.0 0.5} --out "$OUT"
echo "[slurm] done=$(date -Is)"
