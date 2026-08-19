#!/bin/bash
#SBATCH --job-name=sam3_obj
#SBATCH --output=logs/sam3obj_%j.out
#SBATCH --error=logs/sam3obj_%j.err
#SBATCH --account=iscrc_fiche
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=8 --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
cd "$PROJECT"
DATASET=${1:-dronewaste}
export WASTE_VLM_WEIGHTS=$WROOT/weights DINOV3_REPO=$WROOT/dinov3_repo
export WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "[slurm] $(hostname) job=${SLURM_JOB_ID:-?} dataset=$DATASET"
/leonardo_scratch/large/userexternal/adiecidu/envs/waste_sam3/bin/python \
  scripts/sam3_objectness.py --dataset "$DATASET" --limit "${LIMIT:-200}" \
  --threshold "${THR:-0.3}" --out-json "$WROOT/results/sam3_obj_${DATASET}.json"
echo "[slurm] done=$(date -Is)"
