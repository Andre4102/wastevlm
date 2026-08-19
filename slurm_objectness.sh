#!/bin/bash
#SBATCH --job-name=objness
#SBATCH --output=logs/objness_%j.out
#SBATCH --error=logs/objness_%j.err
#SBATCH --account=iscrc_fiche
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Can the encoder we already run produce the proposals?
# Usage:  sbatch slurm_objectness.sh [dronewaste|aw_m2] [image_size]
set -euo pipefail
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/${ENV:-waste_vlm}/bin
cd "$PROJECT"
DATASET=${1:-dronewaste}; IMG=${2:-640}
export WASTE_VLM_WEIGHTS=$WROOT/weights DINOV3_REPO=$WROOT/dinov3_repo
export WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "[slurm] $(hostname) job=${SLURM_JOB_ID:-?} dataset=$DATASET img=$IMG"
"$PYBIN/python" scripts/feature_objectness.py --dataset "$DATASET" --image-size "$IMG" \
  --limit "${LIMIT:-200}" --project "${TEACHER:-none}" \
  --out-json "$WROOT/results/objectness_${DATASET}_${IMG}_${TEACHER:-none}.json"
echo "[slurm] done=$(date -Is)"
