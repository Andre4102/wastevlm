#!/bin/bash
#SBATCH --job-name=attr_map
#SBATCH --output=logs/attr_%j.out
#SBATCH --error=logs/attr_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Attribution of the Yes/No margin onto the visual token grid.
# Usage:  CKPT=<dir> sbatch slurm_attr.sh [aw_m2|dronewaste] [ig|grad|occ]
set -euo pipefail
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/${ENV:-waste_vlm}/bin
cd "$PROJECT"
DATASET=${1:-aw_m2}; METHOD=${2:-ig}
export WASTE_VLM_WEIGHTS=$WROOT/weights DINOV3_REPO=$WROOT/dinov3_repo
export WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "[slurm] $(hostname) job=${SLURM_JOB_ID:-?} dataset=$DATASET method=$METHOD"
"$PYBIN/python" scripts/attribution_maps.py --ckpt "$CKPT" --dataset "$DATASET" \
  --method "$METHOD" --limit "${LIMIT:-40}" ${PERCAT:+--per-category} \
  --png-dir "$WROOT/results/attr_png/${DATASET}_${METHOD}" \
  --out-json "$WROOT/results/attr_${DATASET}_${METHOD}.json"
echo "[slurm] done=$(date -Is)"
