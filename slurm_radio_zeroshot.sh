#!/bin/bash
#SBATCH --job-name=radio_zs
#SBATCH --output=logs/radio_zs_%j.out
#SBATCH --error=logs/radio_zs_%j.err
#SBATCH --account=iscrc_fiche
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00

# Zero-shot naming + dense open-vocab segmentation through C-RADIOv4's SigLIP2 head.
# Usage:  sbatch slurm_radio_zeroshot.sh [dronewaste|aw_m2] [image_size]
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
"$PYBIN/python" scripts/radio_zeroshot.py --dataset "$DATASET" --image-size "$IMG" \
  --modes ${MODES:-crop-summary roi-dense dense-seg} ${SITES:+--held-out-sites} \
  --out-json "$WROOT/results/radio_zs_${DATASET}_${IMG}.json"
echo "[slurm] done=$(date -Is)"
