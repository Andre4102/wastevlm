#!/bin/bash
#SBATCH --job-name=roi_tok
#SBATCH --output=logs/roi_tok_%j.out
#SBATCH --error=logs/roi_tok_%j.err
#SBATCH --account=iscrc_fiche
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# Full image at native resolution, pooling only the tokens inside the object.
# Usage:  sbatch slurm_roi_token.sh [aw_m2|dronewaste] [image_size]
set -euo pipefail
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/${ENV:-waste_vlm}/bin
cd "$PROJECT"
DATASET=${1:-aw_m2}; IMG=${2:-1024}; ENCODER=${ENCODER:-cradiov4-so}
export WASTE_VLM_WEIGHTS=$WROOT/weights DINOV3_REPO=$WROOT/dinov3_repo
export WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "[slurm] $(hostname) job=${SLURM_JOB_ID:-?} dataset=$DATASET img=$IMG enc=$ENCODER pad=${PAD:-0.0}"
"$PYBIN/python" scripts/roi_token_probe.py --dataset "$DATASET" --image-size "$IMG" \
  --encoder "$ENCODER" \
  --pad "${PAD:-0.0}" --batch-size "${BS:-2}" ${SIGLIP2:+--siglip2} ${DUMP:+--dump-emb "$DUMP"} \
  --out-json "$WROOT/results/roi_token_${DATASET}_${IMG}_${ENCODER}${SIGLIP2:+_siglip2}.json"
echo "[slurm] done=$(date -Is)"
