#!/bin/bash
#SBATCH --job-name=vlm_occ
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00

# Remove the annotated waste and ask again, against a matched control edit.
# Usage:  sbatch slurm_vlm_occlusion.sh <ENCODER> <CKPT_DIR> [DATASET]
set -euo pipefail
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/${ENV:-waste_vlm}/bin
cd "$PROJECT"
ENCODER=${1:?encoder}; CKPT_DIR=${2:?ckpt}; DATASET=${3:-aw_m2}
OUT=$WROOT/results/vlm_eval/occ_$(basename "$CKPT_DIR")_${DATASET}.json
export WASTE_VLM_WEIGHTS=$WROOT/weights DINOV3_REPO=$WROOT/dinov3_repo
export WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} ckpt=$CKPT_DIR"
"$PYBIN/python" scripts/occlusion_probe.py --generate \
  --ckpt "$CKPT_DIR" --encoder "$ENCODER" --image-size 768 --pixel-shuffle 2 \
  --dataset "$DATASET" --question "${QUESTION:-categories}" --out "$OUT"
echo "[slurm] done=$(date -Is)"
