#!/bin/bash
#SBATCH --job-name=aw_nameprobe
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Is the material category in the frozen features, or not in the imagery at all?
# Usage:  sbatch slurm_aw_naming_probe.sh [m2|m4] [IMAGE_SIZE]
set -euo pipefail
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/${ENV:-waste_vlm}/bin
cd "$PROJECT"
VERSION=${1:-m2}
IMG=${2:-768}
export WASTE_VLM_WEIGHTS=$WROOT/weights
export DINOV3_REPO=$WROOT/dinov3_repo
export WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} version=$VERSION img=$IMG"
"$PYBIN/python" scripts/aw_naming_probe.py --version "$VERSION" --image-size "$IMG" \
  --out-json "$WROOT/results/aw_naming_probe_${VERSION}_${IMG}.json"
echo "[slurm] done=$(date -Is)"
