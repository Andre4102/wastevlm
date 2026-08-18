#!/bin/bash
#SBATCH --job-name=probe_sweep
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00

# Frozen-feature probe at several input resolutions. This is the direct test of
# "the sub-token problem is the projector, not the encoder": if detection and
# naming both improve as the dense map gets finer, the grounded architecture has
# headroom the LLaVA path cannot reach.
#
# Usage:  sbatch slurm_probe_sweep.sh <ENCODER> <VERSION> <IMAGE_SIZE> <TASK>
set -euo pipefail
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/${ENV:-waste_vlm}/bin
cd "$PROJECT"
ENCODER=${1:-cradiov4-so}; VERSION=${2:-m2}; IMG=${3:-768}; TASK=${4:-naming}
BS=${BS:-8}
export WASTE_VLM_WEIGHTS=$WROOT/weights DINOV3_REPO=$WROOT/dinov3_repo
export WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[slurm] $(hostname) job=${SLURM_JOB_ID:-?} enc=$ENCODER ver=$VERSION img=$IMG task=$TASK bs=$BS"
if [ "$TASK" = "naming" ]; then
  "$PYBIN/python" scripts/aw_naming_probe.py --encoder "$ENCODER" --version "$VERSION" \
    --image-size "$IMG" --batch-size "$BS" \
    --out-json "$WROOT/results/probe_naming_${ENCODER}_${VERSION}_${IMG}.json"
else
  "$PYBIN/python" scripts/aw_feature_probe.py --encoder "$ENCODER" --version "$VERSION" \
    --image-size "$IMG" --batch-size "$BS" \
    --out-json "$WROOT/results/probe_binary_${ENCODER}_${VERSION}_${IMG}.json"
fi
echo "[slurm] done=$(date -Is)"
