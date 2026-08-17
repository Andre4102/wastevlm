#!/bin/bash
#SBATCH --job-name=vlm_convo
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# Record real user<->model conversations for the thesis figures. Only the
# --generate half needs a GPU; rendering runs anywhere from the transcript JSON.
#
# Usage:  IDS=35,193,224 sbatch slurm_vlm_convo.sh <ENCODER> <CKPT_DIR> [DATASET] [OUT]
set -euo pipefail

PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
ENV=${ENV:-waste_vlm}
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/$ENV/bin
cd "$PROJECT"

ENCODER=${1:?encoder id}
CKPT_DIR=${2:?ckpt dir with llm_merged/ + projector.pt}
DATASET=${3:-aw_m2}
OUT=${4:-$WROOT/results/vlm_eval/convos_$(basename "$CKPT_DIR")_${DATASET}.json}
IMG_SIZE=${IMG_SIZE:-768}
PSHUF=${PSHUF:-2}

export WASTE_VLM_WEIGHTS=$WROOT/weights
export DINOV3_REPO=$WROOT/dinov3_repo
export WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

ARGS=""
[ -n "${IDS:-}" ] && ARGS="--ids $IDS"
[ -n "${CALIB:-}" ] && ARGS="$ARGS --calib $CALIB"
[ "${WITH_COMMIT:-1}" = "1" ] && ARGS="$ARGS --with-commit"

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} ckpt=$CKPT_DIR ds=$DATASET"
echo "[slurm] args: $ARGS"

"$PYBIN/python" scripts/make_convo.py --generate \
  --ckpt "$CKPT_DIR" --encoder "$ENCODER" \
  --image-size "$IMG_SIZE" --pixel-shuffle "$PSHUF" \
  --dataset "$DATASET" --out "$OUT" $ARGS

echo "[slurm] done=$(date -Is)"
