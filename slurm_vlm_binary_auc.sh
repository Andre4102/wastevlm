#!/bin/bash
#SBATCH --job-name=vlm_binauc
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

# Threshold-free readout of the binary waste decision: score Yes-vs-No logits
# at the first assistant token and sweep them for an AUC. See the module
# docstring of scripts/vlm_binary_auc.py for why this is the measurement that
# separates "not represented" from "represented, badly verbalised".
#
# Usage:  sbatch slurm_vlm_binary_auc.sh <ENCODER> <CKPT_DIR> [DATASET] [LIMIT]
# Smoke:  sbatch --qos=boost_qos_dbg --time=00:20:00 slurm_vlm_binary_auc.sh \
#            cradiov4-so <ckpt> aw_m2 20
set -euo pipefail

PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
ENV=${ENV:-waste_vlm}
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/$ENV/bin
cd "$PROJECT"

ENCODER=${1:?encoder id}
CKPT_DIR=${2:?ckpt dir with llm_merged/ + projector.pt}
DATASET=${3:-aw_m2}
LIMIT=${4:-0}
# MUST match how the ckpt was trained: projector input dim is patch_dim*PSHUF^2.
IMG_SIZE=${IMG_SIZE:-768}
PSHUF=${PSHUF:-2}

# FIT=1 also scores the train split and reports a threshold fitted there, which
# is the only number safe to quote: a cut chosen on test is an oracle.
FIT_ARGS=""
[ "${FIT:-0}" = "1" ] && FIT_ARGS="--fit-on-train"

STAGE=$(basename "$CKPT_DIR")
OUTNAME=${OUT_TAG:-binauc_${STAGE}_${DATASET}${FIT:+_fit}}
RESULTS=$WROOT/results/vlm_eval/$OUTNAME

export WASTE_VLM_WEIGHTS=$WROOT/weights
export DINOV3_REPO=$WROOT/dinov3_repo
export WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} start=$(date -Is)"
echo "[slurm] encoder=$ENCODER ckpt=$CKPT_DIR dataset=$DATASET limit=$LIMIT img=$IMG_SIZE pshuf=$PSHUF"

mkdir -p "$RESULTS"
"$PYBIN/python" scripts/vlm_binary_auc.py \
  --ckpt "$CKPT_DIR" \
  --encoder "$ENCODER" \
  --image-size "$IMG_SIZE" \
  --pixel-shuffle "$PSHUF" \
  --dataset "$DATASET" \
  --limit "$LIMIT" $FIT_ARGS ${LLM:+--llm "$LLM"} ${SITES:+--sites "$SITES"} ${SITES_ALL:+--sites-all-images} \
  --out-json "$RESULTS/binary_auc.json"

echo "[slurm] done=$(date -Is)"
