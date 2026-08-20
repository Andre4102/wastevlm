#!/bin/bash
#SBATCH --job-name=dinoseg_eval              # overridable via `sbatch --job-name`
#SBATCH --output=logs/%x_%j.out              # Output file (%x job name, %j job ID)
#SBATCH --error=logs/%x_%j.err               # Error file
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --cpus-per-task=8                    # Request 8 CPU cores
#SBATCH --mem=64G                            # CPU RAM cap
#SBATCH --time=02:00:00                      # Time limit (hh:mm:ss)

set -euo pipefail

PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
RESULTS=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/results/seg
PYTHON=${PYTHON:-/leonardo/home/userexternal/adiecidu/miniconda3/envs/waste_vlm/bin/python}
cd "$PROJECT"
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
export WASTE_DATA_ROOT=$WROOT/data WASTE_VLM_WEIGHTS=$WROOT/weights
export DINOV3_REPO=$WROOT/dinov3_repo
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export TOKENIZERS_PARALLELISM=false

# positional args: OUTNAME BATCH   (checkpoint = $RESULTS/$OUTNAME/best.pt)
OUTNAME=${1:?out dir name}; BATCH=${2:?batch size}

echo "[slurm] host=$(hostname)  job=${SLURM_JOB_ID:-?}  eval=$OUTNAME bs=$BATCH  start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export HF_HUB_OFFLINE=1   # DINOv3 weights are local
"$PYTHON" -m src.seg_eval \
  --checkpoint "$RESULTS/$OUTNAME/best.pt" \
  --split test --batch-size "$BATCH" --num-workers 8 \
  ${SITE_HOLDOUT:+--site-holdout "$SITE_HOLDOUT"} \
  --out-json "$RESULTS/$OUTNAME/test_eval.json"

echo "[slurm] done=$(date -Is)"
