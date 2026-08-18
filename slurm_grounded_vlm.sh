#!/bin/bash
#SBATCH --job-name=gvlm
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00

# Baseline 2: grounded VLMs asked to point at waste, zero waste-specific training.
# Usage:  sbatch slurm_grounded_vlm.sh <kosmos2|geochat> [aw_m2|dronewaste]
set -euo pipefail
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/${ENV:-waste_vlm}/bin
cd "$PROJECT"
MODEL=${1:?model}; DATASET=${2:-aw_m2}
export WASTE_VLM_WEIGHTS=$WROOT/weights WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "[slurm] $(hostname) job=${SLURM_JOB_ID:-?} model=$MODEL dataset=$DATASET"
"$PYBIN/python" scripts/grounded_vlm_baseline.py --generate --model "$MODEL" \
  --dataset "$DATASET" --limit "${LIMIT:-0}" \
  --out "$WROOT/results/gvlm_${MODEL}_${DATASET}.json"
echo "[slurm] done=$(date -Is)"
