#!/bin/bash
#SBATCH --job-name=base_auc
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/waste_vlm/bin
cd "$PROJECT"
MODEL=${1:?model}; DATASET=${2:-aw_m2}
export WASTE_VLM_WEIGHTS=$WROOT/weights WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
OUT=${OUT_TAG:-baseauc_${MODEL}_${DATASET}}
echo "[slurm] $(hostname) job=${SLURM_JOB_ID:-?} model=$MODEL dataset=$DATASET sites=${SITES:-all}"
"$PYBIN/python" scripts/baseline_binary_auc.py --model "$MODEL" --dataset "$DATASET" \
  ${SITES:+--sites "$SITES"} ${SITES_ALL:+--sites-all-images} \
  --out-json "$WROOT/results/vlm_eval/$OUT/binary_auc.json"
echo "[slurm] done=$(date -Is)"
