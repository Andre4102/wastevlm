#!/bin/bash
#SBATCH --job-name=describe
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
set -euo pipefail
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/waste_vlm/bin
cd "$PROJECT"
ARM=${1:?arm}; LLM=${2:?llm path}
export WASTE_VLM_WEIGHTS=$WROOT/weights WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "[slurm] $(hostname) job=${SLURM_JOB_ID:-?} arm=$ARM llm=$LLM"
"$PYBIN/python" scripts/describe_eval.py --arm "$ARM" --llm "$LLM" \
  --n "${N:-300}" --n-empty "${NE:-100}" --seed 0 \
  --out-json "$WROOT/results/describe_${ARM}.json"
echo "[slurm] done=$(date -Is)"
