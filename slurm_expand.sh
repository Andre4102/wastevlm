#!/bin/bash
#SBATCH --job-name=expand
#SBATCH --output=logs/expand_%j.out
#SBATCH --error=logs/expand_%j.err
#SBATCH --account=iscrc_fiche
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=8 --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
cd "$PROJECT"
export WASTE_VLM_WEIGHTS=$WROOT/weights WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
/leonardo/home/userexternal/adiecidu/miniconda3/envs/waste_vlm/bin/python \
  scripts/expand_prompts.py --out "$WROOT/results/prompt_bank_expanded.json"
echo "[slurm] done=$(date -Is)"
