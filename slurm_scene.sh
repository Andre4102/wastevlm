#!/bin/bash
#SBATCH --job-name=scene
#SBATCH --output=logs/scene_%j.out
#SBATCH --error=logs/scene_%j.err
#SBATCH --account=iscrc_fiche
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=8 --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
cd /leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
export WASTE_VLM_WEIGHTS=$WROOT/weights WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
DEVSITES="site1 site3 site4 site6 site7 site8 site10 site11 site12 site14 site15 site16"
/leonardo_scratch/large/userexternal/adiecidu/envs/waste_sam3/bin/python \
  scripts/scene_graph.py --source pred --sites $DEVSITES \
  --threshold "${THR:-0.15}" --limit "${LIMIT:-250}" \
  --out "$WROOT/results/scenes_pred_dev.json"
echo "[slurm] done=$(date -Is)"
