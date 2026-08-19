#!/bin/bash
#SBATCH --job-name=router
#SBATCH --output=logs/router_%j.out
#SBATCH --error=logs/router_%j.err
#SBATCH --account=iscrc_fiche
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=8 --mem=64G
#SBATCH --time=03:00:00
set -euo pipefail
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/${ENV:-waste_vlm}/bin
cd "$PROJECT"
export WASTE_VLM_WEIGHTS=$WROOT/weights WASTE_DATA_ROOT=$WROOT/data
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
RES=${RES:-$WROOT/results/radio_zs_dronewaste_640_contrastive_dev.json}
echo "[slurm] $(hostname) job=${SLURM_JOB_ID:-?} defer=${DEFER:-0.30} topk=${TOPK:-5}"
"$PYBIN/python" scripts/two_stage_router.py --result "$RES" \
  --defer "${DEFER:-0.30}" --topk "${TOPK:-5}" ${DUMP:+--dump-prompts} \
  --out-json "$WROOT/results/router_defer${DEFER:-0.30}_top${TOPK:-5}.json"
echo "[slurm] done=$(date -Is)"
