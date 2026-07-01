#!/bin/bash
#SBATCH --job-name=vlm_ddpcheck
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=00:30:00

# Fast validation of the DDP all-reduce fix: real-data 4-GPU pretrain, but
# --logging-steps 2 forces the (previously buggy) logging collective at step 2,
# and --epochs 0.01 caps it to ~21 steps. Clean exit past step 2 = fix holds.
set -euo pipefail
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/waste_vlm/bin
cd "$PROJECT"

export WASTE_VLM_WEIGHTS=$WROOT/weights
export DINOV3_REPO=$WROOT/dinov3_repo
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} start=$(date -Is)"
"$PYBIN/torchrun" --standalone --nproc_per_node=4 -m src.vlm_train \
  --stage pretrain --encoder "${1:-radio-l}" \
  --token-cache "$WROOT/data/llava_pretrain/token_cache" \
  --image-root "$WROOT/data/llava_pretrain/images" \
  --out-dir "$WROOT/results/vlm/ddpcheck" \
  --epochs 0.01 --batch-size 8 --grad-accum 8 --logging-steps 2 --save-steps 100000
echo "[slurm] done=$(date -Is)"
