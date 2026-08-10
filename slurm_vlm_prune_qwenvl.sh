#!/bin/bash
#SBATCH --job-name=qwenvl_prune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_FICHE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --mail-user=andrea.diecidue@polimi.it
#SBATCH --mail-type=BEGIN,FAIL,END

# Joint structural-prune + LoRA fine-tune of the NATIVE Qwen2.5-VL decoder on the
# waste VQA data. Freezes the native ViT+merger; wraps the Qwen2.5 decoder with
# M-RoPE structural masks (custom_attentions/qwen2_5_vl_attention.py) + LoRA and
# learns masks jointly (or sequentially) with the fixed tau schedule + two-phase.
# Saves LoRA adapter + mask_logits.pt (materialize/eval at load via the
# qwen2_5vl_pruned eval adapter).
#
# Smoke (2 GPU dbg): sbatch --qos=boost_qos_dbg --time=00:25:00 --gres=gpu:2 \
#     --cpus-per-task=16 --mem=120G slurm_vlm_prune_qwenvl.sh smoke
# Full:  PRUNE_MODE=joint      sbatch slurm_vlm_prune_qwenvl.sh prune
#        PRUNE_MODE=sequential sbatch slurm_vlm_prune_qwenvl.sh prune
set -euo pipefail

PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
ENV=myenv
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/$ENV/bin
cd "$PROJECT"

MODE=${1:-prune}                 # prune | smoke
PRUNE_MODE=${PRUNE_MODE:-joint}  # joint | sequential
TS=${TARGET_SPARSITY:-0.5}

export PRUNING_REPO=/leonardo/home/userexternal/adiecidu/scripts/pruning
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
LAUNCH="$PYBIN/torchrun --standalone --nproc_per_node=$NGPU -m src.vlm_train_qwenvl"

BASE=$WROOT/weights/Qwen2.5-VL-7B-Instruct
TRAIN_JSON=$WROOT/data/waste_sft/train.json   # absolute image paths
OUT=$WROOT/results/vlm/qwenvl_prune_${PRUNE_MODE}_ts${TS}${RUNTAG:+_$RUNTAG}

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} start=$(date -Is)"
echo "[slurm] mode=$MODE prune_mode=$PRUNE_MODE ngpu=$NGPU target_sparsity=$TS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case "$MODE" in
  smoke)
    $LAUNCH --base "$BASE" --train "$TRAIN_JSON" --smoke \
      --prune --prune-mode "$PRUNE_MODE" --target-sparsity "$TS" \
      --grad-accum 1 --logging-steps 1 \
      --out-dir "$WROOT/results/vlm/qwenvl_prune_smoke"
    ;;
  prune)
    # bs=1 per device (variable image grids) * accum(16) * ngpu(4) = 64
    $LAUNCH --base "$BASE" --train "$TRAIN_JSON" \
      --prune --prune-mode "$PRUNE_MODE" --target-sparsity "$TS" \
      --prune-tau-start 5.0 --prune-tau-min 0.01 --prune-tau-anneal-frac 0.7 \
      --prune-phase2-frac 0.35 --prune-warmup 200 --mask-lr 0.01 \
      --epochs 2 --grad-accum 16 --lora-lr 2e-5 \
      --max-pixels $((1024 * 28 * 28)) --logging-steps 10 \
      --out-dir "$OUT"
    ;;
  *)
    echo "unknown mode: $MODE (use prune|smoke)"; exit 1 ;;
esac

echo "[slurm] done=$(date -Is)"
