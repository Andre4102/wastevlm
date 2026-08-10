#!/bin/bash
#SBATCH --job-name=vlm_prune
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

# Joint structural-prune + LoRA-fine-tune of a *pretrained* VLM on the waste data.
# Starts from cell A1 (radio-l encoder + projector + Qwen2.5-7B decoder), wraps the
# Qwen decoder with learnable structural masks (custom-qwen2 arch in the pruning
# repo) and learns the masks jointly with LoRA + projector on waste_sft — the mask
# search removes the decoder structure least useful for the waste-VLM task. At the
# end it materializes (slices) the pruned decoder at --materialize-threshold.
#
# IMPORTANT: runs in `myenv` (transformers 4.57 + the pruning stack). The default
# VLM env `waste_vlm` (transformers 4.49) is too old for the pruning code.
#
# Smoke (synthetic data, real A1 decoder, 1 GPU, validates the full pipeline incl.
# materialize):  sbatch --qos=boost_qos_dbg --time=00:30:00 --gres=gpu:1 \
#                       --cpus-per-task=8 --mem=80G slurm_vlm_prune.sh smoke
# Full run:      sbatch slurm_vlm_prune.sh prune
set -euo pipefail

PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
ENV=myenv
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
DATA=$WROOT/data
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/$ENV/bin
cd "$PROJECT"

MODE=${1:-prune}                # prune | smoke
ENCODER=${2:-radio-l}
ETAG=${ENCODER//\//_}
TS=${TARGET_SPARSITY:-0.5}
PRUNE_MODE=${PRUNE_MODE:-joint}  # joint | sequential  (override: PRUNE_MODE=sequential sbatch ...)

export PRUNING_REPO=/leonardo/home/userexternal/adiecidu/scripts/pruning
export WASTE_VLM_WEIGHTS=$WROOT/weights
export DINOV3_REPO=$WROOT/dinov3_repo
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
LAUNCH="$PYBIN/torchrun --standalone --nproc_per_node=$NGPU -m src.vlm_train"

# A1 = pretrained VLM to prune (merged Qwen decoder + aligned projector).
A1=$WROOT/results/vlm/${ETAG}_finetune
DECODER=$A1/llm_merged
PROJECTOR=$A1/projector.pt
TRAIN_JSON=$DATA/waste_sft/train.json   # absolute image paths → no --image-root

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} start=$(date -Is)"
echo "[slurm] mode=$MODE prune_mode=$PRUNE_MODE encoder=$ENCODER ngpu=$NGPU target_sparsity=$TS"
echo "[slurm] decoder=$DECODER"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case "$MODE" in
  smoke)
    # synthetic data, real A1 Qwen decoder → exercises wrap+LoRA+prune-loop+materialize.
    # Small total_steps means phase2 (sequential) may not trigger; joint smoke still
    # exercises tau schedule + materialize + sanity-gen.
    $LAUNCH --smoke --prune --prune-mode "$PRUNE_MODE" --encoder "$ENCODER" \
      --llm-path "$DECODER" --projector-init "$PROJECTOR" \
      --target-sparsity "$TS" --materialize-threshold 0.5 \
      --out-dir "$WROOT/results/vlm/${ETAG}_prune_smoke"
    ;;
  prune)
    # global batch = bs(2) * accum(16) * ngpu(4) = 128.
    # materialize-threshold is the 0.5 prob cutoff (STE binarization point), NOT the
    # target sparsity — target is driven by the controller toward --target-sparsity.
    $LAUNCH --prune --prune-mode "$PRUNE_MODE" --stage finetune --encoder "$ENCODER" \
      --llm-path "$DECODER" --projector-init "$PROJECTOR" \
      --train "$TRAIN_JSON" \
      --target-sparsity "$TS" --materialize-threshold 0.5 \
      --prune-tau-start 5.0 --prune-tau-min 0.01 --prune-tau-anneal-frac 0.7 \
      --prune-phase2-frac 0.35 \
      --mask-lr 0.01 --prune-warmup 200 \
      --epochs 3 --batch-size 2 --grad-accum 16 \
      --lora-lr 2e-5 --projector-lr 2e-4 --save-steps 500 --logging-steps 10 \
      --out-dir "$WROOT/results/vlm/${ETAG}_prune_${PRUNE_MODE}_ts${TS}"
    ;;
  *)
    echo "unknown mode: $MODE (use prune|smoke)"; exit 1 ;;
esac

echo "[slurm] done=$(date -Is)"
