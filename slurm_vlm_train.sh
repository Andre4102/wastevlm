#!/bin/bash
#SBATCH --job-name=vlm_train
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00

# Waste-VLM visual stage: frozen DINO/RADIO encoder -> projector -> Qwen2.5-7B.
# Two-stage LLaVA-1.5 recipe, 4x A100 DDP (single A100 is ~44h/epoch -> too slow):
#   pretrain  = connector/alignment, projector-only, LLM frozen (LCS-558K), ~11h
#   finetune  = visual-instruction tuning, projector + LoRA (LLaVA-Instruct-150K)
#
# Smoke (synthetic, 1 GPU, ~1 min):
#   sbatch --qos=boost_qos_dbg --time=00:20:00 --gres=gpu:1 --cpus-per-task=8 --mem=64G \
#          slurm_vlm_train.sh smoke radio-l
# Stage 1:  sbatch slurm_vlm_train.sh pretrain radio-l
# Stage 2:  sbatch slurm_vlm_train.sh finetune radio-l   # warm-starts projector from stage 1
set -euo pipefail

PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
ENV=waste_vlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
DATA=$WROOT/data
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/$ENV/bin
cd "$PROJECT"

MODE=${1:-smoke}                # smoke | pretrain | finetune
ENCODER=${2:-radio-l}
ETAG=${ENCODER//\//_}

export WASTE_VLM_WEIGHTS=$WROOT/weights
export DINOV3_REPO=$WROOT/dinov3_repo
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

# GPUs visible to this job (smoke is launched with --gres=gpu:1).
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
LAUNCH="$PYBIN/torchrun --standalone --nproc_per_node=$NGPU -m src.vlm_train"

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} start=$(date -Is)"
echo "[slurm] mode=$MODE encoder=$ENCODER ngpu=$NGPU"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Prefer a pre-tokenized cache (built by `pretokenize` mode) when present, else
# fall back to raw JSON. The cache is tokenizer-specific (Qwen), not encoder-
# specific, so one cache per dataset is shared by every encoder.
PRE_JSON="$DATA/llava_pretrain/blip_laion_cc_sbu_558k.json"
PRE_IMG="$DATA/llava_pretrain/images"
PRE_CACHE="$DATA/llava_pretrain/token_cache"
FT_JSON="$DATA/llava_instruct/llava_instruct_150k.json"
FT_IMG="$DATA/coco/train2017"
FT_CACHE="$DATA/llava_instruct/token_cache"

src_args() {  # $1=cache_dir $2=json  -> echoes --token-cache X | --train Y
  if [ -d "$1" ]; then echo "--token-cache $1"; else echo "--train $2"; fi
}

case "$MODE" in
  smoke)
    $LAUNCH --smoke --encoder "$ENCODER" --out-dir "$WROOT/results/vlm/${ETAG}_smoke"
    ;;
  pretokenize)
    # CPU-only; run once. Wastes the GPUs on this partition — cheaper on a login
    # node: python -m src.pretokenize_vlm --train ... --out ... --workers 16
    "$PYBIN/python" -m src.pretokenize_vlm --train "$PRE_JSON" \
      --image-root "$PRE_IMG" --out "$PRE_CACHE" --workers 16
    "$PYBIN/python" -m src.pretokenize_vlm --train "$FT_JSON" \
      --image-root "$FT_IMG" --out "$FT_CACHE" --workers 16
    ;;
  pretrain)
    # global batch = bs(8) * accum(8) * ngpu(4) = 256  (LLaVA-1.5 pretrain)
    $LAUNCH --stage pretrain --encoder "$ENCODER" \
      $(src_args "$PRE_CACHE" "$PRE_JSON") --image-root "$PRE_IMG" \
      --out-dir "$WROOT/results/vlm/${ETAG}_pretrain" \
      --epochs 1 --batch-size 8 --grad-accum 8 --save-steps 500
    ;;
  finetune)
    # global batch = bs(4) * accum(8) * ngpu(4) = 128  (LLaVA-1.5 finetune)
    $LAUNCH --stage finetune --encoder "$ENCODER" \
      $(src_args "$FT_CACHE" "$FT_JSON") --image-root "$FT_IMG" \
      --projector-init "$WROOT/results/vlm/${ETAG}_pretrain/projector.pt" \
      --out-dir "$WROOT/results/vlm/${ETAG}_finetune" \
      --epochs 1 --batch-size 4 --grad-accum 8 --save-steps 500
    ;;
  *)
    echo "unknown mode: $MODE (use smoke|pretokenize|pretrain|finetune)"; exit 1 ;;
esac

echo "[slurm] done=$(date -Is)"
