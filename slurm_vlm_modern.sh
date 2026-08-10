#!/bin/bash
#SBATCH --job-name=vlm_modern
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=16:00:00

# Modern general-purpose VLM recipe: frozen C-RADIOv4-SO400M -> pixel-shuffle
# projector -> Qwen2.5-7B. Higher native resolution (RADIO CPE, no AnyRes tiling)
# + dense-caption alignment. 2-stage, encoder frozen throughout:
#   pretrain  = connector/alignment, projector-only, LLM frozen, PixMo dense caps
#   finetune  = visual-instruction tuning, projector + LoRA, LLaVA-Instruct-150K
# Then eval zero-shot on DroneWaste/AerialWaste (existing harness).
#
# "Prove the recipe on 150K, then scale to LLaVA-NeXT-760K." This is the 150K pass.
#
# Resolution/shuffle overridable: IMG_SIZE (default 768, must be x16),
# PSHUF (default 2 -> 1/4 visual tokens; 768/16=48 grid -> 576 tokens).
#
# GPU smoke (synthetic, 1 GPU, ~2 min) — validates the full 768/ps2 forward:
#   sbatch --qos=boost_qos_dbg --time=00:20:00 --gres=gpu:1 --cpus-per-task=8 --mem=64G \
#          slurm_vlm_modern.sh smoke
# Stage 1:  sbatch slurm_vlm_modern.sh pretrain
# Stage 2:  sbatch slurm_vlm_modern.sh finetune   # warm-starts projector from stage 1
set -euo pipefail

PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
ENV=waste_vlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
DATA=$WROOT/data
PYBIN=/leonardo/home/userexternal/adiecidu/miniconda3/envs/$ENV/bin
cd "$PROJECT"

MODE=${1:-smoke}                # smoke | pretrain | finetune
ENCODER=${2:-cradiov4-so}
ETAG=${ENCODER//\//_}
IMG_SIZE=${IMG_SIZE:-768}
PSHUF=${PSHUF:-2}
MAXSTEPS=${MAXSTEPS:-2000}       # stage-1 steps (PixMo is small -> step-capped)
TAG=${ETAG}_r${IMG_SIZE}ps${PSHUF}   # keeps outputs distinct from legacy runs

export WASTE_VLM_WEIGHTS=$WROOT/weights
export DINOV3_REPO=$WROOT/dinov3_repo
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
# High-res batches spike activation memory; defrag reserved-but-unallocated cache.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); NGPU=${NGPU:-1}
LAUNCH="$PYBIN/torchrun --standalone --nproc_per_node=$NGPU -m src.vlm_train"
RES_ARGS="--encoder $ENCODER --image-size $IMG_SIZE --pixel-shuffle $PSHUF"

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} start=$(date -Is)"
echo "[slurm] mode=$MODE encoder=$ENCODER img=$IMG_SIZE pshuf=$PSHUF ngpu=$NGPU tag=$TAG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Stage-1 alignment data: combined dense-caption/grounding mix (ShareGPT4V 200k +
# PixMo-Cap ~700k + PixMo cap/points), built by scripts/build_alignment_mix.sh.
# All records reference images/<name> relative to normalized/ (symlinked, no copy).
PRE_JSON="$DATA/alignment/normalized/alignment_mix.jsonl"
PRE_IMG="$DATA/alignment/normalized"
# Stage-2 SFT: LLaVA-Instruct-150K. Token cache is tokenizer-specific (Qwen), NOT
# resolution/encoder-specific, so the existing cache is reused as-is.
FT_JSON="$DATA/llava_instruct/llava_instruct_150k.json"
FT_IMG="$DATA/coco/train2017"
FT_CACHE="$DATA/llava_instruct/token_cache"
# Stage-2 SCALE-UP (finetune_next): LLaVA-NeXT-760K + Vision-Flan, ~819K, still
# general-purpose (NO waste-SFT). Images are the flat normalized/ tree. One epoch
# is ~6.4k steps > the 16h wall, so it runs as a 2-job chain: job 1 fresh, job 2
# with RESUME=auto (restores optimizer+scheduler+step -> one continuous cosine).
FT_NEXT_JSON="$DATA/alignment/normalized/sft_mix.jsonl"
FT_NEXT_IMG="$DATA/alignment/normalized"

src_args() {  # $1=cache_dir $2=json -> --token-cache X if cache exists else --train Y
  if [ -d "$1" ]; then echo "--token-cache $1"; else echo "--train $2"; fi
}

case "$MODE" in
  smoke)
    $LAUNCH --smoke $RES_ARGS --out-dir "$WROOT/results/vlm/${TAG}_smoke"
    ;;
  pretrain)
    # global batch = bs(4) * accum(16) * ngpu(4) = 256 (matches the align-arm recipe)
    $LAUNCH --stage pretrain $RES_ARGS \
      --train "$PRE_JSON" --image-root "$PRE_IMG" \
      --out-dir "$WROOT/results/vlm/${TAG}_pretrain" \
      --max-steps "$MAXSTEPS" --batch-size 4 --grad-accum 16 \
      --warmup-ratio 0.03 --save-steps 500
    ;;
  finetune)
    # global batch = bs(4) * accum(8) * ngpu(4) = 128 (LLaVA-1.5 finetune, 1 epoch)
    $LAUNCH --stage finetune $RES_ARGS \
      $(src_args "$FT_CACHE" "$FT_JSON") --image-root "$FT_IMG" \
      --projector-init "$WROOT/results/vlm/${TAG}_pretrain/projector.pt" \
      --out-dir "$WROOT/results/vlm/${TAG}_finetune" \
      --epochs 1 --batch-size 4 --grad-accum 8 --save-steps 500
    ;;
  finetune_next)
    # Scale-up stage-2 (~819K, general-purpose). Same global batch 128 + 1 epoch as
    # `finetune` so total_steps is well-defined; --epochs 1 over the fixed mix gives
    # the SAME total_steps in both chain jobs, so the cosine is one curve. save-steps
    # 250 keeps <=250 steps (~1h) of work at risk when the wall kills job 1.
    # RESUME=auto on job 2 restores optimizer+scheduler+step from the newest ckpt.
    RESUME_ARG=""
    [ "${RESUME:-}" = "auto" ] && RESUME_ARG="--resume auto"
    $LAUNCH --stage finetune $RES_ARGS \
      --train "$FT_NEXT_JSON" --image-root "$FT_NEXT_IMG" \
      --projector-init "$WROOT/results/vlm/${TAG}_pretrain/projector.pt" \
      --out-dir "$WROOT/results/vlm/${TAG}_finetune_next" \
      --epochs 1 --batch-size 4 --grad-accum 8 --save-steps 250 $RESUME_ARG
    ;;
  *)
    echo "unknown mode: $MODE (use smoke|pretrain|finetune|finetune_next)"; exit 1 ;;
esac

echo "[slurm] done=$(date -Is)"
