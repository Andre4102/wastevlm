#!/bin/bash
#SBATCH --job-name=vlm_modern
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=boost_usr_prod
#SBATCH --account=iscrc_fiche
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=16:00:00
# ntasks-per-node (not ntasks) so `sbatch -N 4` scales the task count with the
# nodes: multi-node torchrun needs exactly one launcher task per node.
# The partition allows 24h and 64 nodes, so both are raisable on the CLI.

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
# Per-GPU micro-batch and accumulation. Visual tokens scale as
# (IMG_SIZE/(16*PSHUF))^2 -> 768/ps2=576, 768/ps1=2304, 1024/ps2=1024, so a
# pixel-shuffle or resolution change moves activation memory a lot. Keep the
# GLOBAL batch fixed when changing BS (global = BS * ACCUM * NGPU): stage 1 is
# 256, stage 2 is 128. Override both together, e.g. BS=1 ACCUM=64 for ps1.
# Stage-2 warm start. TAG includes IMG_SIZE, but the projector is
# patch_dim*PSHUF^2 -> hidden and does NOT depend on resolution, so a 1024/ps2 arm
# reuses the 768/ps2 stage-1. Override when the two differ; PSHUF must still match.
PROJ_INIT=${PROJ_INIT:-$WROOT/results/vlm/${TAG}_pretrain/projector.pt}
BS=${BS:-4}
# The recipe is defined by the GLOBAL batch (256 pretrain / 128 finetune), not by
# accumulation, so derive ACCUM from the world size instead of hard-coding it for
# 4 GPUs. Adding nodes then cuts wall-clock without changing the recipe at all:
#   4 GPU  -> ACCUM_FT 8   (identical to every arm run so far)
#   16 GPU -> ACCUM_FT 2
#   32 GPU -> ACCUM_FT 1   (the floor at BS=4; more GPUs needs a smaller BS)
# ACCUM= still overrides both, for deliberate recipe changes.
# The derivation itself lives below, after WORLD is known.
GB_PRE=${GB_PRE:-256}
GB_FT=${GB_FT:-128}

export WASTE_VLM_WEIGHTS=$WROOT/weights
export DINOV3_REPO=$WROOT/dinov3_repo
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
# High-res batches spike activation memory; defrag reserved-but-unallocated cache.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l); [ "$NGPU" -gt 0 ] 2>/dev/null || NGPU=1
NNODES=${SLURM_NNODES:-1}
WORLD=$((NGPU * NNODES))

# Must follow WORLD: under `set -u` an earlier reference aborts the job in ~7s.
ACCUM_PRE=${ACCUM:-$(( GB_PRE / (BS * WORLD) ))}
ACCUM_FT=${ACCUM:-$(( GB_FT / (BS * WORLD) ))}
[ "$ACCUM_PRE" -lt 1 ] && ACCUM_PRE=1
[ "$ACCUM_FT" -lt 1 ] && ACCUM_FT=1

# Single node keeps the exact --standalone launch every previous arm used; only a
# multi-node job takes the rendezvous path, so `sbatch` with no -N is unchanged.
if [ "$NNODES" -gt 1 ]; then
  MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  LAUNCH="srun --ntasks=$NNODES --ntasks-per-node=1 $PYBIN/torchrun \
    --nnodes=$NNODES --nproc_per_node=$NGPU \
    --rdzv_id=${SLURM_JOB_ID:-0} --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER}:${MASTER_PORT:-29500} -m src.vlm_train"
else
  LAUNCH="$PYBIN/torchrun --standalone --nproc_per_node=$NGPU -m src.vlm_train"
fi
RES_ARGS="--encoder $ENCODER --image-size $IMG_SIZE --pixel-shuffle $PSHUF"

echo "[slurm] host=$(hostname) job=${SLURM_JOB_ID:-?} start=$(date -Is)"
echo "[slurm] mode=$MODE encoder=$ENCODER img=$IMG_SIZE pshuf=$PSHUF ngpu=$NGPU tag=$TAG"
VTOK=$(( (IMG_SIZE / (16 * PSHUF)) * (IMG_SIZE / (16 * PSHUF)) ))
echo "[slurm] nodes=$NNODES gpus/node=$NGPU world=$WORLD"
echo "[slurm] visual tokens/image=$VTOK  bs=$BS accum=pre:$ACCUM_PRE/ft:$ACCUM_FT"
echo "[slurm] global batch = pretrain $((BS*ACCUM_PRE*WORLD)) | finetune $((BS*ACCUM_FT*WORLD))"
# a silently-wrong global batch is a changed recipe, not a changed footprint
if [ $((BS*ACCUM_FT*WORLD)) -ne "$GB_FT" ] && [ -z "${ACCUM:-}" ]; then
  echo "[slurm] WARN finetune global batch $((BS*ACCUM_FT*WORLD)) != target $GB_FT" \
       "-- BS*WORLD must divide $GB_FT"
fi
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
# Overridable so a new arm can point at a composed mix (scripts/build_train_mix.py)
# without a new mode. ARM appends a suffix to the output dir -- without it two arms
# would write to the same <TAG>_finetune_next/ and the second would eat the first.
# RS records carry ABSOLUTE image paths and general-replay records relative ones;
# vlm_data only prepends image_root to relative paths, so one root serves both.
FT_NEXT_JSON=${FT_NEXT_JSON:-"$DATA/alignment/normalized/sft_mix.jsonl"}
FT_NEXT_IMG=${FT_NEXT_IMG:-"$DATA/alignment/normalized"}
ARM=${ARM:-}

src_args() {  # $1=cache_dir $2=json -> --token-cache X if cache exists else --train Y
  if [ -d "$1" ]; then echo "--token-cache $1"; else echo "--train $2"; fi
}

case "$MODE" in
  smoke)
    $LAUNCH --smoke $RES_ARGS --out-dir "$WROOT/results/vlm/${TAG}_smoke"
    ;;
  pretrain)
    # global batch = bs(4) * accum(16) * ngpu(4) = 256 (matches the align-arm recipe).
    # RESUME=auto restores optimizer+scheduler+step, so a stage 1 that outruns the
    # wall (ps1 is ~36h) continues as ONE cosine over MAXSTEPS instead of restarting
    # its warmup each job.
    RESUME_ARG=""
    [ "${RESUME:-}" = "auto" ] && RESUME_ARG="--resume auto"
    $LAUNCH --stage pretrain $RES_ARGS \
      --train "$PRE_JSON" --image-root "$PRE_IMG" \
      --out-dir "$WROOT/results/vlm/${TAG}_pretrain" \
      --max-steps "$MAXSTEPS" --batch-size "$BS" --grad-accum "$ACCUM_PRE" \
      --warmup-ratio 0.03 --save-steps 500 $RESUME_ARG
    ;;
  finetune)
    # global batch = bs(4) * accum(8) * ngpu(4) = 128 (LLaVA-1.5 finetune, 1 epoch)
    # RESUME=auto for the same reason as finetune_next: at ps1 (2304 tokens) one
    # epoch is ~18h against a 16h wall, so this stage needs the chain too. --epochs 1
    # over a fixed mix gives the SAME total_steps in every chain job, so the cosine
    # stays one curve across the restart.
    RESUME_ARG=""
    [ "${RESUME:-}" = "auto" ] && RESUME_ARG="--resume auto"
    $LAUNCH --stage finetune $RES_ARGS \
      $(src_args "$FT_CACHE" "$FT_JSON") --image-root "$FT_IMG" \
      --projector-init "$PROJ_INIT" \
      --out-dir "$WROOT/results/vlm/${TAG}_finetune" \
      --epochs 1 --batch-size "$BS" --grad-accum "$ACCUM_FT" \
      --save-steps "${SAVE_STEPS:-500}" $RESUME_ARG
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
      --projector-init "$PROJ_INIT" \
      --out-dir "$WROOT/results/vlm/${TAG}_finetune_next${ARM:+_$ARM}" \
      --epochs 1 --batch-size "$BS" --grad-accum "$ACCUM_FT" \
      --save-steps "${SAVE_STEPS:-250}" $RESUME_ARG
    ;;
  *)
    echo "unknown mode: $MODE (use smoke|pretrain|finetune|finetune_next)"; exit 1 ;;
esac

echo "[slurm] done=$(date -Is)"
