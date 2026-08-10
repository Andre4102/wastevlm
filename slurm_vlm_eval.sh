#!/bin/bash
#SBATCH --job-name=vlm_eval                  # overridable via `sbatch --job-name`
#SBATCH --output=logs/%x_%j.out              # Output file (%x job name, %j job ID)
#SBATCH --error=logs/%x_%j.err               # Error file
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_FICHE
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --cpus-per-task=8                    # Request 8 CPU cores
#SBATCH --mem=80G
#SBATCH --time=04:00:00                      # Time limit (hh:mm:ss)

set -euo pipefail

# Leonardo paths. ENV=myenv (transformers 4.57 + pruning stack) so this same
# script also evals the structurally-pruned qwen2_5vl_pruned model (custom-qwen2
# materialize); waste_vlm (4.49) would be too old for that.
PROJECT=/leonardo/home/userexternal/adiecidu/scripts/wastevlm
WROOT=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm
RESULTS=$WROOT/results/vlm_eval
ENV=${ENV:-myenv}
PYTHON=/leonardo/home/userexternal/adiecidu/miniconda3/envs/$ENV/bin/python
cd "$PROJECT"

# positional args: MODEL OUTNAME [TASK] [PROMPT_STYLE] [DATASET] [LIMIT] [CKPT]
#   MODEL        : clip | qwen2_5vl | internvl3 | qwen2_5vl_pruned
#   OUTNAME      : subdir under $RESULTS (e.g. vlm_qwen2_5vl_closed)
#   TASK         : classify (default) | detect  [detect is DW only; not for clip]
#   PROMPT_STYLE : closed_vocab (default) | open_caption | open_cot
#   DATASET      : dw_paper10 (default) | aw_m2 | aw_m4
#   LIMIT        : 0 = full test split (default), >0 = smoke on first N images
#   CKPT         : prune out dir (lora_adapter/ + mask_logits.pt); required for
#                  qwen2_5vl_pruned, unused otherwise
MODEL=${1:?model name}
OUTNAME=${2:?out dir name}
TASK=${3:-classify}
PROMPT_STYLE=${4:-closed_vocab}
DATASET=${5:-dw_paper10}
LIMIT=${6:-0}
CKPT=${7:-}

export PRUNING_REPO=/leonardo/home/userexternal/adiecidu/scripts/pruning
export WASTE_VLM_WEIGHTS=$WROOT/weights
export WASTE_DATA_ROOT=$WROOT/data
export DINOV3_REPO=$WROOT/dinov3_repo
export HF_HOME=/leonardo_scratch/large/userexternal/adiecidu/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "[slurm] host=$(hostname)  job=${SLURM_JOB_ID:-?}  start=$(date -Is)"
echo "[slurm] model=$MODEL task=$TASK dataset=$DATASET prompt=$PROMPT_STYLE limit=$LIMIT out=$OUTNAME ckpt=${CKPT:-none}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

CKPT_ARG=(); [ -n "$CKPT" ] && CKPT_ARG=(--ckpt "$CKPT")
"$PYTHON" -m src.vlm_eval \
  --model "$MODEL" \
  --task "$TASK" \
  --dataset "$DATASET" \
  --prompt-style "$PROMPT_STYLE" \
  --split test --limit "$LIMIT" \
  "${CKPT_ARG[@]}" \
  --out-json "$RESULTS/$OUTNAME/test_eval.json" \
  --save-raw "$RESULTS/$OUTNAME/raw_responses.jsonl"

echo "[slurm] done=$(date -Is)"
