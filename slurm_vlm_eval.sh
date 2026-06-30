#!/bin/bash
#SBATCH --job-name=vlm_eval                  # overridable via `sbatch --job-name`
#SBATCH --output=logs/%x_%j.out              # Output file (%x job name, %j job ID)
#SBATCH --error=logs/%x_%j.err               # Error file
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH --cpus-per-task=8                    # Request 8 CPU cores
#SBATCH --mem=32G
#SBATCH --time=04:00:00                      # Time limit (hh:mm:ss)

set -euo pipefail

PROJECT=/home/ids/diecidue/scripts/waste_vlm
RESULTS=/home/ids/diecidue/results/waste_vlm
PYTHON=/home/ids/diecidue/miniconda3/envs/waste_vlm/bin/python
cd "$PROJECT"

# positional args: MODEL OUTNAME [TASK] [PROMPT_STYLE] [DATASET] [LIMIT]
#   MODEL        : clip | qwen2_5vl | internvl3
#                  clip uses zero-shot CLIP ViT-B/32; --prompt-style is ignored for clip.
#   OUTNAME      : subdir under $RESULTS (e.g. vlm_clip_dw | vlm_qwen2_5vl_closed)
#   TASK         : classify (default) | detect  [detect is DW only; not supported for clip]
#   PROMPT_STYLE : closed_vocab (default) | open_caption | open_cot  (only used when TASK=classify + non-clip model)
#                  open_cot = two-turn chain-of-thought (describe first, then name waste; free vocab, keyword-bag parse)
#   DATASET      : dw_paper10 (default) | aw_m2 | aw_m4    (only used when TASK=classify)
#   LIMIT        : 0 = full test split (default), >0 = smoke test on first N images
MODEL=${1:?model name}
OUTNAME=${2:?out dir name}
TASK=${3:-classify}
PROMPT_STYLE=${4:-closed_vocab}
DATASET=${5:-dw_paper10}
LIMIT=${6:-0}

echo "[slurm] host=$(hostname)  job=${SLURM_JOB_ID:-?}  start=$(date -Is)"
echo "[slurm] model=$MODEL task=$TASK dataset=$DATASET prompt=$PROMPT_STYLE limit=$LIMIT out=$OUTNAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export HF_HUB_OFFLINE=1  # weights are local
"$PYTHON" -m src.vlm_eval \
  --model "$MODEL" \
  --task "$TASK" \
  --dataset "$DATASET" \
  --prompt-style "$PROMPT_STYLE" \
  --split test --limit "$LIMIT" \
  --out-json "$RESULTS/$OUTNAME/test_eval.json" \
  --save-raw "$RESULTS/$OUTNAME/raw_responses.jsonl"

echo "[slurm] done=$(date -Is)"
