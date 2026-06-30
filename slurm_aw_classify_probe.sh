#!/bin/bash
#SBATCH --job-name=aw_classify_probe         # overridable via `sbatch --job-name`
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=L40S
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail

PROJECT=/home/ids/diecidue/scripts/waste_vlm
RESULTS=/home/ids/diecidue/results/waste_vlm
PYTHON=/home/ids/diecidue/miniconda3/envs/waste_vlm/bin/python
cd "$PROJECT"

# positional args: BACKBONE_TYPE VERSION OUTNAME [BACKBONE_ID] [MULTI_BLOCK] [TASK]
#   BACKBONE_TYPE : dinov2 | dinov3 | radio
#   VERSION       : m2 | m4 | ""  (empty for binary task)
#   OUTNAME       : subdir under $RESULTS for the output JSON
#   BACKBONE_ID   : optional override (e.g. RADIO-B path, "vitl16" for DINOv3-L)
#   MULTI_BLOCK   : optional comma-separated block indices, e.g. "3,7,11"
#   TASK          : mcml (default) | binary
BTYPE=${1:?backbone type}
VERSION=${2:-}
OUTNAME=${3:?out dir name}
BACKBONE_ID=${4:-}
MULTI_BLOCK=${5:-}
TASK=${6:-mcml}

echo "[slurm] host=$(hostname)  job=${SLURM_JOB_ID:-?}  start=$(date -Is)"
echo "[slurm] btype=$BTYPE  version=${VERSION:-n/a}  task=$TASK  id=${BACKBONE_ID:-default}  multi_block=${MULTI_BLOCK:-none}  out=$OUTNAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export HF_HUB_OFFLINE=1
EXTRA_ARGS=(--task "$TASK")
if [ -n "$BACKBONE_ID" ]; then EXTRA_ARGS+=(--backbone-id "$BACKBONE_ID"); fi
if [ -n "$MULTI_BLOCK" ]; then EXTRA_ARGS+=(--multi-block "$MULTI_BLOCK"); fi
if [ -n "$VERSION" ]; then EXTRA_ARGS+=(--version "$VERSION"); fi

# Output filename: binary task → aw_binary_probe.json; mcml → aw_<version>_probe.json
if [ "$TASK" = "binary" ]; then
  OUT_JSON="$RESULTS/$OUTNAME/aw_binary_probe.json"
else
  OUT_JSON="$RESULTS/$OUTNAME/aw_${VERSION}_probe.json"
fi

"$PYTHON" -m src.aw_classify_probe \
  --backbone-type "$BTYPE" \
  "${EXTRA_ARGS[@]}" \
  --out-json "$OUT_JSON"

echo "[slurm] done=$(date -Is)"
