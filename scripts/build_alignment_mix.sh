#!/bin/bash
# Build the stage-1 alignment mix from the normalized per-source jsonls.
# All records already reference images/<name> relative to normalized/, so a plain
# concat is valid; we shuffle so any step-capped run sees a balanced sample rather
# than one source up front.
set -euo pipefail
NORM=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data/alignment/normalized
OUT=$NORM/alignment_mix.jsonl
TMP=$OUT.tmp

SRCS=(sharegpt4v.jsonl pixmo_cap.jsonl pixmo.jsonl)
: > "$TMP"
for s in "${SRCS[@]}"; do
  if [ -s "$NORM/$s" ]; then
    n=$(wc -l < "$NORM/$s")
    printf "  + %-22s %s\n" "$s" "$n"
    cat "$NORM/$s" >> "$TMP"
  else
    printf "  ! %-22s MISSING/empty (skipped)\n" "$s"
  fi
done
shuf "$TMP" -o "$OUT"
rm -f "$TMP"
echo "alignment_mix.jsonl total: $(wc -l < "$OUT")"
