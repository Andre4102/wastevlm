"""Merge agent re-annotations back into full.jsonl.

Workflow:
  1. In inspect_captions.ipynb, call suggest(i, "note") on captions you think
     are wrong. Each appends {dataset,image_id,image_path,gt_categories,
     old_caption,suggestion} to recaption_queue.jsonl.
  2. An agent reads that queue, re-opens each image, and writes a corrected
     caption (incorporating your note) to recaption_queue_out.jsonl, schema
     {dataset, image_id, caption}.
  3. This script updates full.jsonl in place (matching on dataset+image_id),
     re-appending the canonical '  Labels: <gt>' line so the target stays
     consistent with ground truth. A timestamped backup is written first, and
     the processed queue files are archived so they are not reapplied.

Usage:
    python -m src.apply_recaptions \
        --full /home/ids/diecidue/data/captions/full.jsonl \
        --recaptions /home/ids/diecidue/data/captions/recaption_queue_out.jsonl
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path


def canonical(desc: str, gt: list[str]) -> str:
    desc = desc.strip()
    # drop any Labels line the agent appended; we re-add the authoritative one
    idx = desc.rfind("Labels:")
    if idx != -1:
        desc = desc[:idx].rstrip()
    labels = ", ".join(sorted(gt)) if gt else "none"
    return desc + "\n  Labels: " + labels


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full", type=Path, required=True)
    p.add_argument("--recaptions", type=Path, required=True)
    p.add_argument("--keep-agent-labels", action="store_true",
                   help="trust the agent's Labels line instead of re-deriving "
                        "from gt_categories (use only if the suggestion corrected "
                        "the label itself).")
    args = p.parse_args()

    records = [json.loads(l) for l in args.full.read_text().splitlines() if l.strip()]
    by_key = {(r["dataset"], str(r["image_id"])): r for r in records}

    updates = [json.loads(l) for l in args.recaptions.read_text().splitlines() if l.strip()]
    applied = missing = 0
    for u in updates:
        key = (u["dataset"], str(u["image_id"]))
        rec = by_key.get(key)
        if rec is None:
            missing += 1
            print(f"  [warn] {key} not in full.jsonl — skipped")
            continue
        if args.keep_agent_labels:
            rec["caption"] = u["caption"].strip()
        else:
            rec["caption"] = canonical(u["caption"], rec["gt_categories"])
        applied += 1

    bak = args.full.with_suffix(f".jsonl.{time.strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy(args.full, bak)
    with args.full.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # archive the processed queue + output so a re-run doesn't double-apply
    for q in (args.recaptions, args.recaptions.parent / "recaption_queue.jsonl"):
        if q.exists():
            q.rename(q.with_suffix(q.suffix + ".done"))

    print(f"applied {applied} re-captions, {missing} unmatched")
    print(f"backup -> {bak}")
    print(f"updated -> {args.full}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
