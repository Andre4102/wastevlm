"""Build the token-budget-matched training arms from normalized/*.jsonl.

Budget B = min(total_text_tokens over the available normalized datasets). Each
single-source arm is subsampled down to B; the combined arm takes ~B/3 from each.
Subsampling is seeded and stratified by `task_type` so heterogeneous sources
(Vision-Flan's 191 tasks, PixMo cap+points) keep their task mix instead of being
skewed by naive random sampling.

Arms written to $DATA_ROOT/normalized/arm_<name>.jsonl (shuffled, seeded), with a
per-arm image count / token count / task_type histogram in logs/arm_stats.json.

    python scripts/build_arms.py --seed 0
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from align_common import data_root

# normalized file -> arm name for the single-source arms
SOURCES = {
    "sharegpt4v": "arm_density",
    "visionflan": "arm_diversity",
    "pixmo": "arm_spatial",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def drop_missing_images(records: list[dict], norm: Path) -> list[dict]:
    """Filter out records whose image is absent on disk (e.g. dangling PixMo
    symlinks from link rot). The dataloader would silently substitute a blank
    image for these, quietly corrupting an arm — so exclude them here and let the
    matched budget re-derive from what is actually trainable."""
    kept = [r for r in records if not r["image"] or (norm / r["image"]).exists()]
    dropped = len(records) - len(kept)
    if dropped:
        print(f"[arms]   dropped {dropped} records with missing images", flush=True)
    return kept


def total_tokens(records: list[dict]) -> int:
    return sum(r["n_text_tokens"] for r in records)


def stratified_to_budget(records: list[dict], budget: int, rng: random.Random) -> list[dict]:
    """Pick a subset whose summed n_text_tokens ~= budget, preserving the
    per-task_type token proportions of `records`."""
    if total_tokens(records) <= budget:
        return list(records)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_type[r["task_type"]].append(r)
    grand = total_tokens(records)
    chosen: list[dict] = []
    for ttype, recs in by_type.items():
        rng.shuffle(recs)
        type_budget = budget * (total_tokens(recs) / grand)
        acc = 0
        for r in recs:
            if acc >= type_budget:
                break
            chosen.append(r)
            acc += r["n_text_tokens"]
    return chosen


def summarize(records: list[dict]) -> dict:
    return {
        "n_records": len(records),
        "n_images": sum(1 for r in records if r["image"]),
        "n_text_tokens": total_tokens(records),
        "task_type_hist": dict(Counter(r["task_type"] for r in records)),
    }


def write_arm(records: list[dict], path: Path, rng: random.Random) -> None:
    rng.shuffle(records)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-combined", action="store_true")
    args = ap.parse_args()

    root = data_root()
    norm = root / "normalized"
    rng = random.Random(args.seed)

    datasets: dict[str, list[dict]] = {}
    for src in SOURCES:
        p = norm / f"{src}.jsonl"
        if p.exists():
            datasets[src] = drop_missing_images(load_jsonl(p), norm)
            print(f"[arms] {src}: {len(datasets[src])} recs, "
                  f"{total_tokens(datasets[src]):,} tokens", flush=True)
        else:
            print(f"[arms] WARN {src}.jsonl absent — skipping its arm", flush=True)
    if not datasets:
        raise SystemExit("no normalized datasets found; run converters first")

    budget = min(total_tokens(r) for r in datasets.values())
    print(f"[arms] matched budget B = {budget:,} text tokens", flush=True)

    stats: dict[str, dict] = {"budget_tokens": budget, "seed": args.seed, "arms": {}}

    # single-source arms
    for src, recs in datasets.items():
        arm = SOURCES[src]
        sub = stratified_to_budget(recs, budget, random.Random(args.seed))
        write_arm(sub, norm / f"{arm}.jsonl", random.Random(args.seed))
        stats["arms"][arm] = summarize(sub)
        print(f"[arms] {arm}: {summarize(sub)}", flush=True)

    # combined arm: ~B/3 from each source, stratified within source
    if not args.no_combined and len(datasets) >= 2:
        share = budget // len(datasets)
        combined: list[dict] = []
        for src, recs in datasets.items():
            combined += stratified_to_budget(recs, share, random.Random(args.seed))
        write_arm(combined, norm / "arm_combined.jsonl", random.Random(args.seed))
        stats["arms"]["arm_combined"] = summarize(combined)
        print(f"[arms] arm_combined: {summarize(combined)}", flush=True)

    (root / "logs/arm_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[arms] wrote {root/'logs/arm_stats.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
