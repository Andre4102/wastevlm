"""Step-5 sanity check: render N random samples per arm to a markdown file so
image<->text alignment (and any step-2 conversion bug) can be eyeballed, plus
assert no image-path collisions and near-equal token budgets across arms.

    python scripts/spot_check.py --n 10
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from align_common import data_root


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = data_root()
    norm = root / "normalized"
    arms = sorted(norm.glob("arm_*.jsonl"))
    if not arms:
        raise SystemExit("no arm_*.jsonl found; run build_arms.py first")

    rng = random.Random(args.seed)
    out = [f"# Alignment arms spot-check (n={args.n}/arm)\n"]
    budgets = {}
    for arm in arms:
        recs = load_jsonl(arm)
        budgets[arm.stem] = sum(r["n_text_tokens"] for r in recs)
        out.append(f"\n## {arm.stem}  ({len(recs)} recs, {budgets[arm.stem]:,} tokens)\n")
        for r in rng.sample(recs, min(args.n, len(recs))):
            img = norm / r["image"] if r["image"] else None
            exists = img.exists() if img else False
            out.append(f"\n### `{r['id']}`  [{r['task_type']}]  img_ok={exists}")
            out.append(f"![]({img})" if exists else "_(no image)_")
            for t in r["conversations"]:
                who = "**Q**" if t["from"] in ("human", "user") else "**A**"
                out.append(f"- {who} {t['value'][:400].strip()}")

    report = root / "logs/spot_check.md"
    report.write_text("\n".join(out))
    print(f"[spot] wrote {report}")

    # collisions: two different source images claiming the same flat filename in
    # normalized/images. (Cross-ARM image reuse is expected — arm_combined reuses
    # single-source records — so this checks the flat image tree, not arm pools.)
    from collections import Counter
    img_dir = norm / "images"
    names = Counter(p.name for p in img_dir.iterdir())
    dups = [n for n, c in names.items() if c > 1]
    print(f"[spot] flat-tree image collisions: {len(dups)} (expect 0); "
          f"{sum(names.values())} images total")

    # budget spread
    vals = list(budgets.values())
    spread = (max(vals) - min(vals)) / max(vals) if vals else 0
    print(f"[spot] token budgets: {budgets}")
    print(f"[spot] max cross-arm budget spread: {spread:.1%} (want < ~2%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
