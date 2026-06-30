"""Aggregate N independent agent 'photo-interpreter' votes into a QA triage.

Each agent inspected the images in a batch and, for every GT (candidate) label,
voted present / absent / unsure, plus listed any unlabeled waste it saw. This
script computes per-(image,label) consensus, inter-rater agreement (Fleiss' kappa),
and emits the subset a human should review: where interpreters DISAGREE, or where
their consensus CONTRADICTS the ground-truth label, or where >=2 agents spotted an
unlabeled material (possible missing GT label).

Usage:
    python -m src.aggregate_votes \
        --batch /home/ids/diecidue/data/captions/qa_calib_batch.json \
        --votes qa_votes_agent1.jsonl qa_votes_agent2.jsonl qa_votes_agent3.jsonl \
        --out-review /home/ids/diecidue/data/captions/qa_review_list.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

CATS = ("present", "absent", "unsure")


def fleiss_kappa(rows: list[list[int]]) -> float:
    """rows[i] = [count_present, count_absent, count_unsure] for item i (sums to n raters)."""
    rows = [r for r in rows if sum(r) > 0]
    if not rows:
        return float("nan")
    N = len(rows)
    n = sum(rows[0])
    if n < 2:
        return float("nan")
    p_j = [sum(r[j] for r in rows) / (N * n) for j in range(len(CATS))]
    P_bar = sum((sum(c * c for c in r) - n) / (n * (n - 1)) for r in rows) / N
    P_e = sum(p * p for p in p_j)
    return (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 1e-9 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--votes", type=Path, nargs="+", required=True)
    ap.add_argument("--out-review", type=Path, required=True)
    args = ap.parse_args()

    gt = {(b["dataset"], str(b["image_id"])): sorted(b["gt_categories"])
          for b in json.loads(args.batch.read_text())}

    # votes[key][label] = list of votes (one per agent)
    votes: dict = defaultdict(lambda: defaultdict(list))
    extra: dict = defaultdict(list)  # key -> list of extra_seen lists (one per agent)
    n_agents = len(args.votes)
    for vf in args.votes:
        for line in vf.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r["dataset"], str(r["image_id"]))
            for lab, v in r.get("votes", {}).items():
                votes[key][lab].append(str(v).lower().strip())
            extra[key].append([e.strip() for e in r.get("extra_seen", []) if e.strip()])

    kappa_rows = []
    review = []
    n_labels = unanimous = majority = gt_conflict = splits = missed = 0

    for key, labs in gt.items():
        rowflags = {"gt_conflict": [], "split": [], "missed_label": []}
        for lab in labs:
            vs = votes.get(key, {}).get(lab, [])
            if not vs:
                continue
            n_labels += 1
            c = Counter(vs)
            kappa_rows.append([c.get("present", 0), c.get("absent", 0), c.get("unsure", 0)])
            if len(set(vs)) == 1:
                unanimous += 1
            present = c.get("present", 0)
            if present > n_agents / 2:
                majority += 1
            else:
                # GT says this label is present, but interpreters don't agree it is
                gt_conflict += 1
                rowflags["gt_conflict"].append(lab)
            if len(set(vs)) > 1:
                splits += 1
                rowflags["split"].append(f"{lab}:{dict(c)}")
        # missed labels: a material >=2 agents saw that isn't in GT
        seen = Counter(e for lst in extra.get(key, []) for e in set(lst))
        miss = [m for m, cnt in seen.items() if cnt >= 2 and m not in labs]
        if miss:
            missed += 1
            rowflags["missed_label"] = miss
        if any(rowflags.values()):
            review.append({"dataset": key[0], "image_id": key[1], "gt": labs, **rowflags})

    args.out_review.parent.mkdir(parents=True, exist_ok=True)
    with args.out_review.open("w") as f:
        for r in review:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    kappa = fleiss_kappa(kappa_rows)
    print(f"agents               : {n_agents}")
    print(f"images judged        : {len(gt)}")
    print(f"label-judgments      : {n_labels}")
    print(f"unanimous            : {unanimous} ({100*unanimous/max(1,n_labels):.0f}%)")
    print(f"majority-present     : {majority} ({100*majority/max(1,n_labels):.0f}%)")
    print(f"GT-conflict labels   : {gt_conflict} (GT present, no interpreter majority)")
    print(f"split (any disagree) : {splits}")
    print(f"images w/ missed-label: {missed}")
    print(f"Fleiss kappa         : {kappa:.3f}")
    print(f"images needing review: {len(review)} / {len(gt)}  -> {args.out_review}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
