"""Does SigLIP2 know when it is confused, and is that knowledge usable?

The confusion matrix said the zero-shot failures are sibling collisions inside the
EWC parents -- Plastic packaging absorbed by Plastic, Excavation by Rubble and C&D,
Wood by Pallets. If the encoder is *unsure* in exactly those cases, the confusion
is detectable at inference without labels, and a second stage can be spent only
where it is needed. If it is confidently wrong, no amount of routing helps and the
prompts have to carry the whole job.

That is a question about the runners-up, so it needs the full similarity vector
rather than the argmax. Three signals, all label-free:

  margin   top1 - top2. Directly measures "two names fit this equally well",
           which is the failure mode actually observed.
  entropy  of softmax(sim / tau) over the class set; sees mass spread across
           several classes rather than only the nearest rival.
  top1     the raw similarity, i.e. "does anything fit at all".

Judged by selective prediction rather than by correlation: sort by confidence,
and ask what accuracy is available at each coverage. A signal that lifts accuracy
sharply as coverage drops is one you can route on; a flat curve means the encoder
is confidently wrong and routing buys nothing.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import Counter

import numpy as np


def softmax(x, tau=0.01):
    z = (x - x.max(-1, keepdims=True)) / tau
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def auroc(score, correct):
    """Mann-Whitney U: P(confidence higher on a correct case than a wrong one)."""
    pos = score[correct == 1]
    neg = score[correct == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    # average ties so a constant score scores 0.5 rather than 1.0
    vals = np.concatenate([pos, neg])
    for v in np.unique(vals):
        m = vals == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--tau", type=float, default=0.01)
    args = ap.parse_args()

    d = json.loads(pathlib.Path(args.result).read_text())
    if "sims" not in d:
        raise SystemExit("this result predates --sims; re-run the arm to record them")
    cats = d["cats"]
    sim = np.array(d["sims"], dtype=np.float32)
    y = np.array(d["y_true"])
    pred = sim.argmax(1)
    correct = (pred == y).astype(int)

    srt = np.sort(sim, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    p = softmax(sim, args.tau)
    entropy = -(p * np.log(p + 1e-12)).sum(1)
    top1 = srt[:, -1]

    print(f"\n{len(y)} objects, {len(cats)} classes, overall accuracy {correct.mean():.3f}\n")
    print("  does the signal know when it is wrong?  (AUROC, 0.5 = knows nothing)")
    sigs = {"margin (top1-top2)": margin, "negative entropy": -entropy, "top1 similarity": top1}
    for name, sc in sigs.items():
        print(f"    {name:22s} {auroc(sc, correct):.3f}")

    print("\n  selective prediction on the margin: accuracy if we only answer the")
    print("  most confident fraction, and defer the rest to a second stage")
    order = np.argsort(-margin)
    print(f"    {'coverage':>9s}  {'accuracy':>9s}  {'lift':>7s}")
    for cov in (0.2, 0.4, 0.6, 0.8, 1.0):
        k = max(1, int(cov * len(order)))
        a = correct[order[:k]].mean()
        print(f"    {cov:9.0%}  {a:9.3f}  {a - correct.mean():+7.3f}")

    lo = order[int(0.7 * len(order)):]          # the least confident 30%
    print(f"\n  in the least-confident 30% ({len(lo)} objects, accuracy "
          f"{correct[lo].mean():.3f}), the top-2 pairs are:")
    pairs = Counter()
    for i in lo:
        a, b = sim[i].argsort()[-2:][::-1]
        pairs[tuple(sorted((cats[a], cats[b])))] += 1
    for (a, b), n in pairs.most_common(8):
        same_parent = ""
        print(f"    {n:4d}  {a}  vs  {b}{same_parent}")

    print(f"\n  and how often the TRUE class is the runner-up rather than the top:")
    top2 = sim.argsort(1)[:, -2]
    rescued = ((pred != y) & (top2 == y)).sum()
    wrong = (pred != y).sum()
    print(f"    {rescued}/{wrong} of errors ({rescued/max(1,wrong):.1%}) have the right "
          f"answer in second place — the ceiling for any pairwise re-ranking stage")


if __name__ == "__main__":
    main()
