"""Per-class results for the naming arms, with the supervised and zero-shot routes side by side.

Pooling the rare classes hides the thing worth seeing. DroneWaste's tail is five
classes holding 61 objects between them, one of which has a single instance, and a
pooled "tail = 0.47x chance" says nothing about which of those five the encoder can
actually find. It also invites a reader to average over an n of 1 as though it were
an estimate.

So: every class on its own row, with its support, and a Wilson interval rather than
a bare rate. Wilson because at n=1 or n=3 the normal approximation is nonsense and
a point estimate pretends to a precision that is not there -- a class scoring 1/1
should read as "somewhere between 21% and 100%", which is the honest statement.

The two routes are shown together because they fail in different places and that
difference is the argument:

  supervised  ROI tokens + a linear head. Strong, and needs labelled instances --
              so it inherits exactly the problem a trained detector has, and can
              say nothing about a class it never saw.
  zero-shot   the SigLIP2 head against text. Weaker on the head classes, and needs
              no instances at all, so it is the only route that reaches the tail.

    python scripts/per_class_report.py --zeroshot <json> --probe <json>
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import Counter

TAIL_MAX = 100


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion; honest where the normal approximation is not."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def per_class(cats, y_true, y_pred):
    out = {}
    for i, c in enumerate(cats):
        idx = [j for j, y in enumerate(y_true) if y == i]
        hit = sum(1 for j in idx if y_pred[j] == i)
        npred = sum(1 for p in y_pred if p == i)
        tp = hit
        rec = (hit / len(idx)) if idx else float("nan")
        pre = (tp / npred) if npred else float("nan")
        f1 = (2 * pre * rec / (pre + rec)) if (pre and rec and pre + rec > 0) else 0.0
        out[c] = {"n": len(idx), "recall": rec, "ci": wilson(hit, len(idx)),
                  "precision": pre, "f1": f1, "n_pred": npred}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zeroshot")
    ap.add_argument("--probe")
    ap.add_argument("--zs-mode", default="roi-head")
    ap.add_argument("--probe-mode", default="roi_mean")
    args = ap.parse_args()

    cats = y_true = None
    cols = {}
    if args.zeroshot and pathlib.Path(args.zeroshot).exists():
        d = json.loads(pathlib.Path(args.zeroshot).read_text())
        cats, y_true = d["cats"], d["y_true"]
        if args.zs_mode in d.get("pred", {}):
            cols["zero-shot"] = per_class(cats, y_true, d["pred"][args.zs_mode])
    if args.probe and pathlib.Path(args.probe).exists():
        d = json.loads(pathlib.Path(args.probe).read_text())
        if args.probe_mode in d.get("pred", {}):
            # the probe holds out sites, so its support differs from the zero-shot run
            cols["supervised"] = per_class(d["cats"], d["y_true"], d["pred"][args.probe_mode])
            cats = cats or d["cats"]
    if not cols:
        raise SystemExit("no per-object predictions found; re-run the arms with the "
                         "current scripts, which write them")

    ref = cols.get("zero-shot") or cols.get("supervised")
    order = sorted(cats, key=lambda c: -ref[c]["n"])
    print(f"\n{'class':38s} {'n':>5s} | " +
          " | ".join(f"{k:>34s}" for k in cols))
    print(f"{'':38s} {'':>5s} | " +
          " | ".join(f"{'recall [95% CI]      P     F1':>34s}" for _ in cols))
    print("-" * (46 + 37 * len(cols)))
    for c in order:
        tail = ref[c]["n"] < TAIL_MAX
        line = f"{c:38s} {ref[c]['n']:5d} |"
        for k, t in cols.items():
            e = t.get(c)
            if e is None or e["n"] == 0:
                line += f" {'--':>26s} |"
            else:
                lo, hi = e["ci"]
                line += (f" {e['recall']:.3f} [{lo:.2f},{hi:.2f}]  "
                         f"{e['precision']:.3f} {e['f1']:.3f} |")
        print(line + ("  TAIL" if tail else ""))

    print()
    # head/tail comes from ONE support definition, the reference column's. Taking
    # it per column made each arm report a different number of head classes purely
    # because their test sets differ in size, which reads as a finding and is not.
    head = [c for c in cats if ref.get(c, {}).get("n", 0) >= TAIL_MAX]
    tail = [c for c in cats if 0 < ref.get(c, {}).get("n", 0) < TAIL_MAX]
    for k, t in cols.items():
        n_here = sum(t.get(c, {}).get("n", 0) for c in cats)
        print(f"  [{k}] scored {n_here} objects", end="  ")
        found = [c for c in tail if t.get(c, {}).get("n", 0) and t[c]["recall"] > 0]
        hd = [c for c in head if t.get(c, {}).get("n", 0)]
        print(f"head {len(hd):2d} classes, mean recall "
              f"{sum(t[c]['recall'] for c in hd)/max(1,len(hd)):.3f}"
              f"   |  tail {len(tail)} classes, "
              f"{len(found)} with any recall at all: {', '.join(found) or 'none'}")


if __name__ == "__main__":
    main()
