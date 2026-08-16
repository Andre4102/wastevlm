"""How few binary labels does the decision threshold actually need?

Fitting the operating point on AW's full 3,689-image train split works, but it
spends a lot of supervision on ONE scalar and muddies the zero-shot claim. Two
cheaper protocols are worth pricing, and both can be answered offline from the
per-image margins already dumped by `scripts/vlm_binary_auc.py` -- no GPU.

  budget   fit the threshold on n randomly drawn images, score the REST. The
           calibration labels are binary present/absent only, so no category is
           ever used for fitting and the multi-label eval stays category-zero-shot.
           1 fitted parameter, disclosed exactly.

  transfer take a threshold fitted on a DIFFERENT dataset and apply it as-is.
           If that held, calibration would cost zero in-domain labels.

Draws are random, not stratified: a real calibration set is "label the next n
images", and stratifying would quietly assume you already know which are
positive.

    python scripts/calib_budget.py
"""

from __future__ import annotations

import glob
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/results/vlm_eval")
SIZES = [10, 20, 30, 50, 75, 100, 200, 400]
DRAWS = 400


def best_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """Threshold maximising Youden J. Midpoints between observed scores, so the
    cut does not sit exactly on a training point it was chosen from."""
    order = np.argsort(s)
    s_sorted = np.unique(s[order])
    if len(s_sorted) < 2:
        return float(s_sorted[0]) if len(s_sorted) else 0.0
    cands = np.concatenate([[s_sorted[0] - 1.0],
                            (s_sorted[:-1] + s_sorted[1:]) / 2.0,
                            [s_sorted[-1] + 1.0]])
    npos, nneg = max((y == 1).sum(), 1), max((y == 0).sum(), 1)
    pred = s[None, :] >= cands[:, None]
    tpr = (pred & (y == 1)[None, :]).sum(1) / npos
    fpr = (pred & (y == 0)[None, :]).sum(1) / nneg
    return float(cands[np.argmax(tpr - fpr)])


def youden(y: np.ndarray, s: np.ndarray, thr: float) -> float:
    pred = s >= thr
    npos, nneg = (y == 1).sum(), (y == 0).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((pred & (y == 1)).sum() / npos - (pred & (y == 0)).sum() / nneg)


def load(pattern: str):
    hits = sorted(glob.glob(str(ROOT / pattern / "binary_auc.json")))
    if not hits:
        return None
    d = json.load(open(hits[0]))
    return np.array(d["labels"]), np.array(d["scores"]), d


def main() -> None:
    runs = {
        "aw_m2": "binauc_cradiov4-so_r768ps2_finetune_next_aw_m2",
        "aw_m4": "binauc_cradiov4-so_r768ps2_finetune_next_aw_m4",
        "dw": "binauc_cradiov4-so_r768ps2_finetune_next_dw_paper10",
    }
    data = {k: load(v) for k, v in runs.items()}
    data = {k: v for k, v in data.items() if v is not None}

    print("=== calibration budget: fit on n images (binary labels only), "
          f"score the remainder | {DRAWS} draws\n")
    print(f"{'dataset':8s} {'n':>5s}  {'median J':>9s} {'p10':>7s} {'p90':>7s}"
          f"  {'% of full-fit':>14s}")
    full = {}
    for name, (y, s, d) in data.items():
        thr_full = best_threshold(y, s)
        full[name] = youden(y, s, thr_full)
    for name, (y, s, d) in data.items():
        rng = np.random.default_rng(0)
        n_all = len(y)
        for n in SIZES:
            if n >= n_all:
                continue
            js = []
            for _ in range(DRAWS):
                idx = rng.choice(n_all, size=n, replace=False)
                mask = np.zeros(n_all, bool); mask[idx] = True
                if (y[mask] == 1).sum() == 0 or (y[mask] == 0).sum() == 0:
                    continue  # a draw with one class present cannot fit a cut
                thr = best_threshold(y[mask], s[mask])
                js.append(youden(y[~mask], s[~mask], thr))
            js = np.array([j for j in js if not np.isnan(j)])
            if not len(js):
                continue
            med = float(np.median(js))
            print(f"{name:8s} {n:5d}  {med:9.4f} {np.percentile(js,10):7.4f} "
                  f"{np.percentile(js,90):7.4f}  {100*med/full[name]:13.1f}%")
        print(f"{name:8s} {'FULL':>5s}  {full[name]:9.4f} {'':7s} {'':7s}"
              f"  {100.0:13.1f}%")
        print()

    print("=== transfer: threshold fitted on A, applied to B (no in-domain labels)\n")
    print(f"{'fit on':8s} -> {'applied to':10s} {'thr':>8s} {'J':>8s} "
          f"{'own-fit J':>10s} {'kept':>7s}")
    for src, (ys, ss, _) in data.items():
        thr = best_threshold(ys, ss)
        for dst, (yd, sd, _) in data.items():
            if src == dst:
                continue
            j = youden(yd, sd, thr)
            print(f"{src:8s} -> {dst:10s} {thr:8.3f} {j:8.4f} "
                  f"{full[dst]:10.4f} {100*j/full[dst]:6.1f}%")


if __name__ == "__main__":
    main()
