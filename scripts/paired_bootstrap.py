"""Paired bootstrap over images for the micro-F1 delta between two eval runs.

Both runs must have been scored on the same split, so every image_id in one
appears in the other; the pairing is what removes per-image difficulty from the
comparison and makes the CI far tighter than two independent bootstraps.

Reads `parsed` and `gt` straight out of raw_responses.jsonl -- these are the
sets the reported micro-F1 was computed from, so no parser runs here and the
point estimate reproduces test_eval.json exactly.

    python scripts/paired_bootstrap.py --a <eval_dir> --b <eval_dir> [--n 2000]
    python scripts/paired_bootstrap.py --grid   # every arm/cell pair below
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

R = pathlib.Path(
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/results/vlm_eval"
)


def read_jsonl(path: pathlib.Path) -> list[dict]:
    # iterate the handle rather than splitlines(): at least one raw_responses
    # file contains a literal U+2028, which str.splitlines() treats as a break
    # and which then tears the JSON record in half.
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def counts_per_image(run_dir: pathlib.Path) -> tuple[list[str], np.ndarray]:
    """-> (image_ids, [n_images, 3] array of per-image tp, fp, fn)."""
    recs = read_jsonl(run_dir / "raw_responses.jsonl")
    ids, rows = [], []
    for r in recs:
        pred, gt = set(r.get("parsed") or []), set(r.get("gt") or [])
        ids.append(str(r["image_id"]))
        rows.append((len(pred & gt), len(pred - gt), len(gt - pred)))
    return ids, np.asarray(rows, dtype=np.int64)


def micro_f1(c: np.ndarray) -> float:
    tp, fp, fn = c.sum(axis=0)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom else 0.0


def paired_bootstrap(
    a_dir: pathlib.Path, b_dir: pathlib.Path, n_boot: int, seed: int
) -> dict:
    a_ids, a_c = counts_per_image(a_dir)
    b_ids, b_c = counts_per_image(b_dir)

    # align on image_id; a positional zip would silently mis-pair if either run
    # emitted its images in a different order.
    b_index = {i: k for k, i in enumerate(b_ids)}
    missing = [i for i in a_ids if i not in b_index]
    if missing:
        raise SystemExit(
            f"{len(missing)} image_ids in {a_dir.name} absent from {b_dir.name} "
            f"(first: {missing[:3]}) -- runs are not on the same split"
        )
    order = np.array([b_index[i] for i in a_ids])
    b_c = b_c[order]

    f1_a, f1_b = micro_f1(a_c), micro_f1(b_c)
    rng = np.random.default_rng(seed)
    n = len(a_ids)
    deltas = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[k] = micro_f1(b_c[idx]) - micro_f1(a_c[idx])
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "n_images": n,
        "f1_a": f1_a,
        "f1_b": f1_b,
        "delta": f1_b - f1_a,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        # share of resamples on the other side of zero: a two-sided p-value proxy
        "p_two_sided": float(2 * min((deltas <= 0).mean(), (deltas >= 0).mean())),
    }


# the 768/ps2 150K arm predates the stage component in OUTNAME for dw
BASELINE = {
    ("aw_m2", "closed_vocab"): "vlm_cradiov4-so_r768ps2_finetune_aw_m2_closed_vocab",
    ("aw_m2", "open_cot"): "vlm_cradiov4-so_r768ps2_finetune_aw_m2_open_cot",
    ("aw_m4", "closed_vocab"): "vlm_cradiov4-so_r768ps2_finetune_aw_m4_closed_vocab",
    ("aw_m4", "open_cot"): "vlm_cradiov4-so_r768ps2_finetune_aw_m4_open_cot",
    ("dw_paper10", "closed_vocab"): "vlm_cradiov4-so_r768ps2_dw_paper10_closed_vocab",
    ("dw_paper10", "open_cot"): "vlm_cradiov4-so_r768ps2_dw_paper10_open_cot",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=pathlib.Path, help="baseline eval dir")
    ap.add_argument("--b", type=pathlib.Path, help="candidate eval dir")
    ap.add_argument(
        "--grid",
        metavar="PATTERN",
        help="compare every baseline cell against PATTERN.format(ds=..., ps=...)",
    )
    ap.add_argument("--n", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.grid:
        hdr = f"{'cell':26s} {'base':>7s} {'cand':>7s} {'delta':>8s} {'95% CI':>18s} {'p':>7s}"
        print(hdr)
        print("-" * len(hdr))
        for (ds, ps), base in BASELINE.items():
            cand = R / args.grid.format(ds=ds, ps=ps)
            if not (cand / "raw_responses.jsonl").exists():
                print(f"{ds + '/' + ps:26s}   -- candidate not run --")
                continue
            r = paired_bootstrap(R / base, cand, args.n, args.seed)
            sig = "*" if r["ci_lo"] > 0 or r["ci_hi"] < 0 else " "
            print(
                f"{ds + '/' + ps:26s} {r['f1_a']:7.3f} {r['f1_b']:7.3f} "
                f"{r['delta']:+8.3f} [{r['ci_lo']:+6.3f},{r['ci_hi']:+6.3f}]{sig} "
                f"{r['p_two_sided']:7.3f}"
            )
        return

    if not (args.a and args.b):
        raise SystemExit("need --a and --b, or --grid")
    r = paired_bootstrap(args.a, args.b, args.n, args.seed)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
