"""Spot the classes whose prompts are too narrow, without using labels.

The conservative classes were identified from labelled dev data -- precision 0.883
against recall 0.179 for Textile, 0.846 against 0.094 for Furniture -- and that rule
cannot run at deployment, where there are no labels to compute either number from.
If prompt widening is only applicable where someone has already annotated the site,
it is not part of an open-world system.

So: signals computable from the similarity matrix alone, and a check that they
actually track the labelled asymmetry rather than merely sounding plausible.

  win_rate        how often the class is the argmax
  top3_rate       how often it is in the top three
  near_miss       top3_rate / win_rate. A class that is repeatedly competitive and
                  repeatedly loses is one whose prompts describe a corner of it --
                  the label-free shape of "precise but blind".
  win_margin      mean top1-top2 on the objects it does win. A class that only ever
                  wins narrowly is barely holding its ground.
  coherence       mean pairwise cosine between the class's own prompt embeddings.
                  Low means the prompts do not agree on what the class looks like.

Validation is the point of the script. Each signal is ranked against the labelled
recall and against the precision-minus-recall gap on the development sites; a signal
that fails here is not usable at runtime whatever its story.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return float("nan")

    def rank(x):
        o = np.argsort(x, kind="mergesort")
        r = np.empty(len(x), float)
        r[o] = np.arange(len(x), dtype=float)
        for v in np.unique(x):
            m = x == v
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r

    ra, rb = rank(a), rank(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--out-json")
    args = ap.parse_args()

    d = json.loads(pathlib.Path(args.result).read_text())
    cats = d["cats"]
    sim = np.array(d["sims"], np.float32)
    y = np.array(d["y_true"])
    pred = sim.argmax(1)
    rank3 = (-sim).argsort(1)[:, :3]
    srt = np.sort(sim, 1)
    margin = srt[:, -1] - srt[:, -2]

    rows = {}
    for i, c in enumerate(cats):
        win = pred == i
        wr = float(win.mean())
        t3 = float(np.mean([i in r for r in rank3]))
        rows[c] = {
            "win_rate": wr,
            "top3_rate": t3,
            "near_miss": float(t3 / wr) if wr > 0 else float("inf"),
            "win_margin": float(margin[win].mean()) if win.any() else float("nan"),
        }
        # When the class is competitive but loses, WHO beats it? Losing repeatedly
        # to one rival is a sibling collision and wants disambiguation; losing to
        # many different rivals means the prompt describes a corner of the class
        # and wants widening. That distinction decides which fix to apply, and it
        # needs no labels.
        near = [j for j, r in enumerate(rank3) if i in r and pred[j] != i]
        if near:
            w = np.bincount([pred[j] for j in near], minlength=len(cats)).astype(float)
            w /= w.sum()
            nz = w[w > 0]
            rows[c]["rival_top_share"] = float(nz.max())
            rows[c]["rival_entropy"] = float(-(nz * np.log(nz)).sum() / np.log(len(cats)))
            rows[c]["rival"] = cats[int(w.argmax())]
        else:
            rows[c].update(rival_top_share=float("nan"),
                           rival_entropy=float("nan"), rival="-")
        # labelled, for validation only -- never available at deployment
        idx = y == i
        if idx.sum():
            rec = float((pred[idx] == i).mean())
            pre = float((y[win] == i).mean()) if win.any() else 0.0
            rows[c].update(n=int(idx.sum()), recall=rec, precision=pre, gap=pre - rec)

    have = [c for c in cats if "recall" in rows[c]]
    hdr = ("class", "n", "nearmiss", "winmarg", "rivalH", "top rival",
           "rec", "prec", "P-R", "label-free call")
    print("\n{:32s} {:>4s} {:>8s} {:>8s} {:>7s} {:>20s} | {:>6s} {:>6s} {:>6s}  {}"
          .format(*hdr))
    for c in sorted(have, key=lambda c: -rows[c]["gap"]):
        r = rows[c]
        # The rule as it would run at deployment: a class being missed
        # (near_miss high) is widened when it loses diffusely and disambiguated
        # when one rival takes most of its losses.
        call = ""
        if r["near_miss"] > 3.0 and r["n"] >= 20:
            call = "DISAMBIGUATE" if r.get("rival_top_share", 0) > 0.45 else "WIDEN"
        print("{:32s} {:4d} {:8.2f} {:8.4f} {:7.3f} {:>20s} | "
              "{:6.3f} {:6.3f} {:+6.3f}  {}".format(
                  c[:32], r["n"], r["near_miss"], r["win_margin"],
                  r.get("rival_entropy", float("nan")),
                  str(r.get("rival", "-"))[:20], r["recall"], r["precision"],
                  r["gap"], call))

    print("\nDoes a label-free signal track the labelled asymmetry?  (Spearman)")
    print(f"  {'signal':12s} {'vs recall':>11s} {'vs (P-R) gap':>14s}")
    for sig in ("win_rate", "top3_rate", "near_miss", "win_margin",
                "rival_top_share", "rival_entropy"):
        v = [rows[c][sig] if np.isfinite(rows[c][sig]) else np.nan for c in have]
        print(f"  {sig:12s} {spearman(v, [rows[c]['recall'] for c in have]):11.3f} "
              f"{spearman(v, [rows[c]['gap'] for c in have]):14.3f}")
    print("\n  A signal that ranks classes by how much they are being missed should")
    print("  show a strong POSITIVE correlation with the P-R gap and a NEGATIVE one")
    print("  with recall. near_miss is the candidate; the numbers decide.")

    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(rows, indent=2))
        print(f"\n[write] {args.out_json}")


if __name__ == "__main__":
    main()
