"""Let each class choose its own prompts, from a signal available at runtime.

Neither bank wins globally. Against the contrastive baseline, widening the prompts
costs the eleven common classes (5.01x -> 4.52x chance) and buys the seven rare ones
(3.29x -> 3.76x), leaving overall macro-recall unchanged at 9.2x. A single bank is
therefore the wrong unit of decision: the choice belongs per class.

Making it per class is only useful if the chooser runs without labels, which is the
whole constraint -- precision and recall are not observable at deployment. Two
signals from `class_diagnostics.py` are, and both were checked against the labelled
asymmetry first: `rival_entropy`, the spread of whoever beats a class when it is
competitive and loses (Spearman -0.710 against the precision-recall gap), and
`near_miss`, how often it is competitive at all (-0.684 against recall).

Which way the rule should point is a measurement, not a guess -- an earlier guess
about this was inverted -- so both directions are scored, along with the two
single-bank baselines and an oracle that picks per class using the labels. The
oracle is the ceiling for any chooser and is never a deployable arm.

Everything re-scores from cached embeddings, so a policy costs seconds rather than
a pass over the images.
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


def macro_recall(y, pred, ncls):
    r = [float((pred[y == c] == c).mean()) for c in range(ncls) if (y == c).sum()]
    return float(np.mean(r))


def score_bank(emb, cats, banks, encode_text, cache):
    """-> [N, C] similarity, one column per class, using that class's chosen prompts."""
    import torch

    cols = []
    for c in cats:
        ps = tuple(banks[c])
        if ps not in cache:
            t = encode_text(list(ps))
            cache[ps] = torch.nn.functional.normalize(
                t.mean(0), dim=-1).cpu().numpy()
        cols.append(cache[ps])
    return emb @ np.stack(cols).T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="stage-1 json with sims + y_true")
    ap.add_argument("--emb")
    ap.add_argument("--expanded-bank", required=True)
    ap.add_argument("--dataset", default="dronewaste")
    ap.add_argument("--rival-threshold", type=float, default=0.45)
    ap.add_argument("--out-json")
    args = ap.parse_args()

    from src.prompt_sets import build as build_prompts, load_bank
    from src.radio_adaptors import siglip2_text
    from scripts.roi_material import cue_prompts

    d = json.loads(pathlib.Path(args.result).read_text())
    cats, y = d["cats"], np.array(d["y_true"])
    sim0 = np.array(d["sims"], np.float32)
    emb = np.load(args.emb or str(pathlib.Path(args.result).with_suffix(".emb.npy")))
    C = len(cats)

    base = cue_prompts(args.dataset, cats)
    contr = build_prompts(cats, base, "contrastive")
    expan = build_prompts(cats, base, "expanded", load_bank(args.expanded_bank))

    # --- label-free diagnosis from the first pass, exactly as at deployment
    pred0 = sim0.argmax(1)
    rank3 = (-sim0).argsort(1)[:, :3]
    diag = {}
    for i, c in enumerate(cats):
        near = [j for j, r in enumerate(rank3) if i in r and pred0[j] != i]
        share = 0.0
        if near:
            w = np.bincount([pred0[j] for j in near], minlength=C).astype(float)
            share = float((w / w.sum()).max())
        wr = float((pred0 == i).mean())
        t3 = float(np.mean([i in r for r in rank3]))
        diag[c] = {"rival_top_share": share, "near_miss": (t3 / wr) if wr else np.inf}

    concentrated = [c for c in cats
                    if diag[c]["rival_top_share"] > args.rival_threshold]
    print(f"[adaptive] {len(concentrated)}/{C} classes lose to one dominant rival: "
          f"{', '.join(concentrated[:6])}{' ...' if len(concentrated) > 6 else ''}")

    encode_text = siglip2_text(device="cuda")
    cache = {}
    policies = {
        "all contrastive": {c: contr[c] for c in cats},
        "all expanded": {c: expan[c] for c in cats},
        "expand the diffuse losers": {
            c: (contr[c] if c in concentrated else expan[c]) for c in cats},
        "expand the concentrated losers": {
            c: (expan[c] if c in concentrated else contr[c]) for c in cats},
    }

    out = {}
    print(f"\n  {'policy':32s} {'acc':>7s} {'macro-recall':>13s} {'xchance':>8s}")
    for name, bank in policies.items():
        s = score_bank(emb, cats, bank, encode_text, cache)
        p = s.argmax(1)
        acc, mr = float((p == y).mean()), macro_recall(y, p, C)
        print(f"  {name:32s} {acc:7.3f} {mr:13.3f} {mr*C:8.2f}x")
        out[name] = {"acc": acc, "macro_recall": mr}

    # oracle: per class, whichever bank gives that class better recall. Uses labels,
    # so it is a ceiling on any chooser and never a deployable arm.
    sc, se = (score_bank(emb, cats, {c: contr[c] for c in cats}, encode_text, cache),
              score_bank(emb, cats, {c: expan[c] for c in cats}, encode_text, cache))
    pc, pe = sc.argmax(1), se.argmax(1)
    pick = {}
    for i, c in enumerate(cats):
        m = y == i
        if not m.sum():
            pick[c] = contr[c]; continue
        pick[c] = expan[c] if (pe[m] == i).mean() > (pc[m] == i).mean() else contr[c]
    s = score_bank(emb, cats, pick, encode_text, cache)
    p = s.argmax(1)
    mr = macro_recall(y, p, C)
    print(f"  {'oracle per-class (uses labels)':32s} {float((p==y).mean()):7.3f} "
          f"{mr:13.3f} {mr*C:8.2f}x   <- ceiling, not deployable")
    out["oracle"] = {"acc": float((p == y).mean()), "macro_recall": mr}
    out["chose_expanded"] = [c for c in cats if pick[c] is expan[c]]
    print(f"\n  the oracle picks the widened bank for: "
          f"{', '.join(out['chose_expanded']) or 'no class'}")

    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"\n[write] {args.out_json}")


if __name__ == "__main__":
    main()
