"""Re-score the confused objects by moving the text vectors, not by writing new text.

Sibling collision is a geometric fact before it is a linguistic one. "Plastic" and
"Plastic packaging" point almost the same way in SigLIP2's space, so the similarity
each earns is dominated by what they share and the part that separates them is a
thin residual. Writing better sentences is one way at that residual; subtracting the
shared component is a more direct one, and it needs no decoder, no training and no
labels.

Three operations on the candidate set's text vectors, all label-free:

  centered    subtract the mean of the candidates being decided between. What is
              common to all five carries no information about which one it is.
  global      subtract the mean over all classes instead -- the standard hubness
              correction, which removes the direction every class shares rather
              than the one this particular group shares.
  whitened    centre, then divide each coordinate by its spread across the
              candidates, so a dimension on which they barely differ stops
              dominating the dot product.

Scored against the same two references as the decoder arm: doing nothing, and the
hand-written contrastive prompts. Anything here that beats 0.348 on the deferred
set is doing what the decoder could not.
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


def norm(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True)
    ap.add_argument("--emb")
    ap.add_argument("--dataset", default="dronewaste")
    ap.add_argument("--defer", type=float, default=0.30)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--out-json")
    args = ap.parse_args()

    import torch

    from src.prompt_sets import build as build_prompts
    from src.radio_adaptors import siglip2_text
    from scripts.roi_material import cue_prompts

    d = json.loads(pathlib.Path(args.result).read_text())
    cats, y = d["cats"], np.array(d["y_true"])
    sim0 = np.array(d["sims"], np.float32)
    emb = np.load(args.emb or str(pathlib.Path(args.result).with_suffix(".emb.npy")))
    C = len(cats)

    # the same bank the winning `fixed` arm used, so the comparison is like for like
    prompts = build_prompts(cats, cue_prompts(args.dataset, cats), "contrastive")
    encode_text = siglip2_text(device="cuda")
    T = np.stack([norm(encode_text(prompts[c]).cpu().numpy().mean(0)) for c in cats])

    rank = (-sim0).argsort(1)
    pred1 = rank[:, 0]
    p = np.exp((sim0 - sim0.max(1, keepdims=True)) / 0.01)
    p /= p.sum(1, keepdims=True)
    entropy = -(p * np.log(p + 1e-12)).sum(1)
    deferred = np.argsort(-entropy)[: int(args.defer * len(y))]

    base_acc = float((pred1[deferred] == y[deferred]).mean())
    cont = float(np.mean([y[i] in rank[i, :args.topk] for i in deferred]))
    print(f"\ndeferred {len(deferred)} objects; stage 1 {base_acc:.3f}; "
          f"top-{args.topk} ceiling {cont:.3f}")
    print("  references: hand-written contrastive re-score reached 0.348, "
          "decoder-written 0.133\n")

    gmean = T.mean(0)
    out = {}
    for name in ("as-is", "centered", "global", "whitened"):
        pred2 = pred1.copy()
        for i in deferred:
            cand = rank[i, :args.topk]
            V = T[cand]
            if name == "centered":
                V = norm(V - V.mean(0))
            elif name == "global":
                V = norm(V - gmean)
            elif name == "whitened":
                V = V - V.mean(0)
                V = norm(V / (V.std(0) + 1e-6))
            pred2[i] = cand[int(np.argmax(V @ emb[i]))]
        acc = float((pred2[deferred] == y[deferred]).mean())
        overall = float((pred2 == y).mean())
        print(f"  {name:10s} deferred {acc:.3f}   overall {overall:.3f}")
        out[name] = {"deferred": acc, "overall": overall}

    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"\n[write] {args.out_json}")


if __name__ == "__main__":
    main()
