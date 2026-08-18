"""Does the model know where "bottom-left" is, or is it reciting a prior?

The correlational evidence (scripts/check_grounding.py) shows that images the model
calls "left" have their ground-truth waste further left than images it calls
"right". That is consistent with grounding, but also with any confound that
happens to correlate with both the words and the boxes.

This is the causal version. Mirror the image and ask the same question. A model
that reads position off the token grid must swap left and right; a model emitting
a content-driven prior ("dumps sit near the bottom of aerial tiles") must not.
Nothing about the scene content changes under a mirror -- same materials, same
textures, same context -- so content-driven explanations predict no swap.

The mechanism is at least available to it: src/vlm_model.py:_shuffle folds the
patch grid row-major, so visual token i is grid cell (i // g, i % g) at a fixed
RoPE position for every image. Whether the decoder learned that convention is
what this measures.

Vertical flip is reported too but reads differently: mirroring an aerial tile
left-right leaves a plausible aerial tile, while flipping it top-bottom inverts
shadow direction, which is a real cue the encoder may key on. Horizontal is the
clean test; vertical is a bonus with a caveat.

    python scripts/flip_probe.py --generate --ckpt <dir> --encoder cradiov4-so \
        --image-size 768 --pixel-shuffle 2 --dataset aw_m2 --out flip.json
    python scripts/flip_probe.py --report flip.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.check_grounding import spoken_position  # noqa: E402


def generate(args) -> None:
    from PIL import Image

    from src.vlm_eval import PROMPT_DESCRIBE, WasteVLMAdapter
    from scripts.make_convo import load_samples

    _cats, samples = load_samples(args.dataset)
    # The AW split indexes more images than the release ships; the eval skips the
    # gaps silently (581 rows from a 690-image split), so this must skip them too
    # or the conditions come out on different image sets.
    pos = [s for s in samples
           if s.extra["gt_categories"] and pathlib.Path(s.image_path).exists()]
    if args.limit:
        pos = pos[: args.limit]
    print(f"[flip] {len(pos)} positive images x 3 conditions", flush=True)

    adapter = WasteVLMAdapter(args.ckpt, encoder=args.encoder,
                              image_size=args.image_size,
                              pixel_shuffle=args.pixel_shuffle,
                              max_new_tokens=args.max_new_tokens)
    adapter.load()

    out = []
    for n, s in enumerate(pos):
        img = Image.open(s.image_path).convert("RGB")
        rec = {"image_id": s.image_id, "image_path": str(s.image_path),
               "gt_categories": s.extra["gt_categories"]}
        for cond, im in (("orig", img),
                         ("hflip", img.transpose(Image.FLIP_LEFT_RIGHT)),
                         ("vflip", img.transpose(Image.FLIP_TOP_BOTTOM))):
            rec[cond] = adapter.generate(im, PROMPT_DESCRIBE)
        out.append(rec)
        if n % 20 == 0:
            print(f"[flip] {n}/{len(pos)}", flush=True)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[write] {args.out}  ({len(out)} images)")


def binom_p(k: int, n: int, p: float = 0.5) -> float:
    """Two-sided exact binomial p-value."""
    from math import comb
    if n == 0:
        return float("nan")
    probs = [comb(n, i) * p**i * (1 - p)**(n - i) for i in range(n + 1)]
    return min(1.0, sum(q for q in probs if q <= probs[k] * (1 + 1e-9)))


def report(args) -> None:
    recs = json.loads(pathlib.Path(args.report).read_text())
    print(f"\n=== flip probe on {len(recs)} positives")

    for cond, axis, lo, hi in (("hflip", "x", "left", "right"),
                               ("vflip", "y", "top", "bottom")):
        pairs = []
        for r in recs:
            a = spoken_position(r["orig"])[axis]
            b = spoken_position(r[cond])[axis]
            if a in (-1, 1) and b in (-1, 1):
                pairs.append((a, b))
        if not pairs:
            print(f"\n  {cond}: no image gave a definite {lo}/{hi} in both conditions")
            continue
        swapped = sum(1 for a, b in pairs if a != b)
        n = len(pairs)
        # Null: the utterance ignores the image, so it repeats. Under a grounded
        # model it inverts. A coin-flip model sits at the marginal rate below.
        marg = sum(1 for a, _ in pairs if a == 1) / n
        chance = 2 * marg * (1 - marg)      # P(differ) if the second draw is iid
        print(f"\n  {cond}: {n} images said {lo} or {hi} in BOTH conditions")
        print(f"    flipped the word: {swapped}/{n} = {swapped/n:.0%}"
              f"   (repeat-the-prior predicts 0%, grounded predicts 100%,"
              f" iid-guessing predicts {chance:.0%})")
        # The null that matters is not a coin flip, it is "the word is drawn from
        # the same marginal no matter where the waste is", which predicts `chance`.
        print(f"    p vs position-blind null ({chance:.0%}): "
              f"{binom_p(swapped, n, chance):.3g}"
              f"   |  p vs fully-grounded (100%): {0.5**n if swapped < n else 1.0:.3g}")
        kept = [(a, b) for a, b in pairs if a == b]
        if kept:
            side = "".join(sorted({lo if a == -1 else hi for a, _ in kept}))
            print(f"    unchanged on {len(kept)} ({side or '-'}) ")

    for cond in ("hflip", "vflip"):
        same = sum(1 for r in recs if r["orig"].strip() == r[cond].strip())
        spoke = sum(1 for r in recs if spoken_position(r["orig"])["x"] in (-1, 1)
                    or spoken_position(r["orig"])["y"] in (-1, 1))
        print(f"\n  {cond}: description byte-identical on {same}/{len(recs)} "
              f"({same/len(recs):.0%})   [orig places waste on {spoke}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--report")
    ap.add_argument("--ckpt")
    ap.add_argument("--encoder", default="cradiov4-so")
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--pixel-shuffle", type=int, default=2)
    ap.add_argument("--dataset", default="aw_m2")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", default="flip.json")
    args = ap.parse_args()
    if args.generate:
        generate(args)
    if args.report:
        report(args)


if __name__ == "__main__":
    main()
