"""Name the material from the full image at native resolution, pooling only the object.

Every earlier naming result handicapped AerialWaste in one of two ways, and the
objection that they both dodge the question is fair.

The resolution sweep fed the whole image but pooled it globally (cls/mean/max).
A 40px object is one token out of 576; at 1024px it is one token out of 4096.
Global pooling dilutes it by exactly as much either way, so a flat sweep is what
that design produces whether or not resolution matters. It could not have
detected the effect it was cited as ruling out.

The ceiling experiment avoided the dilution by cropping to the object, but then
threw the context away and handed a 40px crop to a 224px encoder.

This does neither. The full image goes in at its native size, and only the tokens
whose receptive field lands inside the ground-truth box are pooled. Context is
available to the encoder, nothing is resampled away, and the object is not
competing with 4000 background tokens. At 1024px with patch 16 a token covers
15.6px, so AerialWaste's median object spans ~6.6 tokens instead of 0.9.

If naming still sits at the majority baseline here, it is not the framing, not
the pooling and not the resolution.

    python scripts/roi_token_probe.py --dataset aw_m2 --image-size 1024
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import Counter

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.roi_material import load_rois  # noqa: E402
from src.vision_encoder import VisionEncoder  # noqa: E402


def roi_pool(patches, grid: int, box, native, pad: float):
    """Mean- and max-pool the tokens covering `box`.

    `box` is in the original image's pixels and `native` is that image's size, so
    the box is mapped to the token grid by fraction of the image rather than by
    the resized pixel count. A box thinner than one token still selects the token
    it falls in: an empty selection would silently become a zero vector.
    """
    import torch

    x, y, w, h = box
    W, H = native
    cx, cy = w * pad, h * pad
    x0 = int(np.floor((x - cx) / W * grid))
    y0 = int(np.floor((y - cy) / H * grid))
    x1 = int(np.ceil((x + w + cx) / W * grid))
    y1 = int(np.ceil((y + h + cy) / H * grid))
    x0, y0 = max(0, min(grid - 1, x0)), max(0, min(grid - 1, y0))
    x1, y1 = max(x0 + 1, min(grid, x1)), max(y0 + 1, min(grid, y1))

    g = patches.reshape(grid, grid, -1)[y0:y1, x0:x1].reshape(-1, patches.shape[-1])
    return (g.mean(0), g.max(0).values, (y1 - y0) * (x1 - x0))


def split_by_image(rois, frac=0.25):
    """Hold out whole images, not objects: two crops of one pile are not two samples.

    DroneWaste ships no train/test split, and splitting its objects at random
    would put the same image on both sides.
    """
    import hashlib

    def bucket(p):
        return int(hashlib.md5(str(p).encode()).hexdigest(), 16) % 1000 / 1000.0

    te = [r for r in rois if bucket(r[0]) < frac]
    tr = [r for r in rois if bucket(r[0]) >= frac]
    return tr, te


def encode(enc, rois, pad, batch_size):
    """-> {'roi_mean': [N, D], 'roi_max': [N, D], 'cls': [N, D]}, tokens per object."""
    import torch
    from PIL import Image

    grid = enc.image_size // enc.patch_size
    by_image: dict[str, list[int]] = {}
    for i, (p, _b, _c) in enumerate(rois):
        by_image.setdefault(str(p), []).append(i)
    paths = sorted(by_image)

    # RADIO's summary vector and its patch tokens have different widths
    # (feature_dim is the CLS dim, patch_dim the tokens'), so neither array is
    # sized from a constant -- both are allocated once the first batch is in.
    out = {}
    cls = None
    ntok = []
    for i in range(0, len(paths), batch_size):
        chunk = paths[i:i + batch_size]
        imgs = [Image.open(p).convert("RGB") for p in chunk]
        sizes = [im.size for im in imgs]
        with torch.no_grad():
            o = enc.encode(imgs)
        if cls is None:
            cls = np.zeros((len(rois), o.cls.shape[-1]), dtype=np.float32)
            out = {k: np.zeros((len(rois), o.patches.shape[-1]), dtype=np.float32)
                   for k in ("roi_mean", "roi_max")}
        for b, p in enumerate(chunk):
            for j in by_image[p]:
                m, mx, n = roi_pool(o.patches[b], grid, rois[j][1], sizes[b], pad)
                out["roi_mean"][j] = m.cpu().numpy()
                out["roi_max"][j] = mx.cpu().numpy()
                cls[j] = o.cls[b].cpu().numpy()
                ntok.append(n)
        if i % (batch_size * 20) == 0:
            print(f"  encoded {i + len(chunk)}/{len(paths)} images", flush=True)
    out["cls"] = cls
    return out, ntok


def fit_multiclass(X_tr, y_tr, X_te, y_te, cats):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize

    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", n_jobs=-1)
    clf.fit(normalize(X_tr), y_tr)
    pred = clf.predict(normalize(X_te))
    acc = float((pred == y_te).mean())
    rec = [float((pred[y_te == c] == c).mean()) for c in range(len(cats))
           if (y_te == c).sum()]
    return acc, float(np.mean(rec)), len(set(pred.tolist()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="aw_m2")
    ap.add_argument("--encoder", default="cradiov4-so")
    ap.add_argument("--image-size", type=int, default=1024,
                    help="native, rounded up to a multiple of the patch size")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="grow the box by this fraction before selecting tokens")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--out-json")
    args = ap.parse_args()

    cats, rois = load_rois(args.dataset, "test")
    if args.dataset == "dronewaste":
        tr_rois, rois = split_by_image(rois)   # ships no split of its own
    else:
        _c2, tr_rois = load_rois(args.dataset, "train")
    print(f"[roi-token] {len(tr_rois)} train / {len(rois)} test objects, {len(cats)} classes")

    enc = VisionEncoder(args.encoder, image_size=args.image_size)
    grid = enc.image_size // enc.patch_size
    print(f"[roi-token] {args.encoder} @{args.image_size}px -> {grid}x{grid} grid, "
          f"{enc.image_size / grid:.1f}px per token")

    idx = {c: i for i, c in enumerate(cats)}
    print("[encode] train"); F_tr, n_tr = encode(enc, tr_rois, args.pad, args.batch_size)
    print("[encode] test");  F_te, n_te = encode(enc, rois, args.pad, args.batch_size)
    y_tr = np.array([idx[c] for _p, _b, c in tr_rois])
    y_te = np.array([idx[c] for _p, _b, c in rois])

    prev = Counter(c for _p, _b, c in rois)
    base = max(prev.values()) / len(rois)
    print(f"\n=== {args.dataset}: {len(rois)} test objects, {len(cats)} classes")
    print(f"  tokens per object: median {int(np.median(n_te))}, "
          f"p10 {int(np.percentile(n_te, 10))}, p90 {int(np.percentile(n_te, 90))}")
    print(f"  majority-class accuracy (the bar): {base:.3f}\n")

    rep = {"dataset": args.dataset, "image_size": args.image_size, "cats": cats,
           "majority": base, "tokens_per_object_median": int(np.median(n_te)),
           "readouts": {}}
    for k in ("roi_mean", "roi_max", "cls"):
        acc, mrec, npred = fit_multiclass(F_tr[k], y_tr, F_te[k], y_te, cats)
        tag = "whole image, no ROI" if k == "cls" else "tokens inside the box"
        print(f"  {k:9s} acc {acc:.3f} ({acc - base:+.3f} vs majority)  "
              f"macro-recall {mrec:.3f} = {mrec * len(cats):.2f}x chance  "
              f"predicts {npred}/{len(cats)}   [{tag}]")
        rep["readouts"][k] = {"acc": acc, "macro_recall": mrec, "n_predicted": npred}

    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(rep, indent=2))
        print(f"\n[write] {args.out_json}")


if __name__ == "__main__":
    main()
