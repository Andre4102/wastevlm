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


def split_by_site(rois, held_out=("site5", "site9", "site13", "site17", "site2")):
    """Hold out whole SITES, not images.

    DroneWaste ships no split, and its 4993 images are crops cut from 17
    hand-annotated sites, so crops from one site are near duplicates of each
    other. An image-level split leaves the same pile on both sides of the
    evaluation and reports a number that is mostly memorisation. The default
    held-out set spans large and small sites and 5 of the 17.
    """
    def site(p):
        return pathlib.Path(p).name.split("_")[0]

    te = [r for r in rois if site(r[0]) in held_out]
    tr = [r for r in rois if site(r[0]) not in held_out]
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


def project_siglip2(out: dict, encoder_id: str, device="cuda") -> dict:
    """Add readouts living in SigLIP2's text-aligned summary space.

    The probe above reads RADIO's own features. That leaves the interesting
    question open: is the +0.266 over the bar a property of the FEATURES, or does
    it survive the projection that makes them text-comparable? Same vectors, same
    probe, one extra matmul -- so the only thing that varies is the space.

    `_heads.siglip2-g` is the summary head, which is what `roi-head` naming uses;
    the CLS slice is taken at index 0 because the model concatenates the summary
    tokens of the teachers that set use_summary (siglip2-g, then dino_v3_7b).
    """
    import torch

    from src.radio_adaptors import load_projection

    head = load_projection("siglip2-g", "summary", encoder_id, device=device)
    hin = head.fc1.weight.shape[1] if hasattr(head, "fc1") else None
    dt = next(head.parameters()).dtype
    add = {}
    for src in ("roi_mean", "roi_max", "cls"):
        if src not in out:
            continue
        X = out[src]
        if hin is not None and X.shape[1] != hin:
            if X.shape[1] % hin:
                print(f"  [siglip2] skip {src}: {X.shape[1]}d does not fit the "
                      f"head's {hin}d input")
                continue
            X = X[:, :hin]        # CLS is the teachers' summaries concatenated
            print(f"  [siglip2] {src}: taking slice 0 of {out[src].shape[1]}d "
                  f"-> {hin}d (siglip2-g's own summary token)")
        Z = np.zeros((len(X), head.final[2].out_features), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                b = torch.from_numpy(X[i:i + 2048]).to(device=device, dtype=dt)
                Z[i:i + 2048] = head(b).float().cpu().numpy()
        add[f"{src}@siglip2"] = Z
    out.update(add)
    return out


def fit_centroid(X_tr, y_tr, X_te, y_te, cats):
    """Nearest class-centroid, cosine -- the same functional form as text matching.

    The logistic probe has 5 x D free parameters; a text comparison has 5 fixed
    points and picks the nearest. Those are not comparable readouts, so a probe
    beating text does not by itself show that the TEXT is the weak part. This
    control keeps the form (5 prototypes, nearest wins) and only swaps where the
    prototypes come from: class means of the training images, instead of the text
    encoder. If it lands near the probe, text embeddings are bad prototypes; if it
    lands near the text arm, the prototype form is the limit.
    """
    from sklearn.preprocessing import normalize

    Xtr, Xte = normalize(X_tr), normalize(X_te)
    # A class with no training exemplar has no centroid. Averaging an empty slice
    # gives NaN, which propagates through the whole similarity matrix and makes
    # argmax return 0 for every object -- a silent collapse that reads as
    # "accuracy 0.000, predicts 1/20" rather than as an error. Such classes are
    # excluded from the prototype set instead, so they are simply never predicted.
    have = np.array([c for c in range(len(cats)) if (y_tr == c).sum() > 0])
    C = np.stack([Xtr[y_tr == c].mean(0) for c in have])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    pred = have[(Xte @ C.T).argmax(1)]
    acc = float((pred == y_te).mean())
    rec = [float((pred[y_te == c] == c).mean()) for c in range(len(cats))
           if (y_te == c).sum()]
    return acc, float(np.mean(rec)), len(set(pred.tolist()))


def fit_multiclass(X_tr, y_tr, X_te, y_te, cats, keep=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize

    clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", n_jobs=-1)
    clf.fit(normalize(X_tr), y_tr)
    pred = clf.predict(normalize(X_te))
    acc = float((pred == y_te).mean())
    rec = [float((pred[y_te == c] == c).mean()) for c in range(len(cats))
           if (y_te == c).sum()]
    if keep is not None:
        keep.append(pred.tolist())
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
    ap.add_argument("--siglip2", action="store_true",
                    help="also probe the same vectors after the SigLIP2 summary "
                         "head, to separate the features from their space")
    ap.add_argument("--dump-emb", help="path stem for saving the feature arrays")
    ap.add_argument("--out-json")
    args = ap.parse_args()

    cats, rois = load_rois(args.dataset, "test")
    if args.dataset == "dronewaste":
        tr_rois, rois = split_by_site(rois)   # ships no split of its own
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
    if args.siglip2:
        print("[project] train"); F_tr = project_siglip2(F_tr, args.encoder)
        print("[project] test");  F_te = project_siglip2(F_te, args.encoder)
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
           "y_true": y_te.tolist(), "pred": {}, "readouts": {}}
    keys = [k for k in ("roi_mean", "roi_max", "cls",
                        "roi_mean@siglip2", "roi_max@siglip2", "cls@siglip2")
            if k in F_tr and k in F_te]
    for k in keys:
        keep = []
        acc, mrec, npred = fit_multiclass(F_tr[k], y_tr, F_te[k], y_te, cats, keep)
        rep["pred"][k] = keep[0]
        tag = "whole image, no ROI" if k.startswith("cls") else "tokens inside the box"
        if k.endswith("@siglip2"):
            tag += ", SigLIP2 summary space"
        print(f"  {k:17s} acc {acc:.3f} ({acc - base:+.3f} vs majority)  "
              f"macro-recall {mrec:.3f} = {mrec * len(cats):.2f}x chance  "
              f"predicts {npred}/{len(cats)}   [{tag}]")
        rep["readouts"][k] = {"acc": acc, "macro_recall": mrec, "n_predicted": npred}
        cacc, cmrec, cnp = fit_centroid(F_tr[k], y_tr, F_te[k], y_te, cats)
        print(f"  {'':17s} centroid {cacc:.3f} ({cacc - base:+.3f})  "
              f"macro-recall {cmrec:.3f}  predicts {cnp}/{len(cats)}")
        rep["readouts"][k + "|centroid"] = {"acc": cacc, "macro_recall": cmrec,
                                           "n_predicted": cnp}

    if args.dump_emb:
        # The probe reports accuracy; it cannot say WHY a text query misses. That
        # needs the vectors themselves -- class centroids on one side, text
        # embeddings on the other -- so dump the projected features and do the
        # geometry offline on CPU.
        import numpy as _np
        stem = pathlib.Path(args.dump_emb)
        stem.parent.mkdir(parents=True, exist_ok=True)
        for k in keys:
            _np.save(f"{stem}_{k.replace('@', '_at_')}_test.npy", F_te[k])
            _np.save(f"{stem}_{k.replace('@', '_at_')}_train.npy", F_tr[k])
        _np.save(f"{stem}_y_test.npy", y_te)
        _np.save(f"{stem}_y_train.npy", y_tr)
        (stem.parent / (stem.name + "_cats.json")).write_text(json.dumps(cats))
        print(f"[dump] {stem}_*.npy  ({len(keys)} readouts)")

    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(rep, indent=2))
        print(f"\n[write] {args.out_json}")


if __name__ == "__main__":
    main()
