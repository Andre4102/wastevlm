"""Can the encoder we already run produce the proposals, so the detector can go?

The training-free pipeline currently spends four models: Grounding DINO for
boxes, SAM 2 for masks, GeoRSCLIP for names, our VLM for the gate. Three of those
four jobs may be doable from one frozen C-RADIOv4 pass, which is already computed
for the ROI naming head:

  naming  -- measured. ROI-pooled tokens + a linear head reach 3.52x chance on
             AerialWaste and beat GeoRSCLIP-on-crops by a wide margin.
  gating  -- measured. The same features separate waste from no-waste at AUC 0.970.
  boxes   -- NOT measured, and the only reason Grounding DINO is still in the
             pipeline. This script measures it.

Two objectness estimators, both training-free, both from the same patch grid:

  border    background prototype = the mean border token, objectness = distance
            from it. Aerial crops are mostly homogeneous ground, so "unlike the
            edge of the frame" is a decent first guess at "object".
  spectral  TokenCut: Fiedler vector of the normalised Laplacian of the patch
            affinity graph, i.e. the cut that best separates the grid in two.
            Makes no assumption that the background touches the border.

Scored the way `gdino_baseline.py` scores, against the same ground truth, with
the same random-placement floor, so the numbers are directly comparable to
Grounding DINO's 0.819 recall on DroneWaste and 0.150 on AerialWaste.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.gdino_baseline import iou  # noqa: E402
from scripts.roi_material import load_rois  # noqa: E402
from src.vision_encoder import VisionEncoder  # noqa: E402


def border_objectness(F: np.ndarray, g: int) -> np.ndarray:
    """1 - cosine similarity to the mean border token."""
    X = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    m = np.zeros((g, g), bool)
    m[0], m[-1], m[:, 0], m[:, -1] = True, True, True, True
    proto = X[m.reshape(-1)].mean(0)
    proto /= np.linalg.norm(proto) + 1e-8
    return (1.0 - X @ proto).reshape(g, g)


def spectral_objectness(F: np.ndarray, g: int, tau: float = 0.2) -> np.ndarray:
    """TokenCut: the Fiedler vector of the thresholded patch affinity graph."""
    X = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    A = X @ X.T
    W = np.where(A > tau, 1.0, 1e-5)
    d = W.sum(1)
    D_is = np.diag(1.0 / np.sqrt(d + 1e-8))
    L = D_is @ (np.diag(d) - W) @ D_is
    vals, vecs = np.linalg.eigh(L)
    f = vecs[:, 1]                       # second smallest = Fiedler
    # orient so the smaller partition is the foreground: objects are the minority
    if (f > f.mean()).sum() > f.size / 2:
        f = -f
    return f.reshape(g, g)


def boxes_from_map(a: np.ndarray, size, pct: float = 90.0, min_cells: int = 1):
    """Threshold, take connected components, return each component's box in pixels."""
    from scipy import ndimage

    g = a.shape[0]
    W, H = size
    m = a >= np.percentile(a, pct)
    lab, n = ndimage.label(m)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if ys.size < min_cells:
            continue
        out.append([xs.min() / g * W, ys.min() / g * H,
                    (xs.max() + 1) / g * W, (ys.max() + 1) / g * H])
    return out


def random_floor(boxes, size, gt, rng, thr):
    """The same boxes placed elsewhere -- what the recall would be by luck."""
    W, H = size
    hit = 0
    for b in gt:
        gx, gy, gw, gh = b
        best = 0.0
        for p in boxes:
            w, h = p[2] - p[0], p[3] - p[1]
            x = rng.uniform(0, max(1, W - w)); y = rng.uniform(0, max(1, H - h))
            best = max(best, iou([x, y, x + w, y + h], [gx, gy, gx + gw, gy + gh]))
        hit += best >= thr
    return hit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dronewaste")
    ap.add_argument("--encoder", default="cradiov4-so")
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--pct", type=float, nargs="*", default=[85.0, 90.0, 95.0])
    ap.add_argument("--project", default="none", choices=["none", "sam3", "siglip2-g"],
                    help="run the teacher projection before scoring objectness; "
                         "raw features are heuristics, sam3 is the head trained for this")
    ap.add_argument("--out-json")
    args = ap.parse_args()

    import torch
    from PIL import Image

    _cats, rois = load_rois(args.dataset, "test")
    by_image = defaultdict(list)
    for p, b, _c in rois:
        by_image[str(p)].append(list(b))
    paths = sorted(by_image)[: args.limit]
    print(f"[obj] {len(paths)} annotated images, "
          f"{sum(len(by_image[p]) for p in paths)} objects")

    enc = VisionEncoder(args.encoder, image_size=args.image_size)
    g = enc.image_size // enc.patch_size
    proj = None
    if args.project != "none":
        from src.radio_adaptors import load_projection
        proj = load_projection(args.project, "features", args.encoder, device=enc.device)
        print(f"[obj] projecting patches into {args.project} space")
    print(f"[obj] {args.encoder} @{args.image_size} -> {g}x{g} grid, "
          f"{enc.image_size / g:.1f}px per token")

    rng = np.random.default_rng(0)
    stats = {f"{m}@{p}": {"hit25": 0, "hit50": 0, "n": 0, "floor50": 0, "nbox": 0}
             for m in ("border", "spectral") for p in args.pct}
    for n, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        with torch.no_grad():
            P = enc.encode([img]).patches
            if proj is not None:
                P = proj(P.to(next(proj.parameters()).dtype))
            F = P[0].float().cpu().numpy()
        gt = by_image[path]
        maps = {"border": border_objectness(F, g), "spectral": spectral_objectness(F, g)}
        for mname, a in maps.items():
            for pct in args.pct:
                k = f"{mname}@{pct}"
                pred = boxes_from_map(a, img.size, pct)
                s = stats[k]
                s["nbox"] += len(pred)
                for gx, gy, gw, gh in gt:
                    box = [gx, gy, gx + gw, gy + gh]
                    best = max((iou(p, box) for p in pred), default=0.0)
                    s["hit25"] += best >= 0.25
                    s["hit50"] += best >= 0.50
                    s["n"] += 1
                s["floor50"] += random_floor(pred, img.size, gt, rng, 0.50)
        if n % 25 == 0:
            print(f"  {n}/{len(paths)}", flush=True)

    print(f"\n=== {args.dataset}, training-free objectness from {args.encoder} @{args.image_size}")
    print("  reference: Grounding DINO box recall @IoU 0.5 -- DroneWaste 0.819, AerialWaste 0.150\n")
    rep = {}
    for k, s in stats.items():
        r25, r50 = s["hit25"] / s["n"], s["hit50"] / s["n"]
        fl = s["floor50"] / s["n"]
        print(f"  {k:16s} recall@0.25 {r25:.3f}   recall@0.50 {r50:.3f}"
              f"   (random floor {fl:.3f})   {s['nbox'] / len(paths):.1f} boxes/img")
        rep[k] = {"recall25": r25, "recall50": r50, "floor50": fl,
                  "boxes_per_image": s["nbox"] / len(paths)}
    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(rep, indent=2))
        print(f"\n[write] {args.out_json}")


if __name__ == "__main__":
    main()
