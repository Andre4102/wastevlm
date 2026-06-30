"""Decoding sweep for DinoSeg detection-from-mask (no retraining).

The frozen-backbone seg probe is fixed; this script only varies how the
per-class softmax map is turned into boxes. It caches the model forward
pass ONCE (argmax label map + winning-class prob, both at input res),
then applies several decoding strategies on CPU and scores each with the
exact same COCOeval + paper-10 subset used by `src.seg_eval`.

Decoding strategies
-------------------
  cc        : baseline — scipy.ndimage.label connected components,
              score = mean prob inside the CC (reproduces src.seg_eval).
  cc_max    : CC, but score = max prob inside the CC.
  cc_close  : morphological close (square SE) before CC, score = max.
  watershed : distance-transform watershed to split touching instances
              of the same class, score = max prob inside each instance.

Why maxprob is enough: inside argmax == c pixels, class c IS the argmax,
so softmax[c] there equals the per-pixel max prob. Every score we need
(mean / max of class-c prob over a region) is computable from the single
maxprob map, so we never cache the full [C+1,H,W] tensor.

Usage (SLURM, 1 GPU):
    python -m src.seg_eval_decode \
        --checkpoint /home/ids/diecidue/results/waste_vlm/dinoseg_dw_radio_512/best.pt \
        --split test --out-json .../decode_sweep_radio.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import DRONEWASTE_PAPER_10  # noqa: E402
from src.det_eval import build_split_gt_coco, cocoeval  # noqa: E402
from src.seg_dataset import DroneWasteSegmentation, collate_seg  # noqa: E402
from src.seg_model import DinoSeg, DinoSegConfig  # noqa: E402


@torch.no_grad()
def cache_forward(model: DinoSeg, loader: DataLoader, device: str) -> list[dict]:
    """Run the model once; return per-image {argmax, maxprob, image_id, orig}."""
    model.eval()
    cache: list[dict] = []
    for pix, _masks, meta in loader:
        pix = pix.to(device)
        probs = F.softmax(model(pix), dim=1)  # [B, C+1, H, W]
        maxprob, argmax = probs.max(dim=1)     # [B, H, W], [B, H, W]
        maxprob = maxprob.cpu().numpy().astype(np.float16)
        argmax = argmax.cpu().numpy().astype(np.uint8)
        for b in range(pix.size(0)):
            cache.append({
                "argmax": argmax[b],
                "maxprob": maxprob[b],
                "image_id": int(meta[b]["image_id"].item()),
                "orig_h": int(meta[b]["orig_size"][0]),
                "orig_w": int(meta[b]["orig_size"][1]),
            })
    return cache


def _emit(dets, image_id, fg_idx, ys, xs, score, sx, sy):
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    dets.append({
        "image_id": image_id,
        "category_id": fg_idx,  # 0-based fg idx; remapped to coco id by caller
        "bbox": [float(x0) * sx, float(y0) * sy,
                 float(x1 - x0 + 1) * sx, float(y1 - y0 + 1) * sy],
        "score": score,
    })


def decode_binary(cls_mask, probmap, mode, min_area, close_size, ws_min_distance,
                  image_id, fg_idx, sx, sy, dets):
    """Turn one class's binary mask + prob map into boxes via the chosen mode.

    Shared by the single-label (mask = argmax==c, probmap = maxprob) and
    multi-label (mask = prob[c] > t, probmap = prob[c]) drivers.
    """
    if cls_mask.sum() < min_area:
        return
    if mode == "cc_close" and close_size > 0:
        cls_mask = ndi.binary_closing(cls_mask, structure=np.ones((close_size, close_size)))

    if mode == "watershed":
        dist = ndi.distance_transform_edt(cls_mask)
        coords = peak_local_max(dist, min_distance=ws_min_distance, labels=cls_mask)
        if len(coords) == 0:
            inst, n = ndi.label(cls_mask)
        else:
            markers = np.zeros_like(dist, dtype=np.int32)
            markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
            inst = watershed(-dist, markers, mask=cls_mask)
            n = int(inst.max())
    else:
        inst, n = ndi.label(cls_mask)

    for cid in range(1, n + 1):
        where = inst == cid
        if int(where.sum()) < min_area:
            continue
        ys, xs = np.where(where)
        vals = probmap[where]
        score = float(vals.mean()) if mode == "cc" else float(vals.max())
        _emit(dets, image_id, fg_idx, ys, xs, score, sx, sy)


def decode_image(item: dict, num_logits: int, mode: str, min_area: int,
                 close_size: int, ws_min_distance: int) -> list[dict]:
    """Single-label driver: per class, region = (argmax == c), prob = maxprob."""
    argmax = item["argmax"].astype(np.int32)
    maxprob = item["maxprob"].astype(np.float32)
    H, W = argmax.shape
    sx = item["orig_w"] / W
    sy = item["orig_h"] / H
    dets: list[dict] = []
    for c in range(1, num_logits):
        cls_mask = argmax == c
        decode_binary(cls_mask, maxprob, mode, min_area, close_size,
                      ws_min_distance, item["image_id"], c - 1, sx, sy, dets)
    return dets


def decode_image_multilabel(item: dict, num_classes: int, mode: str, thresh: float,
                            min_area: int, close_size: int,
                            ws_min_distance: int) -> list[dict]:
    """Multi-label driver: per class, region = (prob[c] > thresh), prob = prob[c]."""
    probs = item["probs"].astype(np.float32)  # [C, H, W]
    _C, H, W = probs.shape
    sx = item["orig_w"] / W
    sy = item["orig_h"] / H
    dets: list[dict] = []
    for c in range(num_classes):
        cls_mask = probs[c] > thresh
        decode_binary(cls_mask, probs[c], mode, min_area, close_size,
                      ws_min_distance, item["image_id"], c, sx, sy, dets)
    return dets


def paper10_mAP(per_class: dict) -> float:
    vals = [per_class[c] for c in DRONEWASTE_PAPER_10 if c in per_class
            and not np.isnan(per_class[c])]
    return float(np.mean(vals)) if vals else float("nan")


def score_raw(raw, ds, gt, idx_to_cat_id) -> dict:
    dets = [{**d, "category_id": idx_to_cat_id[d["category_id"]]} for d in raw]
    m = cocoeval(gt, dets) if dets else {
        "overall": {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0},
        "per_class": {n: 0.0 for n in ds.categories},
    }
    return {
        "n_dets": len(dets),
        "mAP": m["overall"]["mAP"],
        "AP50": m["overall"]["AP50"],
        "AP75": m["overall"]["AP75"],
        "paper10_mAP": paper10_mAP(m["per_class"]),
        "per_class": m["per_class"],
    }


def score_config(cache, num_logits, ds, gt, idx_to_cat_id, **kw) -> dict:
    raw = []
    for item in cache:
        raw.extend(decode_image(item, num_logits, **kw))
    return score_raw(raw, ds, gt, idx_to_cat_id)


@torch.no_grad()
def sweep_multilabel(model, loader, device, num_classes, configs) -> dict:
    """One forward pass; decode every config inline (per-class prob is large,
    so we stream rather than cache the full [C,H,W] tensor per image)."""
    model.eval()
    raw = {name: [] for name, _ in configs}
    for pix, _masks, meta in loader:
        pix = pix.to(device)
        probs = torch.sigmoid(model(pix)).cpu().numpy().astype(np.float32)  # [B,C,H,W]
        for b in range(pix.size(0)):
            item = {
                "probs": probs[b],
                "image_id": int(meta[b]["image_id"].item()),
                "orig_h": int(meta[b]["orig_size"][0]),
                "orig_w": int(meta[b]["orig_size"][1]),
            }
            for name, kw in configs:
                raw[name].extend(decode_image_multilabel(item, num_classes, **kw))
    return raw


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--min-area", type=int, default=20)
    p.add_argument("--close-size", type=int, default=3)
    p.add_argument("--ws-min-distance", type=int, nargs="+", default=[7, 15],
                   help="peak_local_max min_distance values to sweep for watershed")
    p.add_argument("--thresh", type=float, nargs="+", default=[0.3, 0.5],
                   help="multi-label only: sigmoid thresholds to sweep")
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = DinoSegConfig(**ckpt["cfg"])
    ml = getattr(cfg, "multilabel", False)
    ds = DroneWasteSegmentation(split=args.split, image_size=cfg.image_size)
    num_classes = len(ds.categories)
    num_logits = num_classes + 1
    print(f"[data] split={args.split} n={len(ds)} classes={num_classes} "
          f"mode={'multilabel' if ml else 'single-label'}")

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_seg, num_workers=args.num_workers,
                        pin_memory=True)
    device = "cuda"
    model = DinoSeg(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[model] {args.checkpoint}  head={cfg.head} backbone={cfg.backbone_type}")

    gt = build_split_gt_coco(ds)
    idx_to_cat_id = {v: k for k, v in ds.cat_id_to_idx.items()}

    if ml:
        configs = []
        for t in args.thresh:
            configs.append((f"cc_max t={t}", dict(mode="cc_max", thresh=t,
                            min_area=args.min_area, close_size=0, ws_min_distance=0)))
        for d in args.ws_min_distance:
            configs.append((f"watershed t=0.5 d={d}", dict(mode="watershed",
                            thresh=0.5, min_area=args.min_area, close_size=0,
                            ws_min_distance=d)))
        raw = sweep_multilabel(model, loader, device, num_classes, configs)
        results = {name: score_raw(raw[name], ds, gt, idx_to_cat_id)
                   for name, _ in configs}
    else:
        cache = cache_forward(model, loader, device)
        print(f"[cache] forward done for {len(cache)} images (CPU sweep follows)")
        configs = [
            ("cc (baseline)", dict(mode="cc", min_area=args.min_area,
                                   close_size=0, ws_min_distance=0)),
            ("cc_max", dict(mode="cc_max", min_area=args.min_area,
                            close_size=0, ws_min_distance=0)),
            ("cc_close", dict(mode="cc_close", min_area=args.min_area,
                              close_size=args.close_size, ws_min_distance=0)),
        ]
        for d in args.ws_min_distance:
            configs.append((f"watershed d={d}",
                            dict(mode="watershed", min_area=args.min_area,
                                 close_size=0, ws_min_distance=d)))
        results = {name: score_config(cache, num_logits, ds, gt, idx_to_cat_id, **kw)
                   for name, kw in configs}

    print()
    print(f"{'config':22s} {'n_dets':>7s} {'mAP':>7s} {'AP50':>7s} "
          f"{'AP75':>7s} {'paper10':>8s}")
    print("-" * 62)
    for name, _ in configs:
        r = results[name]
        print(f"{name:22s} {r['n_dets']:7d} {r['mAP']:7.3f} {r['AP50']:7.3f} "
              f"{r['AP75']:7.3f} {r['paper10_mAP']*100:7.2f}%")

    # Per-class paper-10 mAP for the best config vs the first (baseline) config.
    base_name = configs[0][0]
    base = results[base_name]["per_class"]
    best_name = max(results, key=lambda k: results[k]["paper10_mAP"])
    best = results[best_name]["per_class"]
    print()
    print(f"per-class paper-10 mAP: baseline ({base_name}) vs best ({best_name})")
    for cname in DRONEWASTE_PAPER_10:
        b = base.get(cname, float("nan"))
        w = best.get(cname, float("nan"))
        print(f"  {cname[:40]:40s} {b*100:6.2f}% -> {w*100:6.2f}%")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if kk != "per_class"}
                       for k, v in results.items()}, f, indent=2)
        print(f"[saved] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
