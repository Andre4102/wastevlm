"""COCO-mAP evaluation for DinoDETR on DroneWaste.

Loads a checkpoint, runs inference on the requested split, converts
predictions to COCO format, and runs pycocotools' COCOeval. Reports
overall mAP @ IoU [0.5:0.95] + AP@0.5 + AP@0.75 + per-class AP.

Usage:
    python -m src.det_eval --checkpoint results/dinodetr_dw/best.pt --split test
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.det_dataset import DroneWasteDetection, collate_detection  # noqa: E402
from src.det_model import DinoDETR, DinoDETRConfig  # noqa: E402


def _cxcywh_to_xywh_pixels(boxes_norm: torch.Tensor, h: int, w: int) -> torch.Tensor:
    cx, cy, bw, bh = boxes_norm.unbind(-1)
    x = (cx - bw / 2) * w
    y = (cy - bh / 2) * h
    return torch.stack([x, y, bw * w, bh * h], dim=-1)


@torch.no_grad()
def run_inference(
    model: DinoDETR, loader: DataLoader, device: str, conf_thresh: float = 0.05
) -> list[dict]:
    """Returns COCO-format `results` list (one dict per kept detection)."""
    model.eval()
    results: list[dict] = []
    for pix, tgts in loader:
        pix = pix.to(device)
        out = model(pix)
        logits = out["logits"]  # [B, Q, C+1]
        boxes = out["pred_boxes"]  # [B, Q, 4]
        probs = logits.softmax(-1)
        # Drop the "no object" class (last index)
        scores, labels = probs[..., :-1].max(-1)
        for i, t in enumerate(tgts):
            h, w = int(t["orig_size"][0]), int(t["orig_size"][1])
            keep = scores[i] >= conf_thresh
            if keep.sum() == 0:
                continue
            xywh = _cxcywh_to_xywh_pixels(boxes[i][keep], h, w)
            sc = scores[i][keep].cpu().numpy()
            lb = labels[i][keep].cpu().numpy()
            xywh = xywh.cpu().numpy()
            for box, s_, l_ in zip(xywh, sc, lb):
                results.append(
                    {
                        "image_id": int(t["image_id"].item()),
                        "category_id": int(l_),  # placeholder; remapped below
                        "bbox": [float(x) for x in box],
                        "score": float(s_),
                    }
                )
    return results


def build_split_gt_coco(ds: DroneWasteDetection) -> dict:
    """Build a COCO-format GT dict restricted to the split's images."""
    images = []
    annotations = []
    ann_id = 1
    for s in ds.samples:
        images.append(
            {
                "id": s["id"],
                "file_name": s["file_name"],
                "width": s["width"],
                "height": s["height"],
            }
        )
        for a in s["annotations"]:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": s["id"],
                    "category_id": a["category_id"],
                    "bbox": [float(x), float(y), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    # Reuse the original category list (with original IDs)
    with (Path("/home/ids/diecidue/data/dronewaste") / "dronewaste_v1.0.json").open() as f:
        full = json.load(f)
    return {
        "images": images,
        "annotations": annotations,
        "categories": full["categories"],
    }


def remap_pred_category_ids(results: list[dict], idx_to_cat_id: dict[int, int]) -> list[dict]:
    return [{**r, "category_id": idx_to_cat_id[r["category_id"]]} for r in results]


def cocoeval(gt: dict, dt: list[dict]) -> dict:
    """Run pycocotools COCOeval and return overall + per-class metrics."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(gt, f)
        gt_path = f.name
    coco_gt = COCO(gt_path)
    if not dt:
        return {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "per_class": {}}
    coco_dt = coco_gt.loadRes(dt)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    buf = io.StringIO()
    with redirect_stdout(buf):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    stats = coco_eval.stats  # [mAP, AP50, AP75, APs, APm, APl, AR1, AR10, AR100, ARs, ARm, ARl]
    overall = {
        "mAP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APs": float(stats[3]),
        "APm": float(stats[4]),
        "APl": float(stats[5]),
    }

    # Per-class AP @ IoU [0.5:0.95] averaged across IoU + area + max-dets axes
    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]
    per_class: dict[str, float] = {}
    cat_ids = coco_gt.getCatIds()
    for k, cid in enumerate(cat_ids):
        ap = precision[:, :, k, 0, -1]
        ap = ap[ap > -1]
        per_class[coco_gt.loadCats([cid])[0]["name"]] = float(ap.mean()) if ap.size else float("nan")

    return {"overall": overall, "per_class": per_class}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--conf-thresh", type=float, default=0.05)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = DinoDETRConfig(**ckpt["cfg"])

    ds = DroneWasteDetection(split=args.split, image_size=cfg.image_size)
    print(f"[data] split={args.split} n={len(ds)} classes={len(ds.categories)}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_detection, num_workers=args.num_workers, pin_memory=True,
    )

    device = "cuda"
    model = DinoDETR(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[model] loaded from {args.checkpoint}")

    raw = run_inference(model, loader, device, conf_thresh=args.conf_thresh)
    # Remap 0-based class index -> original COCO category_id
    idx_to_cat_id = {v: k for k, v in ds.cat_id_to_idx.items()}
    dets = remap_pred_category_ids(raw, idx_to_cat_id)
    print(f"[inference] {len(dets)} detections @ conf>={args.conf_thresh}")

    gt = build_split_gt_coco(ds)
    metrics = cocoeval(gt, dets)

    print()
    print(f"=== DinoDETR — DroneWaste {args.split} split ===")
    print(f"mAP @ [.5:.95] = {metrics['overall']['mAP']:.3f}")
    print(f"AP@0.5         = {metrics['overall']['AP50']:.3f}")
    print(f"AP@0.75        = {metrics['overall']['AP75']:.3f}")
    print(f"AP small       = {metrics['overall']['APs']:.3f}")
    print(f"AP medium      = {metrics['overall']['APm']:.3f}")
    print(f"AP large       = {metrics['overall']['APl']:.3f}")
    print()
    print("per-class AP (sorted desc):")
    for name, ap in sorted(metrics["per_class"].items(), key=lambda kv: -kv[1] if not np.isnan(kv[1]) else 0):
        print(f"  {name[:40]:40s}  {ap:.3f}" if not np.isnan(ap) else f"  {name[:40]:40s}  n/a")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[saved] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
