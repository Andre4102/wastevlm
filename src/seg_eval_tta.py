"""Tile-then-aggregate (TTA) eval for DinoSeg checkpoints — tests the
patch-size / effective-ground-resolution hypothesis without retraining.

Standard DinoSeg eval (`src.seg_eval`):
    640²-native DroneWaste tile  →  resize to cfg.image_size (e.g. 512²)
                                 →  encoder + head
                                 →  CC → bbox

TTA eval here:
    640²-native tile             →  split into N × N non-overlapping crops
                                 →  each crop resized to cfg.image_size
                                 →  encoder + head (run N² times)
                                 →  outputs stitched back to 640²
                                 →  CC → bbox

Same checkpoint, same head, no retraining. With N=2 each ViT patch covers
~½ the ground area it saw at training time — that's the real "smaller
effective patch" test (HR 1024² didn't move the needle because the source
data was already at the native resolution).

Usage:
    python -m src.seg_eval_tta \\
        --checkpoint /home/ids/diecidue/results/waste_vlm/dinoseg_dw_dinov3_512/best.pt \\
        --tta-grid 2 --split test \\
        --out-json /home/ids/diecidue/results/waste_vlm/dinoseg_dw_dinov3_512/test_eval_tta2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.det_eval import build_split_gt_coco, cocoeval  # noqa: E402
from src.seg_dataset import DroneWasteSegmentation  # noqa: E402
from src.seg_model import DinoSeg, DinoSegConfig  # noqa: E402

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _crop_to_model_input(crop: Image.Image, model_size: int) -> torch.Tensor:
    tx = transforms.Compose([
        transforms.Resize((model_size, model_size),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tx(crop).unsqueeze(0)  # [1, 3, S, S]


@torch.no_grad()
def run_tta_inference(
    model: DinoSeg, ds: DroneWasteSegmentation, device: str,
    tta_grid: int, min_area: int,
) -> tuple[list[dict], torch.Tensor]:
    model.eval()
    num_logits = model.cfg.num_classes + 1
    cfg_size = model.cfg.image_size
    cm_total = torch.zeros(num_logits, num_logits, dtype=torch.long)
    results: list[dict] = []

    for idx, s in enumerate(ds.samples):
        img_full = Image.open(ds.images_dir / s["file_name"]).convert("RGB")
        full_w, full_h = img_full.size

        # Non-overlapping N × N tiling. Use integer division; trailing pixels
        # (640 % 2 = 0 so harmless for DroneWaste; for non-divisible sizes
        # the last crop is shifted inward to keep equal-sized crops).
        crop_w = full_w // tta_grid
        crop_h = full_h // tta_grid

        # Accumulator at full original resolution.
        probs_full = np.zeros((num_logits, full_h, full_w), dtype=np.float32)
        count_full = np.zeros((full_h, full_w), dtype=np.float32)

        for gi in range(tta_grid):
            for gj in range(tta_grid):
                y0 = gi * crop_h if gi < tta_grid - 1 else full_h - crop_h
                x0 = gj * crop_w if gj < tta_grid - 1 else full_w - crop_w
                y1, x1 = y0 + crop_h, x0 + crop_w
                crop = img_full.crop((x0, y0, x1, y1))
                pix = _crop_to_model_input(crop, cfg_size).to(device)
                logits = model(pix)  # [1, C+1, S, S]
                # Interpolate to original crop size in one shot.
                logits_hw = F.interpolate(
                    logits, size=(crop_h, crop_w),
                    mode="bilinear", align_corners=False,
                )
                probs_hw = F.softmax(logits_hw, dim=1).cpu().numpy()[0]
                probs_full[:, y0:y1, x0:x1] += probs_hw
                count_full[y0:y1, x0:x1] += 1.0

        probs_full /= count_full[None, :, :]
        pred_full = probs_full.argmax(0)  # [H, W] at original resolution

        # Polygon-rasterised GT at the *full* original resolution.
        # (seg_dataset._rasterise produces a model_size-sized mask; for TTA
        # we keep everything at native resolution for the confusion matrix.)
        gt_full = _rasterise_full(
            s["annotations"], full_w, full_h, ds.cat_id_to_idx,
        )
        k = (gt_full >= 0) & (gt_full < num_logits)
        idx_flat = (gt_full[k].astype(np.int64) * num_logits
                    + pred_full[k].astype(np.int64))
        cm_total += torch.from_numpy(
            np.bincount(idx_flat, minlength=num_logits ** 2)
        ).reshape(num_logits, num_logits)

        # --- detections via connected-components at full resolution
        for c in range(1, num_logits):
            cls_mask = pred_full == c
            if cls_mask.sum() < min_area:
                continue
            lab, n = ndimage.label(cls_mask)
            for cid in range(1, n + 1):
                where = lab == cid
                if where.sum() < min_area:
                    continue
                ys, xs = np.where(where)
                y0, y1 = int(ys.min()), int(ys.max())
                x0, x1 = int(xs.min()), int(xs.max())
                score = float(probs_full[c, where].mean())
                results.append({
                    "image_id": int(s["id"]),
                    "category_id": c - 1,
                    "bbox": [float(x0), float(y0),
                             float(x1 - x0 + 1), float(y1 - y0 + 1)],
                    "score": score,
                })

        if (idx + 1) % 50 == 0 or (idx + 1) == len(ds.samples):
            print(f"  [{idx+1}/{len(ds.samples)}] dets={len(results)}", flush=True)

    return results, cm_total


def _rasterise_full(anns, w, h, cat_id_to_idx) -> np.ndarray:
    """Polygon raster at the *original* (w, h) resolution — no resize.

    Mirrors DroneWasteSegmentation._rasterise but without the scaling.
    """
    from PIL import ImageDraw
    mask = Image.new("I", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    anns_sorted = sorted(anns, key=lambda a: -float(a.get("area", 0.0)))
    for a in anns_sorted:
        cls = cat_id_to_idx[a["category_id"]] + 1
        seg = a.get("segmentation")
        if not isinstance(seg, list):
            continue
        for poly in seg:
            if not isinstance(poly, list) or len(poly) < 6:
                continue
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
            if len(pts) >= 3:
                draw.polygon(pts, fill=cls)
    return np.array(mask, dtype=np.int64)


def iou_from_cm(cm: torch.Tensor) -> torch.Tensor:
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    return tp / (tp + fp + fn + 1e-6)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--tta-grid", type=int, default=2,
                   help="N×N non-overlapping crops per tile (1 = no TTA)")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--min-area", type=int, default=20)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = DinoSegConfig(**ckpt["cfg"])
    ds = DroneWasteSegmentation(split=args.split, image_size=cfg.image_size)
    print(f"[data] split={args.split} n={len(ds)} classes={len(ds.categories)} "
          f"tta_grid={args.tta_grid} (effective crop={ds.samples[0]['width']//args.tta_grid}px)")

    device = "cuda"
    model = DinoSeg(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[model] loaded from {args.checkpoint}  cfg.image_size={cfg.image_size}  head={cfg.head}")

    raw, cm = run_tta_inference(model, ds, device, args.tta_grid, args.min_area)
    iou = iou_from_cm(cm)
    iou_per_class = {ds.categories[i]: float(iou[i + 1]) for i in range(len(ds.categories))}
    miou_fg = float(iou[1:].mean())
    miou_all = float(iou.mean())

    idx_to_cat_id = {v: k for k, v in ds.cat_id_to_idx.items()}
    dets = [{**d, "category_id": idx_to_cat_id[d["category_id"]]} for d in raw]
    gt = build_split_gt_coco(ds)
    det_metrics = cocoeval(gt, dets) if dets else {
        "overall": {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0,
                    "APs": 0.0, "APm": 0.0, "APl": 0.0},
        "per_class": {n: 0.0 for n in ds.categories},
    }

    print()
    print(f"=== DinoSeg TTA (grid={args.tta_grid}) — DroneWaste {args.split} ===")
    print(f"mIoU (foreground)  = {miou_fg:.4f}")
    print(f"mIoU (incl bg)     = {miou_all:.4f}")
    print(f"mAP @ [.5:.95]     = {det_metrics['overall']['mAP']:.3f}")
    print(f"AP@0.5             = {det_metrics['overall']['AP50']:.3f}")
    print()
    print("per-class mAP (sorted desc):")
    for name, ap in sorted(det_metrics["per_class"].items(),
                           key=lambda kv: -kv[1] if not np.isnan(kv[1]) else 0):
        tag = "n/a" if np.isnan(ap) else f"{ap:.3f}"
        print(f"  {name[:40]:40s}  mAP={tag}")

    if args.out_json:
        out = {
            "tta_grid": args.tta_grid,
            "segmentation": {
                "mIoU_fg": miou_fg, "mIoU_all": miou_all,
                "bg_IoU": float(iou[0]), "iou_per_class": iou_per_class,
            },
            "detection_from_mask": det_metrics,
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
