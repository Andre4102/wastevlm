"""Eval frozen DINOv2 + seg head on DroneWaste.

Reports two views on the same predictions:
 - Semantic segmentation: per-class IoU + mIoU (foreground / all).
 - Detection-from-mask: per-class connected components → bbox + score
   (mean foreground softmax prob inside CC), evaluated via pycocotools
   COCOeval against the same GT bboxes used for DinoDETR — i.e. apples-
   to-apples with paper YOLOv8/v12/F-RCNN numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.det_eval import build_split_gt_coco, cocoeval  # noqa: E402
from src.seg_dataset import DroneWasteSegmentation, collate_seg  # noqa: E402
from src.seg_model import DinoSeg, DinoSegConfig  # noqa: E402


@torch.no_grad()
def run_inference(
    model: DinoSeg, loader: DataLoader, device: str, min_area: int = 20
) -> tuple[list[dict], torch.Tensor]:
    model.eval()
    num_logits = model.cfg.num_classes + 1
    cm_total = torch.zeros(num_logits, num_logits, dtype=torch.long)
    results: list[dict] = []
    for pix, masks, meta in loader:
        pix = pix.to(device)
        logits = model(pix)
        probs = F.softmax(logits, dim=1).cpu().numpy()  # [B, C+1, H, W]
        pred = logits.argmax(1).cpu()  # [B, H, W]
        for b in range(pix.size(0)):
            m_gt = masks[b]
            m_pred = pred[b]
            k = (m_gt >= 0) & (m_gt < num_logits)
            idx = (m_gt[k] * num_logits + m_pred[k]).long()
            cm_total += torch.bincount(idx, minlength=num_logits ** 2).reshape(num_logits, num_logits)

            orig_h, orig_w = int(meta[b]["orig_size"][0]), int(meta[b]["orig_size"][1])
            H, W = m_pred.shape
            sx = orig_w / W
            sy = orig_h / H
            p_np = probs[b]
            pred_np = m_pred.numpy()
            for c in range(1, num_logits):
                cls_mask = pred_np == c
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
                    score = float(p_np[c, where].mean())
                    results.append(
                        {
                            "image_id": int(meta[b]["image_id"].item()),
                            "category_id": c - 1,  # 0-based fg idx; remapped later
                            "bbox": [
                                float(x0) * sx,
                                float(y0) * sy,
                                float(x1 - x0 + 1) * sx,
                                float(y1 - y0 + 1) * sy,
                            ],
                            "score": score,
                        }
                    )
    return results, cm_total


def iou_from_cm(cm: torch.Tensor) -> torch.Tensor:
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    return tp / (tp + fp + fn + 1e-6)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--min-area", type=int, default=20)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = DinoSegConfig(**ckpt["cfg"])

    ds = DroneWasteSegmentation(split=args.split, image_size=cfg.image_size)
    print(f"[data] split={args.split}  n={len(ds)}  classes={len(ds.categories)}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_seg, num_workers=args.num_workers, pin_memory=True,
    )

    device = "cuda"
    model = DinoSeg(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"[model] loaded from {args.checkpoint}")

    raw, cm = run_inference(model, loader, device, min_area=args.min_area)
    print(f"[inference] {len(raw)} CC-derived detections")

    iou = iou_from_cm(cm)
    iou_per_class = {ds.categories[i]: float(iou[i + 1]) for i in range(len(ds.categories))}
    miou_fg = float(iou[1:].mean())
    miou_all = float(iou.mean())

    idx_to_cat_id = {v: k for k, v in ds.cat_id_to_idx.items()}
    dets = [{**d, "category_id": idx_to_cat_id[d["category_id"]]} for d in raw]
    gt = build_split_gt_coco(ds)
    det_metrics = cocoeval(gt, dets) if dets else {
        "overall": {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "APs": 0.0, "APm": 0.0, "APl": 0.0},
        "per_class": {n: 0.0 for n in ds.categories},
    }

    print()
    print(f"=== DinoSeg ({cfg.head}) — DroneWaste {args.split} ===")
    print(f"mIoU (foreground)  = {miou_fg:.4f}")
    print(f"mIoU (incl bg)     = {miou_all:.4f}")
    print(f"bg IoU             = {float(iou[0]):.4f}")
    print(f"mAP @ [.5:.95]     = {det_metrics['overall']['mAP']:.3f}")
    print(f"AP@0.5             = {det_metrics['overall']['AP50']:.3f}")
    print(f"AP@0.75            = {det_metrics['overall']['AP75']:.3f}")
    print()
    print("per-class IoU (sorted desc):")
    for n, v in sorted(iou_per_class.items(), key=lambda kv: -kv[1]):
        print(f"  {n[:40]:40s}  IoU={v:.3f}")
    print()
    print("per-class mAP (sorted desc):")
    for n, v in sorted(
        det_metrics["per_class"].items(),
        key=lambda kv: -kv[1] if not np.isnan(kv[1]) else 0,
    ):
        if np.isnan(v):
            print(f"  {n[:40]:40s}  mAP=n/a")
        else:
            print(f"  {n[:40]:40s}  mAP={v:.3f}")

    if args.out_json:
        out = {
            "segmentation": {
                "mIoU_fg": miou_fg, "mIoU_all": miou_all, "bg_IoU": float(iou[0]),
                "iou_per_class": iou_per_class,
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
