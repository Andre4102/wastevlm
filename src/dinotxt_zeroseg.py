"""dino.txt zero-shot open-vocabulary semantic segmentation on DroneWaste.

Recipe (from the official dinotxt.ipynb):
 1. Get patch features [B, 1024, h, w] at the model's native grid (h=w=37 for
    518² input); bicubic-upsample to image_size for pixel-level prediction.
 2. L2-normalise patches over the channel dim.
 3. Encode each class name's templates -> mean -> use the patch-aligned slice
    text[:, 1024:] -> L2-norm. [C, 1024].
 4. einsum("bdhw,cd->bchw") -> per-pixel cosine logits.
 5. (Background handling) Either argmax with no bg class, or include a
    "background" prompt and argmax over (C+1).

Reports:
 - Per-class IoU and mIoU on the rasterised GT masks (same as supervised DinoSeg).
 - COCO box-reconstitution mAP (same `cocoeval` path).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.det_eval import build_split_gt_coco, cocoeval  # noqa: E402
from src.dinotxt_runner import AERIAL_TEMPLATES, DinoTxtRunner  # noqa: E402
from src.dinotxt_zeroshot import prompt_term  # noqa: E402
from src.seg_dataset import DroneWasteSegmentation, collate_seg  # noqa: E402


BACKGROUND_PROMPTS = [
    "an aerial photograph of empty ground.",
    "an aerial photograph of grass and trees.",
    "an aerial photograph of farmland.",
    "an aerial photograph of a road or parking lot.",
    "an aerial photograph of a building rooftop.",
    "an aerial photograph of bare soil.",
]


@torch.no_grad()
def background_embedding(runner: DinoTxtRunner) -> torch.Tensor:
    """Patch-aligned background prompt embedding, [1024]."""
    toks = runner.tokenizer.tokenize(BACKGROUND_PROMPTS).to(runner.device)
    with torch.autocast(runner.device, dtype=runner.dtype):
        emb = runner.model.encode_text(toks)
    emb = emb[:, 1024:]  # patch-aligned slice
    emb = F.normalize(emb.float(), p=2, dim=-1)
    emb = emb.mean(dim=0)
    emb = F.normalize(emb, p=2, dim=-1)
    return emb


@torch.no_grad()
def run_inference(
    runner: DinoTxtRunner,
    loader: DataLoader,
    class_text_feats: torch.Tensor,
    bg_text_feat: torch.Tensor | None,
    out_size: int,
    min_area: int = 20,
    bg_bonus: float = 0.0,
) -> tuple[list[dict], torch.Tensor]:
    """Returns (COCO detections, confusion matrix).

    Class indices in pred mask:
        0          -> background
        1..C       -> 0-based foreground class index + 1
    """
    num_logits = class_text_feats.size(0) + 1
    cm_total = torch.zeros(num_logits, num_logits, dtype=torch.long)
    text_feats = class_text_feats.to(runner.device)
    if bg_text_feat is not None:
        bg_text_feat = bg_text_feat.to(runner.device)
    results: list[dict] = []

    for pix, masks, meta in loader:
        patch = runner.get_patch_features(pix, out_size=(out_size, out_size))
        # patch: [B, 1024, H, W] L2-normed over D
        # text_feats: [C, 1024] L2-normed over D
        logits_fg = torch.einsum("bdhw,cd->bchw", patch, text_feats)  # [B, C, H, W]
        if bg_text_feat is not None:
            logits_bg = torch.einsum("bdhw,d->bhw", patch, bg_text_feat).unsqueeze(1)
            logits_bg = logits_bg + bg_bonus
            logits = torch.cat([logits_bg, logits_fg], dim=1)  # bg at channel 0
        else:
            # Inject a constant background channel — pred is bg if no fg score > bg_bonus
            logits_bg = torch.full_like(logits_fg[:, :1], fill_value=bg_bonus)
            logits = torch.cat([logits_bg, logits_fg], dim=1)
        probs = F.softmax(logits * 100.0, dim=1)  # sharpen for nicer reconstitution
        pred = logits.argmax(1).cpu()

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
            p_np = probs[b].cpu().numpy()
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
                            "category_id": c - 1,
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
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--image-size", type=int, default=518)
    p.add_argument("--out-size", type=int, default=518)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--min-area", type=int, default=20)
    p.add_argument("--bg-mode", choices=["learned", "constant"], default="learned",
                   help="learned: use aerial-bg text prompts; constant: just a bias")
    p.add_argument("--bg-bonus", type=float, default=0.0,
                   help="additive bonus on the background logit (positive=more bg)")
    p.add_argument("--out-json", type=Path, default=Path("/home/ids/diecidue/results/waste_vlm/dinotxt_zeroseg_dw_test.json"))
    args = p.parse_args()

    runner = DinoTxtRunner(image_size=args.image_size)
    print(f"[model] dino.txt loaded; image_size={args.image_size}, out_size={args.out_size}")

    ds = DroneWasteSegmentation(split=args.split, image_size=args.image_size)
    print(f"[data] split={args.split}  n={len(ds)}  classes={len(ds.categories)}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_seg, num_workers=args.num_workers, pin_memory=True,
    )

    class_text = runner.encode_text_classes_patch(
        [prompt_term(c) for c in ds.categories], templates=AERIAL_TEMPLATES,
    )  # [C, 1024]
    bg_text = background_embedding(runner) if args.bg_mode == "learned" else None
    if bg_text is not None:
        print("[bg] using learned background embedding from aerial prompts")
    else:
        print(f"[bg] using constant background with bonus={args.bg_bonus}")

    dets, cm = run_inference(
        runner, loader, class_text, bg_text,
        out_size=args.out_size, min_area=args.min_area, bg_bonus=args.bg_bonus,
    )
    print(f"[inference] {len(dets)} CC-derived detections")

    iou = iou_from_cm(cm)
    iou_per_class = {ds.categories[i]: float(iou[i + 1]) for i in range(len(ds.categories))}
    miou_fg = float(iou[1:].mean())
    miou_all = float(iou.mean())
    bg_iou = float(iou[0])

    idx_to_cat_id = {v: k for k, v in ds.cat_id_to_idx.items()}
    dets_remap = [{**d, "category_id": idx_to_cat_id[d["category_id"]]} for d in dets]
    gt = build_split_gt_coco(ds)
    det_metrics = (
        cocoeval(gt, dets_remap)
        if dets_remap
        else {
            "overall": {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0, "APs": 0.0, "APm": 0.0, "APl": 0.0},
            "per_class": {n: 0.0 for n in ds.categories},
        }
    )

    print()
    print(f"=== dino.txt zero-shot open-vocab seg — DroneWaste {args.split} ===")
    print(f"mIoU (foreground)  = {miou_fg:.4f}")
    print(f"mIoU (incl bg)     = {miou_all:.4f}")
    print(f"bg IoU             = {bg_iou:.4f}")
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
                "mIoU_fg": miou_fg, "mIoU_all": miou_all, "bg_IoU": bg_iou,
                "iou_per_class": iou_per_class,
            },
            "detection_from_mask": det_metrics,
            "bg_mode": args.bg_mode,
            "bg_bonus": args.bg_bonus,
            "image_size": args.image_size,
            "out_size": args.out_size,
            "prompt_terms": {c: prompt_term(c) for c in ds.categories},
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, indent=2))
        print(f"[saved] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
