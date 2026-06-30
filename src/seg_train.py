"""Train frozen DINOv2 + seg head on DroneWaste semantic segmentation.

Loss: weighted cross-entropy (background down-weighted) + foreground Dice.
Logs train loss + val loss + val mIoU each epoch, saves best.pt by mIoU_fg.

Usage:
    python -m src.seg_train --epochs 50 --batch-size 32 --lr 1e-3 \
        --head linear --out /home/ids/diecidue/results/waste_vlm/dinoseg_dw
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.seg_dataset import DroneWasteSegmentation, collate_seg  # noqa: E402
from src.seg_model import DinoSeg, DinoSegConfig  # noqa: E402


def dice_loss_fg(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean Dice loss over foreground classes (excludes channel 0 = background)."""
    probs = F.softmax(logits, dim=1)
    C = logits.size(1)
    onehot = F.one_hot(target.clamp_min(0), C).permute(0, 3, 1, 2).float()
    p_fg = probs[:, 1:]
    t_fg = onehot[:, 1:]
    inter = (p_fg * t_fg).sum(dim=(2, 3))
    denom = p_fg.sum(dim=(2, 3)) + t_fg.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def bce_dice_multilabel(
    logits: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Per-class BCE (with pos_weight) + per-class soft Dice.

    logits, target: [B, C, H, W]; target is a float multi-hot mask. Dice is
    computed per class then averaged, which keeps gradient on rare classes
    and prevents the all-negative collapse that pure BCE risks under the
    heavy background imbalance.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(2, 3))
    denom = probs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return bce + (1.0 - dice.mean())


@torch.no_grad()
def run_val_multilabel(
    model: DinoSeg, loader: DataLoader, device: str, num_classes: int,
    pos_weight: torch.Tensor, thresh: float = 0.5,
) -> dict:
    """Per-class IoU at a fixed sigmoid threshold (no argmax / no bg channel)."""
    model.eval()
    inter = torch.zeros(num_classes)
    union = torch.zeros(num_classes)
    loss_sum = 0.0
    n = 0
    for pix, masks, _ in loader:
        pix = pix.to(device)
        masks = masks.to(device)
        logits = model(pix)
        loss = bce_dice_multilabel(logits, masks, pos_weight)
        pred = torch.sigmoid(logits) > thresh
        t = masks > 0.5
        inter += (pred & t).sum(dim=(0, 2, 3)).cpu().float()
        union += (pred | t).sum(dim=(0, 2, 3)).cpu().float()
        loss_sum += float(loss.detach().cpu())
        n += 1
    model.train()
    iou = inter / (union + 1e-6)
    return {
        "loss": loss_sum / max(1, n),
        "mIoU_fg": float(iou.mean()),
        "mIoU_all": float(iou.mean()),
        "iou_per_class": iou.tolist(),
    }


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_logits: int) -> torch.Tensor:
    k = (target >= 0) & (target < num_logits)
    idx = (target[k] * num_logits + pred[k]).long()
    return torch.bincount(idx, minlength=num_logits ** 2).reshape(num_logits, num_logits)


@torch.no_grad()
def run_val(
    model: DinoSeg, loader: DataLoader, device: str, num_logits: int, class_weights: torch.Tensor
) -> dict:
    model.eval()
    cm = torch.zeros(num_logits, num_logits, dtype=torch.long)
    loss_sum = 0.0
    n = 0
    for pix, masks, _ in loader:
        pix = pix.to(device)
        masks = masks.to(device)
        logits = model(pix)
        loss = F.cross_entropy(logits, masks, weight=class_weights) + dice_loss_fg(logits, masks)
        pred = logits.argmax(1)
        cm += confusion_matrix(pred.cpu(), masks.cpu(), num_logits)
        loss_sum += float(loss.detach().cpu())
        n += 1
    model.train()
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    iou = tp / (tp + fp + fn + 1e-6)
    return {
        "loss": loss_sum / max(1, n),
        "mIoU_fg": float(iou[1:].mean()) if iou.numel() > 1 else 0.0,
        "mIoU_all": float(iou.mean()),
        "iou_per_class": iou.tolist(),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=518)
    p.add_argument("--head", choices=["linear", "conv", "fpn"], default="linear")
    p.add_argument("--backbone-id", type=str, default="facebook/dinov2-base")
    p.add_argument("--backbone-type", choices=["dinov2", "dinov3", "radio", "frcnn"],
                   default="dinov2")
    p.add_argument("--bg-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--multi-block", type=str, default="",
                   help="Phase-2 FPN-lite: comma-separated ViT block indices to "
                        "concat patch tokens from (e.g. '3,7,11'). Empty = last "
                        "block only (default). dinov2 / dinov3 only.")
    p.add_argument("--fpn-blocks", type=str, default="2,5,8,11",
                   help="Comma-separated ViT block indices for the FPN head "
                        "(shallowest→deepest). Used only when --head fpn.")
    p.add_argument("--fpn-dim", type=int, default=256,
                   help="FPN pathway channel width (default 256).")
    p.add_argument("--fpn-merge", choices=["add", "concat"], default="add",
                   help="How to merge features in the FPN top-down pass: "
                        "'add' (element-wise, default) or 'concat' (channel concat, "
                        "more expressive but ~2× merge_conv params).")
    p.add_argument("--multilabel", action="store_true",
                   help="Multi-label seg: per-class sigmoid (no bg channel), "
                        "BCE+Dice loss, threshold-based val mIoU. Lets a pixel "
                        "belong to several classes (matches the annotations).")
    p.add_argument("--pos-weight", type=float, default=8.0,
                   help="BCE positive-class weight (multi-label only); offsets "
                        "the heavy background imbalance.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds = DroneWasteSegmentation(split="train", image_size=args.image_size,
                                      seed=args.seed, multilabel=args.multilabel)
    val_ds = DroneWasteSegmentation(split="val", image_size=args.image_size,
                                    seed=args.seed, multilabel=args.multilabel)
    num_classes = train_ds.num_classes
    num_logits = num_classes if args.multilabel else num_classes + 1
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  fg classes={num_classes}"
          f"  mode={'multilabel' if args.multilabel else 'single-label'}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_seg, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_seg, num_workers=args.num_workers, pin_memory=True,
    )

    # Resolve backbone_id defaults per type.
    backbone_id = args.backbone_id
    if args.backbone_type == "radio" and backbone_id == "facebook/dinov2-base":
        backbone_id = "/home/ids/diecidue/results/waste_vlm/weights/RADIO-L"
    if args.backbone_type == "frcnn":
        backbone_id = "torchvision/fasterrcnn_resnet50_fpn"  # sentinel; loaded internally
    if args.backbone_type == "dinov3" and backbone_id == "facebook/dinov2-base":
        backbone_id = "dinov3/vitb16"  # sentinel; seg_model checks for "vitl" to select L

    multi_block = (
        tuple(int(x) for x in args.multi_block.split(",") if x.strip())
        if args.multi_block else None
    )
    fpn_blocks = tuple(int(x) for x in args.fpn_blocks.split(",") if x.strip())
    cfg = DinoSegConfig(
        backbone_id=backbone_id,
        backbone_type=args.backbone_type,
        image_size=args.image_size, num_classes=num_classes,
        head=args.head, freeze_backbone=True,
        multi_block=multi_block, multilabel=args.multilabel,
        fpn_blocks=fpn_blocks, fpn_dim=args.fpn_dim,
        fpn_merge=args.fpn_merge,
    )
    device = "cuda"
    model = DinoSeg(cfg).to(device)

    if args.multilabel:
        # pos_weight shape [C,1,1] broadcasts over B,H,W in BCE.
        pos_weight = torch.full((num_classes, 1, 1), args.pos_weight, device=device)
        class_weights = None
    else:
        class_weights = torch.ones(num_logits, device=device)
        class_weights[0] = args.bg_weight
        pos_weight = None

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"[model] head={args.head}  trainable_params={n_params:,}")

    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs * len(train_loader))

    history: list[dict] = []
    best_miou = -1.0
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        loss_sum = 0.0
        n_steps = 0
        for it, (pix, masks, _) in enumerate(train_loader):
            pix = pix.to(device)
            masks = masks.to(device)
            logits = model(pix)
            if args.multilabel:
                loss = bce_dice_multilabel(logits, masks, pos_weight)
            else:
                loss = F.cross_entropy(logits, masks, weight=class_weights) + dice_loss_fg(logits, masks)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            sched.step()
            loss_sum += float(loss.detach().cpu())
            n_steps += 1
            if (it + 1) % args.log_every == 0:
                avg = loss_sum / n_steps
                lr_now = optim.param_groups[0]["lr"]
                elapsed = time.time() - t_start
                print(f"  e{epoch:03d} it{it+1:04d}/{len(train_loader)} "
                      f"loss={avg:.4f} lr={lr_now:.2e} elapsed={elapsed/60:.1f}min", flush=True)

        train_avg = loss_sum / max(1, n_steps)
        if args.multilabel:
            val_m = run_val_multilabel(model, val_loader, device, num_classes, pos_weight)
        else:
            val_m = run_val(model, val_loader, device, num_logits, class_weights)
        row = {"epoch": epoch, "train_loss": train_avg, **val_m, "lr": optim.param_groups[0]["lr"]}
        history.append(row)
        print(f"[epoch {epoch:03d}] train={train_avg:.4f}  "
              f"val_loss={val_m['loss']:.4f}  mIoU_fg={val_m['mIoU_fg']:.4f}", flush=True)

        with (args.out / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

        if val_m["mIoU_fg"] > best_miou:
            best_miou = val_m["mIoU_fg"]
            torch.save(
                {"model": model.state_dict(), "cfg": cfg.__dict__,
                 "categories": train_ds.categories},
                args.out / "best.pt",
            )
            print(f"  [best] mIoU_fg={best_miou:.4f} saved -> {args.out/'best.pt'}", flush=True)

    torch.save(
        {"model": model.state_dict(), "cfg": cfg.__dict__,
         "categories": train_ds.categories},
        args.out / "last.pt",
    )
    print(f"[done] best_mIoU_fg={best_miou:.4f}  total_time={(time.time()-t_start)/60:.1f}min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
