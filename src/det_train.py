"""Train DinoDETR on DroneWaste.

Usage:
    python -m src.det_train --epochs 50 --batch-size 16 --lr 1e-4 \
        --out /home/ids/diecidue/results/waste_vlm/dinodetr_dw

Frozen DINOv2-B backbone, only the projection / decoder / heads / object
queries are trainable. AdamW + cosine LR. Logs train + val loss each epoch
and saves the best-by-val-loss checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.models.detr.modeling_detr import DetrHungarianMatcher, DetrLoss

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.det_dataset import DroneWasteDetection, collate_detection  # noqa: E402
from src.det_model import DinoDETR, DinoDETRConfig  # noqa: E402


def build_criterion(num_classes: int) -> DetrLoss:
    matcher = DetrHungarianMatcher(class_cost=1.0, bbox_cost=5.0, giou_cost=2.0)
    criterion = DetrLoss(
        matcher=matcher,
        num_classes=num_classes,
        eos_coef=0.1,
        losses=["labels", "boxes", "cardinality"],
    )
    return criterion


LOSS_WEIGHTS = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}


def step(
    model: DinoDETR,
    criterion: DetrLoss,
    pixel_values: torch.Tensor,
    targets: list[dict],
    device: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    pixel_values = pixel_values.to(device)
    targets = [
        {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in t.items()
        }
        for t in targets
    ]
    out = model(pixel_values)
    losses = criterion(out, targets)
    total = sum(LOSS_WEIGHTS.get(k, 0.0) * v for k, v in losses.items() if k in LOSS_WEIGHTS)
    return total, {k: float(v.detach().cpu()) for k, v in losses.items()}


@torch.no_grad()
def run_val(
    model: DinoDETR, criterion: DetrLoss, loader: DataLoader, device: str
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    n = 0
    for pix, tgts in loader:
        total, parts = step(model, criterion, pix, tgts, device)
        sums["loss_total"] = sums.get("loss_total", 0.0) + float(total.detach().cpu())
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
    model.train()
    if n == 0:
        return sums
    return {k: v / n for k, v in sums.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=518)
    p.add_argument("--num-queries", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True, help="output dir for checkpoints + log")
    p.add_argument("--log-every", type=int, default=20)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    train_ds = DroneWasteDetection(split="train", image_size=args.image_size, seed=args.seed)
    val_ds = DroneWasteDetection(split="val", image_size=args.image_size, seed=args.seed)
    num_classes = len(train_ds.categories)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  classes={num_classes}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_detection, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_detection, num_workers=args.num_workers, pin_memory=True,
    )

    cfg = DinoDETRConfig(
        image_size=args.image_size,
        num_classes=num_classes,
        num_queries=args.num_queries,
        freeze_backbone=True,
    )
    device = "cuda"
    model = DinoDETR(cfg).to(device)
    criterion = build_criterion(num_classes).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs * len(train_loader))

    history: list[dict] = []
    best_val = float("inf")
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_loss_parts: dict[str, float] = {}
        n_steps = 0
        for it, (pix, tgts) in enumerate(train_loader):
            total, parts = step(model, criterion, pix, tgts, device)
            optim.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            optim.step()
            sched.step()
            train_loss_sum += float(total.detach().cpu())
            for k, v in parts.items():
                train_loss_parts[k] = train_loss_parts.get(k, 0.0) + v
            n_steps += 1
            if (it + 1) % args.log_every == 0:
                avg = train_loss_sum / n_steps
                lr_now = optim.param_groups[0]["lr"]
                elapsed = time.time() - t_start
                print(f"  e{epoch:03d} it{it+1:04d}/{len(train_loader)} "
                      f"loss={avg:.4f} lr={lr_now:.2e} elapsed={elapsed/60:.1f}min", flush=True)

        train_avg = {"loss_total": train_loss_sum / max(1, n_steps)}
        for k, v in train_loss_parts.items():
            train_avg[k] = v / max(1, n_steps)
        val_avg = run_val(model, criterion, val_loader, device)
        row = {"epoch": epoch, "train": train_avg, "val": val_avg, "lr": optim.param_groups[0]["lr"]}
        history.append(row)
        print(f"[epoch {epoch:03d}] train_loss={train_avg['loss_total']:.4f}  "
              f"val_loss={val_avg.get('loss_total', float('nan')):.4f}", flush=True)

        with (args.out / "history.json").open("w") as f:
            json.dump(history, f, indent=2)

        if val_avg.get("loss_total", float("inf")) < best_val:
            best_val = val_avg["loss_total"]
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "categories": train_ds.categories},
                       args.out / "best.pt")
            print(f"  [best] val_loss={best_val:.4f} saved -> {args.out/'best.pt'}", flush=True)

    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "categories": train_ds.categories},
               args.out / "last.pt")
    print(f"[done] best_val={best_val:.4f}  total_time={(time.time()-t_start)/60:.1f}min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
