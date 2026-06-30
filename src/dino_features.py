"""Extract DINOv2 features for AerialWaste + DroneWaste and cache to disk.

For each image we record the CLS embedding (768-dim for ViT-B/14) at
518×518 input resolution — the DINOv2 paper's recommended fine-grained-
classification setup.

Usage:
    python -m src.dino_features --dataset aerialwaste --split testing
    python -m src.dino_features --dataset aerialwaste --split training
    python -m src.dino_features --dataset dronewaste
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import (  # noqa: E402
    load_aerialwaste,
    load_dronewaste,
)

CACHE_DIR = Path("/home/ids/diecidue/results/waste_vlm/dinov2_features")


def build_transform(image_size: int) -> transforms.Compose:
    # DINOv2 paper uses ImageNet mean/std + bicubic resize, no center crop is
    # needed because we feed exact-size square inputs.
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["aerialwaste", "dronewaste"], required=True)
    p.add_argument("--split", default="testing", help="aerialwaste only")
    p.add_argument("--model-id", default="facebook/dinov2-base")
    p.add_argument("--image-size", type=int, default=518)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    if args.dataset == "aerialwaste":
        samples = load_aerialwaste("/home/ids/diecidue/data/aerialwaste", split=args.split)
        suffix = f"aerialwaste_{args.split}"
    else:
        samples = load_dronewaste("/home/ids/diecidue/data/dronewaste")
        suffix = "dronewaste"
    if args.limit:
        samples = samples[: args.limit]

    print(f"[load] {len(samples)} samples from {args.dataset}/{args.split}")

    device = "cuda"
    print(f"[load] {args.model_id} at {args.image_size}x{args.image_size}")
    model = AutoModel.from_pretrained(args.model_id, torch_dtype=torch.float32).to(device)
    model.eval()
    tfm = build_transform(args.image_size)

    feats: list[np.ndarray] = []
    image_ids: list[str] = []
    labels: list[int] = []
    image_sources: list[str] = []
    gt_categories: list[list[str]] = []

    @torch.inference_mode()
    def embed_batch(pil_images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        tensors = torch.stack([tfm(im) for im in pil_images]).to(device)
        out = model(pixel_values=tensors)
        # pooler_output is the CLS token after the layernorm.
        cls = out.pooler_output  # [B, D]
        # last_hidden_state is [B, N+1, D] where index 0 is CLS, 1: are patches.
        patches = out.last_hidden_state[:, 1:, :]
        patch_mean = patches.mean(dim=1)
        patch_max = patches.max(dim=1).values
        return (
            cls.float().cpu().numpy(),
            patch_mean.float().cpu().numpy(),
            patch_max.float().cpu().numpy(),
        )

    cls_feats: list[np.ndarray] = []
    mean_feats: list[np.ndarray] = []
    max_feats: list[np.ndarray] = []
    pil_buf: list[Image.Image] = []
    meta_buf: list = []

    def flush() -> None:
        c, m, mx = embed_batch(pil_buf)
        cls_feats.append(c)
        mean_feats.append(m)
        max_feats.append(mx)
        for ms in meta_buf:
            image_ids.append(ms.image_id)
            labels.append(ms.label)
            image_sources.append(ms.image_source)
            gt_categories.append(
                list(ms.extra.get("gt_categories", []) or ms.extra.get("categories_present", []))
            )
        pil_buf.clear()
        meta_buf.clear()

    for s in tqdm(samples, desc="embed"):
        try:
            img = Image.open(s.image_path).convert("RGB")
        except Exception as e:
            print(f"  [skip] {s.image_path}: {e}", file=sys.stderr)
            continue
        pil_buf.append(img)
        meta_buf.append(s)
        if len(pil_buf) >= args.batch_size:
            flush()
    if pil_buf:
        flush()

    cls_arr = np.concatenate(cls_feats, axis=0)
    mean_arr = np.concatenate(mean_feats, axis=0)
    max_arr = np.concatenate(max_feats, axis=0)
    y = np.array(labels, dtype=np.int32)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{suffix}.npz"
    np.savez(
        path,
        features=cls_arr,           # back-compat: 'features' == CLS
        features_cls=cls_arr,
        features_patch_mean=mean_arr,
        features_patch_max=max_arr,
        labels=y,
        image_ids=np.array(image_ids),
        image_sources=np.array(image_sources),
        gt_categories=np.array(gt_categories, dtype=object),
        model_id=args.model_id,
        image_size=args.image_size,
    )
    print(f"[saved] {path}  cls={cls_arr.shape}  patch_mean={mean_arr.shape}  patch_max={max_arr.shape}  "
          f"pos={int(y.sum())}/{len(y)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
