"""DroneWaste detection dataset.

Yields (pixel_values, target_dict) where target_dict matches HF DETR's
expectation: `class_labels` LongTensor, `boxes` FloatTensor in normalised
cxcywh. Also tracks image_id + orig_size for COCO-mAP eval at test time.

Site-stratified 70/15/15 train/val/test split with seed=0, identical
partitioning to dino_probe's DroneWaste split so the detection results are
comparable.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_ROOT = Path("/home/ids/diecidue/data/dronewaste")


def _norm_tx(image_size: int) -> transforms.Compose:
    """DINOv2 ImageNet normalisation + resize to fixed square."""
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def site_stratified_split(
    samples: list[dict], seed: int = 0, frac_train: float = 0.70, frac_val: float = 0.15
) -> tuple[list[int], list[int], list[int]]:
    """Stratify image IDs by site, return (train, val, test) index lists."""
    rng = np.random.default_rng(seed)
    by_site: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_site[s["site"]].append(i)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for site, idxs in by_site.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        n = len(idxs)
        n_tr = int(n * frac_train)
        n_va = int(n * frac_val)
        train_idx.extend(idxs[:n_tr])
        val_idx.extend(idxs[n_tr : n_tr + n_va])
        test_idx.extend(idxs[n_tr + n_va :])
    return train_idx, val_idx, test_idx


class DroneWasteDetection(Dataset):
    """COCO-style detection dataset over DroneWaste."""

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        image_size: int = 518,
        split: str = "train",
        seed: int = 0,
    ) -> None:
        root = Path(root)
        with (root / "dronewaste_v1.0.json").open() as f:
            data = json.load(f)

        # category_id (1-based, with possible gaps) -> 0-based contiguous index
        self.categories = [c["name"] for c in data["categories"]]
        cat_id_to_idx = {c["id"]: i for i, c in enumerate(data["categories"])}
        self.cat_id_to_idx = cat_id_to_idx
        self.id_to_cat_id = {i: c["id"] for c, i in zip(data["categories"], range(len(data["categories"])))}

        ann_by_image: dict[int, list[dict]] = defaultdict(list)
        for a in data.get("annotations", []):
            ann_by_image[a["image_id"]].append(a)

        all_images = []
        for img in data["images"]:
            all_images.append(
                {
                    "id": int(img["id"]),
                    "file_name": img["file_name"],
                    "site": img["site"],
                    "width": int(img["width"]),
                    "height": int(img["height"]),
                    "annotations": ann_by_image.get(int(img["id"]), []),
                }
            )

        # Site-stratified split (same partitioning regardless of which split
        # is requested — we just slice).
        train_idx, val_idx, test_idx = site_stratified_split(all_images, seed=seed)
        idx_by_split = {"train": train_idx, "val": val_idx, "test": test_idx}
        if split not in idx_by_split:
            raise ValueError(f"unknown split {split!r}")

        self.samples = [all_images[i] for i in idx_by_split[split]]
        self.images_dir = root / "images"
        self.image_size = image_size
        self.transform = _norm_tx(image_size)
        self.split = split

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        s = self.samples[idx]
        img = Image.open(self.images_dir / s["file_name"]).convert("RGB")
        orig_w, orig_h = s["width"], s["height"]
        pixel_values = self.transform(img)

        boxes_xyxy: list[list[float]] = []
        class_labels: list[int] = []
        for a in s["annotations"]:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes_xyxy.append([x, y, x + w, y + h])
            class_labels.append(self.cat_id_to_idx[a["category_id"]])

        if boxes_xyxy:
            b = torch.tensor(boxes_xyxy, dtype=torch.float32)
            # Convert xyxy -> normalised cxcywh
            cx = (b[:, 0] + b[:, 2]) / 2 / orig_w
            cy = (b[:, 1] + b[:, 3]) / 2 / orig_h
            bw = (b[:, 2] - b[:, 0]) / orig_w
            bh = (b[:, 3] - b[:, 1]) / orig_h
            boxes = torch.stack([cx, cy, bw, bh], dim=1)
            labels = torch.tensor(class_labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        target = {
            "class_labels": labels,
            "boxes": boxes,
            "image_id": torch.tensor([s["id"]]),
            "orig_size": torch.tensor([orig_h, orig_w]),
            "size": torch.tensor([self.image_size, self.image_size]),
        }
        return pixel_values, target


def collate_detection(batch: Iterable[tuple]) -> tuple[torch.Tensor, list[dict]]:
    pix = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return pix, targets
