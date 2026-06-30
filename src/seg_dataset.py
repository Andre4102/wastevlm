"""DroneWaste semantic-segmentation dataset.

Rasterises COCO-style polygon annotations into per-pixel class masks at
the model's working resolution. Same site-stratified 70/15/15 split as
`det_dataset` (seed=0) so seg ↔ det results are directly comparable.

Class indices in the rasterised mask:
    0           -> background
    1..N        -> 0-based foreground category index + 1
                   (i.e. cat_id_to_idx[c] + 1)
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms

from src.det_dataset import DEFAULT_ROOT, site_stratified_split


def _norm_tx(image_size: int) -> transforms.Compose:
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


class DroneWasteSegmentation(Dataset):
    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        image_size: int = 518,
        split: str = "train",
        seed: int = 0,
        multilabel: bool = False,
    ) -> None:
        root = Path(root)
        with (root / "dronewaste_v1.0.json").open() as f:
            data = json.load(f)

        self.categories = [c["name"] for c in data["categories"]]
        self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(data["categories"])}

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

        tr, va, te = site_stratified_split(all_images, seed=seed)
        idx_by_split = {"train": tr, "val": va, "test": te}
        if split not in idx_by_split:
            raise ValueError(f"unknown split {split!r}")

        self.samples = [all_images[i] for i in idx_by_split[split]]
        self.images_dir = root / "images"
        self.image_size = image_size
        self.transform = _norm_tx(image_size)
        self.split = split
        self.multilabel = multilabel
        self.num_classes = len(self.categories)  # foreground only

    def __len__(self) -> int:
        return len(self.samples)

    def _rasterise(self, anns: list[dict], orig_w: int, orig_h: int) -> torch.Tensor:
        """Return [image_size, image_size] long tensor of class indices."""
        mask = Image.new("I", (self.image_size, self.image_size), 0)
        draw = ImageDraw.Draw(mask)
        # Larger area first; smaller polygons overwrite -> they "win" overlaps
        anns_sorted = sorted(anns, key=lambda a: -float(a.get("area", 0.0)))
        sx = self.image_size / orig_w
        sy = self.image_size / orig_h
        for a in anns_sorted:
            cls = self.cat_id_to_idx[a["category_id"]] + 1  # +1 for bg=0
            seg = a.get("segmentation")
            if not isinstance(seg, list):
                continue
            for poly in seg:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                pts = [(poly[i] * sx, poly[i + 1] * sy) for i in range(0, len(poly) - 1, 2)]
                if len(pts) >= 3:
                    draw.polygon(pts, fill=cls)
        return torch.from_numpy(np.array(mask, dtype=np.int64))

    def _rasterise_multilabel(self, anns: list[dict], orig_w: int, orig_h: int) -> torch.Tensor:
        """Return [C, image_size, image_size] float multi-hot mask.

        Each foreground class is drawn into its own channel; overlaps are
        preserved (a pixel can be 1 in several channels). No background
        channel — a pixel is background iff all channels are 0.
        """
        C = self.num_classes
        sx = self.image_size / orig_w
        sy = self.image_size / orig_h
        chans = [Image.new("1", (self.image_size, self.image_size), 0) for _ in range(C)]
        draws = [ImageDraw.Draw(ch) for ch in chans]
        for a in anns:
            cls = self.cat_id_to_idx[a["category_id"]]  # 0-based fg index
            seg = a.get("segmentation")
            if not isinstance(seg, list):
                continue
            for poly in seg:
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                pts = [(poly[i] * sx, poly[i + 1] * sy) for i in range(0, len(poly) - 1, 2)]
                if len(pts) >= 3:
                    draws[cls].polygon(pts, fill=1)
        arr = np.stack([np.asarray(ch, dtype=np.float32) for ch in chans])  # [C,H,W]
        return torch.from_numpy(arr)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        s = self.samples[idx]
        img = Image.open(self.images_dir / s["file_name"]).convert("RGB")
        pixel_values = self.transform(img)
        if self.multilabel:
            mask = self._rasterise_multilabel(s["annotations"], s["width"], s["height"])
        else:
            mask = self._rasterise(s["annotations"], s["width"], s["height"])
        target = {
            "mask": mask,
            "image_id": torch.tensor([s["id"]]),
            "orig_size": torch.tensor([s["height"], s["width"]]),
        }
        return pixel_values, target


def collate_seg(batch: Iterable[tuple]) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    pix = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1]["mask"] for b in batch])
    meta = [{"image_id": b[1]["image_id"], "orig_size": b[1]["orig_size"]} for b in batch]
    return pix, masks, meta
