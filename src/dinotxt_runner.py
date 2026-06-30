"""Thin loader for dino.txt (DINOv2 ViT-L/14 + text head, CVPR 2025).

Loads the public pretrained model from torch.hub and wraps the two
inference paths we use:

    - image-level CLIP-style classification:
        encode_image_batch(...) -> [B, 2048] (L2-normalised)
        encode_text_classes(...) -> [C, 2048] (L2-normalised, mean over templates)

    - pixel-level open-vocabulary segmentation:
        get_patch_features(...) -> [B, 1024, H, W] (L2-normalised over channel)
        encode_text_classes_patch(...) -> [C, 1024] (L2-normalised, mean over templates)
      Use einsum("bdhw,cd->bchw", patch_feats, text_feats) for per-pixel logits.

Model card: dino.txt embed_dim=2048. First 1024 dims are aligned to the
image CLS token, last 1024 dims to the patch-average pooled token.
For pixel-level prediction the patch features (1024-d) must be compared
against the *patch-aligned* slice of the text feature, `text[:, 1024:]`.
This is the recipe from the official dinotxt.ipynb.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
import torch.nn.functional as F


# Patch-aligned text slice. Hardcoded — half of the 2048-d joint embed.
PATCH_TEXT_OFFSET = 1024


AERIAL_TEMPLATES: tuple[Callable[[str], str], ...] = (
    lambda c: f"an aerial photograph of {c}.",
    lambda c: f"a drone photo showing {c}.",
    lambda c: f"a satellite image of {c}.",
    lambda c: f"a top-down aerial view of {c}.",
    lambda c: f"aerial imagery of {c}.",
    lambda c: f"a high-resolution aerial photo of {c}.",
    lambda c: f"a photo of {c} from above.",
    lambda c: f"a bird's-eye view of {c}.",
)


SIMPLE_TEMPLATES: tuple[Callable[[str], str], ...] = (
    lambda c: f"a photo of {c}.",
)


@dataclass
class DinoTxtRunner:
    image_size: int = 518  # multiple of 14 (37 patches)
    device: str = "cuda"
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l",
            verbose=False,
        ).to(self.device).eval()
        from dinov2.hub.dinotxt import get_tokenizer  # type: ignore
        self.tokenizer = get_tokenizer()
        self.transform = self._build_transform(self.image_size)

    def _build_transform(self, image_size: int):
        from torchvision import transforms
        # dino.txt uses ImageNet stats and 224 default. We override resolution
        # and rely on positional-embedding interpolation in the backbone.
        return transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
            ),
        ])

    @torch.no_grad()
    def encode_image_batch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B, 3, H, W] -> [B, 2048] L2-normalised."""
        pixel_values = pixel_values.to(self.device)
        with torch.autocast(self.device, dtype=self.dtype):
            feats = self.model.encode_image(pixel_values)
        feats = F.normalize(feats.float(), p=2, dim=-1)
        return feats

    @torch.no_grad()
    def encode_text_classes(
        self,
        class_names: Sequence[str],
        templates: Sequence[Callable[[str], str]] = AERIAL_TEMPLATES,
    ) -> torch.Tensor:
        """Return [C, 2048] class embeddings (avg over templates, L2-normed)."""
        out: list[torch.Tensor] = []
        for name in class_names:
            prompts = [t(name.lower()) for t in templates]
            toks = self.tokenizer.tokenize(prompts).to(self.device)
            with torch.autocast(self.device, dtype=self.dtype):
                emb = self.model.encode_text(toks)
            emb = F.normalize(emb.float(), p=2, dim=-1)
            emb = emb.mean(dim=0)
            emb = F.normalize(emb, p=2, dim=-1)
            out.append(emb)
        return torch.stack(out, dim=0)

    @torch.no_grad()
    def encode_text_classes_patch(
        self,
        class_names: Sequence[str],
        templates: Sequence[Callable[[str], str]] = AERIAL_TEMPLATES,
    ) -> torch.Tensor:
        """Patch-aligned text slice, [C, 1024], L2-normed.

        Slice MUST happen before normalisation — patches and CLS halves of
        the joint 2048-d embed have different magnitudes by design.
        """
        out: list[torch.Tensor] = []
        for name in class_names:
            prompts = [t(name.lower()) for t in templates]
            toks = self.tokenizer.tokenize(prompts).to(self.device)
            with torch.autocast(self.device, dtype=self.dtype):
                emb = self.model.encode_text(toks)
            emb = emb[:, PATCH_TEXT_OFFSET:]  # [T, 1024]
            emb = F.normalize(emb.float(), p=2, dim=-1)
            emb = emb.mean(dim=0)
            emb = F.normalize(emb, p=2, dim=-1)
            out.append(emb)
        return torch.stack(out, dim=0)

    @torch.no_grad()
    def get_patch_features(
        self,
        pixel_values: torch.Tensor,
        out_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """[B, 3, H, W] -> [B, 1024, h, w] L2-normalised over channel.

        If out_size given, bicubic-upsample to that resolution before
        normalisation (per the notebook recipe).
        """
        pixel_values = pixel_values.to(self.device)
        with torch.autocast(self.device, dtype=self.dtype):
            _cls, patches = self.model.get_visual_class_and_patch_tokens(pixel_values)
        # patches: [B, P, 1024]
        B, P, D = patches.shape
        h = w = int(math.sqrt(P))
        assert h * w == P, f"expected square patch grid, got P={P}"
        x = patches.float().movedim(2, 1).unflatten(2, (h, w))  # [B, D, h, w]
        if out_size is not None:
            x = F.interpolate(x, size=out_size, mode="bicubic", align_corners=False)
        x = F.normalize(x, p=2, dim=1)
        return x
