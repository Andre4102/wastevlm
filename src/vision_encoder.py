"""Uniform vision encoder interface for DINOv2 / DINOv3 / RADIO.

Supported encoder_id strings
-----------------------------
  dinov2-b   HF facebook/dinov2-base          ViT-B/14, patch 14, dim 768
  dinov2-l   HF facebook/dinov2-large         ViT-L/14, patch 14, dim 1024
  dinov3-b   local dinov3-vitb16-pretrain-…   ViT-B/16, patch 16, dim 768
  radio-b    local RADIO-B                    patch 16, dim 768
  radio-l    local RADIO-L                    patch 16, dim 1024
  radio-h    local RADIO-H                    patch 16, dim 1280

All backbones are frozen (no_grad, eval mode). Gradients may still flow
through a projector or probe head placed on top of EncoderOutput tensors.

Quick smoke-test
----------------
    python -m src.vision_encoder --encoder radio-l \
        --image /path/to/image.jpg
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Weights live on Leonardo scratch (big files; home has a 50 GB quota).
# Override with WASTE_VLM_WEIGHTS to relocate.
WEIGHTS_ROOT = Path(
    os.environ.get(
        "WASTE_VLM_WEIGHTS",
        "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/weights",
    )
)

_IN_MEAN = (0.485, 0.456, 0.406)
_IN_STD = (0.229, 0.224, 0.225)


def _sync_remote_code_cache(repo_dirname: str, snapshot_dir: str) -> None:
    """Copy every .py from a local trust_remote_code snapshot into the transformers
    dynamic-modules cache, so an incomplete auto-copy can't break `from_pretrained`."""
    import shutil

    hf_home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache/huggingface"))
    mods = hf_home / "modules" / "transformers_modules" / repo_dirname
    if not mods.exists():
        return  # nothing copied yet; the retry's fresh copy will populate it
    for py in Path(snapshot_dir).glob("*.py"):
        dst = mods / py.name
        if not dst.exists():
            shutil.copy2(py, dst)

# encoder_id → (family, hf_id_or_local_dir, patch_size, default_image_size)
_CONFIGS: dict[str, dict] = {
    "dinov2-b": {"family": "dinov2", "hf_id": "facebook/dinov2-base",  "patch": 14, "size": 518},
    "dinov2-l": {"family": "dinov2", "hf_id": "facebook/dinov2-large", "patch": 14, "size": 518},
    "dinov3-b": {"family": "dinov3", "variant": "vitb16", "local": "dinov3-vitb16-pretrain-lvd1689m", "patch": 16, "size": 512},
    "dinov3-l": {"family": "dinov3", "variant": "vitl16-lvd", "local": "dinov3-vitl16-pretrain-lvd1689m", "patch": 16, "size": 512},
    "radio-b":  {"family": "radio",  "local": "RADIO-B",               "patch": 16, "size": 512},
    "radio-l":  {"family": "radio",  "local": "RADIO-L",               "patch": 16, "size": 512},
    "radio-h":  {"family": "radio",  "local": "RADIO-H",               "patch": 16, "size": 512},
    # C-RADIOv4: agglomerative model distilled from SigLIP2 + DINOv3 + SAM3 teachers.
    # Same HF trust_remote_code interface as RADIO (summary, features); patch_dim
    # is auto-detected so the projector self-sizes. Gated (NVIDIA Open Model License).
    "cradiov4-h":  {"family": "radio", "local": "C-RADIOv4-H",      "patch": 16, "size": 512},
    "cradiov4-so": {"family": "radio", "local": "C-RADIOv4-SO400M", "patch": 16, "size": 512},
}


@dataclass
class EncoderOutput:
    """Raw (un-normalized) encoder features.

    cls     [B, D]    — CLS / summary token
    patches [B, N, D] — patch tokens; N = (image_size / patch_size)^2
    """
    cls: torch.Tensor
    patches: torch.Tensor


class VisionEncoder:
    """Frozen backbone wrapper with a uniform encode() interface.

    Parameters
    ----------
    encoder_id : str
        One of the keys in _CONFIGS.
    device : str
        Target device (e.g. "cuda", "cpu").
    image_size : int | None
        Override the default input resolution. Must be a multiple of
        patch_size (14 for DINOv2, 16 for DINOv3/RADIO).
    """

    def __init__(
        self,
        encoder_id: str,
        device: str = "cuda",
        image_size: int | None = None,
    ) -> None:
        if encoder_id not in _CONFIGS:
            raise ValueError(f"Unknown encoder_id {encoder_id!r}. Choose from {list(_CONFIGS)}")
        cfg = _CONFIGS[encoder_id]
        self.encoder_id = encoder_id
        self.family: str = cfg["family"]
        self.patch_size: int = cfg["patch"]
        self.image_size: int = image_size or cfg["size"]
        self.device = device

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size={self.image_size} must be divisible by patch_size={self.patch_size}"
            )

        self.model = self._load_backbone(cfg)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

        self._transform = self._build_transform()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, images: Union[Image.Image, list[Image.Image]]) -> EncoderOutput:
        """Encode one or more PIL images.

        Returns EncoderOutput with un-normalized features. Caller is
        responsible for L2-normalizing if needed (e.g. for a linear probe).
        """
        if isinstance(images, Image.Image):
            images = [images]
        pixel_values = self._preprocess(images)
        return self.encode_tensor(pixel_values)

    @torch.no_grad()
    def encode_tensor(self, pixel_values: torch.Tensor) -> EncoderOutput:
        """Encode a pre-processed [B, 3, H, W] tensor (already on self.device).

        For RADIO, pixel_values must be in [0, 1] range (not ImageNet-normed).
        For DINOv2/DINOv3, pixel_values must be ImageNet-normalized.
        Use encode() to handle the transform automatically.
        """
        if self.family == "dinov2":
            out = self.model(pixel_values=pixel_values)
            cls = out.pooler_output           # [B, D] — CLS after layernorm
            patches = out.last_hidden_state[:, 1:]  # drop CLS, [B, N, D]
        elif self.family == "dinov3":
            out = self.model.forward_features(pixel_values)
            cls = out["x_norm_clstoken"]      # [B, D]
            patches = out["x_norm_patchtokens"]  # [B, N, D] (register tokens stripped)
        else:  # radio
            summary, spatial = self.model(pixel_values)
            cls = summary                     # [B, D]
            patches = spatial                 # [B, N, D]
        return EncoderOutput(cls=cls.float(), patches=patches.float())

    @property
    def transform(self) -> transforms.Compose:
        """The exact preprocessing pipeline (PIL -> CPU tensor) this encoder
        expects. Exposed so dataloaders can preprocess images identically."""
        return self._transform

    @property
    def patch_dim(self) -> int:
        """Patch-token feature dimension (may differ from the CLS/summary dim,
        e.g. RADIO summary=3072 but patches=1024). This is the dim a LLaVA-style
        projector must consume."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
            out = self.encode_tensor(dummy)
        return int(out.patches.shape[-1])

    @property
    def feature_dim(self) -> int:
        """Backbone feature dimension D."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
            out = self.encode_tensor(dummy)
        return int(out.cls.shape[-1])

    @property
    def n_patches(self) -> int:
        """Number of patch tokens N = (image_size / patch_size)^2."""
        g = self.image_size // self.patch_size
        return g * g

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_backbone(self, cfg: dict) -> torch.nn.Module:
        family = cfg["family"]
        if family == "dinov2":
            from transformers import AutoModel
            return AutoModel.from_pretrained(cfg["hf_id"], torch_dtype=torch.float32).to(self.device)
        if family == "dinov3":
            from src.dinov3_backbone import (
                load_dinov3_vitb16,
                load_dinov3_vitl16,
                load_dinov3_vitl16_lvd,
            )
            loaders = {
                "vitb16": load_dinov3_vitb16,
                "vitl16-sat": load_dinov3_vitl16,       # SAT493M
                "vitl16-lvd": load_dinov3_vitl16_lvd,   # LVD1689M
            }
            return loaders[cfg["variant"]](device=self.device)
        if family == "radio":
            from transformers import AutoModel
            local_path = str(WEIGHTS_ROOT / cfg["local"])
            try:
                model = AutoModel.from_pretrained(local_path, trust_remote_code=True)
            except FileNotFoundError:
                # transformers' trust_remote_code copier can miss some modeling
                # files (e.g. C-RADIOv4's utils.py / dual_hybrid_vit.py) when it
                # scans relative imports. Mirror all .py into the modules cache
                # and retry so a cleared cache can't silently break loading.
                _sync_remote_code_cache(cfg["local"], local_path)
                model = AutoModel.from_pretrained(local_path, trust_remote_code=True)
            return model.to(self.device)
        raise ValueError(family)

    def _build_transform(self) -> transforms.Compose:
        if self.family == "radio":
            # RADIO ingest [0,1]; internal normalisation is baked into the model.
            return transforms.Compose([
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),  # [0,1]
            ])
        return transforms.Compose([
            transforms.Resize(
                (self.image_size, self.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IN_MEAN, std=_IN_STD),
        ])

    def _preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        tensors = [self._transform(img.convert("RGB")) for img in images]
        return torch.stack(tensors).to(self.device)

    def __repr__(self) -> str:
        return (
            f"VisionEncoder(id={self.encoder_id!r}, family={self.family!r}, "
            f"image_size={self.image_size}, patch_size={self.patch_size})"
        )


# ---------------------------------------------------------------------------
# Smoke-test CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Smoke-test a vision encoder on one image.")
    p.add_argument("--encoder", default="radio-l", choices=list(_CONFIGS))
    p.add_argument("--image", required=True, help="Path to a test image")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    print(f"[load] encoder={args.encoder} device={args.device}")
    enc = VisionEncoder(args.encoder, device=args.device)
    print(f"  {enc}")
    print(f"  feature_dim={enc.feature_dim}  n_patches={enc.n_patches}")

    img = Image.open(args.image).convert("RGB")
    out = enc.encode(img)

    cls_norm = F.normalize(out.cls, dim=-1)
    print(f"  cls:     shape={tuple(out.cls.shape)}  norm={cls_norm.norm().item():.4f} (post-norm)")
    print(f"  patches: shape={tuple(out.patches.shape)}")
    print("[ok]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
