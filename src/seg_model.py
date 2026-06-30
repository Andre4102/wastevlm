"""DINOv2 / DINOv3 / RADIO / Faster R-CNN + segmentation head for DroneWaste.

Frozen backbone → patch tokens. A lightweight head predicts per-patch
class logits, then bilinear-upsample to the input resolution for
pixel-level supervision.

Backbones:
    dinov2  : HF facebook/dinov2-base, patch-14 (image_size multiple of 14)
    dinov3  : DINOv3 ViT-B/16, patch-16 (image_size multiple of 16)
    radio   : RADIOv2.5-L, patch-16 (local weights)
    frcnn   : torchvision Faster R-CNN FPN P4 (stride 16, 256-d, COCO pretrained)

Head choices:
    linear  : nn.Linear(D, num_classes + 1)              ~5-22k params
    conv    : 3×3 + GELU + 1×1                           ~1.8M params
    fpn     : lateral + top-down over multi-block tokens  ~2.5M params
              (dinov2 / dinov3 / radio only; set fpn_blocks)
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


@dataclass
class DinoSegConfig:
    backbone_id: str = "facebook/dinov2-base"
    backbone_type: str = "dinov2"  # "dinov2" | "dinov3" | "radio" | "frcnn"
    image_size: int = 518
    num_classes: int = 20  # foreground classes; +1 background added internally
    head: str = "linear"  # "linear" | "conv" | "fpn"
    head_hidden: int = 256
    freeze_backbone: bool = True
    # Phase-2 FPN-lite: concatenate patch tokens from these ViT blocks along
    # the feature dim. Empty / None = use only the last block (default).
    # Block indices are 0-based and post-norm. Supported for dinov2 / dinov3.
    multi_block: tuple[int, ...] | None = None
    # Multi-label seg: one sigmoid channel per foreground class, no background
    # channel (a pixel is bg iff all channels are below threshold).
    multilabel: bool = False
    # FPN head: ViT block indices (shallowest→deepest) and pathway width.
    # Used only when head="fpn"; ignored otherwise.
    fpn_blocks: tuple[int, ...] = (2, 5, 8, 11)
    fpn_dim: int = 256
    # How to merge shallower and deeper feature maps in the top-down pass.
    # "add": element-wise addition (original FPN; merge_conv input = fpn_dim)
    # "concat": channel concat (merge_conv input = 2*fpn_dim; more expressive)
    fpn_merge: str = "add"


# ImageNet stats — DroneWasteSegmentation normalises with these; RADIO needs
# them undone (it ingests [0,1] and conditions internally).
_IN_MEAN = (0.485, 0.456, 0.406)
_IN_STD = (0.229, 0.224, 0.225)


class FPNHead(nn.Module):
    """Lateral + top-down FPN over same-resolution ViT block feature maps.

    Accepts K [B, D, G, G] feature maps (shallowest first), applies one
    1×1 lateral conv per level, then merges top-down (deepest → shallowest).
    Since all ViT blocks share the same spatial grid, no spatial upsampling
    is needed in the merge path.

    merge="add"    : element-wise add then 3×3 conv. Params ~2.56 M (K=4, D=768, C=21)
    merge="concat" : channel concat then 3×3 conv. Params ~4.26 M (merge_convs double)
    """
    def __init__(self, in_dim: int, fpn_dim: int, n_levels: int, num_logits: int,
                 merge: str = "add"):
        super().__init__()
        assert merge in ("add", "concat"), f"fpn_merge must be 'add' or 'concat', got {merge!r}"
        self.merge = merge
        self.laterals = nn.ModuleList(
            [nn.Conv2d(in_dim, fpn_dim, 1) for _ in range(n_levels)]
        )
        merge_in = 2 * fpn_dim if merge == "concat" else fpn_dim
        self.merge_convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(merge_in, fpn_dim, 3, padding=1), nn.GELU())
             for _ in range(n_levels - 1)]
        )
        self.classifier = nn.Conv2d(fpn_dim, num_logits, 1)

    def forward(self, block_feats: list[torch.Tensor]) -> torch.Tensor:
        lats = [lat(f) for lat, f in zip(self.laterals, block_feats)]
        p = lats[-1]
        for i in range(len(lats) - 2, -1, -1):
            merged = torch.cat([lats[i], p], dim=1) if self.merge == "concat" else lats[i] + p
            p = self.merge_convs[i](merged)
        return self.classifier(p)  # [B, num_logits, G, G]


class DinoSeg(nn.Module):
    def __init__(self, cfg: DinoSegConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.backbone_type == "dinov2":
            self.backbone = AutoModel.from_pretrained(
                cfg.backbone_id, torch_dtype=torch.float32
            )
            backbone_dim = self.backbone.config.hidden_size
            patch_size = self.backbone.config.patch_size
            if cfg.multi_block:
                backbone_dim = backbone_dim * len(cfg.multi_block)
        elif cfg.backbone_type == "dinov3":
            if "vitl_lvd" in cfg.backbone_id:
                from src.dinov3_backbone import load_dinov3_vitl16_lvd
                self.backbone = load_dinov3_vitl16_lvd(device="cpu")
                backbone_dim = 1024
            elif "vitl" in cfg.backbone_id:
                from src.dinov3_backbone import load_dinov3_vitl16
                self.backbone = load_dinov3_vitl16(device="cpu")
                backbone_dim = 1024
            else:
                from src.dinov3_backbone import load_dinov3_vitb16
                self.backbone = load_dinov3_vitb16(device="cpu")
                backbone_dim = 768
            patch_size = 16
            if cfg.multi_block:
                backbone_dim = backbone_dim * len(cfg.multi_block)
        elif cfg.backbone_type == "radio":
            self.backbone = AutoModel.from_pretrained(
                cfg.backbone_id, trust_remote_code=True
            )
            patch_size = 16
            self.register_buffer(
                "_in_mean", torch.tensor(_IN_MEAN).view(1, 3, 1, 1), persistent=False
            )
            self.register_buffer(
                "_in_std", torch.tensor(_IN_STD).view(1, 3, 1, 1), persistent=False
            )
            # Probe spatial feature dim — RADIO-B=768, RADIO-L=1024, RADIO-H=1280.
            # We can't rely on config.hidden_size because RADIO's HF config has
            # a deeply-nested `args` dict, not a top-level field. One tiny forward
            # is cheaper and version-proof.
            with torch.no_grad():
                _dummy = torch.zeros(1, 3, 224, 224)
                _summary, _spatial = self.backbone(_dummy)
                backbone_dim = int(_spatial.shape[-1])
            print(f"[radio] backbone_dim probed = {backbone_dim}")
        elif cfg.backbone_type == "frcnn":
            import torchvision
            det = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
            self.backbone = det.backbone  # BackboneWithFPN; frozen below
            backbone_dim = 256            # FPN outputs 256 channels at all levels
            patch_size = 16              # stride-16 → P4 at key "2"
        else:
            raise ValueError(f"unknown backbone_type {cfg.backbone_type!r}")

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        self.grid = cfg.image_size // patch_size
        self.image_size = cfg.image_size
        # Multi-label: one channel per fg class (no bg). Single-label: +1 for bg.
        self.num_logits = cfg.num_classes if cfg.multilabel else cfg.num_classes + 1

        if cfg.head == "linear":
            self.head = nn.Linear(backbone_dim, self.num_logits)
        elif cfg.head == "conv":
            self.head = nn.Sequential(
                nn.Conv2d(backbone_dim, cfg.head_hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(cfg.head_hidden, self.num_logits, kernel_size=1),
            )
        elif cfg.head == "fpn":
            if cfg.backbone_type == "frcnn":
                raise ValueError("FPN head is not compatible with frcnn backbone "
                                 "(frcnn already has a built-in FPN; use head=linear).")
            # For RADIO, register forward hooks on intermediate ViT blocks.
            if cfg.backbone_type == "radio":
                self._fpn_hook_feats: dict[int, torch.Tensor] = {}
                _blocks = None
                for attr_path in ["model.blocks", "radio_model.model.blocks", "blocks"]:
                    obj = self.backbone
                    try:
                        for part in attr_path.split("."):
                            obj = getattr(obj, part)
                        _blocks = obj
                        break
                    except AttributeError:
                        continue
                if _blocks is None:
                    raise AttributeError(
                        "Cannot locate ViT blocks in RADIO model for FPN head. "
                        "Use head='conv' or 'linear' instead."
                    )
                # Resolve num_skip: RADIO prepends num_skip non-patch tokens
                # (CLS + register) before patch tokens. Must slice correctly in
                # _features_multi — cannot assume a single CLS at position 0.
                _pg = None
                for _pg_path in ["radio_model.model.patch_generator", "model.patch_generator"]:
                    _obj = self.backbone
                    try:
                        for _part in _pg_path.split("."): _obj = getattr(_obj, _part)
                        _pg = _obj
                        break
                    except AttributeError:
                        continue
                self._radio_num_skip: int = _pg.num_skip if _pg is not None else 1
                print(f"[radio] FPN num_skip={self._radio_num_skip}")
                for _idx in cfg.fpn_blocks:
                    def _make_hook(i):
                        def _h(m, inp, out):
                            self._fpn_hook_feats[i] = out  # full seq; sliced in _features_multi
                        return _h
                    _blocks[_idx].register_forward_hook(_make_hook(_idx))
                print(f"[radio] FPN hooks registered on blocks {cfg.fpn_blocks}")
            self.head = FPNHead(
                in_dim=backbone_dim,
                fpn_dim=cfg.fpn_dim,
                n_levels=len(cfg.fpn_blocks),
                num_logits=self.num_logits,
                merge=cfg.fpn_merge,
            )
        else:
            raise ValueError(f"unknown head {cfg.head!r}")

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] -> [B, N, D] patch tokens.

        With cfg.multi_block set (DINOv2 / DINOv3 only), returns [B, N, D*K]
        with K patch-token blocks concatenated along the feature dim.
        """
        ctx = torch.no_grad() if self.cfg.freeze_backbone else torch.enable_grad()
        mb = self.cfg.multi_block
        with ctx:
            if self.cfg.backbone_type == "dinov2":
                if mb:
                    out = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
                    parts = [out.hidden_states[i + 1][:, 1:] for i in mb]
                    tokens = torch.cat(parts, dim=-1)
                else:
                    out = self.backbone(pixel_values=pixel_values)
                    tokens = out.last_hidden_state[:, 1:]  # drop CLS
            elif self.cfg.backbone_type == "dinov3":
                if mb:
                    parts = self.backbone.get_intermediate_layers(
                        pixel_values, n=list(mb),
                        reshape=False, return_class_token=False, norm=True,
                    )
                    tokens = torch.cat(list(parts), dim=-1)
                else:
                    out = self.backbone.forward_features(pixel_values)
                    tokens = out["x_norm_patchtokens"]
            elif self.cfg.backbone_type == "radio":
                if mb:
                    raise NotImplementedError("multi_block not supported for RADIO yet")
                x01 = pixel_values * self._in_std + self._in_mean
                _summary, tokens = self.backbone(x01)
            else:  # frcnn
                feats = self.backbone(pixel_values)  # OrderedDict keys 0-3,pool
                f = feats["2"]                       # stride-16 P4: [B, 256, G, G]
                B, C, Hf, Wf = f.shape
                tokens = f.flatten(2).transpose(1, 2)  # [B, G*G, 256]
        return tokens

    def _features_multi(self, pixel_values: torch.Tensor) -> list[torch.Tensor]:
        """[B,3,H,W] -> list of [B, D, G, G] feature maps for fpn_blocks.

        Returns K maps ordered shallowest→deepest (matching cfg.fpn_blocks order).
        Used only by the FPN head.
        """
        blocks = self.cfg.fpn_blocks
        ctx = torch.no_grad() if self.cfg.freeze_backbone else torch.enable_grad()
        with ctx:
            if self.cfg.backbone_type == "dinov2":
                out = self.backbone(pixel_values=pixel_values, output_hidden_states=True)
                maps = []
                for i in blocks:
                    t = out.hidden_states[i + 1][:, 1:]  # [B, N, D]
                    B, N, D = t.shape
                    maps.append(t.transpose(1, 2).reshape(B, D, self.grid, self.grid))
                return maps
            elif self.cfg.backbone_type == "dinov3":
                parts = self.backbone.get_intermediate_layers(
                    pixel_values, n=list(blocks),
                    reshape=False, return_class_token=False, norm=True,
                )
                maps = []
                for t in parts:
                    B, N, D = t.shape
                    maps.append(t.transpose(1, 2).reshape(B, D, self.grid, self.grid))
                return maps
            else:  # radio — populate hook feats via a forward pass
                x01 = pixel_values * self._in_std + self._in_mean
                self.backbone(x01)  # hooks fill self._fpn_hook_feats
                N_patch = self.grid * self.grid
                skip = self._radio_num_skip  # 8 for RADIOv2.5 (4 CLS + 4 register)
                maps = []
                for i in blocks:
                    t = self._fpn_hook_feats[i][:, skip:skip + N_patch]  # [B, N_patch, D]
                    B, N, D = t.shape
                    maps.append(t.transpose(1, 2).reshape(B, D, self.grid, self.grid))
                return maps

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if isinstance(self.head, FPNHead):
            block_feats = self._features_multi(pixel_values)  # list of [B,D,G,G]
            logits = self.head(block_feats)                   # [B, C, G, G]
        else:
            feats = self._features(pixel_values)  # [B, N, D]
            B, N, D = feats.shape
            if isinstance(self.head, nn.Linear):
                logits = self.head(feats)  # [B, N, C+1]
                logits = logits.transpose(1, 2).reshape(B, self.num_logits, self.grid, self.grid)
            else:
                feats_2d = feats.transpose(1, 2).reshape(B, D, self.grid, self.grid)
                logits = self.head(feats_2d)  # [B, C+1, grid, grid]
        logits = F.interpolate(
            logits, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )
        return logits  # [B, C+1, image_size, image_size]
