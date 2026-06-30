"""DINOv2 + DETR decoder for DroneWaste object detection.

- Frozen DINOv2-B backbone (518×518 → 1369 patch tokens + CLS, 768-d).
- Linear projection 768 → 256 (matches DETR hidden_dim).
- 2-D sinusoidal positional embeddings added to backbone tokens (DINOv2
  features already carry position info, but DETR's decoder benefits from
  explicit pos info on keys — same trick as Carion 2020 Eq. 5).
- 6-layer standard transformer decoder with 100 learnable object queries.
- Linear class head (num_classes + 1 for "no object") and 3-layer MLP box
  head outputting cxcywh in [0, 1].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def sinusoidal_2d_pos_embed(h: int, w: int, dim: int) -> torch.Tensor:
    """2-D sinusoidal positional embedding, [h*w, dim]."""
    assert dim % 4 == 0, "dim must be divisible by 4 for 2-D sin/cos"
    half = dim // 2
    omega = torch.arange(half // 2, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (half // 2)))
    y = torch.arange(h, dtype=torch.float32)[:, None] * omega[None]
    x = torch.arange(w, dtype=torch.float32)[:, None] * omega[None]
    pe_y = torch.cat([torch.sin(y), torch.cos(y)], dim=1)  # [h, half]
    pe_x = torch.cat([torch.sin(x), torch.cos(x)], dim=1)  # [w, half]
    pe = torch.cat(
        [pe_y[:, None].expand(h, w, -1), pe_x[None].expand(h, w, -1)], dim=2
    )
    return pe.reshape(h * w, dim)


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden] * (n_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(n_layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


@dataclass
class DinoDETRConfig:
    backbone_id: str = "facebook/dinov2-base"
    image_size: int = 518
    hidden_dim: int = 256
    num_queries: int = 100
    num_decoder_layers: int = 6
    nhead: int = 8
    dim_feedforward: int = 1024
    dropout: float = 0.1
    num_classes: int = 20  # class indices are 0..num_classes-1; class num_classes = "no object"
    freeze_backbone: bool = True


class DinoDETR(nn.Module):
    def __init__(self, cfg: DinoDETRConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone = AutoModel.from_pretrained(cfg.backbone_id, torch_dtype=torch.float32)
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        backbone_dim = self.backbone.config.hidden_size  # 768 for dinov2-base
        patch_size = self.backbone.config.patch_size  # 14 for dinov2-base
        self.grid = cfg.image_size // patch_size
        self.num_tokens = self.grid * self.grid

        self.proj = nn.Linear(backbone_dim, cfg.hidden_dim)

        pe = sinusoidal_2d_pos_embed(self.grid, self.grid, cfg.hidden_dim)
        self.register_buffer("pos_embed", pe)  # [num_tokens, hidden_dim]

        self.query_embed = nn.Embedding(cfg.num_queries, cfg.hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=False,
        )
        decoder_norm = nn.LayerNorm(cfg.hidden_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.num_decoder_layers, norm=decoder_norm)

        self.class_head = nn.Linear(cfg.hidden_dim, cfg.num_classes + 1)
        self.box_head = MLP(cfg.hidden_dim, cfg.hidden_dim, 4, n_layers=3)

    def _features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Returns [B, num_tokens, hidden_dim] backbone+proj features."""
        ctx = torch.no_grad() if self.cfg.freeze_backbone else torch.enable_grad()
        with ctx:
            out = self.backbone(pixel_values=pixel_values)
            tokens = out.last_hidden_state[:, 1:]  # drop CLS
        tokens = self.proj(tokens)
        tokens = tokens + self.pos_embed.unsqueeze(0)
        return tokens

    def forward(self, pixel_values: torch.Tensor) -> dict:
        memory = self._features(pixel_values)  # [B, N, D]
        B = memory.size(0)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]
        dec = self.decoder(queries, memory)  # [B, Q, D]
        logits = self.class_head(dec)  # [B, Q, num_classes+1]
        boxes = self.box_head(dec).sigmoid()  # [B, Q, 4] in (cx, cy, w, h) ∈ [0,1]
        return {"logits": logits, "pred_boxes": boxes}
