"""Load the DINOv3 ViT-B/16 backbone in a torch-2.3 environment.

The official torch.hub DINOv3 code imports fine once a small `torch.amp`
shim is applied, but its pretrained weights are license-gated on a CDN
that 403s. Instead we pull the HF-format checkpoint
(`facebook/dinov3-vitb16-pretrain-lvd1689m`, gated — needs an accepted
licence + HF token) and remap its keys onto the hub `DinoVisionTransformer`.

Key differences HF -> hub:
    embeddings.cls_token            -> cls_token
    embeddings.register_tokens      -> storage_tokens
    embeddings.patch_embeddings.*   -> patch_embed.proj.*
    layer.N.*                       -> blocks.N.*
      attention.{q,k,v}_proj        -> attn.qkv          (fused, concat dim 0)
      attention.o_proj              -> attn.proj
      layer_scale{1,2}.lambda1      -> ls{1,2}.gamma
      mlp.up_proj / mlp.down_proj   -> mlp.fc1 / mlp.fc2
HF has no k-proj bias (mask_k_bias=True) -> k slice of qkv.bias = zeros.
Buffers `rope_embed.periods` and `attn.qkv.bias_mask` are config-derived;
we load strict=False and keep the model's initialised values.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

DINOV3_REPO = "/home/ids/diecidue/.cache/torch/hub/facebookresearch_dinov3_main"
HF_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"
# Full local snapshots of the gated HF repos — no HF round-trip on load.
LOCAL_REPO_DIR = Path(
    "/home/ids/diecidue/results/waste_vlm/weights/dinov3-vitb16-pretrain-lvd1689m"
)
LOCAL_WEIGHTS = LOCAL_REPO_DIR / "model.safetensors"

LOCAL_REPO_DIR_L = Path(
    "/home/ids/diecidue/results/waste_vlm/weights/dinov3-vitl16-pretrain-sat493m"
)
LOCAL_WEIGHTS_L = LOCAL_REPO_DIR_L / "model.safetensors"

LOCAL_REPO_DIR_L_LVD = Path(
    "/home/ids/diecidue/results/waste_vlm/weights/dinov3-vitl16-pretrain-lvd1689m"
)
LOCAL_WEIGHTS_L_LVD = LOCAL_REPO_DIR_L_LVD / "model.safetensors"


def _apply_torch_amp_shim() -> None:
    """torch 2.3 keeps custom_fwd/bwd under torch.cuda.amp, not torch.amp."""
    import torch.cuda.amp as _camp
    if not hasattr(torch.amp, "custom_fwd"):
        torch.amp.custom_fwd = _camp.custom_fwd
        torch.amp.custom_bwd = _camp.custom_bwd


def _remap_hf_state_dict(hf_sd: dict, n_layers: int = 12) -> dict:
    out: dict[str, torch.Tensor] = {}
    out["cls_token"] = hf_sd["embeddings.cls_token"]
    out["storage_tokens"] = hf_sd["embeddings.register_tokens"]
    out["mask_token"] = hf_sd["embeddings.mask_token"].reshape(1, -1)
    out["patch_embed.proj.weight"] = hf_sd["embeddings.patch_embeddings.weight"]
    out["patch_embed.proj.bias"] = hf_sd["embeddings.patch_embeddings.bias"]
    out["norm.weight"] = hf_sd["norm.weight"]
    out["norm.bias"] = hf_sd["norm.bias"]
    for n in range(n_layers):
        h, b = f"layer.{n}.", f"blocks.{n}."
        out[b + "norm1.weight"] = hf_sd[h + "norm1.weight"]
        out[b + "norm1.bias"] = hf_sd[h + "norm1.bias"]
        out[b + "norm2.weight"] = hf_sd[h + "norm2.weight"]
        out[b + "norm2.bias"] = hf_sd[h + "norm2.bias"]
        # fused qkv; HF has no k bias (mask_k_bias)
        q, k, v = (hf_sd[h + f"attention.{x}_proj.weight"] for x in "qkv")
        out[b + "attn.qkv.weight"] = torch.cat([q, k, v], dim=0)
        qb = hf_sd[h + "attention.q_proj.bias"]
        vb = hf_sd[h + "attention.v_proj.bias"]
        out[b + "attn.qkv.bias"] = torch.cat([qb, torch.zeros_like(qb), vb], dim=0)
        out[b + "attn.proj.weight"] = hf_sd[h + "attention.o_proj.weight"]
        out[b + "attn.proj.bias"] = hf_sd[h + "attention.o_proj.bias"]
        out[b + "ls1.gamma"] = hf_sd[h + "layer_scale1.lambda1"]
        out[b + "ls2.gamma"] = hf_sd[h + "layer_scale2.lambda1"]
        out[b + "mlp.fc1.weight"] = hf_sd[h + "mlp.up_proj.weight"]
        out[b + "mlp.fc1.bias"] = hf_sd[h + "mlp.up_proj.bias"]
        out[b + "mlp.fc2.weight"] = hf_sd[h + "mlp.down_proj.weight"]
        out[b + "mlp.fc2.bias"] = hf_sd[h + "mlp.down_proj.bias"]
    return out


def _resolve_checkpoint() -> str:
    """Local snapshot if present; otherwise fetch the gated HF repo once."""
    if LOCAL_WEIGHTS.is_file():
        return str(LOCAL_WEIGHTS)
    import os
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    snapshot_download(HF_REPO, local_dir=str(LOCAL_REPO_DIR), token=token)
    return str(LOCAL_WEIGHTS)


def load_dinov3_vitb16(device: str = "cpu") -> torch.nn.Module:
    """Return a ready-to-eval DINOv3 ViT-B/16 backbone with pretrained weights."""
    from safetensors.torch import load_file

    _apply_torch_amp_shim()
    if DINOV3_REPO not in sys.path:
        sys.path.insert(0, DINOV3_REPO)
    from dinov3.hub.backbones import dinov3_vitb16  # type: ignore

    model = dinov3_vitb16(pretrained=False).eval()
    remapped = _remap_hf_state_dict(load_file(_resolve_checkpoint()))
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # Only config-derived buffers may be missing.
    allowed_missing = {"rope_embed.periods"} | {
        f"blocks.{n}.attn.qkv.bias_mask" for n in range(12)
    }
    bad = set(missing) - allowed_missing
    if bad or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={bad} unexpected={unexpected}")
    return model.to(device)


def load_dinov3_vitl16(device: str = "cpu") -> torch.nn.Module:
    """Return a ready-to-eval DINOv3 ViT-L/16 backbone (SAT493M weights)."""
    from safetensors.torch import load_file

    _apply_torch_amp_shim()
    if DINOV3_REPO not in sys.path:
        sys.path.insert(0, DINOV3_REPO)
    from dinov3.hub.backbones import dinov3_vitl16, Weights  # type: ignore

    model = dinov3_vitl16(pretrained=False, weights=Weights.SAT493M).eval()
    remapped = _remap_hf_state_dict(load_file(str(LOCAL_WEIGHTS_L)), n_layers=24)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    # local_cls_norm is only in the hub model (untie_global_and_local_cls_norm=True);
    # the HF checkpoint doesn't include it — keep hub-initialised values.
    allowed_missing = {
        "rope_embed.periods",
        "local_cls_norm.weight",
        "local_cls_norm.bias",
    } | {f"blocks.{n}.attn.qkv.bias_mask" for n in range(24)}
    bad = set(missing) - allowed_missing
    if bad or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={bad} unexpected={unexpected}")
    return model.to(device)


def load_dinov3_vitl16_lvd(device: str = "cpu") -> torch.nn.Module:
    """Return a ready-to-eval DINOv3 ViT-L/16 backbone (LVD1689M weights)."""
    from safetensors.torch import load_file

    _apply_torch_amp_shim()
    if DINOV3_REPO not in sys.path:
        sys.path.insert(0, DINOV3_REPO)
    from dinov3.hub.backbones import dinov3_vitl16, Weights  # type: ignore

    model = dinov3_vitl16(pretrained=False, weights=Weights.LVD1689M).eval()
    remapped = _remap_hf_state_dict(load_file(str(LOCAL_WEIGHTS_L_LVD)), n_layers=24)
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    allowed_missing = {
        "rope_embed.periods",
        "local_cls_norm.weight",
        "local_cls_norm.bias",
    } | {f"blocks.{n}.attn.qkv.bias_mask" for n in range(24)}
    bad = set(missing) - allowed_missing
    if bad or unexpected:
        raise RuntimeError(f"state_dict mismatch: missing={bad} unexpected={unexpected}")
    return model.to(device)


@torch.no_grad()
def patch_tokens(model: torch.nn.Module, pixel_values: torch.Tensor) -> torch.Tensor:
    """[B,3,H,W] -> [B, P, D] normed patch tokens (CLS + storage stripped)."""
    out = model.forward_features(pixel_values)
    return out["x_norm_patchtokens"]
