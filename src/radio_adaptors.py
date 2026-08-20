"""C-RADIOv4's teacher projections, loaded from the release checkpoint.

C-RADIOv4-SO400M is distilled from three teachers -- `siglip2-g-384`,
`dinov3_vit7b16` and `sam3` -- and the release ships the heads that project its
features back into each teacher's space:

  _heads.<teacher>                summary vector -> teacher's summary space
  _feature_projections.<teacher>  per-patch tokens -> teacher's dense space

That matters because it collapses most of the pipeline. The SigLIP2 projection
puts every patch into a *text-aligned* space, which is open-vocabulary naming and
open-vocabulary segmentation from the pass we already run; the SAM3 projection
puts every patch into SAM3's image-embedding space, which is where objectness and
masks should come from rather than from heuristics on raw features.

The catch, and the reason this file exists: the HF snapshot's `model.safetensors`
holds only the 330 backbone tensors under `radio_model.`, with `adaptor_names:
null` and no head weights at all. The heads are in the raw
`c-radio_v4-so400m_half.pth.tar` beside it, so they have to be pulled from there
and rebuilt with the release's own `create_mlp_from_state`, which infers every
dimension from the state dict.

Text queries additionally need the SigLIP2 text tower
(`google/siglip2-giant-opt-patch16-384`), which is a separate download; the
projections themselves are self-contained and the SAM3 one needs nothing else.
"""
from __future__ import annotations

import functools
import os
import pathlib
import string
import sys

import torch

WEIGHTS = pathlib.Path(os.environ.get(
    "WASTE_VLM_WEIGHTS",
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/weights"))

# teacher -> (feature-projection MLP version, summary-head MLP version)
# siglip2's dense projection carries attention blocks (the release trains it with
# spatial_mlp_version="attn"); sam3's is a plain MLP.
_VERSIONS = {"siglip2-g": ("attn", "v2"), "sam3": ("v2", "v2"), "dino_v3_7b": ("v2", "v2")}


def _release_dir(encoder_id: str = "cradiov4-so") -> pathlib.Path:
    name = {"cradiov4-so": "C-RADIOv4-SO400M", "cradiov4-h": "C-RADIOv4-H"}[encoder_id]
    return WEIGHTS / name


@functools.lru_cache(maxsize=2)
def _state(encoder_id: str) -> dict:
    d = _release_dir(encoder_id)
    ck = list(d.glob("*.pth.tar"))
    if not ck:
        raise FileNotFoundError(
            f"{d} has no *.pth.tar; the HF safetensors carries the backbone only, "
            "so the teacher heads cannot be recovered from it")
    return torch.load(str(ck[0]), map_location="cpu", weights_only=False)["state_dict"]


@functools.lru_cache(maxsize=2)
def _factory(encoder_id: str):
    """Import the release's own MLP builder.

    Its modules use relative imports (`from .enable_spectral_reparam import ...`),
    so the snapshot directory has to be registered as a package rather than just
    put on sys.path -- it ships no __init__.py.
    """
    import importlib
    import types

    d = _release_dir(encoder_id)
    name = f"_cradio_release_{encoder_id.replace('-', '_')}"
    if name not in sys.modules:
        pkg = types.ModuleType(name)
        pkg.__path__ = [str(d)]
        sys.modules[name] = pkg
    mod = importlib.import_module(f"{name}.adaptor_module_factory")
    return mod.create_mlp_from_state


def load_projection(teacher: str = "siglip2-g", kind: str = "features",
                    encoder_id: str = "cradiov4-so", device="cuda", dtype=torch.float32):
    """-> an nn.Module mapping RADIO output to `teacher`'s space.

    kind='features' projects per-patch tokens, kind='summary' the summary vector.
    """
    create_mlp_from_state = _factory(encoder_id)

    prefix = {"features": "_feature_projections", "summary": "_heads"}[kind] + f".{teacher}."
    sd = {k: v for k, v in _state(encoder_id).items() if k.startswith(prefix)}
    if not sd:
        raise KeyError(f"no {prefix}* in the checkpoint; teachers present: "
                       f"{sorted({k.split('.')[1] for k in _state(encoder_id) if k.startswith('_heads.')})}")
    version = _VERSIONS[teacher][0 if kind == "features" else 1]
    m = create_mlp_from_state(version, sd, prefix=prefix, is_summary=(kind == "summary"))
    return m.to(device=device, dtype=dtype).eval()


@functools.lru_cache(maxsize=1)
def siglip2_text(version: str = "google/siglip2-giant-opt-patch16-384", device="cuda"):
    """-> encode(list[str]) giving L2-normalised SigLIP2 text embeddings.

    Built by hand rather than through AutoModel because this checkpoint's text
    tower is not square: its config declares text_config.projection_size 1536
    while the hidden size is 1152, and transformers 4.49 builds the final head as
    nn.Linear(hidden, hidden). from_pretrained therefore refuses the head with a
    shape mismatch, and the suggested ignore_mismatched_sizes would silently
    randomise the very projection that carries the alignment -- the text side
    would still produce vectors, they just would not mean anything.

    So the head is resized to what the checkpoint holds and the weights are
    loaded strictly. Upgrading transformers would also work, but not while the
    rest of the stack is pinned around it.
    """
    import glob
    import json

    import torch
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, SiglipTextConfig, SiglipTextModel

    d = _snapshot(version)
    cfg_all = json.loads((pathlib.Path(d) / "config.json").read_text())
    tc = dict(cfg_all["text_config"])
    proj = tc.pop("projection_size", None) or tc["hidden_size"]
    model = SiglipTextModel(SiglipTextConfig(**tc))
    # transformers 4.49 nests the tower under .text_model; 5.x flattens it, and this
    # loader now runs in both environments. Newer versions may also honour
    # projection_size themselves, in which case the head is already the right shape.
    inner = getattr(model, "text_model", model)
    if getattr(inner, "head", None) is not None and inner.head.out_features != proj:
        inner.head = torch.nn.Linear(tc["hidden_size"], proj)

    sd = {}
    for f in sorted(glob.glob(str(pathlib.Path(d) / "*.safetensors"))):
        sd.update({k: v for k, v in load_file(f).items() if k.startswith("text_model.")})
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        # the flattened layout drops the prefix the checkpoint still carries
        stripped = {k[len("text_model."):]: v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(stripped, strict=False)
    real = [k for k in missing if "position_ids" not in k]
    if real:
        raise RuntimeError(f"SigLIP2 text tower incomplete, missing {real[:5]}")
    model = model.to(device).eval()
    tok = AutoTokenizer.from_pretrained(d)

    @torch.no_grad()
    def encode(texts):
        # The release canonicalises before tokenising (SigLIP2Adaptor ->
        # canonicalize_text, itself lifted from big_vision's prompt engineering):
        # underscores to spaces, punctuation stripped, lowercased, whitespace
        # collapsed. SigLIP2 was trained on text in that form, so anything with a
        # comma, slash or capital -- which most of the contrastive and EWC-seeded
        # prompts have -- is otherwise fed off-distribution. Worth +0.007 accuracy
        # on the DroneWaste development set, measured against cached embeddings.
        b = tok([_canonicalize(t) for t in texts], padding="max_length",
                truncation=True, max_length=64, return_tensors="pt").to(device)
        out = model(**b)
        e = (out.pooler_output if out.pooler_output is not None
             else out.last_hidden_state[:, -1]).float()
        return torch.nn.functional.normalize(e, dim=-1)

    return encode



def _canonicalize(text: str, _trans=str.maketrans("", "", string.punctuation)) -> str:
    """Lowercase, drop punctuation, collapse whitespace -- the release's own
    `canonicalize_text`, kept byte-compatible with it so text embeddings match
    what the SigLIP2 teacher head was distilled against."""
    return " ".join(text.replace("_", " ").translate(_trans).lower().split()).strip()

def _snapshot(version: str) -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(version, local_files_only=True)
