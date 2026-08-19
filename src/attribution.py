"""Attribution of the Yes/No margin back onto the visual token grid.

The margin is a single scalar from a single prefill, and the projected visual
tokens are an ordinary tensor spliced into the prompt, so the whole path from
those tokens to the decision is differentiable. That makes "where did it look"
answerable directly rather than by proxy: no attention weights, which are a poor
attribution signal and hidden behind fused kernels anyway.

Three estimators, deliberately overlapping so they can check each other:

  grad  gradient x input on the projected tokens. One backward pass, but a local
        linearisation of a very non-linear function.
  ig    integrated gradients from a baseline of mean visual tokens along a
        straight path. Costs `steps` passes and satisfies completeness, so the
        cell scores sum to the margin difference the baseline actually produces.
  occ   replace a block of tokens with the mean token and measure the drop. Slow,
        assumption-free, and in the same units as the occlusion experiment that
        established the model is grounded at all -- so it is the reference the
        cheap estimators are calibrated against.
"""
from __future__ import annotations

import numpy as np
import torch

from src.vlm_model import IMAGE_TOKEN_INDEX


def _prompt_ids(model, question: str) -> list[int]:
    """Exactly the framing score_margin uses; a different one measures a different model."""
    def tok(t: str) -> list[int]:
        return model.tokenizer(t, add_special_tokens=False).input_ids

    pre, post = ("<image>\n" + question).split("<image>", 1)
    ids = tok(f"<|im_start|>system\n{model.system_prompt}<|im_end|>\n")
    ids += tok(f"<|im_start|>user\n{pre}") + [IMAGE_TOKEN_INDEX] + tok(f"{post}<|im_end|>\n")
    return ids + tok("<|im_start|>assistant\n")


def _margin_from_embeds(model, vis, ids, yes_ids, no_ids):
    """Run the prefill with `vis` spliced in, return the differentiable margin."""
    device = model.llm.device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    p = ids.index(IMAGE_TOKEN_INDEX)
    pre = model.llm.get_input_embeddings()(input_ids[:, :p])
    post = model.llm.get_input_embeddings()(input_ids[:, p + 1:])
    emb = torch.cat([pre, vis.unsqueeze(0).to(pre.dtype), post], dim=1)
    attn = torch.ones(1, emb.shape[1], dtype=torch.long, device=device)
    logits = model.llm(inputs_embeds=emb, attention_mask=attn).logits[0, -1]
    lp = torch.log_softmax(logits.float(), dim=-1)
    return torch.logsumexp(lp[yes_ids], 0) - torch.logsumexp(lp[no_ids], 0)


def token_grid(model) -> int:
    g = model.encoder.image_size // model.encoder.patch_size
    return g // model.pixel_shuffle


def attribute(model, pixel_values, question: str, yes_ids, no_ids,
              method: str = "ig", steps: int = 32, block: int = 2):
    """-> (map [g, g] float32, margin float). Positive = pushed the answer to Yes."""
    ids = _prompt_ids(model, question)
    g = token_grid(model)

    with torch.no_grad():
        vis0 = model.encode_images(pixel_values)[0]          # [N, hidden]
    N = vis0.shape[0]
    if N != g * g:
        raise ValueError(f"{N} visual tokens is not a {g}x{g} grid")

    if method == "occ":
        with torch.no_grad():
            base = float(_margin_from_embeds(model, vis0, ids, yes_ids, no_ids))
            fill = vis0.mean(0, keepdim=True)
            out = np.zeros((g, g), dtype=np.float32)
            for y0 in range(0, g, block):
                for x0 in range(0, g, block):
                    v = vis0.clone().reshape(g, g, -1)
                    v[y0:y0 + block, x0:x0 + block] = fill
                    m = float(_margin_from_embeds(model, v.reshape(N, -1),
                                                  ids, yes_ids, no_ids))
                    # drop when removed = evidence it was contributing
                    out[y0:y0 + block, x0:x0 + block] = base - m
        return out, base

    if method == "grad":
        vis = vis0.clone().requires_grad_(True)
        m = _margin_from_embeds(model, vis, ids, yes_ids, no_ids)
        (grad,) = torch.autograd.grad(m, vis)
        a = (grad.float() * vis.detach().float()).sum(-1)
        return a.reshape(g, g).cpu().numpy(), float(m)

    if method == "ig":
        baseline = vis0.mean(0, keepdim=True).expand_as(vis0).contiguous()
        total = torch.zeros_like(vis0, dtype=torch.float32)
        for k in range(steps):
            alpha = (k + 0.5) / steps
            vis = (baseline + alpha * (vis0 - baseline)).detach().requires_grad_(True)
            m = _margin_from_embeds(model, vis, ids, yes_ids, no_ids)
            (grad,) = torch.autograd.grad(m, vis)
            total += grad.float()
        a = (total / steps * (vis0 - baseline).float()).sum(-1)
        with torch.no_grad():
            base = float(_margin_from_embeds(model, vis0, ids, yes_ids, no_ids))
        return a.reshape(g, g).cpu().numpy(), base

    raise ValueError(method)


# ------------------------------------------------------------------ validation
def box_mask(boxes, size, g: int) -> np.ndarray:
    """Ground-truth boxes (xywh, original pixels) rasterised onto the token grid."""
    W, H = size
    m = np.zeros((g, g), dtype=bool)
    for x, y, w, h in boxes:
        x0 = max(0, min(g - 1, int(np.floor(x / W * g))))
        y0 = max(0, min(g - 1, int(np.floor(y / H * g))))
        x1 = max(x0 + 1, min(g, int(np.ceil((x + w) / W * g))))
        y1 = max(y0 + 1, min(g, int(np.ceil((y + h) / H * g))))
        m[y0:y1, x0:x1] = True
    return m


def score_map(a: np.ndarray, mask: np.ndarray) -> dict:
    """How much of the evidence lands on the waste, against the null that says
    a map ignorant of the image scores the box's own area fraction."""
    pos = np.clip(a, 0, None)
    area = float(mask.mean())
    mass = float(pos[mask].sum() / pos.sum()) if pos.sum() > 0 else float("nan")
    flat = a.reshape(-1)
    top = int(np.argmax(flat))
    return {
        "box_area_fraction": area,          # what a uniform map would score
        "mass_in_box": mass,
        "mass_lift": mass - area,           # the number that means something
        "hit": bool(mask.reshape(-1)[top]),  # pointing game
    }
