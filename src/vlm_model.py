"""LLaVA-style multimodal assembly: DINO/RADIO encoder -> projector -> Qwen2.5-7B.

This is the Waste-VLM "visual stage": the CLIP encoder of an off-the-shelf VLM
is replaced by the frozen DINOv3 / RADIO backbone chosen by the DINO/RADIO track
(RADIOv2.5-L @ 512² is the headline pick; DINOv3-B is the close runner-up; cf.
`RESULTS_SNAPSHOT.md` §1/§4). The encoder is frozen, a 2-layer MLP projector
maps its *patch* tokens into Qwen's embedding space, and Qwen2.5-7B-Instruct is
adapted with LoRA.

Image tokens are spliced into the text sequence with LLaVA's sentinel trick: a
single `IMAGE_TOKEN_INDEX` (-200) marker in `input_ids` is replaced, at forward
time, by the N projected patch embeddings. The marker is never embedded
directly (it is not a real vocab id), so no embedding resize is needed.

Smoke-test (needs one GPU; loads the 7B LLM):
    python -m src.vlm_model --smoke --encoder radio-l
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.vision_encoder import VisionEncoder

# LLaVA sentinel: a non-vocab id used to mark where visual tokens are spliced in.
IMAGE_TOKEN_INDEX = -200

WEIGHTS_ROOT = Path(
    os.environ.get(
        "WASTE_VLM_WEIGHTS",
        "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/weights",
    )
)
DEFAULT_LLM_PATH = str(WEIGHTS_ROOT / "Qwen2.5-7B-Instruct")

# The learned-mask structural-pruning toolkit (custom Qwen2/Llama3 arch + mask
# wrapper) lives in a sibling repo. Only imported when building a *pruned*
# decoder (joint prune+finetune); a no-op otherwise.
PRUNING_REPO = os.environ.get(
    "PRUNING_REPO", "/leonardo/home/userexternal/adiecidu/scripts/pruning"
)


def _import_pruning():
    """Add the pruning repo to sys.path and return its mask/model primitives.

    Kept lazy so the ordinary (stock-decoder) VLM path never imports the
    pruning stack. Returns a namespace-like dict of the names the masked-decoder
    build + training loop need."""
    import sys
    if PRUNING_REPO not in sys.path:
        sys.path.insert(0, PRUNING_REPO)
    from language_utils import create_qwen2_and_tokenizer  # registers custom-qwen2
    from custom_attentions.qwen2_attention import Qwen2MaskAdapter
    from mask_wrapper import wrap_model_with_masks, MaskedTransformerLayer
    return {
        "create_qwen2_and_tokenizer": create_qwen2_and_tokenizer,
        "Qwen2MaskAdapter": Qwen2MaskAdapter,
        "wrap_model_with_masks": wrap_model_with_masks,
        "MaskedTransformerLayer": MaskedTransformerLayer,
    }


def _is_custom_qwen2(llm_path: str) -> bool:
    """True if `llm_path` holds a materialized pruned decoder (model_type
    custom-qwen2). Such a checkpoint has per-layer flexible GQA/MLP dims that
    stock AutoModel can't rebuild — it must go through the pruning toolkit's
    custom Qwen2ForCausalLM. Cheap config.json peek, no model load."""
    import json, os
    cfg = os.path.join(llm_path, "config.json")
    if not os.path.exists(cfg):
        return False
    try:
        with open(cfg) as f:
            return json.load(f).get("model_type") == "custom-qwen2"
    except Exception:
        return False


DEFAULT_SYSTEM_PROMPT = (
    "You are a remote-sensing assistant that analyzes aerial and drone imagery "
    "to detect and describe illegal waste, dumping sites, and related land cover."
)


class Projector(nn.Module):
    """2-layer MLP from the encoder patch dim to the LLM hidden dim (LLaVA-1.5
    style: Linear -> GELU -> Linear, output width = LLM hidden).

    With ``pixel_shuffle > 1`` an s x s neighbourhood of patch tokens is folded
    into the channel dim before the MLP (LLaVA-NeXT / InternVL style): the square
    patch grid [B, (h*w), D] becomes [B, (h/s * w/s), D*s^2], cutting the token
    count fed to the LLM by s^2 while keeping the high-res signal. ``pixel_shuffle
    == 1`` (default) is the original 2-layer MLP, so old checkpoints load 1:1.
    """

    def __init__(self, in_dim: int, out_dim: int, pixel_shuffle: int = 1) -> None:
        super().__init__()
        self.pixel_shuffle = pixel_shuffle
        eff_in = in_dim * (pixel_shuffle ** 2)
        self.fc1 = nn.Linear(eff_in, out_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)

    def _shuffle(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D] with N a perfect square (h == w == sqrt(N)).
        B, N, D = x.shape
        s = self.pixel_shuffle
        h = w = int(round(N ** 0.5))
        if h * w != N:
            raise ValueError(f"pixel_shuffle needs a square patch grid, got N={N}")
        if h % s or w % s:
            raise ValueError(f"grid {h}x{w} not divisible by pixel_shuffle={s}")
        x = x.view(B, h // s, s, w // s, s, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(B, (h // s) * (w // s), D * s * s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pixel_shuffle > 1:
            x = self._shuffle(x)
        return self.fc2(self.act(self.fc1(x)))


class WasteVLM(nn.Module):
    """Frozen DINO/RADIO encoder + trainable projector + Qwen2.5-7B (LoRA-ready).

    The encoder runs in fp32 (frozen); the projector and LLM run in `dtype`
    (bf16 by default). Only the projector and any LoRA adapters on the LLM are
    trainable; the backbone and the LLM base weights stay frozen.
    """

    def __init__(
        self,
        llm_path: str = DEFAULT_LLM_PATH,
        encoder_id: str = "radio-l",
        image_size: int = 512,
        pixel_shuffle: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        attn_implementation: str = "sdpa",
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        prune: bool = False,
        prune_tau_start: float = 5.0,
        prune_logit_init: float = 2.0,
        prune_enable_layer_drop: bool = False,
        prune_blocks: str = "both",
    ) -> None:
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device_str = device
        self.dtype = dtype
        self.system_prompt = system_prompt
        self.is_pruning = prune
        self.mask_params: list = []
        self.mask_adapter = None

        # --- frozen vision encoder (fp32) ---
        self.encoder = VisionEncoder(encoder_id, device=device, image_size=image_size)
        self.encoder_id = encoder_id
        self.image_size = image_size
        patch_dim = self.encoder.patch_dim

        # --- LLM ---
        if prune:
            # Joint prune+finetune: build the decoder as the custom masked Qwen-2
            # (per-layer flexible GQA + learnable structural masks). The mask
            # search is driven by the VLM task loss during stage-2/2.5 training;
            # the trainer adds the sparsity penalty + controllers + materializes.
            P = _import_pruning()
            self.llm, self.tokenizer = P["create_qwen2_and_tokenizer"](
                llm_path, llm_path, cache_dir=None,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.mask_adapter = P["Qwen2MaskAdapter"]()
            # Wrap on CPU (adds mask params + tau buffers), THEN move the whole
            # module — masks included — to the device, so DDP sees a uniform
            # device (mirrors the pruning trainer's build order).
            self.mask_params = P["wrap_model_with_masks"](
                self.llm, self.mask_adapter,
                tau_init=prune_tau_start, logit_init=prune_logit_init,
                gate_type="sigmoid", enable_layer_drop=prune_enable_layer_drop,
                prune_blocks=prune_blocks,
            )
            self._MaskedTransformerLayer = P["MaskedTransformerLayer"]
            self.llm = self.llm.to(device=device, dtype=dtype)
        elif _is_custom_qwen2(llm_path):
            # Inference on a materialized pruned decoder (per-layer flexible GQA,
            # model_type custom-qwen2). Stock AutoModel can't build it; load via
            # the pruning toolkit's custom Qwen2ForCausalLM, which reads the
            # per-layer mlp/qkv config and loads the pruned weights 1:1 — but do
            # NOT wrap masks: the structure is already baked in, this is the final
            # model. `create_qwen2_and_tokenizer` also registers the arch.
            P = _import_pruning()
            self.llm, self.tokenizer = P["create_qwen2_and_tokenizer"](
                llm_path, llm_path, cache_dir=None,
            )
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm = self.llm.to(device=device, dtype=dtype)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.llm = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=dtype,
                attn_implementation=attn_implementation,
            ).to(device)
        hidden = self.llm.config.hidden_size

        # Qwen turn terminator (<|im_end|>); fall back to eos.
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.turn_end_id = im_end if im_end is not None and im_end >= 0 else self.tokenizer.eos_token_id

        # --- projector (trainable) ---
        self.pixel_shuffle = pixel_shuffle
        self.projector = Projector(
            patch_dim, hidden, pixel_shuffle=pixel_shuffle,
        ).to(device=device, dtype=dtype)
        self.patch_dim = patch_dim
        self.hidden = hidden

    # ------------------------------------------------------------------
    # PEFT / training plumbing
    # ------------------------------------------------------------------
    def apply_lora(self, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                   target_modules: Optional[list[str]] = None) -> None:
        """Wrap the LLM with a LoRA adapter (attention + MLP projections)."""
        from peft import LoraConfig, get_peft_model

        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]
        cfg = LoraConfig(
            r=r, lora_alpha=alpha, lora_dropout=dropout,
            target_modules=target_modules, bias="none", task_type="CAUSAL_LM",
        )
        self.llm = get_peft_model(self.llm, cfg)
        # get_peft_model freezes every non-LoRA param, including the structural
        # mask logits. Re-enable them so the joint prune+finetune keeps learning
        # the masks alongside LoRA.
        if self.is_pruning:
            self.unfreeze_masks()

    def unfreeze_masks(self) -> None:
        """Mark the structural mask logits trainable (idempotent)."""
        for p in self.mask_params:
            p.requires_grad_(True)

    def mask_sparsity_terms(self):
        """(l1_probability_sum, total_prunable_params) summed over masked layers.

        Used by the trainer to form the sparsity penalty and to read achieved
        sparsity for the PI controller. Only valid when built with prune=True."""
        l1 = 0.0
        total = 0
        for m in self.llm.modules():
            if isinstance(m, self._MaskedTransformerLayer):
                l1 = l1 + m.l1_probability_sum()
                total += m.total_params()
        return l1, max(1, total)

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.llm.gradient_checkpointing_enable(**kwargs)
        # inputs_embeds path: make the embedding output require grad so the
        # checkpointed LLM backbone backprops into the projector.
        if hasattr(self.llm, "enable_input_require_grads"):
            self.llm.enable_input_require_grads()

    def freeze_llm(self) -> None:
        """Freeze every LLM parameter (connector/alignment stage: projector-only)."""
        for p in self.llm.parameters():
            p.requires_grad_(False)

    def freeze_for_training(self) -> None:
        """Encoder frozen; projector trainable; LLM trainable only where LoRA
        already marked params (apply_lora handles that). In the projector-only
        stage call freeze_llm() instead of apply_lora()."""
        for p in self.encoder.model.parameters():
            p.requires_grad_(False)
        for p in self.projector.parameters():
            p.requires_grad_(True)

    def trainable_parameter_groups(self, projector_lr: float, lora_lr: float,
                                   mask_lr: float = 0.0):
        mask_ids = {id(p) for p in self.mask_params}
        proj_params = [p for p in self.projector.parameters() if p.requires_grad]
        # LoRA group = trainable LLM params minus the structural mask logits
        # (those go in their own group at mask_lr).
        lora_params = [p for _, p in self.llm.named_parameters()
                       if p.requires_grad and id(p) not in mask_ids]
        groups = []
        if proj_params:
            groups.append({"params": proj_params, "lr": projector_lr})
        if lora_params:
            groups.append({"params": lora_params, "lr": lora_lr})
        if self.is_pruning and mask_lr > 0:
            mp = [p for p in self.mask_params if p.requires_grad]
            if mp:
                groups.append({"params": mp, "lr": mask_lr})
        return groups

    # ------------------------------------------------------------------
    # Multimodal forward
    # ------------------------------------------------------------------
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[B,3,H,W] -> [B, N, hidden] projected visual tokens.

        The frozen backbone runs in fp32 with autocast disabled so features
        match the probe regime exactly; only the projector sees the training
        dtype/autocast.
        """
        pixel_values = pixel_values.to(self.encoder.device)
        with torch.no_grad():
            if pixel_values.is_cuda:
                with torch.autocast(device_type="cuda", enabled=False):
                    feats = self.encoder.encode_tensor(pixel_values).patches
            else:
                feats = self.encoder.encode_tensor(pixel_values).patches  # [B,N,Denc] fp32
        feats = feats.to(self.projector.fc1.weight.dtype)
        return self.projector(feats)

    def _embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(ids)

    def prepare_multimodal(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        image_embeds: Optional[torch.Tensor],
    ):
        """Splice projected visual tokens into the text embedding sequence at the
        IMAGE_TOKEN_INDEX marker, then right-pad the batch.

        Returns (inputs_embeds, attention_mask, labels), all [B, L', ...].
        """
        device = self.llm.device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        batch_embeds, batch_labels = [], []
        for i in range(input_ids.size(0)):
            keep = attention_mask[i].bool()
            ids = input_ids[i][keep]
            labs = labels[i][keep]

            img_pos = (ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=False).flatten()
            if img_pos.numel() == 0 or image_embeds is None:
                emb = self._embed_tokens(ids)
                lab = labs
            else:
                p = int(img_pos[0].item())
                pre = self._embed_tokens(ids[:p])
                post = self._embed_tokens(ids[p + 1:])
                vis = image_embeds[i].to(pre.dtype)  # [N, hidden]
                emb = torch.cat([pre, vis, post], dim=0)
                vis_labels = torch.full((vis.size(0),), -100, dtype=labs.dtype, device=device)
                lab = torch.cat([labs[:p], vis_labels, labs[p + 1:]], dim=0)
            batch_embeds.append(emb)
            batch_labels.append(lab)

        max_len = max(e.size(0) for e in batch_embeds)
        B = len(batch_embeds)
        out_embeds = torch.zeros(B, max_len, self.hidden, dtype=self.dtype, device=device)
        out_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        out_labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
        for i, (emb, lab) in enumerate(zip(batch_embeds, batch_labels)):
            n = emb.size(0)
            out_embeds[i, :n] = emb.to(self.dtype)
            out_mask[i, :n] = 1
            out_labels[i, :n] = lab
        return out_embeds, out_mask, out_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        image_embeds = self.encode_images(pixel_values) if pixel_values is not None else None
        inputs_embeds, attn, labels = self.prepare_multimodal(
            input_ids, attention_mask, labels, image_embeds
        )
        out = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=labels,
            use_cache=False,
        )
        # The image marker expands into patch tokens, so these labels are longer
        # than the caller's and are the ones whose indices line up with
        # `out.logits`. Hand them back so the decision loss can locate the first
        # answer token directly, instead of re-deriving a patch-count-dependent
        # offset that would silently rot if the token geometry changed.
        out.expanded_labels = labels
        return out

    # ------------------------------------------------------------------
    # Inference / generation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.Tensor],
        user_content: str,
        max_new_tokens: int = 256,
    ) -> str:
        """Single-image greedy generation for eval.

        `pixel_values` is a preprocessed [1,3,H,W] tensor (encoder.transform);
        `user_content` is the user-turn text with a single `<image>` placeholder.
        Prompt assembly mirrors `vlm_data.encode_messages` exactly (Qwen ChatML,
        same system prompt + `<image>`→IMAGE_TOKEN_INDEX marker) so the model
        sees the same format it was trained on. Returns the decoded response.
        """
        self.eval()
        IMAGE_PLACEHOLDER = "<image>"

        def tok(t: str) -> list[int]:
            return self.tokenizer(t, add_special_tokens=False).input_ids

        ids: list[int] = tok(f"<|im_start|>system\n{self.system_prompt}<|im_end|>\n")
        if IMAGE_PLACEHOLDER in user_content:
            pre, post = user_content.split(IMAGE_PLACEHOLDER, 1)
            ids += tok(f"<|im_start|>user\n{pre}") + [IMAGE_TOKEN_INDEX] + tok(f"{post}<|im_end|>\n")
        else:
            ids += tok(f"<|im_start|>user\n{user_content}<|im_end|>\n")
        ids += tok("<|im_start|>assistant\n")

        device = self.llm.device
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attn = torch.ones_like(input_ids)
        labels = torch.full_like(input_ids, -100)
        image_embeds = self.encode_images(pixel_values) if pixel_values is not None else None
        inputs_embeds, attn, _ = self.prepare_multimodal(input_ids, attn, labels, image_embeds)

        out = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=self.turn_end_id,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        # inputs_embeds path: generate returns only the new tokens.
        return self.tokenizer.decode(out[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke(encoder_id: str) -> int:
    """Build the model and run one forward+backward on synthetic data."""
    from src.vlm_data import build_collator, synthetic_samples

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[smoke] device={device} encoder={encoder_id}")
    model = WasteVLM(encoder_id=encoder_id, device=device)
    print(f"[smoke] patch_dim={model.patch_dim} hidden={model.hidden} "
          f"n_patches={model.encoder.n_patches}")
    model.apply_lora()
    model.freeze_for_training()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[smoke] trainable params: {n_trainable/1e6:.2f}M / {n_total/1e9:.2f}B")

    collate = build_collator(model.tokenizer, model.encoder.transform, model.system_prompt)
    batch = collate(synthetic_samples(n=2, image_size=model.image_size))

    model.train()
    out = model(**batch)
    loss = out.loss
    print(f"[smoke] forward ok, loss={loss.item():.4f}")
    loss.backward()
    gnorm = sum(p.grad.norm().item() for p in model.projector.parameters() if p.grad is not None)
    print(f"[smoke] backward ok, projector grad-norm sum={gnorm:.4f}")
    assert torch.isfinite(loss), "loss is not finite"
    assert gnorm > 0, "no gradient reached the projector"
    print("[smoke] OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Waste-VLM assembly smoke test.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--encoder", default="radio-l")
    args = ap.parse_args()
    if args.smoke:
        return _smoke(args.encoder)
    print("Nothing to do. Pass --smoke to validate the assembly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
