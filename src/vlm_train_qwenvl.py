"""Joint structural-prune + LoRA fine-tune of the **native Qwen2.5-VL** decoder
on the waste VQA data.

Unlike `vlm_train.py` (our radio+projector+Qwen LLaVA stack, "A1"), this prunes
the decoder *inside* the real `Qwen2_5_VLForConditionalGeneration`:

  * the native ViT + merger are frozen (we only prune/adapt the language model);
  * the Qwen2.5 decoder is wrapped with structural masks via
    `Qwen2_5_VLMaskAdapter` (M-RoPE-aware masked forward, see
    `custom_attentions/qwen2_5_vl_attention.py`) and LoRA is injected on the LLM
    projections only (never the vision tower);
  * inputs are built with the native Qwen2.5-VL processor (chat template + image
    tokens), so image_grid_thw / pixel_values / 3-D position ids are all handled
    by the stock model.

The prune loop itself — schedule-based tau annealing, PI sparsity controller, and
the joint/sequential two-phase logic — is imported wholesale from `vlm_train.py`
(the model-agnostic `train`, `build_prune_ctx`, `_prune_tau_at`, `_set_all_tau`).

    torchrun --standalone --nproc_per_node=N -m src.vlm_train_qwenvl \\
        --base <Qwen2.5-VL dir> --train waste_sft/train.json --prune \\
        --prune-mode joint --target-sparsity 0.5 --out-dir <dir>

Materialization (saving a per-layer-flexible pruned decoder that reloads into
Qwen2.5-VL) needs the flexible-arch work tracked separately; with
`--no-materialize` (default until that lands) the run saves the LoRA adapter +
mask logits + a mask-probability dump for inspection.
"""
from __future__ import annotations

import argparse
import contextlib  # noqa: F401  (parity with vlm_train import surface)
import json
import math
import os
import sys

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from src.vlm_data import _load_records, _record_to_messages, _safe_open_rgb
from src.vlm_train import (
    setup_distributed, is_main_process, trainable_params,
    build_prune_ctx, train, make_prune_scheduler, _mask_group_index,
)

PRUNING_REPO = "/leonardo/home/userexternal/adiecidu/scripts/pruning"
DEFAULT_BASE = "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/weights/Qwen2.5-VL-7B-Instruct"
LLM_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


# ---------------------------------------------------------------------------
# Model: masked + LoRA'd Qwen2.5-VL decoder, frozen vision tower
# ---------------------------------------------------------------------------
def _import_pruning():
    if PRUNING_REPO not in sys.path:
        sys.path.insert(0, PRUNING_REPO)
    from mask_wrapper import wrap_model_with_masks, MaskedTransformerLayer
    from custom_attentions.qwen2_5_vl_attention import Qwen2_5_VLMaskAdapter
    return wrap_model_with_masks, MaskedTransformerLayer, Qwen2_5_VLMaskAdapter


def _select_llm_targets(model) -> list[str]:
    """LoRA target module names for the LLM projections ONLY.

    Return FULL module paths, not bare suffixes: the Qwen2.5-VL vision tower's
    MLP also has gate_proj/up_proj/down_proj, so bare-suffix targeting (PEFT
    matches by endswith) would inject LoRA into the ViT too. Full paths (which
    contain 'language_model') match exactly the decoder projections."""
    targets = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in LLM_PROJ:
            if any(k in name for k in ("visual", "vision", "merger")):
                continue
            targets.append(name)
    return targets


class QwenVLPrune(nn.Module):
    """Wraps Qwen2.5-VL with decoder masks + LoRA; exposes the interface the
    shared `train()` loop expects (`.llm`, `._MaskedTransformerLayer`,
    `.mask_sparsity_terms()`, and `forward(**batch) -> out.loss`)."""

    def __init__(self, base: str, device: str, dtype=torch.bfloat16,
                 prune_blocks: str = "both", tau_start: float = 5.0,
                 logit_init: float = 2.0, lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, freeze_qk: bool = False):
        super().__init__()
        self.freeze_qk = freeze_qk
        from transformers import Qwen2_5_VLForConditionalGeneration
        from peft import LoraConfig, get_peft_model
        wrap_model_with_masks, MaskedTransformerLayer, Qwen2_5_VLMaskAdapter = _import_pruning()

        # eager attention so the text model prepares an explicit additive 4-D mask
        # that our masked forward consumes (parity-validated path).
        vl = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base, torch_dtype=dtype, attn_implementation="eager")
        vl.config.use_cache = False
        for p in vl.parameters():
            p.requires_grad_(False)

        self.mask_adapter = Qwen2_5_VLMaskAdapter()
        self.mask_params = wrap_model_with_masks(
            vl, self.mask_adapter, tau_init=tau_start, logit_init=logit_init,
            gate_type="sigmoid", enable_layer_drop=False, prune_blocks=prune_blocks)
        self._MaskedTransformerLayer = MaskedTransformerLayer
        vl = vl.to(device=device, dtype=dtype)  # move mask params/buffers to device

        targets = _select_llm_targets(vl)
        assert targets, "no LLM projection modules found — check model structure"
        lora = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                          bias="none", task_type="CAUSAL_LM", target_modules=targets)
        vl = get_peft_model(vl, lora)
        # get_peft_model froze everything incl. the mask logits — re-enable masks.
        for p in self.mask_params:
            p.requires_grad_(True)
        # Channel-level QK pruning IS valid under M-RoPE during training: m_qk is a
        # per-ROTARY-PAIR mask ([H, hd_qk//2], expanded x2), and the masked forward
        # zeroes Q/K channels BEFORE apply_multimodal_rotary_pos_emb — a zeroed pair
        # rotates to zero, so no rope misalignment. (The mrope_section [16,24,24]
        # only complicates MATERIALIZE, where cos/sin must be sliced to the kept
        # pairs per section — deferred; the masked-eval path needs no slicing.)
        # So we leave m_qk TRAINABLE — dropping the coarse head-only pruning that
        # otherwise forces the param penalty to annihilate whole attention heads.
        if self.freeze_qk:
            for m in vl.modules():
                if isinstance(m, MaskedTransformerLayer):
                    with torch.no_grad():
                        m.m_qk.fill_(20.0)
                    m.m_qk.requires_grad_(False)
        # no LoRA/grad in the vision tower
        bad = [n for n, p in vl.named_parameters()
               if p.requires_grad and any(k in n for k in ("visual", "vision", "merger"))]
        assert not bad, f"trainable params leaked into vision tower: {bad[:3]}"

        # non-reentrant: compatible with DDP find_unused_parameters (masks freeze
        # at phase 2) and the monkeypatched masked forward.
        vl.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(vl, "enable_input_require_grads"):
            vl.enable_input_require_grads()

        self.vl = vl
        self.llm = vl  # shared train() iterates .llm.modules() for MaskedTransformerLayer

    def forward(self, **batch):
        return self.vl(**batch)

    def save_mask_logits(self, path: str):
        """Persist raw mask logits per layer so a pruned decoder can be
        materialized later (via build_plans_from_probs on _compute_probs)."""
        out = {}
        for i, m in enumerate(mod for mod in self.llm.modules()
                              if isinstance(mod, self._MaskedTransformerLayer)):
            out[i] = {"m_head": m.m_head.detach().cpu(),
                      "m_qk": m.m_qk.detach().cpu(),
                      "m_vo": m.m_vo.detach().cpu(),
                      "m_mlp": m.m_mlp.detach().cpu(),
                      "tau": float(m.tau.item())}
            if getattr(m, "m_qhead", None) is not None:
                out[i]["m_qhead"] = m.m_qhead.detach().cpu()
        torch.save(out, path)

    def mask_sparsity_terms(self):
        l1, total = 0.0, 0
        for m in self.llm.modules():
            if isinstance(m, self._MaskedTransformerLayer):
                l1 = l1 + m.l1_probability_sum()
                total += m.total_params()
        return l1, max(1, total)

    def optim_groups(self, lora_lr: float, mask_lr: float):
        mask_ids = {id(p) for p in self.mask_params}
        lora_ps = [p for p in self.vl.parameters()
                   if p.requires_grad and id(p) not in mask_ids]
        groups = [{"params": lora_ps, "lr": lora_lr}]
        mp = [p for p in self.mask_params if p.requires_grad]
        if mp and mask_lr > 0:
            groups.append({"params": mp, "lr": mask_lr})
        return groups


# ---------------------------------------------------------------------------
# Data: waste_sft records -> Qwen2.5-VL processor inputs (bs=1 per micro-step)
# ---------------------------------------------------------------------------
class QwenVLVQADataset(Dataset):
    def __init__(self, path: str):
        self.records = _load_records(path)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image, messages = _record_to_messages(self.records[idx])
        # messages: list of (role, text) with a single "<image>" placeholder in the
        # first user turn. Use the first user->assistant pair.
        user_text, answer = "", ""
        for role, text in messages:
            if role == "user" and not user_text:
                user_text = text
            elif role == "assistant" and user_text and not answer:
                answer = text
                break
        pil = _safe_open_rgb(_resolve(image)) if image else None
        return {"image": pil, "user": user_text.replace("<image>", "").strip(),
                "answer": answer}


def _resolve(image: str):
    from pathlib import Path
    return Path(image)


class QwenVLCollator:
    """Build labeled processor inputs for one record (batch_size=1). Labels mask
    the prompt (system+user+image) to -100; only the assistant answer is scored."""

    def __init__(self, processor, max_len: int = 2048):
        self.processor = processor
        self.max_len = max_len

    def __call__(self, batch):
        rec = batch[0]
        # Let the processor smart-resize within its min/max_pixels budget (set at
        # load time). High-res drone images otherwise blow up to ~32k vision
        # tokens and OOM the L^2 attention scores.
        img = rec["image"]
        content = ([{"type": "image", "image": img}] if img is not None else []) + \
                  [{"type": "text", "text": rec["user"]}]
        msgs_full = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": rec["answer"]}]},
        ]
        msgs_prompt = [{"role": "user", "content": content}]
        text_full = self.processor.apply_chat_template(
            msgs_full, tokenize=False, add_generation_prompt=False)
        text_prompt = self.processor.apply_chat_template(
            msgs_prompt, tokenize=False, add_generation_prompt=True)
        images = [img] if img is not None else None
        full = self.processor(text=[text_full], images=images, return_tensors="pt")
        plen = self.processor(text=[text_prompt], images=images,
                              return_tensors="pt").input_ids.shape[1]

        labels = full["input_ids"].clone()
        labels[:, :plen] = -100
        out = {k: v for k, v in full.items()}
        out["labels"] = labels
        return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--train", default=None, help="waste_sft train.json (abs image paths)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--min-pixels", type=int, default=256 * 28 * 28,
                    help="processor vision-token floor (min_pixels)")
    ap.add_argument("--max-pixels", type=int, default=1024 * 28 * 28,
                    help="processor vision-token cap (~1024 tokens); keeps the "
                         "L^2 attention scores in memory")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--lora-lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--smoke", action="store_true", help="cap total_steps to a few")
    # prune args mirror vlm_train (build_prune_ctx reads these)
    ap.add_argument("--prune", action="store_true")
    ap.add_argument("--prune-mode", choices=["joint", "sequential"], default="joint")
    ap.add_argument("--target-sparsity", type=float, default=0.5)
    ap.add_argument("--mask-lr", type=float, default=0.01)
    ap.add_argument("--prune-logit-init", type=float, default=2.0)
    ap.add_argument("--prune-freeze-qk", action="store_true",
                    help="freeze QK channel masks (coarse head-only attention "
                         "pruning). Default OFF = channel-level QK pruning.")
    ap.add_argument("--prune-tau-start", type=float, default=5.0)
    ap.add_argument("--prune-tau-min", type=float, default=0.01)
    ap.add_argument("--prune-tau-anneal-frac", type=float, default=0.7)
    ap.add_argument("--prune-phase2-frac", type=float, default=0.35)
    ap.add_argument("--prune-reg-weight", type=float, default=0.05)
    ap.add_argument("--prune-kp", type=float, default=2.0)
    ap.add_argument("--prune-ki", type=float, default=0.1)
    ap.add_argument("--prune-warmup", type=int, default=200)
    ap.add_argument("--prune-blocks", choices=["both", "mlp", "attn"], default="both")
    ap.add_argument("--no-materialize", action="store_true", default=True,
                    help="skip slicing (flexible Qwen2.5-VL materialize arch pending)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rank, local_rank, world_size, distributed = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    main_proc = is_main_process(rank)

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        args.base, min_pixels=args.min_pixels, max_pixels=args.max_pixels)

    model = QwenVLPrune(
        args.base, device=device, prune_blocks=args.prune_blocks,
        tau_start=args.prune_tau_start, logit_init=args.prune_logit_init,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        freeze_qk=args.prune_freeze_qk)

    if args.smoke or not args.train:
        # tiny synthetic-ish loop: reuse the first few real records if given.
        ds = QwenVLVQADataset(args.train) if args.train else None
        if ds is None:
            raise SystemExit("--train required (no synthetic path for the VL processor)")
    else:
        ds = QwenVLVQADataset(args.train)

    collate = QwenVLCollator(processor)
    sampler = DistributedSampler(ds, shuffle=True) if distributed else None
    loader = DataLoader(ds, batch_size=1, sampler=sampler, shuffle=(sampler is None),
                        num_workers=args.num_workers, collate_fn=collate,
                        pin_memory=True, drop_last=True)

    groups = model.optim_groups(args.lora_lr, args.mask_lr)
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    steps_per_epoch = max(len(loader) // args.grad_accum, 1)
    total_steps = 6 if args.smoke else max(int(steps_per_epoch * args.epochs), 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    prune_ctx = build_prune_ctx(args, model, total_steps)
    if prune_ctx is not None:
        scheduler = make_prune_scheduler(
            optimizer, mode=prune_ctx["mode"], warmup=warmup_steps,
            total_steps=total_steps, phase2_start=prune_ctx["phase2_start"],
            mask_group_idx=_mask_group_index(optimizer, model.mask_params))
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    if torch.cuda.is_available():
        model.to(device)
    train_module = model
    if distributed:
        train_module = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if main_proc:
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[qwenvl] world_size={world_size} trainable={n_tr/1e6:.2f}M "
              f"total_steps={total_steps} mode={args.prune_mode} "
              f"target_sparsity={args.target_sparsity}", flush=True)

    train(model, train_module, loader, sampler, optimizer, scheduler,
          device=device, rank=rank, distributed=distributed,
          epochs=args.epochs, grad_accum=args.grad_accum, total_steps=total_steps,
          max_grad_norm=args.max_grad_norm, logging_steps=args.logging_steps,
          save_steps=args.save_steps, out_dir=args.out_dir, stage="finetune",
          do_periodic_save=False, prune_ctx=prune_ctx)

    if distributed:
        dist.barrier()
    if main_proc:
        os.makedirs(args.out_dir, exist_ok=True)
        model.vl.save_pretrained(os.path.join(args.out_dir, "lora_adapter"))
        # dump mask probabilities (inspection) + raw logits (for materialize)
        _dump_mask_probs(model, os.path.join(args.out_dir, "mask_probs.json"))
        model.save_mask_logits(os.path.join(args.out_dir, "mask_logits.pt"))
        print(f"[qwenvl] saved LoRA adapter + mask probs + logits → {args.out_dir}",
              flush=True)
    return 0


def _dump_mask_probs(model, path):
    import torch
    out = {}
    for i, m in enumerate(mod for mod in model.llm.modules()
                          if isinstance(mod, model._MaskedTransformerLayer)):
        with torch.no_grad():
            probs = m._compute_probs()
        out[f"layer{i}"] = {k: float(v.float().mean()) for k, v in probs.items()
                            if hasattr(v, "float")}
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
