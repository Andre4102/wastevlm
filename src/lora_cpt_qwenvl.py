"""Text-only LoRA continued-pretraining of the LANGUAGE layers of Qwen2.5-VL-7B.

Stage 1 of the curriculum: inject waste-domain knowledge into the LLM backbone
of the VLM *before* visual instruction tuning. We feed the packed text corpus
(no images), put LoRA adapters on the language-model projections only (the
Qwen2.5 vision tower uses fused `qkv`/`proj` names, so the standard LLM proj
names never touch it — we assert this), and train a causal-LM objective.

Default is 4-bit QLoRA (safe on any GPU); pass --bf16 for plain bf16 LoRA if VRAM
is ample. Saves the adapter to --out, ready to (a) merge for the visual stage or
(b) load alongside the VLM.

    python -m src.lora_cpt_qwenvl --out <dir> [--bf16] [--max-steps N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

QWEN_VL = "/home/ids/diecidue/results/waste_vlm/weights/Qwen2.5-VL-7B-Instruct"
BLOCKS = "/home/ids/diecidue/data/waste_corpus/cpt_blocks.jsonl"
LLM_PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class BlockDS(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l)["input_ids"] for l in Path(path).read_text().splitlines() if l.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids = self.rows[i]
        t = torch.tensor(ids, dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t), "labels": t.clone()}


def select_llm_targets(model) -> list[str]:
    """LoRA target module names: LLM projections only, never the vision tower."""
    targets = set()
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and name.split(".")[-1] in LLM_PROJ:
            if "visual" in name or "merger" in name or "vision" in name:
                continue
            targets.add(name.split(".")[-1])
    return sorted(targets)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=QWEN_VL)
    ap.add_argument("--blocks", default=BLOCKS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--bf16", action="store_true", help="plain bf16 LoRA instead of 4-bit QLoRA")
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke test: cap steps")
    args = ap.parse_args()

    from transformers import (AutoTokenizer, Trainer, TrainingArguments,
                              default_data_collator, BitsAndBytesConfig)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
    except Exception:
        from transformers import AutoModelForImageTextToText as VLModel
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    load_kw = dict(trust_remote_code=True, torch_dtype=torch.bfloat16)
    if not args.bf16:
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    print(f"[load] {args.base}  mode={'bf16' if args.bf16 else '4bit-QLoRA'}", flush=True)
    model = VLModel.from_pretrained(args.base, **load_kw)
    model.config.use_cache = False

    if not args.bf16:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    targets = select_llm_targets(model)
    print(f"[lora] target modules: {targets}", flush=True)
    assert targets, "no LLM projection modules found — check model structure"

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM", target_modules=targets)
    model = get_peft_model(model, lora)
    # sanity: every trainable LoRA param must live in the language model, not the vision tower
    bad = [n for n, p in model.named_parameters() if p.requires_grad and ("visual" in n or "vision" in n)]
    assert not bad, f"LoRA leaked into vision tower: {bad[:3]}"
    model.print_trainable_parameters()

    ds = BlockDS(args.blocks)
    print(f"[data] {len(ds)} blocks of {len(ds.rows[0])} tokens", flush=True)

    targs = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=5, save_strategy="no", bf16=True, gradient_checkpointing=True,
        report_to=[], dataloader_num_workers=2, optim="paged_adamw_8bit" if not args.bf16 else "adamw_torch",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds, data_collator=default_data_collator)
    trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tok.save_pretrained(str(args.out))
    print(f"[done] adapter saved -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
