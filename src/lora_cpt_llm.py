"""Text LoRA continued-pretraining of standalone Qwen2.5-7B-Instruct.

Stage 1 of the custom-VLM curriculum: domain-adapt the LLM backbone on the
curated waste text corpus, BEFORE it is wired to a DINOv3/RADIO vision encoder
via a projector (LLaVA-style) for the visual stage. Plain causal-LM LoRA — no
vision tower involved here.

    python -m src.lora_cpt_llm --out <dir> [--max-steps N]   # N>0 => smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

QWEN = "/home/ids/diecidue/results/waste_vlm/weights/Qwen2.5-7B-Instruct"
BLOCKS = "/home/ids/diecidue/data/waste_corpus/cpt_blocks.jsonl"
# Qwen2.5 LLM projection names (attention + MLP)
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class BlockDS(Dataset):
    def __init__(self, path):
        self.rows = [json.loads(l)["input_ids"] for l in Path(path).read_text().splitlines() if l.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        t = torch.tensor(self.rows[i], dtype=torch.long)
        return {"input_ids": t, "attention_mask": torch.ones_like(t), "labels": t.clone()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=QWEN)
    ap.add_argument("--blocks", default=BLOCKS)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=4.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()

    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                              TrainingArguments, default_data_collator)
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    print(f"[load] {args.base}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
                      bias="none", task_type="CAUSAL_LM", target_modules=TARGETS)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = BlockDS(args.blocks)
    print(f"[data] {len(ds)} blocks x {len(ds.rows[0])} tok", flush=True)

    targs = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=5, save_strategy="no", bf16=True, gradient_checkpointing=True,
        report_to=[], dataloader_num_workers=2, optim="adamw_torch",
    )
    Trainer(model=model, args=targs, train_dataset=ds,
            data_collator=default_data_collator).train()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tok.save_pretrained(str(args.out))
    print(f"[done] adapter -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
