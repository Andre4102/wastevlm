"""Train the Waste-VLM: frozen DINO/RADIO encoder -> projector -> Qwen2.5-7B (LoRA).

Stage: visual SFT on the VQA train split. The encoder is frozen; the projector
is trained from scratch (higher LR); the LLM is adapted with LoRA (lower LR).
Loss is the standard LM loss on the assistant turn only (the collator masks
everything else to -100).

The training loop is a plain functional PyTorch/DDP loop (no HF Trainer): see
`train()` below. Classes live only in the model (`WasteVLM`) and the data
(`VQADataset`); everything here is a function.

Example (4x A100 DDP via torchrun):
    torchrun --standalone --nproc_per_node=4 -m src.vlm_train \
        --stage finetune --encoder radio-l \
        --train data/vqa/train.json --image-root data/images \
        --out-dir $RESULTS/vlm/run_radio_l --epochs 1 --batch-size 4 --grad-accum 8

Quick end-to-end check on synthetic data (no dataset needed, 1 GPU):
    python -m src.vlm_train --smoke --encoder radio-l
"""
from __future__ import annotations

import argparse
import contextlib
import math
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from src.vlm_data import (
    PretokenizedVQADataset,
    VQADataset,
    build_cached_collator,
    build_collator,
    synthetic_samples,
)
from src.vlm_model import DEFAULT_LLM_PATH, WasteVLM


# ---------------------------------------------------------------------------
# Distributed / device
# ---------------------------------------------------------------------------
def setup_distributed() -> tuple[int, int, int, bool]:
    """Init the process group under torchrun. Returns (rank, local_rank, world, dist)."""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        return dist.get_rank(), local_rank, world_size, True
    return 0, local_rank, 1, False


def is_main_process(rank: int) -> bool:
    return rank == 0


# ---------------------------------------------------------------------------
# Model / optimizer / checkpoint plumbing (all functions)
# ---------------------------------------------------------------------------
def build_model(args, device: str) -> WasteVLM:
    model = WasteVLM(
        llm_path=args.llm_path,
        encoder_id=args.encoder,
        image_size=args.image_size,
        device=device,
    )
    if args.stage == "pretrain":
        # Connector/alignment: train the projector only; LLM stays frozen.
        model.freeze_llm()
    else:
        # Instruction tuning: projector (full) + LoRA on the LLM.
        model.apply_lora(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model.freeze_for_training()

    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if args.projector_init:
        sd = torch.load(args.projector_init, map_location="cpu")
        model.projector.load_state_dict(sd)
        print(f"[train] loaded projector init from {args.projector_init}")
    return model


def build_optimizer(model: WasteVLM, projector_lr: float, lora_lr: float,
                    weight_decay: float = 0.0) -> torch.optim.Optimizer:
    """Two-group AdamW: projector (high LR) + LoRA (low LR)."""
    groups = model.trainable_parameter_groups(projector_lr, lora_lr)
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8,
                             weight_decay=weight_decay)


def trainable_params(model: WasteVLM) -> list[torch.Tensor]:
    return [p for p in model.parameters() if p.requires_grad]


def save_trainables(model: WasteVLM, out_dir: str, stage: str) -> None:
    """Persist only the trainable parts (encoder + LLM base are frozen).

    Used for periodic mid-training checkpoints: the LoRA adapter is saved
    separately (lightweight, resumable). The *final* finetune artifact instead
    folds LoRA into the base weights via `save_merged_model`.
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.projector.state_dict(), os.path.join(out_dir, "projector.pt"))
    if stage == "finetune":
        model.llm.save_pretrained(os.path.join(out_dir, "lora_adapter"))


def save_merged_model(model: WasteVLM, out_dir: str) -> None:
    """Final finetune save: bake the LoRA deltas into the LLM and drop the adapter.

    `merge_and_unload` folds the adapter into the base weights and returns a plain
    LLM with no PEFT wrapper — so the shipped artifact is a single merged model
    (`llm_merged/`) plus the projector, not a base model + kept adapter.
    """
    os.makedirs(out_dir, exist_ok=True)
    merged = model.llm.merge_and_unload()
    merged.save_pretrained(os.path.join(out_dir, "llm_merged"))
    model.tokenizer.save_pretrained(os.path.join(out_dir, "llm_merged"))
    torch.save(model.projector.state_dict(), os.path.join(out_dir, "projector.pt"))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def build_dataloader(dataset, collate, batch_size: int, distributed: bool,
                     num_workers: int = 4):
    """Map-style loader; DistributedSampler under DDP, shuffle otherwise."""
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        drop_last=False,
    )
    return loader, sampler


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(model, train_module, loader, sampler, optimizer, scheduler, *,
          device, rank, distributed, epochs, grad_accum, total_steps,
          max_grad_norm, logging_steps, save_steps, out_dir, stage,
          do_periodic_save):
    """Functional bf16 DDP loop with grad accumulation and trainable-only saves."""
    main = is_main_process(rank)
    model.train()

    global_step = 0
    running_loss = torch.zeros((), device=device)
    running_count = 0
    n_micro = len(loader)

    for epoch in range(math.ceil(epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch)

        for micro_step, batch in enumerate(loader):
            is_boundary = ((micro_step + 1) % grad_accum == 0) or (micro_step + 1 == n_micro)
            # Skip the all-reduce on non-boundary micro-steps (grad accumulation).
            sync_ctx = (train_module.no_sync()
                        if distributed and not is_boundary
                        else contextlib.nullcontext())
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = train_module(**batch)
                    loss = out.loss
                (loss / grad_accum).backward()

            running_loss += loss.detach()
            running_count += 1

            if not is_boundary:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params(model), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            if logging_steps > 0 and global_step % logging_steps == 0:
                # all_reduce is a collective: every rank must call it in lockstep,
                # so this block runs on all ranks and only the print is main-only.
                avg = running_loss / max(running_count, 1)
                if distributed:
                    dist.all_reduce(avg, op=dist.ReduceOp.AVG)
                if main:
                    lr = scheduler.get_last_lr()[0]
                    print(f"[train] step {global_step}/{total_steps} "
                          f"loss={avg.item():.4f} lr={lr:.2e}", flush=True)
                running_loss.zero_()
                running_count = 0

            if do_periodic_save and main and save_steps > 0 and global_step % save_steps == 0:
                save_trainables(model, os.path.join(out_dir, f"checkpoint-{global_step}"), stage)

            if global_step >= total_steps:
                return global_step

    return global_step


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Waste-VLM visual-stage trainer.")
    ap.add_argument("--train", help="VQA train json/jsonl (LLaVA instruction format)")
    ap.add_argument("--token-cache", default=None,
                    help="pre-tokenized cache dir from src.pretokenize_vlm "
                         "(skips per-batch tokenization; overrides --train)")
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--encoder", default="radio-l")
    ap.add_argument("--llm-path", default=DEFAULT_LLM_PATH)
    ap.add_argument("--out-dir", default="./vlm_run")
    ap.add_argument("--stage", choices=["pretrain", "finetune"], default="finetune",
                    help="pretrain = projector-only connector alignment (LLM frozen); "
                         "finetune = projector + LoRA visual-instruction tuning")
    ap.add_argument("--projector-init", default=None,
                    help="projector.pt from a prior stage to warm-start (stage 2)")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--projector-lr", type=float, default=None,
                    help="default: 1e-3 (pretrain) / 2e-4 (finetune)")
    ap.add_argument("--lora-lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="run a few steps on synthetic data, no dataset needed")
    args = ap.parse_args()
    if args.projector_lr is None:
        args.projector_lr = 1e-3 if args.stage == "pretrain" else 2e-4
    return args


def main() -> int:
    args = parse_args()
    rank, local_rank, world_size, distributed = setup_distributed()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    main_proc = is_main_process(rank)

    model = build_model(args, device)

    # --- dataset ---
    # Three sources: synthetic (smoke), a pre-tokenized cache (--token-cache), or
    # raw JSON (--train). The cache path skips per-batch tokenization, so it uses
    # the lightweight cached collator; the other two tokenize on the fly.
    if args.smoke:
        train_ds = synthetic_samples(n=64, image_size=args.image_size)
        collate = build_collator(model.tokenizer, model.encoder.transform,
                                 model.system_prompt, max_len=args.max_len)
        if args.out_dir == "./vlm_run":
            args.out_dir = "./vlm_smoke"
        args.epochs = 1.0
    elif args.token_cache:
        train_ds = PretokenizedVQADataset(args.token_cache, args.image_root)
        meta = train_ds.meta
        if main_proc:
            print(f"[train] token cache: {len(train_ds)} records, "
                  f"max_len={meta.get('max_len')} tokenizer={meta.get('llm_path')}",
                  flush=True)
            if (meta.get("system_prompt") != model.system_prompt
                    or meta.get("max_len") != args.max_len):
                print("[train] WARNING: token cache was built with a different "
                      "system_prompt/max_len than this run", flush=True)
        collate = build_cached_collator(model.tokenizer.pad_token_id,
                                        model.encoder.transform)
    else:
        if not args.train:
            raise SystemExit("--train or --token-cache is required unless --smoke is set")
        train_ds = VQADataset(args.train, args.image_root)
        collate = build_collator(model.tokenizer, model.encoder.transform,
                                 model.system_prompt, max_len=args.max_len)

    loader, sampler = build_dataloader(
        train_ds, collate, args.batch_size, distributed, args.num_workers
    )

    # --- optimizer + schedule ---
    optimizer = build_optimizer(model, args.projector_lr, args.lora_lr, args.weight_decay)

    steps_per_epoch = max(len(loader) // args.grad_accum, 1)
    total_steps = 4 if args.smoke else max(int(steps_per_epoch * args.epochs), 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- DDP wrap (only the trainables carry grad, so no unused params) ---
    train_module = model
    if distributed:
        train_module = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if main_proc:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[train] stage={args.stage} encoder={args.encoder} world_size={world_size} "
              f"trainable={n_trainable/1e6:.2f}M total_steps={total_steps} "
              f"warmup={warmup_steps}", flush=True)

    final_step = train(
        model, train_module, loader, sampler, optimizer, scheduler,
        device=device, rank=rank, distributed=distributed,
        epochs=args.epochs, grad_accum=args.grad_accum, total_steps=total_steps,
        max_grad_norm=args.max_grad_norm, logging_steps=args.logging_steps,
        save_steps=args.save_steps, out_dir=args.out_dir, stage=args.stage,
        do_periodic_save=not args.smoke,
    )

    if distributed:
        dist.barrier()
    if main_proc:
        if args.stage == "finetune":
            save_merged_model(model, args.out_dir)
            what = "merged LLM (LoRA folded in) + projector"
        else:
            save_trainables(model, args.out_dir, args.stage)
            what = "projector (connector stage)"
        print(f"[train] saved {what} to {args.out_dir} (step {final_step})", flush=True)

    if distributed:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
