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
import time

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
from src.vlm_model import DEFAULT_LLM_PATH, PRUNING_REPO, WasteVLM


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
    prune = getattr(args, "prune", False)
    model = WasteVLM(
        llm_path=args.llm_path,
        encoder_id=args.encoder,
        image_size=args.image_size,
        pixel_shuffle=getattr(args, "pixel_shuffle", 1),
        device=device,
        prune=prune,
        prune_tau_start=getattr(args, "prune_tau_start", 5.0),
        prune_logit_init=getattr(args, "prune_logit_init", 2.0),
        prune_enable_layer_drop=getattr(args, "prune_enable_layer_drop", False),
        prune_blocks=getattr(args, "prune_blocks", "both"),
    )
    if args.stage == "pretrain":
        # Connector/alignment: train the projector only; LLM stays frozen.
        model.freeze_llm()
    else:
        # Instruction tuning: projector (full) + LoRA on the LLM.
        # apply_lora re-enables the mask logits when prune=True.
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


def decision_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """First-token ids of the Yes / No surface forms, deduplicated.

    Only the first token is scored: "Yes" and "No" tokenize to different lengths,
    so summing whole-word logprobs would compare sequences of unequal length and
    bake a length bias into the decision.
    """
    def ids(words):
        out = []
        for w in words:
            enc = tokenizer(w, add_special_tokens=False).input_ids
            if enc:
                out.append(enc[0])
        return sorted(set(out))
    return ids(["Yes", " Yes", "yes", " yes", "YES"]), ids(["No", " No", "no", " no", "NO"])


def decision_margin_loss(logits, labels, decision, yes_ids, no_ids, pos_weight=None):
    """BCE on the Yes-vs-No logit margin at the first answer token.

    Token CE weights a record by how long its answer is, so a 3-token decision
    sitting in a mix whose captions average 185 tokens contributes almost nothing
    -- which is how a 6.4% answer-token share came to flip the entire decision
    policy. This term is one scalar per record, so it cannot be diluted by answer
    length, and BCE puts the operating point at margin 0 by construction instead
    of leaving it wherever the answer prior happens to land. That is the failure
    we measured: AW ranks at AUC 0.84 but speaks at J 0.11 because its positives
    sit at margin -1.9.

    `labels` must be the EXPANDED mask returned by `WasteVLM.forward`, whose
    indices line up with `logits`. Records with decision < 0 are skipped.
    """
    import torch.nn.functional as F

    tgt = labels != -100
    decision = decision.to(logits.device)
    keep = (decision >= 0) & tgt.any(dim=1)
    if not bool(keep.any()):
        return logits.new_zeros(())
    # logits[t] predicts token t+1, so the position that predicts the first
    # answer token is one before it.
    first = tgt.float().argmax(dim=1)
    pos = (first - 1).clamp(min=0)
    sel = logits[torch.arange(logits.size(0), device=logits.device), pos].float()
    lp = torch.log_softmax(sel, dim=-1)
    margin = (torch.logsumexp(lp[:, yes_ids], dim=-1)
              - torch.logsumexp(lp[:, no_ids], dim=-1))
    return F.binary_cross_entropy_with_logits(
        margin[keep], decision[keep].float(), pos_weight=pos_weight)


def build_optimizer(model: WasteVLM, projector_lr: float, lora_lr: float,
                    weight_decay: float = 0.0, mask_lr: float = 0.0
                    ) -> torch.optim.Optimizer:
    """AdamW over projector (high LR) + LoRA (low LR) [+ mask logits (mask_lr)]."""
    groups = model.trainable_parameter_groups(projector_lr, lora_lr, mask_lr=mask_lr)
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


# --- resume plumbing (finetune 2-job chain: continue one cosine across jobs) ---
# A wall-clock-limited job saves periodic checkpoints; the next job restores the
# optimizer moments + scheduler position + global_step so the LR curve is ONE
# continuous cosine over `total_steps` (identical in both jobs because it is
# derived from the same dataset+world_size), not two restarted warmups. Only the
# non-prune finetune path uses this; prune runs have their own finalize.
def save_train_state(ckpt_dir: str, optimizer, scheduler, global_step: int,
                     sampler_offset: int) -> None:
    """Atomically persist optimizer+scheduler+step beside the trainable weights.

    Written LAST (after projector.pt + lora_adapter) and via os.replace, so a job
    killed mid-save leaves either a complete train_state.pt or none — the resumer
    only picks checkpoints that have a complete one."""
    tmp = os.path.join(ckpt_dir, "train_state.pt.tmp")
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": int(global_step),
        "sampler_offset": int(sampler_offset),
    }, tmp)
    os.replace(tmp, os.path.join(ckpt_dir, "train_state.pt"))


def find_latest_resumable(out_dir: str):
    """Newest checkpoint-<N> dir under out_dir that has a complete train_state.pt."""
    import glob
    import re
    best, best_step = None, -1
    for d in glob.glob(os.path.join(out_dir, "checkpoint-*")):
        if not os.path.exists(os.path.join(d, "train_state.pt")):
            continue
        m = re.search(r"checkpoint-(\d+)$", d)
        if not m:
            continue
        s = int(m.group(1))
        if s > best_step:
            best, best_step = d, s
    return best


def load_resume_weights(model: WasteVLM, ckpt_dir: str, stage: str) -> None:
    """Load the trainable weights (projector [+ LoRA adapter]) saved by a prior job."""
    proj = os.path.join(ckpt_dir, "projector.pt")
    model.projector.load_state_dict(torch.load(proj, map_location="cpu"))
    if stage == "finetune":
        from peft import set_peft_model_state_dict
        from peft.utils import load_peft_weights
        w = load_peft_weights(os.path.join(ckpt_dir, "lora_adapter"), device="cpu")
        res = set_peft_model_state_dict(model.llm, w)
        missing = getattr(res, "unexpected_keys", None)
        if missing:
            print(f"[train] WARNING: {len(missing)} unexpected LoRA keys on resume",
                  flush=True)


def _optimizer_state_to(optimizer, device) -> None:
    """Move loaded optimizer state tensors (Adam moments) onto the compute device."""
    for st in optimizer.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(device)


def _prune_tau_at(step: int, warmup: int, anneal_end: int,
                  tau_start: float, tau_min: float) -> float:
    """Linear tau schedule: hold tau_start through warmup, linearly anneal to
    tau_min by `anneal_end`, hold tau_min after. Schedule-based (NOT tied to
    sparsity progress) so it can't deadlock — sharpening the gates is what lets
    the discrete sparsity move AND makes the soft training forward match the hard
    materialized model."""
    if step <= warmup:
        return tau_start
    if step >= anneal_end:
        return tau_min
    f = (step - warmup) / max(anneal_end - warmup, 1)
    return tau_start + (tau_min - tau_start) * f


def _set_all_tau(module, tau: float, masked_layer_cls) -> None:
    for m in module.modules():
        if isinstance(m, masked_layer_cls):
            m.tau.fill_(tau)


def make_prune_scheduler(optimizer, *, mode: str, warmup: int, total_steps: int,
                         phase2_start: int, mask_group_idx=None,
                         phase2_warmup: int = 50, mask_floor: float = 0.2):
    """Per-group LR schedule for pruning — the mask logits and the weights get
    SEPARATE learning-rate schedules.

    The old single cosine over all groups broke pruned models two ways: it
    decayed `mask_lr` to ~0 so sparsity froze below target, and (sequential) it
    left the phase-2 weight-heal running on the dead tail of the cosine so the
    model never recovered. Here, per group:
      * WEIGHTS → joint: cosine(warmup, total). sequential: cosine over phase 1,
        then a FRESH warmup+cosine over the phase-2 recovery window.
      * MASKS → its OWN schedule, decoupled from the weights. joint: a gentle
        cosine decay from 1.0 to `mask_floor` (never 0 → keeps enough drive to
        reach target sparsity, while decaying so the masks commit toward the end).
        sequential: constant through phase 1, then 0 (masks frozen at phase 2).
    """
    def cos(step, w, span):
        if span <= 0:
            return 0.0
        if step < w:
            return step / max(1, w)
        p = min(1.0, (step - w) / max(1, span - w))
        return 0.5 * (1.0 + math.cos(math.pi * p))

    def weight_lambda(step):
        if mode == "sequential":
            if step < phase2_start:
                return cos(step, warmup, phase2_start)
            return cos(step - phase2_start, phase2_warmup, total_steps - phase2_start)
        return cos(step, warmup, total_steps)

    def mask_lambda(step):
        if mode == "sequential":
            return 1.0 if step < phase2_start else 0.0
        # joint: gentle cosine 1.0 -> mask_floor over the whole run (no warmup).
        p = min(1.0, step / max(1, total_steps))
        return mask_floor + (1.0 - mask_floor) * 0.5 * (1.0 + math.cos(math.pi * p))

    lambdas = [mask_lambda if i == mask_group_idx else weight_lambda
               for i in range(len(optimizer.param_groups))]
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambdas)


def _mask_group_index(optimizer, mask_params):
    """Index of the optimizer group holding the structural mask logits (or None)."""
    ids = {id(p) for p in mask_params}
    for i, g in enumerate(optimizer.param_groups):
        if any(id(p) in ids for p in g["params"]):
            return i
    return None


def build_prune_ctx(args, model: WasteVLM, total_steps: int):
    """Assemble the structural-pruning controllers for the prune loop.

    Tau annealing is SCHEDULE-based (see `_prune_tau_at`), fixing the old
    controller-driven deadlock (tau waited on sparsity, sparsity waited on tau).
    In `sequential` mode the run splits into phase 1 (learn masks) and a phase-2
    weight-heal (masks frozen + STE) at `phase2_start`; in `joint` mode masks are
    learned the whole way and phase2_start is never reached."""
    if not getattr(args, "prune", False):
        return None
    import sys
    if PRUNING_REPO not in sys.path:
        sys.path.insert(0, PRUNING_REPO)
    from mask_wrapper import MaskScheduler
    from sparsity_controller import SparsityController

    controller = SparsityController(
        target_sparsity=args.target_sparsity,
        Kp=args.prune_kp, Ki=args.prune_ki,
        rw_init=args.prune_reg_weight, warmup_steps=args.prune_warmup,
    )
    warmup = args.prune_warmup
    if args.prune_mode == "sequential":
        phase2_start = max(int((1.0 - args.prune_phase2_frac) * total_steps), warmup + 1)
    else:
        phase2_start = total_steps + 1  # joint: never freeze
    mask_window_end = phase2_start if args.prune_mode == "sequential" else total_steps
    anneal_end = warmup + int((mask_window_end - warmup) * args.prune_tau_anneal_frac)
    return {
        "controller": controller, "MaskScheduler": MaskScheduler,
        "target_sparsity": args.target_sparsity,
        "lam": float(args.prune_reg_weight), "cur_sp": 0.0,
        "mode": args.prune_mode, "warmup": warmup, "anneal_end": anneal_end,
        "phase2_start": phase2_start, "tau_start": args.prune_tau_start,
        "tau_min": args.prune_tau_min, "joint_enabled": False,
    }


def finalize_prune(model: WasteVLM, args, out_dir: str) -> str:
    """End-of-training: extract mask probs, fold in LoRA, unwrap masks, then
    materialize (slice) the trained decoder in-memory at the target threshold
    and save it. Also saves the projector. Returns the pruned-decoder dir.
    Run on the main process only.

    Materialization is done in-memory (not via pruning_vicuna_masked.py) so it
    only touches the lm_eval-free primitives, keeping the VLM training env light.
    """
    import os
    import sys
    import json
    if PRUNING_REPO not in sys.path:
        sys.path.insert(0, PRUNING_REPO)
    from mask_wrapper import (
        extract_probabilities, unwrap_model,
        build_plans_from_probs, compute_sparsity_from_plans,
    )
    from structure_pruning_utils_combined import (
        generic_slice_attention_weights_gqa, generic_slice_mlp_weights,
    )
    from custom_attentions.qwen2_attention import update_qwen2_config

    # 1. mask probabilities (while masks are still installed). The adapter's
    # get_layers wants the base Qwen2ForCausalLM (`.model.layers`); model.llm is
    # the PEFT wrapper here, whose `.model` resolves one level too shallow, so
    # reach through to the wrapped base model first.
    base_llm = (model.llm.get_base_model()
                if hasattr(model.llm, "get_base_model") else model.llm)
    probs = extract_probabilities(base_llm, model.mask_adapter)
    # 2. fold LoRA into the decoder → plain Qwen2ForCausalLM (masks still wrapped)
    merged = model.llm.merge_and_unload()
    # 3. remove the mask hooks / restore the plain forward
    unwrap_model(merged, model.mask_adapter)
    merged = merged.to("cpu")

    # 4. slice the trained decoder in-place per the thresholded plan (GQA path)
    plans = build_plans_from_probs(
        probs, threshold=args.materialize_threshold, enforce_uniform_count=True,
    )
    achieved = compute_sparsity_from_plans(plans, probs)
    for i, layer in enumerate(merged.model.layers):
        if i not in plans:
            continue
        sa, mlp = layer.self_attn, layer.mlp
        if "attn" in plans[i] and hasattr(sa, "reconstruct_weights"):
            s_args, s_kw = generic_slice_attention_weights_gqa(sa, *plans[i]["attn"])
            sa.reconstruct_weights(*s_args, **s_kw)
        if "mlp" in plans[i] and hasattr(mlp, "reconstruct_weights"):
            s_args, s_kw = generic_slice_mlp_weights(mlp, plans[i]["mlp"])
            mlp.reconstruct_weights(*s_args, **s_kw)
    merged.config = update_qwen2_config(merged, merged.config)
    if hasattr(merged, "generation_config"):
        # Greedy defaults; clear every sampling-only field or transformers'
        # strict save-time validation rejects the config (e.g. top_k=20 left
        # over from Qwen with do_sample=False).
        merged.generation_config.do_sample = False
        merged.generation_config.temperature = None
        merged.generation_config.top_p = None
        merged.generation_config.top_k = None

    # 5. save the pruned decoder + tokenizer + projector
    pruned_dir = os.path.join(out_dir, "decoder_pruned")
    os.makedirs(pruned_dir, exist_ok=True)
    merged.save_pretrained(pruned_dir)
    model.tokenizer.save_pretrained(pruned_dir)
    torch.save(model.projector.state_dict(), os.path.join(out_dir, "projector.pt"))
    with open(os.path.join(pruned_dir, "prune_summary.json"), "w") as f:
        json.dump({"target_sparsity": args.target_sparsity,
                   "materialize_threshold": args.materialize_threshold,
                   "achieved_sparsity": float(achieved)}, f, indent=2)
    print(f"[prune] materialized pruned decoder → {pruned_dir} "
          f"(achieved sparsity {achieved:.4f})", flush=True)

    # Sanity gate: a broken materialization/recipe produces a decoder that
    # generates degenerate text (empty or one token on repeat). Cheap greedy
    # probe on CPU so we never again trust a silently-broken 6h artifact.
    try:
        tok = model.tokenizer
        prompt = ("<|im_start|>user\nName three primary colors.<|im_end|>\n"
                  "<|im_start|>assistant\n")
        ids = tok(prompt, return_tensors="pt").input_ids.to(merged.device)
        with torch.no_grad():
            gen = merged.generate(ids, max_new_tokens=24, do_sample=False,
                                  pad_token_id=tok.pad_token_id or tok.eos_token_id)
        text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
        uniq = len(set(text.split()))
        tag = "OK" if uniq >= 4 else "DEGENERATE"
        print(f"[prune] sanity-gen [{tag}] uniq_words={uniq} :: {text!r}", flush=True)
        if uniq < 4:
            print("[prune] WARNING: pruned decoder generates degenerate text — the "
                  "materialized model is likely broken (soft!=hard gap or over-prune). "
                  "Inspect before trusting; consider sequential mode / lower target / "
                  "more anneal.", flush=True)
    except Exception as e:
        print(f"[prune] sanity-gen skipped ({e})", flush=True)
    return pruned_dir


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
          do_periodic_save, prune_ctx=None, dec_ctx=None, start_step=0,
          sampler_offset=0):
    """Functional bf16 DDP loop with grad accumulation and trainable-only saves.

    When ``prune_ctx`` is set, a learned-mask structural-sparsity penalty is
    added once per optimizer step: the PI controller reads the current expected
    (post-threshold) sparsity and sets the penalty weight λ toward the target;
    the tau scheduler anneals the mask sharpness as sparsity approaches target."""
    main = is_main_process(rank)
    model.train()

    # On resume, continue the step count; the scheduler/optimizer were already
    # restored to `start_step` in main(). sampler_offset shifts the shuffle so a
    # resumed job draws a fresh permutation instead of replaying job-1's prefix.
    global_step = start_step
    running_loss = torch.zeros((), device=device)
    running_dec = torch.zeros((), device=device)
    running_count = 0
    n_micro = len(loader)
    # Step time and peak memory are what size the next arm: visual-token count
    # scales as (image_size / (patch * pixel_shuffle))^2, so a pixel-shuffle or
    # resolution change moves both, and guessing the batch size wastes a queue
    # slot. Reported on the periodic log line rather than only at the end.
    step_timer = time.time()
    last_log_step = global_step
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(math.ceil(epochs)):
        if sampler is not None:
            sampler.set_epoch(epoch + sampler_offset)

        for micro_step, batch in enumerate(loader):
            is_boundary = ((micro_step + 1) % grad_accum == 0) or (micro_step + 1 == n_micro)

            # At each optimizer step's first micro-batch, refresh the sparsity
            # controller: measure current sparsity, set λ.
            if prune_ctx is not None and (micro_step % grad_accum == 0):
                with torch.no_grad():
                    cur_sp = prune_ctx["MaskScheduler"].expected_sparsity(model.llm)
                prune_ctx["cur_sp"] = float(cur_sp)
                # Only regulate lambda while masks are still being learned. Once
                # phase 2 (STE heal) starts the masks are frozen, so the penalty
                # + controller are inert.
                if not prune_ctx["joint_enabled"]:
                    prune_ctx["lam"] = float(prune_ctx["controller"].step(float(cur_sp)))

            # Skip the all-reduce on non-boundary micro-steps (grad accumulation).
            sync_ctx = (train_module.no_sync()
                        if distributed and not is_boundary
                        else contextlib.nullcontext())
            with sync_ctx:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = train_module(**batch)
                    loss = out.loss
                    if dec_ctx is not None and "decision" in batch:
                        d_loss = decision_margin_loss(
                            out.logits, out.expanded_labels, batch["decision"],
                            dec_ctx["yes_ids"], dec_ctx["no_ids"],
                            pos_weight=dec_ctx["pos_weight"])
                        loss = loss + dec_ctx["weight"] * d_loss
                        running_dec += d_loss.detach()
                step_loss = loss
                # Add the structural-sparsity penalty once per step (on the
                # boundary micro, inside the synced backward). The ×grad_accum
                # cancels the /grad_accum below so the penalty lands at weight 1.
                if prune_ctx is not None and is_boundary and not prune_ctx["joint_enabled"]:
                    l1, total = model.mask_sparsity_terms()
                    pen = (l1 / total) * prune_ctx["lam"]
                    step_loss = loss + pen * grad_accum
                (step_loss / grad_accum).backward()

            running_loss += loss.detach()
            running_count += 1

            if not is_boundary:
                continue

            torch.nn.utils.clip_grad_norm_(trainable_params(model), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1
            # Anneal mask sharpness (tau) on a fixed schedule so the gates become
            # ~binary (soft-train forward -> hard-materialize model), and in
            # sequential mode flip to the phase-2 STE weight-heal at phase2_start.
            if prune_ctx is not None:
                tau = _prune_tau_at(global_step, prune_ctx["warmup"],
                                    prune_ctx["anneal_end"], prune_ctx["tau_start"],
                                    prune_ctx["tau_min"])
                _set_all_tau(model.llm, tau, model._MaskedTransformerLayer)
                if (not prune_ctx["joint_enabled"]
                        and global_step >= prune_ctx["phase2_start"]):
                    prune_ctx["MaskScheduler"].enable_joint_phase(model.llm)
                    prune_ctx["joint_enabled"] = True
                    if main:
                        print(f"[prune] --> phase 2: froze masks + STE weight-heal "
                              f"at step {global_step} "
                              f"(sparsity={prune_ctx['cur_sp']:.4f})", flush=True)

            if logging_steps > 0 and global_step % logging_steps == 0:
                # all_reduce is a collective: every rank must call it in lockstep,
                # so this block runs on all ranks and only the print is main-only.
                avg = running_loss / max(running_count, 1)
                avg_dec = running_dec / max(running_count, 1)
                if distributed:
                    dist.all_reduce(avg, op=dist.ReduceOp.AVG)
                    dist.all_reduce(avg_dec, op=dist.ReduceOp.AVG)
                if main:
                    lr = scheduler.get_last_lr()[0]
                    extra = ""
                    if dec_ctx is not None:
                        extra += f" dec={avg_dec.item():.4f}"
                    if prune_ctx is not None:
                        tau_now = _prune_tau_at(
                            global_step, prune_ctx["warmup"], prune_ctx["anneal_end"],
                            prune_ctx["tau_start"], prune_ctx["tau_min"])
                        phase = "P2" if prune_ctx["joint_enabled"] else "P1"
                        extra = (f" sparsity={prune_ctx['cur_sp']:.4f}/"
                                 f"{prune_ctx['target_sparsity']:.2f} "
                                 f"lam={prune_ctx['lam']:.3g} tau={tau_now:.3g} {phase}")
                    now = time.time()
                    s_per_step = (now - step_timer) / max(global_step - last_log_step, 1)
                    peak_gb = (torch.cuda.max_memory_allocated() / 2**30
                               if torch.cuda.is_available() else 0.0)
                    eta_h = s_per_step * max(total_steps - global_step, 0) / 3600
                    print(f"[train] step {global_step}/{total_steps} "
                          f"loss={avg.item():.4f} lr={lr:.2e}{extra} "
                          f"{s_per_step:.1f}s/step peak={peak_gb:.1f}GB eta={eta_h:.1f}h",
                          flush=True)
                    step_timer, last_log_step = now, global_step
                running_loss.zero_()
                running_dec.zero_()
                running_count = 0

            if do_periodic_save and main and save_steps > 0 and global_step % save_steps == 0:
                ck = os.path.join(out_dir, f"checkpoint-{global_step}")
                save_trainables(model, ck, stage)
                # Resume state only for the non-prune finetune chain; prune runs
                # materialize via finalize_prune and never resume.
                if prune_ctx is None:
                    save_train_state(ck, optimizer, scheduler, global_step,
                                     sampler_offset)

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
    ap.add_argument("--resume", default=None,
                    help="resume a finetune from a checkpoint-<N> dir (restores "
                         "projector+LoRA+optimizer+scheduler+step so the cosine LR "
                         "continues as one curve). 'auto' = newest resumable "
                         "checkpoint in --out-dir; used for the 2-job full-epoch chain.")
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--pixel-shuffle", type=int, default=1,
                    help="fold s x s patch neighbourhoods into channels before the "
                         "projector (s=2 -> 1/4 the visual tokens). 1 = off (LLaVA-1.5).")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="if >0, train exactly this many optimizer steps and ignore "
                         "--epochs (used to hold optimizer steps equal across the "
                         "matched-budget alignment arms; epochs are derived to cover it)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--projector-lr", type=float, default=None,
                    help="default: 1e-3 (pretrain) / 2e-4 (finetune)")
    ap.add_argument("--lora-lr", type=float, default=2e-5)
    ap.add_argument("--decision-loss-weight", type=float, default=0.0,
                    help="weight of the Yes/No margin BCE on records carrying a "
                         "`decision` field; 0 disables (default, so existing arms "
                         "are bit-identical)")
    ap.add_argument("--decision-pos-weight", type=float, default=None,
                    help="BCE pos_weight for the decision term; use >1 when the "
                         "decision records are negative-heavy")
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
    # --- joint structural pruning (prune the decoder while fine-tuning) ---
    ap.add_argument("--prune", action="store_true",
                    help="build a masked Qwen-2 decoder and learn structural masks "
                         "jointly with LoRA+projector (forces --stage finetune)")
    ap.add_argument("--target-sparsity", type=float, default=0.5,
                    help="fraction of prunable decoder params to drop")
    ap.add_argument("--mask-lr", type=float, default=0.01,
                    help="LR for the structural mask logits")
    ap.add_argument("--prune-logit-init", type=float, default=2.0,
                    help="mask logit init; σ(2)≈0.88 → most channels start kept")
    ap.add_argument("--prune-tau-start", type=float, default=5.0)
    ap.add_argument("--prune-tau-min", type=float, default=0.01)
    ap.add_argument("--prune-reg-weight", type=float, default=0.05,
                    help="PI-controller rw_init (seeds the sparsity penalty weight)")
    ap.add_argument("--prune-kp", type=float, default=2.0)
    ap.add_argument("--prune-ki", type=float, default=0.1)
    ap.add_argument("--prune-warmup", type=int, default=200,
                    help="steps before the sparsity controller starts acting")
    ap.add_argument("--prune-enable-layer-drop", action="store_true")
    ap.add_argument("--prune-blocks", choices=["both", "mlp", "attn"], default="both")
    ap.add_argument("--materialize-threshold", type=float, default=0.5,
                    help="probability threshold for materializing the pruned decoder")
    ap.add_argument("--prune-mode", choices=["joint", "sequential"], default="joint",
                    help="joint = learn masks WHILE fine-tuning (tau annealed sharp "
                         "throughout so the forward tracks the real hard pruning); "
                         "sequential = learn masks (phase 1), then freeze+STE and "
                         "heal the weights against the discrete structure (phase 2)")
    ap.add_argument("--prune-phase2-frac", type=float, default=0.35,
                    help="sequential mode: fraction of total steps for the phase-2 heal")
    ap.add_argument("--prune-tau-anneal-frac", type=float, default=0.7,
                    help="fraction of the mask-learning window (joint: whole run; "
                         "sequential: phase 1) over which tau linearly anneals "
                         "tau_start->tau_min, then holds at tau_min. Annealing to a "
                         "near-binary tau is what makes soft(train) == hard(materialize).")
    args = ap.parse_args()
    if args.prune and args.stage != "finetune":
        print("[train] --prune requires --stage finetune; forcing it.")
        args.stage = "finetune"
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
    mask_lr = args.mask_lr if getattr(args, "prune", False) else 0.0
    optimizer = build_optimizer(model, args.projector_lr, args.lora_lr,
                                args.weight_decay, mask_lr=mask_lr)
    steps_per_epoch = max(len(loader) // args.grad_accum, 1)
    if args.smoke:
        total_steps = 4
    elif args.max_steps > 0:
        # Fixed-step mode: hold optimizer steps equal across arms. Derive epochs
        # to cover max_steps so the outer epoch loop doesn't stop early (the loop
        # breaks on global_step >= total_steps regardless).
        total_steps = args.max_steps
        args.epochs = math.ceil(total_steps / steps_per_epoch)
    else:
        total_steps = max(int(steps_per_epoch * args.epochs), 1)
    warmup_steps = int(total_steps * args.warmup_ratio)
    prune_ctx = build_prune_ctx(args, model, total_steps)
    dec_ctx = None
    if args.decision_loss_weight > 0:
        yes_ids, no_ids = decision_token_ids(model.tokenizer)
        if not yes_ids or not no_ids:
            raise SystemExit("could not resolve Yes/No token ids for the decision loss")
        pw = (torch.tensor(args.decision_pos_weight, device=device)
              if args.decision_pos_weight else None)
        dec_ctx = {"yes_ids": yes_ids, "no_ids": no_ids,
                   "weight": args.decision_loss_weight, "pos_weight": pw}
        if main_proc:
            print(f"[train] decision margin BCE on: weight={args.decision_loss_weight} "
                  f"pos_weight={args.decision_pos_weight} "
                  f"yes={yes_ids} no={no_ids}", flush=True)
    if prune_ctx is not None:
        scheduler = make_prune_scheduler(
            optimizer, mode=prune_ctx["mode"], warmup=warmup_steps,
            total_steps=total_steps, phase2_start=prune_ctx["phase2_start"],
            mask_group_idx=_mask_group_index(optimizer, model.mask_params))
    else:
        scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # --- resume (finetune 2-job chain): restore weights + optimizer + schedule ---
    # Runs on every rank (each built its own model/optimizer/scheduler). Weight
    # loads copy into the already-on-device params; optimizer moments are moved to
    # device explicitly. total_steps is unchanged, so the cosine stays one curve.
    start_step = 0
    sampler_offset = 0
    resume_dir = getattr(args, "resume", None)
    if resume_dir == "auto":
        resume_dir = find_latest_resumable(args.out_dir)
        if resume_dir is None and main_proc:
            print("[train] --resume auto: no resumable checkpoint yet, "
                  "starting fresh", flush=True)
    if resume_dir:
        load_resume_weights(model, resume_dir, args.stage)
        st = torch.load(os.path.join(resume_dir, "train_state.pt"),
                        map_location="cpu")
        optimizer.load_state_dict(st["optimizer"])
        _optimizer_state_to(optimizer, device)
        scheduler.load_state_dict(st["scheduler"])
        start_step = int(st["global_step"])
        sampler_offset = int(st.get("sampler_offset", 0)) + 1
        if main_proc:
            print(f"[train] RESUMED from {resume_dir}: start_step={start_step} "
                  f"total_steps={total_steps} sampler_offset={sampler_offset}",
                  flush=True)

    # --- DDP wrap (only the trainables carry grad, so no unused params) ---
    # Guard: the masked-Qwen + LoRA build can leave stray params/buffers on CPU
    # (PEFT-injected adapters, lazily-built rope buffers), which makes DDP abort
    # with "input module parameters locate in {'cpu', 'cuda'}". Diagnose, then
    # force a uniform device before wrapping. Cheap and idempotent.
    stray_p = [n for n, p in model.named_parameters() if p.device.type != "cuda"]
    stray_b = [n for n, b in model.named_buffers()
               if b is not None and b.device.type != "cuda"]
    if (stray_p or stray_b) and main_proc:
        print(f"[train] pre-DDP: {len(stray_p)} params + {len(stray_b)} buffers "
              f"on CPU → moving to {device}. "
              f"params e.g. {stray_p[:6]} buffers e.g. {stray_b[:6]}", flush=True)
    if torch.cuda.is_available():
        model.to(device)
    train_module = model
    if distributed:
        # Sequential mode freezes the mask logits at phase 2, so they stop
        # producing gradients mid-run — DDP must tolerate now-unused params.
        find_unused = bool(getattr(args, "prune", False))
        train_module = DDP(model, device_ids=[local_rank],
                           find_unused_parameters=find_unused)

    if main_proc:
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[train] stage={args.stage} encoder={args.encoder} world_size={world_size} "
              f"trainable={n_trainable/1e6:.2f}M total_steps={total_steps} "
              f"warmup={warmup_steps}", flush=True)

    run_t0 = time.time()
    final_step = train(
        model, train_module, loader, sampler, optimizer, scheduler,
        device=device, rank=rank, distributed=distributed,
        epochs=args.epochs, grad_accum=args.grad_accum, total_steps=total_steps,
        max_grad_norm=args.max_grad_norm, logging_steps=args.logging_steps,
        save_steps=args.save_steps, out_dir=args.out_dir, stage=args.stage,
        do_periodic_save=not args.smoke, prune_ctx=prune_ctx, dec_ctx=dec_ctx,
        start_step=start_step, sampler_offset=sampler_offset,
    )

    if distributed:
        dist.barrier()
    if main_proc:
        # Always-on cost report. A --smoke run is only 4 steps, so it never hits
        # the periodic log line, yet sizing the next arm (pixel-shuffle or
        # resolution change) is exactly what the smoke is for.
        ran = max(final_step - start_step, 1)
        wall = time.time() - run_t0
        vis_tok = (args.image_size // (16 * getattr(args, "pixel_shuffle", 1))) ** 2
        peak_gb = (torch.cuda.max_memory_allocated() / 2**30
                   if torch.cuda.is_available() else 0.0)
        print(f"[cost] img={args.image_size} pshuf={getattr(args, 'pixel_shuffle', 1)} "
              f"visual_tokens={vis_tok} bs={args.batch_size} accum={args.grad_accum} "
              f"world={world_size} | {wall/ran:.1f}s/step peak={peak_gb:.1f}GB "
              f"({ran} steps in {wall/60:.1f}min)", flush=True)
        if prune_ctx is not None:
            finalize_prune(model, args, args.out_dir)
            what = "pruned+merged decoder + projector"
        elif args.stage == "finetune":
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
