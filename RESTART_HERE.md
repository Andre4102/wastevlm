# Waste-VLM — restart checkpoint (Leonardo)

*Written 2026-06-30. Read this first to resume the VLM visual-stage work.*

## What we're doing

Building the **VLM visual stage**: replace a stock VLM's CLIP encoder with the
frozen **DINO/RADIO** backbone chosen by the backbone-probing track
(RADIOv2.5-L @ 512² headline pick; DINOv3-B close 2nd — see `RESULTS_SNAPSHOT.md`
§1/§4), wire it through a projector into **Qwen2.5-7B-Instruct**, and train it
LLaVA-style. The repo was ported from the old IDS cluster to **Leonardo (CINECA)**.

## Decisions taken (with the user)

1. **Env**: fresh `waste_vlm` conda env built from the actual imports (no
   wastevlm-specific requirements file existed). Captured in `requirements.txt`.
2. **Scope**: build env + weights + code now; train on standard data (the
   project's own aerial-waste VQA isn't on Leonardo yet).
3. **Datasets**: **Standard LLaVA-1.5** two-stage recipe (not remote-sensing).
   - connector/alignment: LLaVA-Pretrain LCS-558K
   - visual-instruction tuning: LLaVA-Instruct-150K + COCO train2017
4. **Compute**: single A100 is ~44 h/epoch (too slow, >24 h limit) → **4×A100
   DDP** on one node (~11 h, same GPU-hours, no recipe compromise).

## Environment & paths (Leonardo)

| What | Path |
|---|---|
| Repo | `/leonardo/home/userexternal/adiecidu/scripts/wastevlm` |
| Conda env | `waste_vlm` (Python 3.11; torch 2.5.1+cu121, transformers 4.49, peft 0.15.2) |
| Scratch root `$WROOT` | `/leonardo_scratch/large/userexternal/adiecidu/waste_vlm` |
| Weights | `$WROOT/weights/{Qwen2.5-7B-Instruct, RADIO-L, dinov3-vitb16-pretrain-lvd1689m}` |
| DINOv3 hub repo | `$WROOT/dinov3_repo` (cloned facebookresearch/dinov3) |
| Datasets | `$WROOT/data/{llava_pretrain, llava_instruct, coco}` |
| Results/checkpoints | `$WROOT/results/vlm/` |
| HF cache | `/leonardo_scratch/large/userexternal/adiecidu/hf_cache` |
| SLURM | partition `boost_usr_prod`, account `iscrc_fiche`, A100 64GB |

Env vars the code reads (set by `slurm_vlm_train.sh`): `WASTE_VLM_WEIGHTS`,
`DINOV3_REPO`, `HF_HOME`, `HF_HUB_OFFLINE=1`.

## DONE ✅

- **Env** built + `requirements.txt` written.
- **Weights** downloaded: Qwen2.5-7B-Instruct (15 GB), RADIO-L (`nvidia/RADIO-L`),
  DINOv3-B (`facebook/...`, gated — the `HF_TOKEN` has access). DINOv3 hub repo cloned.
- **Paths repointed** from IDS → Leonardo in `src/vision_encoder.py` and
  `src/dinov3_backbone.py` (with env-var overrides). Added `patch_dim`/`transform`
  hooks to `VisionEncoder`.
- **Datasets downloaded & verified**: 558,128 LLaVA-Pretrain images + captions json;
  LLaVA-Instruct-150K json; 118,287 COCO train2017 images (46 GB total).
- **VLM code written** (the CLIP→DINO/RADIO swap):
  - `src/vlm_model.py` — `WasteVLM`: frozen encoder → 2-layer MLP projector →
    Qwen2.5-7B; LLaVA `-200` image-token splice; encoder kept fp32; `apply_lora()`,
    `freeze_llm()`, `freeze_for_training()`, `gradient_checkpointing_enable()`.
  - `src/vlm_data.py` — `VQADataset` (JSON-array or jsonl; LLaVA-conversations or
    flat QA; multi-turn) + collator with **assistant-only loss masking** (manual
    Qwen ChatML) + `synthetic_samples`.
  - `src/vlm_train.py` — **functional PyTorch/DDP loop (no HF Trainer)**: plain
    functions (`setup_distributed`, `build_model`, `build_optimizer`,
    `build_dataloader`, `train`, `save_trainables`, `save_merged_model`); classes
    live only in the model/data per the repo convention. Two optimizer groups
    (projector `1e-3`/`2e-4` + LoRA `2e-5`); `--stage pretrain|finetune`;
    `--projector-init`; bf16 autocast; grad-checkpoint; `no_sync` grad-accum;
    cosine+warmup; grad-clip; rank-0-only saving. **Periodic checkpoints save only
    the trainables** (projector + LoRA adapter, for resume); the **final finetune
    save merges LoRA into the LLM and drops the adapter** (`merge_and_unload` →
    `llm_merged/` + `projector.pt`) — the shipped artifact is one merged model.
  - `slurm_vlm_train.sh` — 4-GPU `torchrun` DDP launcher; modes `smoke|pretrain|finetune`.
- **Smokes passed** (debug QOS): single-GPU radio-l + dinov3-b, both stages
  (pretrain projector-only + finetune projector+LoRA). 4-GPU DDP smoke on the
  original HF-Trainer code (job 48155663). **After the functional-loop refactor +
  LoRA-merge-on-save, re-ran and both passed**: 1-GPU (48177723) and 4-GPU DDP
  (48177724), clean exit, merged-model save verified.

## Key facts / gotchas (so we don't relearn them)

- The VLM uses **Qwen2.5-7B-Instruct (text LLM)**, NOT `Qwen2.5-VL` (that's only
  the zero-shot baseline in `src/vlm_eval.py`).
- Projector input = encoder **patch** dim: radio-l → **1024**, dinov3-b → **768**.
  RADIO's CLS/summary is 3072 and is *unused*.
- 512² → **1024 visual tokens** (patch 16). This is the main cost driver.
- Qwen's `eos_token` **is** `<|im_end|>` (151645); pad is `<|endoftext|>` (151643).
- Single A100 ≈ 36 s/step (global batch 128) → **~44 h/epoch**. Hence 4-GPU DDP.
- Don't write checkpoints to compute-node `/tmp` (only ~9.6 GB) — use `$WROOT`.
- Mid-training checkpoints save only projector + LoRA adapter (small). The final
  finetune save *intentionally* writes the full merged 7B (`llm_merged/`, ~15 GB)
  — LoRA is folded in and the adapter dropped. Write it to `$WROOT`, not `/tmp`.

## 2026-07-01 update — DDP crash fixed + data pre-tokenized

- **Bug found & fixed**: the first real 4-GPU pretrain (radio-l + dinov3-b) crashed
  at **step 10** with an NCCL 600 s collective timeout → SIGABRT. Cause: in
  `src/vlm_train.py` the logging block's `dist.all_reduce(loss)` sat **inside the
  `if main:` guard**, so only rank 0 called the collective → ranks desynced. Fixed:
  the all-reduce + running-loss reset now run on **all** ranks; only the print is
  rank-0. (This is why the smokes passed: `total_steps=4` < `logging_steps=10`, so
  the logging path never executed.) Periodic-save block is fine — no collective.
- **Pre-tokenization added** (avoids re-parsing the 558K JSON per rank + per-batch
  tokenization; images still decoded lazily — can't cache decoded pixels):
  - `src/vlm_data.py`: extracted `encode_messages()` (single source of truth),
    added `PretokenizedVQADataset` (mmap'd token cache) + `build_cached_collator`.
  - `src/pretokenize_vlm.py`: functional, multiprocess builder → cache dir
    (`tokens/labels/offsets.npy` + `images.txt` + `meta.json`).
  - `slurm_pretokenize.sh`: CPU-only job (`lrd_all_serial`, 4 CPUs). Built both
    caches in ~3 min: `data/llava_pretrain/token_cache` (558,128 rec) and
    `data/llava_instruct/token_cache` (157,712 rec). Cache is Qwen-tokenizer-
    specific → shared by every encoder.
  - `slurm_vlm_train.sh`: pretrain/finetune auto-use `--token-cache` when the dir
    exists (fallback to `--train`); added `pretokenize` mode. Verified the cached
    batch is byte-identical to the on-the-fly collator.
- **Relaunched** pretrain radio-l (48184128) + dinov3-b (48184130) with fix+cache.
  Watch for `[train] step 10/2180` — clearing step 10 confirms the fix.

## 2026-07-01 — C-RADIOv4-H encoder added (DINOv3-teacher RADIO)

- **C-RADIOv4** = NVIDIA agglomerative model distilled from **SigLIP2 + DINOv3-7B +
  SAM3** teachers; C-RADIOv4-H (653M) is ~DINOv3-7B-competitive on dense tasks →
  good fit for aerial waste. Downloaded `nvidia/C-RADIOv4-H` →
  `$WROOT/weights/C-RADIOv4-H` (HF token already had access; NVIDIA Open Model Lic).
- **Drop-in with the RADIO path**: same `summary, features = model(x)` interface,
  `do_normalize=False`/`do_rescale=True` → **[0,1] input** (our radio transform is
  already correct), native 512², patch 16. Registry keys `cradiov4-h` / `cradiov4-so`
  in `src/vision_encoder.py`. **Validated smoke**: patch_dim=**1280** (projector
  auto-sizes), 1024 tokens @512², summary dim 2560 (unused), finite.
- **Gotcha fixed**: transformers' trust_remote_code copier skipped `utils.py` /
  `dual_hybrid_vit.py` → `FileNotFoundError`. Added self-healing `_sync_remote_code_cache`
  in `vision_encoder.py` (mirrors snapshot .py into the modules cache and retries).
- **No extra data prep**: the token cache is Qwen-keyed, so C-RADIOv4 reuses it.
  Launch when ready: `sbatch slurm_vlm_train.sh pretrain cradiov4-h`.
  (`$WROOT/weights/C-RADIOv4-H/c-radio_v4-h_half.pth.tar` is a 1.7 GB redundant
  copy of the safetensors — safe to delete.)

## RESUME — exact next steps

1. **Confirm the 4-GPU DDP smoke passed** (job 48155663):
   `grep train_loss logs/vlm_ddp_smoke_48155663.out`
2. **Launch connector pretrain (stage 1, ~11 h):**
   ```bash
   cd /leonardo/home/userexternal/adiecidu/scripts/wastevlm
   sbatch slurm_vlm_train.sh pretrain radio-l
   # -> $WROOT/results/vlm/radio-l_pretrain/projector.pt
   ```
3. **Launch instruction finetune (stage 2)** — warm-starts the projector:
   ```bash
   sbatch slurm_vlm_train.sh finetune radio-l
   # -> $WROOT/results/vlm/radio-l_finetune/{projector.pt, lora_adapter/}
   ```
4. (Optional) repeat both with `dinov3-b` to compare encoders.

## TODO / not done yet

- Run the two real training stages (above) to completion.
- **No eval/inference path yet** for the trained VLM (generate + metrics). Need a
  `src/vlm_infer.py` that loads encoder + projector.pt + LoRA and runs generation;
  then per-q_type metrics. `src/vlm_eval.py` currently only hosts the *zero-shot*
  baselines (and still has IDS paths).
- Swap stage-2 data to the project's **own aerial-waste VQA** when it's built/ported
  (same schema; just point `--train` at it).
- DINOv3-B uses the hub-repo + key-remap loader; RADIO-L is pure HF trust_remote_code.
