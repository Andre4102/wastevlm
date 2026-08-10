# Resume — Pruned-Decoder VLM (start here 2026-07-04 morning)

_Written 2026-07-03 evening. This is the cold-start handoff; the full design lives in
`PLAN.md` (keep pristine), status in `EXPERIMENTS.md`. This file = "what to do next"._

## TL;DR
The next step is **PLAN.md's B-row**: build a VLM on the **pruned + recovered
ts0.5 llama-3.1-8b decoder** (~45% sparse), then eval on DroneWaste + AerialWaste.
The whole B-path is **gated on one queued job** — the recovery CPT. Everything
else needed (waste-SFT data, A1 baseline, the pruned checkpoint) already exists.

**Framing lock (important):** the pruned-VLM claim is **iso-accuracy at lower cost**
(B2 ≥ A2 at fewer params / VRAM / latency), *not* beating InternVL/Qwen-VL on raw F1.
Absolute-accuracy lift comes from Phase-3 **waste-SFT**, not from pruning. So the
pruning result rides on the SFT working first.

---

## Where we are vs PLAN.md headline table

| cell | what | status | path |
|---|---|---|---|
| A1 | Qwen2.5-7B full, generic stage-2 | ✅ done + evaluated | `…/waste_vlm/results/vlm/radio-l_finetune/` (`llm_merged/` + `projector.pt`) |
| B1 | pruned-decoder VLM, generic stage-2 | ⛔ blocked on recovery CPT | — |
| A2 | A1 + waste-SFT | ⬜ not started (data ready) | — |
| B2 | B1 + waste-SFT | ⬜ blocked on B1 | — |
| Phase-0 data | 46,713-record waste-SFT `train.json` | ✅ done, leakage-checked | `…/waste_vlm/data/waste_sft/train.json` |

### The one blocker
- **`cpt_recover_prune45`** = job **`48424050`** — currently `PENDING` (Reason=Priority).
  Output dir `…/waste_vlm/results/llm/cpt_recover_prune45/` (only `args.json`+`tb/` so far
  → hasn't produced weights yet). **When this finishes, the recovered checkpoint is the
  B-row decoder.** Check first thing: `squeue -u adiecidu` / `sacct -j 48424050`.
- Also queued: `cpt_waste` base CPT = job **`48424051`** (secondary — "prune from the
  CPT'd model" idea, PLAN §non-headline; not on the B-row critical path).
- **NOTE:** these two jobs are the first real test of the DDP `static_graph` fix in
  `pruning/pruning_llama3_pretrain.py` (~L268). **Not yet verified** — nothing has run
  through it. Watch the first-startup log when they schedule.

### Key existing paths
- Pruned decoder (PRE-recovery, ~45% sparse, materialized):
  `…/waste_vlm/results/llm/mask_pruning_prune_waste_ts0.5/materialized_thr0.5`
  → can build/dry-run against this **today**; hot-swap recovered weights when 48424050 lands.
- radio-l stage-1 projector (Qwen): `…/waste_vlm/results/vlm/radio-l_pretrain/projector.pt`
- Recovery job script: `slurm_cpt_waste.sh` (override `MODEL_PATH=` + `RUN_NAME=` for recovery).

---

## Tomorrow's decision (pick one to start)
While recovery CPT runs, the productive unblocked work is one of:

1. **Build the pruned-decoder VLM path** (recommended, highest parallelism).
   Wire the LLaMA pruned decoder into the Qwen-built VLM against the *pre-recovery*
   `materialized_thr0.5` ckpt, then hot-swap. Non-trivial bits:
   - `WasteVLM` must load `custom-llama3` via `src/eval/_arch.py::register_pruning_arch`.
   - projector output dim **4096** (LLaMA) vs 3584 (Qwen) → new stage-1 projector.
   - **chat template: llama-3, NOT ChatML** — `encode_messages`/`generate` are
     hardcoded to `<|im_start|>…<|im_end|>`. Needs a template switch keyed on decoder
     family, applied identically in train + eval (PLAN Phase-1 flags this).
   - Then Phase-1 stage-1 projector pretrain (LCS-558K, gate: loss ≤ ~2.5).
2. **Prep Phase-4 comparison harness** — add efficiency columns (median latency,
   peak VRAM, materialized param count) + auto 2×2 A/B table to the eval.
3. **Just monitor** — wait until the recovered checkpoint exists, build against the real thing.

---

## Loose ends from today (VLM eval)
- **closed_vocab eval done, all 6** — micro-F1:

  | | dw_paper10 | aw_m2 | aw_m4 |
  |---|---|---|---|
  | dinov3-b | 0.062 | 0.219 | 0.165 |
  | radio-l | 0.062 | **0.293** | 0.186 |
  | *dino.txt (base)* | 0.40 | 0.37 | 0.38 |
  | *InternVL3-8B* | 0.368 | 0.452 | 0.304 |
  | *Qwen2.5-VL-7B* | 0.239 | 0.375 | 0.266 |

  - **radio > dino within our pipeline** (aw_m2/aw_m4) → the "better features help" thesis
    holds in the controlled comparison. On `dw` both are identical (0.062, recall≈1.0,
    prec≈0.03) = degenerate "say-yes-to-everything"; visual signal isn't reaching the
    decision there → projector/SFT bottleneck, not encoder. Feature advantage is real
    (frozen-probe data already positive) but doesn't transfer through the generative
    path given current training data.
- **LLM-judge: 3 radio-l cells still invalid (0.0000)** = `claude` CLI quota fallback
  (recall=0 across all → empty predictions), NOT real scores. Hit the account session
  limit (resets 11:50pm Brussels). **Re-run after reset:**
  `$CLAUDE_JOB_DIR/tmp/run_reparse_retry.sh` (dinov3-b/dw already fixed → 0.120; only the
  3 radio-l pairs remain: dw_paper10, aw_m2, aw_m4). kw-bag numbers are all valid.

## Quick status commands
```
squeue -u adiecidu -o "%.10i %.30j %.8T %.10M %R"
sacct -j 48424050,48424051 --format=JobID,JobName%16,State,Elapsed,Start
ls …/waste_vlm/results/llm/cpt_recover_prune45/   # weights appear here when recovery done
```
