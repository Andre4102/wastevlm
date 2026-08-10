# Waste-LLM / Waste-VLM — Experiment Status

_Last updated: 2026-07-04_

## Track A — LLM structural pruning (pruning-paper case study)

**Setup.** Learned-mask structural pruning of **llama-3.1-8b (base)**, with LoRA-FT
*during* mask search. Calibrated on 1,384 waste-mostly samples (seq 2048).
Target-sparsity sweep {0.5, 0.9}. Masks temperature-anneal to (near-)binary, so
all materialization thresholds collapse to a single operating point per run.

**Waste-benchmark results (base vs. pruned, pre-recovery):**

| Model | Sparsity | low_qa ↑ | concept_qa ↑ | appearance PPL ↓ | corpus PPL ↓ | wikitext PPL ↓ |
|-------|:--------:|:--------:|:------------:|:----------------:|:------------:|:--------------:|
| base llama-3.1-8b | 0%   | 0.341 | 0.491 | 11.1 | **7.14** | ~6–7 |
| ts0.5             | ~45% | 0.343 | 0.509 | 23.2 | 13.6 | 12.5 |
| ts0.9             | ~88% | 0.316 | 0.455 | 325  | 97.9 | 70.5 |
| _random_          | —    | 0.25  | 0.25  | —    | —    | —    |

**Reading it.**
- MC accuracy is robust even at 88%; **PPL is where the damage shows** (1.9× at
  45%, ~14× at 88%). Classic signature of pruning **with ~zero recovery
  training** — the 1,384 samples were only for mask search.
- 45% is a usable operating point; 88% breaks free-form generation.
- These pruned the **base** model, not a domain-adapted one. Base itself is weak
  on low_qa (0.34, barely > random) — a knowledge gap CPT targets, not pruning
  damage.

**Recovery anchor (from prior work):** 20k×2048 (~41M tokens) recovery took a
pruned Vicuna-1.5-7B to PPL ~9 — recovery is cheap and proven in this setup.

**Status.** baseline eval ✅ · both prunes ✅ · waste-eval of prunes ✅ ·
base CPT (`cpt_waste`, 300M tok, job `48355745`) PENDING · **recovery-CPT on
45% ckpt (`cpt_recover_prune45`, 150M tok, job `48355743`) PENDING**.

> **CPT-trainer regression (2026-07-03, fixed).** First submissions of both CPT
> jobs (`48278134`, `48316869`) died at startup: the pruning repo's
> `pruning_llama3_pretrain.py` was refactored so the trainer now *always* builds
> a PI `SparsityController` (`if sparsity_controller is None: raise`) and seeds
> its `rw_init` from `--reg_weight`, which is rejected if `≤ 0`
> (`ValueError: All rw values must be positive`). Our mask-off recipe passed
> `--reg_weight 0.0`. Fix: `slurm_cpt_waste.sh` now uses `--reg_weight 0.05`
> (inert — `mask_lr=0` + `logit_init=10` freeze masks ≈1.0, so the controller is
> unused; reg_weight only seeds it). Resubmitted as `48355743`/`48355745`.

## Track B — VLM (LLaVA-1.5: frozen encoder → projector → Qwen2.5-7B)

**Stage 1 — connector pretrain (projector only, LCS-558K): ✅ both done**

| Encoder | Loss start→end | Steps | Wall | Output |
|---------|:--------------:|:-----:|:----:|--------|
| radio-l (RADIO)   | 8.70 → **2.23** | 2180 | 11h03m | `results/vlm/radio-l_pretrain/projector.pt` |
| dinov3-b (DINOv3) | 8.66 → **2.75** | 2180 | 9h59m  | `results/vlm/dinov3-b_pretrain/projector.pt` |

radio-l is the stronger encoder at stage 1 (~0.5 nats lower).

**Stage 2 — visual instruction tuning (projector + LoRA, LLaVA-Instruct-150K):**
✅ both done — radio-l = 48312695 → `results/vlm/radio-l_finetune/`,
dinov3-b = 48312696 → `results/vlm/dinov3-b_finetune/` (`llm_merged/` + `projector.pt`).
These are cell **A1** (generic stage-2, no waste-SFT yet).

**Third encoder:** `vlm_pre_cradiov4h` (48237602) — stage-1 pretrain, now
**RUNNING** (landed 2026-07-03, ~11h in of 24h limit).

### Waste-benchmark eval of A1 (2026-07-04) — micro-F1

Two-turn CoT protocol, same as the Qwen/InternVL baselines. Three parsers reported
(kw-bag = keyword substring, negation-blind; LLM-judge = negation-aware, via `claude`
CLI; closed_vocab = forced-choice, single-turn — the same regime as the baseline table).

| encoder | parser | dw_paper10 | aw_m2 | aw_m4 |
|---------|--------|:----------:|:-----:|:-----:|
| dinov3-b | kw-bag        | 0.101 | 0.126 | 0.250 |
| dinov3-b | LLM-judge     | 0.120 | 0.173 | 0.212 |
| dinov3-b | closed_vocab  | 0.062 | 0.219 | 0.165 |
| **radio-l** | kw-bag     | 0.126 | 0.174 | **0.310** |
| **radio-l** | LLM-judge  | ⚠️ n/a | ⚠️ n/a | ⚠️ n/a |
| **radio-l** | closed_vocab | 0.062 | **0.293** | 0.186 |
| _dino.txt (base, e2e)_ | _closed_vocab_ | _0.40_ | _0.37_ | _0.38_ |
| _InternVL3-8B_ | _closed_vocab_ | _0.368_ | _0.452_ | _0.304_ |
| _Qwen2.5-VL-7B_ | _closed_vocab_ | _0.239_ | _0.375_ | _0.266_ |

**Reading it (this is the supervisor-facing takeaway):**
- **radio-l ≥ dinov3-b in every valid cell** (aw_m2 0.293 vs 0.219; aw_m4 0.310 vs
  0.250 kw-bag) with *everything else held fixed* (same LLM, projector recipe, data —
  only the encoder swapped). This is the controlled test of "better visual features
  help," and it holds. It agrees with the frozen-feature probe (already positive in
  prior work): radio separates waste classes better than dino at the representation
  level.
- **But the advantage does not fully transfer through the generative path.** On
  `dw_paper10` both encoders are *identical* (0.062, precision 0.032, **recall ≈1.0**) —
  the model predicts nearly every class regardless of the encoder → the visual signal
  isn't reaching the decision there. This is a projector/SFT bottleneck (A1 has **no
  waste-SFT**), not an encoder failure. radio only pulls ahead on AerialWaste, where the
  task is enough in-distribution for the features to leak through.
- **All A1 cells sit well below the baselines** (best 0.29 vs InternVL 0.45). Expected:
  A1 is projector-only + generic stage-2, no waste-domain SFT. This is the confound —
  the baselines are full multimodal-SFT VLMs — so it is **not** evidence against the
  feature-quality thesis; it locates the gap on the training-data/SFT axis.
- **Parser regime matters:** closed_vocab helps radio on aerial (aw_m2 0.174→0.293) but
  collapses on dw (over-prediction, recall 1.0). kw-bag and LLM-judge track each other.

**Caveats / provenance:**
- **radio-l LLM-judge cells are missing, not zero.** They show 0.0000 = `claude` CLI
  quota fallback (recall=0 across all records → empty predictions), triggered by the
  account session limit. Re-run after reset via `$CLAUDE_JOB_DIR/tmp/run_reparse_retry.sh`
  (dinov3-b/dw already recovered → 0.120). kw-bag and closed_vocab are unaffected/valid.
- **Chat template verified correct** (asked during this session): decoder is Qwen2.5 →
  ChatML, and eval `generate()` matches training `encode_messages` byte-for-byte
  (system/user/assistant, same system prompt, same `<image>`→IMAGE_TOKEN splice). No
  Vicuna involved. The only train/eval gap is the two-turn CoT being off-*distribution*
  (model trained single-turn), not off-template.

## Infrastructure (this session)

- Waste eval env switched `gausdino` → **`myenv`** (matching tokenizers +
  registers the pruned `custom-llama3` arch). Added `src/eval/_arch.py`
  (`register_pruning_arch`) wired into `mc_score._load` / `ppl_eval._load` so
  materialized-pruned checkpoints load via AutoModel. **`gausdino` env deleted.**
- `slurm_cpt_waste.sh`: `MODEL_PATH` now overridable → enables post-prune
  recovery CPT on a pruned checkpoint.
- **VLM waste-eval of trained models enabled** (this session): `WasteVLM.generate()`
  (greedy, ChatML) + `WasteVLMAdapter` in `src/vlm_eval.py` + `slurm_vlm_eval_trained.sh`
  (args: `ENCODER CKPT DATASET LIMIT PROMPT_STYLE`). Made det/seg imports lazy so
  classify runs without pycocotools. Recovered AerialWaste MCML splits (aw_m2/aw_m4).
- **Phase-0 waste-SFT data built**: `data/waste_sft/train.json` = 46,713 records
  (leakage-checked vs dw/aw). See `data/waste_sft/stats.md`.
- **Two CPT infra bugs fixed** (jobs resubmitted `48424050` recovery / `48424051` base,
  both PENDING — fix unverified until they schedule): (1) trainer now always builds a PI
  SparsityController seeded from `--reg_weight` (rejects ≤0) → set `0.0`→`0.05` (inert);
  (2) DDP + gradient-checkpointing "marked ready twice" on mask params → conditional
  `static_graph=True` (no layer-drop) in `pruning/pruning_llama3_pretrain.py` ~L268.

## Open decisions / next experiments

1. **Recovery is the clear win** — `cpt_recover_prune45` (running) tests how much
   of the 13.6 corpus-PPL heals back toward 7.1. Re-eval on the waste benchmark
   after. If it recovers well, then push sparsity.
2. **Prune from the CPT'd model, not base** — once `cpt_waste` lands, re-prune the
   domain-adapted model for a better PPL floor.
3. **Gradual/iterative sparsity** for high targets (0.5→0.9 curriculum) instead of
   one-shot 0.9.
4. **VLM** — decide whether to carry both encoders forward or drop dinov3-b early;
   waste-eval both after finetune; `cradiov4h` still needs to start.

## Agreed next direction (2026-07-04): Pruned-Decoder VLM (PLAN.md B-row)

Build a VLM on the **pruned + recovered ts0.5 llama-3.1-8b decoder** (~45% sparse) and
eval on dw/aw. Headline claim = **iso-accuracy at lower cost** (B2 ≥ A2 at fewer
params/VRAM/latency), *not* beating InternVL/Qwen-VL on raw F1 — absolute accuracy comes
from Phase-3 waste-SFT. Whole B-path is **gated on the recovery CPT** (`48424050`).
Cold-start handoff + concrete steps: **`NEXT_STEPS.md`**.
