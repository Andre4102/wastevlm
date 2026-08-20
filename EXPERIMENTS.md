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

## Modern-recipe VLM — waste-benchmark eval (2026-08-11)

**Model.** `cradiov4-so_r768ps2`: frozen C-RADIOv4-SO400M @768 → pixel-shuffle 2
(576 visual tokens) → projector → Qwen2.5-7B. Stage 1 = 2000 steps on the
ShareGPT4V+PixMo alignment mix; stage 2 = **LLaVA-NeXT-760K + Vision-Flan (~819K),
1 epoch, 6081 steps** (`finetune_next`, chain finished 2026-07-22, final loss 0.82).
No waste-SFT anywhere — this is a zero-shot transfer number.

**Grid.** 3 benchmarks × {closed_vocab, open_cot}, plus the earlier 150K-SFT
checkpoint on AerialWaste as a stage-2 **data-scale ablation** (150K vs 819K,
everything else fixed). Micro-F1, `labels/img` = mean predicted labels per image:

| stage-2 data | parser | dw_paper10 | aw_m2 | aw_m4 |
|---|---|:---:|:---:|:---:|
| 150K (LLaVA-Instruct) | closed_vocab | 0.060 | 0.289 | 0.166 |
| **819K (NeXT+Vision-Flan)** | closed_vocab | **0.310** | 0.097 | 0.161 |
| 150K | open_cot (kw-bag) | 0.106 | — | — |
| 819K | open_cot (kw-bag) | 0.258 | 0.008 | 0.025 |
| _dino.txt (base, e2e)_ | _closed_vocab_ | _0.40_ | _0.37_ | _0.38_ |
| _InternVL3-8B_ | _closed_vocab_ | _0.368_ | _0.452_ | _0.304_ |
| _Qwen2.5-VL-7B_ | _closed_vocab_ | _0.239_ | _0.375_ | _0.266_ |

**Reading it.**
- **DroneWaste: 0.060 → 0.310 (5×) from stage-2 data scale alone.** This is the
  first cell of ours that is competitive with the full-SFT baselines — above
  Qwen2.5-VL-7B (0.239), within reach of InternVL3-8B (0.368) and dino.txt (0.40),
  with **no waste-SFT**. Binary waste/no-waste on dw: F1 0.670 (P 0.562 / R 0.829).
- **AerialWaste went the other way** (aw_m2 0.289 → 0.097; aw_m4 flat 0.166 → 0.161).
  The scale-up helps drone-altitude imagery and hurts satellite/aerial-altitude
  imagery — the SFT mix is natural-image-heavy, and dw is much closer to that
  distribution than AW's Google-Earth/AGEA tiles.
- **The failure mode inverted, and that is the real story.** The 150K model
  over-predicted (aw_m4: 5.23 labels/img, recall 0.905, precision 0.091 — the old
  "say yes to everything" regime); the 819K model **under-predicts** (aw_m4: 0.22
  labels/img; answers literally `none` on 457/581 AW images, 1072/1504 dw images).
  Precision is now the strong side (dw open_cot P=0.714, aw_m2 closed P=0.490).
  Formatting is not the problem — closed_vocab replies copy the label strings
  exactly; the model simply asserts nothing. Micro-F1 alone cannot see this, which
  is why `labels_per_image` + `binary_presence` are now in every report.
- **CoT hurts this model** (dw 0.310 → 0.258): turn-1 descriptions come back
  terse and generic ("The image is of a green field.", 1504/1504 non-empty but
  contentless), so turn 2 has nothing to classify from and answers `none`
  (turn-2 non-empty: dw 146/1504, aw 12–16/581). Short-answer bias from the
  Vision-Flan / LLaVA-NeXT mix is the likely cause.

**Where the numbers live.** `results/vlm_eval/vlm_cradiov4-so_r768ps2_<stage>_<dataset>_<style>/`
(`test_eval.json` + `raw_responses.jsonl`). Eval output dirs now carry the ckpt
**stage** (`slurm_vlm_eval_trained.sh`); before this they did not, so a second
stage silently overwrote the first stage's results.

**Next.** (a) waste-SFT on top of `finetune_next` — the model is precise but
mute, which is exactly what in-domain SFT should fix; (b) an AerialWaste-specific
look at whether the loss is resolution/altitude or vocabulary; (c) LLM-judge
re-parse of the open_cot runs (`src/reparse_llm.py --backend api`).

---

## Token geometry: resolution and pixel-shuffle (2026-08-12)

Follow-up to (b) above — is AerialWaste's weakness a *resolution* problem? Three
arms, **stage-2 held at LLaVA-Instruct-150K** so only token geometry varies:

| arm | encoder input | pixel-shuffle | visual tokens | stage 1 | stage 2 |
|---|---:|---:|---:|---|---|
| A (baseline) | 768 | 2 | 576 | reused | 5h31m |
| **C** | **1024** | 2 | 1024 | reused (ps unchanged) | 8h54m |
| B | 768 | **1** | 2304 | from scratch, ~29h | pending |

Stage 1 only reruns when `PSHUF` changes: the projector is
`patch_dim * PSHUF^2 -> hidden` applied per token, so it is independent of
`IMG_SIZE`. Arm C therefore reused the 768/ps2 `projector.pt` and went straight
to stage 2; Arm B needs a fresh stage 1.

### Arm C — training at 1024 does not help

Paired bootstrap over images, 2000 resamples (`scripts/paired_bootstrap.py`),
micro-F1, `*` = 95% CI excludes zero:

| cell | 768/ps2 | 1024/ps2 | Δ | 95% CI |
|---|---:|---:|---:|---|
| aw_m2 closed_vocab | **0.289** | 0.101 | **−0.188** | [−0.249, −0.129] * |
| aw_m2 open_cot | 0.191 | 0.206 | +0.015 | [−0.034, +0.062] |
| aw_m4 closed_vocab | 0.166 | 0.172 | +0.006 | [−0.000, +0.012] |
| aw_m4 open_cot | 0.248 | 0.268 | +0.019 | [−0.010, +0.049] |
| dw closed_vocab | 0.060 | 0.060 | +0.000 | [−0.002, +0.002] |
| dw open_cot | 0.106 | 0.109 | +0.002 | [−0.016, +0.022] |

**One significant delta out of six, and it is negative.** The extra pixels at
1024 are *real* on AerialWaste (native modal 1000x1000, previously downsampled
0.77x) and the model still cannot use them. This closes the resolution branch
for AW: the earlier zero-training 1024 probe showed the same losses, and the
hypothesis that those were a train/test resolution shift is now falsified —
training at the target geometry does not recover them.

**The 0.289 baseline is a refusal artifact, not a detection result.** On that
cell the model emits an empty parse on **505/581 images (86.9%)** and scores
0.289 off the 76 it answers, by being precise when it speaks. Arm C simply
refuses harder (96.6% empty, 20 images), so the −0.188 is mostly a refusal
delta: `binary_presence` F1 0.411 -> 0.188 is driven entirely by recall
(0.291 -> 0.104) while precision *rises* (0.697 -> 0.950). Any future "best AW
score" claim has to be read next to its empty-parse rate.

**CoT helps on AerialWaste, for both arms** — the opposite of the dw finding
above: aw_m4 0.166 -> 0.248 (baseline) and 0.172 -> 0.268 (Arm C); aw_m2 gains
only on Arm C. The mechanism is refusal, not reasoning: open_cot cuts AW empty
parses from 86.9% to 34.3% (aw_m2, baseline). Prompt style moves AW numbers
further than a 78% increase in visual tokens does.

**Still open.** Arm B (ps1, 2304 tokens) is the stronger version of the same
lever — 4x the token grid at identical pixels, so it tests spatial
*addressability* rather than raw resolution. Object-scale measurements predict
it should matter most on AW, where 51-54% of annotated objects are sub-token at
ps2. Arm C is weak evidence against, but does not settle it.

### Arm B — pixel-shuffle 1 (2304 tokens) — and the control that reframes all of it

Arm B trained clean (1232 steps, 12h51m, 37.4s/step, final loss 1.157 — the same
loss as Arm C's 1.158 and within noise of the baseline, despite a 4x spread in
visual tokens). Paired bootstrap vs 768/ps2:

| cell | 768/ps2 | 768/ps1 | Δ | 95% CI |
|---|---:|---:|---:|---|
| aw_m2 closed_vocab | 0.289 | 0.004 | −0.285 | [−0.342, −0.225] * |
| aw_m2 open_cot | 0.191 | 0.192 | +0.000 | [−0.042, +0.044] |
| aw_m4 closed_vocab | 0.166 | 0.142 | −0.024 | [−0.081, +0.034] |
| aw_m4 open_cot | 0.248 | 0.255 | +0.006 | [−0.017, +0.028] |
| dw closed_vocab | 0.060 | 0.067 | +0.006 | [+0.003, +0.010] * |
| dw open_cot | 0.106 | 0.098 | −0.009 | [−0.030, +0.012] |

**The object-scale hypothesis is falsified.** 4x the token grid at *identical*
pixels — the clean test of spatial addressability, which the sub-token
measurements predicted should help AerialWaste specifically — produces no gain
anywhere. The only positive significant delta is dw closed_vocab at +0.006.

**Why the closed_vocab deltas are not measurements.** Counting distinct predicted
label sets over 581 images:

| cell | arm | unique sets | modal share |
|---|---|---:|---:|
| aw_m4 closed_vocab | 768/ps2 | 5 | 51.5% |
| aw_m4 closed_vocab | 768/ps1 | 4 | 95.7% |
| aw_m4 open_cot | 768/ps2 | 26 | 19.1% |
| aw_m4 open_cot | 768/ps1 | 28 | 27.4% |

Under closed_vocab the models emit a near-constant string — the ps2 baseline
answers `Rubble/excavated earth and rocks, Scrap, Sludge-Zootechnical` on 302/581
images; Arm B answers `none` on 556/581. These are two degenerate operating
points, and the −0.285 between them is a behaviour swing, not a difference in
perception. **Under open_cot, where predictions are genuinely image-conditioned,
all three arms are statistically indistinguishable on all three benchmarks.**

**The control I should have run first.** Micro-F1 on a skewed multi-label prior
has a high floor. The best *image-independent* predictor — one fixed label set on
every image, pixels ignored entirely:

| benchmark | best constant micro-F1 | best arm we have |
|---|---:|---:|
| aw_m2 | **0.307** (`Bulky items`+`Unknown material`) | 0.289 |
| aw_m4 | **0.359** (`Other waste`) | 0.268 |
| dw_paper10 | 0.093 (`Mixed items`+`Scrap`) | 0.310 (819K arm) |

**No arm of ours beats the constant predictor on AerialWaste.** Not the 0.289
that this whole branch was launched to beat, not Arm B, not Arm C. The published
reference numbers do clear it (Qwen2.5-VL 0.375, dino.txt 0.37, InternVL3-8B
0.452), so the benchmark is not degenerate — our AW operating point is. Every AW
comparison in the section above is therefore noise around a sub-trivial baseline,
and the token-geometry arms were never capable of resolving it. DroneWaste is
unaffected: 0.310 clears its 0.093 floor by a wide margin, and that result stands.

`src/vlm_eval.py` now reports `constant_baseline` and `prediction_diversity` in
every run and prints `<-- AT OR BELOW CONSTANT BASELINE` when the score does not
clear the floor, so this cannot go unnoticed again.

**What this closes and what it opens.** Closed: resolution and pixel-shuffle as
levers for AerialWaste — three token geometries spanning 576-2304 tokens, all
null under the only prompt style that produces image-conditioned answers. Open:
AerialWaste needs the model to *answer at all* in a discriminative way, which is
a refusal/calibration problem, not a perception-budget one. In-domain waste-SFT
is the lever that addresses that; token geometry is not.

---

## Why AerialWaste is bad: the decomposition (2026-08-14)

The constant-baseline control said *that* our AW numbers are worthless; it did not
say *why*. This section decomposes it. Everything below reproduces with
`scripts/aw_diagnose.py --gt --noise --runs` (see LAUNCHERS.md §3).

### The test split is mostly negatives, and micro-F1 scores them

| split | images | with any GT label | labels/image (positives) |
|---|---:|---:|---:|
| aw_m2 | 581 | 182 (31.3%) | 2.56 |
| aw_m4 | 581 | 172 (29.6%) | 1.78 |
| dw_paper10 | 1504 | 355 (23.6%) | 1.84 |

So ~70% of AW images are "say nothing" images, and every label emitted on one is a
pure false positive. Note DW is *more* negative-heavy than AW, so this alone is not
the difference.

### Split the score into detection and naming

For each run: *detection* = did the model emit any label, vs did the image have any
GT label. *naming* = micro-F1 over GT-positive images the model answered.
*Oracle-gated* = our own predictions with every false alarm deleted (identical to
scoring on the positives-only split the dataset ships and nothing uses).

| run | micro-F1 | detection P / R / F1 | false alarms | oracle-gated | naming on answered gt+ |
|---|---:|---|---:|---:|---:|
| radio-l aw_m4 open_cot | 0.310 | .314 / .808 / .453 | 303/409 | 0.527 | 0.602 |
| 1024ps2 aw_m4 open_cot | 0.268 | .338 / .773 / .470 | 261/409 | 0.504 | 0.556 |
| 768ps2 aw_m4 open_cot | 0.248 | .315 / .971 / .476 | 363/409 | 0.566 | 0.572 |
| 768ps2 aw_m2 closed_vocab | 0.289 | .697 / .291 / .411 | 23/399 | 0.320 | 0.602 |
| **819K dw closed_vocab** | **0.310** | **.562 / .829 / .670** | **189/1211** | 0.391 | 0.427 |

**Detection is the whole gap.** Delete the false alarms and aw_m4 goes
0.248 → 0.566. The model's decision to answer carries almost no signal on AW —
it answers on 77% of positives and 64% of negatives (768ps2: 97% vs 89%) — while
on DW the same decision separates 83% from 16%. That single contrast is the
result.

The 0.289 aw_m2 arm is the opposite failure of the same kind: it barely answers
(detection recall 0.291, 76 of 581 images) and buys precision with silence.

### Naming is not the problem, but it is not informative either

Absolute naming quality on AW (0.556-0.602) is *better* than on DW (0.427). But
relative to the prior it is worse: with a perfect detector, a constant
`Other waste`+`Rubble/excavated earth and rocks` scores **0.732**, above our
best oracle-gated 0.566. On DW the reverse holds — ours 0.391 beats constant
0.345. AW's taxonomy is prior-dominated: `Other waste` is 52% of aw_m4 label mass
and `Unknown material` 26% of aw_m2's, i.e. two of the biggest classes literally
mean "not identifiable".

### Why the domain is hard — and why resolution was still the wrong lever

AW is nadir satellite/aerial at 0.199-0.301 m GSD, 1000px tiles downsampled 0.77x
to 768; median annotated object is **0.92 tokens**, 53% are sub-token, 89% under 4
tokens. DW is 640px drone imagery upsampled 1.2x, median object 4.4 tokens. Our
whole SFT stack (LLaVA-150K/NeXT, PixMo, COCO, Vision-Flan) is ground-level
natural images.

That sub-token statistic is what motivated the resolution branch — and the branch
came back null at 4x tokens (Arm B) and at native resolution (Arm C). Both facts
are true: the objects are tiny *and* giving the model more perception budget
changes nothing, because the binding constraint is downstream, in the decision to
answer.

### Label noise: real, small, and not the explanation

All 409 aw_m4 empty-GT images have `valid_fine_grain = 0`, but 22 of them carry a
real `site_type`/`severity`/`evidence` — genuine waste sites with no usable
fine-grained labels, scored as negatives. The model answers on them at the same
rate as on true background (50% vs 65%, 77% vs 74%, 82% vs 89% across three arms),
so it is not being punished for being right; it is not discriminating at all.
5% mislabelling does not move a 0.25.

### Correction to the reference row

"The published references clear the constant floor" is true for aw_m2 (floor
0.307: dino.txt 0.37, Qwen2.5-VL 0.375, InternVL3-8B 0.452) but **false for
aw_m4** (floor 0.359: InternVL3-8B 0.304 and Qwen2.5-VL 0.266 are both *below*
it; only dino.txt 0.38 clears). aw_m4 as scored on the full split is a benchmark
almost nothing beats trivially. dino.txt clearing both floors on our own harness
and split is the useful existence proof — and it is a discriminative model with
per-class thresholds, i.e. it has exactly the calibrated decision our generative
path lacks.

### What to do

1. **In-domain AW SFT with negatives.** The train split has 3689 on-disk images at
   31% positive — the right supervision for "no waste here" on nadir imagery,
   which no general-purpose mix provides. This is the single highest-value change.
2. **Report positives-only alongside the full split.** The dataset ships
   `only_pos` (217/204 test images) and no eval of ours uses it. It separates
   naming from detection by construction.
3. **Give the decision a threshold**, not a sampled token — score the closed
   vocabulary by per-class likelihood and threshold it, the way dino.txt does.

### The cause: the SFT task never has "nothing" as an answer

The decomposition above says the failure is the decision to answer. This says why
that decision is broken, and it is a property of the stage-2 data, not the encoder.

**The captions are unconditioned on the image.** `open_cot` turn 1 asks for a
description; the keyword parser then reads waste terms out of it. Rate of each term
on GT-positive vs GT-negative images (`scripts/aw_diagnose.py --captions`):

| arm / benchmark | `pile` gt+ / gt- | `debris` gt+ / gt- | `construction` gt+ / gt- |
|---|---|---|---|
| 150K, aw_m4 | 100.0% / **99.8%** | 84.9% / **85.1%** | 95.9% / **96.1%** |
| 150K, dw | 100.0% / **97.1%** | 84.6% / 82.3% | 84.6% / 88.5% |
| **819K, dw** | **60.1% / 6.0%** | **30.4% / 5.1%** | 2.0% / 0.2% |
| 819K, aw_m4 | 3.5% / 0.5% | 1.7% / 0.0% | 5.2% / 0.0% |

The 150K arm asserts *piles, debris and construction on essentially every image it
is shown*, positive or negative, on both benchmarks. 20% of its AW captions open
with the same twelve words ("the aerial drone photograph shows a rural area with
various materials, objects, ..."). It is not describing these images; it is
emitting a caption template, and the parser dutifully converts the template into
waste labels. That is the false-alarm epidemic, and it explains why the constant
baseline is unbeatable: **the model was itself behaving as a constant predictor.**

The 819K arm on DroneWaste is the control that makes this causal: same encoder,
same projector recipe, same prompt, same parser — only the stage-2 mix scaled —
and the caption profile separates (60.1% vs 6.0% on `pile`). That is exactly the
arm with detection F1 0.670 and micro-F1 0.310. **Caption conditioning and
detection quality move together.**

On AerialWaste the 819K arm swings to the opposite failure: it almost never
mentions waste-like content at all (`pile` 3.5% / 0.5%), so it is mute rather than
indiscriminate, and scores 0.025. Scaling general-purpose SFT fixed the "narrate
piles unconditionally" prior and replaced it, on nadir imagery, with "nothing to
report" — because nothing in the mix looks like a waste pile at 0.21 m GSD.

**And the SFT task itself never teaches abstention.** Share of assistant turns
that assert the absence of something (first 200K turns of each mix):

| mix | turns asserting absence |
|---|---:|
| LLaVA-Instruct-150K | 1.56% |
| 819K NeXT + Vision-Flan | 0.18% |

and most of even those are incidental clauses ("without any other vehicles
visible"), not an answer of *nothing*. The stage-2 distribution is one where every
image has a subject worth naming and every question has a positive answer. AW is
70% images whose correct answer is "nothing". The model has never once been
rewarded for saying it.

### How much the known parser leak costs (measured, and it is *not* the explanation)

`parse_keywords` being negation-blind is documented above in the A1 section — that
is why the LLM-judge parser exists. This quantifies it for the first time. A
response reading "there is **no** clear indication of rubble, scrap metal, manure,
wood waste, discarded tires or mixed waste" emits **all six** labels as positives.
The `Rubble/excavated earth and rocks` bag also contains `soil`, `earth`, `rock`,
`sand`, and aw_m2's `Rubble` contains `construction`/`demolition` — terms present
in almost any aerial scene description.

Both are real and worth fixing, but re-parsing with a negation-aware, clause-scoped
matcher (and again with the generic terrain words dropped) does **not** rescue the
result:

| aw_m4, 768ps2 open_cot | micro-F1 | fp | false alarms |
|---|---:|---:|---:|
| as scored | 0.2484 | 1154 | 363/409 |
| negation-aware | 0.2857 | 744 | **359/409** |
| + terrain words dropped | 0.2718 | 597 | 309/409 |

False positives nearly halve, but the model still *asserts* waste on 88% of the
negatives, because a negated sentence almost always sits next to a non-negated
assertion. On aw_m2 the reparse makes the score *worse* (0.191 → 0.121 → 0.082):
the substring parser was collecting true positives by accident too. Conclusion:
the parser leaks, the model is unconditioned, and only the second one is the
result. Fixing the parser is worth doing on its own merits — validated against the
hand-labelled agreement sample, not against F1 — but it is not the fix for AW.

### Revised conclusion

AerialWaste does not fail on perception (three token geometries, all null), on
label noise (5% of negatives, and the model does not exploit it), or on parsing
(halving the FPs moves nothing). It fails because **stage-2 SFT teaches the model
that every image contains nameable objects**, so it either narrates waste
unconditionally (150K) or, once better conditioned on natural images, sees nothing
it recognises at 0.21 m GSD (819K). The lever is stage-2 data that contains nadir
imagery *and* negative answers — i.e. AW's own train split, 3689 on-disk images at
31% positive, where the majority target is "no waste".

### Where the signal is lost: frozen-feature probe (2026-08-14, job 52297931)

If the failure is the binary decision, the question that decides what to build next
is whether the encoder represents that decision at all. Linear probe on the frozen
**C-RADIOv4-SO400M** features — same 768px preprocessing, same train/test images,
same binary waste/no-waste target as the VLM, logistic regression on top
(`scripts/aw_feature_probe.py`, 48 min on 1 GPU):

| split | pooling | AUC | Youden J | TPR@J | FPR@J |
|---|---|---:|---:|---:|---:|
| aw_m4 | **cls** | **0.9662** | **0.837** | 0.930 | 0.093 |
| aw_m4 | mean patches | 0.9381 | 0.772 | 0.890 | 0.117 |
| aw_m4 | max patches | 0.9145 | 0.681 | 0.959 | 0.279 |
| aw_m2 | **cls** | **0.9696** | **0.853** | 0.956 | 0.103 |
| aw_m2 | mean patches | 0.9436 | 0.774 | 0.885 | 0.110 |
| aw_m2 | max patches | 0.9215 | 0.687 | 0.951 | 0.263 |

Against **J = 0.139 (aw_m4) / 0.234 (aw_m2) for the best of 13 full-VLM eval runs**
on the identical decision. A logistic regression on frozen features nearly solves
what the projector → LLM → sampled-token path fails at. AerialWaste waste/no-waste
is *linearly separable* in the encoder we are already using.

Shortcut check: positive rate is 30.8% / 29.8% / 33.5% across AGEA / GE / WV3 and
29.8% / 31.2% by image size, so the probe is not reading sensor or resolution
identity off the features.

**This settles the alignment-vs-instruction-tuning question.** The alignment stage
and the encoder are not the problem — the information arrives and is discarded
downstream. It also explains, after the fact, why three token geometries were all
null: the branch was widening a pipe that was already delivering.

Two caveats worth carrying:
- At 0.966 AUC on a site-level task the probe is plausibly separating "industrial /
  degraded / disturbed site" from "ordinary countryside" rather than "waste" as
  such. For AW that is arguably the intended task (the labels *are* site-level
  inspection records), but it bounds the **detection** half only, not fine-grained
  naming.
- `cls` is the best pooling, and `src/vlm_model.py:326,328` feeds `.patches` to the
  projector — **the summary token never reaches the LLM**. That is LLaVA-standard,
  but on a task whose signal is a global scene property it drops the single most
  discriminative feature at the door. Prepending the projected summary token is a
  one-line change and a cheap arm. Do not over-read it though: mean-over-patches
  still reaches J=0.772 and the LLM sees all 576 patch tokens, which is strictly
  more information than their mean. The signal is reachable from what the LLM
  already receives; it is not being used. Stage-2 supervision stays the primary
  lever, summary token a cheap secondary.

### Direction, with the evidence attached

| direction | verdict | evidence |
|---|---|---|
| stage-2 dataset with abstention | **primary** | 1.56% / 0.18% of SFT turns assert absence; AW train = 3689 imgs @31% pos |
| prompt style / decision threshold | real but bounded | closed_vocab cuts false alarms 65%→5.8%, but best AW J=0.234 vs dw 0.673 |
| alignment / encoder redesign | **not indicated** | frozen-feature probe AUC 0.966-0.970, J 0.837-0.853 |
| token geometry (resolution, pixel-shuffle) | **closed** | 3 geometries 576-2304 tokens, all null |
| negation-aware parser | worth fixing, not the fix | halves FPs, false alarms 363→359/409 |
| feed the summary token to the LLM | cheap secondary arm | cls J=0.837 > mean 0.772 > max 0.681; patches-only today |

Trap to design against: a 69%-negative SFT set is exactly what teaches a model to
answer "none" unconditionally — the 819K arm's AW failure mode (`pile` on 3.5% of
positives, micro 0.025). Both degenerate attractors are reachable, so the mix ratio
and scoring *both* error directions matter more than total size.

## Arm "pilot": the RS SFT injection, measured (2026-08-15, jobs 52364030 / 52408835-40)

First test of the abstention-SFT design. Mix = LLaVA-150K x1 + `rs_sft_p2` x3
(320,828 records, 2506 steps, 10h26 on 4x A100, loss 2.13 -> 1.00, cosine run to
term). Evaluated zero-shot on all three benchmarks in both prompt styles.

**The matched control is the 150K arm (`_finetune_`), not 819K (`_finetune_next_`).**
The pilot's general component is 150K, so comparing it against 819K would conflate
the injection with a base-mix swap. Both are shown; only the 150K column is the
controlled contrast.

### Detection (Youden J = TPR - FPR)

| eval | 150K control | **pilot (+rs_sft x3)** | 819K | encoder ceiling |
|---|---|---|---|---|
| aw_m2 closed_vocab | **0.233** | -0.003 | 0.200 | 0.853 |
| aw_m4 closed_vocab | -0.008 | 0.006 | 0.139 | 0.837 |
| aw_m4 open_cot | 0.083 | 0.000 | 0.062 | 0.837 |
| **dw_paper10 closed_vocab** | 0.000 | **0.292** | 0.673 | — |
| dw_paper10 open_cot | — | 0.119 | — | — |

### It flipped the degeneracy instead of removing it

The control was a constant-YES predictor: on DroneWaste it answered on all 293
positives *and* all 1211 negatives (TPR 1.000, FPR 1.000, J exactly 0.000); on
aw_m4 closed_vocab, 393 false alarms out of 409 negatives. The pilot is a
constant-NO predictor on AerialWaste: 0/182 positives on aw_m2 (both styles),
1/172 on aw_m4. Two degenerate attractors, opposite signs, J ~ 0 either way.

This is exactly the trap written down at the end of the previous section before
the arm was built. Writing it down did not prevent it: the design's four negative
types and 45% `absent_category` share were chosen without a dose control.

### Where it genuinely worked

DroneWaste closed_vocab, J 0.000 -> 0.292, detection precision 0.195 -> 0.798,
micro-F1 0.0603 -> 0.1910. That is a real decision boundary where the control had
none. Naming on answered positives also holds up (0.369, and 0.508 on open_cot,
vs the control's 0.281). Still below the 819K arm's J 0.673.

### The asymmetry is the mechanism

AerialWaste is nadir, high-altitude imagery -- what DIOR and VRSBench look like.
The abstention prior transferred most strongly to the eval domain that most
resembles the SFT source, and AerialWaste's diffuse low-contrast piles are
precisely the case the `not_determinable` rule was written to refuse. DroneWaste's
low-altitude oblique views are less similar and kept the discrimination benefit
without the silence. Collapse tracks domain similarity to the injection.

### Dose, and why this does not directly predict a1

| arm | RS record share | RS answer-token share |
|---|---|---|
| pilot | **50.8%** | 6.4% |
| a1 | **17.3%** | 1.5% |

At x3 against the small 150K base the injection stopped being an injection and
became half the training records. Against 819K it is 17%.

**6.4% of answer tokens flipped the entire decision policy.** That is direct
support for the composition note in `scripts/build_train_mix.py`: for teaching a
decision policy, record share governs and token share badly understates the
effect. Any future mix must be specified in record share.

### What this changes

| lever | status after the pilot |
|---|---|
| abstention SFT as a concept | **not refuted** — produced real discrimination on dw where the control had none |
| x3 upsampling on a small base | **refuted** — 50.8% record share is a dose, not an injection |
| 45% `absent_category` + `not_determinable` | suspect on diffuse-target imagery; AW is the worst case for both |
| domain match SFT<->eval | newly implicated: collapse is strongest where the eval looks most like the SFT |

Next: a1 (819K + rs_sft x3, 17.3% record share) is the arm whose control is
J 0.139/0.200/0.673. If it also collapses, the fix is dose (`rs_sft x1`) and the
negative-type mix, not the concept.

### Stage 3 (2026-08-16, jobs 52413403/52413404, evals 52443964-78)

A third alignment pass on top of the finished 819K stage 2: warm-start from
`_finetune_next/{llm_merged,projector.pt}`, fit a fresh LoRA on `rs_sft` + 25%
general replay. Arms `s3` (72,496 recs, 566 steps, 2h21) and `s3a` (42,644 recs,
15% negatives, 333 steps, 1h24). Both ran clean at world=4, global batch 128.

Youden J against the 819K control, all six eval cells:

| eval | control | s3 | s3a |
|---|---|---:|---:|
| aw_m2 closed_vocab | **0.200** | 0.079 | 0.159 |
| aw_m2 open_cot | **0.044** | 0.011 | 0.027 |
| aw_m4 closed_vocab | **0.139** | 0.005 | 0.028 |
| aw_m4 open_cot | **0.062** | 0.012 | 0.041 |
| dw closed_vocab | **0.673** | 0.564 | 0.634 |
| dw open_cot | **0.324** | 0.194 | 0.280 |

control > s3a > s3 in 6/6. Read alone this says "sequencing changes magnitude,
not sign, and dose is the knob". **The AUC section below retracts that reading**:
the ordering is threshold displacement, not learning damage.

Caveats carried: `s3a` differs from `s3` in BOTH negative share (15% vs 50%) and
steps (333 vs 566), so it does not isolate abstention; `checkpoint-350` of s3 is
the matched-step probe (needs a merge step -- checkpoints hold `lora_adapter/`,
the eval script wants `llm_merged/`). Stage 3 also moved the PROJECTOR at 2e-4
(the finetune default) alongside LoRA at 1e-5, so "steps x LoRA LR was the dose
knob" is not established by these runs.

Naming-on-answered rises on dw open_cot (0.538 -> 0.630) but the answered set
shrinks 97 -> 58 images. That is selection, not skill; do not quote it.

### The decision is calibration, not perception (2026-08-16, jobs 52454126-52455987)

Every AW number above comes from sampled tokens through a parser, which fixes one
operating point and cannot distinguish "not represented" from "represented,
verbalised at the wrong threshold". The DW split makes that concrete: identical
weights give recall 0.83/FPR 0.16 under closed_vocab and 0.33/0.007 under
open_cot.

So score the Yes-vs-No logit margin at the first answer token and sweep it
(`scripts/vlm_binary_auc.py`). Threshold-free, parser-free:

| ckpt | eval | AUC | best J | J at margin 0 | pos margin | neg margin |
|---|---|---:|---:|---:|---:|---:|
| 819K | aw_m2 | **0.837** | 0.526 | 0.107 | -1.93 | -3.27 |
| 819K | aw_m4 | **0.829** | 0.501 | 0.114 | -1.90 | -3.24 |
| 819K | dw | **0.899** | 0.690 | 0.663 | **+1.48** | -2.15 |
| s3 | aw_m2 | 0.869 | 0.569 | **0.000** | -3.71 | -4.85 |
| s3 | dw | 0.897 | 0.678 | 0.389 | -0.80 | -4.04 |
| s3a | aw_m2 | 0.793 | 0.458 | 0.071 | -2.45 | -3.56 |

AUC 0.84 on AerialWaste -- nowhere near chance. The LLM ranks AW images almost as
well as drone images (0.90). What differs is where zero sits: DW positives are at
+1.48 (above the cut, so the model speaks), AW positives at -1.93 (below it, so
both classes read as "No"). The whole AW distribution is ~3.4 logits low.

**This retracts the stage-3 conclusion.** s3 on aw_m2 has AUC 0.869 vs the
control's 0.837 -- a slightly BETTER representation -- while scoring J=0.000
because it says No to all 581 images. On dw, s3 and control are identical in AUC
(0.897 vs 0.899) while spoken J falls 0.663 -> 0.389. The measured "dose-response
damage" was threshold displacement. `s3a` is the one genuine regression
(AUC 0.793 < 0.837).

Threshold fitted on the TRAIN split, applied to test untouched (the only quotable
number; an oracle cut on test is not a result):

| eval | as spoken | **train-fitted** | oracle-on-test | gap | frozen probe |
|---|---:|---:|---:|---:|---:|
| aw_m2 | 0.200 | **0.5075** (TPR 0.643, FPR 0.135) | 0.526 | +0.019 | 0.853 |
| aw_m4 | 0.139 | **0.4944** (TPR 0.680, FPR 0.186) | 0.501 | +0.006 | 0.837 |
| dw    | 0.673 | **0.6714** (TPR 0.823, FPR 0.151) | 0.690 | +0.019 | -- |

AW detection improves 2.5-3.6x with **no retraining**, and the threshold
generalises (gap <= 0.019; train AUC 0.822 vs test 0.837, no overfit). DW is the
control that makes this credible: already calibrated, so fitting a threshold
changes nothing (0.673 -> 0.671). A sweep cannot manufacture J where the model is
not broken.

Decomposition on aw_m2, in J: spoken 0.200 -> calibrated 0.508 -> probe 0.853.
Calibration recovers ~47% of the gap for free; the remaining ~53% is genuine
projector->LLM representation loss and needs the summary-token arm / stronger
nadir supervision. Note the AUC uses a plain yes/no question while J=0.200 comes
from the closed_vocab labelling prompt, so those two are not a matched protocol.

Caption-conditioning, same 819K model: "pile" in captions gt+ vs gt- is
**DW +54.0** (60.1% vs 6.0%) against **AW +2.8** (3.3% vs 0.5%). Across five
older AW configs (cradiov4 r1024ps2 / r768ps1 / r768ps2, dinov3-b, radio-l) the
same diff is -0.1 / -0.9 / +0.3 / +0.0 / -0.7 -- those arms say "pile" on ~100%
of ALL images. Both degenerate modes (always-yes, always-no) are threshold
pathologies, consistent with the AUC result.

### Training routine implied by the above (implemented 2026-08-16)

Loss is token-CE over answer strings (`vlm_data.py`), so a record's weight is its
answer length: a 3-token decision inside a mix whose captions average 185 tokens
is a rounding error. That is how a 6.4% answer-token share flipped the whole
policy. Added `--decision-loss-weight`: BCE on the Yes-vs-No margin at the first
answer token, for records carrying a `decision` field.

- one scalar per record -> immune to answer length
- BCE centres the operating point at margin 0 by construction, instead of leaving
  it wherever the answer prior lands (the measured failure)
- no new parameters; still pure LM
- `--decision-pos-weight` for negative-heavy decision sets (the 69%-negative
  "answer none" attractor)

Plumbing: `WasteVLM.forward` returns `expanded_labels` (the image marker expands
into patch tokens, so caller-side label indices do not line up with `logits`);
the answer position is derived as `first(labels != -100) - 1` because `logits[t]`
predicts `t+1`. Default weight 0, so existing arms are bit-identical. Unit-tested
with decoys at the answer index itself to pin the offset.

Multi-label, not yet built: replace "list the categories" with one binary
question per class, each with its own margin and threshold fitted on train. That
removes the parser from the causal path, gives per-class confidence, and lets
class-balanced sampling kill the "answer none" attractor. Cost is N forwards per
image at eval (AW 5-6 cheap, DW 20 ~2.5h); the image encoding is shared.

### How few labels the threshold needs (2026-08-16, `scripts/calib_budget.py`, CPU)

Fitting the cut on AW's full 3,689-image train split spends a lot of supervision
on ONE scalar and muddies the zero-shot claim. Priced offline from the per-image
margins already dumped by `vlm_binary_auc.py` -- no GPU. Draws are random, not
stratified: a real calibration set is "label the next n images", and stratifying
would assume you already know which are positive.

| n | aw_m2 median J | p10 | % of full-fit | aw_m4 | dw |
|---:|---:|---:|---:|---:|---:|
| 10 | 0.461 | 0.306 | 88% | 88% | 89% |
| 30 | 0.477 | 0.403 | 91% | 91% | 94% |
| **50** | **0.491** | **0.422** | **93%** | 93% | 95% |
| 100 | 0.494 | 0.437 | 94% | 93% | 96% |
| 400 | 0.495 | 0.427 | 94% | 93% | 98% |
| full | 0.526 | -- | 100% | 100% | 100% |

Saturates at n~50-75; 50 -> 400 labels buys ~1 point. A p10 draw at n=50 (0.422)
is still double the uncalibrated 0.200, so downside risk is small. Below n~30 the
median holds but the spread opens (p10 0.306 at n=10).

Threshold transfer, fitted on A applied to B with no in-domain labels:

| fit -> apply | J | % of own-fit |
|---|---:|---:|
| aw_m2 -> aw_m4 | 0.498 | **99.5%** |
| aw_m4 -> aw_m2 | 0.514 | **97.6%** |
| aw_m2 -> dw | 0.399 | 57.9% |
| dw -> aw_m2 | 0.160 | 30.5% |

**The cut is a property of the DOMAIN, not the label set**: a threshold fitted for
the 5-class m2 taxonomy transfers to 6-class m4 intact. Calibrate once per domain,
evaluate any taxonomy on top, and the multi-label result stays category-zero-shot.
Caveat: m2/m4 are the same images at different granularity (182 vs 172 positives),
so their binary targets are correlated -- suggestive of taxonomy-independence, not
proof. Cross-DOMAIN transfer fails (58% / 31%), consistent with the ~3-logit offset
between AW and DW margins, so "zero labels anywhere" is not available.

**Protocol to quote**: detection threshold calibrated on 50 images with binary
waste/no-waste labels only; no category labels used for fitting; all multi-label
results category-zero-shot; one scalar parameter.

### s3b: the nadir description corpus (2026-08-16, jobs 52463710/52463712, evals 52492112-147)

`nadir_desc` = VRSBench [caption] (20,264 grounded captions, mean 53 words) +
LoveDA (8,482 records from 3,035 masked tiles at ~0.3 m/px, true absence, 5,477
carrying a `decision` field). 38,328-record mix at 75% nadir / 25% general replay,
299 steps, ~75 min. Two arms differing ONLY in loss: `s3b` CE, `s3bd` CE + margin
BCE (`--decision-loss-weight 1.0`, decision loss 0.234 -> 0.075).

**Caption conditioning on AerialWaste -- the thing the corpus was built to fix:**

| arm | "pile" gt+ / gt- | diff | "debris" diff | distinct caps | modal opening |
|---|---|---:|---:|---:|---:|
| 819K control | 3.3 / 0.5 | **+2.8** | +1.6 | 330/581 | 9% |
| **s3b** | 12.6 / 0.8 | **+11.9** | +4.4 | 544/581 | 3% |
| s3bd | 4.9 / 0.5 | +4.4 | +2.8 | 511/581 | 3% |

s3b **quadruples** AW caption discrimination and breaks the template: distinct
captions 330 -> 544 of 581, modal 12-word opening 9% -> 3%. The negative rate stays
low (0.8%), so this is discrimination, not verbosity. On dw, "debris" goes
+25.3 -> +40.5 and distinct captions 900 -> 1230.

**Detection, threshold fitted on train:**

| eval | control | s3b | s3bd | | AUC control | s3b | s3bd |
|---|---:|---:|---:|---|---:|---:|---:|
| aw_m2 | **0.5075** | 0.4606 | 0.4940 | | **0.837** | 0.801 | 0.815 |
| aw_m4 | **0.4944** | 0.4323 | 0.4788 | | **0.829** | 0.791 | 0.806 |
| dw    | 0.6714 | 0.6595 | **0.6930** | | 0.899 | 0.897 | **0.908** |

Two findings, and they point opposite ways:

1. **The margin BCE works.** s3bd beats s3b on all three evals in BOTH AUC and
   calibrated J (+0.034 / +0.047 / +0.034 J), same corpus, same steps, same LR --
   only the loss differs. Cleanest controlled evidence we have for the training
   routine change. On dw, s3bd is the best model recorded: AUC 0.908, J 0.693.
2. **The corpus did NOT fix AW detection.** Both arms sit below the 819K control
   on AW AUC. Descriptions improved sharply while detection slid.

**Description and detection dissociate**, and the two arms trade off against each
other: s3b has the better captions (+11.9) and the worse detection (0.801); s3bd
the reverse (+4.4, 0.815). The margin BCE pulls capacity toward the decision and
away from description.

Note the decision loss did NOT recentre the AW operating point (J at margin 0 is
still 0.072 / 0.080). BCE centres the cut for the QUESTIONS IT TRAINED ON -- LoveDA
land cover -- and "is there waste" was never one of them. Centring does not
transfer across question semantics; only the ranking does.

**Product reading**: for an open-description demo on satellite imagery, `s3b` is
the best model we have -- 4.3x caption conditioning, template collapse fixed. For
a calibrated triage gate, the 819K control still wins on AW and `s3bd` on drone.

### Rejected calibration source: Yuan et al. 2026 global landfill segmentation

(Eng. Appl. Artif. Intell. 176:114721, doi 10.1016/j.engappai.2026.114721.)
1,380 landfills, 118 countries, six continents, Google WMTS, 2,000x2,000 px tiles
covering 16 km^2 at **2 m/px**. Proposed as a calibration set; **rejected**:

- **All 1,380 images are landfill images**, randomly split into train/val/test.
  There are no negative (no-landfill) images at all -- their "recall 99.51% for
  presence detection" has no FPR to go with it. **A threshold cannot be fitted
  without negatives**, so the dataset cannot do the one job proposed for it.
- **2 m/px vs AerialWaste's ~0.2**, and 16 km^2 per image vs AW's 0.04 km^2 --
  a 400x difference in ground coverage. Our own transfer numbers (AW->DW keeps
  31%) say a cut does not survive that kind of domain gap.
- Data is "available on request"; the GitHub repo carries code only.

Still interesting as a TRAINING corpus (globally diverse landfills with masks,
semantically on-target), but it would need the request fulfilled and a site-level
leakage check against AW -- Google-sourced imagery over 118 countries plausibly
includes Lombardy, and site overlap would not be caught by pHash at 10x GSD
difference.

### The calibrated gate is now the default eval readout (2026-08-16)

`src/vlm_calib.py` is the single source of truth for the margin readout, imported
by BOTH the calibration step (`scripts/vlm_binary_auc.py`, which now writes a
`calibration.json`) and the eval that applies it (`src/vlm_eval.py --calib`). If
the question wording, token set, or scored position ever differed between fitting
and applying, the threshold would be measured against one quantity and used on
another -- and it would fail silently, looking merely like a bad number.

Margin scoring is ON by default for models that support it (one prefill per image,
no sampling -- cheaper than the generation it accompanies); `--no-gate` opts out.
Without `--calib` only AUC is reported, because a cut fitted on the split being
scored is an oracle, not a result. Every reported J now carries `calib_meta` with
the provenance of its threshold.

Smoke (819K ckpt, aw_m2, 60 imgs, thr -2.625 from the train fit):

```
calibrated gate: AUC=0.8194  margin pos=-1.89 neg=-3.12
  thr=-2.625 -> J=0.4501 (TPR 0.736, FPR 0.286)   vs spoken J=0.2453
  micro F1 gated=0.0857 (ungated 0.0857); suppressed 0, gate-positive but unnamed 28
```

**The gate is one-directional on micro-F1** and this is worth internalising: it can
suppress labels the parser emitted, never add ones it withheld. So it pays where a
model OVER-asserts (the 150K arm, dw closed_vocab at FPR 0.156) and does nothing
where it under-asserts. On AW today the model answers empty on ~78% of images, so
gated and ungated micro-F1 coincide *while detection J nearly doubles*. The new
`n_gate_pos_parser_empty` counter sizes what a namer would have to fill: 28 of 60
images here are called waste by the gate and left blank by the parser. That number
is the AW naming gap, made countable for the first time.

## The AUC re-scoring sweep: two retractions (2026-08-17, jobs 52562245-52566514)

With the gate wired, every arm on file was re-scored through the threshold-free
readout: 8 arms x 3 benchmarks, AUC plus a threshold fitted on the TRAIN split.
This was meant to close the dose curve with `a1`. It did that, and it also
invalidated two conclusions the record was built on.

### The full table (aw_m2; AUC, fitted threshold, calibrated J, J at margin 0)

| arm | AUC | fitted thr | calib J | J@0 | spoken J |
|---|---:|---:|---:|---:|---:|
| 150K base (`_finetune_`) | 0.8615 | -4.000 | **0.6053** | 0.066 | 0.234 |
| s3 | **0.8688** | -4.498 | 0.5500 | 0.000 | 0.079 |
| pilot | 0.8436 | -5.625 | 0.5573 | 0.000 | -0.003 |
| 819K (`_finetune_next_`) | 0.8368 | -2.625 | 0.5075 | 0.107 | 0.200 |
| s3bd | 0.8148 | -3.625 | 0.4940 | 0.080 | 0.051 |
| a1 | 0.8237 | -4.617 | 0.4629 | 0.060 | 0.074 |
| s3b | 0.8006 | -3.250 | 0.4606 | 0.072 | 0.103 |
| s3a | 0.7933 | -3.249 | 0.4432 | 0.071 | 0.159 |

Spearman across the 8 arms, spoken J vs calibrated J: **rho = +0.02**. Spoken J
vs AUC: **+0.02**. Calibrated J vs AUC: **+0.91**.

**The spoken readout carries no information about arm quality.** Not inverted --
uninformative. Every arm-vs-arm decision made by reading generated text was made
by coin flip, and the two retractions below are what that cost.

### Retraction 1: the 819K base-mix swap was a regression

The 150K arm dominates the 819K arm it was replaced by, on all six calibrated
measures:

| eval | 150K AUC | 819K AUC | 150K calib J | 819K calib J |
|---|---:|---:|---:|---:|
| aw_m2 | **0.8615** | 0.8368 | **0.6053** | 0.5075 |
| aw_m4 | **0.8591** | 0.8294 | **0.5886** | 0.4944 |
| dw    | **0.9042** | 0.8987 | **0.6725** | 0.6714 |

**The 150K arm is the best model on file for waste detection**, and it has been
sitting in `results/vlm/cradiov4-so_r768ps2_finetune` since before the 819K arm
was built. It was passed over because of its spoken behaviour: on dw closed_vocab
it named a category on all 1504 images (`n_empty_parse=0`), TPR 1.000 / FPR 1.000,
J exactly 0.000 -- filed as a constant-YES degenerate. Its margin AUC on that same
split is 0.9042, and even at its natural cut J@0 is 0.666.

So the model separated DroneWaste cleanly the whole time and emitted a constant.
The degeneracy was in the verbalisation, never in the decision.

### Retraction 2: abstention SFT is a pure cost

The pilot's matched control is the 150K arm, and against it `rs_sft x3` loses on
every AW measure while wrecking speakability:

| | 150K base | pilot (+rs_sft x3) | delta |
|---|---:|---:|---:|
| aw_m2 AUC | 0.8615 | 0.8436 | **-0.018** |
| aw_m4 AUC | 0.8591 | 0.8423 | **-0.017** |
| aw_m2 calib J | 0.6053 | 0.5573 | **-0.048** |
| fitted thr | -4.000 | -5.625 | -1.63 |
| dw J@0 | 0.666 | 0.334 | **-0.332** |

Earlier today, before the 150K column existed, the pilot's AUC 0.8436 looked like
the best on file and the abstention track looked like it deserved promotion. It
does not: the base was already at 0.8615 and the injection **subtracted** ranking
while displacing the operating point 1.6 logits further out of reach. DW AUC is
flat (0.9042 -> 0.9050) so nothing was gained there either.

`x3 upsampling on a small base` stays refuted -- but for the opposite reason than
recorded. It was not that 50.8% record share destroyed the decision (it did not;
AUC held at 0.844). It is that the injection bought no ranking at any dose and
cost the operating point at every dose. Confirmed by `a1` at 17.3% on the 819K
base: AUC 0.8237 vs the control's 0.8368, calib J 0.4629 vs 0.5075, thr -4.617 vs
-2.625. Same sign, smaller size, monotone in dose.

**Recommendation: drop the abstention-SFT track.** Two doses on two bases, four
arms, no ranking gain anywhere, consistent operating-point damage.

### a1, and the dose curve closed

| eval | a1 AUC | 819K control | a1 calib J | control |
|---|---:|---:|---:|---:|
| aw_m2 | 0.8237 | **0.8368** | 0.4629 | **0.5075** |
| aw_m4 | 0.8153 | **0.8294** | 0.4782 | **0.4944** |
| dw    | 0.8928 | **0.8987** | 0.6674 | **0.6714** |

Loses on 6/6, small and consistent. `a1` cost 12h53 on 4x A100 and was built to
test a dose hypothesis that the instrument had manufactured.

### The naming gap, sized on the full splits

`a1`'s calibrated-gate evals give the first full-split counts of images the gate
calls positive and the parser leaves blank:

| eval | gate-positive but unnamed | of |
|---|---:|---:|
| aw_m2 closed_vocab | 177 | 581 |
| aw_m2 open_cot | 191 | 581 |
| aw_m4 closed_vocab | 199 | 581 |
| dw closed_vocab | 299 | 1504 |
| dw open_cot | 381 | 1504 |

Suppression stays near zero (0-10 images) because the gate is one-directional and
these models under-assert. **~30% of AerialWaste images are detected and unnamed.**
That is the work item, and it is a naming problem behind a working detector.

### What this changes

| lever | status |
|---|---|
| spoken J as an arm-selection metric | **refuted** -- rho +0.02 with calibrated J over 8 arms |
| abstention SFT (`rs_sft`) | **drop** -- no ranking gain at 17.3% or 50.8%, costs the operating point |
| 819K base mix | **regression** vs 150K on 6/6; the swap was made on spoken numbers |
| 150K base (`_finetune_`) | **best detector on file**; new baseline to beat |
| nadir description corpus (s3b) | re-read as sound: it targeted naming, and naming is what is broken |
| next arm | build on the 150K base, target naming, score with AUC only |

## Naming arms n1/n2/n2d on the 150K base (2026-08-17, jobs 52570906-52585553)

First arms built after the AUC sweep, and the first built on `_finetune_` (150K)
rather than `_finetune_next_` (819K). Target: the ~30% of AW images that come back
gate-positive with an empty parse. Replay is `general_150k`, matching the base's
own distribution. New `_source_cap` build knob caps the two close-up tiers of
`waste_sft` (trashbox 6000, wastebench 3000) -- they are also the TERSE tiers
(wastebench answers have a median length of one word, 70% <=3), so uncapped this
corpus teaches terseness, the opposite of naming. After the caps the four
naming-shaped aerial sources are ~51% of the component instead of 19%.

| arm | mix | records | steps | wall |
|---|---|---:|---:|---:|
| n1 | `waste_sft` + 25% `general_150k` | 24,449 | 191 | 45 min |
| n2 | + `nadir_desc` | 62,777 | 490 | 115 min |
| n2d | n2 with `--decision-loss-weight 1.0` | 62,777 | 490 | 115 min |

Leakage re-verified before quoting any of this: MD5 + pHash (Hamming <=6) over all
15,025 AW+DW images, **0 leaks from SWAD's 1,996 tiles**, 0 from TrashBox / TACO /
ZeroWaste, and the single UAVVaste flag manually confirmed a false positive. SWAD
is 1.8 m/px Henan; AerialWaste ~0.2 m/px Italy.

### Detection

| | aw_m2 AUC | aw_m2 calib J | aw_m2 J@0 | aw_m4 AUC | dw AUC | neg>0 aw / dw |
|---|---:|---:|---:|---:|---:|---:|
| 150K base | 0.8615 | 0.6053 | 0.066 | 0.8591 | **0.9042** | 0.00 / 0.10 |
| **n1** | **0.9410** | **0.7338** | **0.695** | **0.9418** | 0.8639 | 0.26 / 0.38 |
| n2 | 0.9171 | 0.6778 | 0.610 | 0.9190 | 0.8563 | **0.03** / 0.27 |
| n2d | 0.9205 | 0.6563 | 0.498 | 0.9195 | **0.8751** | 0.04 / 0.33 |

**n1 is the largest single gain in the project.** aw_m2 AUC 0.8615 -> 0.9410 closes
73% of the remaining gap to the frozen-feature probe (0.970). AUC is
shift-invariant, so this is ranking, not recentring. Separately, the operating
point recentred on its own -- threshold -4.00 -> +1.38, J@0 0.066 -> 0.695 -- so
the model now SAYS it with no calibration at all.

### Naming (closed_vocab micro-F1, gated / ungated)

| arm | aw_m2 | aw_m4 | dw | unnamed aw_m2 |
|---|---:|---:|---:|---:|
| 150K base | 0.289 | 0.166 | **0.310** | — |
| **n1** | **0.355** / 0.352 | **0.345** / 0.190 | 0.212 / 0.142 | **90**/581 |
| n2 | 0.089 / 0.088 | 0.291 / 0.165 | 0.175 / 0.089 | 186/581 |
| n2d | 0.054 / 0.054 | 0.136 / 0.132 | 0.219 / 0.139 | 204/581 |

n1 wins naming on both AW splits and cuts gate-positive-but-unnamed from a1's
177/581 to **90/581**. Hold the claim to the baseline though: the aw_m2 constant
predictor scores 0.3073 and aw_m4's scores 0.3585, so n1 clears the degenerate
baseline by 0.048 on m2 and **still loses to it on m4**. Naming moved; it has not
yet been won.

### The gate stopped being inert

Every earlier arm under-asserted, so the one-directional gate had nothing to
suppress and gated == ungated micro-F1 throughout. These arms over-assert, and the
gate now does real work: aw_m4 n1 0.190 -> **0.345** (199 images suppressed), dw n2
0.089 -> 0.175 (529 suppressed). This is the first evidence that the calibrated
gate improves the multi-label answer and not just the binary decision.

### `nadir_desc` controls false alarms and mutes naming

Adding it cut the fraction of AW negatives scoring positive from 0.26 to **0.03**,
exactly the nadir-absence mechanism it was included for, and the risk written into
the arm before the run. But it did NOT restore dw AUC (0.856/0.875 vs the base's
0.904) and it cost AW ranking (0.941 -> 0.917) and *collapsed* aw_m2 naming
(0.355 -> 0.089). Suppressing false positives is not the same as restoring
discrimination -- the arm design conflated them.

Mechanism, consistent with s3b: `nadir_desc`'s vocabulary is LAND COVER. It
competes with waste vocabulary for the same answer slot, so the model describes
the tile instead of naming the material. It improved captions on the 819K base for
the same reason it mutes the taxonomy here.

### The margin BCE is base-dependent, not a free win

n2d vs n2 -- same corpus, same steps, only the loss differs. On the 819K base the
same contrast (s3bd vs s3b) won all six. Here it splits: better on dw (AUC 0.8751
vs 0.8563) and marginally on aw_m2 AUC, but worse on calibrated J (0.656 vs 0.678),
much worse at zero (0.498 vs 0.610), and worse on naming everywhere. Its decision
loss fell 0.078 -> 0.051 then ROSE to 0.097, where s3bd's fell monotonically
0.234 -> 0.075: on a base that already arrives calibrated there is little for the
BCE to fix, so it ends up trading against the CE term. Do not carry it forward as
a default.

### Where this leaves the model

| domain | best model | AUC | calib J | naming |
|---|---|---:|---:|---:|
| AerialWaste | **n1** | 0.9410 | 0.7338 | 0.355 |
| DroneWaste | **150K base** | 0.9042 | 0.6725 | 0.310 |

No arm wins both, and the split is stable across all three readouts. Since the
threshold is already per-domain (AW->DW transfers at 58%), **selecting the model
per domain costs nothing operationally** -- that is the recommendation, rather than
hunting for one arm that wins everywhere.

Open: `open_cot` collapsed on AW for n1 (spoken J -0.063, micro-F1 0.000). n1
trained on short answers and the two-turn format needs a turn-1 description to
commit from; the caps fixed the vocabulary balance but not the format. A
format-anchor tier scaled past its current 899 records is the obvious fix.

### Descriptions: n2/n2d are the best on file, and it reverses the arm ranking

The naming table above scores the taxonomy commit (turn 2). Scoring the
DESCRIPTION (turn 1) instead puts the same arms in a different order, and for the
open-description product framing this is the metric that matters.

Caption conditioning on aw_m2, `open_cot` turn-1:

| arm | "pile" gt+ - gt- | "debris" | any waste term gt+ / gt- | distinct caps | modal | mean len |
|---|---:|---:|---|---:|---:|---:|
| 819K control | +2.8 | +1.6 | 3.8% / 0.5% | 330/581 | 9% | 13w |
| s3b | +11.9 | +4.4 | 12.6% / 1.5% | 544/581 | 3% | 26w |
| n1 | **-35.3** | -12.5 | — | 468/581 | 2% | — |
| n2 | +31.0 | +8.8 | 38.5% / 10.8% | 574/581 | 5% | 38w |
| **n2d** | **+35.8** | **+9.7** | **61.0%** / 32.6% | 570/581 | 3% | 38w |

**n2d triples s3b's caption conditioning** (+35.8 vs +11.9), mentions a waste term
on 61% of AW positives against the 819K control's 3.8%, and all but eliminates the
template (570/581 distinct, modal opening 3%). On dw it mentions waste on 92.5% of
positives. Sampled turn-1: *"There is a small pile of debris in the bottom-left
corner, which appears to be scattered or dumped materials."*

Two things this exposes:

**n1's `open_cot` collapse is vocabulary substitution, not muteness.** Its turn-1 is
the swad template verbatim -- *"Solid waste is visible in the top-left, centre and
bottom-left areas"* -- and turn 2 answers `solid waste`, which is not in the AW
taxonomy, so the parser returns empty and micro-F1 is 0.000. n1 learned swad's
single class name and uses it in place of every material word: "pile" conditioning
is **-35.3** while "solid waste" is +46.7. Under `closed_vocab`, where the menu is
supplied, it names correctly (0.355). The capability is there; the open vocabulary
was overwritten.

**s3b describes the scene and misses the waste.** On gt+ images it produces *"a
large parking lot with many vehicles parked in rows"* -- fluent, grounded, and
silent about the dumping. That is why its captions looked good on lexical
diversity while its detection sat at AUC 0.801. Diversity is not conditioning.

The cost of n2d's descriptions is over-mention: waste terms appear on 32.6% of AW
negatives (s3b 1.5%). Same over-assertion as its detection, and the same fix --
gate first, describe second, so the false mentions never surface.

### Recommended product stack (open descriptions)

1. **Decide with the calibrated gate** on `n2d`: one prefill, AUC 0.9205 aw_m2 /
   0.8751 dw, threshold from 50 in-domain binary labels.
2. **Describe with n2d turn-1** only when the gate fires: waste term on 61% of AW
   and 92.5% of dw positives, 38-word grounded descriptions, template broken.
3. **Do not use the turn-2 taxonomy commit.** It answers `none` even on images it
   has just described as containing debris; it is the broken component, and the
   product framing does not need a closed vocabulary.

For a closed-vocabulary benchmark number instead, `n1` remains the naming arm
(aw_m2 0.355) and the 150K base the drone arm.

## Grounded-architecture batch: what the six experiments settled (2026-08-18)

Six experiments run against the Track A/B plan. Two contradict predictions made
earlier the same day, and those come first.

### Retraction: resolution does not unlock naming

I argued that AerialWaste's 0.90-token median object was a *projector* artefact and
that a finer dense map would lift the material head. The frozen-feature sweep says
no. Naming micro-F1, positives only, best constant predictor **0.697**:

| input | 768px | 1024px | 1536px |
|---|---:|---:|---:|
| naming micro-F1 | 0.727 | 0.727 | 0.720 |
| detection AUC | 0.9696 | 0.9703 | 0.9692 |
| detection J | 0.853 | 0.864 | 0.863 |

Flat, then slightly down. The encoder does not matter either — C-RADIOv4-H 0.734,
DINOv3-L 0.730, C-RADIOv4-SO400M 0.727, all within noise of each other *and of the
constant predictor*. Whatever bounds material naming, it is neither spatial
resolution nor backbone choice.

### Ceiling experiment: naming fails with perfect localisation

The 1350 labelled AerialWaste object crops (`scripts/roi_material.py`), detection
removed entirely. Majority-class bar **0.399**:

| readout | accuracy | macro-recall |
|---|---:|---:|
| GeoRSCLIP ViT-L/14 | 0.384 | 0.290 |
| RemoteCLIP ViT-L/14 | 0.375 | 0.274 |
| our VLM (Yes/No on crop) | 0.345 | 0.330 |

Every readout is **below the bar**, at both context margins. Macro-recall against a
0.20 chance floor says a trace of signal exists, but nothing here can name an
AerialWaste material. With the supervised probe barely clearing its constant
predictor, the honest reading is that these five categories are **not recoverable
from this imagery** — a statement about the task, not the method, and one that
bounds any material head built on top.

### Retraction: the model *is* looking at the waste, for detection

I wrote repeatedly that the decoder loses the signal. For the presence decision
that is wrong. Removing the annotated waste (2.44% of the tile) vs removing an
equal area elsewhere (`scripts/occlusion_probe.py`):

| arm | Δ waste removed | Δ control removed | p | images |
|---|---:|---:|---:|---:|
| n2d | **−1.782** | +0.086 | 8e-26 | 84% |
| n1 | **−1.389** | +0.033 | 3e-26 | 87% |

Sign flips Yes→No: 44 vs 1 (n2d), 24 vs 1 (n1). Crop of the largest box scores
+0.002 against **−3.174** for a same-sized background crop. This is genuine
grounding and the strongest positive result in the project.

Category-level grounding is far weaker. Drop for categories that ARE present vs
ABSENT on the same image: n2d +1.235 vs +0.886 (p=2e-9) — real, but **72% of the
effect is presence, not identity**. n1 shows none (p=0.207).

### Object scale is the lever for grounding, not for naming

Grounding DINO, zero waste-specific training, identical code and prompts:

| dataset | tokens/object | box recall @ IoU 0.5 |
|---|---:|---:|
| AerialWaste | 0.90 | 0.150 |
| DroneWaste | 7.96 | **0.819** |

A 5.5× recall difference tracking a 9× object-size difference. Generic
open-vocabulary grounding **transfers** to aerial waste when objects are
resolvable and collapses when they are not. Neither dataset gets image-level
discrimination at the low threshold — both fire on ~100% of negatives — so this is
a localisation result, not a detector.

Kosmos-2 returns one box and a language-prior caption ("garbage or dumped waste is
a major problem in the world"), the exact failure mode the grounding plan exists to
avoid.

### Consequence for the plan

- **Detection + localisation is the defensible thesis story.** Grounding is real
  (occlusion), transfers zero-shot at adequate object scale (GDINO on DroneWaste),
  and the frozen probe reaches AUC 0.970.
- **The material branch (B4/B5/Model 6-7) has no headroom on AerialWaste.** Three
  independent readouts and a supervised probe all sit at or below their baselines,
  and neither resolution nor encoder moves it.
- **DroneWaste should carry the grounding experiments**, AerialWaste the detection
  ones. Reporting AerialWaste grounding numbers measures object scale, not method.

## The material branch is dataset-bound, not impossible (2026-08-19)

The ceiling experiment run on DroneWaste, same code and same readouts as the
AerialWaste run above, on 5135 ground-truth object crops across 20 categories.

| readout | AerialWaste (5 cats) | DroneWaste (20 cats) |
|---|---:|---:|
| majority-class baseline | 0.399 | 0.198 |
| GeoRSCLIP-L14 @ctx 0.5 | 0.356 (**−0.044**) | **0.314** (+0.116, p=7e-87) |
| GeoRSCLIP-L14 @ctx 0.0 | 0.384 (−0.016) | 0.303 (+0.105, p=4e-72) |
| RemoteCLIP-L14 @ctx 0.5 | 0.375 (−0.024) | 0.273 (+0.076, p=4e-39) |
| RemoteCLIP-L14 @ctx 0.0 | 0.299 (−0.101) | 0.212 (+0.015, p=0.005) |

Every AerialWaste readout is **at or below** the majority baseline. Every
DroneWaste readout is above it, the best by 11.6 points on a problem with four
times as many classes. Normalising for that, macro-recall against 1/K chance:

| | best macro-recall | classes | × chance |
|---|---:|---:|---:|
| AerialWaste | 0.330 | 5 | 1.65× |
| DroneWaste | 0.244 | 20 | **4.89×** |

So material identity **is** readable from a remote-sensing CLIP given a correct
crop — just not from AerialWaste's crops. This matches the grounding result
exactly: 0.90 tokens/object versus 7.96. The same 9× scale gap that separates
0.150 from 0.819 box recall separates naming that fails from naming that works.

Context helps on DroneWaste (+0.011 to +0.061 for ctx 0.5 over a tight crop) and
hurts or does nothing on AerialWaste, which is what one expects when the crop
itself carries no signal to contextualise.

### Revision to the previous section

"The material branch has no headroom" was too broad — it holds for AerialWaste
and does not generalise. **Model 6-7's material head is worth building, on
DroneWaste.** AerialWaste's site-level labels (mean 2.56 of 5 categories per
positive, P(B|A) ≈ marginal P(B)) describe a site, not the pixels in a crop, and
no readout can recover from that.

## It is the labels, not the sensor (2026-08-19)

The ceiling experiment crops to the object and upsamples, so the token-budget gap
that explains grounding (0.90 tokens per object against 7.96) is already removed
there. AerialWaste failed anyway, leaving two live explanations: too few pixels
on the object, or a label that was never a statement about that crop.

`--degrade` separates them. Crops are thrown down to N pixels across before the
usual preprocess, so on DroneWaste -- whose labels are drawn per object -- label
quality is held fixed and only pixels vary. GeoRSCLIP, ctx 0.5, macro-recall as a
multiple of 1/K chance so the 5-class and 20-class problems are comparable:

| pixels across the object | AerialWaste (5 cls) | DroneWaste (20 cls) |
|---|---:|---:|
| 16 | 1.03x | 2.37x |
| 24 | 1.13x | 3.27x |
| **40** (AerialWaste's native median) | **1.40x** | **3.99x** |
| 64 | 1.50x | 4.46x |
| native | 1.54x | 4.89x |

**At a matched pixel budget DroneWaste is at 3.99x chance and AerialWaste at
1.40x.** Same number of pixels on the object, roughly seven times the
discriminative signal. Resolution cannot be the explanation.

Both datasets degrade at the same rate -- 73% and 77% of their above-chance
signal survives to 40px -- so the imagery behaves identically under pixel loss.
The difference is that one of them has signal to lose (3.89x above chance) and
the other never had any (0.54x).

Note the accuracy column inverts on AerialWaste: degrading to 16px *raises*
accuracy 0.384 -> 0.424 while dropping macro-recall 0.290 -> 0.237 and predicted
classes 5/5 -> 4/5. Blur collapses the readout onto "Bulky items", which is a
better bet than anything it knew. That is why macro-recall is the reported
column.

### What this settles, and what it does not

CORRECTION. An earlier draft of this section said AerialWaste's material labels
come from site inspection records and are therefore not statements about a given
crop. Both halves are wrong. They are hand annotations made by looking at the
imagery, and they are genuinely per-object: of the 191 test images carrying more
than one box, 88.5% have boxes with differing categories, so the label is not an
image-level tag broadcast onto every box.

What the degradation curve does support is narrow and within-dataset: AerialWaste
material signal is weak at *every* resolution it was measured at, 1.03x chance at
16px rising only to 1.54x at native, so the last factor of two in pixels is not
what stands between it and a usable classifier.

What it does NOT support is the cross-dataset reading. Matched pixels are not
matched information: AerialWaste's ground sampling distance is far coarser, so a
40px AerialWaste crop is a downsampled patch of many metres of mixed ground while
a 40px DroneWaste crop is a downsampled single object. Degrading DroneWaste
removes pixels from one object; it does not reproduce what an AerialWaste crop
contains. The 1.40x against 3.99x comparison is suggestive, not controlled.

The most direct evidence that this task is hard for *people* and not only for
models is in the label distribution itself: **25.8% of AerialWaste's annotated
objects are labelled "Unknown material"**. Annotators working from the full
image, at full resolution, with scene context, declined to name a quarter of what
they boxed. Whatever a model recovers, it is bounded by a target that its own
authors could not assign a quarter of the time -- alongside ordinary annotator
noise, which the label set gives no way to estimate without a re-annotated
subset.

## Attribution maps: it looks at the waste, and at the same waste whatever you ask (2026-08-19)

`src/attribution.py` maps the Yes/No margin back onto the 24x24 visual token grid.
The margin is one scalar from one prefill and the projected tokens are an ordinary
tensor spliced into the prompt, so the path is differentiable end to end -- no
attention weights, which are a poor attribution signal and behind fused kernels
anyway. Integrated gradients over 40 AerialWaste images with drawn boxes, plus
token-block occlusion on 12 as the assumption-free reference.

| | mass-in-box | lift over uniform | peak inside box |
|---|---:|---:|---:|
| uniform null (= box area fraction) | 0.055 | +0.000 | -- |
| centre prior | 0.096 | +0.040 | 60% |
| **integrated gradients** | **0.393** | **+0.338** | **70%** |
| token occlusion (12 imgs) | 0.257 | +0.193 | 25% |

Evidence for "yes, waste" lands on the annotated waste at **8.5x the lift of a
centre-weighted map that never saw the image** -- which matters because
AerialWaste piles are often central and a centre prior would inherit that for
free. The two estimators agree in direction and magnitude, so the cheap one is
usable at scale.

Note the two columns disagree for the centre prior: it wins 60% peak-inside-box
while scoring lift +0.040. Peak position is easy to get right by accident on
centred data; mass is not. Lift is the column to read.

### The same map, whatever material is asked about

| correlation | |
|---|---:|
| between two DIFFERENT category maps | **+0.791** |
| between a category map and the presence map | **+0.824** |

Asking "is there plastic" and asking "is there rubble" produce the same map, and
both are essentially the map for "is there waste". This is the occlusion result --
72% of the category effect was presence rather than identity -- reproduced by an
independent method, and it is now visible rather than inferred.

**Consequence.** The decoder is a usable, training-free *localiser* and not a
namer. Open-vocabulary naming cannot be obtained by varying the question text on
this stack, because the attribution does not respond to the question's content.
Naming has to come from a text-aligned encoder.

## The decoder writes prompts, and they are worse than doing nothing (2026-08-19)

Two-stage routing works. The first stage names an object by comparing C-RADIOv4's
SigLIP2-projected embedding against text, is right 47.8% of the time on the
development sites, and knows when it is not: entropy separates its right answers
from its wrong ones at AUROC 0.817. Deferring the least-confident 30% and
re-scoring them against only their top-5 candidates lifts that subset from 0.212 to
0.348 and the whole set from 0.478 to 0.519.

Every bit of that comes from hand-written prompts. The decoder-written arm, which
is the one job in the design with no non-LLM substitute -- the candidate set does
not exist until inference, so no fixed bank could have been written for it -- makes
the deferred subset **worse than leaving it alone**.

| deferred set, 913 objects | accuracy |
|---|---:|
| stage 1, no routing | 0.212 |
| decoder prompts, first template | 0.097 |
| decoder prompts, EWC-grounded, worked examples, cross-naming guard | 0.133 |
| hand-written contrastive prompts | **0.348** |
| oracle: top-5 containment | 0.802 |

Dumping the generations explains the first template and not the gap that remains.
Asbestos came back as "dark, irregular patches" in one candidate set and "small,
uniform, dark grey rectangles, very flat" in another, when asbestos sheeting from a
drone is light grey corrugated panel: fluent, confident, wrong about the material,
and inconsistent between candidate sets so the scores stop being comparable across
objects. Handing the decoder the official EWC entry fixed the facts -- 12.11 now
produces "dark gray concrete chunks / red brick fragments / white gypsum boards",
which is what the entry says -- and bought 0.036. It did not approach the
hand-written arm.

### The asymmetry worth stating before drawing a conclusion

The hand-written prompts were written *after reading the confusion matrix*. They
encode which classes collide and how. The decoder writes blind: it gets the
candidate names and definitions, no feedback about what is being confused with
what, and no signal about whether its wording works for this particular encoder.
Prompt effectiveness for a CLIP-family model is not the same thing as
human-readable accuracy, and nothing in the current design lets the decoder
discover the difference.

So the honest claim is narrow: **a decoder writing prompts open-loop, from names and
definitions alone, is worse than not routing.** Whether it can do better with a
selection signal is untested, and the signal to try is label-free -- generate
several candidate prompt sets, keep the one that separates the candidates best on
the actual image embeddings, since margin is already known to track correctness at
AUROC 0.82. The failure mode to check is that confidently-wrong prompts also score
high margin, which is what the development labels are for.

## Detection, not reasoning or naming, is what bounds the compositional system (2026-08-19)

The four-arm harness separates perception error from reasoning error, which an
end-to-end evaluation cannot. On the development sites, 4496 questions over 759
images with the symbolic solver verified at exactly 1.000 over ground-truth scenes:

| family | trivial bar | symbolic / GT | symbolic / predicted |
|---|---:|---:|---:|
| count | 0.604 | 1.000 | **0.142** |
| presence (control) | 1.000 | 1.000 | 0.692 |
| area_compare | 0.508 | 1.000 | 0.585 |
| negation_spatial | 0.517 | 1.000 | 0.584 |
| spatial | 0.508 | 1.000 | 0.551 |
| count_compare | 0.536 | 1.000 | 0.524 |
| superlative | 0.268 | 1.000 | 0.309 |
| **overall** | | **1.000** | **0.462** |

**Perception costs 0.538 with reasoning held perfect**, and most families barely
clear the majority-or-oracle bar they are scored against.

The obvious explanation was over-proposal: the scene graph carried 29.3 detections
per image against 4.8 real objects, with no deduplication. It is wrong. Filtering
by score and applying NMS makes things monotonically WORSE -- 0.462 unfiltered, 0.447
at 10.1 per image, 0.404 at 4.1 per image, which is the ground truth's own density
almost exactly. Counting never recovers, at 0.142, 0.140 and 0.103.

So the detections do not correspond to the annotated objects at any operating point,
and matching the count does not make them the right objects. That agrees with the
detector measured directly: at a proposal budget of one the bridged arm recalls
0.258 and native SAM3 0.340, and at a budget matched to the 4.795 objects actually
present, 0.43-0.53 and 0.56-0.67 (interpolating the budget-3 and budget-10 rows of
`sam3_obj_dronewaste.json`). Roughly half the objects are not there to reason over.

CORRECTION (2026-08-20). This paragraph previously described the budget-of-one
figures, 0.258 and 0.340, as "matched to the number of objects present" and
concluded that three quarters of the objects are missing. The budget was 1, not
4.795. The conclusion holds in direction -- detection is the binding constraint --
but the magnitude was overstated by roughly a factor of two.

### Consequence

Detection recall at a sane budget is the binding constraint on the whole agentic
design. Naming, routing, prompt engineering and the decoder's composition all
operate downstream of a scene graph that is missing most of its objects, and no
amount of work on them can recover what was never detected. The complementarity
already measured -- Grounding DINO 0.909 on large objects and 0.273 on small, SAM3
0.716 and 0.682 -- makes their union the first thing to try, ahead of any further
work on the language half.

### Closed-loop prompt selection, the last untested arm (2026-08-19, job 52881523)

The open-loop result left one version of the decoder's prompt-writing role alive:
generate several candidate prompt sets and keep whichever separates the candidates
best on the cached image embeddings, using no labels. 236 distinct candidate sets
over the 913 deferred objects, best separation rising 0.0164 -> 0.0347.

| deferred set, 913 objects | accuracy |
|---|---:|
| stage 1, no routing | 0.212 |
| decoder prompts, open-loop, first template | 0.097 |
| decoder prompts, open-loop, EWC-grounded | 0.133 |
| **decoder prompts, closed-loop selection** | **0.161** |
| hand-written contrastive prompts | 0.348 |
| oracle: top-5 containment | 0.802 |

Closed-loop beats open-loop and still loses to doing nothing. Overall it moves the
full set 0.478 -> 0.462, i.e. routing with decoder prompts is a net cost, where
routing with hand-written prompts gives 0.478 -> 0.519.

**The claim is now unconditional for this design**: a decoder writing prompts from
names and definitions is worse than not routing, with or without a label-free
selection signal. The separation signal it optimises is real (margin tracks
correctness at AUROC 0.82) but confidently-wrong prompts score high separation too,
which is the failure mode the development labels were held for. Nothing further to
try on this axis; the remaining lever is detection recall.

## The AerialWaste material signal is there; the text readout cannot reach it (2026-08-20, jobs 52667316-52671854)

Written up late -- these runs happened on 2026-08-19 and never made it into this
file, which matters because they qualify one of the strongest claims in it.

The ceiling experiment cropped each annotated object and re-encoded the crop.
That throws away the full-resolution context and forces a second forward pass.
`scripts/roi_token_probe.py` does the opposite: encode the image ONCE at native
resolution, then mean-pool the patch tokens that fall inside the box. Readouts are
a linear probe on frozen features -- supervised, so this measures what the
features CONTAIN, not what the open-world pipeline can currently extract.

| dataset | test objects | classes | majority | `roi_mean` | vs bar | macro-recall |
|---|---:|---:|---:|---:|---:|---|
| aw_m2 @1024 | 1350 | 5 | 0.399 | **0.665** | **+0.266** | 0.704 = **3.52x** chance |
| dronewaste @640 | 2090 | 20 | 0.234 | **0.733** | **+0.500** | 0.586 = **11.72x** chance |

The AerialWaste row is on **the same 1350 objects** as the GT-crop ceiling
experiment, so it is directly comparable to it:

| readout on the same 1350 AerialWaste objects | accuracy | vs majority 0.399 |
|---|---:|---:|
| GeoRSCLIP ViT-L/14 (crop, re-encode) | 0.384 | -0.015 |
| RemoteCLIP ViT-L/14 (crop, re-encode) | 0.375 | -0.024 |
| our VLM (Yes/No on crop) | 0.345 | -0.054 |
| C-RADIOv4 -> SigLIP2 head, zero-shot (crop-summary) | 0.395 | -0.004 |
| **C-RADIOv4 ROI-pool + linear probe** | **0.665** | **+0.266** |

**CORRECTION.** The ceiling section concluded that AerialWaste's five categories
are "not recoverable from this imagery -- a statement about the task, not the
method". That is refuted. A linear probe on ROI-pooled C-RADIO tokens clears the
majority bar by 0.266 and predicts all 5 classes. The information is in the
imagery and in the features; what fails is every TEXT-ALIGNED ZERO-SHOT readout,
all four of which sit within +/-0.05 of the bar.

The honest claim is therefore about the readout, not the data: **AerialWaste
material identity is linearly decodable from frozen features and is not reachable
by any open-vocabulary text comparison tried.** That is a much narrower statement,
and it prices the open-world constraint: on this dataset, insisting on no fixed
label vector costs ~0.27 accuracy on the material branch.

Two controls that make the reading specific:

- **Localisation is what unlocks it, not the probe.** The `cls` readout is the
  same supervised probe on the whole-image summary vector with no ROI: aw_m2
  0.346, BELOW the majority bar, and dronewaste 0.473 against roi_mean's 0.733.
  Pooling the right tokens is worth +0.32 / +0.26 over having the same classifier
  read a global vector.
- **`roi_max` ~ `roi_mean`** (0.668 / 0.689), so this is not an artefact of the
  pooling operator.

### Why `roi-dense` fails, and what it says about the two SigLIP2 heads

`radio_zeroshot.py --mode roi-dense` pools the same tokens but sends them through
`_feature_projections.siglip2-g` instead of `_heads.siglip2-g`:

| mode | aw_m2 (5 cls, bar 0.399) | dronewaste (20 cls, bar 0.234) |
|---|---:|---:|
| crop-summary | 0.395 | **0.552** |
| roi-dense | 0.107 | 0.080 |

Both collapse far below their bars. The two heads target different spaces:
`_heads.siglip2-g` distils SigLIP2's **pooled, text-aligned** embedding, while
`_feature_projections.siglip2-g` distils its **patch** space, which is NOT
text-aligned. Comparing patch-space vectors against text embeddings is a type
error, and the dense segmentation arm shows the same thing directly --
`dense-seg` lift over mask area is **0.0021**, i.e. nothing.

So the pooled-token naming branch must route through the summary head. That is
what `roi-head` does, and it is the mode all the routing numbers use.
