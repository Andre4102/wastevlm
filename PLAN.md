# Pruned-Decoder VLM — Experiment Plan

_Target: CV4E @ ECCV (non-archival, deadline 2026-08-14). Companion analysis feeds the ICLR interp paper._

## Hypothesis

Aerial waste detection with a VLM needs **perception-conditioned reasoning and
instruction-following, not parametric world knowledge**. Structured pruning
preferentially destroys stored facts while preserving reasoning (prior result:
~48× fact-retention gap, reasoning causally dissociated from fact storage), so a
**pruned + recovered decoder should match a full decoder** on waste detection
while cutting inference cost — making the VLM competitive with e2e classifiers
on deployment cost, not just accuracy.

## Core comparison (the headline table)

All cells: **radio-l encoder**, same LLaVA-1.5-style pipeline (frozen encoder →
projector → decoder), same eval protocol.

| | generic stage-2 only | + waste SFT (stage 2.5) |
|---|---|---|
| **Qwen2.5-7B (full)** | A1 — already running (48312695) | A2 |
| **llama-3.1-8b ts0.5, recovered (~45% sparse)** | B1 | B2 |

Metrics per cell:
1. **Waste F1 / mAP** on DroneWaste + AerialWaste (two-turn CoT, kw-bag + LLM-judge parsing — existing eval code)
2. **Parse rate / format compliance** — first-class metric, reported alongside F1. Distinguishes "perception failure" from "generation/formatting failure"
3. **Efficiency**: per-image latency, peak VRAM, params (pruned decoder materialized, not masked)
4. (optional, if probe set built) attribute-probe accuracy

**Decision reads:**
- B2 ≥ A2 on F1 at lower cost → headline claim holds.
- B* accuracy drop concentrated in parse-rate → recovery insufficiency, not refuted hypothesis → extend recovery tokens, rerun.
- A2 ≫ A1 → waste SFT is the main lever regardless of decoder (still a useful result).

## Prerequisites / current state (verify before starting)

- [x] llama-3.1-8b ts0.5 pruned checkpoint (~45% sparsity), pre-recovery
- [ ] `cpt_recover_prune45` (150M tok recovery CPT) — **in flight, wait for it**; this checkpoint is the B-row decoder
- [x] radio-l stage-1 projector: `results/vlm/radio-l_pretrain/projector.pt` (loss 2.23)
- [ ] radio-l stage-2 (job 48312695) — in flight; produces cell A1
- [x] Pruned-arch loading: `src/eval/_arch.py::register_pruning_arch` (env: `myenv`)
- [x] Raw caption corpus `full.jsonl` (9,297 records, known noise: duplicates, short captions, prompt-scaffolding artifacts)
- [x] WasteBench QA pairs (~9.5k)
- [x] DroneWaste / AerialWaste with labels

> Claude Code: confirm actual paths for the caption corpus, WasteBench, and the
> stage-1/2 training scripts before Phase 1. Paths below marked `TODO(path)`
> where unknown.

---

## Phase 0 — Waste SFT dataset construction (no GPU, start immediately)

Build `data/waste_sft/` as LLaVA-format instruction data (`conversations` JSON
with `<image>` token, image paths resolvable on the cluster).

**Sources → ~30–40k samples total** (captions + WasteBench + format anchors +
external templated VQA; comfortably LoRA-SFT scale):

1. **Caption corpus (~9.3k → expect ~7–8k after cleaning)** — `TODO(path)/full.jsonl`
   - Dedup exact + near-duplicates (normalize whitespace/case; minhash or embedding-sim at 0.95 if easy)
   - Drop captions < 15 tokens
   - Strip prompt-scaffolding artifacts (leading "Here is a description…", "This image shows" boilerplate is fine to keep once, but remove meta-text about the prompt itself)
   - Format: image → "Describe this scene, focusing on any waste, its materials, and appearance."
   - **Hallucination guard:** sample 200 records for VLM-as-judge screening (Qwen2.5-VL-7B judging caption↔image consistency); report flag-rate. If >10% flagged, judge-filter the full set before training.
2. **WasteBench QA (~9.5k)** — convert to single-turn VQA format as-is.
3. **Format-anchor samples (~1k)** — replicas of the two-turn CoT eval structure with gold answers, so the decoder learns the exact commit-format the parser expects. Generate from WasteBench / caption-corpus / external-dataset images only (rewrite a subset of their QA/captions/labels into the two-turn structure), **never** from DroneWaste/AerialWaste.
4. **External templated VQA (~12–20k, downloaded public datasets)** — replaces
   the benchmark-derived VQA that was cut. Frame as a **viewpoint curriculum**:
   ground-level attributes → drone litter → satellite scene-level.
   - **TrashBox** (~17.8k imgs, 7 material classes incl. e-waste/medical; ground-level, web-scraped) — use all, template as material-identification VQA ("what material is this object? justify from appearance"); source imbalance is handled by dynamic resampling at training time (see Phase 3), not by subsampling. Web-scraped → dedup against every other source.
   - **UAVVaste** (~770 drone imgs, ~3.7k COCO litter annotations) — the only public drone-viewpoint waste data that isn't our benchmark; template detection labels as "is litter visible? what and roughly where?". Use all of it. **Mandatory provenance check: verify zero image overlap with DroneWaste before use** (both are drone-litter datasets).
   - **TACO** (~1.5k imgs, ~4.8k in-context litter annotations) — "describe the waste and its surroundings" samples; use all.
   - **SWAD** (~2k satellite imgs, 1.8 m/px, binary waste labels, Henan) — closest public distribution to AerialWaste without being it; template as yes/no + justification scene-level VQA. Use all; this is the main aerial scene-level exposure.
   - **ZeroWaste** (MRF conveyor segmentation, ~4.5k frames) — optional; a ~1–2k subsample for hard cluttered/occluded attribute-binding samples if budget allows.
   - Templates: ≥5 phrasings per question type to avoid template overfitting; balance yes/no answers within each source.
   - `TODO(path)`: download locations + licenses to be confirmed by Claude Code before use (all are public research datasets; record license per source in `stats.md`).

**DroneWaste / AerialWaste are benchmark-only — no split of either appears in SFT
in any form** (no images, no templated VQA from their labels). This makes the
Phase-4 numbers genuine transfer, not partial benchmark memorization.

**Leakage assertion (mandatory):** programmatically check that no SFT image
(caption corpus, WasteBench, format anchors, external datasets) overlaps with
*any* split of DroneWaste/AerialWaste — hash-based exact match (e.g. MD5 +
perceptual hash for re-encoded copies) plus filename check. Two hot spots:
- the caption corpus: if `full.jsonl` was generated *over* DroneWaste/AerialWaste
  imagery, those records must be dropped (or, if that guts the corpus, escalate —
  the SFT design changes);
- UAVVaste vs DroneWaste (both drone-litter; provenance + hash check).
Also cross-dedup the external sources against each other (TrashBox is
web-scraped and may contain TACO/TrashNet images).

**Deliverables:** `data/waste_sft/train.json`, `stats.md` (per-source counts, dedup/filter losses, judge flag-rate).

## Phase 1 — Stage 1 for the pruned decoder (1 GPU-day class)

Projector pretrain, LCS-558K, radio-l encoder, **decoder = recovered ts0.5
llama-3.1** (materialized checkpoint from `cpt_recover_prune45`, loaded via
`register_pruning_arch`).

- Same hyperparams as the Qwen stage-1 runs (2180 steps; expect ~10–11h on the same allocation)
- Decoder + encoder frozen; projector only
- **Gate:** final loss should land in the same band as the Qwen runs (≤ ~2.5, radio-l reached 2.23 with Qwen). If it's > ~3.0, recovery was insufficient — stop, extend recovery CPT, don't burn stage 2.
- Output: `results/vlm/radio-l_prune45_pretrain/projector.pt`

Also verify the chat template: recovered checkpoint is a **base** model CPT'd on
corpus text, not instruct-tuned. Stage 2 provides instruction tuning, but the
prompt template used in stage 2 / eval must match (llama-3 template or plain
LLaVA v1 template — pick one, use it everywhere for the B row).

## Phase 2 — Stage 2 for the pruned decoder (cell B1)

Visual instruction tuning: projector + LoRA on decoder, LLaVA-Instruct-150K,
~1,170 steps, 4×A100 — mirror job 48312695's config, swapping decoder + stage-1
projector.

Note: this doubles as additional recovery training (~same token order as the
41M-token Vicuna recovery anchor), so B1's decoder gets fluency healing for free.

Output: `results/vlm/radio-l_prune45_finetune/`

## Phase 3 — Waste SFT (cells A2, B2)

Stage 2.5: LoRA SFT on `data/waste_sft/train.json`, starting from the
respective stage-2 checkpoints.

- 2–3 epochs over ~30–40k samples; small LR (1e-5 class), projector unfrozen optional (run frozen first)
- Two jobs: A2 (from 48312695 output), B2 (from Phase 2 output)
- Cheap (few GPU-hours each) — if budget allows, sweep 1/2/3 epochs on A2 and pick by val F1, reuse choice for B2

**Data mixture — dynamic batch loading (Sheared-LLaMA-style):** instead of
static subsampling to balance sources, resample at training time.
- Tag every sample with its source domain: {captions, wastebench, format_anchor, trashbox, uavvaste, taco, swad, zerowaste}
- Hold out a small per-source dev slice (~100 samples each)
- Every N steps (~50–100), evaluate per-source dev loss and update sampling
  weights ∝ excess loss vs. reference (Sheared-LLaMA uses loss gap to a
  reference model; here use the simpler proxy — gap to each source's own
  loss-EMA target, or equivalently upweight sources whose loss is decreasing
  slowest). Renormalize; floor each source at ~2% so nothing starves.
- Log the weight trajectory in `stats.md` — it's a result in itself: if the
  sampler upweights uavvaste/swad (the scarce aerial sources), that's direct
  evidence the aerial domain is the hard/underlearned one.
- Fallback: if training is unstable or the implementation burns too much time,
  static mix with aerial sources upweighted ~3× (uavvaste, swad) and trashbox
  downweighted to an effective ~6–8k epoch-size.

## Phase 4 — Evaluation (all four cells + external baselines)

Existing two-turn CoT eval on DroneWaste + AerialWaste test splits, extended with:

1. **Parse-rate logging**: fraction of responses the kw-bag/LLM-judge pipeline can extract a committed answer from; log raw generations for failure inspection
2. **Efficiency benchmark**: fixed 200-image subset, batch 1, report median latency + peak VRAM per cell; pruned decoder must run **materialized**
3. Per-category breakdown (materials) where labels allow

**External baseline — TrashVLM-style (cell C):** reproduce the TrashVLM recipe
(QLoRA fine-tune of the compact Perception-Encoder-based VLM on TrashBox) and
evaluate it on DroneWaste/AerialWaste under the same protocol. Expected result:
strong on ground-level classification, collapses on aerial scene-level — the
motivating "ground-level waste VLMs don't transfer to aerial" row.
- Claude Code: check first whether code/weights were released (ATC 2025 paper,
  IEEE doc 11268574); if not, reproduce from the paper's recipe (QLoRA, ~1% params,
  TrashBox). If reproduction cost is disproportionate, a QLoRA'd
  Qwen2.5-VL-3B-on-TrashBox stands in as "TrashVLM-style" with a footnote.
- Output-space mapping needed: their model emits material categories; map to the
  benchmark's binary/multilabel scheme (any-waste-category → positive) and
  report both the mapped score and raw category outputs.
- **Fair-framing note for the paper:** TrashVLM never claimed aerial capability —
  present cell C as a domain-transfer baseline demonstrating the gap our
  viewpoint-curriculum SFT closes, not as a head-to-head defeat.

Note: with DroneWaste/AerialWaste fully held out, Phase 4 measures **cross-domain
transfer** — SFT data may skew ground-level/close-range while the benchmarks are
aerial. If both A2 and B2 underperform A1/B1, suspect the domain gap before the
decoder: check per-GSD / per-altitude breakdowns if metadata allows, and inspect
whether failures are scale/viewpoint-driven.

**Deliverable:** `results/vlm/pruned_decoder_comparison.md` with the 2×2 table + the C baseline row + parse-rate + efficiency columns, plus a short failure-mode section (perception vs parsing errors, ~30 manually-inspected failures per B cell and for C on aerial).

## Phase 5 (secondary, interp track) — Modality-attribution mask diff

Only after Phase 2; independent of Phases 3–4. On the **Qwen2.5-7B stage-2
model** (best-trained VLM):

1. Train two masks at **matched sparsity** (start 0.5), decoder frozen, projector frozen, **no LoRA during search** (attribution purity):
   - `mask_mm`: multimodal calibration (image+instruction through projector, loss on response tokens; sample from LLaVA-Instruct-150K + waste SFT val)
   - `mask_txt`: text-only calibration (same text, images dropped/blank)
2. Diff: per-layer IoU of kept-unit sets; define `visual_set = kept(mm) − kept(txt)` and reverse
3. **Causal validation:** ablate `visual_set` from the full model → expect VQA/captioning to crater while text-only QA + wikitext PPL stay flat; then reverse set → double dissociation
4. Compare layer profile to FastV-style prior (visual divergence concentrated in early layers); MLP-channel divergence in mid layers is the interesting-if-found result
5. Complementary: per-unit activation stats over visual-token vs text-token positions on held-out data; flag units where the two attributions disagree

**Deliverable:** layer-wise IoU plot, ablation table, `results/interp/modality_masks/notes.md`

---

## Sequencing & dependencies

```
now:        Phase 0 (CPU only)  ──────────────┐
wait:       cpt_recover_prune45 ──► Phase 1 ──► Phase 2 ──► Phase 3 (B2) ─┐
wait:       job 48312695 (A1)  ───────────────────────────► Phase 3 (A2) ─┼─► Phase 4
                                                Phase 2 done ─────────────► Phase 5 (parallel)
```

Rough GPU budget: Phase 1 ~11h×(stage-1 alloc); Phase 2 ≈ one 48312695-class job;
Phase 3 a few hours ×2; Phase 4 eval-only; Phase 5 two mask searches (cheap
relative to CPT) + ablation evals.

## Related work (positioning notes for the paper)

- **TrashVLM** (Trang, Pham, Vu, Le, Tran, Dao — ATC 2025, IEEE doc 11268574):
  QLoRA fine-tuning of a compact VLM (Perception Encoder backbone) on TrashBox
  (17,785 imgs); tunes ~1% of params, >10× less training memory/time, >10%
  accuracy gain over prior fine-tuning baselines. **Positioning:** validates the
  "cheap VLM adaptation for waste" premise, but ground-level single-object
  classification only — no aerial/scene-level imagery, no reasoning/CoT
  protocol, and efficiency via QLoRA + small backbone rather than decoder
  pruning. **Used as baseline (Phase 4, cell C)** to demonstrate the
  ground-level→aerial transfer gap. Our gap: aerial scene-level detection +
  pruned-decoder efficiency + fully held-out benchmarks.
- The prompt-engineering line (e.g. VLM waste recognition via prompt
  optimization, Waste Management 2025) shows zero-shot prompting alone lifts
  waste classification substantially — supports the "perception + prompting >
  parametric knowledge" hypothesis; cite when motivating the two-turn protocol.

## Explicit non-goals (this cycle)

- No ts0.9 / high-sparsity decoders — 88% breaks free-form generation; revisit only if B2 holds at 45%
- No dinov3-b / cradiov4h cells — encoder comparison is a separate axis; radio-l only here
- No CPT-scale domain pretraining — SFT-and-eval regime per prior corpus analysis
- Vicuna is retired as an artifact; it survives only as the recovery-cost anchor