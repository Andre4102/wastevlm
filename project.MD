# Multimodal LLM for Illegal Waste Detection from Aerial / Satellite Imagery

A vision-language system to support Italian Law Enforcement Agencies (LEAs) and environmental protection agencies (ARPA) in identifying illegal waste deposits from aerial and UAV imagery. The system combines a domain-pruned LLM with a remote-sensing-adapted vision encoder, aligned LLaVA-style and instruction-tuned for waste-specific reasoning.

This document captures the architectural design, the rationale behind the key choices, the immediate baseline experiment, and the longer-term research directions.

---

## 1. Problem framing

Illegal waste dumping is a major environmental and public-health problem in Italy, particularly acute in regions like Campania (the "Terra dei Fuochi" context) and well-documented in Lombardy through ARPA's photo-interpretation work. Manual scanning of aerial imagery for suspect sites is slow, expensive, and skill-bound. The aim of this project is to build a tool that supports — not replaces — LEA analysts and ARPA inspectors by:

1. Triaging large volumes of aerial / satellite imagery to surface candidate suspect sites.
2. Producing structured analyst-facing summaries: waste type, severity, evidence indicators, surrounding land use, regulatory context.
3. Cross-referencing against Italian environmental law (D.Lgs. 152/2006 and related jurisprudence) and ARPA classification conventions.

The output is **analyst-supporting, not legally authoritative**. The model paraphrases and reasons; the human inspector confirms and authors the formal documentation.

### Operational requirements

- **Italian comprehension** for ARPA documents and analyst queries (mandatory).
- **Italian or English generation** (flexible — paraphrasing is fine; only statute references and controlled-vocabulary classifications must be exact, and those are extractive rather than generative).
- **Hardware-agnostic deployment**: workstations, edge devices on UAV payloads, possibly Jetson-class accelerators. No assumption of NVIDIA Ampere/Hopper-class sparse-tensor cores.
- **Evidence-grade outputs** with calibrated uncertainty: a confidently-wrong classification is worse than a "low confidence" one.

---

## 2. Architecture (target state)

A LLaVA-style multimodal architecture with three modifications motivated by the deployment context:

```
                 ┌─────────────────────────────┐
   Aerial /      │   Vision encoder (DINOv2)   │     fine-grained patch features
   UAV image  ─→ │   + scene-adaptive token    │ ─→  for waste discrimination
                 │   compression layer         │
                 └─────────────────────────────┘
                              │
                              ▼ (cross-attention OR concatenation,
                              │  decided post-baseline)
                 ┌─────────────────────────────┐
   Italian /     │   Mistral-7B (or Nemo-12B)  │
   English   ─→  │   - LEA-fine-tuned          │ ─→  Structured analyst-facing
   query         │   - structurally pruned     │     output
                 │     with LEA calibration    │
                 │   - recovered via LoRA +    │
                 │     distillation from       │
                 │     dense LEA teacher       │
                 └─────────────────────────────┘
```

### Key design choices and their rationale

**LLM base — Mistral-7B (preferred) or Mistral-Nemo-12B.**
The user-facing requirement is Italian comprehension, not Italian-native generation. Mistral models have meaningfully better European-language coverage than LLaMA-2/Vicuna out of the box, ship with permissive licensing (Apache-2.0), and are a healthier ecosystem for ongoing work. Native-Italian models (Minerva, LLaMAntino) are over-specified for our generation needs and weaker on reasoning. Mistral-Nemo has longer context (helpful for full ARPA documents) at a 12B param cost; pick based on hardware budget after baselines.

**Vision encoder — DINOv2-B.**
DINOv2 captures fine-grained patch-level localization information that CLIP discards for global semantics. Aerial waste discrimination is a fine-grained heterogeneous-texture problem, which favors DINOv2's strengths. Precedent: COMM (Jiang et al. 2023) showed DINOv2 aligns to LLMs with just an MLP; Ferret-v2 uses a CLIP+DINOv2 fusion; dino.txt provides a clean text-alignment recipe for frozen DINOv2.

**Pruning the LLM (not the vision encoder, at least not at first).**
DINOv2-B is ~86M params; Vicuna/Mistral-7B is ~7B. The LLM dominates the inference budget. Pruning the LLM yields significant speed/memory gains across heterogeneous deployment hardware. We adopt the learning-based per-head channel structural pruning method ("Don't carry that weight", NeurIPS 2026 submission) as the pruning routine — it tests on both DinoV2-B and Vicuna-7B, beats SparseGPT-2:4 on perplexity at 50% structural sparsity, and produces hardware-agnostic speedups (no sparse-tensor-core dependency).

**Calibration with LEA documents, not English-only corpora.**
Calibration data drives which substructures survive pruning. An English-only calibration set would preserve generic English capability and damage Italian and domain-specific competence. We use an Italian-heavy mix (60–70%) including ARPA reports, Italian environmental jurisprudence, and AerialWaste metadata text, plus English instruction data (30–40%) to preserve cross-lingual reasoning.

**Important caveat on pruning**: the NeurIPS paper documents a dissociation — structural pruning preserves general linguistic competence but damages specific factual recall (BoolQ-style tasks). LEA-specific facts are at risk. Mitigation: dense LEA fine-tuning *before* pruning, then distillation from the dense LEA teacher into the pruned student during recovery. This re-injects the factual content the mask risks erasing.

**Pipeline ordering** (this is non-obvious and matters):

```
Stage 1: Mistral-7B (dense) ──▶ LoRA fine-tune on LEA documents
                                       │
                                       ▼
                              dense LEA-Mistral (TEACHER)
                                       │
                                       │ (calibrate from same corpus)
                                       ▼
Stage 2: Apply structural pruning ──▶ pruned-but-untrained LEA-Mistral
                                       │
                                       ▼
Stage 3: Recovery — LoRA + distillation from teacher
                                       │
                                       ▼
                              pruned-and-recovered LEA-Mistral
                                       │
                                       ▼
Stage 4: LLaVA-style alignment with frozen DINOv2 (MLP projector trained,
         then unfreeze for full alignment)
                                       │
                                       ▼
Stage 5: Instruction tuning on AerialWaste-derived VQA data
                                       │
                                       ▼
                              deployable model
```

The dense LEA fine-tune at Stage 1 is what converts pruning from "lossy operation that hopefully preserves what matters" into "compression of a model that already has what matters."

---

## 3. Baseline experiment (start here)

Before implementing any custom architecture, establish strong zero-shot baselines on AerialWaste using existing VLMs. The deltas between baselines are diagnostic for which capabilities matter most in this domain.

### Models to evaluate

| Model | Resolution | RS-adapted | Waste-adapted | Role |
|---|---|---|---|---|
| LLaVA-NeXT-7B | ~672 tiled | No | No | Generic VLM floor |
| GeoChat | 504×504 | Yes | No | RS-adapted floor |
| GeoLLaVA-8K | up to 8192×8192 | Yes | No | UHR + RS-adapted |
| GPT-4o or Claude 3.5 Sonnet (sample only) | high | No | No | Frontier ceiling reference |

The three open-source models give a clean ablation:
- `GeoChat − LLaVA-NeXT` → does generic RS adaptation help?
- `GeoLLaVA-8K − GeoChat` → does extreme resolution help on top of RS adaptation?
- `Frontier API − GeoLLaVA-8K` → how much open-source-vs-frontier headroom remains?

### Dataset

**AerialWaste v2.0** (Politecnico di Milano + ARPA Lombardia, https://aerialwaste.org).
- 10,434 images: ~3,478 positive (waste sites), ~6,956 negative.
- Three image sources at different resolutions: AGEA orthophotos, WorldView-3, Google Earth.
- Metadata on positives: evidence (0–3), severity (0–3), waste object class (22 categories), area type, storage mode.
- 715 samples annotated with object classes; 169 with segmentation masks.

Stratified sample: ~200 images, balanced positives/negatives, balanced across the three image sources.

### Query templates (apply identically across all models)

**Q1 — Binary detection.** "Does this image contain illegal waste dumping or accumulation? Answer yes or no, then briefly justify."

**Q2 — Open-ended description.** "Describe what you see in this aerial image." (Diagnostic for whether waste is even mentioned without prompting.)

**Q3 — Constrained classification.** "Is this image showing: (a) legitimate industrial site, (b) illegal waste dump, (c) construction site, (d) agricultural land, (e) other? Answer with a single letter."

**Q4 — Counting / localization.** "How many distinct waste piles or accumulations are visible, and where in the image are they located? Use coordinates relative to the image (top-left, center, etc.)."

Fix the prompts across models. Resist per-model prompt tuning; the comparison must be apples-to-apples even if some models' formats are not preferred.

### Reporting

Results split by image source (AGEA / WorldView-3 / Google Earth). The resolution-driven gains should be most visible on WorldView-3 (highest native resolution); their absence there would indicate the resolution argument is wrong for our domain.

Metrics:
- **Binary (Q1, Q3)**: accuracy, precision, recall, F1, **calibration (Brier score, ECE)**. Calibration matters as much as accuracy: a confident-wrong model is worse than a calibrated-uncertain one for LEA support.
- **Open-ended (Q2)**: human eval on 50 images, scoring whether waste was mentioned, whether the description was domain-appropriate, whether hallucinated specifics appeared.
- **Localization (Q4)**: human eval against the 169 segmentation-mask subset; coarse (top-left, center, etc.) is fine for now.

### Pre-flight checks (before week 1)

- Verify GeoLLaVA-8K loads and runs on available hardware. UHR models have non-trivial memory footprints; if it doesn't fit, the rest of the plan changes.
- Pull AerialWaste v2.0 and confirm image-source metadata is accessible.
- Decide whether tile/crop preprocessing is needed for high-resolution sources fed into LLaVA-NeXT and GeoChat. WorldView-3 native resolution far exceeds either model's input; downsampling will hide microdumps. Tile and report per-tile.

### Expected findings (predictions to test)

- LLaVA-NeXT: 55–70% binary accuracy, frequent confident hallucinations in non-waste images, generic descriptions.
- GeoChat: 70–80% binary accuracy, more domain-appropriate vocabulary, still misses regulatory-context cues.
- GeoLLaVA-8K: best on WorldView-3 specifically, comparable elsewhere; still struggles with the discrimination problem (legitimate yard vs illegal dump) because it lacks the regulatory priors.
- Frontier model (GPT-4o / Claude): probably best overall, useful as a ceiling reference but not deployable.

If GeoLLaVA-8K dominates substantially, build on top of it and skip the from-scratch route. If it underperforms on the discrimination problem regardless of resolution, the bottleneck is domain knowledge, validating the LEA-fine-tuning + waste-instruction-tuning plan.

---

## 3.5 Baseline results and plan refinement

The §3 ablation has been run on n ≈ 170–200 stratified samples per dataset for each of the three open-source VLMs (frontier APIs deferred). Headline numbers, failure-mode analysis, and the implications for the §2 architecture follow.

### 3.5.1 Results

**Q1 binary detection** (project.MD §3 prompt, identical across models):

| Model                  | AW acc / F1   | DW acc / F1   |
|------------------------|---------------|---------------|
| LLaVA-NeXT-7B Mistral  | 0.68 / 0.51   | **0.77 / 0.80** |
| GeoChat-7B             | 0.65 / 0.71   | 0.61 / 0.73   |
| GeoLLaVA-8K (Qwen2-7B) | 0.55 / 0.00   | 0.51 / 0.09   |

**Q1 by image source (AW)** — RS adaptation helps where the imagery looks orthorectified, hurts elsewhere:

| Source            | LLaVA-NeXT acc / F1 | GeoChat acc / F1 |
|-------------------|---------------------|------------------|
| AGEA (1000×1000)  | 0.61 / 0.35         | **0.83 / 0.85** (+22pp acc) |
| WorldView-3 (~1050²) | 0.67 / 0.56     | 0.62 / 0.72      |
| Google Earth (~700²) | **0.82 / 0.67** | 0.42 / 0.50 (−40pp acc) |

**Q5 combined-prompt multi-label** (single prompt listing all 20–22 classes):

| Model                  | AW micro F1 | DW micro F1 |
|------------------------|-------------|-------------|
| LLaVA-NeXT-7B          | 0.05        | **0.28**    |
| GeoChat-7B             | 0.00        | 0.01        |
| GeoLLaVA-8K            | 0.05        | 0.00        |

**Supervised reference** (Torres & Fraternali 2023, ResNet-50 on AW binary): ≈ 90–92% accuracy. The zero-shot ceiling sits 10–35 pp below that.

### 3.5.2 Failure-mode analysis

Diagnostic probes (`src/diagnose.py`, `src/diagnose_geollava.py`) confirmed dataloaders, prompt formats, and image conditioning are all correct. The three models have *distinct, systematic* failure modes — not random output:

- **LLaVA-NeXT** has waste vocabulary and engages with the task. Its p_yes spans 0.12–0.60 across probe images (real positives, real negatives, random noise, solid black/white), so vision *is* driving the decision. The limitation is that the CLIP-336 vision tower (trained on web photos) doesn't reliably activate "waste" concepts when shown top-down views — under-detects on aerial, over-detects on close-range drone.
- **GeoChat** was trained on RS-VQA where most answers are "yes, X exists." It pattern-matches into yes-saturation regardless of content (p_yes ≈ 0.54 even on random noise). The RS-adaptation lifts AGEA orthophotos by +22 pp over LLaVA-NeXT but costs −40 pp on Google Earth — its training distribution is brittle outside the imagery it has seen.
- **GeoLLaVA-8K** is trained on multiple-choice + counting + spatial reasoning RS-VQA (`initiacms/GeoLLaVA-Data`); the instruction data contains essentially no waste vocabulary. The model defaults to "no" on yes/no and shows strong positional bias on 2-option MC (always picks the "yes-waste" letter regardless of which letter it is). Its free-text descriptions (Q2) are coherent and image-specific but never mention waste even on positives. Vision works; the task is out-of-distribution. **The honest comparison for this model is Q3 (5-option MC)**, where it reaches DW F1 0.50 / AW F1 0.02.

Net: the gap to supervised baselines isn't a vision problem and isn't a language problem; it's the missing *alignment* between aerial-waste visual patterns and waste vocabulary that no upstream training has provided.

### 3.5.3 Architectural refinement — text-level interface vs feature-level

The §2 plan (LEA-fine-tune Mistral → prune → LLaVA-align DINOv2 → instruction-tune on AW VQA) is the most expressive option but has two heavy dependencies: assembling the LEA corpus (D.Lgs. 152/2006 + ARPA reports + EWC) and training the projector against AW-derived VQA. A lighter near-term path preserves most of the operational value while sidestepping both:

1. **Vision-only detector.** Train DINOv2-B (LoRA on the encoder, or just a probe head) on AW binary + DroneWaste multi-label. Output is structured: `P(waste)`, top-K categories, optionally bounding boxes.
2. **Text-level interface to an off-the-shelf LLM.** Pipe the detector's structured output into a generic LLM prompt as a system note (e.g. *"Detector: P(waste) = 0.83, top categories: rubble, scrap"*). The LLM does paraphrasing + statute lookup over its general knowledge; it never sees AW-specific Q–A pairs.

This sidesteps both the LEA corpus dependency and LLaVA-style projector training. The "zero-shot" claim is honest under this scheme: the LLM never saw AW outputs, so it can't have memorized image→severity mappings. Generalization across sites/regions is normal supervised generalization in the vision encoder, not test-set leakage. The full §2 plan remains the long-term target once the LEA corpus is available; the text-level scheme is the near-term operational baseline.

### 3.5.4 Current step — DINOv2-B linear probe

Before committing to encoder fine-tuning, we measure how much waste-relevant signal is already in stock DINOv2-B features (`src/dino_features.py` → `src/dino_probe.py`). Linear-probe protocol: ViT-B/14 features at 518×518 input (the DINOv2 paper's recommended fine-grained-classification setup), L2-normalised CLS embedding, sklearn `LogisticRegression` head with balanced class weights.

Three measurements:

- **AW binary**: train on AW training (9 096 imgs, ~3 032 positive), evaluate on AW testing (2 607 imgs).
- **AW multi-label**: 22 classes, restricted to the fine-grain-annotated subset on both splits (~165 test images).
- **DroneWaste multi-label**: 20 classes, site-stratified 70/30 split on the full 4 993 imgs.

Decision branches:

| Probe binary accuracy | Action |
|-----------------------|--------|
| ≥ 85%                 | features already strong; only train MLP head + small projector; skip encoder LoRA |
| 70–85%                | LoRA fine-tune DINOv2-B on AW binary; re-measure |
| < 70%                 | encoder is the bottleneck; consider RemoteCLIP or SatMAE as a starting point |

Multi-label probe results inform whether stock features can support 22-class discrimination or whether category-level supervision is also needed.

### 3.5.5 DINOv2-B probe results

**Binary AW** (linear probe, frozen DINOv2-B, 518×518, CLS embedding):

| Metric              | Value | vs supervised bar (90–92%) |
|---------------------|-------|---------------------------|
| Accuracy            | **0.891** | within 1–3 pp |
| Precision / Recall  | 0.79 / 0.92 | – |
| F1                  | 0.849 | – |
| AP                  | 0.921 | – |
| Brier / ECE         | 0.079 / 0.024 | well-calibrated |

Per image source: AGEA 0.898, Google Earth 0.933, WorldView-3 0.876 — all > 87%, beating every zero-shot VLM Q1 result by 20–40 pp.

**AW multi-label — official 5-class mapping (mcml/m2)** (train 4 304 / val 616 / test 664, CLS features, F1-tuned per-class thresholds):

| Class             | support | DINOv2 F1 / AP | dino.txt F1 / AP | LLaVA-NeXT F1 | GeoChat F1 | GeoLLaVA-8K F1 |
|-------------------|--------:|----------------|------------------|---------------|------------|----------------|
| Bulky items       |     150 | **0.72 / 0.76** | 0.59 / 0.64      | 0.34          | 0.04       | 0.00           |
| Unknown material  |     143 | **0.66 / 0.67** | 0.31 / 0.31      | 0.00          | 0.00       | 0.00           |
| Containers        |     110 | **0.64 / 0.67** | 0.46 / 0.48      | 0.40          | 0.05       | 0.00           |
| Rubble            |      90 | **0.49 / 0.50** | 0.27 / 0.30      | 0.32          | 0.04       | 0.00           |
| Plastic           |      57 | **0.32 / 0.27** | 0.17 / 0.16      | 0.21          | 0.03       | 0.00           |
| **micro F1**      |   ----- | **0.602**      | **0.374**        | 0.262         | 0.032      | 0.000          |
| **macro F1**      |   ----- | **0.565**      | **0.360**        | 0.253         | 0.034      | 0.000          |

**dino.txt zero-shot** (no parameters trained, only operating points tuned on train) lands +11 pp above the best generative VLM (LLaVA-NeXT 0.26) and ~23 pp below the supervised DINOv2 probe. It is the only zero-shot baseline that engages "Unknown material" at all (F1 0.31 vs 0.00 across all CLIP-based VLMs) — open-vocabulary text alignment lets us prompt with `unidentified waste material` and get a real signal. The supervised probe still wins by a wide margin because the LR head can absorb class-conditional shape priors that pure cosine similarity cannot.

**AW multi-label — fine-grain 6-class mapping (mcml/m4)** (train 4 304 / test 664):

| Class                              | support | DINOv2 F1 / AP | dino.txt F1 / AP | LLaVA-N F1 | GeoChat F1 | GeoLL8K F1 |
|------------------------------------|--------:|----------------|------------------|------------|------------|------------|
| Other waste                        |     186 | **0.77 / 0.84** | 0.55 / 0.58      | 0.00       | 0.00       | 0.00       |
| Rubble/excavated earth and rocks   |      90 | **0.49 / 0.50** | 0.32 / 0.34      | 0.28       | 0.06       | 0.00       |
| Wood                               |      35 | **0.34 / 0.30** | 0.18 / 0.18      | 0.15       | 0.00       | 0.00       |
| Tires                              |      25 | **0.29 / 0.27** | 0.10 / 0.06      | 0.07       | 0.00       | 0.00       |
| Scrap                              |       9 | **0.22 / 0.26** | n/a              | 0.15       | 0.00       | 0.00       |
| Sludge-Zootechnical waste-Manure   |      10 | **0.19 / 0.13** | 0.06 / 0.06      | 0.00       | 0.00       | 0.00       |
| **micro F1**                       |   ----- | **0.562**      | **0.381**        | 0.107      | 0.017      | 0.000      |
| **macro F1**                       |   ----- | **0.384**      | **0.202**        | 0.109      | 0.011      | 0.000      |

**DroneWaste multi-label — paper-10 subset** (train 3 489 / test 1 504, site-stratified split; the 10 classes the DroneWaste paper reports detection mAP on):

| Class                                  | YOLOv8 mAP | YOLOv12 mAP | F-RCNN mAP | DINOv2 AP (img) | dino.txt F1 | LLaVA-N F1 | GeoChat F1 | GeoLL8K F1 |
|----------------------------------------|-----------:|------------:|-----------:|-----------------|------------:|-----------:|-----------:|-----------:|
| Construction and demolition materials  |   13.1%    |    9.2%     |   8.8%     |  **0.50**       | 0.26        | 0.23       | 0.00       | 0.00       |
| Metal barrels                          |   37.6%    |   50.5%     |  27.6%     |  0.24           | 0.13        | 0.00       | 0.00       | 0.00       |
| Plastic packaging                      |    5.4%    |    6.5%     |  11.5%     |  **0.39**       | 0.30        | 0.00       | 0.00       | 0.00       |
| Pallets                                |   63.7%    |   67.2%     |  61.4%     |  **0.68**       | 0.45        | 0.42       | 0.00       | 0.00       |
| Scrap                                  |   14.7%    |   17.0%     |  16.4%     |  **0.72**       | 0.46        | 0.00       | 0.00       | 0.00       |
| Vehicles                               |    5.7%    |    1.6%     |  19.7%     |  **0.55**       | 0.52        | 0.40       | 0.00       | 0.00       |
| Tyres                                  |   64.9%    |   61.1%     |  56.9%     |  0.46           | 0.27        | 0.00       | 0.00       | 0.00       |
| Asbestos                               |   76.2%    |   76.9%     |  73.8%     |  **0.83**       | 0.53        | 0.00       | 0.00       | 0.00       |
| Textile                                |   78.5%    |   75.8%     |  70.2%     |  0.66           | 0.41        | 0.00       | 0.00       | 0.00       |
| Mixed items                            |   22.5%    |   19.1%     |  19.4%     |  **0.72**       | 0.51        | 0.26       | 0.00       | 0.00       |
| **mean mAP / image-level AP / micro F1**|  38.2%    |   38.5%     |  36.6%     | **0.57**        | **0.40**    | 0.18       | 0.00       | 0.00       |

Bold = DINOv2 image-level AP higher than the best of the three detection mAPs. Caveat: **classification ≠ detection.** Detection mAP requires bounding-box localisation with COCO IoU matching — strictly harder than the image-level "is class X anywhere in this image" task DINOv2 is doing. To convert this comparison into a real claim we would need either:

- a detection head on top of DINOv2 (DETR-style) trained on the DroneWaste boxes, then compute COCO mAP; or
- the detector papers' image-level pooled scores (typically 10–20 pp above their detection mAP), which would put the YOLO/F-RCNN line around 48–58 % — bracketing the DINOv2 0.57.

The conservative read is therefore: **DINOv2 frozen image-level classification is at-or-near the detectors' implicit image-level capability**, while requiring zero detection-supervision and zero waste-specific fine-tuning. Specifically strong on the classes where the detectors *underperform* (Scrap, Plastic packaging, Mixed items, Vehicles — all detection-hard heterogeneous textures), and weaker on cleanly-bounded instance classes the detectors handle well (Asbestos panels, Tyres).

> **Loader note.** An earlier multi-label run on AW with the legacy fine-grain 22-class taxonomy showed F1 = 0.00 — caused by a loader bug: AW training GT was stored in a per-image `categories` list (IDs), not in a COCO-style `annotations` array. Fixed in `src/datasets.py::load_aerialwaste_multilabel`. The official mcml/m2 split distributed by the AW team makes the 22→5 mapping explicit and is what we use for headline numbers.

### 3.5.6 Three approaches to detection on DroneWaste

To convert the "classification ≠ detection" caveat from §3.5.5 into real numbers, we report COCO mAP on the same site-stratified test split for three frozen-DINOv2 heads, side-by-side with the DroneWaste paper's end-to-end-trained detectors:

| approach | backbone | head | trained on DW? | trainable params | train wall-clock |
|---|---|---|---|---:|---:|
| **YOLOv8 / YOLOv12 / Faster R-CNN** | unfrozen | task-specific | yes (full e2e) | ≥ 25 M each | hours |
| **DinoDETR** | DINOv2-B *frozen* | 6-layer DETR decoder + box head | yes | 6.7 M | 95 min |
| **DinoSeg** | DINOv2-B *frozen* | linear semantic-seg head + CC→box | yes | **16 k** | 92 min |
| **dino.txt zero-shot seg** | dino.txt ViT-L/14 *frozen* | cosine-sim vs text prompts + CC→box | **no** | 0 | — |

Same 518² input, same site-stratified test split, same `pycocotools` eval against the same bbox GT.

**Per-class mAP @ IoU [0.5:0.95] on the paper-10 classes**:

| Class                                  | YOLOv8 mAP | YOLOv12 mAP | F-RCNN mAP | DinoDETR mAP | **DinoSeg mAP** | dino.txt zs mAP |
|----------------------------------------|-----------:|------------:|-----------:|-------------:|----------------:|----------------:|
| Construction and demolition materials  |   13.1%    |    9.2%     |    8.8%    |    1.8%      |     **6.9%**    |     0.0%        |
| Metal barrels                          |   37.6%    |   50.5%     |   27.6%    |    0.1%      |     0.0%        |     0.0%        |
| Plastic packaging                      |    5.4%    |    6.5%     |   11.5%    |    0.0%      |     **5.6%**    |     0.0%        |
| Pallets                                |   63.7%    |   67.2%     |   61.4%    |    0.9%      |     7.2%        |     **8.2%**    |
| Scrap                                  |   14.7%    |   17.0%     |   16.4%    |    2.7%      |    **12.3%**    |     0.1%        |
| Vehicles                               |    5.7%    |    1.6%     |   19.7%    |    0.2%      |    **23.8%**    |     0.1%        |
| Tyres                                  |   64.9%    |   61.1%     |   56.9%    |    0.2%      |     **5.6%**    |     2.3%        |
| Asbestos                               |   76.2%    |   76.9%     |   73.8%    |   10.4%      |    **47.6%**    |     1.8%        |
| Textile                                |   78.5%    |   75.8%     |   70.2%    |    1.6%      |     **6.0%**    |     1.2%        |
| Mixed items                            |   22.5%    |   19.1%     |   19.4%    |    3.2%      |    **18.2%**    |     0.1%        |
| **mean mAP (paper-10)**                | **38.2%**  |  **38.5%**  | **36.6%**  | **2.1%**     |   **13.3%**     |   **1.4%**      |
| AP@0.5                                 |     —      |     —       |     —      |    2.6%      |    16.2%        |     2.0%        |
| AP@0.75                                |     —      |     —       |     —      |    1.1%      |     8.7%        |     0.7%        |

For DinoSeg we additionally report semantic-seg quality (no equivalent in the paper detectors):

| | mIoU foreground | mIoU incl. bg | bg IoU | best class IoU |
|---|---:|---:|---:|---:|
| **DinoSeg** | **0.372** | 0.401 | 0.975 | Pallets 0.80, Asbestos 0.78, Textile 0.76, Tyres 0.72 |
| dino.txt zero-shot seg | 0.042 | 0.056 | 0.342 | Asbestos 0.28, Scrap 0.16, Pallets 0.15 |

**Read.**

1. **Head choice matters more than backbone-freeze.** Swapping the head from "DETR box regression" (DinoDETR) to "per-patch classification" (DinoSeg) on the *same* frozen DINOv2-B backbone moves paper-10 mean mAP from **2.1% → 13.3% (6.3×)** with **400× fewer trainable params** (16 k vs 6.7 M). Box regression from frozen patch features is the bottleneck; per-pixel classification followed by connected-components is far more honest about what DINOv2 can deliver.

2. **DinoSeg closes ~⅓ of the gap to YOLO with only 16k trained parameters.** 13.3% paper-10 mAP vs YOLOv8's 38.2%. On Vehicles, DinoSeg (23.8%) actually beats YOLOv8 (5.7%) and YOLOv12 (1.6%) — frozen DINOv2 semantics is more reliable than YOLO's detection-from-scratch on visually-distinctive heterogeneous classes the detectors struggle with. The classes where the gap remains large (Pallets, Tyres, Asbestos, Textile, with Asbestos at 47.6% mAP vs paper's ~76%) are cleanly-bounded instance categories where bbox-supervised end-to-end training is genuinely advantageous.

3. **Zero-shot dino.txt seg is weak in absolute terms (1.4% paper-10 mAP)** but the per-class story is informative: Pallets reaches 8.2% mAP with **zero training**, beating supervised DinoDETR's 0.9% on that class. Pallets are visually distinctive (rectangular wooden grids) and have unambiguous text descriptions, so the CLIP-style alignment lands. On classes with vague names (Plastic, Foundry, Furniture), zero-shot collapses. This is the open-vocab failure mode: text needs to disambiguate visual concepts well enough for cosine similarity to pick out the right patches.

4. **Resolution remains the ceiling for the cheap heads.** All three frozen-backbone variants operate at 518² (37×37 patch grid, 14-px stride). Small objects (DroneWaste's typical 50-pixel debris items) span well under one patch token — both supervised and zero-shot heads bottom out for those. The paper detectors handle small objects via FPN + multi-resolution training; matching that requires either (a) unfreezing DINOv2 and going multi-scale, or (b) tile-based inference at higher resolution.

**Implication for the §2 pipeline.** Keep the backbone frozen, but **prefer the segmentation head over the detection head** for any spatial information we need to feed the LLM. The structured note can now include both `P(class)` *and* coarse spatial location (`asbestos in the south-west quadrant, mIoU 0.78`) at zero additional training cost. dino.txt zero-shot remains useful for **out-of-taxonomy queries** — when an LEA inspector asks about a waste type not in our 20-class menu, we can still produce a heatmap by prompting dino.txt with their phrase.

### 3.5.7 Verdict and revised plan

- **Vision-side ceiling is high enough without encoder fine-tuning.** Frozen DINOv2-B + linear probe gets within 1–3 pp of the supervised binary bar on AW (89.1 % vs 90–92 %) and posts **micro F1 0.60 on the official 5-class AW mcml/m2 multi-label**, beating the best zero-shot VLM (LLaVA-NeXT 0.26) by +34 pp. On DroneWaste, image-level AP 0.57 is at-or-near the detectors' implicit image-level capability.
- **dino.txt (CVPR 2025) is the strongest zero-shot baseline by a wide margin.** Frozen DINOv2-ViT-L/14 aligned to text via CLIP-style contrastive training, no parameters fine-tuned on waste data, hits **micro F1 0.37 (AW m2) / 0.38 (AW m4) / 0.40 (DW paper-10)** — between supervised DINOv2 probe and all four CLIP-based VLMs. Crucially, it is the only zero-shot baseline that handles "Unknown material" (F1 0.31 vs 0.00 elsewhere) because open-vocabulary prompts can substitute `unidentified waste material` for the bare label.
- **Zero-shot CLIP-based VLMs are not competitive on multi-label.** Best is LLaVA-NeXT at F1 0.26 on AW m2 / 0.18 on DW paper-10. GeoChat collapses to F1 ≈ 0.03 (refuses to enumerate from a fixed class menu). GeoLLaVA-8K → 0.00 on every multi-label split (task-format OOD; see §3.5.2). The generative LLM head adds noise and instruction-following overhead — once we have dino.txt, the "VLM as multi-label classifier" pattern is uniformly worse.
- **Segmentation > detection on this frozen-backbone setup.** DinoSeg (linear head on frozen DINOv2-B, 16 k trainable params) hits **13.3% paper-10 mAP** vs DinoDETR's 2.1% — a 6× jump from changing the head, no backbone change. With 92 min of training. See §3.5.6. We now have a per-class spatial heatmap to feed the LLM, not just image-level class scores.
- **No need to fine-tune the encoder yet.** LoRA on DINOv2 is the natural next lever if (a) the supervised binary bar must be exceeded (> 92 %), or (b) the small-support classes in m4 / fine-grain matter operationally. For the current operational ask (binary triage + 5-class summary + coarse spatial localisation) the frozen probe + seg head + dino.txt combination is already enough.

Concrete next step: build the **(DINOv2 probe + DinoSeg + dino.txt) → text → LLM** pipeline. Convert the three signals into a structured text note (`P(waste) = 0.83, top categories: Bulky items (0.76), Containers (0.67), Unknown material (0.67); largest detected region: rubble pile in NW quadrant covering ~22% of image; open-vocab matches: "asbestos roofing sheet" (0.32)`), inject alongside the image in an off-the-shelf LLM's prompt, and re-run Q1–Q4 to see whether grounding closes the zero-shot gap on free-text severity / area-type / spatial-localisation questions.

### 3.5.8 Open architectural questions

The findings above leave two design choices that can be revisited later:

- **Vision backbone choice**: DINOv2-B at 89.1% binary makes RemoteCLIP / SatMAE / GeoChat-CLIP comparisons interesting but not urgent. Defer until the text-level pipeline is built and we know which axis needs improvement.
- **LLM scope**: the §2 plan does dense LEA fine-tune *then* pruning *then* alignment. Under the text-level-interface scheme, the LLM stays off-the-shelf (or LEA-fine-tuned but not pruned/aligned with vision) and just consumes detector output. This collapses §2 stages 1–4 into a much smaller LLM-side investment. Decide after the text-level pipeline shows whether off-the-shelf paraphrasing is good enough.

### 3.5.9 Backbone-exploration TODO

Prioritised list of vision-backbone experiments. Ordered by expected impact-per-cost — the ablation in §3.5.6 already showed scale alone (DINOv2-B→L) barely moves the headline, so capacity upgrades sit *below* the resolution axis.

1. **[done — see §3.5.10] Resolution + patch grid.** Hypothesised in §3.5.6 to be the ceiling for cheap heads. Tested via PCA sweep (512/768/1024 → 32²/48²/64² grids; `figures/pca_dinov3_resolutions.png`) and DinoSeg-DINOv3 probes at 512² and 1024². Result: **HR slightly hurts test mIoU** (DINOv3 0.410 @1024 < 0.419 @512) — the 32 × 32 token grid already covers DroneWaste's median 62-px objects. Axis closed; the ceiling is the head and the encoder, not the resolution.
2. **[deferred] DINOv3-g (ViT-g, ~1.1 B).** The giant variant. The DINOv2 B → L result (0 pp paper-10 mAP) is the cautionary precedent, and §3.5.10 now confirms scale is not the bottleneck on this dataset. Below "tile-then-aggregate" and "head-side augmentation" in priority. Single PCA + one seg-probe run still suffices if we ever revisit.
3. **[done — see §3.5.10] RADIOv2.5 (AM-RADIO line, CVPR 2025).** NVIDIA agglomerative model — distils CLIP + DINOv2 + SAM into one ViT, purpose-built as a **VLM vision encoder**: a language-aligned summary token *and* dense spatial features from a single backbone. Unlike DINO (pure SSL, no native text alignment) and unlike our dino.txt setup (text alignment bolted on via a separately-trained head), RADIO bakes the CLIP alignment in through distillation. Candidate to **replace the (DINOv2 probe + dino.txt) pair with one encoder**. Test with the same protocol as DINOv2/DINOv3: zero-shot classification, linear seg probe, PCA viz.

### 3.5.10 Backbone × head × resolution ablation

§3.5.6 fixed the head (linear, DINOv2-B, 518²) and reported 13.3 % paper-10 mAP as the DinoSeg baseline. This section sweeps the three remaining axes — **backbone family** (DINOv2 vs DINOv3 vs RADIOv2.5-L), **head capacity** (linear vs 3×3+1×1 conv ≈ 1.8 M params, 3-patch RF), and **input resolution** (512–518² vs 1024–1036²) — to identify which axis actually moves the needle. Same site-stratified test split as §3.5.6, same `pycocotools` paper-10 mAP, same frozen-backbone protocol. RADIO conv and RADIO HR runs are still in the SLURM queue and will be appended once their `test_eval.json` lands.

| Backbone | Res. | Head | Trainable | val mIoU | test mIoU_fg | paper-10 mAP | AP50 |
|---|---:|---|---:|---:|---:|---:|---:|
| DINOv2-B (baseline §3.5.6) | 518² | linear | 16 k | 0.409 | 0.372 | 13.3 % | 16.2 % |
| DINOv2-L | 518² | linear | 22 k | — | 0.384 | 13.3 % | 17.7 % |
| DINOv2-B HR | 1036² | linear | 16 k | — | 0.390 | 13.6 % | 16.8 % |
| DINOv3-B | 512² | linear | 16 k | 0.457 | **0.419** | 14.7 % | 17.1 % |
| DINOv3-B HR | 1024² | linear | 16 k | 0.449 | 0.410 | 15.4 % | 16.9 % |
| DINOv2-B + conv head | 518² | conv | 1.8 M | 0.466 | 0.416 | 15.5 % | 18.6 % |
| DINOv3-B + conv head | 512² | conv | 1.8 M | 0.486 | 0.408 | 14.1 % | 17.0 % |
| **RADIOv2.5-L** | **512²** | **linear** | **22 k** | **0.457** | **0.416** | **16.1 %** | **19.5 %** |
| RADIOv2.5-L + conv head | 512² | conv | 1.8 M | 0.484 | **0.427** | 15.4 % | 19.1 % |
| RADIOv2.5-L HR | 1024² | linear | 22 k | 0.438 | 0.417 | 15.8 % | 18.7 % |

**Read.**

1. **Scale and resolution barely move DINOv2.** B → L adds ~1 pp mIoU (0.372 → 0.384) and 0 pp paper-10 mAP. 518² → 1036² adds another +0.6 pp mIoU and +0.3 pp mAP. Both deltas are inside the run-to-run noise we'd expect from a 16 k-parameter head on 555 test images. Capacity and pixels are not the bottleneck inside the DINOv2 family.

2. **DINOv3-B is the first backbone change that materially helps.** Same head, same resolution, same trainable count, **+4.7 pp test mIoU** (0.372 → 0.419) and **+1.4 pp paper-10 mAP** (13.3 → 14.7 %) over DINOv2-B. The PCA already foreshadowed it: PC1 explains 14.2 % of patch-feature variance for DINOv3 vs 7.8 % for DINOv2 (`figures/pca_dinov2_vs_dinov3.png`), and the per-component maps in `figures/nmf_channels_dinov2_vs_dinov3.png` show DINOv3 components aligning with object parts where DINOv2 components mostly track texture. The Gram-anchoring objective DINOv3 adds to keep dense features locally consistent translates into a real probe gain.

3. **Higher resolution still does not help.** DINOv3 @1024² loses 0.9 pp test mIoU vs @512² (0.410 < 0.419) at 4 × the inference cost. Paper-10 mAP inches up (+0.7 pp) but AP50 falls. This contradicts §3.5.6's "resolution remains the ceiling" hypothesis: with DroneWaste tiles at 640², median object √area ≈ 62 px, even a 32 × 32 token grid covers each object with ≥ 1 patch. Once each object owns a token, adding tokens just inflates the GT-bg ratio per token. The resolution sweep figure `figures/pca_dinov3_resolutions.png` (512/768/1024) shows the components stabilise by 512 and gain nothing visually at 1024.

4. **The conv head helps DINOv2 but overfits with DINOv3.** Replacing the linear head with `Conv3×3 → GELU → Conv1×1` (1.8 M params, 3-patch RF) on DINOv2-B lifts paper-10 mAP from 13.3 → 15.5 % (+2.2 pp, the single biggest gain on the DINOv2 baseline). On DINOv3-B the same head *drops* test mIoU 0.419 → 0.408 (−1.1 pp) and paper-10 mAP 14.7 → 14.1 %, despite climbing on val (0.457 → 0.486). 8-pp val→test gap. The mechanism is now visible: a conv head adds local-context smoothing, which compensates for DINOv2's noisier per-patch features, but **DINOv3 already has that smoothing built into its encoder** (Gram anchoring) — extra head capacity becomes dead weight that the optimiser uses to memorise training tiles. **Use the conv head only when the encoder needs it.**

5. **RADIOv2.5-L is the new test-set leader.** Frozen RADIO-L @512² + linear head reaches **0.416 mIoU, 16.1 % paper-10 mAP, 19.5 % AP50** — best paper-10 mAP and best AP50 of the linear-head runs, with the same 22 k trainable parameters as DINOv2-L. Notable because RADIO's PCA is visibly grainier than DINOv3's (`figures/pca_dinov2_dinov3_radio.png`): the joint-PCA RGB doesn't cleanly separate waste objects from background. The supervised head pulls out a signal the unsupervised joint decomposition hides — RADIO is distilled from CLIP + DINOv2 + SAM, so its features carry SAM-style boundary cues that the linear classifier can exploit even though PCA discards them as low-variance. The AP50 jump (17.1 → 19.5 %) is the smoking gun: better object boundaries → tighter connected-components → tighter bboxes.

6. **The conv head behaves differently on each backbone, and the pattern reveals the bottleneck.** Three frozen backbones × {linear, conv head} now in hand:

   | backbone | val Δ (linear→conv) | test mIoU Δ | test paper-10 mAP Δ | verdict |
   |---|---:|---:|---:|---|
   | DINOv2-B | +0.057 | **+0.044** | **+2.2 pp** | conv head clearly helps |
   | DINOv3-B | +0.029 | **−0.011** | **−0.6 pp** | conv head overfits (val→test gap 8 pp) |
   | RADIOv2.5-L | +0.027 | **+0.011** | **−0.7 pp** | mixed: better mIoU, worse mAP |

   Reading the rightmost column: the conv head's 1.8 M parameters only pay back on DINOv2, the encoder with the noisiest per-patch features. On DINOv3 the encoder has Gram anchoring built in, so the head's extra capacity becomes memorisation. RADIO sits in between — its CLIP/SAM-distilled features are already locally consistent enough that the conv head adds little, and the val→test gap (5.7 pp) is smaller than DINOv3's (8 pp) but not zero. **Conclusion: head capacity is a substitute for encoder feature quality, not a complement.** Adding a conv head to a clean encoder doesn't stack.

   The RADIO-L+conv test mIoU (**0.427**) is the highest of all 9 configs, but its paper-10 mAP (15.4 %) and AP50 (19.1 %) are both *below* RADIO-L+linear. The conv head smooths the segmentation map enough to raise mIoU, but the smoothing also merges neighbouring connected components, which hurts the downstream detection-from-mask step. **For DroneWaste paper-10 mAP, the headline configuration stays RADIO-L @512² + linear head (16.1 %).**

**Stack-up.** Three cheap interventions actually pay off on DroneWaste seg-probes: (a) **upgrade the encoder** (DINOv2-B → DINOv3-B or RADIOv2.5-L, free at inference time, 16 k–22 k trainable params); (b) **only add the conv head with DINOv2** — on cleaner encoders (DINOv3, RADIO) it overfits on val and hurts paper-10 mAP on test; (c) **stay at 512².** Scale (B → L) and resolution (512 → 1024) do not move the headline on either DINOv3 or RADIO. The §3.5.9 TODO collapses: RADIOv2.5 is now done (it leads paper-10 mAP) and DINOv3-g is unlikely to help given the DINOv2-B → L precedent — it sits below "head-side augmentation" and "tile-then-aggregate at higher effective resolution" in priority.

### 3.5.11 Encoder choice

**Chosen encoder: RADIOv2.5-L @ 512² + linear head.**

RADIOv2.5-L reaches **16.1% paper-10 mAP / 19.5% AP50** on the DroneWaste frozen-backbone seg probe — the best of all nine backbone × head × resolution configurations in §3.5.10, with only 22 k trainable parameters and the same 512² input as all other runs. It leads DINOv3-B (14.7% mAP) by +1.4 pp on the headline detection metric and by +2.4 pp on AP50. The gap is driven by RADIO's SAM-distilled boundary cues: its AP50 jump (DINOv3-B 17.1 → RADIO-L 19.5) directly reflects tighter connected-components from cleaner object boundaries in the patch feature map. For the VLM track, RADIO's language-aligned summary token (distilled from CLIP) means the encoder simultaneously supports dense spatial grounding (patch tokens → seg head) and global semantic retrieval (summary token → projector → LLM) with a single backbone — the (DINOv2 probe + dino.txt) pair collapses into one model.

Encoder identifier for downstream use: `"radio-l"` in `src.vision_encoder.VisionEncoder`. Local weights at `results/waste_vlm/weights/RADIO-L/`. Image size: 512², patch size 16, patch grid 32×32 = 1024 tokens. Actual output dims: **summary token 3072-dim** (language-aligned aggregated representation), **patch tokens 1024-dim** (ViT-L hidden size). The VLM track imports this encoder via `VisionEncoder("radio-l")`, reads `.cls` for the summary token and `.patches` for spatial features, and keeps the backbone frozen during projector training.

Linear-probe VQA-split numbers will be appended here once `data/vqa/split_index.jsonl` is produced by the ANNOTATION track and `src.encoder_probe` is run. The binary AW and DroneWaste multi-label numbers from §3.5.5 and §3.5.10 already establish that frozen RADIO-L features are competitive without any waste-specific fine-tuning.

---

## 4. Research directions for the architecture

These are the architectural contributions that distinguish our pipeline from existing RS VLMs. They are deferred until baselines are complete, but framed here so we know what to test for in the baseline analysis.

### 4.1 Scene-adaptive token compression

GeoLLaVA-8K uses Background Token Pruning + Anchored Token Selection: drops uninformative visual tokens (ocean, forest) before they reach the LLM. This is correct for **sparse scenes** (one suspect site against a wide rural backdrop).

For **dense scenes** (zoomed-in candidate site full of heterogeneous debris), pruning drops discriminative signal. **Token merging** (ToMe-style, similarity-based) preserves more information when the dropped tokens are partially redundant rather than uninformative.

**Proposal**: a two-stage scene-adaptive compression — pruning at the wide-area triage tier, merging at the candidate-region classification tier. A cheap upfront classifier decides which regime applies. This aligns with how ARPA inspectors work (broad scan, then zoom).

Empirical hook: compare pruning, merging, and hybrid on AerialWaste full tiles vs. the positive-only subset. If the hybrid wins the combined task, this is a publishable contribution that improves on GeoLLaVA-8K's specific weakness.

### 4.2 Cross-attention vs concatenation for visual tokens

LLaVA's concatenation paradigm (visual tokens prepended to text in the LLM) won the general-purpose race in 2023–2024 but is increasingly questioned for high-resolution and multi-image settings — see mPLUG-Owl3, EVE, Mini-Gemini.

The argument that "concatenation lets the LLM reason richly over visual tokens" assumes the LLM's reasoning capacity isn't saturated by visual sequence length. At >1000 visual tokens (which UHR aerial imagery routinely produces even after compression), that assumption breaks: most of the LLM's attention budget gets spent on vision-vision interactions, which is the vision encoder's job.

For aerial waste analysis, queries are extractive and structured ("is there waste? what type? what severity?"), not free-form open visual reasoning. Cross-attention fits this regime better.

Cross-attention also composes more cleanly with structural pruning: the cross-attention layers are *new* layers added to the LLM, not part of original Mistral self-attention blocks. Pruning can target original self-attention (where the NeurIPS method was validated) without affecting the more sensitive vision-language alignment layers. With concatenation, visual and text reasoning share parameters, and pruning hits both indiscriminately — possibly a contributor to the factual-recall dissociation observed in the NeurIPS paper.

**Decision criterion**: if the baseline visual token count after compression is reliably under ~500, concatenation is fine. If it routinely exceeds 1000, switch to cross-attention.

### 4.3 (Future) Temporal change detection

Illegal dumping is fundamentally a temporal phenomenon — sites accumulate waste over months. The Elgendy et al. GeoLLaVA (the change-detection one, distinct from GeoLLaVA-8K) fine-tunes Video-LLaVA on fMoW image pairs across years. ARPA already has multi-temporal AGEA orthophotos for Italy.

A combined "high-resolution + temporal" VLM for waste monitoring would be a genuinely novel contribution — neither GeoLLaVA paper does both. Defer until the static pipeline works; flag now as a planned extension.

### 4.4 Structured output schema

Some outputs must be exact, not paraphrased: statute references, CER waste codes, ARPA severity levels, controlled-vocabulary classifications. These should be **extractive** (retrieved from inputs or selected from a fixed list) rather than generated.

Output schema sketch:
```json
{
  "waste_detected": "yes|no|uncertain",
  "confidence": 0.0,
  "free_text_summary": "...",                 // free-form Italian or English
  "primary_waste_class": "CER_code_from_list",  // controlled vocabulary
  "severity_estimate": 0|1|2|3,                 // ARPA scale, controlled
  "evidence_indicators": ["...", "..."],        // controlled
  "area_type": "...",                           // controlled
  "statute_references": ["art. 256, D.Lgs. 152/2006", ...],  // extractive
  "spatial_localization": "..."
}
```

Constrained decoding handles the controlled-vocabulary fields; the model is freed to generate naturally only in the summary field. This keeps the parts that must be precise out of the generation pathway.

---

## 5. Datasets

| Dataset | Size | Source | Use |
|---|---|---|---|
| AerialWaste v2.0 | 10,434 images | Politecnico di Milano + ARPA Lombardia | Primary training and evaluation |
| DroneTrashNet | varies | UAV-collected | Secondary UAV evaluation |
| ARPA inspection reports | TBD | ARPA Lombardia (subject to availability) | LEA fine-tuning corpus |
| D.Lgs. 152/2006 + jurisprudence | ~ | Italian legal databases | LEA fine-tuning corpus |
| EU Waste Catalogue (CER/EWC) | ~ | EU | Controlled vocabulary |
| Italian instruction-following sets | varies | Open Italian LLM community | Calibration mix (Italian portion) |
| OpenOrca / ShareGPT subset | varies | Public | Calibration mix (English portion, dialog preservation) |

The PERIVALLON Horizon Europe project has direct relevance and existing connections to ARPA Lombardia and Politecnico di Milano. Worth coordinating to avoid duplicating dataset effort.

---

## 6. Evaluation methodology (beyond the zero-shot baseline)

Once the pipeline is built, evaluate on:

**Detection / classification metrics on AerialWaste**: precision, recall, F1, confusion matrix on waste classes, calibration metrics.

**Operational metrics**: false-positive rate per km² scanned (how much human-review time does the system save vs cost?), time-to-detection on simulated incident streams.

**Hallucination / faithfulness**: rate of statute references that exist vs invented; rate of waste-class assignments outside the controlled vocabulary; rate of spatial claims unsupported by image content.

**Italian comprehension probes**: a held-out set of ARPA-style Italian queries with known correct answers, measuring whether the pruning + alignment pipeline degraded Italian understanding from the dense LEA-Mistral baseline.

**Hardware-agnostic latency**: measured on at least three deployment targets (workstation GPU, consumer GPU without sparse cores, Jetson-class). The whole point of structured pruning is universal speedup; verify it.

---

## 7. Schedule

**Weeks 1–2**: Three-baseline zero-shot comparison on stratified AerialWaste sample. Deliverable: baseline report split by image source, with calibration metrics, that informs which foundation to build on.

**Weeks 3–4**: Implement scene-adaptive token compression on top of chosen foundation. Compare against pure pruning and pure merging. Deliverable: token-compression ablation.

**Month 2**: Decide cross-attention vs concatenation based on visual token counts after compression. If switching, implement and re-baseline.

**Month 3**: Stage 1 — dense LEA fine-tune of Mistral. Build LEA calibration corpus. Build dense LEA-Mistral teacher.

**Month 4**: Stage 2 — apply structural pruning with LEA-heavy calibration. Stage 3 — recovery via LoRA + distillation from dense teacher. Verify Italian comprehension hasn't regressed.

**Month 5**: Stage 4 — LLaVA-style alignment with DINOv2 on AerialWaste. Stage 5 — instruction tuning on waste-specific VQA built from AerialWaste metadata (GeoChat-style synthetic conversation pipeline).

**Month 6**: End-to-end evaluation, hardware-agnostic latency validation, documentation.

---

## 8. Key references

### Application domain
- Torres & Fraternali, "AerialWaste dataset for landfill discovery in aerial and satellite images", *Scientific Data* 2023. https://www.nature.com/articles/s41597-023-01976-9
- Torres et al., "A Deep Learning Pipeline for Solid Waste Detection in Remote Sensing Images", arXiv 2502.06607, 2025.
- Persello et al., "Illegal Microdumps Detection in Multi-Mission Satellite Images", *IEEE JSTARS* 2024 (Campania context).
- Sun et al., "Revealing influencing factors on global waste distribution via deep-learning based dumpsite detection from satellite imagery", *Nature Communications* 2023.
- YOLO-SAT, arXiv 2511.03888, 2025 (UAV-deployable waste detection).

### RS Vision-Language Models
- Kuckreja et al., "GeoChat: Grounded Large Vision-Language Model for Remote Sensing", CVPR 2024. https://github.com/mbzuai-oryx/GeoChat
- Wang et al., "GeoLLaVA-8K: Scaling Remote-Sensing Multimodal Large Language Models to 8K Resolution", NeurIPS 2025 Spotlight. https://github.com/MiliLab/GeoLLaVA-8K
- Elgendy et al., "GeoLLaVA: Efficient Fine-Tuned Vision-Language Models for Temporal Change Detection", arXiv 2410.19552, 2024 (temporal version, distinct from GeoLLaVA-8K).
- Bazi et al., "RS-LLaVA", *Remote Sensing* 2024.
- Liu et al., "RemoteCLIP: A Vision Language Foundation Model for Remote Sensing", 2023.
- Liu et al., "Remote Sensing SpatioTemporal Vision-Language Models: A Comprehensive Survey", arXiv 2412.02573, 2024–2025.
- Awesome list: https://github.com/ZhanYang-nwpu/Awesome-Remote-Sensing-Multimodal-Large-Language-Model

### DINOv2 in MLLMs
- Jiang et al., "From CLIP to DINO: Visual Encoders Shout in Multi-modal LLMs" (COMM), arXiv 2310.08825, 2023.
- Zhang et al., "Ferret-v2: An Improved Baseline for Referring and Grounding with Large Language Models", arXiv 2404.07973, 2024.
- "DINOv2 Meets Text" (dino.txt), arXiv 2412.16334, Meta 2024.
- VIRAL, arXiv 2025 — visual representation alignment regularization for MLLMs.

### Pruning and compression
- "Don't carry that weight: learning-based pruning for transformer architectures", NeurIPS 2026 submission (the routine adopted here).
- ATP — "All-in-One Tuning and Structural Pruning for Domain-Specific LLMs", arXiv 2412.14426, 2024.
- SparseGPT (Frantar & Alistarh, 2023), Wanda (Sun et al., 2024) — unstructured baselines.
- Bolya et al., "Token Merging: Your ViT But Faster" (ToMe), ICLR 2023.
- LLaVA-PruMerge, FastV, VisionZip — visual token compression for MLLMs.

### Cross-attention vs concatenation
- mPLUG-Owl3 (Hyper Attention), 2024.
- EVE, Mini-Gemini, 2024.
- Original Flamingo, BLIP-2, LLaVA papers.

### Italian / multilingual LLMs
- Mistral-7B and Mistral-Nemo, Mistral AI.
- Minerva (Sapienza NLP), 2024.
- LLaMAntino (Università di Bari), 2023–2024.
- Qwen2.5, Alibaba, 2024.