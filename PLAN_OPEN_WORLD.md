# Open-world detection and segmentation of waste — plan

## Where this starts from

Settled by the experiments in `EXPERIMENTS.md`:

- **Waste/no-waste detection works.** Frozen-feature probe AUC 0.970; the VLM
  reaches micro-F1 0.601 once the readout is a Yes/No margin rather than free text.
- **The decision is genuinely grounded.** Occluding the waste moves the margin
  −1.782 against +0.086 for a matched control edit (p=8e-26, 84% of images).
- **Object scale governs localisation.** Grounding DINO, no waste training:
  0.819 box recall on DroneWaste against 0.150 on AerialWaste, tracking
  7.96 against 0.90 visual tokens per object.
- **Material naming works on DroneWaste and not on AerialWaste.** GeoRSCLIP on
  ground-truth crops: 4.89× chance macro-recall over 20 classes, against 1.54×
  over 5 on AerialWaste, where every readout sits at or below the majority baseline.
- **Off-the-shelf grounded VLMs fail at AerialWaste's scale.** Kosmos-2 returns
  language-prior captions; GeoChat sits at the random-placement floor and fires
  on 100% of negatives.

So the thesis claim is: *detection on aerial imagery, and on drone imagery the
same pipeline both detects and identifies materials.* Everything below builds the
second half out to open-world detection and segmentation.

## What DroneWaste actually gives us

| | |
|---|---|
| images | 4993, 640×640 |
| annotated images | **1171** (the other 3822 need checking — negatives or unlabelled?) |
| objects | 5135, **every one carrying a segmentation mask** |
| categories | 20, mapped to EWC-Stat codes |
| sites | 17, recorded per image |

Two structural facts drive the whole design.

**Splits must be by site, never by image.** Images from one site are near
duplicates of each other; an image-level split leaks. Site sizes are very uneven
(site16 has 848 images but 186 objects; site14 has 279 images and 792 objects),
so the split has to be built by object count, not image count.

**The class tail is severe.** Usable (≥100 objects): Pallets 1016, Textile 876,
C&D materials 397, Scrap 394, Mixed items 365, Plastic packaging 342, Tyres 311,
Asbestos 210, Furniture 189, Plastic 186, Vehicles 178, Wood 175, Metal barrels
172, Rubble 161, Excavation materials 102. Unusable: Appliances 35, Paper 11,
Electronic equipment 11, Foundry 3, Asphalt milling 1. **15 classes, not 20** —
reporting a 20-way number invites a reviewer to ask about the class with one
instance.

The EWC-Stat codes give a real hierarchy, which is what makes "open-world" more
than a slogan here:

| parent | leaves |
|---|---|
| 06 metallic | Scrap 06.11, Metal barrels 06.31 |
| 07 packaging / consumables | Paper 07.2, Tyres 07.31, Plastic packaging 07.41, Plastic 07.42, Pallets 07.51, Wood 07.53, Textile 07.6 |
| 08 discarded equipment | Vehicles 08.12, Appliances 08.21, Electronic 08.23 |
| 10 household / mixed | Furniture 10.11, Mixed items 10.2 |
| 12 mineral | C&D 12.11, Asphalt milling 12.12, Asbestos 12.21, Excavation 12.31, Foundry 12.42, Rubble 12.61 |

A held-out leaf can be queried by its own name, by a paraphrase, or by its parent
category — three difficulty levels from one split.

## Phase 0 — close out what is running (~6 GPU-h)

Finish the attribution maps and the ROI-token probe, fold both into
`EXPERIMENTS.md`. Nothing downstream depends on them; they close arguments
already made.

**Decide the negatives question first**: if the 3822 unannotated images are true
negatives, they are training background and a source of false-positive
measurement. If they are merely unlabelled, training on them as background
teaches the detector to suppress real waste. This gates Phase 1 and costs an hour
of inspection, not GPU.

## Phase 1 — closed-set detection and segmentation (~30 GPU-h)

The reference ceiling everything open-world is measured against. Without it, an
open-vocabulary number has no denominator.

- Mask R-CNN and a DETR-style detector, site-split, 15 classes.
- Report box AP/AP50/AP75, mask AP, and per-class AP so the tail is visible.
- Run the same detector on AerialWaste for the transfer claim. Expect it to be
  poor, and say so — that is the object-scale result restated in detection terms.

**Deliverable:** the number a reviewer compares every later arm to.

## Phase 2 — open-vocabulary detection (~40 GPU-h)

Two arms, deliberately different in where the openness lives.

**A. Class-agnostic proposals + RS-CLIP naming.** SAM or a class-agnostic
detector proposes regions; GeoRSCLIP names them from text. This is the ceiling
experiment with predicted boxes instead of ground-truth ones, so its upper bound
is already measured (4.89× chance) and the gap to it is exactly proposal quality.
Cheap, strong, and interpretable.

**B. Text-conditioned detector.** Fine-tune Grounding DINO on DroneWaste with
category names as queries. It already reaches 0.819 recall zero-shot here, so
this arm should be strong; the question is whether fine-tuning on 15 names
destroys the open-vocabulary behaviour that made it work in the first place.
Measure that directly rather than assuming it.

**Open-world protocol.** Hold out 3–4 usable classes from training entirely
(candidates: Tyres 311, Asbestos 210, Wood 175, Metal barrels 172 — chosen to
span different EWC parents). At test, query them by name. Report base AP and
novel AP separately, plus the three query levels (own name / paraphrase /
parent). Baseline to beat: proposals scored by a *fixed* class prior, which
measures how much of "novel detection" is really just proposal quality.

## Phase 3 — segmentation (~20 GPU-h)

Masks exist for all 5135 objects, so this is free supervision we are currently
ignoring.

- Box→mask via SAM (GeoGround already ships this path) as the training-free arm.
- A learned mask head as the supervised arm.
- Report mask AP against box AP; where they diverge tells you whether the model
  finds objects or merely regions.

Segmentation also gives a better attribution target than boxes: mass-inside-mask
is a tighter test than mass-inside-box, and the machinery in `src/attribution.py`
takes a mask already.

## Phase 4 — the language layer, on object tokens (~40 GPU-h)

This is the original "separate perception from language" design, and it now has
evidence behind it rather than intuition. The LLM never needed to *find* the
object — at 0.90 tokens per object on AerialWaste it could not — it needs to talk
about one that has already been found.

- Feed pooled region features plus the predicted class as object tokens.
- Evaluate description quality and VQA, and re-run the hallucination battery
  (§8: crop / mask / shuffle / irrelevant-region / occlusion ΔP) on the new stack
  using the existing occlusion and attribution code.

The specific thing to test: whether per-category attribution maps become distinct
once the model is given object tokens. On the current stack they are not, and
that is the measured reason naming fails — 72% of the category effect is presence
rather than identity.

## Phase 5 — grounding verification as a first-class result (~10 GPU-h)

Reuse `scripts/attribution_maps.py` and the occlusion probe against the final
pipeline, reporting mass-in-mask lift over the uniform and centre nulls. This is
the part of the thesis that distinguishes "the model is right" from "the model is
right for the right reason", and the machinery already exists.

## Budget

~150 GPU-hours across all five phases, against 750 remaining. Comfortable, with
room for the ablation matrix **provided ablation arms are stage-3 finetunes
(~8–10 h each) rather than full pipelines (~110–150 h each)**. Six full-pipeline
arms would be 900 hours and would need the other cluster; the same six as
stage-3 arms are 60.

## Risks, stated up front

- **1171 annotated images is small** for 15-class detection. Expect wide
  confidence intervals on novel classes; report them rather than point estimates.
- **Novel-class AP on ~200 instances is noisy.** Use several held-out splits and
  report the spread, not one favourable draw.
- **17 sites is few.** A site-level split has high variance; cross-validate over
  site folds rather than trusting a single split.
- **AerialWaste material labels carry unmeasured annotator noise**, and 25.8% of
  its objects are labelled "Unknown material". Any AerialWaste material number is
  bounded by a target its own annotators could not assign a quarter of the time.
  A small re-annotated subset would let that bound be quantified instead of
  asserted; without it, AerialWaste material results should be reported as a
  limitation rather than a finding.
