# Training-free open-world detection and segmentation of waste — plan

Every weight this needs is already on disk. Nothing here is trained.

## Why training-free is a real option and not a compromise

Each component's ceiling has already been measured on these datasets, so the
pipeline's performance can be predicted before it is built:

| component | role | measured on DroneWaste unless noted |
|---|---|---|
| Grounding DINO (base) | open-vocabulary proposals | **0.819** box recall @ IoU 0.5, zero waste training |
| SAM 2 (hiera-large) | box/point -> mask | masks exist for all 5135 objects, so mask AP is directly scorable |
| GeoRSCLIP / RemoteCLIP / SkyCLIP | naming, open vocabulary | **4.89x chance** macro-recall over 20 classes on GT crops |
| our VLM (Yes/No margin) | presence gate | micro-F1 **0.601**; grounded (occlusion p=8e-26) |
| `src/attribution.py` | where it looked | mass-in-box lift **+0.338**, 8.5x a centre prior |

The 3822 unannotated DroneWaste crops are **true negatives** -- hand-annotated
sites, then cropped -- which makes false-positive rate measurable rather than
assumed. That matters more than it sounds: Grounding DINO returns a box on
essentially every negative it is shown. A detector that cannot say "nothing here"
is not a detector, and the gate is the part that fixes it.

## The architecture, and why each part is where it is

```
image ─► Grounding DINO ─► boxes ─► SAM 2 ─► masks ─┬─► GeoRSCLIP ─► material name
             (open vocab)                            │      (open vocab, text)
                                                     └─► VLM Yes/No ─► keep / drop
                                                            (presence gate)
                              attribution map ◄─────────────┘  (why here)
```

Each part does the thing it is measured to be good at, and nothing is asked to do
the thing it is measured to be bad at. Specifically:

**The VLM localises; it does not name.** Attribution maps for "is there plastic"
and "is there rubble" correlate at **+0.791**, and each correlates with the plain
presence map at **+0.824**. The map does not respond to the content of the
question. So open-vocabulary naming cannot be had by varying the prompt on this
stack -- that route is closed by measurement, not by assumption -- and naming
belongs to the text-aligned encoder.

**The VLM's real job is the gate.** Presence is where it is strong (0.601 F1,
grounded), and suppressing proposals on 3822 true negatives is exactly a presence
problem. Detector proposes, VLM disposes.

## Phase 1 — class-agnostic detection and segmentation (~15 GPU-h)

The denominator for everything else.

- Grounding DINO with waste queries -> boxes; SAM 2 -> masks.
- Score box AP/AP50 and **mask AP** against DroneWaste's ground-truth masks.
- Score on the negatives: what fraction of the 3822 produce a proposal.
- Then add the VLM gate and re-score. The gate's whole value is the change in
  false positives per image at fixed recall, so report both points, not one.

**Sub-question worth its own arm:** SAM 2's automatic mask generation as a
proposal source, with no text at all. If class-agnostic SAM proposals plus the
VLM gate match Grounding DINO, the text-conditioned detector is not earning its
place and the pipeline gets simpler.

## Phase 2 — open-vocabulary naming (~10 GPU-h)

All training-free, all measurable against the 4.89x ceiling already established
on ground-truth crops. The gap between predicted-box naming and GT-crop naming
*is* proposal quality, which makes this a rare ablation that costs nothing.

Arms, in increasing order of cleverness:

1. **Box crop -> GeoRSCLIP.** The measured baseline.
2. **Mask crop -> GeoRSCLIP**, background suppressed rather than merely cropped.
   Untested here and plausibly a real gain for material ID, since it removes the
   surrounding ground that the box drags in.
3. **Context ratio.** ctx 0.5 beat ctx 0.0 on DroneWaste (+0.011 to +0.061) and
   hurt on AerialWaste. With masks available, context and object can be varied
   independently for the first time.
4. **Prompt ensembling** over multiple templates per class -- free, typically
   worth a couple of points, and `cue_prompts` already exists.
5. **Encoder ensembling** across GeoRSCLIP / RemoteCLIP / SkyCLIP.

**Open-world protocol.** The 20 categories carry EWC-Stat codes, which gives a
real hierarchy rather than an arbitrary split:

| parent | leaves |
|---|---|
| 06 metallic | Scrap 06.11, Metal barrels 06.31 |
| 07 packaging / consumables | Paper 07.2, Tyres 07.31, Plastic packaging 07.41, Plastic 07.42, Pallets 07.51, Wood 07.53, Textile 07.6 |
| 08 discarded equipment | Vehicles 08.12, Appliances 08.21, Electronic 08.23 |
| 10 household / mixed | Furniture 10.11, Mixed items 10.2 |
| 12 mineral | C&D 12.11, Asphalt milling 12.12, Asbestos 12.21, Excavation 12.31, Foundry 12.42, Rubble 12.61 |

Because nothing is trained there is no "held-out class" in the usual sense --
every class is already zero-shot. That is a genuine advantage, and it changes the
experiment into something better: **query the same object by its own name, by a
paraphrase, and by its EWC parent**, and report how far accuracy falls as the
query gets more abstract. Also query with names that are *absent* from the
dataset entirely, to measure how readily the vocabulary hallucinates.

### The long tail is the argument, not a nuisance

An earlier draft of this plan said to report 15 classes and drop the other five.
That was wrong, and backwards. The tail is the case *for* this approach.

| | classes | objects | can a detector be trained on it? |
|---|---:|---:|---|
| head (>=100 objects) | 15 | 5074 | yes |
| tail (<100 objects) | 5 | **61** | no -- Appliances 35, Paper 11, Electronic equipment 11, Foundry 3, Asphalt milling **1** |

A YOLO or Faster R-CNN cannot learn a class from one training instance; there is
nothing to fit. A text-driven pipeline needs zero instances, only the class name.
So the honest comparison, and the one that makes the case:

- Train the supervised detector on the head. Show that it scores ~0 on the tail,
  because it must.
- Run the zero-shot pipeline on all 20 without ever seeing any of them, and show
  non-trivial tail performance.
- Report **per-class results for all 20 with instance counts beside them**, plus
  a *pooled* tail figure. Pooled over 61 objects the tail is a real measurement;
  per-class at n=1 it is an anecdote, and should be labelled as one rather than
  quietly averaged into a macro score.

This also sharpens which naming arm carries the thesis. **A linear probe on ROI
tokens needs labels per class and therefore inherits exactly the detector's tail
problem.** Only the zero-shot text arm covers classes with no training instances.
The strongest system is likely a hybrid -- probe on the head where labels are
plentiful, text matching on the tail -- and the strongest *argument* is the tail
column, where the supervised baseline is structurally at zero.

Note that this makes the supervised detector a **baseline**, not a component. It
gets trained; the proposed system still does not. Budget ~8 GPU-h for a
Faster R-CNN / YOLO head-class baseline, and treat it as the thing to beat.

## Phase 3 — grounding verification (~8 GPU-h)

The part that separates "right" from "right for the right reason", and the
machinery exists.

- Attribution mass **inside the mask** rather than the box -- a strictly tighter
  test, and `score_map` already takes a mask.
- The occlusion battery against the assembled pipeline.
- Report every map against both nulls, uniform and centre, as now.

## Phase 4 — AerialWaste as the transfer test (~5 GPU-h)

Run the identical pipeline, unchanged, on AerialWaste. Expect class-agnostic
detection to work and naming to fail; that is the object-scale result restated
end to end, and it is a finding rather than a disappointment provided it is
framed as one.

## Budget

**~40 GPU-hours total**, against 750 remaining. Training-free is roughly a
quarter the cost of the trained plan it replaces (~150 h), and it removes the
risk that a training run consumes the budget and then has to be repeated.

## What training-free costs, stated plainly

Bounded by the ceilings in the table above. Recall cannot exceed Grounding DINO's
0.819; naming cannot exceed roughly 0.31 accuracy against a 0.198 majority
baseline. **Class-agnostic detection and segmentation should be genuinely good;
class-aware numbers will be modest.** If a stronger class-aware number is needed
later, the cheapest additions in order of cost are: prompt/encoder ensembling
(still training-free), a linear probe on frozen features (minutes), and only then
anything resembling a fine-tune.

## Risks

- **Proposal recall is the binding constraint** and it is measured on
  ground-truth-annotated images only. On the 3822 negatives the relevant number
  is the false-positive rate, which nobody has measured yet.
- **17 sites is few.** Cross-validate over site folds; never split by image,
  since crops from one site are near duplicates.
- **SAM 2 on 640px drone crops is untested here.** Small objects at 32-75px are
  where promptable segmentation is weakest; verify on a handful before scaling.

## The decoder as a composer over tools, not as a perceiver

Every measurement points the same way. Reading pixels through image tokens, the
7B decoder loses to a linear head on the same frozen features (naming 0.665/0.733
against a readout bounded at +0.043 over a constant predictor; the gate 0.029 AUC
behind the probe), and its attribution maps do not respond to the content of the
question (+0.791 between different category maps). It is not a perceiver.

What it has never been asked to do is compose an answer out of facts that
something else established. That is a different job and the evidence does not
speak against it, because nothing has tested it.

### Architecture

```
C-RADIOv4, one pass ─┬─ proposals   (Grounding DINO now; SAM3 decoder pending)
                     ├─ masks       (SAM3)
                     ├─ material    (ROI+linear on the head, SigLIP2 head on the tail)
                     └─ presence    (linear gate)
                                    │
                                    ▼
                        SCENE GRAPH: [{id, category, confidence,
                                       box, mask, area, centroid}]
                                    │
                                    ▼
                        decoder: question -> tool calls / program -> answer
```

The decoder plans and phrases; it never does arithmetic over a list, because code
does that better and auditably. This is the visual-programming family
(VisProg/ViperGPT) rather than the LLaVA family, and it is what the numbers have
been arguing for since the attribution maps came back.

### The evaluation that makes it a result

Four arms on the same compositional query set, chosen so perception error and
reasoning error come apart:

| arm | what it isolates |
|---|---|
| end-to-end VLM on image tokens | the current design |
| symbolic program over **ground-truth** detections | pure reasoning ceiling, ~100% by construction |
| symbolic program over **predicted** detections | how much perception error costs |
| decoder over predicted detections | how much the decoder costs on top of that |

The gap between rows 2 and 3 is perception. Between 3 and 4 is the decoder. Most
work of this kind reports only row 4 and cannot say which half is failing.

### The caveat that decides whether this is honest

**The query set is schema-computable by construction.** Counting, comparison,
superlatives and spatial relations were generated from boxes and masks precisely
so ground truth could be derived automatically -- which means a symbolic program
should win them, and a decoder that merely matches it has demonstrated nothing.
Reporting "the agentic system beats the end-to-end VLM" on this set is a fair
claim about the *pipeline*; reporting it as evidence that the *decoder* earns its
place is not.

There is also a job in this design that only the decoder can do, and it is worth
separating from composition. **The class list need not be fixed.** A question like
"is there anything here that could leak into the soil" corresponds to no category
in the taxonomy, but it does correspond to a set of text queries -- drums, oil
barrels, asbestos sheeting, chemical containers -- and writing that set is a
language task. The decoder generates the SigLIP2 queries, the encoder scores them,
and the vocabulary stops being the twenty names anyone wrote down. Neither a
linear head nor a fixed prompt bank can do this at all, so unlike composition it
has no non-LLM baseline to tie against; the comparison is instead against a fixed
prompt bank, and the measure is recall of objects whose class was never named.

The decoder's advantage, if it has one, lives where automatic ground truth is
hard: free-form description, unanticipated attributes, ambiguous or
underspecified reference, and questions the schema never anticipated. Two honest
ways to reach it, in order of cost:

1. **Held-out query families.** Write questions whose answers need a *combination*
   the schema exposes but no template covers, and check whether the decoder
   composes tool calls for them zero-shot. Ground truth stays automatic.
2. **A small human-scored free-form set.** A few hundred descriptions, scored for
   factual grounding against the scene graph. Expensive, but it is the only way
   to measure the thing the decoder is actually for.

Without at least the first, the honest conclusion available from this project is
"a frozen encoder with tools beats a 7B VLM at aerial waste perception, and the
decoder's remaining value is untested" -- which is a real result, and better than
an unsupported claim in either direction.

## The constraint: no fixed label vector (2026-08-19)

Every component must take its vocabulary at inference time, as text or as
exemplars. A class nobody enumerated when the system was built must still be
askable. This is an architectural constraint, not a claim about training data, and
it is the one that actually matters -- two earlier drafts of this section framed it
as "no supervision" and then as "nothing fitted on the evaluation datasets", and
both drew the line in the wrong place.

Training is allowed, including on external waste data, and the standing rule that
AerialWaste and DroneWaste stay out of every training mix still holds. What is not
allowed is a head that emits a fixed vector of class logits, however it was
trained. A model that must be rebuilt to answer a new class is not open-world, and
rebuilding it is exactly what a waste taxonomy will demand -- anything can be
dumped.

### What this admits and excludes

| component | verdict | why |
|---|---|---|
| C-RADIOv4 -> SigLIP2 head, scored against text | **in** | vocabulary is the prompt |
| Grounding DINO | **in** | text-conditioned proposals |
| SAM3 (native or C-RADIO-bridged) | **in** | text-promptable detection and masks |
| decoder writing the text queries | **in** | this is what makes the vocabulary genuinely open |
| nearest-prototype from exemplars | **in** | a new class is a new exemplar, not a new architecture |
| ROI tokens + 20-way linear head | **out** | fixed label vector, whatever it was fitted on |
| trained binary presence gate | **out** | same, for a vocabulary of one |

Gating becomes a text comparison in the SigLIP2 space, or SAM3's own detection
score. Fine-tuning a text-image model on external waste data stays admissible,
because the result is still queried by text; fine-tuning it into a 20-logit
classifier does not.

```
C-RADIOv4 (frozen, one pass) ─┬─ →SAM3 → FPN → DETR → boxes + masks
                              └─ →SigLIP2 summary → naming from text
Grounding DINO (frozen) ────────► boxes, where its recall still leads
decoder ────────────────────────► writes the queries, composes the answer
```

### What it costs, and what it buys

Naming falls from the fitted head's 0.733 to 0.442 on DroneWaste -- still 6.66x
chance against 20 classes on a 0.198 majority. On AerialWaste zero-shot naming
sits at or below the majority baseline, so AerialWaste carries detection only.

What it buys is the part that cannot be quoted as a single number: the system is
not limited to 20 classes, or to 15, or to whatever a published Faster R-CNN was
trained on. **The comparison stops being "our accuracy against theirs on their
label set" and becomes "their label set against no label set at all."** That is
also why per-class reporting matters more here than a macro average -- the
interesting rows are the ones a closed-set model could not have had.

The probes keep a role, as diagnostics rather than components: they bound what any
text-driven readout could extract from the same features, and they stand in for the
trained detector in the Faster R-CNN / YOLO comparison. Measuring the gap is the
result; shipping the probe is what stops.

### The leak to watch

An open-vocabulary pipeline still has thresholds -- SAM3's detection score, the
gating margin, the matching IoU -- and choosing them on AerialWaste or DroneWaste
labels is closed-set fitting by another name. Not hypothetical: the SAM3 sweep
already run moved recall from 0.277 to 0.446 across thresholds 0.30 to 0.15, wide
enough that picking the best on the test set would be a real distortion. With 17
sites, spending one on threshold selection is cheap and makes the cost explicit;
that is the recommendation over fixing values a priori.
