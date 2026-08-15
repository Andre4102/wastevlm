# Instruction-tuning design — non-waste main arm, waste ablation

_Drafted 2026-08-14, after the AerialWaste diagnosis in EXPERIMENTS.md
("Why AerialWaste is bad" → "The cause" → "Where the signal is lost")._

**Held out, always: AerialWaste and DroneWaste.** Both splits of both datasets stay
out of every training mix — not just their test splits. Every number is zero-shot.

Three capabilities to train:

1. **Describe the image broadly** — a grounded, scene-level read.
2. **Answer questions about specific details** — which object, where, how many.
3. **Answer negatively when the question is wrong** — absence, false premise, and
   "not determinable at this resolution".

Two arms:

| arm | mix | question it answers |
|---|---|---|
| **A — non-waste** | general SFT + aerial-viewpoint non-waste data teaching (1)(2)(3) | Is the missing thing **behaviour and viewpoint**? |
| **B — + waste ablation** | A + the on-disk non-AW/DW waste tiers | Does **waste domain knowledge** add anything on top? |

Arm A is the main arm. Arm B is the ablation, and it uses only *other* waste
datasets (TrashBox, TACO, UAVVaste, SWAD, ZeroWaste-f/WasteBench) — never
AerialWaste or DroneWaste, so the benchmarks stay clean in both arms.

---

## 1. What the evidence constrains before we start

| finding | constraint it imposes |
|---|---|
| frozen-encoder probe AUC 0.966–0.970, J 0.837–0.853 vs full-VLM J 0.139–0.234 | Perception is already there. This dataset's job is the **decision layer**, not new visual capability. |
| 1.56% / 0.18% of current SFT turns assert absence | Abstention must be a **first-class answer type**, not a rare tail. |
| 150K arm: `pile` on 100% of positives and 99.8% of negatives | The failure mode is a **caption template**. Templated targets reproduce it exactly; style variety is a correctness requirement. |
| 819K arm on AW: `pile` 3.5%/0.5%, mute, micro 0.025 | The **opposite** attractor is equally reachable. Negative-heavy data teaches unconditional "none". |
| AW median annotated object = 0.92 tokens, 53% sub-token | At nadir GSD many detail questions are genuinely unanswerable. "Cannot be determined" is often the *correct* target. |
| stage-2 mix is 100% ground-level natural images | **Viewpoint is a separate axis from behaviour.** Both need fixing, and they can be fixed by the same data. |

**The single most important design rule:** the model must see *both* answer
polarities on *visually similar* images, ideally the same image. Balance at the
level of **answers**, never images.

---

## 2. Why non-waste data can still fix this

The diagnosis says the broken thing is the decision to answer, not the ability to
see. Abstention — "say no when the thing is not there" — is a domain-general
behaviour. If we teach it on nadir imagery using non-waste categories (vehicles,
ships, storage tanks, harbours), and it transfers to "no waste here" on
AerialWaste, that **is** the result: the deficit was behavioural.

This makes Arm A a real experiment rather than a fallback, with three outcomes and
a defined next step for each:

| outcome | reading | next |
|---|---|---|
| Youden J rises, naming F1 flat | Deficit was behaviour + viewpoint | Arm B to see if domain knowledge adds naming |
| Only Arm B moves J | Waste knowledge was needed after all | in-domain data becomes unavoidable |
| Neither moves J | The generative path cannot express what the probe reads | architectural arm: feed the projected summary token (`src/vlm_model.py:326`) |

**Prediction to record before running:** Arm A should move **detection** (Youden J)
and leave **naming** (micro-F1 on answered positives, currently 0.56–0.60) roughly
unchanged, because it never sees a waste pile. Writing this down now makes the
result interpretable either way.

---

## 3. Sources

### On disk already

| source | records | viewpoint | role |
|---|---:|---|---|
| `alignment/normalized/sft_mix.jsonl` | 738,601 llavanext + 39,754 visionflan | ground-level | general replay, both arms |
| `llava_instruct_150k` | 150k | ground-level | small-mix option |
| `waste_sft/train.json` | 46,713 | ground-level + conveyor | **Arm B only** |
| UAVVaste / SWAD | 772 / 1,996 | drone / satellite | **Arm B only** |

### To download (login node — compute nodes have no internet)

| source | size | licence | gives us |
|---|---:|---|---|
| **`xiang709/VRSBench`** ✅ *downloaded* | 12.5 GB | CC-BY-4.0 | 20,264 train images, 142,390 records: 85,813 VQA / 36,313 referring / 20,264 captions. Covers capabilities 1 and 2. **Not a safe source for derived negatives — see §3.1.** |
| **`torchgeo/dior`** | **7.4 GB** | other | DIOR: 23,463 nadir images, 192,472 boxes, 20 classes, **exhaustively annotated** (`Annotations_trainval.zip`, PASCAL-VOC XML). **Required** for capability 3 — the only downloaded-and-verified source where absence in the annotation means absence in the image. |
| `arampacha/rsicd` | 0.5 GB | — | 10,921 images × 5 captions. Cheap capability-1 diversity. |
| RSVQA-LR / HR (Zenodo) | ~1 / ~15 GB | CC-BY | Presence yes/no questions at nadir with **genuine "no" answers** — capability 3 without derivation. |
| `MBZUAI/GeoChat_Instruct` | 102 GB | Apache-2.0 | 318k RS instructions. Only if VRSBench proves insufficient — the size is not obviously worth it. |
| VizWiz-VQA (optional) | ~10 GB | CC BY 4.0 | ~28% explicitly *unanswerable* questions, ground-level. Tests whether abstention transfers **across** viewpoint — a clean sub-ablation. |

### 3.1 Measured after download — VRSBench cannot supply the negatives

Two properties of the shipped data, both measured, both of which change the plan:

**(a) The object annotations are not exhaustive.** `objects[]` holds
referring-expression *targets*, not every instance: **mean 1.8 objects per image**
(median 2, max 8) in aerial scenes that plainly contain more. Cross-checking
against the captions, a category is named in the caption but **absent from
`objects[]` 9.1% of the time** (238/2611 over a 3000-image sample) — and captions
are not exhaustive either, so that is a floor, not an estimate.

⇒ Generating "Is there a vehicle?" → "No" from absence in `objects[]` would
**teach the model to deny things that are present**. That is the mute collapse with
actively wrong labels — strictly worse than the disease. **Do not derive type-(a)
or type-(c) negatives from VRSBench annotations.**

**(b) The shipped answer distribution is skewed the wrong way.**

| answer | count | share of VQA |
|---|---:|---:|
| yes | 21,112 | 24.6% |
| **no** | **3,720** | **4.3%** |

A 5.7:1 yes:no skew. Training on VRSBench VQA as shipped *reinforces* the
affirmative bias we are trying to remove. Its ~3.8k genuine "no" answers (from the
`object existence` QA type, verified against the image by the VRSBench pipeline)
are safe to use but nowhere near a 40k negative block.

**(c) 73% of captions name the image source** — "The high-resolution image sourced
from GoogleEarth shows…". Content-free boilerplate that would teach the model to
say "sourced from GoogleEarth" when describing AerialWaste tiles. Strip it from
caption targets. The captions are otherwise excellent: 20,259 of 20,264 distinct
and genuinely image-conditioned.

**Consequence:** VRSBench supplies capabilities 1 and 2 plus ~3.8k safe negatives.
Capability 3 at volume needs an **exhaustively annotated** detection corpus, where
absence in the annotation really is absence — **DIOR** (`torchgeo/dior`, 20 classes,
192k boxes, ~8 objects/image) is the pick. Start with **VRSBench + DIOR** (~20 GB).

### Hard gate: leakage check

VRSBench, GeoChat and RSICD are largely **Google Earth** imagery. AerialWaste's GE
tier is **391 of 581** test images, also Google Earth. Different regions (AW is
Lombardy) so overlap is unlikely, but it is not zero and it would silently
invalidate everything.

**Before any training:** perceptual-hash dedup of every downloaded image against
AW test (581) + AW train (3689) + DW (4993). The repo already ran this pattern in
Phase 0 (`waste_sft/leakage_report.json`, job `lrd_all_serial`) — reuse it. Any
collision is dropped, and the report is committed alongside the mix.

---

## 4. Capability 1 — broad description

**Source.** VRSBench detailed captions (29,614) + RSICD (10,921 × 5). Both are
human/verified RS captions, so no captioner-hallucination risk and no template.

**What to check, not assume.** Run the caption-conditioning probe on the *training
targets* before training:

```
python scripts/aw_diagnose.py --captions   # pointed at the SFT targets
```

Here the probe terms are the RS categories rather than waste terms, and the test is
whether a category term appears preferentially on images that contain it. If
targets mention categories at similar rates on images with and without them, the
data teaches unconditional narration. Require **≥40 points separation**; the DW arm
that actually worked shows +54.

RSICD is known to be repetitive (several near-identical captions per image). Cap
n-gram overlap across accepted captions, or take one caption per image.

## 5. Capability 2 — specific details

VRSBench ships ~123k VQA pairs and referring expressions over annotated objects,
which covers presence, counting, location, size and relation at nadir viewpoint
directly. Where we want tighter control, derive from the object annotations:

| family | example | derived from |
|---|---|---|
| presence | "Is there a storage tank in this image?" | annotation categories |
| location | "In which part of the image is the harbour?" | box centre → quadrant |
| count | "How many ships are visible?" | box count per category |
| extent | "Which is the largest structure?" | box areas |
| co-occurrence | "Besides vehicles, what else is present?" | category set minus one |

Keep granularity honest: do not generate questions about texture or fine material
identity that the GSD cannot support. Those belong in §6 type (d).

## 6. Capability 3 — answering negatively

Four types, not interchangeable; each blocks a different shortcut. Derived from
**DIOR's exhaustive** annotations, at nadir, with zero waste content — *not* from
VRSBench, for the reason in §3.1. VRSBench's ~3.8k verified `object existence`
negatives can be folded in as-is.

| type | prompt → target | why it matters |
|---|---|---|
| **(a) category-absent on a populated image** | "Is there a roundabout here?" → "No. The visible structures are storage tanks and a harbour." | **Highest value.** The image *does* contain objects, so the model cannot answer from "is this a busy scene?". The only type forcing per-category discrimination. |
| **(b) whole-image absence** | "What man-made structures are visible?" → "None — this is undeveloped terrain." | Teaches the empty answer. Needs images with no annotated objects. |
| **(c) false premise** | "Describe the aircraft in the lower right." → "There is no aircraft in this image; the objects visible are …" | Rejects a presupposition instead of complying with the question's frame. |
| **(d) not determinable** | "What is the small object near the centre?" → "At this resolution it cannot be identified." | Calibration. Sample from the smallest boxes — the sub-token regime that dominates AerialWaste. |

Suggested emphasis: **(a) ≈ 45%, (b) ≈ 25%, (c) ≈ 15%, (d) ≈ 15%.**

**Built and measured (2026-08-14, `scripts/build_rs_sft.py`).** Two things the
design got wrong before the data was on disk:

- **The type mix only binds at one negative per image.** Each DIOR image offers at
  most one candidate of each type, so emitting 3-4 negatives per image takes nearly
  all of them and the weights merely decide which is dropped — the realised mix
  tracks *availability* (36/28/24/3) rather than intent. At `--per-image 2` the
  weighted draw fully determines the type and the mix lands exactly on target
  (45/25/15/…). The cost is ~47k DIOR records instead of ~72k, which is cheap:
  the extras are more questions about the *same* 23,463 images.
- **Type (d)'s 15% target was wrong and should stay ~2-3%.** It was set before
  measuring DIOR's boxes. The median annotated object is **952 px², below one
  visual token at 768px/ps2**, so a generous threshold would call half the dataset
  unidentifiable and teach refusal on objects the model can resolve — building the
  mute collapse by hand. The generator uses a 0.2-token cutoff and the resulting
  low share is the correct outcome, not a shortfall.

Realised builds, both passing every hard gate with **100% of DIOR images carrying
both answer polarities**:

| build | records | images | negative-type mix (a/b/c/d) |
|---|---:|---:|---|
| `rs_sft/` (`--per-image 4`) | 90,542 | 35,918 | 36 / 28 / 24 / 3 |
| **`rs_sft_p2/` (`--per-image 2`)** | **54,372** | **29,698** | **45 / 25 / 15 / 2** |

Types (b) and (c) are cheap to generate and dangerous to over-weight — they are
exactly what produces the mute 819K-style collapse.

Type (d) deserves care: it is the only type whose target is *uncertainty* rather
than *absence*, and it is the one that most directly addresses AerialWaste's 0.92-
token median object. Derive it from the bottom decile of box areas, converted to
visual-token units at the training resolution so the threshold means the same thing
it means at eval.

## 7. Mix, balance and volume

| block | records | source |
|---|---:|---|
| RS captions (cap. 1) | ~35k | VRSBench + RSICD, 1 per image |
| RS detail QA, affirmative (cap. 2) | ~40k | VRSBench VQA (subsampled) |
| RS negative answers (cap. 3) | ~40k | derived from VRSBench annotations, §6 mix |
| eval-format anchors **including the none case** | ~2k | two-turn CoT, both outcomes |
| **RS total (Arm A injection)** | **~117k** | over ~30k images |
| general replay | majority of mix | `sft_mix.jsonl` |

**Balance rules:**

- **~50/50 affirmative vs negative over answers**, not over images.
- **Every populated image contributes at least one "no"** (type a); every empty
  image at least one substantive description. Same image, both polarities.
- **Keep general replay as the majority.** The 819K arm's DroneWaste gain
  (0.060 → 0.310) came from general SFT scale; ~117k in-domain records will not
  survive contact with a 7B decoder alone.
- **Arm B adds the waste tiers on top of A, changing nothing else** — otherwise the
  ablation is confounded and tells us nothing.

**Augmentation.** Nadir imagery is genuinely rotation-invariant, unlike natural
images: 4 rotations × 2 flips is an honest 8× with no label surgery. Applies to the
RS blocks only, never to ground-level tiers. Note it interacts with type (b)/(d)
location questions — rotate the derived answers too, or exclude location families
from augmentation.

## 8. Build pipeline

1. Download VRSBench + RSICD on a **login node**; verify checksums.
2. **Leakage gate** (§3): perceptual-hash dedup against AW + DW; commit the report.
3. `scripts/build_rs_sft.py` — emit the four blocks from VRSBench annotations at the
   §6 type mix. Deterministic.
4. **Dataset validation gates — all must pass before a GPU is booked:**
   - caption-conditioning separation ≥40 points
   - answer-polarity balance within 45–55%
   - every populated image has ≥1 negative-answer record
   - zero AW/DW image hashes present
   - answer-length distribution not collapsed (`waste_sft/stats.md` records 55%
     single-word answers in the WasteBench tier — do not repeat that)
5. Merge with general replay; pre-tokenize; train stage 2. Arm B = same, plus the
   waste tiers.

## 9. How to evaluate

Micro-F1 alone hid this failure for the entire resolution branch. Dev metrics:

- **Youden J on the binary decision** — primary. Baseline 0.234 (aw_m2) / 0.139
  (aw_m4) / 0.673 (dw). Encoder ceiling 0.85 / 0.84.
- **naming micro-F1 on answered positives** — currently 0.56–0.60; must beat the
  oracle-gated constant (0.697 / 0.732) to mean anything.
- **`constant_baseline` and `prediction_diversity`** — already emitted by
  `src/vlm_eval.py`; a run at or below the floor is not a result.
- **Both error directions explicitly** — false-alarm rate on negatives *and*
  refusal rate on positives. Both collapses are live; one number cannot separate them.
- **The `only_pos` split** (217/204 images) alongside the full split.
- **Caption-conditioning probe on the model's outputs** (`aw_diagnose.py --captions`)
  — the direct read on whether the template collapse is gone.

## 10. Known risks

- **Nothing in Arm A teaches waste naming.** Expected, and the prediction in §2
  accounts for it — but it means Arm A alone will not produce a competitive AW
  micro-F1, and that must not be read as failure. J is the metric that matters.
- **VRSBench category granularity is coarse** relative to waste taxonomies, so
  type-(a) discrimination may be easier than the AW task requires. Watch for a
  large J gain that does not survive to fine-grained naming.
- **Google Earth overlap** with AerialWaste — the §3 gate is not optional.
- **Type (b)/(c) over-weighting** produces the mute collapse. The §8 polarity gate
  is the guard.
- **The probe's 0.85 J may not be reachable through a generative path at all.** If a
  well-built Arm A still lands far below it, the next arm is architectural, not more
  data (`src/vlm_model.py:326` feeds `.patches` only; `cls` was the best pooling).
