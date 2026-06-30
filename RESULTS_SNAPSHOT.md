# DroneWaste + AerialWaste — current results snapshot

*Snapshot date: 2026-05-28 (post Phase-2 multi-block, Qwen + InternVL3 full-split VLM runs).*

DroneWaste numbers on the **DroneWaste test split** (site-stratified
70/15/15, seed=0). AerialWaste numbers on the **published AW MCML test
split** (filtered to images on disk, matching dinotxt's regime).
All ViT backbones are **frozen**; only the head is trained.

---

## 1. DroneWaste — backbone × head × resolution ablation (DinoSeg probes)

| # | Backbone        | Res.  | Head   | Train. params | val mIoU | test mIoU_fg | paper-10 mAP | AP50  |
|---|-----------------|------:|--------|--------------:|---------:|-------------:|-------------:|------:|
| 1 | DINOv2-B (baseline) | 518²  | linear |          16 k | 0.409    | 0.372        | 13.3 %       | 16.2 %|
| 2 | DINOv2-L        | 518²  | linear |          22 k | —        | 0.384        | 13.3 %       | 17.7 %|
| 3 | DINOv2-B HR     | 1036² | linear |          16 k | —        | 0.390        | 13.6 %       | 16.8 %|
| 4 | DINOv3-B        | 512²  | linear |          16 k | 0.457    | 0.419        | 14.7 %       | 17.1 %|
| 5 | DINOv3-B HR     | 1024² | linear |          16 k | 0.449    | 0.410        | 15.4 %       | 16.9 %|
| 6 | DINOv2-B        | 518²  | conv   |         1.8 M | 0.466    | 0.416        | 15.5 %       | 18.6 %|
| 7 | DINOv3-B        | 512²  | conv   |         1.8 M | 0.486    | 0.408        | 14.1 %       | 17.0 %|
| 8 | **RADIOv2.5-L** | **512²** | **linear** | **22 k** | **0.457** | 0.416   | **16.1 %**   | **19.5 %** |
| 9 | RADIOv2.5-L     | 512²  | conv   |         1.8 M | 0.484    | **0.427**    | 15.4 %       | 19.1 %|
|10 | RADIOv2.5-L HR  | 1024² | linear |          22 k | 0.438    | 0.417        | 15.8 %       | 18.7 %|
|11 | RADIOv2.5-B     | 512²  | linear |          16 k | —        | 0.397        | 14.9 %       | 17.0 %|
|12 | DINOv3-B mb{3,7,11} | 512² | linear |        ~50 k | —        | **0.429**    | 15.3 %       | 18.6 %|

**Headline winners**
- Best test **mIoU**: RADIOv2.5-L @ 512² + conv head (0.427)
- Best test **paper-10 mAP**: RADIOv2.5-L @ 512² + linear head (16.1 %)
- Best test **AP50**: same — RADIOv2.5-L @ 512² + linear (19.5 %)

**Takeaways**
1. Resolution (512 → 1024) **does not** move the headline on this dataset.
2. **DINOv3 > DINOv2** by +4.7 pp test mIoU on linear head, same params (16 k → 16 k).
3. **AM-RADIO objective ≈ DINOv3 Gram anchor at the same scale.** At B-scale (16 k head), DINOv3-B 14.7 % paper-10 mAP vs RADIO-B 14.85 % — within 0.1 pp. RADIO-L's earlier 16.1 % lead (row 8) was the **B → L scale step**, not the AM-RADIO objective. Scale only buys ~1.4 pp here, comparable to the DINOv2 B → L step (rows 1 → 2, 0 pp). The encoder family is essentially saturated at this head budget.
4. **Conv head helps DINOv2 only**: +2.2 pp mAP on DINOv2; *overfits* DINOv3 (val 0.486 → test 0.408); mixed on RADIO (best mIoU, slightly worse mAP).
5. The conv head's smoothing inflates mIoU but merges connected components, hurting the CC→bbox detection step.
6. **Multi-block FPN-lite (row 12)** — concatenating DINOv3-B patch tokens from ViT blocks {3,7,11} (2304-d head input, ~50 k params) gives a small *consistent* lift over the last-block DINOv3-B baseline (row 4): mIoU 0.419 → 0.429, paper-10 14.7 → 15.3 %, AP50 17.1 → 18.6 %. Unlike the conv head it does **not** overfit (linear head on richer features). Still below RADIO-L's 16.1 % — earlier ViT blocks help but don't close the gap.

---

## 2. DroneWaste — where the probes sit relative to fully-trained detectors

Same DroneWaste paper-10 metric, same site-stratified test split.
Detector numbers reproduced from the DroneWaste paper (Mora et al.).

| Method                                          | Trained on DW?  | Trainable params | paper-10 mAP |
|-------------------------------------------------|-----------------|-----------------:|-------------:|
| YOLOv8                                          | full e2e        | ≥ 25 M           | 38.2 %       |
| YOLOv12                                         | full e2e        | ≥ 25 M           | 38.5 %       |
| Faster R-CNN                                    | full e2e        | ≥ 25 M           | 36.6 %       |
| DinoDETR (frozen DINOv2-B + 6-layer DETR)       | yes             | 6.7 M            | 2.1 %        |
| DinoSeg DINOv2-B + linear (§ 1 row 1, baseline) | yes             | **16 k**         | 13.3 %       |
| **DinoSeg RADIO-L + linear (§ 1 row 8)**        | **yes**         | **22 k**         | **16.1 %**   |
| dino.txt zero-shot seg                          | **no**          | 0                | 1.4 %        |

**Read.** Best frozen-backbone probe (22 k trainable params) closes roughly
**40 %** of the gap between the DinoDETR floor (2.1 %) and YOLOv8 (38.2 %).
Per-class mAP for the YOLO winners is much higher on tightly-bounded
instance classes (Pallets, Asbestos, Tyres) and lower on visually
heterogeneous ones (Vehicles, Mixed items); DinoSeg sometimes *beats* YOLO
on those, e.g. Vehicles (DinoSeg 23.8 % vs YOLOv8 5.7 %, § 3.5.6).

---

## 3. DroneWaste — Phase 1 (tile-then-aggregate) verdict

Tests the "is patch size / effective ground resolution the bottleneck?"
hypothesis without retraining. For each 640² source tile, the image is
split into 2×2 non-overlapping 320² crops, each crop is resized to the
model's native input (512²) and run independently, then the per-class
softmax maps are stitched back to a 640² mask before the CC→bbox step.

| Config                       | mIoU baseline | mIoU TTA 2×2 | paper-10 baseline | paper-10 TTA 2×2 | AP50 baseline | AP50 TTA 2×2 |
|------------------------------|--------------:|-------------:|------------------:|-----------------:|--------------:|-------------:|
| DINOv3-B @512 linear         | 0.4191        | 0.3986 (−2.0)| 14.65 %           | 15.48 % (+0.8)   | 0.171         | 0.165 (−0.6) |
| RADIO-L @512 linear          | 0.4160        | 0.3767 (−3.9)| 16.09 %           | 13.14 % (−3.0)   | 0.195         | 0.154 (−4.1) |

**Verdict: patch size is *not* the bottleneck.** TTA hurts on both backbones —
each 256² crop loses 75 % of the surrounding context, which the encoder
was trained to use, and the marginal +0.8 pp mAP gain on DINOv3 is wiped
out by every other metric. The 22-pp gap to YOLO is therefore about
**FPN-equivalent + instance grouping** (Phase 2 / Phase 3 ablations), not
effective resolution.

---

## 4. AerialWaste — supervised classification probes

Same protocol as the existing DINOv2-B probe in § 3.5.1 (L2-normalised
embeddings → per-class LogReg with class_weight=balanced → F1-tuned
per-class threshold on train), now run for DINOv3-B and RADIO-L. Same AW
MCML train/test splits as `dinotxt_zeroshot.run_aw_mcml`.

| Method                                         | AW m2 micro F1 | AW m4 micro F1 |
|------------------------------------------------|---------------:|---------------:|
| dino.txt zero-shot (DINOv2-L/14 + CLIP head)   | 0.37           | 0.38           |
| DINOv2-B supervised (§ 3.5.1)                  | 0.60           | —              |
| **DINOv3-B supervised**                        | **0.640**      | **0.601**      |
| **RADIOv2.5-L supervised**                     | **0.635**      | **0.610**      |

**Read.** DINOv3-B and RADIO-L are tied (within 0.5 pp) on both AW
splits, and both clear DINOv2-B by ~ +4 pp. The §3.5.10 picture where
RADIO-L beat DINOv3-B on DroneWaste paper-10 mAP **does not generalise**
to AW classification — RADIO's lead on DW was specific to the
seg→CC→bbox pipeline (likely SAM-distilled boundary cues helping the CC
step). For pure multi-label recognition the choice of frozen encoder
above DINOv2-B barely matters.

---

## 5. Zero-shot multi-label classification — DW and AW (VLM full splits)

`src.vlm_eval --task classify`, **closed_vocab** prompt, full test
splits (DW paper-10 = 1504 imgs, AW m2/m4 = 664 imgs each). Prompts
include per-class visual cues from Wikipedia + EU LoW (Decision
2014/955/EU). dino.txt rows are from § 3.5.1. Micro F1; empty-parse =
fraction where the model returned no usable label.

|                                              | DW paper-10 | AW m2 | AW m4 |
|----------------------------------------------|------------:|------:|------:|
| dino.txt zero-shot (DINOv2-L/14)             | 0.40        | 0.37  | 0.38  |
| Qwen2.5-VL-7B closed_vocab (full)            | 0.239       | 0.375 | 0.266 |
| **InternVL3-8B closed_vocab (full)**         | **0.368**   | **0.452** | **0.304** |
| Qwen2.5-VL-7B empty-parse                    | 54 %        | 62 %  | 53 %  |
| InternVL3-8B empty-parse                     | 59 %        | 74 %  | 68 %  |

**Headline.** **InternVL3-8B on AW m2 = 0.452, beating dino.txt by +8 pp
on the full 664-image split** — the first open VLM in this project to
clear that bar on a full split (the earlier Qwen "0.500 +13 pp" was
20-image smoke noise; its full-split AW m2 is 0.375, a tie). InternVL3
beats Qwen on all three splits. But DW paper-10 (0.368) and AW m4 (0.304)
stay below dino.txt, and **empty-parse rates are 53–74 %** — both models
default to "none" on the majority of images. That abstention is the
dominant failure mode, and the exact thing the caption-distillation LoRA
(`src/caption_pilot.py` → Qwen2.5-VL fine-tune) is meant to fix.

**Correction note.** The prior snapshot's §5 headline ("Qwen2.5-VL beats
dino.txt by +13 pp on AW m2") was based on a 20-image smoke (F1 0.500).
The full 664-image split collapses that to 0.375 — a tie, not a win. The
real full-split win belongs to InternVL3 (0.452). The InternVL3
shape-mismatch bug (`[1024,3584] vs [256,3584]`) was fixed via the
official `dynamic_preprocess` (448² tiles + thumbnail, `num_patches_list`
to `chat()`); these full-split numbers confirm the fix works.

---

## 6. In-flight / planned ablations

| Phase | Status | What it tests |
|------:|--------|---------------|
| 1     | **done** — patch size NOT the bottleneck (§ 3) | Tile-then-aggregate at native 640² |
| 2     | **done** — small consistent +0.6 pp mAP, no overfit (§ 1 row 12) | Multi-block FPN-lite (concat ViT blocks {3, 7, 11}) |
| 3     | pending — *highest-leverage next test*    | Mask R-CNN-style RoI head on frozen DINO/RADIO (instance grouping) |
| —     | **done** — RADIO-B = 14.9 % mAP, ≈ DINOv3-B (§ 1 row 11) | Apples-to-apples DINOv3-B vs RADIO-B comparison |

---

## Reproducibility notes

- DinoSeg probes: 80 epochs (linear head) or 30 epochs (1024² HR),
  AdamW `lr=1e-3`, `wd=1e-4`, cosine schedule, batch=32 (512²) or 16 (1024²).
- Loss: weighted cross-entropy (bg weight = 0.1) + foreground Dice.
- `best.pt` selected by best val `mIoU_fg`; test numbers reported on
  that checkpoint.
- AW supervised probe: per-class LogReg, C=1.0, class_weight="balanced",
  F1-tuned per-class threshold on train scores.
- VLM eval: zero-shot, no fine-tuning, `do_sample=False`,
  max_new_tokens=512. Per-class cue tables:
  `src/paper10_descriptions.json` (DW),
  `src/aw_m2_descriptions.json`, `src/aw_m4_descriptions.json` (AW).
- Code: `src/seg_train.py`, `src/seg_model.py`, `src/seg_eval.py`,
  `src/seg_eval_tta.py`, `src/aw_classify_probe.py`, `src/vlm_eval.py`.
