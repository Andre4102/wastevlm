# Track: DINO/RADIO

You are the DINO/RADIO track of the Waste-VLM project. Read
`/home/ids/diecidue/scripts/waste_vlm/project.md` and
`/home/ids/diecidue/scripts/waste_vlm/tracks/README.md` first — they have the
full design and the shared rules.

## Your scope

You own the vision encoder. Pick between DINOv2/DINOv3 and NVLabs RADIO, set up
clean feature extraction over AerialWaste m2 + DroneWaste, and produce a
linear-probe baseline number so the VLM track knows what the encoder is worth
before training the projector.

You do NOT touch: the VQA dataset, the projector wiring, the LLM, or the
training loop. Those are owned by ANNOTATION / VLM / LLM tracks.

Files you own (mostly new):
- `src/vision_encoder.py` — encoder loader (DINOv2/v3, RADIO) with a uniform
  `encode(image) -> features` interface.
- `src/encoder_probe.py` — linear-probe trainer + eval.
- `results/waste_vlm/encoder_probe/` — probe metrics per encoder.

Weights live under `results/waste_vlm/weights/{dinov3-*, RADIO-*}`.

## Decisions to make

1. **Encoder family** — DINOv3 (Meta) vs RADIO v2.5 (NVLabs). Both are
   foundation-grade SSL encoders; RADIO unifies CLIP+DINO+SAM signals via
   distillation, DINOv3 is purer SSL on the LVD-1689M curated set. For aerial
   imagery, DINOv3-Sat (the satellite-trained variant) is the natural fit if
   weights are available locally — check `ls results/waste_vlm/weights/` first.
2. **Patch resolution** — aerial-waste signal can be small (a single dump pile is
   tens of pixels). Prefer ViT-L/14 or smaller patches over ViT-G/14 if
   compute permits.
3. **Adapter strategy** — none for the probe (frozen encoder + linear head). The
   VLM track decides projector design.

## Next steps (in order)

1. **Audit local weights.**
   ```bash
   ls -lh /home/ids/diecidue/results/waste_vlm/weights/
   ```
   Note which DINOv3 and RADIO variants are already downloaded; do not re-download
   if a usable variant is on disk.
2. **Write `src/vision_encoder.py`** — uniform interface that loads either family
   and returns CLS + patch features. Smoke-test on one image from
   `data/dronewaste/images/` and one from `data/aerialwaste/images/`.
3. **Linear probe on classification.** Multi-label classifier head over the 15
   waste categories (10 DW paper-10 + 5 AW m2). Train on the VQA train split's
   `gt_categories` (sourced from `data/vqa/split_index.jsonl` — owned by
   ANNOTATION; read-only here), eval on val. Report per-class F1 + macro-F1 per
   encoder variant.
4. **Optional second probe: segmentation.** Linear DPT-style head on patch
   features, trained on DW masks. Confirm the encoder retains spatial detail at
   the candidate patch size.
5. **Decision memo.** Write a 1-paragraph recommendation to `project.md` under a
   new "Encoder choice" section, with the linear-probe numbers.

## Dependencies on other tracks

- **Needs from ANNOTATION:** `data/vqa/split_index.jsonl` (already exists).
  Read-only. Do not regenerate.
- **Produces for VLM:** the chosen encoder identifier + a known-good loader in
  `src/vision_encoder.py`. The VLM track imports and freezes it.

## Hard constraints

- Use the existing `waste_vlm` conda env. Do not create a sibling env.
- Probe must use the site-stratified seed=0 split (already encoded in
  `split_index.jsonl`) to keep numbers comparable with the seg/detection
  baselines in `RESULTS_SNAPSHOT.md`.
- One H100 on SLURM partition `mm`, node `nodemm07`. Do not consume both GPUs
  on this node — the VLM track will need the other.
