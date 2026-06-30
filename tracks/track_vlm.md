# Track: VLM

You are the VLM track of the Waste-VLM project. Read
`/home/ids/diecidue/scripts/waste_vlm/project.md` and
`/home/ids/diecidue/scripts/waste_vlm/tracks/README.md` first — they have the
full design and the shared rules.

## Your scope

You own the full multimodal model: vision encoder (loaded from the DINO/RADIO
track) → projector → Qwen2.5-7B-Instruct, plus the training loop on the VQA
train split and the eval loop on the VQA val/test benchmark.

You do NOT generate the dataset, build the encoder, or own the LLM weights' text
fine-tuning. Those are owned by ANNOTATION / DINO/RADIO / LLM tracks.

Files you own (mostly new):
- `src/vlm_model.py` — LLaVA-style assembly (encoder + projector + Qwen).
- `src/vlm_train.py` — LoRA fine-tune (projector full-train, LM LoRA, encoder
  frozen).
- `src/vlm_eval.py` — per-q_type metrics (presence acc, type_open F1,
  type_choice exact-match, grounding IoU). Also hosts the zero-shot baselines
  (CLIP, Qwen2.5-VL, InternVL3) run via `--task classify`.
- `slurm_vlm_train.sh`, `slurm_vlm_eval.sh`.
- `results/waste_vlm/vlm/{run_*}` — checkpoints and metrics.

## Zero-shot baselines in `src/vlm_eval.py` (current state)

Three models; three prompt styles; three datasets (dw_paper10, aw_m2, aw_m4).

| Model | `--model` key | Inference path |
|---|---|---|
| CLIP ViT-B/32 | `clip` | `CLIPAdapter.classify_image()` — per-class log-ratio scoring, zero-shot, no vocab menu |
| Qwen2.5-VL-7B | `qwen2_5vl` | generative, `generate()` → parse |
| InternVL3-8B | `internvl3` | generative, `generate()` → parse |

| `--prompt-style` | Parsing | Notes |
|---|---|---|
| `closed_vocab` | exact label substring-match | label menu in prompt |
| `open_caption` | keyword bags | free-form description prompt |
| `open_cot` | keyword bags | two-turn CoT: describe first → classify from description (no label menu) |

CLIP ignores `--prompt-style` (uses `classify_image()` duck-type path).
`open_cot` calls `VLMAdapter.generate_cot()` — default impl on the base class, works for
both Qwen and InternVL3 without extra adapter code.

**Pending runs** (add results to RESULTS_SNAPSHOT §5 when done):
- `clip` on all three datasets (zero-shot contrastive baseline)
- `qwen2_5vl` + `open_cot` on all three datasets
- `internvl3` + `open_cot` on all three datasets

## Architecture (decided)

LLaVA-style: image → encoder (frozen) → projector (MLP 2-layer) → token-prefix →
Qwen2.5-7B-Instruct (LoRA on attn + MLP).

- **Encoder:** whatever the DINO/RADIO track picks. Imported via
  `src.vision_encoder`. Frozen.
- **Projector:** 2-layer MLP from encoder hidden dim → Qwen hidden dim (4096).
  Fully trained.
- **LLM:** Qwen2.5-7B-Instruct, LoRA r=16/alpha=32 on attention + MLP modules.

## Next steps (in order)

1. **Wait for DINO/RADIO track to finalize the encoder.** Do not start training
   until `src/vision_encoder.py` exists and the probe numbers are in
   `project.md`. Without that, the projector input dim isn't pinned.
2. **Write `src/vlm_model.py`** — the assembly. Smoke-test forward pass on one
   VQA sample; confirm token counts and that the projector output drops into
   the right place in Qwen's input embeddings.
3. **Write `src/vlm_train.py`** — LoRA + projector training. Loss = standard LM
   loss on the assistant turn only. Use `data/vqa/train.jsonl` (owned by
   ANNOTATION, read-only).
4. **Smoke run on 100 samples / 1 epoch** — verify loss decreases, no shape
   errors, eval loop runs.
5. **Full train.** 1 H100, partition `mm`, node `nodemm07`. The DINO/RADIO
   track has the other GPU.
6. **Eval on val + test.** Report per q_type:
   - `presence` — balanced accuracy.
   - `type_open` — multi-label F1 vs `gt_categories`.
   - `type_choice` — exact-match accuracy ("None of the above" counts as a
     class).
   - `grounding` — IoU between predicted bbox and the union GT bbox; the bbox
     answer format is `[x_min, y_min, x_max, y_max]` normalised to [0,1].
   - `scenery` — defer; needs a separate judge. Skip for v1 main numbers.
7. **Snapshot.** Add a VLM section to `RESULTS_SNAPSHOT.md` with all four
   numbers per dataset.

## Dependencies on other tracks

- **Blocks on DINO/RADIO** for `src.vision_encoder` and the encoder identifier.
- **Blocks on ANNOTATION** for `data/vqa/{train,val,test}.jsonl` containing Q4
  (scenery) answers. Without Q4, the LM never learns the scene-description
  channel; you can still train without it but eval is incomplete.
- **Optional input from LLM track** — if the LLM track fine-tunes Qwen on text
  first, you swap the base weights and re-train. Don't pre-couple to that.

## Hard constraints

- Encoder is frozen. Don't quietly unfreeze it "to see if it helps."
- LoRA only on the LM. Full fine-tune of Qwen is out of scope (no headroom on
  1 H100).
- Use the existing `waste_vlm` conda env.
- Eval splits = the official VQA val + test (`data/vqa/val.jsonl`,
  `data/vqa/test.jsonl`) — both come from the site-stratified seed=0 split, so
  results are comparable with the seg/detection numbers in
  `RESULTS_SNAPSHOT.md`.

---

## Future idea: VLM-spatial feature fusion for waste localisation

**Motivation.** The PCA visualisation (§3.5.12) shows that DINO/RADIO patch
features contain rich spatial semantics but cannot trivially be mapped to
segmentation masks from text alone. VLMs know *what* waste looks like
semantically but their vision encoder (typically CLIP ViT-L/14 at 336²) has
coarser spatial resolution than DINO/RADIO. The idea is to combine the
*semantic reasoning* of the VLM with the *spatial precision* of DINO/RADIO to
produce grounded waste activation maps — without any bbox/mask supervision.

**Approach.**

1. Run InternVL3 (or Qwen2.5-VL) on a waste image with a grounding prompt
   ("Where is the waste in this image?") and register forward hooks on the
   LLM cross-attention layers to capture attention weights over visual tokens
   during generation.
2. Average the captured attention maps over heads and over the subset of
   generated tokens that correspond to waste-describing text (e.g. everything
   before the first punctuation mark after "waste").  Result: a spatial weight
   map [N_visual] over the VLM's visual token grid.
3. Upsample and align this VLM attention map to the DINO/RADIO spatial grid
   (32×32 at 512²) via bilinear interpolation.
4. Element-wise multiply the VLM attention weights with the DINO/RADIO PCA
   map — this highlights the spatial patches that are (a) semantically coherent
   according to the SSL encoder *and* (b) attended to by the LLM while
   describing waste.
5. Threshold the fused map to obtain a soft waste-presence mask; compare
   against GT segmentation masks to get a proxy IoU without any task-specific
   training.

**Why it matters for the paper.** This would be a direct visual proof that the
VLM bridges the "rich features, poor geometry" gap: DINO knows *where* patches
cluster, the VLM knows *which cluster is waste*, the fusion produces a
localisation the individual models cannot achieve alone.

**Implementation notes.**
- InternVL3's LLM is InternLM2; cross-attention to visual tokens lives in
  `llm.model.layers[i].self_attn` (InternLM2 uses full self-attention over the
  concatenated visual + text token sequence, not cross-attention). Hook the
  attention output weights from any layer in the top half of the model (layers
  16–32 for a 32-layer model are most semantic).
- Visual tokens appear first in the concatenated sequence; their count is
  `num_patches_list` (dynamic tiling) or a fixed constant in single-image mode.
- The fused activation map is a figure, not a number — present it alongside the
  PCA comparison in the paper.
- Implementation entry point: new script `src/vlm_spatial_fusion.py`.
- Prerequisite: DINO/RADIO track finalises encoder (done as of 2026-06-18,
  RADIOv2.5-L @ 512² + linear head chosen).
