# Alignment Dataset Setup — Act 2 Alignment-Budget Probe

Goal: download and normalize three datasets into matched-format shards so they can
be combined into 3 token-budget-matched training arms (density / diversity / combined)
for the RADIO-frozen VLM alignment-budget scaling experiment.

Datasets:
1. ShareGPT4V-PT (dense captions, ~1.2M) — caption density arm
2. Vision-Flan (187 tasks, ~1.66M instances) — task diversity arm
3. PixMo (Molmo dataset family — dense captions + point grounding) — spatial grounding arm

---

## 0. Setup

```bash
export DATA_ROOT=${DATA_ROOT:-/scratch/$USER/alignment_data}   # override per cluster (Jean Zay / Leonardo scratch path)
mkdir -p $DATA_ROOT/{raw,normalized,logs}
pip install --break-system-packages -U huggingface_hub datasets pillow tqdm
```

Set `HF_TOKEN` in env if any of the source repos are gated.

---

## 1. Download

### 1a. ShareGPT4V-PT
```bash
huggingface-cli download Lin-Chen/ShareGPT4V ShareGPT4V-PT.json \
  --repo-type dataset --local-dir $DATA_ROOT/raw/sharegpt4v
huggingface-cli download Lin-Chen/ShareGPT4V \
  --repo-type dataset --local-dir $DATA_ROOT/raw/sharegpt4v --include "*.zip" --include "*.tar"
# ShareGPT4V-PT images are drawn from existing corpora (COCO, SAM, LAION, CC, SBU, TextCaps, WikiArt).
# Confirm in the repo README which image archives are actually required for -PT vs the smaller
# -Captioner-100K instruct set, and only pull those to save disk.
```

### 1b. Vision-Flan
```bash
huggingface-cli download vision-flan/vision-flan_191-task_1k \
  --repo-type dataset --local-dir $DATA_ROOT/raw/vision_flan
# Note: check whether you want the 191-task (paper) release or a later revision; the repo has
# gone through naming changes (Vision-Flan vs Vision-Flan_v2 etc.) — list files first:
huggingface-cli download vision-flan/vision-flan_191-task_1k --repo-type dataset \
  --local-dir $DATA_ROOT/raw/vision_flan --dry-run 2>&1 | tee $DATA_ROOT/logs/vision_flan_manifest.txt
```

### 1c. PixMo
```bash
huggingface-cli download allenai/pixmo-cap --repo-type dataset --local-dir $DATA_ROOT/raw/pixmo_cap
huggingface-cli download allenai/pixmo-points --repo-type dataset --local-dir $DATA_ROOT/raw/pixmo_points
huggingface-cli download allenai/pixmo-cap-qa --repo-type dataset --local-dir $DATA_ROOT/raw/pixmo_capqa
# PixMo images are hosted as URLs, not bundled — there is a download script in the dataset repo
# (pixmo_datasets / download.py in the Molmo repo). Use it, expect ~5-15% link rot, log failures
# rather than silently dropping (affects final sample count you report).
```

Log the actual downloaded sample counts per dataset to `$DATA_ROOT/logs/download_counts.txt` —
you'll need real (not nominal paper) counts for token-budget matching in step 3.

---

## 2. Normalize to a common schema

Convert each dataset to the same JSONL schema (LLaVA-style conversation format), one file
per dataset under `$DATA_ROOT/normalized/<name>.jsonl`:

```json
{"id": "sharegpt4v_000123", "image": "sharegpt4v/coco/train2017/000000123.jpg",
 "conversations": [{"from": "human", "value": "<image>\nDescribe this image in detail."},
                    {"from": "gpt", "value": "..."}],
 "source": "sharegpt4v", "task_type": "dense_caption"}
```

Write one converter script per source under `scripts/convert_<name>.py`. Required per script:
- Resolve/copy or symlink images into a single flat `$DATA_ROOT/normalized/images/` tree with
  source-prefixed filenames to avoid collisions (`sharegpt4v_*`, `visionflan_*`, `pixmo_*`).
- Preserve `task_type` (dense_caption / vqa / grounding / referring / point / ocr / relation ...)
  so arms can be filtered/rebalanced later without re-parsing raw data.
- Drop or flag samples with missing/corrupt images (`PIL.Image.open(...).verify()`), write
  failures to `$DATA_ROOT/logs/<name>_bad_images.txt` rather than crashing the run.
- Tokenize each conversation's text with the target LLM tokenizer and record `n_text_tokens`
  in the JSONL record — this is what "token-budget-matched" will subsample against.

For PixMo specifically: point-grounding annotations (x,y + referring phrase) need a template
to render as text, e.g. `"gpt"` value `"The <object> is located at approximately (x, y)."` or
whatever coordinate-representation convention matches how you're evaluating grounding
downstream — decide this before conversion, don't leave it to the training script.

---

## 3. Build the three matched-budget arms

Compute total text-token count for each normalized dataset. Set the budget to
`min(tokens_sharegpt4v, tokens_visionflan, tokens_pixmo)` and subsample the larger ones down
to that budget (random sample, seeded, stratified by `task_type` if subsampling Vision-Flan
or PixMo, since those are task-heterogeneous and naive random sampling could skew task mix).

Arms:
- `arm_density`   = ShareGPT4V-PT only, at budget B
- `arm_diversity` = Vision-Flan only, at budget B
- `arm_spatial`   = PixMo (cap + points) only, at budget B
- `arm_combined`  = equal 1/3-budget split of all three, total = B (for a 4th "does combining
  help beyond either alone" control — optional but cheap once the above exist)

Write manifests as `$DATA_ROOT/normalized/arm_<name>.jsonl` (concatenated, shuffled, seeded).
Log final image count, text-token count, and task_type histogram per arm to
`$DATA_ROOT/logs/arm_stats.json` — needed for the paper/thesis table regardless of results.

---

## 4. Wire into training config

- Point the existing LLaVA-style pretrain script's `--data_path` / `--image_folder` at
  `arm_<name>.jsonl` and `normalized/images/` respectively.
- Keep RADIO 2.5 encoder frozen, LLM frozen (stage 1 / connector-only) — do not change any
  other hyperparameter (LR, batch size, warmup, connector architecture) across arms; the
  only variable should be which arm's data is loaded.
- Run all arms for the same number of optimizer steps, not the same number of epochs
  (epoch counts differ across arms until step 3's subsampling makes token counts equal —
  should be a non-issue if step 3 was done correctly, but assert `len(arm) == expected`
  before launching as a sanity check).
- Log to the same eval harness (staged evaluation protocol) used for Act 1/Act 3 so results
  are directly comparable on faithfulness as the headline metric.

---

## 5. Sanity checks before launching real runs

- [ ] Spot-check 10 random samples per arm by rendering image + conversation to a scratch
      HTML/markdown file and eyeballing — catches image/text misalignment from step 2 bugs.
- [ ] Confirm no image path collisions across datasets (`sort images/ | uniq -d` should be empty).
- [ ] Confirm token budgets are within ~2% of each other across arms.
- [ ] Dry-run 50 training steps per arm before committing full Jean Zay/Leonardo allocation.
