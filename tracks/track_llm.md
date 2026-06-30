# Track: LLM

You are the LLM track of the Waste-VLM project. Read
`/home/ids/diecidue/scripts/waste_vlm/project.md` and
`/home/ids/diecidue/scripts/waste_vlm/tracks/README.md` first — they have the
full design and the shared rules.

## Your scope

You own LLM-side fine-tuning of Qwen2.5-7B-Instruct **before** it ever sees an
image. Your output is either a fine-tuned base checkpoint that the VLM track
swaps in, OR a decision to skip this stage entirely. Either is a valid outcome.

You do NOT touch: the VQA generation pipeline, the vision encoder, the projector,
or the VLM training loop. Those are owned by ANNOTATION / DINO/RADIO / VLM
tracks.

Files you own:
- `src/lora_cpt_llm.py` — shelved text-CPT trainer (`accelerate` was bumped
  0.34.2 → 1.2.1, so it runs again).
- `src/build_cpt_data.py` — corpus → block builder.
- `slurm_lora_cpt.sh` — SLURM launcher.
- `data/waste_corpus/cpt_blocks.jsonl` — shelved 389K-token block-packed corpus.
- Anything new your chosen strategy needs (`src/llm_instruct_tune.py`, etc.).

## STEP 0 — Scope the strategy (do this first, write a memo to project.md)

Three plausible reads of "LLM fine-tuning" for this project. Pick ONE before
writing any code:

1. **Text-only instruction-tune on the VQA QA pairs.** Take
   `data/vqa/train.jsonl` (owned by ANNOTATION, read-only), strip the `<image>`
   token, train Qwen on the text-only question→answer pairs. Gives the LM the
   waste vocabulary and answer-format priors before it ever sees a projector.
   - Pros: cheap; reuses the same QA we already generated; aligns LM behavior
     with the eval format.
   - Cons: model learns to answer waste questions without an image — could
     reinforce hallucination (saying "yes" without evidence).
   - Verdict: high-yield if you mask the GT-derived answers carefully or only
     train on Q4 (scenery, language-only side).

2. **Domain-adapt on the waste corpus (revive CPT).** The original
   text-CPT idea, now with `accelerate >= 1.2.1` and unblocked. Train on the
   49-doc, ~955K-token waste corpus + EU List of Waste / Commission Decision
   2014/955/EU.
   - Pros: gives the LM a real waste / regulatory vocabulary; sets up the
     deferred "risk & regulatory-reasoning" Q-type for v2.
   - Cons: 389K tokens after block-packing is still tiny; the May-29 decision
     memo argued this was low-yield. If you revive it, scope tightly — e.g.,
     LoRA r=8, 1 epoch, eval ppl on a held-out slice.
   - Verdict: only if (1) and (3) are ruled out.

3. **Skip.** Use Qwen2.5-7B-Instruct as-is. The VLM track LoRA-tunes the LM
   side during multimodal training anyway, so a separate text stage may not
   pay for itself.
   - Pros: zero work; one less coupling point; less risk of drift.
   - Cons: leaves the waste/regulatory vocabulary untouched.
   - Verdict: the safe default. Pick this if your evidence for (1) or (2) is
     thin.

**Write a 1-paragraph decision memo to `project.md`** under a new "LLM-side
strategy" section before writing any training code. Include why you picked
the option and how you'll evaluate it.

## Next steps (after the memo)

- If option **1** (text-only instruction-tune): build a stripped train split
  from `data/vqa/train.jsonl` (no images, no `<image>` token), LoRA-tune Qwen
  on it, eval on `data/vqa/val.jsonl` text-only. Output checkpoint goes to
  `results/waste_vlm/weights/qwen-vqa-text-lora/`.
- If option **2** (revive CPT): `sbatch slurm_lora_cpt.sh`. Reuse the existing
  pipeline; smoke-test on `cpt_blocks.jsonl[:100]` first.
- If option **3** (skip): note the decision in `project.md`, mark this track
  done, and pause.

## Dependencies on other tracks

- **Optional read** of `data/vqa/train.jsonl` and `data/vqa/val.jsonl` (owned
  by ANNOTATION). Read-only.
- **Produces for VLM:** if you train a checkpoint, the VLM track swaps the
  Qwen base weights in `src/vlm_model.py`. Coordinate naming.

## Hard constraints

- Use the existing `waste_vlm` conda env.
- One H100, partition `mm`. The DINO/RADIO and VLM tracks have the other
  H100; coordinate scheduling.
- LoRA only. No full fine-tune of Qwen.
- If you revive CPT, do not regenerate `cpt_blocks.jsonl` — it's shelved
  intentionally, and ANNOTATION owns nothing in `data/waste_corpus/`.
