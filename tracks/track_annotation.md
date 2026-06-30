# Track: ANNOTATION

You are the ANNOTATION track of the Waste-VLM project. Read
`/home/ids/diecidue/scripts/waste_vlm/project.md` and
`/home/ids/diecidue/scripts/waste_vlm/tracks/README.md` first — they have the
full design and the shared rules.

## Your scope (and only yours)

You own the VQA dataset: scenery captions, the Q4 generation pipeline, the
merge/regenerate step, the eval-slice QC, and the stats snapshot. Do NOT touch
the vision encoder, the projector, the LLM, or any training code — those are
owned by the other three tracks.

Files you own:
- `data/captions/scenery_chunks/` — per-agent JSONL outputs (append-only).
- `data/captions/scenery.jsonl` — merged final scenery file.
- `data/vqa/{split_index.jsonl, paraphrase_bank.json, dispatch_queue.json,
  train.jsonl, val.jsonl, test.jsonl}`.
- `src/scenery_gen.py`, `src/vqa_gen.py`, `src/vqa_split.py`,
  `src/vqa_labels.py`.
- `inspect_scenery.ipynb`.

## Current state (as of 2026-06-17)

- 2,464 / 10,535 scenery captions on disk across 83 chunks; **0 forbidden-word
  violations**.
- Sonnet sub-agent pool just got capped after a 2-wide wave for chunks 0083–0084
  refused with the permission-style failure pattern (~12k subagent tokens, 1–2
  tool uses). Haiku pool also recently capped. Wait for resets before resuming.
- Locked scenery prompt and the worker-task assembly live in `src/scenery_gen.py`
  (`LOCKED_PROMPT`, `build_agent_task`). Do NOT alter the no-waste constraint
  without updating `project.md`.

## Steady-state dispatch loop

```bash
# 1. refresh the dispatch queue (idempotent; reads chunks-on-disk to find pending)
python -m src.scenery_gen emit_tasks --n 2 --chunk-size 30 \
  > /home/ids/diecidue/data/vqa/dispatch_queue.json

# 2. dispatch 2 sub-agents in parallel (one per queue index)
#    use the queue[N] ONLY + interleaved-Read-describe prompt — see git log
#    for the latest version (rewritten with the research-framing preamble after
#    Sonnet's safety misread on the no-waste constraint).
#    Model: haiku or sonnet (alternate to spread cap pressure).

# 3. wait for both completion notifications, then loop.
```

The worker prompt MUST include the research-framing preamble (Q1–Q3 carry the
waste signal via GT labels; Q4 scenery-only is anti-hallucination, not a
cover-up). Without it, Sonnet workers will refuse on safety grounds.

## Next steps (in order)

1. **Finish scenery generation.** ~8,071 images remain. Steady-state 2-wide
   waves until 10,535 captions are on disk. Alternate Haiku/Sonnet to spread
   cap pressure across pools.
2. **Opus QA pass.** Batched verifier — for each scenery caption, give Opus the
   image + the text and have it flag drift / forbidden-word leak / wrong
   landcover phrasing. Output a separate JSONL of flagged rows; do NOT
   overwrite the original captions.
3. **Merge + regenerate VQA.**
   ```bash
   python -m src.scenery_gen merge   # -> data/captions/scenery.jsonl
   python -m src.vqa_gen --seed 0    # rebuilds train/val/test.jsonl with Q4
   ```
4. **Sanity-check eval slice.** ~50 items from `data/vqa/test.jsonl` — verify
   answerable from image alone, GT-derived answer matches a human read, scenery
   no-waste compliance held.
5. **Stats + snapshot.** Counts per q_type × dataset × split; add a VQA section
   to `RESULTS_SNAPSHOT.md`.

## Shared-cap warning

The other three tracks (DINO/RADIO, VLM, LLM) share your Anthropic subscription.
Heavy dispatch waves from this track will starve their sessions. Stagger your
waves accordingly, and watch for the permission-style refusal pattern (1–2 tool
uses, ~12k subagent tokens) as the cap-hit signal.

## Hard constraints

- Caption/QA generation uses **Claude Code agent-dispatch** (subscription), never
  the paid API.
- All v1 waste-related answers (Q1–Q3) are derived 100% from GT. Captions are
  NOT consulted for waste answers.
- The Q4 scenery prompt's no-waste constraint is locked. Do not relax it.
