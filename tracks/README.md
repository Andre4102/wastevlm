# Tracks — parallel Claude Code instances

Four streams of work, one Claude Code session per track. Open a fresh Claude Code
session in this repo, paste the corresponding `track_*.md` as the first message,
and the session has full context to pick up cold.

| Track | File | Owns |
|---|---|---|
| ANNOTATION | [track_annotation.md](track_annotation.md) | scenery generation, Opus QA pass, merge + regenerate VQA, eval-slice QC, stats |
| DINO/RADIO | [track_dino_radio.md](track_dino_radio.md) | vision encoder selection, feature extraction, linear-probe baseline |
| VLM | [track_vlm.md](track_vlm.md) | encoder + projector + Qwen2.5-7B-Instruct, train on VQA, eval on benchmark |
| LLM | [track_llm.md](track_llm.md) | LLM-side fine-tuning strategy (scope before starting) |

## Shared rules — all tracks

- **Roadmap of record:** [`../project.md`](../project.md). Read it once at session start.
- **Compute env:** `/home/ids/diecidue/miniconda3/envs/waste_vlm`. SLURM partition `mm`, `nodemm07`.
- **Hard constraint:** caption/QA generation uses **Claude Code agent-dispatch**
  (subscription), never the paid API.
- **GT provenance:** manual multi-interpreter photo-interpretation, NOT on-site
  inspection.
- **Two-person scope:** no external reviewers — don't gate on colleague feedback.
- **Fresh conda env per project**, never reuse a sibling env.

## Cross-track coordination

- Anthropic session caps are **shared across all sessions on the same
  subscription**. The ANNOTATION track is the heavy dispatcher (multi-wave
  sub-agent fan-out); the other three are light on dispatch. If you hit a cap in
  one session, all four are affected. Stagger heavy dispatch waves accordingly.
- **No track edits another track's files.** Boundaries:
  - ANNOTATION owns: `data/captions/scenery_chunks/`, `data/captions/scenery.jsonl`,
    `data/vqa/*.jsonl`, `src/scenery_gen.py`, `src/vqa_gen.py`,
    `src/vqa_split.py`, `src/vqa_labels.py`, `inspect_scenery.ipynb`.
  - DINO/RADIO owns: `src/vision_encoder*.py` (new), `results/.../encoder_probe/`.
  - VLM owns: `src/vlm_*.py` (new), `slurm_vlm_*.sh`, `results/.../vlm/`.
  - LLM owns: `src/lora_cpt_llm.py` (shelved), `src/build_cpt_data.py`,
    `slurm_lora_cpt.sh`, plus whatever the chosen strategy adds.
- If a track needs an artifact from another, **wait for it** — don't fork the file.
