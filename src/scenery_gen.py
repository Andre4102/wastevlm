"""Scenery re-caption generation for the VQA Q4 (scenery-description) task.

Produces a 2-3 sentence factual scenery description for every image in the
DW + AW m2 universe, with the explicit constraint that the description MUST
NOT mention waste / debris / garbage / dumping / pollution — even to deny
their presence. This makes the dangerous failure mode (fabricating waste
content in the answer) structurally impossible; the worst residual
hallucination is landcover mis-labelling (cropland vs pasture), which cannot
bias the VLM toward wrong waste predictions.

Mass generation runs via Claude Code agent-dispatch from a session: enumerate
pending chunks here, then dispatch one general-purpose subagent per chunk.

Output schema per JSONL line:
  {"image_path": "...", "scenery": "...", "model": "claude-code-agent"}

Files:
  data/captions/scenery_chunks/chunk_NNNN.jsonl   — per-agent outputs
  data/captions/scenery.jsonl                      — final merged file

    python -m src.scenery_gen plan --chunk-size 50
    python -m src.scenery_gen merge
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SPLIT_INDEX = Path("/home/ids/diecidue/data/vqa/split_index.jsonl")
CHUNKS_DIR = Path("/home/ids/diecidue/data/captions/scenery_chunks")
MERGED = Path("/home/ids/diecidue/data/captions/scenery.jsonl")

# Locked prompt — do NOT alter the no-waste constraint without updating project.md.
LOCKED_PROMPT = (
    "Describe the visible scenery in this aerial / drone photograph: terrain, "
    "vegetation, water, structures, roads, and visible human activity. Write "
    "two to three concise factual sentences. CRITICAL CONSTRAINT: you must "
    "NOT mention waste, debris, garbage, dumping, rubble, scrap, litter, or "
    "pollution — and you must NOT mention their absence either (do not write "
    "'no waste is visible' or any similar denial). Describe only the scene "
    "itself."
)


def build_agent_task(chunk_id: str, image_paths: list[str], out_path: str) -> str:
    """Full task string for one Claude Code general-purpose agent."""
    paths_block = "\n".join(f"  - {p}" for p in image_paths)
    return f"""You are an annotation worker. For each image listed below, look at it and write a short scenery description, then save all of them to ONE JSONL file.

THE PROMPT (apply identically to every image):
{LOCKED_PROMPT}

Images to caption (chunk {chunk_id}, {len(image_paths)} images):
{paths_block}

Procedure:
1. Read each image with the Read tool (PNG / JPG supported natively).
2. Compose the two-to-three-sentence scenery description following the prompt above. NEVER mention waste, debris, garbage, dumping, rubble, scrap, litter, pollution — or their absence.
3. Accumulate one JSON object per image:
     {{"image_path": "<path>", "scenery": "<your description>", "model": "claude-code-agent"}}
4. After processing all {len(image_paths)} images, write the full JSONL file in ONE Write call to:
     {out_path}
   One JSON object per line, in the same order as the input list. No extra commentary, no wrapping array.

Hard rules:
- Do not skip images. If a Read fails, still write a JSON line for it with "scenery": "ERROR: <reason>".
- Do NOT use Edit/append tools — accumulate in memory, then ONE final Write call.
- Your reply to me should be a single line summarising the count and any errors.

Begin."""


def _load_done() -> set[str]:
    """Image paths already covered by a non-error scenery line in any chunk or the merged file."""
    done: set[str] = set()
    files = [MERGED] + (list(CHUNKS_DIR.glob("chunk_*.jsonl")) if CHUNKS_DIR.exists() else [])
    for f in files:
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("scenery") and not d["scenery"].startswith("ERROR:"):
                done.add(d["image_path"])
    return done


def chunks_pending(chunk_size: int) -> list[dict]:
    """List of {chunk_id, image_paths, out_path} for images still missing scenery."""
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    done = _load_done()

    pending: list[str] = []
    for line in SPLIT_INDEX.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["image_path"] not in done:
            pending.append(r["image_path"])

    used = {p.name for p in CHUNKS_DIR.glob("chunk_*.jsonl")}
    chunks: list[dict] = []
    next_id = 0
    for i in range(0, len(pending), chunk_size):
        while f"chunk_{next_id:04d}.jsonl" in used:
            next_id += 1
        name = f"chunk_{next_id:04d}.jsonl"
        used.add(name)
        chunks.append({
            "chunk_id": f"{next_id:04d}",
            "image_paths": pending[i : i + chunk_size],
            "out_path": str(CHUNKS_DIR / name),
        })
        next_id += 1
    return chunks


def merge() -> int:
    """Concatenate all chunk_*.jsonl files into MERGED, deduped by image_path."""
    by_path: dict[str, dict] = {}
    for f in sorted(CHUNKS_DIR.glob("chunk_*.jsonl")):
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if not d.get("scenery") or d["scenery"].startswith("ERROR:"):
                continue
            by_path[d["image_path"]] = d
    MERGED.parent.mkdir(parents=True, exist_ok=True)
    with MERGED.open("w") as f:
        for path in sorted(by_path):
            f.write(json.dumps(by_path[path], ensure_ascii=False) + "\n")
    print(f"[merge] {len(by_path):,} scenery rows -> {MERGED}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--chunk-size", type=int, default=50)
    e = sub.add_parser("emit_tasks", help="emit JSON [{chunk_id, task, out_path}, ...] for orchestrator dispatch")
    e.add_argument("--n", type=int, default=16, help="how many pending chunks to emit")
    e.add_argument("--chunk-size", type=int, default=30)
    sub.add_parser("merge")
    args = ap.parse_args()

    if args.cmd == "plan":
        chunks = chunks_pending(args.chunk_size)
        total = sum(len(c["image_paths"]) for c in chunks)
        print(f"[plan] {len(chunks)} chunks, {total:,} images pending "
              f"(chunk_size={args.chunk_size})")
        for c in chunks[:3]:
            print(f"  {c['chunk_id']}: {len(c['image_paths'])} imgs -> {c['out_path']}")
        if len(chunks) > 3:
            print(f"  ... ({len(chunks) - 3} more)")
    elif args.cmd == "emit_tasks":
        chunks = chunks_pending(args.chunk_size)[: args.n]
        tasks = [{
            "chunk_id": c["chunk_id"],
            "out_path": c["out_path"],
            "task": build_agent_task(c["chunk_id"], c["image_paths"], c["out_path"]),
        } for c in chunks]
        print(json.dumps(tasks))
    else:
        return merge()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
