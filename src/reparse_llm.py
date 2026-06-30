"""LLM-judge re-parser for VLM open_cot raw responses.

Reads an existing raw_responses.jsonl (no new VLM inference), sends batches
of VLM text to the claude CLI (already authenticated via Claude Code), gets
back JSON bool dicts of which waste classes are present, recomputes F1.
Saves test_eval_llm_judge.json alongside the source file.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dinotxt_zeroshot import ml_metrics  # noqa: E402

# ── dataset metadata ──────────────────────────────────────────────────────────

DATASET_TO_DESC_JSON = {
    "dw_paper10": ROOT / "src" / "paper10_descriptions.json",
    "aw_m2":      ROOT / "src" / "aw_m2_descriptions.json",
    "aw_m4":      ROOT / "src" / "aw_m4_descriptions.json",
}

DATASET_META = {
    "dw_paper10": {"task": "dw_paper_10"},
    "aw_m2":      {"task": "aw_mcml_m2"},
    "aw_m4":      {"task": "aw_mcml_m4"},
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_classes(desc_path: Path) -> dict[str, dict]:
    raw = json.loads(desc_path.read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def is_empty(raw: str) -> bool:
    return not raw or raw.strip().lower() in {"none", "none.", ""}


def find_claude_bin() -> str:
    # prefer the Claude Code binary already in the environment
    cc = os.environ.get("CLAUDE_CODE_EXECPATH", "")
    if cc and Path(cc).exists():
        return cc
    return "claude"  # fall back to PATH


def build_batch_prompt(classes: dict[str, dict], texts: list[str]) -> str:
    lines = [
        "A drone captured aerial images of illegal waste dump sites.",
        "Several vision models described the images; their texts are listed below.",
        "",
        "For EACH numbered description, decide which waste categories are",
        "mentioned or described (even indirectly — by paraphrase, visual",
        "description, or synonym).",
        "",
        "Waste categories (aerial appearance + synonyms):",
    ]
    for name, info in classes.items():
        cue  = info.get("aerial_cue", "")
        tags = ", ".join(info.get("clip_tags", []))
        lines.append(f'  "{name}": {cue}. Synonyms: {tags}.')
    lines += [
        "",
        "Descriptions:",
    ]
    for idx, txt in enumerate(texts):
        # truncate very long texts so the prompt stays manageable
        snippet = txt[:800].replace('"', "'")
        lines.append(f'[{idx}] "{snippet}"')
    lines += [
        "",
        "Return ONLY a JSON array with one object per description (same order).",
        "Each object maps every category name exactly to true or false.",
        "No explanation, no markdown fences — raw JSON only.",
        "Example for 2 descriptions: "
        + json.dumps([{k: False for k in classes}, {k: False for k in classes}]),
    ]
    return "\n".join(lines)


def _call_claude(bin_path: str, model: str, prompt: str, max_retries: int = 4) -> str:
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                [bin_path, "-p", prompt, "--model", model],
                capture_output=True, text=True, timeout=120,
            )
            return result.stdout.strip()
        except subprocess.TimeoutExpired:
            wait = 2 ** attempt
            print(f"  [timeout] retry in {wait}s …", flush=True)
            time.sleep(wait)
        except Exception as exc:
            print(f"  [error] {exc}", flush=True)
            time.sleep(2 ** attempt)
    return "[]"


def call_judge_batch(
    bin_path: str,
    model: str,
    classes: dict[str, dict],
    cats: list[str],
    texts: list[str],
) -> list[dict[str, bool]]:
    """Run one batch; returns a list of {class: bool} dicts, one per text."""
    fallback = [{c: False for c in cats}] * len(texts)
    prompt = build_batch_prompt(classes, texts)
    raw_out = _call_claude(bin_path, model, prompt)

    # strip accidental markdown fences
    if "```" in raw_out:
        parts = raw_out.split("```")
        for part in parts:
            p = part.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("["):
                raw_out = p
                break

    try:
        parsed = json.loads(raw_out)
        if not isinstance(parsed, list) or len(parsed) != len(texts):
            return fallback
        result = []
        for item in parsed:
            row = {}
            for c in cats:
                row[c] = bool(item.get(c, False))
            result.append(row)
        return result
    except json.JSONDecodeError:
        return fallback


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Re-parse VLM CoT raw responses with an LLM judge.")
    p.add_argument("--raw-jsonl", type=Path, required=True)
    p.add_argument("--dataset", choices=list(DATASET_TO_DESC_JSON), required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--model", default="claude-haiku-4-5-20251001",
                   help="Claude model alias or full ID (default: claude-haiku-4-5-20251001)")
    p.add_argument("--turn", type=int, choices=[1, 2], default=2,
                   help="Which CoT turn to judge: 1=pure description, 2=classification output (default: 2)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Records per claude CLI call (default: 8)")
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel subprocess workers (default: 2)")
    args = p.parse_args()

    bin_path = find_claude_bin()
    print(f"[judge] claude binary: {bin_path}")

    desc_path = DATASET_TO_DESC_JSON[args.dataset]
    classes_dict = load_classes(desc_path)
    cats = list(classes_dict.keys())
    n_cats = len(cats)
    print(f"[judge] dataset={args.dataset}  classes={n_cats}  model={args.model}")

    records = [json.loads(l) for l in
               args.raw_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[judge] records: {len(records)}")

    Y_true = np.zeros((len(records), n_cats), dtype=np.int32)
    Y_pred = np.zeros((len(records), n_cats), dtype=np.int32)

    # ground truth
    for i, rec in enumerate(records):
        for label in rec.get("gt", []):
            if label in cats:
                Y_true[i, cats.index(label)] = 1

    # select which turn's text to judge
    raw_field = "raw" if args.turn == 2 else "raw_turn1"
    if args.turn == 1 and not any("raw_turn1" in r for r in records):
        print(f"[judge] ERROR: --turn 1 requested but no 'raw_turn1' field found in {args.raw_jsonl}")
        print("[judge] Re-run vlm_eval.py with open_cot to generate raw_turn1 fields.")
        return 1

    # collect non-empty records to judge
    to_judge: list[tuple[int, str]] = [
        (i, rec[raw_field]) for i, rec in enumerate(records)
        if not is_empty(rec.get(raw_field, ""))
    ]
    n_empty = len(records) - len(to_judge)
    print(f"[judge] to judge: {len(to_judge)}  empty skipped: {n_empty}")

    # build batches
    batches: list[list[tuple[int, str]]] = []
    for start in range(0, len(to_judge), args.batch_size):
        batches.append(to_judge[start:start + args.batch_size])

    print(f"[judge] batches: {len(batches)}  batch_size: {args.batch_size}  workers: {args.workers}")

    done = 0

    def run_batch(batch: list[tuple[int, str]]) -> tuple[list[int], list[dict[str, bool]]]:
        idxs  = [x[0] for x in batch]
        texts = [x[1] for x in batch]
        preds = call_judge_batch(bin_path, args.model, classes_dict, cats, texts)
        return idxs, preds

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_batch, b): b for b in batches}
        for fut in as_completed(futures):
            idxs, preds = fut.result()
            for i, pred in zip(idxs, preds):
                for c, present in pred.items():
                    if present and c in cats:
                        Y_pred[i, cats.index(c)] = 1
            done += len(idxs)
            print(f"  judged {done}/{len(to_judge)} …", flush=True)

    scores = Y_pred.astype(np.float32)
    rep = ml_metrics(cats, Y_true, Y_pred, scores)
    meta = DATASET_META[args.dataset]
    rep["task"]          = meta["task"]
    rep["dataset"]       = args.dataset
    rep["judge_model"]   = args.model
    rep["prompt_style"]  = f"open_cot_turn{args.turn}+llm_judge"
    rep["cot_turn"]      = args.turn
    rep["n_empty_raw"]   = n_empty
    rep["n_judged"]      = len(to_judge)
    rep["raw_jsonl"]     = str(args.raw_jsonl)

    print(f"\n[judge] micro F1 = {rep['micro']['f1']:.4f}  macro F1 = {rep['macro']['f1']:.4f}")
    for name, pc in rep["per_class"].items():
        f1 = pc["f1"]
        f1_str = f"{f1:.3f}" if f1 is not None else "  N/A"
        print(f"  {name:<45s}  F1={f1_str}  support={pc['support']}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(rep, indent=2))
    print(f"\n[judge] saved -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
