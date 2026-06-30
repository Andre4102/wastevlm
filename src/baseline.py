"""Run LLaVA-NeXT-7B zero-shot baseline on AerialWaste and/or DroneWaste.

Usage:
    python -m src.baseline --dataset aerialwaste --n 200 --out results/aerial.jsonl
    python -m src.baseline --dataset dronewaste --n 200 --out results/drone.jsonl

The output is JSONL: one row per sample with all four Q1-Q4 responses, plus
calibrated p(yes) and per-letter probabilities. `report.py` consumes this.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

# Allow `python -m src.baseline` and direct `python src/baseline.py`
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import Sample, load_aerialwaste, load_dronewaste  # noqa: E402
from src.prompts import ALL_QUESTIONS  # noqa: E402
from src.runner import LlavaNextRunner, open_image  # noqa: E402
from src.sampling import balanced_binary_sample  # noqa: E402


AERIALWASTE_ROOT = "/home/ids/diecidue/data/aerialwaste"
DRONEWASTE_ROOT = "/home/ids/diecidue/data/dronewaste"


def build_runner(model: str, max_new_tokens: int):
    if model == "llava-next":
        return LlavaNextRunner(max_new_tokens=max_new_tokens)
    if model == "geochat":
        from src.geochat_runner import GeoChatRunner  # lazy: heavy vendored deps
        return GeoChatRunner(max_new_tokens=max_new_tokens)
    if model == "geollava8k":
        from src.geollava8k_runner import GeoLlava8KRunner
        return GeoLlava8KRunner(max_new_tokens=max_new_tokens)
    raise ValueError(f"unknown model {model!r}")


def load_samples(dataset: str, split: str) -> list[Sample]:
    if dataset == "aerialwaste":
        return load_aerialwaste(AERIALWASTE_ROOT, split=split)
    if dataset == "dronewaste":
        return load_dronewaste(DRONEWASTE_ROOT)
    raise ValueError(f"unknown dataset {dataset!r}")


def serialize_sample(s: Sample) -> dict:
    d = asdict(s)
    d["image_path"] = str(s.image_path)
    return d


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["aerialwaste", "dronewaste"], required=True)
    p.add_argument("--split", default="testing", help="aerialwaste only: training|testing")
    p.add_argument("--n", type=int, default=200, help="balanced sample size")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--model",
        choices=["llava-next", "geochat", "geollava8k"],
        default="llava-next",
    )
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--out", required=True, help="output JSONL path")
    p.add_argument("--limit", type=int, default=None, help="hard cap (debug)")
    p.add_argument("--questions", default="Q1,Q2,Q3,Q4")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.dataset} ({args.split if args.dataset == 'aerialwaste' else 'all'})")
    samples = load_samples(args.dataset, args.split)
    print(f"  {len(samples)} total samples; sampling {args.n} balanced...")
    chosen = balanced_binary_sample(samples, n=args.n, seed=args.seed)
    if args.limit:
        chosen = chosen[: args.limit]
    print(f"  {len(chosen)} samples selected")

    questions = [q for q in args.questions.split(",") if q.strip()]
    for q in questions:
        if q not in ALL_QUESTIONS:
            raise ValueError(f"unknown question {q!r}")

    print(f"[load] model={args.model}")
    runner = build_runner(args.model, max_new_tokens=args.max_new_tokens)

    t0 = time.time()
    n_done = 0
    with out_path.open("w") as fout:
        for s in tqdm(chosen, desc="infer"):
            try:
                image = open_image(s.image_path)
            except Exception as e:
                print(f"  [skip] {s.image_path}: {e}", file=sys.stderr)
                continue

            row = {
                "sample": serialize_sample(s),
                "responses": {},
            }
            for q_id in questions:
                question = ALL_QUESTIONS[q_id]
                resp = runner.ask(
                    image,
                    question,
                    compute_yes_no=(q_id == "Q1"),
                    compute_letters=(q_id == "Q3"),
                )
                row["responses"][q_id] = {
                    "text": resp.text,
                    "p_yes": resp.p_yes,
                    "p_no": resp.p_no,
                    "letter_probs": resp.letter_probs,
                }

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1

    elapsed = time.time() - t0
    print(f"[done] wrote {n_done} rows to {out_path} in {elapsed:.1f}s ({elapsed/max(1,n_done):.2f}s/img)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
