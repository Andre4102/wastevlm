"""Combined-prompt multi-label baseline runner.

For each model, asks Q5 (one prompt enumerating all categories) per image and
records the canonicalised set of predicted categories alongside ground truth.

Usage:
    python -m src.mlbaseline --model llava-next --dataset aerialwaste --n 200 \
        --out results/aerialwaste_ml_llava.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.baseline import build_runner  # noqa: E402
from src.datasets import (  # noqa: E402
    DRONEWASTE_PAPER_10,
    Sample,
    load_aerialwaste_mcml,
    load_aerialwaste_multilabel,
    load_dronewaste_multilabel,
)
from src.prompts import build_q5_multilabel, parse_q5_multilabel  # noqa: E402
from src.runner import open_image  # noqa: E402
from src.sampling import balanced_binary_sample  # noqa: E402


AERIALWASTE_ROOT = "/home/ids/diecidue/data/aerialwaste"
DRONEWASTE_ROOT = "/home/ids/diecidue/data/dronewaste"


def load_ml_samples(
    dataset: str,
    split: str,
    dw_paper_10: bool = False,
    aw_mcml_version: str | None = None,
) -> tuple[list[str], list[Sample]]:
    if dataset == "aerialwaste":
        if aw_mcml_version:
            # mcml splits use 'train'/'val'/'test' naming
            mcml_split = {"training": "train", "testing": "test"}.get(split, split)
            return load_aerialwaste_mcml(AERIALWASTE_ROOT, mcml_split, aw_mcml_version)
        return load_aerialwaste_multilabel(AERIALWASTE_ROOT, split=split)
    if dataset == "dronewaste":
        cf = DRONEWASTE_PAPER_10 if dw_paper_10 else None
        return load_dronewaste_multilabel(DRONEWASTE_ROOT, categories_filter=cf)
    raise ValueError(dataset)


def serialize_sample(s: Sample) -> dict:
    d = asdict(s)
    d["image_path"] = str(s.image_path)
    return d


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["aerialwaste", "dronewaste"], required=True)
    p.add_argument("--split", default="testing", help="aerialwaste only: training|testing")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", choices=["llava-next", "geochat", "geollava8k"], required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--dw-paper-10", action="store_true",
                   help="DroneWaste only: prompt + eval restricted to the paper's 10 classes")
    p.add_argument("--aw-mcml-version", choices=["m2", "m4"], default=None,
                   help="AerialWaste only: use the multi-class multi-label split (m2=5 cats, m4=6 cats)")
    p.add_argument("--no-balance", action="store_true",
                   help="Skip balanced subsampling; iterate all loaded samples")
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[load] {args.dataset} multi-label (dw_paper_10={args.dw_paper_10}, mcml={args.aw_mcml_version})")
    categories, samples = load_ml_samples(
        args.dataset, args.split,
        dw_paper_10=args.dw_paper_10, aw_mcml_version=args.aw_mcml_version,
    )
    print(f"  {len(samples)} total samples ({sum(s.label for s in samples)} positives), "
          f"{len(categories)} categories")

    if args.no_balance:
        chosen = list(samples)
    else:
        chosen = balanced_binary_sample(samples, n=args.n, seed=args.seed)
    if args.limit:
        chosen = chosen[: args.limit]
    print(f"  {len(chosen)} chosen "
          f"({sum(s.label for s in chosen)} pos / {len(chosen) - sum(s.label for s in chosen)} neg)")

    prompt = build_q5_multilabel(categories)
    print(f"[prompt] {len(prompt)} chars, {len(categories)} categories listed")

    print(f"[load] model={args.model}")
    runner = build_runner(args.model, max_new_tokens=args.max_new_tokens)

    t0 = time.time()
    n_done = 0
    with out_path.open("w") as fout:
        for s in tqdm(chosen, desc="ml-infer"):
            try:
                image = open_image(s.image_path)
            except Exception as e:
                print(f"  [skip] {s.image_path}: {e}", file=sys.stderr)
                continue

            resp = runner.ask(image, prompt)
            predicted = parse_q5_multilabel(resp.text, categories)

            row = {
                "sample": serialize_sample(s),
                "text": resp.text,
                "predicted": sorted(predicted),
                "categories": categories,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1

    elapsed = time.time() - t0
    print(f"[done] wrote {n_done} rows to {out_path} in {elapsed:.1f}s "
          f"({elapsed / max(1, n_done):.2f}s/img)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
