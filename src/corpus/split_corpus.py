"""Partition the corpus into train / held-out eval documents BEFORE CPT.

This is the fix for train==eval leakage: CPT trains only on the *train* docs,
and generalisation is measured on *eval* docs the model never saw (held-out
waste-domain perplexity; optionally held-out QA). The split is deterministic
(md5 of doc id) and stratified per source.

By default only *prose* sources (Wikipedia) contribute held-out eval docs, and a
fraction is held out per source. The EU List of Waste (2014/955/EU) is kept
entirely in TRAIN — it is the memorisation axis (`low_qa`), not a generalisation
target (arbitrary 6-digit codes aren't derivable from other text).

    python -m src.corpus.split_corpus --eval-frac 0.12

Writes ``corpus_train.jsonl`` + ``corpus_eval.jsonl`` next to ``corpus.jsonl``.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List

from .common import CORPUS_ROOT, read_jsonl, write_jsonl

# Sources eligible to contribute held-out eval docs (prose only).
DEFAULT_EVAL_SOURCES = {"wikipedia"}
# Docs pinned to TRAIN regardless (memorisation targets).
PIN_TRAIN_IDS = {"eurlex:32014D0955"}


def _hash_unit(s: str) -> float:
    """Deterministic float in [0,1) from a string (stable across runs)."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(1 << 48)


def split_docs(docs: List[Dict], eval_frac: float,
               eval_sources: set) -> Dict[str, List[Dict]]:
    train, evl = [], []
    for d in docs:
        pin = d["id"] in PIN_TRAIN_IDS or d["source"] not in eval_sources
        if not pin and _hash_unit(d["id"]) < eval_frac:
            evl.append(d)
        else:
            train.append(d)
    return {"train": train, "eval": evl}


def _summary(docs: List[Dict]) -> str:
    from collections import Counter
    by_src = Counter(d["source"] for d in docs)
    words = sum(len(d["text"].split()) for d in docs)
    return f"{len(docs)} docs, ~{words/1e6:.2f}M words, by_source={dict(by_src)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=CORPUS_ROOT / "corpus.jsonl")
    ap.add_argument("--eval-frac", type=float, default=0.12)
    ap.add_argument("--eval-sources", nargs="+", default=sorted(DEFAULT_EVAL_SOURCES))
    args = ap.parse_args()

    docs = list(read_jsonl(args.corpus))
    parts = split_docs(docs, args.eval_frac, set(args.eval_sources))

    out_train = args.corpus.with_name("corpus_train.jsonl")
    out_eval = args.corpus.with_name("corpus_eval.jsonl")
    write_jsonl(out_train, parts["train"])
    write_jsonl(out_eval, parts["eval"])

    print(f"eval-frac={args.eval_frac} eval-sources={sorted(set(args.eval_sources))}")
    print(f"  train: {_summary(parts['train'])}")
    print(f"  eval : {_summary(parts['eval'])}")
    print(f"Wrote {out_train}\n      {out_eval}")
    if not parts["eval"]:
        print("⚠  eval split is empty — check --eval-frac / --eval-sources.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
