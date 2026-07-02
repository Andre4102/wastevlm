"""Merge the per-source JSONL files into one deduplicated corpus + stats.

Reads ``wikipedia.jsonl``, ``eurlex.jsonl``, ``lea.jsonl`` from the corpus root,
drops exact and near-duplicate documents, applies a min-length filter, and
writes ``corpus.jsonl`` + ``stats.json``.

    python -m src.corpus.build_corpus --min-chars 400
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .common import CORPUS_ROOT, read_jsonl, write_jsonl

_SOURCE_FILES = ["wikipedia.jsonl", "eurlex.jsonl", "lea.jsonl"]
_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm_key(text: str) -> str:
    """Normalised content hash for exact/near-exact dedup."""
    norm = _ALNUM.sub(" ", text.lower()).strip()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _shingles(text: str, k: int = 8) -> frozenset:
    toks = _ALNUM.sub(" ", text.lower()).split()
    return frozenset(
        hash(" ".join(toks[i:i + k])) for i in range(0, max(1, len(toks) - k), 4)
    )


def dedup(docs: List[Dict], jaccard_thresh: float = 0.8) -> List[Dict]:
    kept: List[Dict] = []
    seen_ids: set = set()
    seen_hash: set = set()
    sig_index: List[frozenset] = []
    for d in sorted(docs, key=lambda x: -x["n_chars"]):  # keep longest first
        if d["id"] in seen_ids:
            continue
        h = _norm_key(d["text"])
        if h in seen_hash:
            continue
        sig = _shingles(d["text"])
        dup = False
        for prev in sig_index:
            if not sig or not prev:
                continue
            inter = len(sig & prev)
            union = len(sig | prev)
            if union and inter / union >= jaccard_thresh:
                dup = True
                break
        if dup:
            continue
        seen_ids.add(d["id"])
        seen_hash.add(h)
        sig_index.append(sig)
        kept.append(d)
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=CORPUS_ROOT)
    ap.add_argument("--min-chars", type=int, default=400)
    ap.add_argument("--jaccard", type=float, default=0.8)
    args = ap.parse_args()

    docs: List[Dict] = []
    for fn in _SOURCE_FILES:
        p = args.root / fn
        if p.exists():
            batch = [d for d in read_jsonl(p) if d["n_chars"] >= args.min_chars]
            docs.extend(batch)
            print(f"  {fn}: {len(batch)} docs (>= {args.min_chars} chars)")
        else:
            print(f"  {fn}: MISSING (skipped)")

    before = len(docs)
    docs = dedup(docs, args.jaccard)
    print(f"Dedup: {before} → {len(docs)} docs")

    out = args.root / "corpus.jsonl"
    write_jsonl(out, docs)

    by_source = Counter(d["source"] for d in docs)
    by_license = Counter(d["license"] for d in docs)
    stats = {
        "n_docs": len(docs),
        "n_chars": sum(d["n_chars"] for d in docs),
        "approx_tokens_word": sum(len(d["text"].split()) for d in docs),
        "by_source": dict(by_source),
        "by_license": dict(by_license),
    }
    (args.root / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"\nWrote {out}")
    if any(k == "verify" for k in by_license):
        print("\n⚠  Some docs carry license='verify' — review provenance "
              "before training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
