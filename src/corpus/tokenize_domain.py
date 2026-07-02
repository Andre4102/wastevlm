"""Pack the merged corpus into a ``waste/`` domain of pruning-style Arrow shards.

Reuses the *exact* ``DomainPacker`` from the pruning repo so the output drops
into the DBL loader (``domain_sampled_dataset.py``) alongside the general
RedPajama/SlimPajama domains. Produces::

    <out_root>/waste/shard_00000.arrow …
    <out_root>/waste/manifest_waste.json    # token count + packing config

Run in an env with ``datasets`` (e.g. ``gausdino``), on a compute or login node
(no internet needed — reads local ``corpus.jsonl``)::

    python -m src.corpus.tokenize_domain \\
        --tokenizer /path/to/Llama-3.1-8B \\
        --pruning-repo /leonardo/home/userexternal/adiecidu/scripts/pruning

The general domains are produced separately by the pruning repo's
``tokenize_slimpajama.py`` into the *same* ``<out_root>``; then the DBL trainer
sees ``waste`` + the general domains as one mix.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .common import CORPUS_ROOT, read_jsonl


def make_tokenizer(path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    # Default to the TRAIN split so held-out eval docs never enter CPT.
    ap.add_argument("--corpus", type=Path, default=CORPUS_ROOT / "corpus_train.jsonl")
    ap.add_argument("--out-root", type=Path, default=CORPUS_ROOT / "shards",
                    help="root that also holds the general-domain shard dirs")
    ap.add_argument("--tokenizer", required=True,
                    help="path/HF-id of the BASE MODEL tokenizer (must match "
                         "the model you will prune — LLaMA family for now)")
    ap.add_argument("--pruning-repo", type=Path,
                    default=Path("/leonardo/home/userexternal/adiecidu/scripts/pruning"),
                    help="path to the pruning repo (for DomainPacker)")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--shard-size-seqs", type=int, default=50_000)
    ap.add_argument("--domain", default="waste")
    args = ap.parse_args()

    # Import the exact packer used to build the general domains.
    sys.path.insert(0, str(args.pruning_repo))
    from tokenize_slimpajama import DomainPacker  # noqa: E402

    if not args.corpus.exists():
        raise SystemExit(
            f"{args.corpus} not found — run `python -m src.corpus.split_corpus` "
            f"first so CPT trains on the train split only (avoids eval leakage)."
        )

    tok = make_tokenizer(args.tokenizer)
    eos = tok.eos_token_id
    out_dir = args.out_root / args.domain
    out_dir.mkdir(parents=True, exist_ok=True)

    packer = DomainPacker(args.domain, args.seq_len, eos, out_dir,
                          args.shard_size_seqs)

    n_docs = 0
    for d in read_jsonl(args.corpus):
        ids = tok(d["text"], add_special_tokens=False)["input_ids"]
        if ids:
            packer.add_doc(ids)
            n_docs += 1
    packer.finalize(keep_tail=True)

    tokens = packer.tokens_written()
    manifest = {
        "domain": args.domain,
        "n_docs": n_docs,
        "tokens": tokens,
        "seq_len": args.seq_len,
        "tokenizer": args.tokenizer,
        "vocab_size": len(tok),
    }
    (out_dir / f"manifest_{args.domain}.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\nPacked {n_docs} docs → {tokens/1e6:.1f}M tokens in {out_dir}")
    print("Next: build the general domains into the SAME --out-root with the "
          "pruning repo's tokenize_slimpajama.py, then point the DBL trainer at it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
