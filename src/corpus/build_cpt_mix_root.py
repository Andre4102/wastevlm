"""Assemble the CPT data root that mixes the general RedPajama domains with the
small ``waste`` domain, for ``pruning_llama3_pretrain.py``'s DBL dataloader.

``DomainSampledArrowDataset`` discovers domains as sub-directories of ``--data_root``
and intersects them with the trainer's domain-weight vector. It never copies data,
so we just symlink the six pre-packed RedPajama domain dirs and our ``waste`` shard
dir into one root and drop a ``manifest.json`` next to them::

    <out>/common_crawl -> redpajama .../common_crawl
    ...
    <out>/waste        -> waste_corpus_web/shards/waste
    <out>/manifest.json

Login or compute node, no GPU::

    python -m src.corpus.build_cpt_mix_root
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import CORPUS_ROOT

REDPAJAMA_ROOT = Path("/leonardo_work/IscrC_FICHE/redpajama_llama3_packed")
GENERAL_DOMAINS = ["common_crawl", "c4", "github", "arxiv", "wikipedia", "stackexchange"]
TOKENIZER = ("/leonardo_scratch/large/userexternal/adiecidu/pruning/results/"
             "llama/basemodel/llama-3.1-8b")


def link(target: Path, link_path: Path) -> None:
    if not target.exists():
        raise SystemExit(f"missing source domain dir: {target}")
    if link_path.is_symlink() or link_path.exists():
        link_path.unlink()
    link_path.symlink_to(target, target_is_directory=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redpajama-root", type=Path, default=REDPAJAMA_ROOT)
    ap.add_argument("--waste-domain", type=Path,
                    default=CORPUS_ROOT / "shards" / "waste")
    ap.add_argument("--out", type=Path, default=CORPUS_ROOT.parent / "cpt_mix")
    ap.add_argument("--seq-len", type=int, default=2048)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for d in GENERAL_DOMAINS:
        link(args.redpajama_root / d, args.out / d)
    link(args.waste_domain, args.out / "waste")

    manifest = {
        "seq_len": args.seq_len,
        "tokenizer_path": TOKENIZER,
        "domains": GENERAL_DOMAINS + ["waste"],
        "note": "general domains symlinked from redpajama_llama3_packed; "
                "waste from waste_corpus_web/shards/waste",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"CPT mix root → {args.out}")
    for p in sorted(args.out.iterdir()):
        tgt = f" -> {p.resolve()}" if p.is_symlink() else ""
        print(f"  {p.name}{tgt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
