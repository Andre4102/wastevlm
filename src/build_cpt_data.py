"""Tokenize the curated waste corpus into packed blocks for LoRA continued-pretraining.

The corpus is dominated by our own teacher captions (~67% of chars); for a TEXT
domain-adaptation stage we want the model to learn waste *knowledge* from the
authoritative encyclopedic/regulatory text, not just re-memorise its captions
(which are anyway the targets of the later visual SFT stage). So caption tokens
are capped to `--caption-frac` of the authoritative token budget.

Output: data/waste_corpus/cpt_blocks.jsonl  — one line per block: {"input_ids": [...]}
packed to BLOCK tokens, EOS between documents.

    python -m src.build_cpt_data --block 2048 --caption-frac 0.3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CORPUS = Path("/home/ids/diecidue/data/waste_corpus/corpus.jsonl")
OUT = Path("/home/ids/diecidue/data/waste_corpus/cpt_blocks.jsonl")
QWEN = "/home/ids/diecidue/results/waste_vlm/weights/Qwen2.5-7B-Instruct"


def load_tokenizer(path: str):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    except Exception:
        from transformers import AutoProcessor
        return AutoProcessor.from_pretrained(path, trust_remote_code=True).tokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=2048)
    ap.add_argument("--caption-frac", type=float, default=0.3,
                    help="cap caption tokens to this fraction of authoritative tokens")
    ap.add_argument("--tokenizer", default=QWEN, help="tokenizer/model path")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    eos = tok.eos_token_id
    docs = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    auth = [d for d in docs if d["source"] != "captions"]
    caps = [d for d in docs if d["source"] == "captions"]

    def encode(d):
        return tok(d["text"], add_special_tokens=False)["input_ids"] + [eos]

    auth_ids = [t for d in auth for t in encode(d)]
    cap_ids_all = [t for d in caps for t in encode(d)]
    cap_budget = int(len(auth_ids) * args.caption_frac)
    cap_ids = cap_ids_all[:cap_budget]

    stream = auth_ids + cap_ids
    # pack into fixed-length blocks (drop the short tail)
    blocks = [stream[i:i + args.block] for i in range(0, len(stream), args.block)]
    blocks = [b for b in blocks if len(b) == args.block]

    with OUT.open("w") as f:
        for b in blocks:
            f.write(json.dumps({"input_ids": b}) + "\n")

    print(f"authoritative tokens : {len(auth_ids):,}  (from {len(auth)} docs)")
    print(f"caption tokens (all) : {len(cap_ids_all):,}  -> capped to {len(cap_ids):,} "
          f"({args.caption_frac:.0%} of authoritative)")
    print(f"packed blocks        : {len(blocks):,} x {args.block} tok = {len(blocks)*args.block:,} train tokens")
    print(f"[saved] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
