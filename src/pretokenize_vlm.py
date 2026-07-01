"""Offline tokenizer for the Waste-VLM training data.

Renders every record to Qwen ChatML token ids + assistant-only label mask *once*
and writes a compact, memory-mappable cache (see `src/vlm_data.py`). Training then
skips both the 558K-record JSON parse in every rank and per-batch tokenization —
only image decoding remains lazy (decoded images are far too large to cache).

The tokenization is `src.vlm_data.encode_messages`, the exact function the live
collator uses, so the cache is byte-identical to on-the-fly training.

Usage:
    python -m src.pretokenize_vlm \
        --train $WROOT/data/llava_pretrain/blip_laion_cc_sbu_558k.json \
        --image-root $WROOT/data/llava_pretrain/images \
        --out $WROOT/data/llava_pretrain/token_cache \
        --workers 16
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Optional

from src.vlm_data import (
    _load_records,
    _record_to_messages,
    encode_messages,
    save_token_cache,
)
from src.vlm_model import DEFAULT_LLM_PATH, DEFAULT_SYSTEM_PROMPT

# Module globals so forked workers inherit the big record list via copy-on-write
# (passing it through the Pool initializer would pickle a copy per worker).
_RECORDS: list = []
_TOK = None
_SYS = ""
_MAXLEN = 2048
_IMAGE_ROOT: Optional[Path] = None
_CHECK = False


def _worker_init(tok_path: str, system_prompt: str, max_len: int,
                 image_root: Optional[str], check_images: bool) -> None:
    global _TOK, _SYS, _MAXLEN, _IMAGE_ROOT, _CHECK
    from transformers import AutoTokenizer

    _TOK = AutoTokenizer.from_pretrained(tok_path)
    _SYS = system_prompt
    _MAXLEN = max_len
    _IMAGE_ROOT = Path(image_root) if image_root else None
    _CHECK = check_images


def _encode_idx(idx: int):
    """Return (input_ids, labels, image_str) or None if the image is missing."""
    image, messages = _record_to_messages(_RECORDS[idx])
    image = image or ""
    if _CHECK and image:
        p = Path(image)
        if not p.is_absolute() and _IMAGE_ROOT is not None:
            p = _IMAGE_ROOT / p
        if not p.exists():
            return None
    ids, labs = encode_messages(_TOK, _SYS, messages, _MAXLEN)
    return ids, labs, image


def build_cache(args: argparse.Namespace) -> None:
    global _RECORDS
    t0 = time.time()
    _RECORDS = _load_records(args.train)
    n = len(_RECORDS)
    print(f"[pretok] loaded {n} records from {args.train} "
          f"({time.time() - t0:.1f}s)", flush=True)

    init_args = (args.llm_path, args.system_prompt, args.max_len,
                 args.image_root, args.check_images)
    encoded: list = []
    images: list[str] = []
    dropped = 0

    if args.workers > 1:
        ctx = mp.get_context("fork")  # fork → workers inherit _RECORDS via COW
        with ctx.Pool(args.workers, initializer=_worker_init,
                      initargs=init_args) as pool:
            it = pool.imap(_encode_idx, range(n), chunksize=1000)
            for i, res in enumerate(it):
                if res is None:
                    dropped += 1
                else:
                    ids, labs, img = res
                    encoded.append((ids, labs))
                    images.append(img)
                if (i + 1) % 50000 == 0:
                    print(f"[pretok] {i + 1}/{n} "
                          f"({(i + 1) / (time.time() - t0):.0f} rec/s)", flush=True)
    else:
        _worker_init(*init_args)
        for i in range(n):
            res = _encode_idx(i)
            if res is None:
                dropped += 1
            else:
                ids, labs, img = res
                encoded.append((ids, labs))
                images.append(img)
            if (i + 1) % 50000 == 0:
                print(f"[pretok] {i + 1}/{n}", flush=True)

    meta = {
        "max_len": args.max_len,
        "llm_path": os.path.basename(args.llm_path.rstrip("/")),
        "system_prompt": args.system_prompt,
        "image_root": args.image_root,
        "source": os.path.abspath(args.train),
    }
    save_token_cache(args.out, encoded, images, meta)
    total_tok = sum(len(ids) for ids, _ in encoded)
    print(f"[pretok] wrote {len(encoded)} records "
          f"({dropped} dropped for missing images), "
          f"{total_tok} tokens → {args.out} ({time.time() - t0:.1f}s)", flush=True)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pre-tokenize Waste-VLM training data.")
    ap.add_argument("--train", required=True, help="LLaVA train json/jsonl")
    ap.add_argument("--image-root", default=None)
    ap.add_argument("--out", required=True, help="output cache directory")
    ap.add_argument("--llm-path", default=DEFAULT_LLM_PATH,
                    help="tokenizer source (must match training)")
    ap.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check-images", action="store_true",
                    help="stat each image and drop records whose file is missing")
    return ap.parse_args()


if __name__ == "__main__":
    build_cache(parse_args())
