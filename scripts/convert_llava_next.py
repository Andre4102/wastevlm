"""Normalize LLaVA-NeXT-Data (779k) -> common alignment schema.

Unlike the other converters, LLaVA-NeXT ships as 250 parquet shards with the
image BYTES embedded per row (col `image` = {bytes, path}); there is no image
tree to symlink. So this writes the bytes straight into
`$DATA_ROOT/normalized/images/llavanext_<shard>_<row>.jpg` (one copy, no extra
staging) and emits records with the SAME schema `emit_records` produces, so the
training loader (`--image-root $DATA_ROOT/normalized`) reads it unchanged.

Resumable per shard: each shard writes `normalized/llava_next_parts/part-<NNN>.jsonl`
and is skipped if that part already exists. `--finalize` concatenates the parts
(shuffled) into `normalized/llava_next.jsonl`.

    DATA_ROOT=.../data/alignment python scripts/convert_llava_next.py --workers 16
    DATA_ROOT=.../data/alignment python scripts/convert_llava_next.py --finalize
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_common import count_text_tokens, data_root  # noqa: E402

IMAGE_PLACEHOLDER = "<image>"
SOURCE = "llavanext"


def parquet_dir(root: Path) -> Path:
    # normalized/ lives under DATA_ROOT (.../data/alignment); the LLaVA-NeXT
    # shards are a sibling of `alignment`: .../data/llava_next/data
    return root.parent / "llava_next" / "data"


def shard_paths(root: Path) -> list[Path]:
    return sorted(parquet_dir(root).glob("train-*-of-*.parquet"))


def _one_placeholder(conversations: list[dict]) -> bool:
    n = sum(t.get("value", "").count(IMAGE_PLACEHOLDER) for t in conversations)
    return n == 1


def process_shard(task: tuple[int, str, str, str]) -> dict:
    """Extract one shard: write images + a part-jsonl. Returns per-shard stats."""
    shard_idx, shard_path, images_dir, parts_dir = task
    part = Path(parts_dir) / f"part-{shard_idx:03d}.jsonl"
    if part.exists():
        return {"shard": shard_idx, "skipped": True}

    images = Path(images_dir)
    tmp = part.with_suffix(".jsonl.tmp")
    emitted = text_only = bad_conv = 0
    pf = pq.ParquetFile(shard_path)
    with open(tmp, "w") as out:
        row_base = 0
        for rg in range(pf.num_row_groups):
            rows = pf.read_row_group(
                rg, columns=["id", "conversations", "data_source", "image"]
            ).to_pylist()
            for j, r in enumerate(rows):
                row = row_base + j
                img = r.get("image") or {}
                data = img.get("bytes")
                if not data:
                    text_only += 1
                    continue
                conv = r.get("conversations") or []
                if not _one_placeholder(conv):
                    bad_conv += 1
                    continue
                name = f"{SOURCE}_{shard_idx:03d}_{row:05d}.jpg"
                (images / name).write_bytes(data)
                rec = {
                    "id": f"{SOURCE}_{shard_idx:03d}_{row:05d}",
                    "image": f"images/{name}",
                    "conversations": conv,
                    "source": SOURCE,
                    "task_type": r.get("data_source") or "llava-next-instruct",
                    "n_text_tokens": count_text_tokens(conv),
                }
                out.write(json.dumps(rec) + "\n")
                emitted += 1
            row_base += len(rows)
    os.replace(tmp, part)
    return {"shard": shard_idx, "emitted": emitted,
            "text_only": text_only, "bad_conv": bad_conv}


def finalize(root: Path, seed: int) -> None:
    parts_dir = root / "normalized" / "llava_next_parts"
    parts = sorted(parts_dir.glob("part-*.jsonl"))
    if not parts:
        raise SystemExit(f"no parts in {parts_dir}; run extraction first")
    lines: list[str] = []
    for p in parts:
        with open(p) as f:
            lines.extend(l for l in f if l.strip())
    random.Random(seed).shuffle(lines)
    out = root / "normalized" / "llava_next.jsonl"
    with open(out, "w") as f:
        f.writelines(lines)
    print(f"[llavanext] finalized {len(lines)} records from {len(parts)} parts "
          f"-> {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--finalize", action="store_true",
                    help="concatenate+shuffle part files into llava_next.jsonl")
    args = ap.parse_args()

    root = data_root()
    if args.finalize:
        finalize(root, args.seed)
        return 0

    shards = shard_paths(root)
    if not shards:
        raise SystemExit(f"no parquet shards under {parquet_dir(root)}")
    images_dir = root / "normalized" / "images"
    parts_dir = root / "normalized" / "llava_next_parts"
    images_dir.mkdir(parents=True, exist_ok=True)
    parts_dir.mkdir(parents=True, exist_ok=True)
    print(f"[llavanext] {len(shards)} shards, workers={args.workers}", flush=True)

    tasks = [(i, str(p), str(images_dir), str(parts_dir))
             for i, p in enumerate(shards)]
    tot = {"emitted": 0, "text_only": 0, "bad_conv": 0, "skipped": 0}
    done = 0
    with Pool(args.workers) as pool:
        for st in pool.imap_unordered(process_shard, tasks):
            done += 1
            if st.get("skipped"):
                tot["skipped"] += 1
            else:
                for k in ("emitted", "text_only", "bad_conv"):
                    tot[k] += st.get(k, 0)
            if done % 10 == 0 or done == len(tasks):
                print(f"[llavanext] shards {done}/{len(tasks)}  {tot}", flush=True)
    print(f"[llavanext] EXTRACT DONE {tot}", flush=True)
    print("[llavanext] now run with --finalize to build llava_next.jsonl", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
