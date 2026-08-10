"""Normalize the locally-downloaded PixMo-Cap images -> common schema.

Unlike convert_pixmo.py (which maps images by url-hash `_sha_name`), the images the
user downloaded locally and uploaded (raw/pixmo_cap_images/NNNNNNN.jpg) are named by
their **0-based global row index** across the 4 sorted cap parquet shards, saved with
a `.jpg` extension regardless of the true image format. Verified: 0000000.jpg == row 0
(birds/1491.png, 1024x685).

So we walk the 4 parquets in sorted order with a running index i, and for every row
that has a caption AND whose image file `NNNNNNN.jpg` is present on disk, emit a
dense-caption record. Link-rot rows (image never downloaded) are simply absent and
skipped. Reuses align_common.emit_records (symlink into normalized/images, PIL-verify,
Qwen token count) -> writes normalized/pixmo_cap.jsonl (separate from pixmo.jsonl).

    python scripts/convert_pixmo_cap_indexed.py [--limit N] [--no-verify]
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pyarrow.parquet as pq

from align_common import Sample, data_root, emit_records

CAP_PROMPT = "Describe this image in detail."


def iter_samples(parquet_dir: Path, images_dir: Path):
    files = sorted(glob.glob(str(parquet_dir / "*.parquet")))
    if not files:
        raise SystemExit(f"no cap parquet in {parquet_dir}")
    i = -1
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=4096,
                                                     columns=["image_url", "caption"]):
            for r in batch.to_pylist():
                i += 1
                cap = r.get("caption")
                if not cap:
                    continue
                img = images_dir / f"{i:07d}.jpg"
                if not img.exists():
                    continue  # link-rot: never downloaded
                yield Sample(
                    id=f"pixmocap_{i:07d}",
                    image_src=str(img),
                    image_rel=f"{i:07d}.jpg",
                    conversations=[
                        {"from": "human", "value": f"<image>\n{CAP_PROMPT}"},
                        {"from": "gpt", "value": cap},
                    ],
                    task_type="dense_caption",
                )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="0 = all present on disk")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip PIL header verify (faster; trust local downloads)")
    args = ap.parse_args()

    root = data_root()
    parquet_dir = root / "raw/pixmo_cap/data"
    images_dir = root / "raw/pixmo_cap_images"
    print(f"[pixmocap] parquet={parquet_dir} images={images_dir}", flush=True)

    stats = emit_records(
        source="pixmocap",
        samples=iter_samples(parquet_dir, images_dir),
        out_jsonl=root / "normalized/pixmo_cap.jsonl",
        images_dir=root / "normalized/images",
        bad_log=root / "logs/pixmo_cap_bad_images.txt",
        limit=(args.limit or None),
        verify=not args.no_verify,
    )
    (root / "logs/pixmo_cap_convert.json").write_text(__import__("json").dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
