"""Normalize ShareGPT4V-PT (share-captioner_coco_lcs_sam_1246k) -> common schema.

Density arm: ~1.25M dense-caption pairs. Images are drawn from COCO / LLaVA-CC-SBU
(LCS) / SAM. On Leonardo, COCO and the LCS-558K images are already on scratch, so
the pilot resolves those two prefixes locally for free and skips SAM (not mirrored
locally). Pass --allow-sam once the SAM archive is fetched to include it.

    python scripts/convert_sharegpt4v.py --limit 40000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from align_common import Sample, data_root, emit_records

# Local roots for each ShareGPT4V image prefix on Leonardo scratch.
DATA = Path("/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data")
COCO_ROOT = DATA / "coco"                       # coco/train2017/xxx.jpg -> here/train2017/..
LLAVA_ROOT = DATA / "llava_pretrain" / "images"  # llava/llava_pretrain/images/AB/xxx.jpg
SAM_ROOT = DATA / "sam"                          # sam/images/sa_x.jpg (usually absent)


def resolve_local(image: str, allow_sam: bool) -> tuple[str | None, str]:
    """Map a ShareGPT4V image field to (abs_local_path_or_None, source_rel_name)."""
    if image.startswith("coco/"):
        return str(COCO_ROOT / image[len("coco/"):]), image
    if image.startswith("llava/llava_pretrain/images/"):
        rel = image[len("llava/llava_pretrain/images/"):]
        return str(LLAVA_ROOT / rel), image
    if image.startswith("sam/"):
        if not allow_sam:
            return None, image
        return str(SAM_ROOT / image[len("sam/"):]), image
    return None, image


def iter_samples(records: list[dict], allow_sam: bool):
    for r in records:
        src, rel = resolve_local(r["image"], allow_sam)
        if src is None:
            continue  # prefix not mirrored locally (pilot skips SAM)
        yield Sample(
            id=f"sharegpt4v_{r['id']}",
            image_src=src,
            image_rel=rel,
            conversations=r["conversations"],
            task_type="dense_caption",
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=None,
                    help="ShareGPT4V captioner json (default: raw/sharegpt4v/...1246k)")
    ap.add_argument("--limit", type=int, default=40000,
                    help="max emitted records (pilot cap; None-like 0 = all)")
    ap.add_argument("--allow-sam", action="store_true")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip PIL verify (trusted local COCO/LCS)")
    args = ap.parse_args()

    root = data_root()
    src_json = args.json or str(
        root / "raw/sharegpt4v/share-captioner_coco_lcs_sam_1246k_1107.json")
    print(f"[sharegpt4v] loading {src_json} ...", flush=True)
    records = json.load(open(src_json))
    print(f"[sharegpt4v] {len(records)} source records", flush=True)

    stats = emit_records(
        source="sharegpt4v",
        samples=iter_samples(records, args.allow_sam),
        out_jsonl=root / "normalized/sharegpt4v.jsonl",
        images_dir=root / "normalized/images",
        bad_log=root / "logs/sharegpt4v_bad_images.txt",
        limit=(args.limit or None),
        verify=not args.no_verify,
    )
    (root / "logs/sharegpt4v_convert.json").write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
