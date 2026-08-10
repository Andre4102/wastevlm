"""Normalize Vision-Flan (191-task_1k) -> common schema.

Diversity arm: ~186K instances over 191 tasks. Two things matter here that a naive
first-N pass gets wrong:
  * We sample EVENLY across the 191 tasks (not first-N), so the diversity arm
    actually spans tasks in balance -- the whole point of this arm.
  * We SELECTIVELY extract only the chosen images from the 34 GB zip (random access
    by name), instead of a full 34 GB -> 68 GB extraction for a ~40 GB pilot subset.

The fine `task_name` is kept as `task_type` so build_arms.py's stratified
subsampling preserves the task mix.

    python scripts/convert_vision_flan.py --limit 40000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from align_common import Sample, data_root, emit_records

ZIP_PREFIX = "images_191task_1k/"


def stratified_pick(records: list[dict], limit: int, seed: int) -> list[dict]:
    """Pick ~limit records spread evenly over task_name (seeded)."""
    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_task[r["task_name"]].append(r)
    rng = random.Random(seed)
    per_task = max(1, math.ceil(limit / len(by_task)))
    picked: list[dict] = []
    for task, recs in by_task.items():
        rng.shuffle(recs)
        picked.extend(recs[:per_task])
    rng.shuffle(picked)
    return picked[:limit]


def extract_selected(zip_path: Path, records: list[dict], stage_dir: Path,
                     bad_log: Path) -> dict[str, str]:
    """Extract only the images referenced by `records` from the zip into
    `stage_dir`. Returns {image_field -> local_path} for those found."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    n_miss = 0
    with zipfile.ZipFile(zip_path) as z, open(bad_log, "w") as bad:
        names = set(z.namelist())
        for i, r in enumerate(records):
            img = r.get("image")
            if not img:
                continue
            key = ZIP_PREFIX + img
            if key not in names:
                n_miss += 1
                bad.write(f"NOTINZIP\t{img}\n")
                continue
            dest = stage_dir / img
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(key) as src, open(dest, "wb") as f:
                    f.write(src.read())
            out[img] = str(dest)
            if (i + 1) % 5000 == 0:
                print(f"[vision_flan] extracted {len(out)} (miss={n_miss})", flush=True)
    print(f"[vision_flan] extraction done: {len(out)} images, {n_miss} not in zip",
          flush=True)
    return out


def iter_samples(records: list[dict], img_map: dict[str, str]):
    for r in records:
        img = r.get("image")
        src = img_map.get(img) if img else None
        if img and src is None:
            continue  # image not recoverable from zip
        yield Sample(
            id=f"visionflan_{r['id']}",
            image_src=src,
            image_rel=img or "",
            conversations=r["conversations"],
            task_type=r.get("task_name", "vqa"),  # fine task -> stratification key
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()

    root = data_root()
    raw = root / "raw/vision_flan"
    records = json.load(open(raw / "annotation_191-task_1k.json"))
    print(f"[vision_flan] {len(records)} source records over "
          f"{len({r['task_name'] for r in records})} tasks", flush=True)

    picked = stratified_pick(records, args.limit, args.seed)
    print(f"[vision_flan] stratified-picked {len(picked)} "
          f"({len({r['task_name'] for r in picked})} tasks)", flush=True)

    img_map = extract_selected(
        raw / "image_191-task_1k.zip", picked, raw / "images_sel",
        root / "logs/visionflan_notinzip.txt")

    stats = emit_records(
        source="visionflan",
        samples=iter_samples(picked, img_map),
        out_jsonl=root / "normalized/visionflan.jsonl",
        images_dir=root / "normalized/images",
        bad_log=root / "logs/visionflan_bad_images.txt",
        limit=(args.limit or None),
        verify=not args.no_verify,
    )
    (root / "logs/visionflan_convert.json").write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
