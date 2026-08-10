"""Repair dangling PixMo image symlinks in normalized/pixmo.jsonl.

Some `images_dl/` targets went missing after conversion (partial downloads across
the retry loop / link rot), leaving dangling symlinks in normalized/images/ that
the dataloader silently blanks. All observed losses are `point` (grounding) records
-- the scientifically important part of the spatial arm -- so we try to re-fetch
them from the source parquets before build_arms.py filters the unrecoverable rest.

The flat name in images/ is `pixmo_<sha>` and the symlink target is
raw/pixmo/images_dl/<sha>, where <sha> == convert_pixmo._sha_name(image_url). We
rebuild that url->sha map by scanning the parquets and re-download only the shas we
are missing, straight into images_dl/ (the existing symlink then resolves).

    python scripts/recover_pixmo_images.py --workers 32
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

from align_common import data_root, verify_image
from convert_pixmo import _sha_name, download


def missing_target_shas(root: Path) -> set[str]:
    """sha filenames (basename of images_dl target) referenced by pixmo.jsonl but
    absent on disk."""
    norm = root / "normalized"
    missing: set[str] = set()
    for line in open(norm / "pixmo.jsonl"):
        if not line.strip():
            continue
        rec = json.loads(line)
        if os.path.exists(norm / rec["image"]):
            continue
        flat = Path(rec["image"]).name           # pixmo_<sha>.<ext>
        assert flat.startswith("pixmo_"), flat
        missing.add(flat[len("pixmo_"):])         # <sha>.<ext>
    return missing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--subsets", default="points,cap",
                    help="parquet subsets to scan for the missing urls")
    args = ap.parse_args()

    root = data_root()
    dl_dir = root / "raw/pixmo/images_dl"
    want = missing_target_shas(root)
    print(f"[recover] {len(want)} missing image targets to re-fetch", flush=True)
    if not want:
        return 0

    # url -> sha for every row across the requested subsets; keep only wanted shas
    sub_dir = {"points": "raw/pixmo_points/data", "cap": "raw/pixmo_cap/data"}
    to_fetch: dict[str, str] = {}                 # sha -> url
    for sub in (s.strip() for s in args.subsets.split(",") if s.strip()):
        for f in sorted((root / sub_dir[sub]).glob("*.parquet")):
            for batch in pq.ParquetFile(f).iter_batches(batch_size=8192,
                                                        columns=["image_url"]):
                for url in batch.column("image_url").to_pylist():
                    if not url:
                        continue
                    sha = _sha_name(url)
                    if sha in want and sha not in to_fetch:
                        to_fetch[sha] = url
            if len(to_fetch) == len(want):
                break
    print(f"[recover] resolved urls for {len(to_fetch)}/{len(want)} shas "
          f"({len(want) - len(to_fetch)} not found in parquets)", flush=True)

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download, url, dl_dir / sha): sha
                for sha, url in to_fetch.items()}
        for fut in as_completed(futs):
            sha = futs[fut]
            dest = dl_dir / sha
            if fut.result() and verify_image(dest):
                ok += 1
            else:
                fail += 1
                if dest.exists() and not verify_image(dest):
                    dest.unlink(missing_ok=True)  # drop corrupt partials
            if (ok + fail) % 200 == 0:
                print(f"[recover] {ok} ok / {fail} fail", flush=True)

    print(f"[recover] DONE recovered={ok} failed={fail} "
          f"unresolved={len(want) - len(to_fetch)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
