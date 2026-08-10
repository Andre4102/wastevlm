"""Emit a manifest of still-missing PixMo `points` images to fetch out-of-band.

The login node's 15-min network window can't re-pull the link-rotted point images,
so we download them locally and upload a tarball instead. This lists what's actually
missing -- point items whose image is neither already emitted in pixmo.jsonl nor
present-and-valid on disk in raw/pixmo/images_dl -- as `url<TAB>filename` lines,
where `filename` is exactly the sha name convert_pixmo expects (so a plain
`--no-download` re-run folds them in after upload).

Capped at (points_target - have) * margin so we collect enough to hit the quota even
if some re-downloads also rot, without dumping the whole 2.4M-row table.

    python scripts/list_missing_pixmo_points.py [--target 7500] [--margin 1.6]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from align_common import data_root, sanitize, verify_image
from convert_pixmo import _sha_name, points_item


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=7500, help="points quota")
    ap.add_argument("--margin", type=float, default=1.6,
                    help="collect (target-have)*margin urls to survive re-download rot")
    args = ap.parse_args()

    root = data_root()
    dl_dir = root / "raw/pixmo/images_dl"
    out_jsonl = root / "normalized/pixmo.jsonl"
    manifest = root / "logs/pixmo_points_missing.tsv"

    done: set[str] = set()
    have_points = 0
    if out_jsonl.exists():
        for line in open(out_jsonl):
            if not line.strip():
                continue
            rec = json.loads(line)
            done.add(Path(rec["image"]).name)          # pixmo_<sha>.ext
            if rec["task_type"] == "point":
                have_points += 1

    need = max(0, args.target - have_points)
    cap = int(need * args.margin)
    print(f"[missing] have {have_points} point recs, target {args.target} "
          f"-> need {need}, collecting up to {cap} candidate urls", flush=True)

    files = sorted((root / "raw/pixmo_points/data").glob("*.parquet"))
    n = 0
    seen_urls: set[str] = set()
    with open(manifest, "w") as out:
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=4096):
                for r in batch.to_pylist():
                    if n >= cap:
                        break
                    item = points_item(r)
                    if item is None:
                        continue
                    url = item[0]
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)
                    fname = _sha_name(url)                # <sha>.ext
                    flat = f"pixmo_{sanitize(fname)}"     # matches pixmo.jsonl image name
                    if flat in done:
                        continue                          # already emitted
                    dest = dl_dir / fname
                    if dest.exists() and dest.stat().st_size > 0 and verify_image(dest):
                        continue                          # already on disk & valid
                    out.write(f"{url}\t{fname}\n")
                    n += 1
                if n >= cap:
                    break
            if n >= cap:
                break
    print(f"[missing] wrote {n} urls -> {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
