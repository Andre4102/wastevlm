"""Normalize PixMo (cap + points) -> common schema.

Spatial-grounding arm. Two subsets:
  * pixmo-cap    : dense captions            -> task_type "dense_caption"
  * pixmo-points : point grounding / counting -> task_type "point"

PixMo images are NOT bundled; the parquet holds `image_url`s hosted across S3 /
imgur / arbitrary sites, so this script downloads them (expect ~5-15% link rot,
logged to logs/pixmo_bad_images.txt, never silently dropped).

Memory: the login node is shared and often near-full, so this streams the parquet
in batches and keeps only a bounded set of in-flight downloads + counters --
records are verified, linked and written incrementally (peak RSS well under 1 GB,
vs. OOM when the whole multi-million-row table was materialized).

Coordinate convention (decided here, not left to the trainer): PixMo points are
normalized PERCENTAGES in [0,100]. Each point renders as `(x.x%, y.y%)`; a
counting/pointing turn lists them after the count. Keep this in sync with however
grounding is scored downstream.

    python scripts/convert_pixmo.py --limit 15000 --workers 32
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq

from align_common import (count_text_tokens, data_root, link_image, sanitize,
                          verify_image)

MAX_POINTS_RENDERED = 30  # bound the text length for high-count images


def _sha_name(url: str) -> str:
    h = hashlib.sha256(url.encode()).hexdigest()[:20]
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if ext not in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"):
        ext = ".jpg"
    return f"{h}{ext}"


def render_points(label: str, points: list[dict], count: int | None) -> str:
    pts = points[:MAX_POINTS_RENDERED]
    coords = "; ".join(f"({p['x']:.1f}%, {p['y']:.1f}%)" for p in pts)
    n = count if count is not None else len(points)
    if n == 1:
        return f"The {label} is located at approximately {coords}."
    tail = "" if len(points) <= MAX_POINTS_RENDERED else f" (first {MAX_POINTS_RENDERED} shown)"
    return (f"There are {n} instances of {label}, located at approximately "
            f"{coords}{tail}.")


def cap_item(r: dict):
    if not r.get("caption"):
        return None
    return (r["image_url"], "dense_caption", [
        {"from": "human", "value": "<image>\nDescribe this image in detail."},
        {"from": "gpt", "value": r["caption"]},
    ])


def points_item(r: dict):
    if not r.get("points") or not r.get("label"):
        return None
    q = (f"<image>\nPoint to each {r['label']} in the image and give its "
         f"approximate location.")
    a = render_points(r["label"], r["points"], r.get("count"))
    return (r["image_url"], "point", [
        {"from": "human", "value": q},
        {"from": "gpt", "value": a},
    ])


def download(url: str, dest: Path, timeout: float = 10.0) -> bool:
    import requests
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        r = requests.get(url, timeout=timeout, stream=True,
                         headers={"User-Agent": "Mozilla/5.0 (waste-vlm align)"})
        r.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
        tmp.rename(dest)
        return True
    except Exception:
        if dest.exists():
            dest.unlink(missing_ok=True)
        return False


def run_subset(files, build_fn, quota: int, source: str, dl_dir: Path,
               images_dir: Path, out, bad, workers: int, stats: dict,
               done_images: set, already: int, offline: bool = False) -> int:
    """Stream `files`, download images with a bounded in-flight pool, and write
    verified records to `out` until `quota` emitted. Resumable: `done_images`
    (flat filenames already in the output) are skipped, `already` seeds the count
    so a restart tops up to quota instead of restarting. Returns emitted count.

    `offline=True` skips every network fetch and only emits records whose image is
    already on disk in `dl_dir` -- used to fold in images pulled by a separate
    recovery/upload pass without re-hammering (and timing out on) the login node."""
    emitted = already
    seq = already
    inflight: dict = {}

    def harvest(target_inflight: int) -> None:
        nonlocal emitted, seq
        while len(inflight) > target_inflight:
            done, _ = wait(list(inflight), return_when=FIRST_COMPLETED)
            for fut in done:
                url, dest, ttype, convs = inflight.pop(fut)
                ok = fut.result()
                if not ok:
                    stats["dl_fail"] += 1
                    bad.write(f"DLFAIL\t{url}\n")
                    continue
                flat = f"{source}_{sanitize(dest.name)}"
                if flat in done_images:
                    continue                       # already emitted in a prior run
                if not verify_image(dest):
                    stats["corrupt"] += 1
                    bad.write(f"CORRUPT\t{dest}\n")
                    continue
                if emitted >= quota:
                    continue
                image_field = link_image(dest, images_dir, flat)
                done_images.add(flat)
                seq += 1
                rec = {"id": f"{source}_{ttype}_{seq}_{dest.stem}", "image": image_field,
                       "conversations": convs, "source": source,
                       "task_type": ttype, "n_text_tokens": count_text_tokens(convs)}
                out.write(json.dumps(rec) + "\n")
                out.flush()                        # durable per record (resumable)
                emitted += 1
                if emitted % 1000 == 0:
                    print(f"[pixmo:{ttype[:4]}] emitted={emitted}/{quota} "
                          f"dl_fail={stats['dl_fail']}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=4096):
                for r in batch.to_pylist():
                    if emitted >= quota:
                        break
                    item = build_fn(r)
                    if item is None:
                        continue
                    url, ttype, convs = item
                    dest = dl_dir / _sha_name(url)
                    if f"{source}_{sanitize(dest.name)}" in done_images:
                        continue                   # skip work for done images
                    if offline and not (dest.exists() and dest.stat().st_size > 0):
                        continue                   # on-disk images only, no network
                    fut = ex.submit(download, url, dest)
                    inflight[fut] = (url, dest, ttype, convs)
                    if len(inflight) >= workers * 3:
                        harvest(workers)          # keep a small backlog only
                    if emitted >= quota:
                        break
                if emitted >= quota:
                    break
            if emitted >= quota:
                break
        harvest(0)                                 # drain the rest
    return emitted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subsets", default="cap,points")
    ap.add_argument("--limit", type=int, default=15000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--no-download", action="store_true",
                    help="only emit records for images already on disk (no network)")
    args = ap.parse_args()

    root = data_root()
    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    dl_dir = root / "raw/pixmo/images_dl"
    dl_dir.mkdir(parents=True, exist_ok=True)
    images_dir = root / "normalized/images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = root / "normalized/pixmo.jsonl"
    bad_log = root / "logs/pixmo_bad_images.txt"

    quota_each = {"cap": args.limit // 2, "points": args.limit - args.limit // 2}
    stats = {"source": "pixmo", "dl_fail": 0, "corrupt": 0, "emitted": 0}

    # Resume: read whatever a prior (possibly OOM-killed) run already emitted, so
    # restarts top up to quota instead of restarting from zero.
    done_images: set[str] = set()
    have = {"dense_caption": 0, "point": 0}
    if out_jsonl.exists():
        for line in open(out_jsonl):
            if not line.strip():
                continue
            rec = json.loads(line)
            done_images.add(Path(rec["image"]).name)
            have[rec["task_type"]] = have.get(rec["task_type"], 0) + 1
        print(f"[pixmo] resuming: {len(done_images)} records already present "
              f"{have}", flush=True)

    print(f"[pixmo] streaming subsets={subsets} target={args.limit} "
          f"(cap={quota_each['cap']}, points={quota_each['points']})", flush=True)

    total = 0
    with open(out_jsonl, "a") as out, open(bad_log, "a") as bad:
        if "cap" in subsets:
            total += run_subset(
                sorted((root / "raw/pixmo_cap/data").glob("*.parquet")),
                cap_item, quota_each["cap"], "pixmo", dl_dir, images_dir,
                out, bad, args.workers, stats, done_images, have["dense_caption"],
                offline=args.no_download)
        if "points" in subsets:
            total += run_subset(
                sorted((root / "raw/pixmo_points/data").glob("*.parquet")),
                points_item, quota_each["points"], "pixmo", dl_dir, images_dir,
                out, bad, args.workers, stats, done_images, have["point"],
                offline=args.no_download)
    stats["emitted"] = total
    (root / "logs/pixmo_convert.json").write_text(json.dumps(stats, indent=2))
    print(f"[pixmo] DONE {stats}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
