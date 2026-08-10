"""LOCAL (run on your laptop, not the HPC) downloader for PixMo-Cap images.

The login node keeps killing long scrapes, so we download the images on a machine
with a stable connection and ship a tarball up, exactly like we did for points.

It writes files named with the SAME scheme the on-HPC converter expects
(`_sha_name` = sha256(url)[:20] + ext), so the offline fold-in
(`convert_pixmo.py --subsets cap --no-download`) matches every file with zero
network. Only deps: pyarrow + requests (you already have both locally).

Usage (locally):
    # 1. grab the cap parquet metadata (small) if you don't have it:
    #    huggingface-cli download allenai/pixmo-cap --repo-type dataset \
    #        --local-dir pixmo_cap
    # 2. download images:
    python local_download_pixmo_cap.py --parquet-dir pixmo_cap/data \
        --out-dir pixmo_cap_dl --limit 40000 --workers 32
    # 3. tar and upload:
    tar czf pixmo_cap_dl.tar.gz pixmo_cap_dl
    #    -> upload to  raw/pixmo_cap/  on the HPC (like the points tar)

Resumable: rerun to top up; files already present are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import pyarrow.parquet as pq
import requests

_IMG_EXT = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")


def sha_name(url: str) -> str:
    """Identical to convert_pixmo._sha_name so names match on the HPC side."""
    h = hashlib.sha256(url.encode()).hexdigest()[:20]
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if ext not in _IMG_EXT:
        ext = ".jpg"
    return f"{h}{ext}"


def download(url: str, dest: Path, timeout: float = 15.0) -> bool:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", required=True,
                    help="dir holding pixmo-cap *.parquet (has image_url + caption)")
    ap.add_argument("--out-dir", default="pixmo_cap_dl")
    ap.add_argument("--limit", type=int, default=40000,
                    help="max images to fetch (rows without a caption are skipped)")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(args.parquet_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet in {args.parquet_dir}")

    # collect URLs (dedup, only rows with a caption) up to a small overshoot of the
    # limit so link-rot still lets us land ~limit files.
    urls: list[str] = []
    seen: set[str] = set()
    target_pool = int(args.limit * 1.25)
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=4096,
                                                     columns=["image_url", "caption"]):
            for r in batch.to_pylist():
                u = r.get("image_url")
                if not u or not r.get("caption") or u in seen:
                    continue
                seen.add(u)
                urls.append(u)
            if len(urls) >= target_pool:
                break
        if len(urls) >= target_pool:
            break
    print(f"[cap] {len(urls)} candidate urls; fetching up to {args.limit}", flush=True)

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(download, u, out / sha_name(u)): u for u in urls}
        for fut in as_completed(futs):
            if fut.result():
                ok += 1
            else:
                fail += 1
            done = ok + fail
            if done % 1000 == 0:
                print(f"[cap] {done}/{len(urls)} ok={ok} fail={fail}", flush=True)
            if ok >= args.limit:
                break
    print(f"[cap] DONE ok={ok} fail={fail} -> {out}", flush=True)
    print(f"[cap] now:  tar czf {out.name}.tar.gz {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
