"""Download the candidate SFT corpora for the non-waste arm (SFT_DESIGN.md §3).

Login node only -- compute nodes have no internet. Each dataset is fetched
independently and a failure is recorded rather than raised, so one dead repo does
not lose the whole batch. Writes a manifest with per-dataset status and on-disk
size so the pick-which-ones decision has real numbers behind it.

    python scripts/fetch_candidates.py            # everything not already present
    python scripts/fetch_candidates.py --only lrv_small,xbd_qa
    python scripts/fetch_candidates.py --list

Nothing here enters a training mix until it has passed scripts/leakage_check.py.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import time

DEST = pathlib.Path(
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data/external"
)
MANIFEST = DEST / "candidates_manifest.json"

# key -> (repo_id, gap it fills, note)
CANDIDATES: dict[str, tuple[str, str, str]] = {
    # gap 1: native abstention supervision
    "lrv_small": ("sionic-ai/lrv_instruction", "abstention",
                  "LRV-Instruction mirror, negative instructions at 3 levels"),
    "lrv_full": ("Mayfull/LRV-Instruction", "abstention",
                 "full LRV-Instruction (~400k), 23.9 GB"),
    "nlvr2": ("lmms-lab/NLVR2", "abstention",
              "balanced true/false statements, 50/50 by construction"),
    "vizwiz": ("lmms-lab-encoder/VizWiz-VQA", "abstention",
               "native unanswerable flag; renamed from lmms-lab/VizWiz-VQA"),
    # gap 2: diffuse, region-level targets rather than discrete objects
    "xbd_qa": ("DakshJ27/xbd-damage-qa", "diffuse-target",
               "building damage assessment, pre/post nadir, undamaged negatives"),
    "floodnet_vqa": ("FrancescoCiccone/FloodNet_VQA", "diffuse-target",
                     "drone flood condition + counting; altitude between DW and AW"),
    "loveda": ("chloechia/loveda", "diffuse-target",
               "land-cover segmentation, 7 region classes, urban/rural"),
    "isaid": ("ariG23498/iSAID", "diffuse-target",
              "655k instance masks over DOTA imagery -- irregular shapes"),
    # gap 3: GSD / viewpoint diversity and volume
    "fmow": ("danielz01/fMoW", "gsd-diversity",
             "100k+ satellite, 62 classes, wide GSD range"),
    "adaptllm_rs": ("AdaptLLM/remote-sensing-visual-instructions", "gsd-diversity",
                    "RS captions + synthetic tasks"),
}


def du(path: pathlib.Path) -> int:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def fetch(key: str, repo_id: str, workers: int = 8) -> dict:
    from huggingface_hub import snapshot_download

    out = DEST / key
    t0 = time.time()
    try:
        snapshot_download(
            repo_id=repo_id, repo_type="dataset",
            local_dir=str(out), max_workers=workers,
        )
        size = du(out)
        return {"status": "ok", "path": str(out), "bytes": size,
                "gb": round(size / 1e9, 2), "seconds": round(time.time() - t0)}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"[:300],
                "seconds": round(time.time() - t0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated subset of keys")
    ap.add_argument("--list", action="store_true", help="print the candidate table and exit")
    ap.add_argument("--force", action="store_true", help="re-fetch even if the dir exists")
    # a fast batch at 8 workers trips HF's rate limiter (429), and every dataset
    # queued behind it then fails with a misleading LocalEntryNotFoundError.
    ap.add_argument("--workers", type=int, default=8, help="parallel file downloads")
    ap.add_argument("--pause", type=float, default=0.0,
                    help="seconds to wait between datasets (use on a retry after a 429)")
    args = ap.parse_args()

    if args.list:
        for k, (rid, gap, note) in CANDIDATES.items():
            print(f"  {k:14s} {gap:15s} {rid:46s} {note}")
        return

    keys = args.only.split(",") if args.only else list(CANDIDATES)
    unknown = [k for k in keys if k not in CANDIDATES]
    if unknown:
        raise SystemExit(f"unknown keys: {unknown}; --list to see them")

    # compute nodes are offline; this must run on a login node
    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ["HF_HOME"] = "/leonardo_scratch/large/userexternal/adiecidu/hf_cache"
    DEST.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    for k in keys:
        repo_id, gap, note = CANDIDATES[k]
        target = DEST / k
        if target.exists() and not args.force and manifest.get(k, {}).get("status") == "ok":
            print(f"[skip] {k}: already present", flush=True)
            continue
        print(f"[fetch] {k} <- {repo_id}", flush=True)
        rec = fetch(k, repo_id, workers=args.workers)
        rec.update({"repo_id": repo_id, "gap": gap, "note": note})
        manifest[k] = rec
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        if rec["status"] == "ok":
            print(f"   ok  {rec['gb']} GB in {rec['seconds']}s", flush=True)
        else:
            print(f"   FAILED  {rec['error']}", flush=True)
            shutil.rmtree(target, ignore_errors=True)
        if args.pause:
            time.sleep(args.pause)

    print("\n=== summary")
    ok = [k for k, v in manifest.items() if v.get("status") == "ok"]
    bad = [k for k, v in manifest.items() if v.get("status") != "ok"]
    total = sum(manifest[k].get("bytes", 0) for k in ok)
    for k in ok:
        v = manifest[k]
        print(f"  ok      {k:14s} {v['gb']:7.2f} GB  {v['gap']:15s} {v['repo_id']}")
    for k in bad:
        print(f"  FAILED  {k:14s} {manifest[k].get('error','')[:90]}")
    print(f"  total on disk: {total/1e9:.1f} GB   manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
