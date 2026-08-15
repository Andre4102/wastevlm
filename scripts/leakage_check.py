"""Perceptual-hash leakage check: any new image corpus vs the held-out benchmarks.

AerialWaste and DroneWaste are held out of every training mix (SFT_DESIGN.md), and
the remote-sensing corpora we train on are largely Google Earth imagery -- as is
391/581 of AerialWaste's test split. A silent tile collision would invalidate every
zero-shot number, so this runs before any new corpus enters a mix.

Method: pHash (DCT, 64-bit) per image, then Hamming distance against every
benchmark hash. pHash is robust to the resize/recompress/format differences that
separate two copies of the same tile, and to mild crops; it is NOT robust to
rotation, which is the right trade-off here (a rotated tile of the same ground is
still a collision we want, but nadir corpora do not usually re-derive tiles that
way -- see --also-rotations to cover it).

    python scripts/leakage_check.py --corpus <dir> --name vrsbench \\
        --out results/vlm_eval/_analysis/leakage_vrsbench.json

Exit code is 1 if any collision at or under --threshold is found, so it can gate a
build script.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

DATA = pathlib.Path("/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data")
# The held-out benchmarks, in full -- both splits of both datasets.
BENCHMARKS = {
    "aerialwaste": DATA / "aerialwaste" / "images",
    "dronewaste": DATA / "dronewaste",
}


def iter_images(root: pathlib.Path):
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in IMG_EXT and p.is_file():
            yield p


def phash_dir(root: pathlib.Path, label: str, rotations: bool = False,
              cache_dir: pathlib.Path | None = None) -> dict:
    """-> {int_hash: [relative paths]}. Collisions within a corpus are fine.

    The benchmark hashes are identical for every corpus we check, so they are
    cached: each additional corpus otherwise re-hashes the same 15k AW+DW images.
    """
    cache = None
    if cache_dir is not None:
        key = f"{label}_{'rot' if rotations else 'norot'}.json"
        cache = cache_dir / key
        if cache.exists():
            raw = json.loads(cache.read_text())
            print(f"  {label}: {len(raw)} hashes from cache", flush=True)
            return {int(k): v for k, v in raw.items()}

    import imagehash
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # AW tiles are large; the bomb guard is noise here
    out: dict[int, list[str]] = collections.defaultdict(list)
    n = 0
    for p in iter_images(root):
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                variants = [im]
                if rotations:
                    variants += [im.rotate(a, expand=True) for a in (90, 180, 270)]
                for v in variants:
                    out[int(str(imagehash.phash(v)), 16)].append(str(p.relative_to(root)))
        except Exception as exc:  # unreadable file: report, do not crash the gate
            print(f"  [warn] {label}: cannot hash {p.name}: {exc}", file=sys.stderr)
            continue
        n += 1
        if n % 2000 == 0:
            print(f"  {label}: hashed {n}", flush=True)
    print(f"  {label}: {n} images, {len(out)} distinct hashes", flush=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({str(k): v for k, v in out.items()}))
    return out


def collisions(corpus: dict, bench: dict, threshold: int) -> list[dict]:
    """Hamming distance <= threshold between any corpus hash and any bench hash.

    Exact (threshold 0) is a dict lookup. Otherwise it is a full cross product --
    20k x 15k = 3e8 pairs here, far too many for a Python loop -- so it runs as a
    chunked numpy XOR with a byte-wise popcount table.
    """
    hits = []
    if threshold == 0:
        for h, paths in corpus.items():
            if h in bench:
                hits.append({"distance": 0, "corpus": paths[:3], "benchmark": bench[h][:3]})
        return hits

    import numpy as np

    c_keys = np.fromiter(corpus.keys(), dtype=np.uint64, count=len(corpus))
    b_keys = np.fromiter(bench.keys(), dtype=np.uint64, count=len(bench))
    c_paths = list(corpus.values())
    b_paths = list(bench.values())
    popcount = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)

    chunk = 512
    for start in range(0, len(c_keys), chunk):
        block = c_keys[start:start + chunk]
        xor = block[:, None] ^ b_keys[None, :]              # [chunk, n_bench] uint64
        dist = popcount[xor.view(np.uint8).reshape(*xor.shape, 8)].sum(axis=2)
        ci, bi = np.nonzero(dist <= threshold)
        for c_i, b_i in zip(ci, bi):
            hits.append({
                "distance": int(dist[c_i, b_i]),
                "corpus": c_paths[start + int(c_i)][:3],
                "benchmark": b_paths[int(b_i)][:3],
            })
    return hits


def verify_pair(p_corpus: pathlib.Path, p_bench: pathlib.Path) -> dict:
    """Adjudicate one pHash collision at the pixel level.

    pHash's known false-positive mode is low-texture images: a uniform field of
    water, forest or farmland carries almost no DCT energy, so unrelated tiles
    land on the same hash. Distinguish that from a genuine duplicate by
    correlating the actual pixels, and report each image's contrast so a
    low-texture false positive is visible as such.
    """
    import numpy as np
    from PIL import Image

    def load(p):
        with Image.open(p) as im:
            return np.asarray(im.convert("L").resize((64, 64), Image.BILINEAR),
                              dtype=np.float64)

    try:
        a, b = load(p_corpus), load(p_bench)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"[:120]}
    sa, sb = float(a.std()), float(b.std())
    if sa < 1e-6 or sb < 1e-6:
        corr = 1.0 if abs(a.mean() - b.mean()) < 1.0 else 0.0
    else:
        corr = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    return {
        "pixel_corr": round(corr, 4),
        "std_corpus": round(sa, 2),
        "std_benchmark": round(sb, 2),
        # a real duplicate correlates near-perfectly; a low-texture false
        # positive shows weak correlation and/or very low contrast on both sides
        "verdict": "DUPLICATE" if corr > 0.90 else "phash_false_positive",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path, required=True,
                    help="root of the new image corpus to check")
    ap.add_argument("--name", required=True, help="corpus name for the report")
    ap.add_argument("--threshold", type=int, default=6,
                    help="max Hamming distance counted as a collision (0 = exact)")
    ap.add_argument("--also-rotations", action="store_true",
                    help="hash 90/180/270 rotations of benchmark images too")
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument("--max-verify", type=int, default=2000,
                    help="cap on pixel-level adjudication of pHash collisions")
    ap.add_argument("--cache-dir", type=pathlib.Path,
                    default=pathlib.Path("/leonardo_scratch/large/userexternal/"
                                         "adiecidu/waste_vlm/results/vlm_eval/"
                                         "_analysis/phash_cache"),
                    help="where benchmark hashes are cached between corpora")
    args = ap.parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"corpus not found: {args.corpus}")

    print(f"[hash] corpus {args.name}: {args.corpus}", flush=True)
    corpus = phash_dir(args.corpus, args.name)

    report = {
        "corpus": args.name,
        "corpus_root": str(args.corpus),
        "threshold": args.threshold,
        "n_corpus_hashes": len(corpus),
        "benchmarks": {},
    }
    total = 0
    for bname, broot in BENCHMARKS.items():
        if not broot.exists():
            print(f"[skip] {bname}: {broot} not found", flush=True)
            continue
        print(f"[hash] benchmark {bname}: {broot}", flush=True)
        bench = phash_dir(broot, bname, rotations=args.also_rotations,
                          cache_dir=args.cache_dir)
        hits = collisions(corpus, bench, args.threshold)
        print(f"  -> {len(hits)} pHash collisions vs {bname}; verifying at pixel level",
              flush=True)

        # adjudicate: a pHash hit is a candidate, not a finding
        n_dup = 0
        for h in hits[:args.max_verify]:
            v = verify_pair(args.corpus / h["corpus"][0], broot / h["benchmark"][0])
            h.update(v)
            if v.get("verdict") == "DUPLICATE":
                n_dup += 1
        dups = [h for h in hits if h.get("verdict") == "DUPLICATE"]
        total += n_dup
        report["benchmarks"][bname] = {
            "n_benchmark_hashes": len(bench),
            "n_phash_collisions": len(hits),
            "n_verified_duplicates": n_dup,
            "n_verified": min(len(hits), args.max_verify),
            "duplicates": dups[:50],
            "examples": hits[:20],
        }
        print(f"     {n_dup} confirmed duplicates, "
              f"{min(len(hits), args.max_verify) - n_dup} pHash false positives",
              flush=True)
        # write after each benchmark: the first run of this was SIGKILLed on a
        # login node midway and lost a completed AerialWaste comparison
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2))

    report["n_duplicates_total"] = total
    report["verdict"] = "CLEAN" if total == 0 else "DUPLICATES FOUND"
    print(f"\n=== {args.name}: {report['verdict']} ({total} verified duplicates "
          f"at Hamming <= {args.threshold})")

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
        print(f"[write] {args.out}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
