"""Input-side audit for the waste benchmarks: geometry, GSD, object scale.

Answers two questions that decide whether a resolution branch is worth taking:

  1. What does the dataloader actually feed the encoder? Native pixel size and
     ground sample distance per test set, and the resize that is really applied
     (`src/vision_encoder.py` → square bicubic Resize to image_size, no aspect
     preservation, no crop, no tiling).
  2. How big is a waste object once it reaches the token grid? Object areas are
     converted into **token units**: at image_size 768 with patch 16 and
     pixel-shuffle 2 the decoder sees a 24x24 grid, so one token covers a
     32x32 px cell of the *resized* image. An object under ~1 token cannot be
     resolved by more tokens unless the resize itself is changed.

Uses the eval's own sample loader, so the audit describes exactly the images
that were scored, not the whole dataset.

  python scripts/input_geometry.py --out-dir <dir> [--image-size 768] [--pixel-shuffle 2]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vlm_eval import DATASETS, WASTE_DATA_ROOT, _load_classification_samples  # noqa: E402

# WGS84 metres per degree, adequate at these latitudes for a GSD estimate.
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON = 111_320.0


def gsd_from_geometry(geom: dict, width_px: int, height_px: int) -> tuple[float, float] | None:
    """Ground sample distance (m/px) from an image's WGS84 footprint polygon."""
    try:
        ring = geom["coordinates"][0]
    except (KeyError, IndexError, TypeError):
        return None
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    lat_mid = math.radians(sum(lats) / len(lats))
    width_m = (max(lons) - min(lons)) * M_PER_DEG_LON * math.cos(lat_mid)
    height_m = (max(lats) - min(lats)) * M_PER_DEG_LAT
    if width_px <= 0 or height_px <= 0 or width_m <= 0 or height_m <= 0:
        return None
    return width_m / width_px, height_m / height_px


def describe(values: list[float]) -> dict:
    if not values:
        return {}
    a = np.asarray(values, dtype=float)
    return {
        "n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
        "p05": float(np.percentile(a, 5)), "p95": float(np.percentile(a, 95)),
        "min": float(a.min()), "max": float(a.max()),
    }


def fmt(d: dict, unit: str = "") -> str:
    if not d:
        return "n/a"
    return (f"median {d['median']:.3g}{unit}  mean {d['mean']:.3g}{unit}  "
            f"p05-p95 {d['p05']:.3g}-{d['p95']:.3g}{unit}  "
            f"range {d['min']:.3g}-{d['max']:.3g}{unit}")


def aw_image_index(version: str) -> dict[str, dict]:
    """file_name -> raw AerialWaste test record (geometry, annotations, size)."""
    sub = "mcml_split_dataset_1" if version == "m2" else "mcml_split_dataset_2"
    path = WASTE_DATA_ROOT / "aerialwaste" / sub / "test.json"
    data = json.loads(path.read_text())
    return {img["file_name"]: img for img in data["images"]}


def dw_annotation_index() -> tuple[dict[str, list[dict]], dict[int, str]]:
    """file_name -> annotations, and the category id -> name map."""
    data = json.loads((WASTE_DATA_ROOT / "dronewaste" / "dronewaste_v1.0.json").read_text())
    cat_name = {c["id"]: c["name"] for c in data["categories"]}
    by_id = {img["id"]: img["file_name"] for img in data["images"]}
    per_file: dict[str, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        fn = by_id.get(ann["image_id"])
        if fn is not None:
            per_file[fn].append(ann)
    return per_file, cat_name


def polygon_area(seg) -> float:
    """Shoelace area of a COCO polygon (list of flat [x1,y1,x2,y2,...] rings)."""
    if not isinstance(seg, list) or not seg:
        return 0.0
    total = 0.0
    for ring in seg:
        if not isinstance(ring, list) or len(ring) < 6:
            continue
        xs, ys = ring[0::2], ring[1::2]
        s = 0.0
        for i in range(len(xs)):
            j = (i + 1) % len(xs)
            s += xs[i] * ys[j] - xs[j] * ys[i]
        total += abs(s) / 2.0
    return total


def geometry_report(dataset: str, image_size: int) -> dict:
    """Native size / aspect / GSD / applied resize for one test set."""
    _cats, samples = _load_classification_samples(dataset, DATASETS[dataset], 0)
    sizes = Counter((s.width, s.height) for s in samples)
    aspects = [s.width / s.height for s in samples]
    # the transform is Resize((image_size, image_size)): independent x/y scale
    scale_x = [image_size / s.width for s in samples]
    scale_y = [image_size / s.height for s in samples]
    longest = [max(s.width, s.height) for s in samples]

    gsd_by_source: dict[str, list[float]] = defaultdict(list)
    gsd_all: list[float] = []
    if dataset.startswith("aw"):
        idx = aw_image_index(dataset.split("_")[1])
        for s in samples:
            rec = idx.get(s.image_path.name)
            if rec is None:
                continue
            g = gsd_from_geometry(rec.get("geometry") or {}, rec["width"], rec["height"])
            if g:
                gsd_by_source[rec.get("source") or "?"].append((g[0] + g[1]) / 2)
                gsd_all.append((g[0] + g[1]) / 2)

    return {
        "dataset": dataset,
        "n_images": len(samples),
        "sizes_top": sizes.most_common(6),
        "n_distinct_sizes": len(sizes),
        "aspect": describe(aspects),
        "scale_x": describe(scale_x),
        "scale_y": describe(scale_y),
        "longest_edge": describe(longest),
        "n_upsampled": int(sum(1 for l in longest if l < image_size)),
        "n_downsampled": int(sum(1 for l in longest if l > image_size)),
        "gsd_all": describe(gsd_all),
        "gsd_by_source": {k: describe(v) for k, v in sorted(gsd_by_source.items())},
    }


def object_areas(dataset: str, image_size: int, pixel_shuffle: int,
                 patch: int = 16) -> dict:
    """Per-object area in native px2 and in token units on the resized grid."""
    px_per_token = patch * pixel_shuffle          # 32 px at 768/ps2
    grid = image_size // px_per_token             # 24
    _cats, samples = _load_classification_samples(dataset, DATASETS[dataset], 0)
    cats_kept = set(DATASETS[dataset]["cats"])

    rows: list[dict] = []
    src = ""
    if dataset == "dw_paper10":
        src = "polygon segmentation (all 5135 annotations carry masks)"
        per_file, cat_name = dw_annotation_index()
        for s in samples:
            for ann in per_file.get(s.image_path.name, []):
                name = cat_name.get(ann["category_id"], "?")
                if name not in cats_kept:
                    continue
                area = float(ann.get("area") or 0.0) or polygon_area(ann.get("segmentation"))
                bx = ann.get("bbox") or [0, 0, 0, 0]
                rows.append({"file": s.image_path.name, "cat": name, "area_px": area,
                             "bbox_px": float(bx[2]) * float(bx[3]),
                             "w": s.width, "h": s.height})
    else:
        src = ("bounding boxes (only a handful of test images carry masks — see "
               "mask_vs_bbox for the shrinkage measured on those)")
        idx = aw_image_index(dataset.split("_")[1])
        cat_by_id = {c["id"]: c["name"] for c in
                     json.loads((WASTE_DATA_ROOT / "aerialwaste" /
                                 ("mcml_split_dataset_1" if dataset == "aw_m2"
                                  else "mcml_split_dataset_2") / "test.json").read_text())["categories"]}
        for s in samples:
            rec = idx.get(s.image_path.name)
            for ann in (rec or {}).get("annotations") or []:
                name = cat_by_id.get(ann.get("category_id"), "?")
                bx = ann.get("bbox") or [0, 0, 0, 0]
                bbox_area = float(bx[2]) * float(bx[3])
                seg_area = polygon_area(ann.get("segmentation")) if ann.get("segmentation") else 0.0
                rows.append({"file": s.image_path.name, "cat": name,
                             "area_px": seg_area or bbox_area,
                             "bbox_px": bbox_area, "seg_px": seg_area,
                             "w": rec["width"], "h": rec["height"]})

    for r in rows:
        # square resize: area scales by (image_size^2)/(w*h)
        r["area_resized_px"] = r["area_px"] * (image_size ** 2) / (r["w"] * r["h"])
        r["area_tokens"] = r["area_resized_px"] / (px_per_token ** 2)

    tokens = [r["area_tokens"] for r in rows]
    mask_vs_bbox = [r["seg_px"] / r["bbox_px"] for r in rows
                    if r.get("seg_px") and r.get("bbox_px")]
    per_cat = {}
    for cat in sorted({r["cat"] for r in rows}):
        v = [r["area_tokens"] for r in rows if r["cat"] == cat]
        per_cat[cat] = describe(v)

    return {
        "dataset": dataset, "source": src, "n_objects": len(rows),
        "n_annotated_images": len({r["file"] for r in rows}),
        "px_per_token": px_per_token, "token_grid": f"{grid}x{grid}",
        "area_tokens": describe(tokens),
        "frac_below_1_token": float(np.mean([t < 1 for t in tokens])) if tokens else None,
        "frac_below_4_tokens": float(np.mean([t < 4 for t in tokens])) if tokens else None,
        "frac_below_quarter_token": float(np.mean([t < 0.25 for t in tokens])) if tokens else None,
        "mask_vs_bbox_ratio": describe(mask_vs_bbox),
        "per_class_tokens": per_cat,
        "_rows": rows,
    }


def resolution_headroom(geo: dict, area: dict, image_size: int, patch: int,
                        pixel_shuffle: int) -> dict:
    """Is there anything left to resolve, or would scaling just interpolate?

    Two distinct levers get conflated as "more resolution":
      * feeding more pixels — bounded by the native image, beyond which the
        resize is pure interpolation and adds no information;
      * a finer token grid over the same pixels (lower pixel-shuffle), which
        costs tokens but invents nothing.
    """
    native = geo["longest_edge"]["median"]
    med_tok = area["area_tokens"]["median"]
    tokens_now = (image_size // (patch * pixel_shuffle)) ** 2
    # largest input size that still consumes real pixels
    honest_size = int(native // (patch * pixel_shuffle) * (patch * pixel_shuffle))
    return {
        "native_longest_edge_median": native,
        "current_image_size": image_size,
        "current_scale": image_size / native,
        "is_upsampling_today": image_size > native,
        "median_object_tokens_now": med_tok,
        "visual_tokens_now": tokens_now,
        # cap on genuine (non-interpolated) gain, area terms
        "max_real_pixel_gain_area": (native / image_size) ** 2 if native > image_size else 1.0,
        "median_tokens_at_native_size": med_tok * ((native / image_size) ** 2)
                                        if native > image_size else med_tok,
        # same pixels, finer grid
        "median_tokens_at_pshuf1": med_tok * (pixel_shuffle ** 2),
        "visual_tokens_at_pshuf1": (image_size // patch) ** 2,
        "honest_max_size": honest_size,
    }


def plot_histogram(area_reports: list[dict], out_png: Path, image_size: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bins = np.logspace(-3, 3, 55)
    colors = {"dw_paper10": "#2563eb", "aw_m2": "#f97316", "aw_m4": "#16a34a"}
    for rep in area_reports:
        vals = [r["area_tokens"] for r in rep["_rows"] if r["area_tokens"] > 0]
        if not vals:
            continue
        ax.hist(vals, bins=bins, histtype="step", linewidth=1.9, density=True,
                color=colors.get(rep["dataset"], "#666"),
                label=f"{rep['dataset']}  (n={len(vals)}, "
                      f"median {np.median(vals):.2f} tok)")
    ax.axvline(1.0, color="#b91c1c", linestyle="--", linewidth=1.3)
    ax.text(1.05, ax.get_ylim()[1] * 0.92, "1 token", color="#b91c1c", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel(f"object area in token units  "
                  f"(1 token = 32x32 px of the {image_size}x{image_size} resized image)")
    ax.set_ylabel("density")
    ax.set_title("Waste object scale as the decoder sees it")
    ax.legend(fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=170)
    print(f"[saved] {out_png}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--image-size", type=int, default=768)
    p.add_argument("--pixel-shuffle", type=int, default=2)
    p.add_argument("--patch", type=int, default=16)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    datasets = ["dw_paper10", "aw_m2", "aw_m4"]

    print("=" * 96)
    print(f"GEOMETRY — what the loader feeds the encoder "
          f"(Resize(({args.image_size},{args.image_size})), bicubic, no aspect preservation)")
    print("=" * 96)
    geo = []
    for ds in datasets:
        g = geometry_report(ds, args.image_size)
        geo.append(g)
        print(f"\n{ds}  ({g['n_images']} test images, {g['n_distinct_sizes']} distinct sizes)")
        print(f"  sizes      : {g['sizes_top']}")
        print(f"  aspect w/h : {fmt(g['aspect'])}")
        print(f"  longest edge: {fmt(g['longest_edge'], ' px')}")
        print(f"  resize x   : {fmt(g['scale_x'], 'x')}")
        print(f"  resize y   : {fmt(g['scale_y'], 'x')}")
        print(f"  upsampled  : {g['n_upsampled']}/{g['n_images']}   "
              f"downsampled: {g['n_downsampled']}/{g['n_images']}")
        if g["gsd_all"]:
            print(f"  GSD        : {fmt(g['gsd_all'], ' m/px')}")
            for src, d in g["gsd_by_source"].items():
                print(f"    {src:<6}: {fmt(d, ' m/px')}")
        else:
            print("  GSD        : not derivable — no geographic footprint in the metadata")

    print("\n" + "=" * 96)
    print(f"OBJECT SCALE — area in token units "
          f"(1 token = {args.patch * args.pixel_shuffle}x{args.patch * args.pixel_shuffle} px "
          f"of the resized image)")
    print("=" * 96)
    areas = []
    for ds in datasets:
        a = object_areas(ds, args.image_size, args.pixel_shuffle, args.patch)
        areas.append(a)
        print(f"\n{ds}  ({a['n_objects']} objects on {a['n_annotated_images']} images)")
        print(f"  area source: {a['source']}")
        print(f"  area       : {fmt(a['area_tokens'], ' tok')}")
        print(f"  below 1 token   : {a['frac_below_1_token']:.1%}")
        print(f"  below 4 tokens  : {a['frac_below_4_tokens']:.1%}")
        print(f"  below 0.25 token: {a['frac_below_quarter_token']:.1%}")
        if a["mask_vs_bbox_ratio"]:
            print(f"  mask/bbox area ratio (where masks exist): {fmt(a['mask_vs_bbox_ratio'])}")
        print("  per class (median tok):  " + ", ".join(
            f"{c}={d['median']:.2f}" for c, d in a["per_class_tokens"].items() if d))

    print("\n" + "=" * 96)
    print("RESOLUTION HEADROOM — real pixels vs interpolation vs a finer grid")
    print("=" * 96)
    heads = []
    for g, a in zip(geo, areas):
        h = resolution_headroom(g, a, args.image_size, args.patch, args.pixel_shuffle)
        h["dataset"] = g["dataset"]
        heads.append(h)
        verdict = ("input is already UPSAMPLED — more input size is pure interpolation"
                   if h["is_upsampling_today"] else
                   f"real pixels remain: up to {h['honest_max_size']}px "
                   f"(x{h['max_real_pixel_gain_area']:.2f} object area)")
        print(f"\n{g['dataset']}")
        print(f"  native longest edge (median): {h['native_longest_edge_median']:.0f} px  "
              f"-> fed at {args.image_size} px  (x{h['current_scale']:.2f})")
        print(f"  {verdict}")
        print(f"  median object now: {h['median_object_tokens_now']:.2f} tok "
              f"({h['visual_tokens_now']} visual tokens)")
        print(f"  at native size, same ps: {h['median_tokens_at_native_size']:.2f} tok")
        print(f"  same pixels, pixel-shuffle 1: {h['median_tokens_at_pshuf1']:.2f} tok "
              f"({h['visual_tokens_at_pshuf1']} visual tokens)")

    plot_histogram(areas, args.out_dir / "object_area_tokens.png", args.image_size)

    payload = {
        "image_size": args.image_size, "pixel_shuffle": args.pixel_shuffle,
        "patch": args.patch, "px_per_token": args.patch * args.pixel_shuffle,
        "geometry": geo,
        "resolution_headroom": heads,
        "object_scale": [{k: v for k, v in a.items() if k != "_rows"} for a in areas],
    }
    out_json = args.out_dir / "input_geometry.json"
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[saved] {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
