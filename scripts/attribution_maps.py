"""Where did the decoder look, and did it look somewhere different per material?

Two questions, one machinery.

The first is whether the evidence for "yes, waste" lands on the annotated waste.
The occlusion experiment already showed it does in aggregate -- editing the waste
moved the margin by -1.782 against +0.086 for a matched control edit -- but that
is one number per image. A map says where, and can be checked per image against
the drawn boxes.

The second is the one the project actually turns on. Category grounding measured
weakly before: categories present versus absent on the same image moved the
margin +1.235 against +0.886, and 72% of that was presence rather than identity.
If that is right, the map for "is there plastic" and the map for "is there
rubble" should be the same map. Correlating them tests it directly.

Every map is scored against two nulls, because a map that ignores the image
entirely can still look convincing:
  uniform  -- scores the box's own area fraction, so lift is what matters
  centre   -- AerialWaste piles are often central, and a centre-weighted map
              that never saw the image would inherit that
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.roi_material import load_rois  # noqa: E402


def centre_prior(g: int) -> np.ndarray:
    y, x = np.mgrid[0:g, 0:g]
    c = (g - 1) / 2
    return np.exp(-(((x - c) ** 2 + (y - c) ** 2) / (2 * (g / 4) ** 2))).astype(np.float32)


def overlay(img, a: np.ndarray, out_path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), dpi=130)
    ax.imshow(img)
    v = np.abs(a).max() or 1.0
    ax.imshow(a, cmap="bwr", vmin=-v, vmax=v, alpha=0.45,
              extent=(0, img.size[0], img.size[1], 0), interpolation="bilinear")
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="aw_m2")
    ap.add_argument("--encoder", default="cradiov4-so")
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--pixel-shuffle", type=int, default=2)
    ap.add_argument("--method", default="ig", choices=["ig", "grad", "occ"])
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--block", type=int, default=2)
    ap.add_argument("--limit", type=int, default=40, help="images, not objects")
    ap.add_argument("--per-category", action="store_true",
                    help="also map each category question and correlate the maps")
    ap.add_argument("--png-dir")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    import torch
    from PIL import Image

    from src import attribution as A
    from src import vlm_calib
    from src.vlm_eval import WasteVLMAdapter

    cats, rois = load_rois(args.dataset, "test")
    by_image = defaultdict(list)
    for p, b, c in rois:
        by_image[str(p)].append((b, c))
    paths = sorted(by_image)[: args.limit]
    print(f"[attr] {len(paths)} annotated images, {len(cats)} categories, method={args.method}")

    ad = WasteVLMAdapter(args.ckpt, encoder=args.encoder,
                         image_size=args.image_size, pixel_shuffle=args.pixel_shuffle)
    ad.load()
    model = ad.model
    yes_ids, no_ids = vlm_calib.decision_token_ids(model.tokenizer)
    g = A.token_grid(model)
    print(f"[attr] token grid {g}x{g}")

    presence_q = "Is there solid waste visible in this image? Answer Yes or No."
    cat_q = {c: f"Is there {c.lower()} visible in this image? Answer Yes or No."
             for c in cats}
    cprior = centre_prior(g)

    if args.png_dir:
        pathlib.Path(args.png_dir).mkdir(parents=True, exist_ok=True)

    rows, cat_corr, presence_corr = [], [], []
    for n, p in enumerate(paths):
        img = Image.open(p).convert("RGB")
        px = ad._transform(img).unsqueeze(0)
        boxes = [b for b, _c in by_image[p]]
        mask = A.box_mask(boxes, img.size, g)

        a, margin = A.attribute(model, px, presence_q, yes_ids, no_ids,
                                method=args.method, steps=args.steps, block=args.block)
        rec = {"image": p, "margin": margin,
               "gt_cats": sorted({c for _b, c in by_image[p]}),
               "presence": A.score_map(a, mask),
               "centre_null": A.score_map(cprior, mask),
               "uniform_null": A.score_map(np.ones((g, g), np.float32), mask)}

        if args.png_dir:
            overlay(img, a, pathlib.Path(args.png_dir) / f"{pathlib.Path(p).stem}_presence.png",
                    f"{pathlib.Path(p).name}  margin={margin:+.2f}  "
                    f"mass-in-box lift={rec['presence']['mass_lift']:+.3f}")

        if args.per_category:
            maps = {}
            for c in cats:
                m, _mg = A.attribute(model, px, cat_q[c], yes_ids, no_ids,
                                     method=args.method, steps=args.steps, block=args.block)
                maps[c] = m
                if args.png_dir:
                    overlay(img, m,
                            pathlib.Path(args.png_dir) / f"{pathlib.Path(p).stem}_{c[:12]}.png",
                            f"{pathlib.Path(p).name}  '{c}'")
            ks = list(cats)
            for i in range(len(ks)):
                presence_corr.append(float(np.corrcoef(a.ravel(), maps[ks[i]].ravel())[0, 1]))
                for j in range(i + 1, len(ks)):
                    cat_corr.append(float(np.corrcoef(maps[ks[i]].ravel(),
                                                      maps[ks[j]].ravel())[0, 1]))
            rec["per_category"] = {c: A.score_map(maps[c], mask) for c in cats}

        rows.append(rec)
        if n % 5 == 0:
            print(f"  {n}/{len(paths)}  margin={margin:+.2f} "
                  f"lift={rec['presence']['mass_lift']:+.3f}", flush=True)

    # ------------------------------------------------------------------ report
    def agg(key, sub):
        v = [r[key][sub] for r in rows if not np.isnan(r[key][sub])]
        return float(np.mean(v)) if v else float("nan")

    print(f"\n=== {args.dataset}, {args.method}, {len(rows)} images")
    print(f"  mean box area fraction (what a uniform map scores): {agg('presence','box_area_fraction'):.3f}")
    for tag, key in [("attribution", "presence"), ("centre prior", "centre_null")]:
        hit = float(np.mean([r[key]["hit"] for r in rows]))
        print(f"  {tag:12s}  mass-in-box {agg(key,'mass_in_box'):.3f}  "
              f"lift {agg(key,'mass_lift'):+.3f}   peak-inside-box {hit:.0%}")

    rep = {"dataset": args.dataset, "method": args.method, "n": len(rows),
           "grid": g, "rows": rows}
    if cat_corr:
        rep["category_map_corr"] = float(np.mean(cat_corr))
        rep["presence_map_corr"] = float(np.mean(presence_corr))
        print(f"\n  mean correlation between two DIFFERENT category maps: "
              f"{np.mean(cat_corr):+.3f}")
        print(f"  mean correlation of a category map with the presence map: "
              f"{np.mean(presence_corr):+.3f}")
        print("  (both near 1.0 => the model looks in the same place whatever "
              "material it is asked about)")

    pathlib.Path(args.out_json).write_text(json.dumps(rep, indent=2))
    print(f"\n[write] {args.out_json}")


if __name__ == "__main__":
    main()
