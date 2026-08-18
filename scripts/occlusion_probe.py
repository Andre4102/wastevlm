"""Does the answer depend on the waste, or on everything else in the tile?

The concern that motivates the whole grounded-VLM plan is that a fluent answer is
weak evidence: the model may be reading context ("industrial site, therefore
waste") rather than the object. AerialWaste's per-image boxes make that testable
without any new model -- remove the annotated waste and ask again.

    delta_waste = margin(I) - margin(I \\ R)

A large delta means the answer depended on the region. On its own it means little,
because ANY edit to an image moves a logit. So every image also gets a control
edit: the same number of boxes, the same total area, the same fill, placed
elsewhere in the same tile and chosen not to overlap the waste.

    delta_control = margin(I) - margin(I \\ R_random)

The comparison that carries the claim is delta_waste vs delta_control, paired
per image. If they are the same size, the model is not reading the object.

Fill matters. A black rectangle is itself a salient object and a model may well
answer "yes, dumped material" to it, which would show up as a spuriously LARGE
delta. Each region is instead filled with the median colour of a ring around it,
so the patch blends into local context and the edit removes information rather
than adding a new thing to look at.

Four conditions, matching the crop / mask / shuffle / irrelevant-region tests:

  occ_waste   -- waste regions filled with local surround
  occ_control -- same area filled elsewhere (the control for occ_waste)
  crop_waste  -- only the largest waste box, upscaled to full input
  crop_bg     -- a same-sized background crop, the control for crop_waste

Run with --question presence for the gate, or --question categories to ask the
same thing of each material name.

    python scripts/occlusion_probe.py --generate --ckpt <dir> --encoder cradiov4-so \
        --image-size 768 --pixel-shuffle 2 --out occ.json
    python scripts/occlusion_probe.py --report occ.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

MCML = {"aw_m2": "mcml_split_dataset_1", "aw_m4": "mcml_split_dataset_2"}


def load_boxes_px(dataset: str, split: str = "test") -> dict:
    """{image_id: [(x, y, w, h), ...]} in pixel coords."""
    import os
    root = (pathlib.Path(os.environ.get(
        "WASTE_DATA_ROOT",
        "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data"))
        / "aerialwaste" / MCML[dataset])
    d = json.loads((root / f"{split}.json").read_text())
    return {str(im["id"]): [tuple(a["bbox"]) for a in (im.get("annotations") or [])]
            for im in d["images"] if im.get("annotations")}


def ring_fill(img, box, pad: float = 0.6):
    """Median colour of a ring around `box`, i.e. what the surroundings look like."""
    import numpy as np
    from PIL import Image

    x, y, w, h = box
    W, H = img.size
    px, py = int(w * pad), int(h * pad)
    x0, y0 = max(0, int(x) - px), max(0, int(y) - py)
    x1, y1 = min(W, int(x + w) + px), min(H, int(y + h) + py)
    outer = np.asarray(img.crop((x0, y0, x1, y1)))
    mask = np.ones(outer.shape[:2], bool)
    ix0, iy0 = int(x) - x0, int(y) - y0
    mask[max(0, iy0):iy0 + int(h), max(0, ix0):ix0 + int(w)] = False
    vals = outer[mask]
    return tuple(int(v) for v in np.median(vals, axis=0)) if len(vals) else (128, 128, 128)


def occlude(img, boxes):
    """Fill each box with its own local surround colour."""
    from PIL import ImageDraw
    out = img.copy()
    d = ImageDraw.Draw(out)
    for b in boxes:
        x, y, w, h = b
        d.rectangle([x, y, x + w, y + h], fill=ring_fill(img, b))
    return out


def random_boxes(img, boxes, rng, tries: int = 200):
    """Same count and sizes, placed elsewhere, not overlapping any real box."""
    W, H = img.size
    out = []
    for (_x, _y, w, h) in boxes:
        for _ in range(tries):
            nx = rng.uniform(0, max(1, W - w))
            ny = rng.uniform(0, max(1, H - h))
            if all(nx + w < bx or bx + bw < nx or ny + h < by or by + bh < ny
                   for (bx, by, bw, bh) in boxes):
                out.append((nx, ny, w, h))
                break
        else:
            out.append((rng.uniform(0, max(1, W - w)), rng.uniform(0, max(1, H - h)), w, h))
    return out


def crop_box(img, box, ctx: float = 0.25):
    """Crop `box` with a little context, as a standalone image."""
    x, y, w, h = box
    W, H = img.size
    cx, cy = w * ctx, h * ctx
    return img.crop((max(0, x - cx), max(0, y - cy),
                     min(W, x + w + cx), min(H, y + h + cy)))


def generate(args) -> None:
    from PIL import Image

    from src import vlm_calib
    from src.vlm_eval import WasteVLMAdapter
    from scripts.make_convo import load_samples
    from scripts.name_probe import questions

    boxes_by_id = load_boxes_px(args.dataset)
    _cats, samples = load_samples(args.dataset)
    pos = [s for s in samples
           if s.extra["gt_categories"] and pathlib.Path(s.image_path).exists()
           and s.image_id in boxes_by_id]
    if args.limit:
        pos = pos[: args.limit]
    qs = {"presence": vlm_calib.QUESTION}
    if args.question == "categories":
        qs.update(questions(args.dataset))
    print(f"[occ] {len(pos)} boxed positives x {len(qs)} questions x 5 conditions",
          flush=True)

    adapter = WasteVLMAdapter(args.ckpt, encoder=args.encoder,
                              image_size=args.image_size,
                              pixel_shuffle=args.pixel_shuffle)
    adapter.load()
    rng = random.Random(0)

    out = []
    for n, s in enumerate(pos):
        img = Image.open(s.image_path).convert("RGB")
        bx = boxes_by_id[s.image_id]
        ctrl = random_boxes(img, bx, rng)
        biggest = max(bx, key=lambda b: b[2] * b[3])
        bg = random_boxes(img, [biggest], rng)[0]

        variants = {
            "full": img,
            "occ_waste": occlude(img, bx),
            "occ_control": occlude(img, ctrl),
            "crop_waste": crop_box(img, biggest),
            "crop_bg": crop_box(img, bg),
        }
        rec = {"image_id": s.image_id, "gt": sorted(s.extra["gt_categories"]),
               "n_boxes": len(bx),
               "box_area_frac": sum(b[2] * b[3] for b in bx) / (img.size[0] * img.size[1])}
        for qname, q in qs.items():
            rec[qname] = {k: adapter.decision_margin(v, q) for k, v in variants.items()}
        out.append(rec)
        if n % 20 == 0:
            print(f"[occ] {n}/{len(pos)}", flush=True)

    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[write] {args.out}  ({len(out)} images)")


def wilcoxon(pairs) -> float:
    """Two-sided Wilcoxon signed-rank p, normal approximation."""
    import math
    d = [b - a for a, b in pairs if b != a]
    if len(d) < 6:
        return float("nan")
    order = sorted(range(len(d)), key=lambda i: abs(d[i]))
    ranks = [0.0] * len(d)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    wp = sum(r for r, x in zip(ranks, d) if x > 0)
    n = len(d)
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24) or 1e-9
    return math.erfc(abs((wp - mu) / sd) / math.sqrt(2))


def report(args) -> None:
    recs = json.loads(pathlib.Path(args.report).read_text())
    qnames = [k for k in recs[0] if isinstance(recs[0][k], dict)]
    print(f"\n=== occlusion probe on {len(recs)} boxed positives")
    print(f"  waste covers a mean {sum(r['box_area_frac'] for r in recs)/len(recs):.2%} "
          f"of the tile, median {sorted(r['n_boxes'] for r in recs)[len(recs)//2]} boxes")

    for q in qnames:
        sub = [r for r in recs if q in r]
        if q != "presence":
            sub = [r for r in sub if q in r["gt"]]
            if len(sub) < 10:
                continue
        full = [r[q]["full"] for r in sub]
        dw = [r[q]["full"] - r[q]["occ_waste"] for r in sub]
        dc = [r[q]["full"] - r[q]["occ_control"] for r in sub]
        mean = lambda v: sum(v) / len(v)
        print(f"\n  --- {q}  (n={len(sub)})")
        print(f"    margin on the full image          {mean(full):+7.3f}")
        print(f"    drop when WASTE is removed        {mean(dw):+7.3f}")
        print(f"    drop when a CONTROL area is removed {mean(dc):+7.3f}")
        print(f"    waste-vs-control paired Wilcoxon p = "
              f"{wilcoxon(list(zip(dc, dw))):.3g}")
        bigger = sum(1 for a, b in zip(dc, dw) if b > a)
        print(f"    waste drop exceeds control drop on {bigger}/{len(sub)} "
              f"({bigger/len(sub):.0%})")
        flips = sum(1 for r in sub if r[q]["full"] >= 0 > r[q]["occ_waste"])
        flipc = sum(1 for r in sub if r[q]["full"] >= 0 > r[q]["occ_control"])
        print(f"    sign flips Yes->No: waste {flips}, control {flipc}")
        print(f"    crop of the largest box           {mean([r[q]['crop_waste'] for r in sub]):+7.3f}")
        print(f"    same-sized background crop        {mean([r[q]['crop_bg'] for r in sub]):+7.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--report")
    ap.add_argument("--ckpt")
    ap.add_argument("--encoder", default="cradiov4-so")
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--pixel-shuffle", type=int, default=2)
    ap.add_argument("--dataset", default="aw_m2")
    ap.add_argument("--question", default="categories", choices=["presence", "categories"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="occ.json")
    args = ap.parse_args()
    if args.generate:
        generate(args)
    if args.report:
        report(args)


if __name__ == "__main__":
    main()
