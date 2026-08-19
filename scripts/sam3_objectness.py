"""Does C-RADIOv4's SAM3 projection substitute for SAM3's own backbone?

The earlier objectness run thresholded projected features with hand-written
heuristics and got 0.194 recall at IoU 0.5 against Grounding DINO's 0.819. That
tested the heuristics, not the encoder: SAM3 does not threshold its backbone, it
runs an FPN and a DETR head over it. This runs the real thing.

`native` is the control and is not optional. It runs SAM3's own backbone on the
same images with the same prompts, so a weak bridged number can be attributed:
if native is strong and bridged is weak the projection does not substitute, and if
both are weak SAM3 simply does not transfer to aerial waste. Reporting bridged
alone would confound the two.
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

from scripts.gdino_baseline import iou  # noqa: E402
from scripts.roi_material import load_rois  # noqa: E402


def post_process(proc, out, sizes, threshold):
    """The processor's API name has moved between releases; find it once."""
    for name in ("post_process_instance_segmentation",
                 "post_process_grounded_object_detection",
                 "post_process_object_detection"):
        fn = getattr(proc, name, None)
        if fn is None:
            continue
        try:
            return fn(out, threshold=threshold, target_sizes=sizes), name
        except TypeError:
            try:
                return fn(out, target_sizes=sizes), name
            except Exception:
                continue
    raise RuntimeError(f"no usable post-process on {type(proc).__name__}: "
                       f"{[m for m in dir(proc) if 'post' in m]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dronewaste")
    ap.add_argument("--encoder", default="cradiov4-so")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--text", default="garbage or dumped waste")
    ap.add_argument("--arms", nargs="+", default=["native", "bridged"])
    ap.add_argument("--out-json")
    args = ap.parse_args()

    import torch
    from PIL import Image

    from src.sam3_bridge import GRID, RADIO_SIZE, detect, load_sam3, radio_vision_embeds

    _cats, rois = load_rois(args.dataset, "test")
    by_image = defaultdict(list)
    for p, b, _c in rois:
        by_image[str(p)].append(list(b))
    paths = sorted(by_image)[: args.limit]
    print(f"[sam3] {len(paths)} images, {sum(len(by_image[p]) for p in paths)} objects, "
          f"prompt={args.text!r}")

    sam3, proc = load_sam3(device="cuda")
    enc = projection = None
    if "bridged" in args.arms:
        from src.radio_adaptors import load_projection
        from src.vision_encoder import VisionEncoder
        enc = VisionEncoder(args.encoder, image_size=RADIO_SIZE)
        projection = load_projection("sam3", "features", args.encoder, device="cuda")
        print(f"[sam3] C-RADIOv4 @{RADIO_SIZE}px -> {GRID}x{GRID} -> SAM3 neck")

    # Recall alone is not comparable across arms that propose different numbers of
    # boxes. DroneWaste genuinely holds many small objects -- 4.8 per image on
    # average, and Pallets alone averages 10.2 where it appears, up to 35 -- so a
    # high count is not automatically padding. But 94 boxes against 4.8 objects is
    # still twentyfold over-proposal, and at that rate the random-placement floor
    # reaches 0.117. So: precision as well as recall, recall again under a budget
    # matched to the number of objects actually present, and recall split by object
    # size, since small objects are where a proposal stage is expected to fail.
    rng = np.random.default_rng(0)
    BUDGETS = (1, 3, 10)
    SMALL, LARGE = 32.0, 96.0
    stats = {a: {"h25": 0, "h50": 0, "n": 0, "nbox": 0, "floor": 0, "tp": 0,
                 "bud": {k: 0 for k in BUDGETS},
                 "size": {"small": [0, 0], "medium": [0, 0], "large": [0, 0]}}
             for a in args.arms}
    seen = None
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        gt = by_image[path]
        for arm in args.arms:
            ve = None
            if arm == "bridged":
                with torch.no_grad():
                    px = enc.transform(img).unsqueeze(0).to(enc.device)
                    patches = enc.encode_tensor(px).patches
                ve = radio_vision_embeds(sam3, patches, projection)
            out = detect(sam3, proc, [img], args.text, vision_embeds=ve,
                         threshold=args.threshold)
            r = out[0]
            boxes = r["boxes"].tolist() if hasattr(r.get("boxes"), "tolist") else list(r.get("boxes", []))
            scores = r["scores"].tolist() if hasattr(r.get("scores"), "tolist") else list(r.get("scores", []))
            order = sorted(range(len(boxes)), key=lambda i: -(scores[i] if scores else 0.0))
            boxes = [boxes[i] for i in order]
            if seen is None:
                seen = sorted(r.keys())
                print(f"[sam3] post-process fields: {seen}", flush=True)
            s = stats[arm]
            s["nbox"] += len(boxes)
            # precision: a predicted box counts once if it matches any object
            for b in boxes:
                if any(iou(b, [g[0], g[1], g[0] + g[2], g[1] + g[3]]) >= 0.50 for g in gt):
                    s["tp"] += 1
            for k in BUDGETS:
                cap = boxes[: max(1, k * len(gt))]
                for gx, gy, gw, gh in gt:
                    box = [gx, gy, gx + gw, gy + gh]
                    s["bud"][k] += max((iou(b, box) for b in cap), default=0.0) >= 0.50
            for gx, gy, gw, gh in gt:
                box = [gx, gy, gx + gw, gy + gh]
                best = max((iou(b, box) for b in boxes), default=0.0)
                s["h25"] += best >= 0.25
                s["h50"] += best >= 0.50
                s["n"] += 1
                side = (gw * gh) ** 0.5
                bucket = "small" if side < SMALL else ("medium" if side < LARGE else "large")
                s["size"][bucket][0] += best >= 0.50
                s["size"][bucket][1] += 1
                W, H = img.size
                fb = 0.0
                for b in boxes:
                    w, h = b[2] - b[0], b[3] - b[1]
                    x = rng.uniform(0, max(1, W - w)); y = rng.uniform(0, max(1, H - h))
                    fb = max(fb, iou([x, y, x + w, y + h], box))
                s["floor"] += fb >= 0.50
        if i % 20 == 0:
            print(f"  {i}/{len(paths)}", flush=True)

    print(f"\n=== {args.dataset}, SAM3 text-prompted, threshold {args.threshold}")
    print("  reference: Grounding DINO 0.819 @IoU 0.5; feature heuristics 0.194\n")
    rep = {}
    ngt = sum(len(by_image[p]) for p in paths)
    print(f"  ground truth: {ngt} objects over {len(paths)} images "
          f"({ngt/len(paths):.1f} per image)\n")
    for a, s in stats.items():
        r25, r50 = s["h25"] / s["n"], s["h50"] / s["n"]
        prec = s["tp"] / max(1, s["nbox"])
        print(f"  {a}")
        print(f"    recall@0.25 {r25:.3f}   recall@0.50 {r50:.3f}   "
              f"precision@0.50 {prec:.3f}")
        print(f"    {s['nbox']/len(paths):6.1f} boxes/img vs {ngt/len(paths):.1f} objects/img"
              f"   (random-placement floor {s['floor']/s['n']:.3f})")
        print("    recall@0.50 under a matched proposal budget: " +
              "  ".join(f"{k}x objects {s['bud'][k]/s['n']:.3f}" for k in BUDGETS))
        print("    by object size: " + "  ".join(
            f"{b} {v[0]/v[1]:.3f} (n={v[1]})" for b, v in s["size"].items() if v[1]))
        rep[a] = {"recall25": r25, "recall50": r50, "precision50": prec,
                  "floor": s["floor"] / s["n"],
                  "boxes_per_image": s["nbox"] / len(paths),
                  "objects_per_image": ngt / len(paths),
                  "recall_at_budget": {str(k): s["bud"][k] / s["n"] for k in BUDGETS},
                  "recall_by_size": {b: (v[0] / v[1] if v[1] else None)
                                     for b, v in s["size"].items()}}
    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(rep, indent=2))
        print(f"\n[write] {args.out_json}")


if __name__ == "__main__":
    main()
