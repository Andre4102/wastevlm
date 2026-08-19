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

    rng = np.random.default_rng(0)
    stats = {a: {"h25": 0, "h50": 0, "n": 0, "nbox": 0, "floor": 0} for a in args.arms}
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
            if seen is None:
                seen = sorted(r.keys())
                print(f"[sam3] post-process fields: {seen}", flush=True)
            s = stats[arm]
            s["nbox"] += len(boxes)
            for gx, gy, gw, gh in gt:
                box = [gx, gy, gx + gw, gy + gh]
                best = max((iou(b, box) for b in boxes), default=0.0)
                s["h25"] += best >= 0.25
                s["h50"] += best >= 0.50
                s["n"] += 1
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
    for a, s in stats.items():
        r25, r50 = s["h25"] / s["n"], s["h50"] / s["n"]
        print(f"  {a:9s} recall@0.25 {r25:.3f}   recall@0.50 {r50:.3f}   "
              f"(random floor {s['floor']/s['n']:.3f})   {s['nbox']/len(paths):.1f} boxes/img")
        rep[a] = {"recall25": r25, "recall50": r50, "floor": s["floor"] / s["n"],
                  "boxes_per_image": s["nbox"] / len(paths)}
    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(rep, indent=2))
        print(f"\n[write] {args.out_json}")


if __name__ == "__main__":
    main()
