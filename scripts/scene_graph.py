"""Run the pipeline over whole images and emit the structured scene the harness consumes.

Detection and naming both come off ONE C-RADIOv4 pass. The patch tokens go to SAM3's
neck and DETR head for boxes and masks; the same tokens are ROI-pooled per detection
and pushed through _heads.siglip2-g to be named against text. Nothing is fitted on
either evaluation dataset and no component carries a fixed label vector.

The output matches what `src/scene_reason.py` already solves over ground truth:

    {"image": path, "size": [W, H],
     "objs": [{"category", "box":[x,y,w,h], "area", "cx", "cy", "score", "margin"}]}

`--source gt` writes the same structure straight from the annotations, which is what
makes the harness interpretable: the symbolic solver over GT scenes is the reasoning
ceiling, the same solver over predicted scenes isolates what perception costs, and
the difference between those two is the number no end-to-end evaluation can report.
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

from scripts.roi_material import cue_prompts, load_rois  # noqa: E402


def gt_scenes(dataset, sites=None):
    from PIL import Image

    _cats, rois = load_rois(dataset, "test")
    if sites:
        rois = [r for r in rois if pathlib.Path(r[0]).name.split("_")[0] in sites]
    by = defaultdict(list)
    for p, b, c in rois:
        by[str(p)].append((b, c))
    out = []
    for p, items in sorted(by.items()):
        W, H = Image.open(p).size
        objs = []
        for (x, y, w, h), c in items:
            objs.append({"category": c, "box": [x, y, w, h], "area": float(w * h),
                         "cx": x + w / 2, "cy": y + h / 2, "score": 1.0})
        out.append({"image": p, "size": [W, H], "objs": objs})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dronewaste")
    ap.add_argument("--source", default="pred", choices=["pred", "gt"])
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--encoder", default="cradiov4-so")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--text", default="garbage or dumped waste")
    ap.add_argument("--prompt-set", default="contrastive")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.source == "gt":
        scenes = gt_scenes(args.dataset, set(args.sites) if args.sites else None)
        if args.limit:
            scenes = scenes[: args.limit]
        pathlib.Path(args.out).write_text(json.dumps(scenes))
        n = sum(len(s["objs"]) for s in scenes)
        print(f"[scene] {len(scenes)} images, {n} ground-truth objects -> {args.out}")
        return

    import torch
    from PIL import Image

    from src.prompt_sets import build as build_prompts
    from src.radio_adaptors import load_projection, siglip2_text
    from src.sam3_bridge import GRID, RADIO_SIZE, detect, load_sam3, radio_vision_embeds
    from src.vision_encoder import VisionEncoder

    cats, rois = load_rois(args.dataset, "test")
    imgs = sorted({str(p) for p, _b, _c in rois})
    if args.sites:
        imgs = [p for p in imgs if pathlib.Path(p).name.split("_")[0] in set(args.sites)]
    if args.limit:
        imgs = imgs[: args.limit]
    print(f"[scene] {len(imgs)} images, naming against {len(cats)} class prompts")

    sam3, proc = load_sam3(device="cuda")
    enc = VisionEncoder(args.encoder, image_size=RADIO_SIZE)
    proj_sam = load_projection("sam3", "features", args.encoder, device="cuda")
    head = load_projection("siglip2-g", "summary", args.encoder, device="cuda")
    hd = next(head.parameters()).dtype
    encode_text = siglip2_text(device="cuda")

    prompts = build_prompts(cats, cue_prompts(args.dataset, cats), args.prompt_set)
    T = torch.stack([torch.nn.functional.normalize(
        encode_text(prompts[c]).mean(0), dim=-1) for c in cats]).cuda()

    out = []
    for n, path in enumerate(imgs):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        with torch.no_grad():
            px = enc.transform(img).unsqueeze(0).to(enc.device)
            patches = enc.encode_tensor(px).patches            # ONE pass, both uses
            ve = radio_vision_embeds(sam3, patches, proj_sam)
            det = detect(sam3, proc, [img], args.text, vision_embeds=ve,
                         threshold=args.threshold)[0]
            boxes = det["boxes"].tolist() if hasattr(det.get("boxes"), "tolist") else []
            scores = det["scores"].tolist() if hasattr(det.get("scores"), "tolist") else []
            P = patches[0]
            objs = []
            for b, sc in zip(boxes, scores):
                x0, y0, x1, y1 = b
                gx0 = max(0, min(GRID - 1, int(x0 / W * GRID)))
                gx1 = max(gx0 + 1, min(GRID, int(np.ceil(x1 / W * GRID))))
                gy0 = max(0, min(GRID - 1, int(y0 / H * GRID)))
                gy1 = max(gy0 + 1, min(GRID, int(np.ceil(y1 / H * GRID))))
                pooled = P.reshape(GRID, GRID, -1)[gy0:gy1, gx0:gx1]
                pooled = pooled.reshape(-1, P.shape[-1]).mean(0)
                e = torch.nn.functional.normalize(
                    head(pooled.unsqueeze(0).to(hd)).float(), dim=-1)
                sim = (e @ T.T)[0]
                top = torch.topk(sim, 2)
                objs.append({
                    "category": cats[int(top.indices[0])],
                    "box": [x0, y0, x1 - x0, y1 - y0],
                    "area": float((x1 - x0) * (y1 - y0)),
                    "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2,
                    "score": float(sc),
                    "margin": float(top.values[0] - top.values[1]),
                })
        out.append({"image": path, "size": [W, H], "objs": objs})
        if n % 25 == 0:
            print(f"  {n}/{len(imgs)}  {len(objs)} objects", flush=True)

    pathlib.Path(args.out).write_text(json.dumps(out))
    tot = sum(len(s["objs"]) for s in out)
    print(f"[scene] {len(out)} images, {tot} detections "
          f"({tot/max(1,len(out)):.1f}/img) -> {args.out}")


if __name__ == "__main__":
    main()
