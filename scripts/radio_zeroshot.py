"""Zero-shot naming and dense open-vocabulary segmentation from C-RADIOv4 alone.

C-RADIOv4-SO400M distils siglip2-g-384, and the release ships the heads that put
its output back into SigLIP2's space (see `src/radio_adaptors.py`). With SigLIP2's
text tower on the other side, the encoder we already run for ROI naming and for
the presence gate can also answer text queries -- which is the whole reason
GeoRSCLIP was in the pipeline. This measures whether it actually can.

Three modes, and the distinction between the first two is not cosmetic:

  crop-summary  crop the object, encode it, push the SUMMARY through
                _heads.siglip2-g. SigLIP2 aligns text with its pooled image
                embedding, so this is the route whose alignment is guaranteed by
                construction. Directly comparable to the GeoRSCLIP-on-crops
                numbers (0.384 AerialWaste, 0.314 DroneWaste).

  roi-dense     encode the whole image once, project every patch through
                _feature_projections.siglip2-g, pool the patches inside the box.
                One pass for the whole image instead of one per object, and it
                reuses the ROI machinery that reached 0.665/0.733 supervised.
                CAVEAT: SigLIP2's per-patch tokens are pre-pooling and are not
                guaranteed to be text-aligned the way the pooled embedding is.
                That is exactly why it is measured rather than assumed.

  dense-seg     per-patch text similarity for the object's own class, scored
                against the ground-truth MASK rather than the box, with the same
                uniform and centre nulls the attribution maps are scored against.

Head and tail classes are reported separately throughout. The tail is the point:
a supervised head needs labels per class and cannot touch a class with one
instance, and text needs none.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.roi_material import DATA, crop, cue_prompts, load_rois  # noqa: E402
from src.attribution import score_map  # noqa: E402
from src.vision_encoder import VisionEncoder  # noqa: E402

TAIL_MAX = 100          # classes with fewer instances than this cannot be trained on


def load_masks(dataset: str) -> dict:
    """image path -> [(polygon list, category name)]; DroneWaste only."""
    if dataset != "dronewaste":
        return {}
    w = json.loads((DATA / "dronewaste" / "dronewaste_v1.0.json").read_text())
    cat = {c["id"]: c["name"] for c in w["categories"]}
    img = {i["id"]: i["file_name"] for i in w["images"]}
    out = defaultdict(list)
    for a in w["annotations"]:
        p = DATA / "dronewaste" / "images" / img[a["image_id"]]
        if a.get("segmentation"):
            out[str(p)].append((a["segmentation"], cat[a["category_id"]]))
    return out


def mask_on_grid(seg, size, g: int) -> np.ndarray:
    """Rasterise COCO polygons and max-pool onto the g x g token grid."""
    from PIL import Image, ImageDraw

    W, H = size
    im = Image.new("1", (W, H), 0)
    d = ImageDraw.Draw(im)
    for poly in seg:
        if len(poly) >= 6:
            d.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)], fill=1)
    a = np.array(im, dtype=bool)
    ys = np.array_split(np.arange(H), g)
    xs = np.array_split(np.arange(W), g)
    return np.array([[a[np.ix_(y, x)].any() for x in xs] for y in ys], dtype=bool)


def text_bank(cats, dataset, encode_text, variant="base"):
    """One L2-normalised embedding per class, averaged over its prompt list."""
    import torch

    from src.prompt_sets import build as build_prompts
    prompts = build_prompts(cats, cue_prompts(dataset, cats), variant)
    T = []
    for c in cats:
        e = encode_text(prompts[c])
        T.append(torch.nn.functional.normalize(e.mean(0), dim=-1))
    return torch.stack(T)


def report(name, y_true, y_pred, cats, prev):
    head = [c for c in cats if prev[c] >= TAIL_MAX]
    tail = [c for c in cats if 0 < prev[c] < TAIL_MAX]
    idx = {c: i for i, c in enumerate(cats)}
    out = {}
    for tag, group in [("all", cats), ("head", head), ("tail", tail)]:
        keep = np.array([c in set(group) for c in [cats[i] for i in y_true]])
        if not keep.any():
            continue
        acc = float((y_pred[keep] == y_true[keep]).mean())
        rec = [float((y_pred[y_true == idx[c]] == idx[c]).mean()) for c in group if prev[c]]
        mrec = float(np.mean(rec))
        base = max(prev[c] for c in group) / sum(prev[c] for c in group)
        print(f"  {name:14s} {tag:5s} n={int(keep.sum()):5d} {len(group):2d} cls  "
              f"acc {acc:.3f} (majority {base:.3f})  macro-recall {mrec:.3f} "
              f"= {mrec * len(group):.2f}x chance")
        out[tag] = {"n": int(keep.sum()), "classes": len(group), "acc": acc,
                    "majority": base, "macro_recall": mrec}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dronewaste")
    ap.add_argument("--encoder", default="cradiov4-so")
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--modes", nargs="+",
                    default=["crop-summary", "roi-dense", "dense-seg"])
    ap.add_argument("--ctx", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-set", default="base", choices=["base", "contrastive"])
    ap.add_argument("--dev-sites", action="store_true",
                    help="score the TRAINING sites, for developing prompts without "
                         "touching the evaluation set")
    ap.add_argument("--held-out-sites", action="store_true",
                    help="score only the sites the supervised probe holds out")
    ap.add_argument("--out-json")
    args = ap.parse_args()

    import torch
    from PIL import Image

    from src.radio_adaptors import load_projection, siglip2_text

    cats, rois = load_rois(args.dataset, "test")
    if args.dev_sites and args.dataset == "dronewaste":
        from scripts.roi_token_probe import split_by_site
        rois, _te = split_by_site(rois)
        print(f"[zs] DEVELOPMENT sites only: {len(rois)} objects "
              f"(the held-out sites are untouched)")
    elif args.held_out_sites and args.dataset == "dronewaste":
        # The supervised probe holds out sites; scoring this arm on all 5135
        # objects made the two columns different evaluations, which is not a
        # comparison however carefully the rows are lined up.
        from scripts.roi_token_probe import split_by_site
        _tr, rois = split_by_site(rois)
        print(f"[zs] restricted to the probe's held-out sites: {len(rois)} objects")
    if args.limit:
        rois = rois[: args.limit]
    prev = Counter(c for _p, _b, c in rois)
    idx = {c: i for i, c in enumerate(cats)}
    y_true = np.array([idx[c] for _p, _b, c in rois])
    print(f"[zs] {len(rois)} objects, {len(cats)} classes "
          f"({sum(1 for c in cats if prev[c] >= TAIL_MAX)} head / "
          f"{sum(1 for c in cats if 0 < prev[c] < TAIL_MAX)} tail)")

    enc = VisionEncoder(args.encoder, image_size=args.image_size)
    g = enc.image_size // enc.patch_size
    dev = enc.device
    encode_text = siglip2_text(device=dev)
    T = text_bank(cats, args.dataset, encode_text, args.prompt_set).to(dev)
    print(f"[zs] {args.encoder} @{args.image_size} -> {g}x{g}; text bank {tuple(T.shape)}")

    # Keep per-object predictions, not only the aggregates. A tail of five classes
    # holding 61 objects -- one of them a single instance -- says nothing when
    # pooled, and re-running a GPU job to answer "which class" is a waste when the
    # predictions could simply have been written down.
    rep = {"dataset": args.dataset, "image_size": args.image_size,
           "cats": cats, "y_true": y_true.tolist(), "modes": {}, "pred": {}}
    E = None

    if "crop-summary" in args.modes:
        head = load_projection("siglip2-g", "summary", args.encoder, device=dev)
        hdim = head.fc1.weight.shape[1] if hasattr(head, "fc1") else 1152
        # RADIO returns its CLS tokens concatenated. This build emits 2304 = 2 x
        # 1152, which matches the two teachers whose config sets use_summary
        # (siglip2-g and dino_v3_7b; sam3 does not), so the SigLIP2 token should
        # be the first slice. "Should be" is not a measurement, so both slices are
        # scored and the answer is read off rather than assumed.
        nslice = 1
        S = None
        for n in range(0, len(rois), 16):
            ims = [crop(Image.open(p).convert("RGB"), b, args.ctx)
                   for p, b, _c in rois[n:n + 16]]
            with torch.no_grad():
                cls = enc.encode(ims).cls
                if S is None:
                    E = [[] for _ in range(4)]
                    nslice = max(1, cls.shape[-1] // hdim)
                    S = [[] for _ in range(nslice)]
                    print(f"[zs] CLS {cls.shape[-1]} = {nslice} x {hdim}; "
                          f"scoring every slice", flush=True)
                for k in range(nslice):
                    e = head(cls[:, k * hdim:(k + 1) * hdim].to(
                        next(head.parameters()).dtype))
                    e = torch.nn.functional.normalize(e.float(), dim=-1)
                    if k == 0:
                        E[0].append(e.cpu().numpy())
                    S[k].append((e @ T.T).cpu().numpy())
            if n % 640 == 0:
                print(f"   crop-summary {n}/{len(rois)}", flush=True)
        for k in range(nslice):
            sim = np.concatenate(S[k])
            yp = sim.argmax(1)
            rep["pred"][f"crop-summary[cls{k}]"] = yp.tolist()
            # Keep the whole similarity vector, not just its argmax. Whether the
            # encoder knows it is confused is a property of the runners-up, and
            # recomputing them costs another GPU pass.
            if k == 0:
                rep["sims"] = np.round(sim, 4).tolist()
            rep["modes"][f"crop-summary[cls{k}]"] = report(
                f"crop-sum[cls{k}]", y_true, yp, cats, prev)

    if "roi-head" in args.modes:
        # Pool the RAW patch tokens inside the box, then push the pooled 1152-d
        # vector through the SUMMARY head -- the text-aligned projection -- rather
        # than projecting each patch through the dense head, which targets
        # SigLIP2's patch space and is not text-aligned (that is `roi-dense`, and
        # it lands below chance). If this works it removes one encoder pass per
        # detected object and the whole image is genuinely encoded once.
        #
        # It is not guaranteed: the head was trained on RADIO's summary token, and
        # a mean of patch tokens has different statistics. That is the question.
        head = load_projection("siglip2-g", "summary", args.encoder, device=dev)
        hd = next(head.parameters()).dtype
        by_image = defaultdict(list)
        for i, (p, b, c) in enumerate(rois):
            by_image[str(p)].append((i, b))
        S = np.zeros((len(rois), len(cats)), np.float32)
        EMB = np.zeros((len(rois), T.shape[1]), np.float32)
        for n, p in enumerate(sorted(by_image)):
            img = Image.open(p).convert("RGB")
            W, H = img.size
            with torch.no_grad():
                P = enc.encode([img]).patches[0]          # [N, 1152], raw
                for i, (x, y, w, h) in by_image[p]:
                    x0 = max(0, min(g - 1, int(x / W * g)))
                    x1 = max(x0 + 1, min(g, int(np.ceil((x + w) / W * g))))
                    y0 = max(0, min(g - 1, int(y / H * g)))
                    y1 = max(y0 + 1, min(g, int(np.ceil((y + h) / H * g))))
                    pooled = P.reshape(g, g, -1)[y0:y1, x0:x1].reshape(-1, P.shape[-1]).mean(0)
                    e = head(pooled.unsqueeze(0).to(hd))
                    e = torch.nn.functional.normalize(e.float(), dim=-1)
                    EMB[i] = e[0].cpu().numpy()
                    S[i] = (e @ T.T)[0].cpu().numpy()
            if n % 100 == 0:
                print(f"   roi-head {n}/{len(by_image)}", flush=True)
        rep["pred"]["roi-head"] = S.argmax(1).tolist()
        rep["sims"] = np.round(S, 4).tolist()
        rep["sims_from"] = "roi-head"
        E = [[EMB]]
        rep["modes"]["roi-head"] = report("roi-head", y_true, S.argmax(1), cats, prev)

    if "roi-dense" in args.modes or "dense-seg" in args.modes:
        fp = load_projection("siglip2-g", "features", args.encoder, device=dev)
        by_image = defaultdict(list)
        for i, (p, b, c) in enumerate(rois):
            by_image[str(p)].append((i, b, c))
        masks = load_masks(args.dataset)
        S = np.zeros((len(rois), len(cats)), np.float32)
        seg_rows = []
        for n, p in enumerate(sorted(by_image)):
            img = Image.open(p).convert("RGB")
            W, H = img.size
            with torch.no_grad():
                P = fp(enc.encode([img]).patches.to(next(fp.parameters()).dtype))
                P = torch.nn.functional.normalize(P[0].float(), dim=-1)
                sim = (P @ T.T).reshape(g, g, len(cats)).cpu().numpy()
            for i, (x, y, w, h) in [(i, b) for i, b, _c in by_image[p]]:
                x0 = max(0, min(g - 1, int(x / W * g))); x1 = max(x0 + 1, min(g, int(np.ceil((x + w) / W * g))))
                y0 = max(0, min(g - 1, int(y / H * g))); y1 = max(y0 + 1, min(g, int(np.ceil((y + h) / H * g))))
                S[i] = sim[y0:y1, x0:x1].reshape(-1, len(cats)).mean(0)
            if "dense-seg" in args.modes and p in masks:
                for seg, c in masks[p]:
                    m = mask_on_grid(seg, img.size, g)
                    if m.any():
                        seg_rows.append(score_map(sim[:, :, idx[c]], m))
            if n % 100 == 0:
                print(f"   dense {n}/{len(by_image)}", flush=True)
        if "roi-dense" in args.modes:
            rep["pred"]["roi-dense"] = S.argmax(1).tolist()
            rep["modes"]["roi-dense"] = report("roi-dense", y_true, S.argmax(1), cats, prev)
        if seg_rows:
            mass = float(np.mean([r["mass_in_box"] for r in seg_rows]))
            lift = float(np.mean([r["mass_lift"] for r in seg_rows]))
            area = float(np.mean([r["box_area_fraction"] for r in seg_rows]))
            hit = float(np.mean([r["hit"] for r in seg_rows]))
            print(f"\n  dense-seg: text similarity for the object's own class, "
                  f"scored on {len(seg_rows)} masks")
            print(f"    mean mask area fraction (uniform null): {area:.3f}")
            print(f"    mass-in-mask {mass:.3f}   lift {lift:+.3f}   peak-inside-mask {hit:.0%}")
            rep["modes"]["dense-seg"] = {"n": len(seg_rows), "mask_area": area,
                                         "mass_in_mask": mass, "lift": lift, "peak_in": hit}

    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(rep, indent=2))
        print(f"\n[write] {args.out_json}")
        # Image embeddings beside the summary. Once these exist every further
        # prompt experiment is text-only and costs seconds instead of a GPU pass
        # over five thousand crops.
        if E and E[0]:
            npy = str(pathlib.Path(args.out_json).with_suffix(".emb.npy"))
            np.save(npy, np.concatenate(E[0]))
            print(f"[write] {npy}  (SigLIP2-space image embeddings)")


if __name__ == "__main__":
    main()
