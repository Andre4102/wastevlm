"""Given PERFECT localisation, can anything name the material?

Every experiment so far confounds two failures: the model has to find the waste
and then say what it is. AerialWaste's boxes each carry a category id, so ~1350
labelled object crops already exist in the test split and have never been used.
Cropping them hands the model perfect localisation for free and leaves only the
semantic question.

That makes this the ceiling experiment. If material accuracy is poor even on a
ground-truth crop, no detector, no grounding head and no projector change can fix
it, and the honest result is that the material is not recoverable from imagery at
this resolution. If accuracy is good, then localisation is the whole problem and
the grounded architecture is exactly the right response.

Three families of readout, which fail for different reasons:

  rs-clip  -- RemoteCLIP / GeoRSCLIP / SkyCLIP, zero-shot text matching. These
              are 224px models, useless for dense small-object work, but a crop is
              upscaled so resolution stops being the handicap and their remote-
              sensing language alignment is exactly what a material head wants.
  vlm      -- our own decoder, per-category Yes/No margin on the crop
  chance   -- the best constant predictor, which on a skewed label set is a
              stronger baseline than it sounds

Crops get a context margin. A waste object cut to its exact box loses the cues
that identify it (a drum is identified partly by the yard it sits in), and
`--context 0` measures how much of the answer was context in the first place.

    python scripts/roi_material.py --generate --dataset aw_m2 --out roi.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

WEIGHTS = pathlib.Path(os.environ.get(
    "WASTE_VLM_WEIGHTS",
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/weights"))
DATA = pathlib.Path(os.environ.get(
    "WASTE_DATA_ROOT",
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data"))
MCML = {"aw_m2": "mcml_split_dataset_1", "aw_m4": "mcml_split_dataset_2"}

RS_CLIPS = {
    "remoteclip-L14": ("ViT-L-14", WEIGHTS / "encoders/remoteclip-official/RemoteCLIP-ViT-L-14.pt"),
    "georsclip-L14":  ("ViT-L-14", WEIGHTS / "encoders/georsclip/ckpt/RS5M_ViT-L-14.pt"),
}


def load_rois(dataset: str, split: str = "test"):
    """-> [(image_path, box_xywh, category)] one entry per annotated object.

    DroneWaste is the control for the AerialWaste result. Its objects are 7.96
    visual tokens against AerialWaste's 0.90, and its labels come from drawn
    annotations rather than site-level inspection records, so if material naming
    works anywhere it should work there. Failure on BOTH would mean material
    naming from aerial imagery is hard in general; failure on AerialWaste alone
    would localise the problem to that dataset's scale and label semantics.
    """
    if dataset == "dronewaste":
        w = json.loads((DATA / "dronewaste" / "dronewaste_v1.0.json").read_text())
        cat = {c["id"]: c["name"] for c in w["categories"]}
        img = {i["id"]: i["file_name"] for i in w["images"]}
        out = []
        for a in w["annotations"]:
            p = DATA / "dronewaste" / "images" / img[a["image_id"]]
            if p.exists() and a["category_id"] in cat:
                out.append((p, tuple(a["bbox"]), cat[a["category_id"]]))
        return sorted({c for c in cat.values()}), out

    d = json.loads((DATA / "aerialwaste" / MCML[dataset] / f"{split}.json").read_text())
    cat = {c["id"]: c["name"] for c in d["categories"]}
    out = []
    for im in d["images"]:
        p = DATA / "aerialwaste" / "images" / im["file_name"]
        if not p.exists():
            continue
        for a in (im.get("annotations") or []):
            if a["category_id"] in cat:
                out.append((p, tuple(a["bbox"]), cat[a["category_id"]]))
    return sorted({c for c in cat.values()}), out


def crop(img, box, ctx: float, degrade: int = 0):
    x, y, w, h = box
    W, H = img.size
    cx, cy = w * ctx, h * ctx
    c = img.crop((max(0, x - cx), max(0, y - cy),
                  min(W, x + w + cx), min(H, y + h + cy)))
    if degrade:
        # Throw away detail down to `degrade` pixels across, then let the usual
        # preprocess upsample back. This is the only knob that separates "the
        # sensor cannot resolve the material" from "the label does not describe
        # the crop": the crop already fills the frame either way, so anything
        # lost here is pixels, not framing.
        from PIL import Image as _I
        c = c.resize((degrade, degrade), _I.LANCZOS)
    return c


def cue_prompts(dataset: str, cats: list[str]) -> dict[str, list[str]]:
    """Text prompts per category: the plain name plus the hand-written aerial cue."""
    stem = {"dronewaste": "paper10"}.get(dataset, dataset)
    path = pathlib.Path(__file__).resolve().parents[1] / "src" / f"{stem}_descriptions.json"
    d = json.loads(path.read_text()) if path.exists() else {}
    out = {}
    for c in cats:
        p = [f"an aerial photo of {c.lower()}",
             f"a satellite image of {c.lower()}",
             f"{c.lower()} seen from above"]
        cue = (d.get(c) or {}).get("aerial_cue")
        if cue:
            p.append(f"an aerial photo of {cue.split('—')[0].strip()}")
        out[c] = p
    return out


def generate(args) -> None:
    import torch
    from PIL import Image

    cats, rois = load_rois(args.dataset)
    if args.limit:
        rois = rois[: args.limit]
    print(f"[roi] {len(rois)} labelled object crops, {len(cats)} categories")
    print("  " + ", ".join(f"{k}={v}" for k, v in Counter(c for _, _, c in rois).most_common()))

    prompts = cue_prompts(args.dataset, cats)
    recs = [{"image": str(p), "box": list(b), "gt": c} for p, b, c in rois]

    # A full pass is hours of GPU, so let a resubmission pick up whatever the last
    # one banked rather than paying for it twice.
    out = pathlib.Path(args.out)
    if args.resume and out.exists():
        prev = json.loads(out.read_text())
        if prev["cats"] == cats and len(prev["rows"]) == len(recs):
            recs = prev["rows"]
            done = Counter(k for r in recs for k in r if k not in ("image", "box", "gt"))
            print("[resume] " + ", ".join(f"{k} {v}/{len(recs)}" for k, v in sorted(done.items())))
        else:
            print(f"[resume] {out} does not match this run, starting clean")

    def flush(tag):
        out.write_text(json.dumps({"cats": cats, "rows": recs}))
        print(f"[write] {out}  ({tag})", flush=True)

    # --- remote-sensing CLIPs -------------------------------------------------
    import open_clip
    for name, (arch, ckpt) in RS_CLIPS.items():
        if not ckpt.exists():
            print(f"[skip] {name}: {ckpt} missing")
            continue
        want = [f"{name}@{c}" + (f"d{d}" if d else "")
                for c in args.contexts for d in ([0] + list(args.degrade))]
        if all(k in r for k in want for r in recs):
            print(f"[skip] {name}: already scored")
            continue
        print(f"[roi] {name} ({arch})", flush=True)
        model, _, preprocess = open_clip.create_model_and_transforms(arch)
        sd = torch.load(ckpt, map_location="cpu")
        sd = sd.get("state_dict", sd)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing = model.load_state_dict(sd, strict=False)
        print(f"   loaded (missing {len(missing.missing_keys)}, "
              f"unexpected {len(missing.unexpected_keys)})")
        model = model.cuda().eval()
        tok = open_clip.get_tokenizer(arch)
        with torch.no_grad():
            T = []
            for c in cats:
                e = model.encode_text(tok(prompts[c]).cuda())
                T.append(torch.nn.functional.normalize(e, dim=-1).mean(0))
            T = torch.nn.functional.normalize(torch.stack(T), dim=-1)
            for ctx in args.contexts:
              for dg in ([0] + list(args.degrade)):
                key = f"{name}@{ctx}" + (f"d{dg}" if dg else "")
                if all(key in r for r in recs):
                    continue
                for i in range(0, len(rois), 64):
                    chunk = rois[i:i + 64]
                    ims = torch.stack([preprocess(crop(Image.open(p).convert("RGB"), b, ctx, dg))
                                       for p, b, _ in chunk]).cuda()
                    F = torch.nn.functional.normalize(model.encode_image(ims), dim=-1)
                    for j, s in enumerate((F @ T.T).cpu().tolist()):
                        recs[i + j].setdefault(key, s)
                    if i % 1280 == 0:
                        print(f"   {key} {i}/{len(rois)}", flush=True)
        del model
        torch.cuda.empty_cache()

    flush("CLIP stage complete")

    # --- our VLM, per-category Yes/No on the crop -----------------------------
    if args.ckpt:
        from src.vlm_eval import WasteVLMAdapter
        qs = {c: f"Is there {c.lower()} visible in this image? Answer Yes or No."
              for c in cats}
        ad = WasteVLMAdapter(args.ckpt, encoder=args.encoder,
                             image_size=args.image_size, pixel_shuffle=args.pixel_shuffle)
        ad.load()
        print("[roi] VLM yes/no on crops", flush=True)
        for ctx in args.contexts:
            key = f"vlm@{ctx}"
            for n, (p, b, _c) in enumerate(rois):
                if key in recs[n]:
                    continue
                im = crop(Image.open(p).convert("RGB"), b, ctx)
                recs[n][key] = [ad.decision_margin(im, qs[c]) for c in cats]
                if n % 200 == 0:
                    print(f"   ctx={ctx} {n}/{len(rois)}", flush=True)
                if n % 500 == 0 and n:
                    flush(f"{key} {n}/{len(rois)}")
            flush(f"{key} complete")

    flush("done")


def report(args) -> None:
    d = json.loads(pathlib.Path(args.report).read_text())
    cats, rows = d["cats"], d["rows"]
    keys = [k for k in rows[0] if k not in ("image", "box", "gt")]
    prev = Counter(r["gt"] for r in rows)
    n = len(rows)
    print(f"\n=== material naming on {n} ground-truth crops, {len(cats)} classes")
    print("  prevalence: " + ", ".join(f"{c} {prev[c]/n:.1%}" for c in cats))
    base = max(prev.values()) / n
    print(f"  majority-class accuracy (the bar): {base:.3f}\n")
    for k in sorted(keys):
        pred = [cats[max(range(len(cats)), key=lambda i: r[k][i])] for r in rows]
        acc = sum(1 for p, r in zip(pred, rows) if p == r["gt"]) / n
        # macro recall: accuracy is dominated by the majority class otherwise
        rec = []
        for c in cats:
            idx = [i for i, r in enumerate(rows) if r["gt"] == c]
            if idx:
                rec.append(sum(1 for i in idx if pred[i] == c) / len(idx))
        print(f"  {k:22s} acc {acc:.3f} ({acc-base:+.3f} vs majority)  "
              f"macro-recall {sum(rec)/len(rec):.3f}  "
              f"predicts {len(set(pred))}/{len(cats)} classes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--degrade", type=int, nargs="*", default=[],
                    help="also score crops thrown down to N pixels across")
    ap.add_argument("--resume", action="store_true",
                    help="keep whatever --out already holds and only fill the gaps")
    ap.add_argument("--report")
    ap.add_argument("--dataset", default="aw_m2",
                    choices=["aw_m2", "aw_m4", "dronewaste"])
    ap.add_argument("--ckpt")
    ap.add_argument("--encoder", default="cradiov4-so")
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--pixel-shuffle", type=int, default=2)
    ap.add_argument("--contexts", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="roi.json")
    args = ap.parse_args()
    if args.generate:
        generate(args)
    if args.report:
        report(args)


if __name__ == "__main__":
    main()
