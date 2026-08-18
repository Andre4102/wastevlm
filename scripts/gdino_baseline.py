"""Baseline 1: open-vocabulary grounding, no waste-specific training at all.

Grounding DINO is prompted with waste phrases and its boxes are scored against
the annotations -- AerialWaste's per-image boxes, DroneWaste's COCO boxes. This
is the conventional-perception yardstick the VLM has to justify itself against,
and it is the cleanest test of "does generic grounding transfer to aerial waste",
because the detector has never seen this domain or this vocabulary.

Reported per prompt set:

  detection AP-style recall  -- fraction of GT boxes matched at IoU >= t by any
                               prediction above a score threshold
  precision                  -- fraction of predictions that hit a GT box
  image-level detection      -- does it fire at all on a positive tile, and how
                               often does it fire on a negative one

Recall matters more than precision here and the asymmetry is not a preference.
AerialWaste's boxes come from three different annotation passes, so a prediction
with no matching box may be unannotated waste rather than a false alarm; recall
against annotated boxes is sound, precision is a lower bound. The image-level
negative rate is the honest false-positive signal, since negatives are whole
tiles asserted to contain nothing.

Two prompt registers, because open-vocabulary detectors are sensitive to it and
the difference is itself a result:
  generic  -- "garbage. trash. debris. pile of waste."
  material -- the dataset's own category names

    python scripts/gdino_baseline.py --dataset aw_m2 --out gdino_aw.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

WEIGHTS = pathlib.Path(os.environ.get(
    "WASTE_VLM_WEIGHTS",
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/weights"))
DATA = pathlib.Path(os.environ.get(
    "WASTE_DATA_ROOT",
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data"))
MCML = {"aw_m2": "mcml_split_dataset_1", "aw_m4": "mcml_split_dataset_2"}

PROMPTS = {
    "generic": "garbage. trash. debris. pile of waste. dumped material. rubbish heap.",
    "material": None,   # filled from the dataset's categories
}


def load_gt(dataset: str):
    """-> [(image_path, [xyxy boxes], has_waste)] over the whole test split."""
    if dataset.startswith("aw_"):
        d = json.loads((DATA / "aerialwaste" / MCML[dataset] / "test.json").read_text())
        cats = [c["name"] for c in d["categories"]]
        out = []
        for im in d["images"]:
            p = DATA / "aerialwaste" / "images" / im["file_name"]
            if not p.exists():
                continue
            bx = [(a["bbox"][0], a["bbox"][1],
                   a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3])
                  for a in (im.get("annotations") or [])]
            out.append((p, bx, bool(im.get("categories"))))
        return cats, out

    d = json.loads((DATA / "dronewaste" / "dronewaste_v1.0.json").read_text())
    cats = [c["name"] for c in d["categories"]]
    by = {}
    for a in d["annotations"]:
        by.setdefault(a["image_id"], []).append(a)
    out = []
    for im in d["images"]:
        p = DATA / "dronewaste" / "images" / im["file_name"]
        if not p.exists():
            continue
        anns = by.get(im["id"], [])
        bx = [(a["bbox"][0], a["bbox"][1],
               a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]) for a in anns]
        out.append((p, bx, bool(anns)))
    return cats, out


def iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def generate(args) -> None:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    cats, items = load_gt(args.dataset)
    PROMPTS["material"] = ". ".join(c.lower() for c in cats) + "."
    if args.limit:
        items = items[: args.limit]
    print(f"[gdino] {len(items)} images, {sum(1 for _p,_b,h in items if h)} positive")
    for k, v in PROMPTS.items():
        print(f"  prompt[{k}] = {v}")

    mdl_dir = WEIGHTS / "grounding" / "grounding-dino-base"
    proc = AutoProcessor.from_pretrained(str(mdl_dir))
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        str(mdl_dir), torch_dtype=torch.float32).to("cuda").eval()

    out = []
    for n, (path, gt, has) in enumerate(items):
        img = Image.open(path).convert("RGB")
        rec = {"image": str(path), "gt": gt, "has_waste": has, "preds": {}}
        for pname, ptext in PROMPTS.items():
            inputs = proc(images=img, text=ptext, return_tensors="pt").to("cuda")
            with torch.no_grad():
                o = model(**inputs)
            r = proc.post_process_grounded_object_detection(
                o, inputs.input_ids, threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                target_sizes=[img.size[::-1]])[0]
            rec["preds"][pname] = [
                {"box": [float(v) for v in b], "score": float(s), "label": lab}
                for b, s, lab in zip(r["boxes"].tolist(), r["scores"].tolist(),
                                     r.get("text_labels", r.get("labels", [])))]
        out.append(rec)
        if n % 50 == 0:
            print(f"[gdino] {n}/{len(items)}", flush=True)

    pathlib.Path(args.out).write_text(json.dumps(out))
    print(f"[write] {args.out}")


def report(args) -> None:
    recs = json.loads(pathlib.Path(args.report).read_text())
    pos = [r for r in recs if r["has_waste"]]
    neg = [r for r in recs if not r["has_waste"]]
    print(f"\n=== Grounding DINO zero-shot, {len(recs)} images "
          f"({len(pos)} positive, {len(neg)} negative)")
    for pname in recs[0]["preds"]:
        print(f"\n  --- prompt: {pname}")
        for thr in (0.10, 0.25, 0.35):
            for t_iou in (0.25, 0.50):
                matched = total = npred = hit = 0
                for r in pos:
                    P = [p["box"] for p in r["preds"][pname] if p["score"] >= thr]
                    npred += len(P)
                    total += len(r["gt"])
                    used = set()
                    for g in r["gt"]:
                        for i, b in enumerate(P):
                            if i not in used and iou(g, b) >= t_iou:
                                used.add(i); matched += 1; break
                    hit += len(used)
                rec_ = matched / total if total else 0.0
                prec = hit / npred if npred else 0.0
                if t_iou == 0.25:
                    fire_p = sum(1 for r in pos
                                 if any(p["score"] >= thr for p in r["preds"][pname])) / max(1, len(pos))
                    fire_n = sum(1 for r in neg
                                 if any(p["score"] >= thr for p in r["preds"][pname])) / max(1, len(neg))
                    print(f"    score>={thr:.2f}  fires on {fire_p:.0%} of positives, "
                          f"{fire_n:.0%} of negatives  ({npred} boxes)")
                print(f"      IoU>={t_iou:.2f}  box recall {rec_:.3f}  "
                      f"precision(lower bound) {prec:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--report")
    ap.add_argument("--dataset", default="aw_m2",
                    choices=["aw_m2", "aw_m4", "dronewaste"])
    ap.add_argument("--box-threshold", type=float, default=0.05)
    ap.add_argument("--text-threshold", type=float, default=0.05)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="gdino.json")
    args = ap.parse_args()
    if args.generate:
        generate(args)
    if args.report:
        report(args)


if __name__ == "__main__":
    main()
