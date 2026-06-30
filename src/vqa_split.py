"""Build the VQA split index over the full DW + AW m2 universe.

Enumerates every image in both source datasets (not just the captioned ones)
so the VQA dataset can include the ~1,306 AerialWaste val+test images that
weren't captioned in the earlier teacher-distillation pass — Q1–Q3 answers
come from GT (which exists for all images), and the upcoming scenery
re-caption pass (step 4) will fill in Q4 answers for the gap. This keeps the
VQA train/val/test boundaries aligned with the official MCML split (AW) and
the site-stratified seed=0 split (DW), so VQA results stay comparable with
the seg/detection numbers in `RESULTS_SNAPSHOT.md`.

- DroneWaste paper-10: site-stratified seed=0 via `src.det_dataset.site_stratified_split`.
- AerialWaste m2: dataset-provided splits `mcml_split_dataset_1/{train,val,test}.json`.

Output: `data/vqa/split_index.jsonl` — one JSON per image:
  {"image_path", "dataset", "image_id", "split", "gt_categories", "captioned"}

    python -m src.vqa_split
"""
from __future__ import annotations

import collections
import json
import os
from pathlib import Path

from src.datasets import DRONEWASTE_PAPER_10
from src.det_dataset import site_stratified_split

CAPTIONS = Path("/home/ids/diecidue/data/captions/full.jsonl")
DW_ROOT = Path("/home/ids/diecidue/data/dronewaste")
AW_ROOT = Path("/home/ids/diecidue/data/aerialwaste")
AW_M2_SUB = "mcml_split_dataset_1"
OUT = Path("/home/ids/diecidue/data/vqa/split_index.jsonl")


def build_dw_universe() -> list[dict]:
    """All DroneWaste images with split + paper-10-restricted gt_categories.

    Annotations whose category is outside the paper-10 set (e.g. Wood, Foundry,
    Asphalt milling) are ignored — matching the rest of the pipeline, which
    treats those as out-of-task. An image with ONLY non-paper-10 annotations
    is therefore classified as a paper-10 negative.
    """
    with (DW_ROOT / "dronewaste_v1.0.json").open() as f:
        data = json.load(f)
    cat_by_id = {c["id"]: c["name"] for c in data["categories"]}
    paper10 = set(DRONEWASTE_PAPER_10)
    ann_by_image: dict[int, list[dict]] = collections.defaultdict(list)
    for a in data.get("annotations", []):
        if cat_by_id.get(a["category_id"]) in paper10:
            ann_by_image[int(a["image_id"])].append(a)

    images: list[dict] = []
    all_for_split: list[dict] = []
    for img in data["images"]:
        iid = int(img["id"])
        cats = sorted({cat_by_id[a["category_id"]] for a in ann_by_image[iid]})
        images.append({
            "image_id": str(iid),
            "image_path": str(DW_ROOT / "images" / img["file_name"]),
            "gt_categories": cats,
            "dataset": "dronewaste_paper10",
        })
        all_for_split.append({"id": iid, "site": img["site"]})

    tr, va, te = site_stratified_split(all_for_split, seed=0)
    split_by_idx: dict[int, str] = {}
    for sp, idxs in (("train", tr), ("val", va), ("test", te)):
        for i in idxs:
            split_by_idx[i] = sp
    for i, im in enumerate(images):
        im["split"] = split_by_idx[i]
    return images


def build_aw_universe() -> list[dict]:
    """All AerialWaste m2 images across the official train/val/test MCML splits."""
    images: list[dict] = []
    for split in ("train", "val", "test"):
        with (AW_ROOT / AW_M2_SUB / f"{split}.json").open() as f:
            data = json.load(f)
        cat_by_id = {c["id"]: c["name"] for c in data["categories"]}
        for img in data["images"]:
            cat_ids = img.get("categories") or []
            cats = sorted({cat_by_id[cid] for cid in cat_ids if cid in cat_by_id})
            images.append({
                "image_id": str(img["id"]),
                "image_path": str(AW_ROOT / "images" / img["file_name"]),
                "gt_categories": cats,
                "split": split,
                "dataset": "aerialwaste_m2",
            })
    return images


def main() -> int:
    dw = build_dw_universe()
    aw = build_aw_universe()

    captioned_keys: set[tuple[str, str]] = set()
    for line in CAPTIONS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        captioned_keys.add((r["dataset"], str(r["image_id"])))

    out_rows: list[dict] = []
    missing = collections.Counter()
    for im in dw + aw:
        # Drop rows whose image file isn't on disk. The AW MCML JSONs reference
        # ~554 Agrate-Brianza images that were never delivered with our local
        # data (zero entries in any images*.zip); also catches stragglers.
        if not os.path.exists(im["image_path"]):
            missing[im["dataset"]] += 1
            continue
        out_rows.append({
            "image_path": im["image_path"],
            "dataset": im["dataset"],
            "image_id": im["image_id"],
            "split": im["split"],
            "gt_categories": im["gt_categories"],
            "captioned": (im["dataset"], im["image_id"]) in captioned_keys,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        for o in out_rows:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    counts = collections.Counter((r["dataset"], r["split"]) for r in out_rows)
    pos = collections.Counter(
        (r["dataset"], r["split"]) for r in out_rows if r["gt_categories"]
    )
    cap = collections.Counter(
        (r["dataset"], r["split"]) for r in out_rows if r["captioned"]
    )
    print(f"[universe] {len(out_rows):,} images on disk "
          f"({len(dw):,} DW + {len(aw):,} AW m2 listed; "
          f"{sum(missing.values()):,} missing-on-disk filtered: {dict(missing)})")
    print(f"[captioned] {sum(cap.values()):,} / {len(out_rows):,}")
    print()
    print(f"{'dataset':<22} {'split':<6} {'n':>6}  {'pos':>6}  {'neg':>6}  {'cap':>6}  {'uncap':>6}")
    for (ds, sp), n in sorted(counts.items()):
        p = pos[(ds, sp)]
        c = cap[(ds, sp)]
        print(f"{ds:<22} {sp:<6} {n:>6}  {p:>6}  {n - p:>6}  {c:>6}  {n - c:>6}")
    print(f"[saved] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
