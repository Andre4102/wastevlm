"""Generate the VQA dataset (train/val/test JSONLs) from curated inputs.

Per image, up to 4 question types are emitted:
  Q1 presence       — yes/no from gt emptiness. Every image.
  Q2 type-ID open   — list types verbally. POSITIVES only.
  Q2 type-ID choice — multiple-choice over the dataset's verifiable subset.
                      Every image (negatives answer "None of the above").
  Q3 grounding      — bbox of the waste region (union of paper-10 annotations,
                      normalised to [0,1]). DW POSITIVES only (AW has no
                      per-instance bboxes in our local data).
  Q4 scenery        — scenery description from `data/captions/scenery.jsonl`.
                      Image must be present there; otherwise Q4 is skipped
                      silently (re-run after the scenery pass completes).

Hallucination defense (per project.md):
  Answers to Q1–Q3 are derived 100% from GT. Captions are NOT consulted.
  Q4 answers come from the constrained no-waste scenery pass.

Output schema (LLaVA-style, one JSON object per QA):
  {"id": "<dataset>__<image_id>__<q_type>",
   "image": "<absolute path>",
   "split": "train|val|test",
   "dataset": "...", "image_id": "...", "q_type": "...",
   "conversations": [
     {"from": "human", "value": "<image>\\n{question}"},
     {"from": "gpt", "value": "{answer}"}
   ]}

    python -m src.vqa_gen --seed 0
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from src.datasets import DRONEWASTE_PAPER_10
from src.vqa_labels import verifiable

SPLIT_INDEX = Path("/home/ids/diecidue/data/vqa/split_index.jsonl")
PARAPHRASES = Path("/home/ids/diecidue/data/vqa/paraphrase_bank.json")
SCENERY = Path("/home/ids/diecidue/data/captions/scenery.jsonl")
DW_COCO = Path("/home/ids/diecidue/data/dronewaste/dronewaste_v1.0.json")
OUT_DIR = Path("/home/ids/diecidue/data/vqa")


def _load_scenery() -> dict[str, str]:
    """{image_path -> scenery text}. Empty dict if scenery.jsonl absent."""
    if not SCENERY.exists():
        return {}
    out: dict[str, str] = {}
    for line in SCENERY.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("scenery") and not d["scenery"].startswith("ERROR:"):
            out[d["image_path"]] = d["scenery"]
    return out


def _load_dw_geometry() -> dict[str, dict]:
    """{image_id_str -> {'w': W, 'h': H, 'bboxes': [[x,y,w,h], ...]}} for DW images.

    bboxes are restricted to paper-10 categories (matches vqa_split.py's filter).
    Used for Q3 grounding.
    """
    with DW_COCO.open() as f:
        data = json.load(f)
    cat_by_id = {c["id"]: c["name"] for c in data["categories"]}
    paper10 = set(DRONEWASTE_PAPER_10)
    out: dict[str, dict] = {}
    for img in data["images"]:
        out[str(img["id"])] = {"w": int(img["width"]), "h": int(img["height"]), "bboxes": []}
    for a in data.get("annotations", []):
        if cat_by_id.get(a["category_id"]) not in paper10:
            continue
        bb = a.get("bbox")  # [x, y, w, h] pixel
        if bb and len(bb) == 4:
            out[str(a["image_id"])]["bboxes"].append(bb)
    return out


def _union_norm_bbox(geom: dict) -> list[float] | None:
    """Union of pixel bboxes -> normalised [x_min, y_min, x_max, y_max] rounded to 3dp."""
    if not geom["bboxes"]:
        return None
    xs1, ys1, xs2, ys2 = [], [], [], []
    for x, y, w, h in geom["bboxes"]:
        xs1.append(x)
        ys1.append(y)
        xs2.append(x + w)
        ys2.append(y + h)
    W, H = geom["w"], geom["h"]
    return [
        round(max(0.0, min(xs1) / W), 3),
        round(max(0.0, min(ys1) / H), 3),
        round(min(1.0, max(xs2) / W), 3),
        round(min(1.0, max(ys2) / H), 3),
    ]


def _make_qa(image_path: str, dataset: str, image_id: str, split: str,
             q_type: str, question: str, answer: str) -> dict:
    return {
        "id": f"{dataset}__{image_id}__{q_type}",
        "image": image_path,
        "split": split,
        "dataset": dataset,
        "image_id": image_id,
        "q_type": q_type,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }


def _options_str(candidates: list[str]) -> str:
    return ", ".join(candidates)


def generate(rng_seed: int = 0) -> dict[str, list[dict]]:
    rng = random.Random(rng_seed)
    paraphrases = json.loads(PARAPHRASES.read_text())
    scenery = _load_scenery()
    dw_geom = _load_dw_geometry()

    rows = [json.loads(l) for l in SPLIT_INDEX.read_text().splitlines() if l.strip()]
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    counts: dict = collections.Counter()

    for r in rows:
        ds = r["dataset"]
        iid = r["image_id"]
        path = r["image_path"]
        split = r["split"]
        gt = list(r["gt_categories"])
        is_pos = bool(gt)

        # Q1 — presence (every image)
        q = rng.choice(paraphrases["presence"])
        a = "Yes." if is_pos else "No."
        by_split[split].append(_make_qa(path, ds, iid, split, "presence", q, a))
        counts[("presence", ds, split)] += 1

        # Q2 — type-ID
        verifiable_set = verifiable(ds)
        # Open form (positives only)
        if is_pos:
            q = rng.choice(paraphrases["waste_type_open"])
            # canonical order = the dataset's verifiable subset ordering
            ordered = [c for c in verifiable_set if c in gt]
            a = ", ".join(ordered) + "." if ordered else "No waste is visible."
            by_split[split].append(_make_qa(path, ds, iid, split, "type_open", q, a))
            counts[("type_open", ds, split)] += 1
        # Choice form (all images)
        candidates = list(verifiable_set)
        rng.shuffle(candidates)
        q_tmpl = rng.choice(paraphrases["waste_type_choice"])
        q = q_tmpl.replace("{options}", _options_str(candidates))
        present = sorted(set(gt) & set(candidates), key=candidates.index)
        a = ", ".join(present) + "." if present else "None of the above."
        by_split[split].append(_make_qa(path, ds, iid, split, "type_choice", q, a))
        counts[("type_choice", ds, split)] += 1

        # Q3 — grounding (DW positives with paper-10 bboxes only)
        if is_pos and ds == "dronewaste_paper10":
            geom = dw_geom.get(iid)
            bbox = _union_norm_bbox(geom) if geom else None
            if bbox:
                q = rng.choice(paraphrases["grounding"])
                a = f"[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]"
                by_split[split].append(_make_qa(path, ds, iid, split, "grounding", q, a))
                counts[("grounding", ds, split)] += 1

        # Q4 — scenery (if available)
        scen = scenery.get(path)
        if scen:
            q = rng.choice(paraphrases["scenery"])
            by_split[split].append(_make_qa(path, ds, iid, split, "scenery", q, scen))
            counts[("scenery", ds, split)] += 1

    return {"by_split": by_split, "counts": counts, "n_images": len(rows),
            "n_scenery": len(scenery)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    result = generate(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for split, items in result["by_split"].items():
        rng = random.Random(args.seed)
        rng.shuffle(items)
        out = OUT_DIR / f"{split}.jsonl"
        with out.open("w") as f:
            for row in items:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        total += len(items)
        print(f"[{split:<5}] {len(items):>6,} QA  -> {out}")

    print()
    print(f"images: {result['n_images']:,};  scenery rows: {result['n_scenery']:,};  total QA: {total:,}")
    print()
    print(f"{'q_type':<14} {'dataset':<22} {'split':<6} {'n':>6}")
    for (qt, ds, sp), n in sorted(result["counts"].items()):
        print(f"{qt:<14} {ds:<22} {sp:<6} {n:>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
