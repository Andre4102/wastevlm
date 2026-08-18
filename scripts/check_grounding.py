"""Is the model's spatial talk grounded, and are its material names right?

Two claims get tested here, both of which look plausible from reading a handful
of samples and neither of which reading can settle:

  1. "It says bottom-left, but does it know where bottom-left is?"
  2. "When it names materials it generally gets them wrong."

AerialWaste carries per-image bounding boxes with a category id, nested under
`images[i]["annotations"]` -- the mcml loader drops them, so nothing in this
project has used them until now. Every *positive* test image is boxed (217/217
in aw_m2), which is exactly the set where "where is the waste" has an answer;
the unboxed remainder are the negatives.

The eval resizes the full image to a square (src/vision_encoder.py), with no
crop, so normalised box coordinates map onto what the model actually saw.

The localisation test is deliberately two-sample rather than accuracy-against-a-
label. Asking "was `bottom-left` correct?" needs a quadrant discretisation that
invents an answer for waste lying on a boundary, and it is scored against a
class distribution the model can game by always naming the commonest quadrant.
Asking instead "when it says left, is the waste further left than when it says
right?" compares two groups of images on a continuous ground-truth coordinate,
needs no threshold, and has an exact null: shuffle the utterances.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
from collections import Counter, defaultdict

DATA = pathlib.Path("/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data")
EVAL = pathlib.Path("/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/results/vlm_eval")
MCML = {"aw_m2": "mcml_split_dataset_1", "aw_m4": "mcml_split_dataset_2"}

WASTE_TERMS = ("waste", "debris", "rubble", "pile", "dump", "litter", "trash",
               "garbage", "scrap", "junk", "refuse", "discard", "rubbish")


# --------------------------------------------------------------------------- gt
def load_boxes(dataset: str, split: str = "test") -> dict:
    """{image_id: {boxes: [(cx, cy, area, cat)], w, h}} in normalised coords."""
    d = json.loads((DATA / "aerialwaste" / MCML[dataset] / f"{split}.json").read_text())
    cat = {c["id"]: c["name"] for c in d["categories"]}
    out = {}
    for img in d["images"]:
        anns = img.get("annotations") or []
        if not anns:
            continue
        W, H = float(img["width"]), float(img["height"])
        boxes = []
        for a in anns:
            x, y, w, h = a["bbox"]
            boxes.append(((x + w / 2) / W, (y + h / 2) / H, (w * h) / (W * H),
                          cat.get(a["category_id"], "?")))
        out[str(img["id"])] = {"boxes": boxes, "w": W, "h": H}
    return out


def gt_centroid(rec: dict) -> tuple[float, float]:
    """Area-weighted centroid of the waste boxes, normalised to [0, 1]."""
    tot = sum(b[2] for b in rec["boxes"]) or 1e-9
    return (sum(b[0] * b[2] for b in rec["boxes"]) / tot,
            sum(b[1] * b[2] for b in rec["boxes"]) / tot)


# ----------------------------------------------------------------------- spoken
CLAUSE_SPLIT = re.compile(r"[.;]|,\s+(?:and|with|while|but)\s+|,(?=\s+[a-z]+ing\b)")
DIRECTIONS = [
    (r"\b(?:bottom|lower)[- ](?:left)\b", (-1, +1)),
    (r"\b(?:bottom|lower)[- ](?:right)\b", (+1, +1)),
    (r"\b(?:top|upper)[- ](?:left)\b", (-1, -1)),
    (r"\b(?:top|upper)[- ](?:right)\b", (+1, -1)),
    (r"\b(?:left)[- ](?:bottom|lower)\b", (-1, +1)),
    (r"\b(?:right)[- ](?:bottom|lower)\b", (+1, +1)),
    (r"\bcent(?:er|re|ral)\b|\bmiddle\b", (0, 0)),
    (r"\bleft\b", (-1, None)),
    (r"\bright\b", (+1, None)),
    (r"\b(?:top|upper)\b", (None, -1)),
    (r"\b(?:bottom|lower)\b", (None, +1)),
]


def spoken_position(text: str) -> dict:
    """Direction words that occur in a clause that also mentions waste.

    Clause scoping is the whole point: "buildings on the left, debris in the
    bottom-right" is one sentence with two positions in it, and a bag-of-words
    reading of the description credits the model for the wrong one.
    """
    xs, ys, hits = [], [], []
    for clause in CLAUSE_SPLIT.split(text.lower()):
        if not any(t in clause for t in WASTE_TERMS):
            continue
        for pat, (dx, dy) in DIRECTIONS:
            for m in re.finditer(pat, clause):
                hits.append(m.group(0))
                if dx is not None:
                    xs.append(dx)
                if dy is not None:
                    ys.append(dy)
                break            # one hit per pattern per clause
    def collapse(v):
        if not v:
            return None
        s = set(v)
        return v[0] if len(s) == 1 else ("conflict" if len(s) > 1 else None)
    return {"x": collapse(xs), "y": collapse(ys), "phrases": hits}


# ------------------------------------------------------------------------ stats
def mannwhitney(a: list[float], b: list[float]) -> tuple[float, float]:
    """Return (AUC of a>b, two-sided p) via normal approximation on ranks."""
    import math
    n1, n2 = len(a), len(b)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i = {}, 0
    vals = [v for v, _ in allv]
    r = [0.0] * len(vals)
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[k] = avg
        i = j + 1
    r1 = sum(r[k] for k, (_, g) in enumerate(allv) if g == 0)
    u1 = r1 - n1 * (n1 + 1) / 2
    auc = u1 / (n1 * n2)
    mu = n1 * n2 / 2
    sd = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12) or 1e-9
    z = (u1 - mu) / sd
    p = math.erfc(abs(z) / math.sqrt(2))
    return auc, p


def two_sample(name: str, neg: list[float], pos: list[float],
               lo: str, hi: str) -> None:
    """`neg` = GT coord where the model said `lo`; `pos` = where it said `hi`."""
    if not neg or not pos:
        print(f"  {name:12s} n/a  (said {lo}: {len(neg)}, said {hi}: {len(pos)})")
        return
    auc, p = mannwhitney(pos, neg)
    print(f"  {name:12s} said {lo:6s} n={len(neg):3d} mean {sum(neg)/len(neg):.3f}   "
          f"said {hi:6s} n={len(pos):3d} mean {sum(pos)/len(pos):.3f}   "
          f"AUC {auc:.3f}  p={p:.2g}")


# ----------------------------------------------------------------------- report
def load_rows(arm: str, dataset: str, style: str = "open_cot") -> list[dict]:
    run = EVAL / f"vlm_cradiov4-so_r768ps2_{arm}_{dataset}_{style}"
    rows = []
    for line in (run / "raw_responses.jsonl").open():
        if line.strip():
            r = json.loads(line)
            rows.append({"image_id": str(r["image_id"]), "gt": r["gt"],
                         "turn1": r.get("raw_turn1") or "",
                         "parsed": r.get("parsed") or []})
    return rows


def report_localisation(arm: str, dataset: str, gt: dict, seed: int = 0) -> None:
    rows = [r for r in load_rows(arm, dataset) if r["image_id"] in gt and r["gt"]]
    print(f"\n=== {arm} / {dataset}: localisation on {len(rows)} boxed positives")

    said = [(r, spoken_position(r["turn1"])) for r in rows]
    n_any = sum(1 for _, s in said if s["x"] or s["y"])
    print(f"  places the waste at all: {n_any}/{len(rows)} ({n_any/len(rows):.0%})")
    ph = Counter(p for _, s in said for p in s["phrases"])
    print(f"  phrases: {', '.join(f'{k} {v}' for k, v in ph.most_common(8))}")

    for axis, lo, hi, idx in (("horizontal", "left", "right", 0),
                              ("vertical", "top", "bottom", 1)):
        key = "x" if idx == 0 else "y"
        neg = [gt_centroid(gt[r["image_id"]])[idx] for r, s in said if s[key] == -1]
        pos = [gt_centroid(gt[r["image_id"]])[idx] for r, s in said if s[key] == +1]
        two_sample(axis, neg, pos, lo, hi)

    # Does "centre" actually mean centre? Compare distance-from-centre.
    import math
    cen = [math.dist(gt_centroid(gt[r["image_id"]]), (0.5, 0.5))
           for r, s in said if s["x"] == 0 and s["y"] == 0]
    off = [math.dist(gt_centroid(gt[r["image_id"]]), (0.5, 0.5))
           for r, s in said if (s["x"] in (-1, 1)) or (s["y"] in (-1, 1))]
    two_sample("centre-dist", cen, off, "centre", "edge")


def report_premise(gt: dict, dataset: str) -> None:
    """The user's premise: is AerialWaste waste just always in the middle?"""
    import math
    cs = [gt_centroid(v) for v in gt.values()]
    d = sorted(math.dist(c, (0.5, 0.5)) for c in cs)
    n = len(cs)
    print(f"\n=== where the waste actually is ({n} boxed positives, {dataset})")
    print(f"  centroid x: mean {sum(c[0] for c in cs)/n:.3f}  "
          f"y: mean {sum(c[1] for c in cs)/n:.3f}")
    print(f"  distance from image centre: median {d[n//2]:.3f}  "
          f"p10 {d[n//10]:.3f}  p90 {d[9*n//10]:.3f}   "
          f"(uniform-random point would be ~0.38)")
    q = Counter(("top" if c[1] < .5 else "bottom") + "-" + ("left" if c[0] < .5 else "right")
                for c in cs)
    print("  quadrant of centroid: " + ", ".join(f"{k} {v} ({v/n:.0%})"
                                                 for k, v in q.most_common()))
    inner = sum(1 for c in cs if 0.33 < c[0] < 0.67 and 0.33 < c[1] < 0.67)
    print(f"  centroid inside the middle third: {inner}/{n} ({inner/n:.0%}) "
          f"-- a uniform centroid would give 11%")


def report_naming(arm: str, dataset: str, style: str = "open_cot") -> None:
    rows = load_rows(arm, dataset, style)
    pos = [r for r in rows if r["gt"]]
    named = [r for r in pos if r["parsed"]]
    print(f"\n=== {arm} / {dataset} / {style}: naming on {len(pos)} positives "
          f"({len(named)} of which get any label at all)")

    cats = sorted({c for r in rows for c in r["gt"]} |
                  {c for r in rows for c in r["parsed"]})
    tp = Counter(); fp = Counter(); fn = Counter()
    conf = defaultdict(Counter)
    for r in pos:
        g, p = set(r["gt"]), set(r["parsed"])
        for c in g & p:
            tp[c] += 1
        for c in p - g:
            fp[c] += 1
        for c in g - p:
            fn[c] += 1
        for t in g:
            for s in (p or {"(said nothing)"}):
                conf[t][s] += 1

    # Precision alone flatters a common class: on a set where 71% of images
    # contain bulky items, saying "bulky items" at random already scores 0.71.
    # The column that matters is precision MINUS prevalence -- how much the
    # label being emitted actually changes the odds that it is there. At or
    # below zero, the word carries no information about the image.
    print(f"  {'category':32s} {'supp':>5s} {'prev':>6s} {'P':>6s} {'lift':>6s} "
          f"{'R':>6s} {'F1':>6s} {'said':>6s}")
    for c in cats:
        supp = tp[c] + fn[c]
        said = tp[c] + fp[c]
        prev = supp / len(pos) if pos else 0.0
        p = tp[c] / said if said else 0.0
        rr = tp[c] / supp if supp else 0.0
        f1 = 2 * p * rr / (p + rr) if p + rr else 0.0
        mark = "" if said == 0 else ("   <-- at chance" if p - prev <= 0.03 else "")
        print(f"  {c[:32]:32s} {supp:5d} {prev:6.3f} {p:6.3f} {p-prev:+6.3f} "
              f"{rr:6.3f} {f1:6.3f} {said:6d}{mark}")
    TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
    micro_p = TP / (TP + FP) if TP + FP else 0.0
    micro_r = TP / (TP + FN) if TP + FN else 0.0
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    print(f"  micro  P {micro_p:.3f}  R {micro_r:.3f}  F1 {micro_f:.3f}"
          f"   (TP {TP}  FP {FP}  FN {FN})")

    # "when it names materials it generally gets it wrong" is a claim about the
    # images where it names something, not about the ones where it stays silent.
    if named:
        hit = sum(1 for r in named if set(r["parsed"]) & set(r["gt"]))
        exact = sum(1 for r in named if set(r["parsed"]) == set(r["gt"]))
        print(f"  given that it named something ({len(named)} images): "
              f"at least one label right {hit}/{len(named)} ({hit/len(named):.0%}), "
              f"label set exactly right {exact}/{len(named)} ({exact/len(named):.0%})")

    print("\n  when the truth is X, it says:")
    for t in cats:
        if not conf[t]:
            continue
        tot = sum(conf[t].values())
        top = ", ".join(f"{s[:28]} {v/tot:.0%}" for s, v in conf[t].most_common(3))
        print(f"    {t[:30]:30s} (n={sum(1 for r in pos if t in r['gt']):3d})  {top}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="stage3_n2d,stage3_n1,stage3_s3b")
    ap.add_argument("--dataset", default="aw_m2")
    ap.add_argument("--styles", default="open_cot,closed_vocab")
    args = ap.parse_args()

    gt = load_boxes(args.dataset)
    report_premise(gt, args.dataset)
    for arm in args.arms.split(","):
        run = EVAL / f"vlm_cradiov4-so_r768ps2_{arm}_{args.dataset}_open_cot"
        if not run.exists():
            print(f"\n-- {arm}: no run at {run.name}")
            continue
        report_localisation(arm, args.dataset, gt)
    for style in args.styles.split(","):
        for arm in args.arms.split(","):
            if (EVAL / f"vlm_cradiov4-so_r768ps2_{arm}_{args.dataset}_{style}").exists():
                report_naming(arm, args.dataset, style)


if __name__ == "__main__":
    main()
