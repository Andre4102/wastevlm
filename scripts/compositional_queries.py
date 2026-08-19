"""A query set that separates what the decoder can do from what the encoder can.

Every task measured so far has been one region, one label -- exactly the shape a
CLIP-style text head is built for, and exactly where a frozen C-RADIOv4 plus a
linear head beats the 7B decoder outright (0.665 / 0.733 against a readout bounded
at +0.043 over a constant predictor). If the decoder earns its place anywhere, it
is on questions whose answer is not a property of any single region.

The whole design turns on one control. For each question we also compute what a
**presence oracle** answers -- an agent told exactly which categories appear in
the image and nothing else: no counts, no geometry, no areas. That is an upper
bound on what the text-head pipeline can do, because scoring every class against
every patch and thresholding is precisely how you recover the presence set. So:

  presence oracle near ceiling  -> the question is presence-solvable, and is a
                                   CONTROL where a tie is the expected result
  presence oracle near chance   -> the question needs composition, and is where
                                   a decoder could actually show a difference

A family that looks compositional but that the oracle solves is not evidence, and
publishing it as evidence would be the error this whole project has been spent
avoiding. Both kinds are generated and labelled, because the controls are needed
to show the decoder does not *lose* on the simple cases.

Answers are balanced within each family, so a constant answerer scores 0.5 on the
binary ones rather than whatever the class prior happens to give.

    python scripts/compositional_queries.py --out queries.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = pathlib.Path(os.environ.get(
    "WASTE_DATA_ROOT",
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data"))

# the same held-out sites the ROI probe uses, so arms stay comparable
TEST_SITES = ("site5", "site9", "site13", "site17", "site2")


def load(sites) -> dict:
    """-> image path -> {'size', 'objs': [{'cat','box','area','cx','cy'}]}"""
    w = json.loads((DATA / "dronewaste" / "dronewaste_v1.0.json").read_text())
    cat = {c["id"]: c["name"] for c in w["categories"]}
    img = {i["id"]: i for i in w["images"]}
    out = defaultdict(lambda: {"size": None, "objs": []})
    for a in w["annotations"]:
        im = img[a["image_id"]]
        if sites and im["site"] not in sites:
            continue
        p = str(DATA / "dronewaste" / "images" / im["file_name"])
        x, y, bw, bh = a["bbox"]
        out[p]["size"] = (im["width"], im["height"])
        out[p]["objs"].append({
            "cat": cat[a["category_id"]], "box": [x, y, bw, bh],
            # Box area, not the annotation's polygon area. A detector only ever
            # produces a box, so grounding the questions in polygon area makes the
            # ceiling arm unreachable by construction -- it scored 0.977 on
            # area_compare and 0.974 on superlative purely from this mismatch,
            # which would then have been read as reasoning error.
            "area": float(bw * bh),
            "cx": x + bw / 2, "cy": y + bh / 2,
        })
    return dict(out)


def q(image, family, question, answer, presence_answer, meta=None):
    """One record. `presence_answer` is what the oracle says knowing only the
    set of categories present -- None when the oracle must guess."""
    return {"image": image, "family": family, "question": question,
            "answer": answer, "presence_answer": presence_answer, "meta": meta or {}}


def build(rec, path, rng, cats_all):
    size, objs = rec["size"], rec["objs"]
    W, H = size
    present = sorted({o["cat"] for o in objs})
    if not present:
        return []
    absent = [c for c in cats_all if c not in present]
    n_by_cat = Counter(o["cat"] for o in objs)
    area_by_cat = defaultdict(float)
    for o in objs:
        area_by_cat[o["cat"]] += o["area"]
    out = []

    # --- CONTROL: presence. The oracle is right by construction; so should the
    # pipeline be. Included to show the decoder does not regress on the easy case.
    c = rng.choice(present)
    out.append(q(path, "presence", f"Is there any {c.lower()} visible in this image?",
                 "yes", "yes", {"cat": c}))
    if absent:
        c = rng.choice(absent)
        out.append(q(path, "presence", f"Is there any {c.lower()} visible in this image?",
                     "no", "no", {"cat": c}))

    # --- COUNT. The oracle knows the category is there and nothing more.
    for c in present:
        n = n_by_cat[c]
        out.append(q(path, "count",
                     f"How many separate piles of {c.lower()} are visible?",
                     str(n if n < 4 else "4+"), None, {"cat": c, "n": n}))

    # --- COUNT COMPARISON, restricted to pairs BOTH present so presence cannot
    # decide it. Skipped on ties, which have no defensible answer.
    if len(present) >= 2:
        a, b = rng.sample(present, 2)
        if n_by_cat[a] != n_by_cat[b]:
            ans = "yes" if n_by_cat[a] > n_by_cat[b] else "no"
            out.append(q(path, "count_compare",
                         f"Are there more piles of {a.lower()} than of {b.lower()}?",
                         ans, None, {"a": a, "b": b,
                                     "na": n_by_cat[a], "nb": n_by_cat[b]}))

    # --- AREA COMPARISON, same restriction, with a 20% margin so near-ties are
    # not scored as if they had a right answer.
    if len(present) >= 2:
        a, b = rng.sample(present, 2)
        ra, rb = area_by_cat[a], area_by_cat[b]
        if max(ra, rb) > 1.2 * min(ra, rb):
            out.append(q(path, "area_compare",
                         f"Does {a.lower()} cover more ground than {b.lower()}?",
                         "yes" if ra > rb else "no", None,
                         {"a": a, "b": b, "area_a": ra, "area_b": rb}))

    # --- SUPERLATIVE over the categories actually present, so the answer cannot
    # be reached by naming the globally commonest class.
    if len(present) >= 2:
        top = max(present, key=lambda c: area_by_cat[c])
        second = sorted(present, key=lambda c: -area_by_cat[c])[1]
        if area_by_cat[top] > 1.2 * area_by_cat[second]:
            out.append(q(path, "superlative",
                         "Which kind of waste covers the largest area in this image?",
                         top, None, {"present": present}))

    # --- SPATIAL RELATION between two categories, both present.
    #
    # The truth has to be evaluated over EVERY pair of instances, not over one
    # chosen pair: with three tyres and two pallets, "is there a tyre left of a
    # pallet" can be true of some pair and false of the one we happened to pick,
    # and scoring the picked pair would mark a correct answer wrong. So a
    # direction is true if ANY instance pair realises it with a margin of a tenth
    # of the frame, and a "no" is only emitted for a direction NO pair realises.
    if len(present) >= 2:
        a, b = rng.sample(present, 2)
        A = [o for o in objs if o["cat"] == a]
        B = [o for o in objs if o["cat"] == b]
        holds = set()
        for o1 in A:
            for o2 in B:
                dx, dy = o1["cx"] - o2["cx"], o1["cy"] - o2["cy"]
                if abs(dx) > W / 10 and abs(dx) > abs(dy):
                    holds.add("to the right of" if dx > 0 else "to the left of")
                elif abs(dy) > H / 10 and abs(dy) > abs(dx):
                    holds.add("below" if dy > 0 else "above")
        DIRS = ("to the left of", "to the right of", "above", "below")
        false_dirs = [d for d in DIRS if d not in holds]
        for d, ans in ([(rng.choice(sorted(holds)), "yes")] if holds else []) + \
                      ([(rng.choice(false_dirs), "no")] if false_dirs else []):
            out.append(q(path, "spatial",
                         f"Is there any {a.lower()} {d} any {b.lower()}?",
                         ans, None, {"a": a, "b": b, "dir": d}))

    # --- NEGATION WITH GEOMETRY. Plain "is there anything other than X" is
    # presence-solvable; requiring the other pile to be NEAR X is not.
    if len(present) >= 2:
        a = rng.choice(present)
        near = False
        for o1 in objs:
            if o1["cat"] != a:
                continue
            for o2 in objs:
                if o2["cat"] == a:
                    continue
                d = ((o1["cx"] - o2["cx"]) ** 2 + (o1["cy"] - o2["cy"]) ** 2) ** 0.5
                if d < 0.25 * ((W ** 2 + H ** 2) ** 0.5):
                    near = True
        out.append(q(path, "negation_spatial",
                     f"Is there a pile that is not {a.lower()} close to the "
                     f"{a.lower()}?", "yes" if near else "no", None, {"cat": a}))
    return out


def balance(rows, rng):
    """Equalise yes/no within each binary family so a constant answer scores 0.5."""
    out, by = [], defaultdict(list)
    for r in rows:
        (by[(r["family"], r["answer"])] if r["answer"] in ("yes", "no")
         else out).append(r)
    fams = {f for f, _a in by}
    for f in fams:
        y, n = by[(f, "yes")], by[(f, "no")]
        k = min(len(y), len(n))
        rng.shuffle(y); rng.shuffle(n)
        out += y[:k] + n[:k]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-sites", action="store_true")
    ap.add_argument("--dev-sites", action="store_true",
                    help="generate over the TRAINING sites, for developing the "
                         "harness without touching the evaluation set")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    if args.all_sites:
        data = load(None)
    elif args.dev_sites:
        every = load(None)
        data = {k: v for k, v in every.items()
                if pathlib.Path(k).name.split("_")[0] not in TEST_SITES}
    else:
        data = load(TEST_SITES)
    cats_all = sorted({o["cat"] for r in data.values() for o in r["objs"]})
    print(f"[cq] {len(data)} annotated images, {len(cats_all)} categories, "
          f"sites={'all' if args.all_sites else ('dev' if args.dev_sites else ','.join(TEST_SITES))}")

    rows = []
    for p, rec in sorted(data.items()):
        rows += build(rec, p, rng, cats_all)
    rows = balance(rows, rng)
    rng.shuffle(rows)

    with open(args.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # ---------------------------------------------------------------- summary
    print(f"\n{len(rows)} questions over {len({r['image'] for r in rows})} images\n")
    print(f"  {'family':18s} {'n':>5s}  {'majority':>8s}  {'presence oracle':>15s}  "
          f"{'bar':>6s}  verdict")
    print("  (bar = the higher of the two; that is the number an arm must beat)")
    for fam in sorted({r["family"] for r in rows}):
        sub = [r for r in rows if r["family"] == fam]
        cnt = Counter(r["answer"] for r in sub)
        maj = max(cnt.values()) / len(sub)
        known = [r for r in sub if r["presence_answer"] is not None]
        if known:
            orc = sum(r["presence_answer"] == r["answer"] for r in known) / len(sub)
        else:
            # the oracle knows only the present set; on a balanced binary family
            # that leaves it guessing, which is the majority rate by construction
            orc = maj if len(cnt) > 1 else 1.0
            orc = 1.0 / len(cnt) if len(cnt) > 2 else maj
        verdict = "CONTROL (presence-solvable)" if orc > 0.9 else "compositional"
        bar = max(maj, orc)
        print(f"  {fam:18s} {len(sub):5d}  {maj:8.3f}  {orc:15.3f}  {bar:6.3f}  {verdict}")
    print(f"\n[write] {args.out}")


if __name__ == "__main__":
    main()
