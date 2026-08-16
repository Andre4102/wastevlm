"""Build the nadir DESCRIPTION corpus -- the corrected s3b component.

`rs_sft` taught the model to REFUSE at nadir (DIOR/VRSBench VQA, mean answer 1.1
words, 15% `not_determinable`), and we measured what that did: AW positives sit at
margin -1.93, the model says "No" to all 581 images, and captions mention "pile"
on 3.3% of positives vs 60.1% on drone imagery. The missing capability is not
abstention and not perception -- the frozen encoder separates AW at AUC 0.97 --
it is *saying what is there* when the target is diffuse, low-contrast and nadir.

Two sources, chosen from what is on disk rather than reputation:

  vrsbench_cap  20,264 grounded nadir captions (one per image, mean 53 words)
                naming objects, positions and scene context. The [vqa]/[refer]
                splits are excluded: their answers average 1.1 words and would
                re-teach terseness.

  loveda        1024x1024 nadir tiles at ~0.3 m/px -- the closest GSD we hold to
                AerialWaste's ~0.2 -- with PER-PIXEL masks over 7 land-cover
                classes. The masks are the point: they give AREA-FRACTION targets
                (diffuse regions, not crisp objects, which is exactly AW's failure
                mode) and TRUE absence, so negatives are facts about the image
                rather than an invented refusal policy.

xBD was the original plan and is REJECTED on inspection: GSD 1.25-3.15 m/px
(median 2.11) against AerialWaste's ~0.2, only 300 materialised scenes, and its
QA is quantitative damage-inventory reporting over pre/post image PAIRS plus
geojson layers -- a shape our single-image model cannot consume.

Ambiguity rule: a class counts as present only above `--min-present` of the tile
and absent only at exactly zero pixels; the band between is DROPPED. Teaching a
decision on marginal coverage is how `not_determinable` taught refusal on
resolvable objects.

    python scripts/build_nadir_desc.py --out <dir>
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import random

import numpy as np
from PIL import Image

DATA = pathlib.Path("/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data")
EXT = DATA / "external"
VRSBENCH = EXT / "VRSBench"
LOVEDA = EXT / "loveda"

# id2label.json; 0 = Ignore (unlabelled, excluded from area), 1 = Background.
LOVEDA_CLASSES = {2: "buildings", 3: "roads", 4: "water",
                  5: "barren ground", 6: "forest", 7: "agricultural land"}
# Count nouns take "are", mass nouns take "is". Getting this wrong would print
# "There is no roads" on every negative -- a systematic grammatical tell that
# marks exactly the records whose answer is No.
PLURAL = {"buildings", "roads"}


def be(name: str) -> str:
    return "are" if name in PLURAL else "is"


DESCRIBE_PROMPTS = [
    "Describe what you see in this aerial image.",
    "This is an overhead satellite image. Describe its contents.",
    "What does this aerial photograph show?",
    "Describe the land cover visible in this overhead image.",
]
# "contain" is number-agnostic; the other two carry an explicit {be}.
PRESENCE_PROMPTS = [
    "{Be} there any {c} visible in this aerial image? Answer Yes or No.",
    "Looking at this overhead image, {be} any {c} present? Answer Yes or No.",
    "Does this aerial image contain {c}? Answer Yes or No.",
]


def presence_prompt(rng: random.Random, name: str) -> str:
    v = be(name)
    return rng.choice(PRESENCE_PROMPTS).format(c=name, be=v, Be=v.capitalize())


def rid(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]


def quadrant_phrase(mask: np.ndarray) -> str:
    """Rough location of a class, from the centroid of its pixels.

    Deliberately coarse: at 0.3 m/px over a 1024 tile the model should learn
    "upper left", not a box it cannot actually resolve from 576 visual tokens.
    """
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return ""
    h, w = mask.shape
    cy, cx = ys.mean() / h, xs.mean() / w
    if 0.33 < cy < 0.67 and 0.33 < cx < 0.67:
        return "in the centre"
    v = "upper" if cy < 0.5 else "lower"
    return f"in the {v} {'left' if cx < 0.5 else 'right'}"


def coverage_phrase(frac: float) -> str:
    if frac >= 0.45:
        return "dominating the scene"
    if frac >= 0.20:
        return "covering a large part of the scene"
    if frac >= 0.07:
        return "covering a moderate area"
    return "in a small patch"


def build_loveda(rng: random.Random, min_present: float, max_images: int) -> list[dict]:
    recs: list[dict] = []
    pairs: list[tuple[pathlib.Path, pathlib.Path]] = []
    for split in ("train", "val"):
        img_dir = LOVEDA / f"urban:rural {split} images"
        msk_dir = LOVEDA / f"urban:rural {split} masks"
        if not img_dir.is_dir() or not msk_dir.is_dir():
            continue
        for img in sorted(img_dir.glob("*.png")):
            m = msk_dir / img.name
            if m.exists():
                pairs.append((img, m))
    if not pairs:
        raise SystemExit(f"no LoveDA image/mask pairs under {LOVEDA}")
    rng.shuffle(pairs)
    if max_images > 0:
        pairs = pairs[:max_images]

    n_drop = 0
    pos_used: collections.Counter = collections.Counter()
    neg_used: collections.Counter = collections.Counter()
    for img_path, msk_path in pairs:
        arr = np.array(Image.open(msk_path))
        labelled = arr != 0                      # 0 = Ignore
        n_lab = int(labelled.sum())
        if n_lab < arr.size * 0.5:               # mostly unlabelled tile
            continue
        fracs = {cid: float((arr == cid).sum()) / n_lab for cid in LOVEDA_CLASSES}

        present = {c: f for c, f in fracs.items() if f >= min_present}
        absent = [c for c, f in fracs.items() if f == 0.0]
        n_drop += sum(1 for f in fracs.values() if 0.0 < f < min_present)
        if not present:
            continue

        # --- affirmative description, ordered by how much of the tile it covers
        parts = []
        for cid, f in sorted(present.items(), key=lambda kv: -kv[1]):
            loc = quadrant_phrase(arr == cid)
            parts.append(f"{LOVEDA_CLASSES[cid]} {coverage_phrase(f)}"
                         + (f" {loc}" if loc else ""))
        if len(parts) == 1:
            body = f"This overhead image shows {parts[0]}."
        else:
            body = ("This overhead image shows " + ", ".join(parts[:-1])
                    + f", and {parts[-1]}.")
        recs.append({
            "id": f"loveda_cap__{rid(img_path.name, 'cap')}",
            "image": str(img_path), "source": "loveda", "component": "nadir_desc",
            "conversations": [
                {"from": "human", "value": "<image>\n" + rng.choice(DESCRIBE_PROMPTS)},
                {"from": "gpt", "value": body},
            ],
        })

        # --- presence decisions: one positive + one negative per image, so the
        # component is balanced by construction rather than by post-hoc sampling.
        # Pick the LEAST-USED eligible class on each side: rural tiles are missing
        # `roads` far more often than anything else, and choosing uniformly over
        # absent classes would make "roads -> No" learnable as a class prior
        # without looking at the image.
        pos_cid = min(present, key=lambda c: (pos_used[c], rng.random()))
        neg_cid = min(absent, key=lambda c: (neg_used[c], rng.random())) if absent else None
        for cid, want in ((pos_cid, 1), (neg_cid, 0)):
            if cid is None:
                continue
            (pos_used if want else neg_used)[cid] += 1
            name = LOVEDA_CLASSES[cid]
            v = be(name)
            if want:
                f = fracs[cid]
                loc = quadrant_phrase(arr == cid)
                ans = (f"Yes. There {v} {name} {coverage_phrase(f)}"
                       + (f" {loc}." if loc else "."))
            else:
                ans = f"No. There {v} no {name} visible in this image."
            recs.append({
                "id": f"loveda_dec__{rid(img_path.name, name, str(want))}",
                "image": str(img_path), "source": "loveda",
                "component": "nadir_desc", "decision": want,
                "polarity": "affirmative" if want else "negative",
                "conversations": [
                    {"from": "human", "value": "<image>\n" + presence_prompt(rng, name)},
                    {"from": "gpt", "value": ans},
                ],
            })
    print(f"[loveda] {len(pairs)} tiles -> {len(recs)} records "
          f"({n_drop} class/image pairs dropped as ambiguous, "
          f"0 < coverage < {min_present})")
    # Print the per-class split: if any class is overwhelmingly one polarity, the
    # decision is learnable from the class name alone and the component is a
    # shortcut rather than a perception task.
    print("  class            pos    neg")
    for cid, name in LOVEDA_CLASSES.items():
        print(f"  {name:16s} {pos_used[cid]:5d}  {neg_used[cid]:5d}")
    return recs


def build_vrsbench_captions(rng: random.Random) -> list[dict]:
    src = VRSBENCH / "VRSBench_train.json"
    img_root = VRSBENCH / "images_train" / "Images_train"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    recs = []
    for r in json.load(src.open()):
        conv = r.get("conversations", [])
        if not conv or "[caption]" not in conv[0].get("value", ""):
            continue                      # [vqa]/[refer] answers average 1.1 words
        img = img_root / r["image"]
        # Rewrite the task tag out of the prompt: at eval nobody types "[caption]",
        # and leaving it in makes the behaviour conditional on a token the
        # benchmark prompts never contain.
        recs.append({
            "id": f"vrsbench_cap__{rid(r['image'])}",
            "image": str(img), "source": "vrsbench_cap", "component": "nadir_desc",
            "conversations": [
                {"from": "human", "value": "<image>\n" + rng.choice(DESCRIBE_PROMPTS)},
                {"from": "gpt", "value": conv[1]["value"]},
            ],
        })
    print(f"[vrsbench] {len(recs)} caption records")
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path, default=DATA / "nadir_desc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-present", type=float, default=0.02,
                    help="coverage at or above which a class counts as present; "
                         "strictly between 0 and this is dropped as ambiguous")
    ap.add_argument("--max-loveda", type=int, default=0, help="0 = all")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    recs = build_vrsbench_captions(rng) + build_loveda(
        rng, args.min_present, args.max_loveda)

    missing = [r for r in recs[:400] if not pathlib.Path(r["image"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)}/400 sampled images missing, "
                         f"e.g. {missing[0]['image']}")

    by_src = collections.Counter(r["source"] for r in recs)
    n_dec = sum(1 for r in recs if "decision" in r)
    n_pos = sum(1 for r in recs if r.get("decision") == 1)
    words = sum(len(r["conversations"][1]["value"].split()) for r in recs)
    print(f"\n=== nadir_desc: {len(recs)} records, ~{words/1e6:.2f}M answer words")
    for s, n in by_src.most_common():
        print(f"  {s:16s} {n:7d}")
    print(f"  decision records {n_dec} ({n_pos} positive / {n_dec-n_pos} negative)")
    print("  not_determinable 0  <- by construction: no abstention in this corpus")

    if args.dry_run:
        return
    args.out.mkdir(parents=True, exist_ok=True)
    rng.shuffle(recs)
    path = args.out / "nadir_desc.jsonl"
    with path.open("w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    (args.out / "nadir_desc.meta.json").write_text(json.dumps({
        "n_records": len(recs), "by_source": dict(by_src),
        "decision_records": n_dec, "decision_positive": n_pos,
        "min_present": args.min_present, "seed": args.seed,
        "held_out": ["aerialwaste", "dronewaste"],
        "excluded": "xbd (GSD 1.25-3.15 m/px, pre/post pairs, 300 scenes); "
                    "vrsbench [vqa]/[refer] (1.1-word answers)",
    }, indent=2))
    print(f"\n[write] {path}")


if __name__ == "__main__":
    main()
