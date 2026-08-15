"""Build the non-waste aerial instruction-tuning mix (Arm A of SFT_DESIGN.md).

Teaches three behaviours, none of them waste-specific, on nadir imagery:
  1. describe an aerial scene         <- VRSBench captions
  2. answer about specific details    <- DIOR boxes + VRSBench VQA
  3. answer NEGATIVELY when the question is wrong  <- DIOR, exhaustively annotated

AerialWaste and DroneWaste are held out of this entirely; every eval is zero-shot.

Why the negatives come from DIOR and not VRSBench
-------------------------------------------------
Derived negatives are only sound over an *exhaustive* annotation. VRSBench's
`objects[]` holds referring-expression targets (1.8 per image; a category named in
the caption is missing from `objects[]` 9.1% of the time), so "is there an X?" ->
"no" built from it would teach the model to deny things that are present. DIOR
labels every instance of its 20 classes, so absence there is real. VRSBench's own
`object existence` QA pairs are verified against the image and are folded in as-is.

Two scoping rules that keep the derived answers truthful:
  * DIOR's 20 classes do NOT include generic buildings or roads, so a
    zero-object image is not an empty scene. Whole-image absence questions are
    therefore always scoped to the taxonomy ("are any of the following present:
    ...") and never open-ended ("what structures are visible?").
  * "Cannot be determined" targets are only emitted for objects that fall below
    one visual token at the training resolution -- where the claim is honest --
    and are phrased as low confidence rather than as a flat refusal.

    python scripts/build_rs_sft.py --out <dir> [--img-size 768 --pixel-shuffle 2]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import pathlib
import random
import re
import xml.etree.ElementTree as ET

EXT = pathlib.Path(
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data/external"
)
DIOR_ANN = EXT / "DIOR/annotations_trainval/Annotations/trainval"
DIOR_IMG_DIRS = [EXT / "DIOR/images_trainval/Images/trainval",
                 EXT / "DIOR/images_test/Images/test"]
VRS_ROOT = EXT / "VRSBench"

# DIOR class id -> human phrasing. The raw labels are lowercase-concatenated
# ("golffield", "Expressway-Service-area"); putting those in a prompt verbatim
# teaches the model a vocabulary no user will ever type.
PRETTY = {
    "airplane": "airplane", "airport": "airport", "baseballfield": "baseball field",
    "basketballcourt": "basketball court", "bridge": "bridge", "chimney": "chimney",
    "dam": "dam", "Expressway-Service-area": "expressway service area",
    "Expressway-toll-station": "expressway toll station", "golffield": "golf course",
    "groundtrackfield": "ground track field", "harbor": "harbor",
    "overpass": "overpass", "ship": "ship", "stadium": "stadium",
    "storagetank": "storage tank", "tenniscourt": "tennis court",
    "trainstation": "train station", "vehicle": "vehicle", "windmill": "windmill",
}

# Prompt phrasings. Several per family, sampled -- a single template per family is
# how the 150K model learned to emit one caption for every image (EXPERIMENTS.md).
Q_PRESENCE = [
    "Is there {a} {c} in this image?",
    "Does this image contain {a} {c}?",
    "Can you see {a} {c} here?",
    "Are there any {p} visible in this scene?",
]
Q_LOCATION = [
    "In which part of the image is the {c}?",
    "Where in this scene is the {c} located?",
    "Whereabouts can the {c} be found?",
]
Q_COUNT = [
    "How many {p} are visible in this image?",
    "Count the {p} in this scene.",
    "What number of {p} can you see?",
]
Q_EXTENT = [
    "Which is the largest object visible in this image?",
    "What is the most prominent object in this scene?",
]
Q_COOCCUR = [
    "Besides the {c}, what else is present in this image?",
    "Apart from the {c}, what other objects can you identify?",
]
Q_FALSE_PREMISE = [
    "Describe the {c} in this image.",
    "What can you tell me about the {c} shown here?",
    "Tell me about the {c} visible in this scene.",
]
Q_SCOPED_ABSENCE = [
    "Are any of the following present in this image: {list}?",
    "Does this scene contain any of these: {list}?",
]
Q_SMALL = [
    "What is the small object near the {loc} of the image?",
    "Can you identify the small feature toward the {loc}?",
]

A_NO = [
    "No, there {be} no {p} in this image.",
    "No — I don't see any {p} here.",
    "No, this image contains no {p}.",
]
A_YES = [
    "Yes, {a} {c} is visible in this image.",
    "Yes — there {be} {c_with_count} here.",
    "Yes, I can see {c_with_count}.",
]
A_FALSE_PREMISE = [
    "There is no {c} in this image. What I can see is {alt}.",
    "This image doesn't contain {a} {c}. It shows {alt}.",
    "I don't see {a} {c} here — the visible objects are {alt}.",
]
A_UNCERTAIN = [
    "There is a small object there, but at this resolution I can't determine what "
    "it is with confidence.",
    "Something is present at that spot, though it is too small here to identify "
    "reliably.",
    "I can make out a small feature, but its type isn't resolvable at this scale.",
]


# SFT_DESIGN.md §6: each negative type blocks a different shortcut, so the mix is
# specified rather than left to whatever the annotations happen to yield most of.
NEG_MIX = {
    "absent_category": 0.45,    # asked on a POPULATED image -- forces per-category
                                # discrimination, unanswerable from "is this busy?"
    "scoped_absence": 0.25,     # the none-of-the-above answer shape
    "false_premise": 0.15,      # reject a presupposition rather than comply with it
    "not_determinable": 0.15,   # calibrated uncertainty on sub-token objects
}


def an(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def plural(word: str) -> str:
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    return word + "s"


def quadrant(cx: float, cy: float) -> str:
    v = "top" if cy < 0.4 else ("bottom" if cy > 0.6 else "middle")
    h = "left" if cx < 0.4 else ("right" if cx > 0.6 else "center")
    if v == "middle" and h == "center":
        return "center"
    if v == "middle":
        return f"{h} side"
    if h == "center":
        return f"{v} center"
    return f"{v}-{h}"


def load_dior() -> list[dict]:
    """-> [{image, classes: {cls: [boxes]}, size}] over every annotated DIOR image."""
    index: dict[str, pathlib.Path] = {}
    for d in DIOR_IMG_DIRS:
        for p in d.glob("*.jpg"):
            index[p.stem] = p
    out = []
    for f in sorted(glob.glob(str(DIOR_ANN / "*.xml"))):
        stem = pathlib.Path(f).stem
        img = index.get(stem)
        if img is None:      # trainval annotations cover images shipped in both zips
            continue
        root = ET.parse(f).getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        boxes: dict[str, list[tuple[int, int, int, int]]] = collections.defaultdict(list)
        for o in root.findall("object"):
            name = o.find("name").text
            b = o.find("bndbox")
            boxes[name].append((int(b.find("xmin").text), int(b.find("ymin").text),
                                int(b.find("xmax").text), int(b.find("ymax").text)))
        out.append({"image": str(img), "w": w, "h": h, "classes": dict(boxes)})
    return out


def rec(image: str, question: str, answer: str, source: str, kind: str,
        polarity: str) -> dict:
    return {
        "id": f"{source}__{kind}__{abs(hash((image, question))) % 10**10}",
        "image": image,
        "source": source,
        "kind": kind,
        "polarity": polarity,          # "affirmative" | "negative" -- drives balancing
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }


def build_dior(data: list[dict], rng: random.Random, token_px: float,
               per_image: int) -> list[dict]:
    """Affirmative detail QA (cap. 2) + the four negative types (cap. 3)."""
    out: list[dict] = []
    all_cls = sorted(PRETTY)
    for d in data:
        img, present = d["image"], d["classes"]
        names = sorted(present)
        absent = [c for c in all_cls if c not in present]
        picks: list[tuple[str, str, str]] = []   # (question, answer, kind)

        if names:
            # --- affirmative: presence, location, count, extent, co-occurrence
            c = rng.choice(names)
            pc = PRETTY[c]
            n = len(present[c])
            cwc = f"{n} {plural(pc)}" if n > 1 else f"{an(pc)} {pc}"
            picks.append((
                rng.choice(Q_PRESENCE).format(a=an(pc), c=pc, p=plural(pc)),
                rng.choice(A_YES).format(a=an(pc), c=pc, be="are" if n > 1 else "is",
                                         c_with_count=cwc),
                "presence_yes"))
            x0, y0, x1, y1 = present[c][0]
            picks.append((
                rng.choice(Q_LOCATION).format(c=pc),
                f"The {pc} is in the {quadrant((x0 + x1) / 2 / d['w'], (y0 + y1) / 2 / d['h'])} "
                f"of the image.", "location"))
            picks.append((
                rng.choice(Q_COUNT).format(p=plural(pc)),
                f"There {'are' if n > 1 else 'is'} {n} {plural(pc) if n > 1 else pc}.",
                "count"))
            big_c, big_b = max(
                ((cc, bb) for cc, bs in present.items() for bb in bs),
                key=lambda t: (t[1][2] - t[1][0]) * (t[1][3] - t[1][1]))
            picks.append((rng.choice(Q_EXTENT),
                          f"The largest object is {an(PRETTY[big_c])} {PRETTY[big_c]}, "
                          f"in the {quadrant((big_b[0]+big_b[2])/2/d['w'], (big_b[1]+big_b[3])/2/d['h'])}.",
                          "extent"))
            if len(names) > 1:
                others = [PRETTY[o] for o in names if o != c]
                picks.append((rng.choice(Q_COOCCUR).format(c=pc),
                              "Also visible: " + ", ".join(others) + ".", "cooccurrence"))

            # --- (a) category-absent on a POPULATED image: the highest-value negative
            if absent:
                ac = PRETTY[rng.choice(absent)]
                vis = ", ".join(PRETTY[o] for o in names)
                picks.append((
                    rng.choice(Q_PRESENCE).format(a=an(ac), c=ac, p=plural(ac)),
                    f"No, there {'are'} no {plural(ac)} in this image. "
                    f"What is visible: {vis}.", "absent_category"))
                # --- (c) false premise
                fc = PRETTY[rng.choice(absent)]
                picks.append((
                    rng.choice(Q_FALSE_PREMISE).format(c=fc),
                    rng.choice(A_FALSE_PREMISE).format(a=an(fc), c=fc, alt=vis),
                    "false_premise"))

            # --- (b) "none of the above". Every one of DIOR's 23,463 images has at
            # least one annotated object, so there is no empty-scene branch to take;
            # the none-of-the-above answer shape is instead taught by asking about a
            # set of categories that are all absent from a populated image.
            if len(absent) >= 4:
                sample = rng.sample(absent, 4)
                listing = ", ".join(PRETTY[s] for s in sample)
                picks.append((
                    rng.choice(Q_SCOPED_ABSENCE).format(list=listing),
                    "No, none of those are present in this image.", "scoped_absence"))

            # --- (d) not determinable. Threshold is deliberately harsh: at
            # 768px/ps2 one visual token is ~1111 source px and DIOR's MEDIAN box is
            # 952 px, so a 1-token cutoff would call over half the dataset
            # unidentifiable -- teaching refusal on objects the model can actually
            # resolve, i.e. manufacturing the mute collapse.
            small = [(cc, bb) for cc, bs in present.items() for bb in bs
                     if (bb[2] - bb[0]) * (bb[3] - bb[1]) / (token_px ** 2) < 0.2]
            if small:
                _cc, bb = rng.choice(small)
                loc = quadrant((bb[0] + bb[2]) / 2 / d["w"], (bb[1] + bb[3]) / 2 / d["h"])
                picks.append((rng.choice(Q_SMALL).format(loc=loc),
                              rng.choice(A_UNCERTAIN), "not_determinable"))

        # Emit as an affirmative/negative PAIR per image, not as a random sample.
        # The pairing is the point: with both answers riding on the same pixels, the
        # model cannot satisfy the data by reading image-level salience ("does this
        # scene look busy?") and must actually answer the question asked. Sampling
        # picks[:n] at random left only 20.6% of images carrying both polarities.
        NEG_KINDS = {"absent_category", "false_premise", "scoped_absence",
                     "not_determinable"}
        affs = [p for p in picks if p[2] not in NEG_KINDS]
        negs = [p for p in picks if p[2] in NEG_KINDS]
        rng.shuffle(affs)
        # choose the negative type by the designed mix rather than by whatever the
        # annotations yielded most of
        chosen_negs: list[tuple[str, str, str]] = []
        if negs:
            weights = [NEG_MIX.get(k, 0.0) for _q, _a, k in negs]
            if sum(weights) > 0:
                # weighted order, not just a weighted first pick: the unpaired
                # extras are numerous enough that shuffling them uniformly drags
                # the realised mix away from NEG_MIX (measured: false_premise
                # 25% against a 15% target)
                pool = list(zip(negs, weights))
                while pool:
                    items, ws = zip(*pool)
                    pick = rng.choices(items, weights=ws, k=1)[0]
                    chosen_negs.append(pick)
                    pool = [(i, w) for i, w in pool if i is not pick]

        paired: list[tuple[str, str, str, bool]] = []
        if affs and chosen_negs:
            paired.append((*affs[0], True))
            paired.append((*chosen_negs[0], True))
            extra = [(*p, False) for p in affs[1:] + chosen_negs[1:]]
            rng.shuffle(extra)
            paired.extend(extra[:max(per_image - 2, 0)])
        else:
            paired = [(*p, False) for p in (affs + chosen_negs)[:per_image]]

        for q, a, kind, is_paired in paired:
            pol = "negative" if kind in NEG_KINDS else "affirmative"
            r = rec(img, q, a, "dior", kind, pol)
            r["paired"] = is_paired
            out.append(r)
    return out


_SOURCE_BOILERPLATE = re.compile(
    r"\b(the|this)\s+(high-resolution\s+)?(satellite\s+|aerial\s+)?image[,]?\s+"
    r"(sourced\s+from|from)\s+GoogleEarth(\s+with[^,]*)?[,]?\s*", re.I)


def clean_caption(text: str) -> str:
    """Strip the 'sourced from GoogleEarth' opener carried by 73% of VRSBench captions.

    It is content-free boilerplate; trained on, it teaches the model to announce
    the imagery provider when asked to describe an AerialWaste tile.
    """
    out = _SOURCE_BOILERPLATE.sub("The image ", text)
    out = re.sub(r"\s*,?\s*sourced from GoogleEarth\s*,?\s*", " ", out, flags=re.I)
    out = re.sub(r"\bfrom GoogleEarth\b\s*", "", out, flags=re.I)
    return re.sub(r"\s{2,}", " ", out).strip()


def build_vrsbench(rng: random.Random, max_vqa: int) -> list[dict]:
    """Captions (cap. 1) + the VQA pairs verified against the image (cap. 2/3)."""
    img_root = VRS_ROOT / "images_train" / "Images_train"
    out: list[dict] = []
    ann_files = sorted(glob.glob(str(VRS_ROOT / "annotations_train/Annotations_train/*.json")))
    vqa_pool: list[dict] = []
    for f in ann_files:
        try:
            a = json.load(open(f))
        except Exception:
            continue
        img = img_root / (a.get("image") or pathlib.Path(f).stem + ".png")
        if not img.exists():
            continue
        cap = clean_caption(a.get("caption", ""))
        if cap:
            out.append(rec(str(img),
                           rng.choice(["Describe this aerial image.",
                                       "What do you see in this scene?",
                                       "Give a description of this image."]),
                           cap, "vrsbench", "caption", "affirmative"))
        for q in a.get("qa_pairs", []):
            ans = str(q.get("answer", "")).strip()
            if not ans:
                continue
            pol = "negative" if ans.lower() in {"no", "none"} else "affirmative"
            vqa_pool.append(rec(str(img), q["question"], ans, "vrsbench",
                                f"vqa_{q.get('type', 'other')}", pol))
    # keep every genuine negative, subsample the affirmatives -- shipped ratio is
    # 5.7:1 yes:no, which would reinforce exactly the bias we are removing
    neg = [r for r in vqa_pool if r["polarity"] == "negative"]
    pos = [r for r in vqa_pool if r["polarity"] == "affirmative"]
    rng.shuffle(pos)
    out.extend(neg)
    out.extend(pos[:max(max_vqa - len(neg), 0)])
    return out


def balance(records: list[dict], rng: random.Random, target: float = 0.5) -> list[dict]:
    """Trim the majority polarity to `target` share of negative ANSWERS.

    Balancing over answers, not images: most DIOR tiles are densely populated
    while VRSBench contributes affirmative-only captions, so an image-level
    balance yields a skewed answer distribution -- and a negative skew is the
    documented route to unconditional 'none' (the 819K arm's AerialWaste collapse).

    Trimming takes UNPAIRED records first. The affirmative/negative pairs on a
    single image are the structure that stops the model answering from image-level
    salience, so they are the last thing to sacrifice for a ratio.
    """
    neg = [r for r in records if r["polarity"] == "negative"]
    pos = [r for r in records if r["polarity"] == "affirmative"]
    if not neg or not pos:
        return records

    def order(rs):
        # unpaired first => they are dropped first by the truncations below
        rng.shuffle(rs)
        return sorted(rs, key=lambda r: bool(r.get("paired")))

    neg, pos = order(neg), order(pos)
    if len(neg) / (len(neg) + len(pos)) > target:
        keep = int(target / (1 - target) * len(pos))
        neg = neg[len(neg) - keep:] if keep < len(neg) else neg
    else:
        keep = int((1 - target) / target * len(neg))
        pos = pos[len(pos) - keep:] if keep < len(pos) else pos
    merged = neg + pos
    rng.shuffle(merged)
    return merged


def gates(records: list[dict]) -> tuple[bool, list[str]]:
    """The pre-training checks from SFT_DESIGN.md §8. Failing any is a hard stop."""
    msgs, ok = [], True
    n = len(records)
    neg = sum(1 for r in records if r["polarity"] == "negative")
    share = neg / max(n, 1)
    if not 0.45 <= share <= 0.55:
        ok = False
    msgs.append(f"{'PASS' if 0.45 <= share <= 0.55 else 'FAIL'}  answer polarity: "
                f"{100*share:.1f}% negative (target 45-55%)")

    answers = [r["conversations"][1]["value"] for r in records]
    one_word = sum(1 for a in answers if len(a.split()) <= 1)
    frac = one_word / max(n, 1)
    if frac > 0.35:
        ok = False
    msgs.append(f"{'PASS' if frac <= 0.35 else 'FAIL'}  answer length: "
                f"{100*frac:.1f}% single-word (cap 35%)")

    prompts = [r["conversations"][0]["value"] for r in records]
    modal = collections.Counter(prompts).most_common(1)[0][1] / max(n, 1)
    if modal > 0.15:
        ok = False
    msgs.append(f"{'PASS' if modal <= 0.15 else 'FAIL'}  prompt diversity: modal "
                f"prompt is {100*modal:.1f}% of records (cap 15%)")

    # every populated image must contribute at least one negative answer, or the
    # model can pass by keying on "does this scene look busy" alone
    by_img = collections.defaultdict(set)
    for r in records:
        if r["source"] == "dior":
            by_img[r["image"]].add(r["polarity"])
    both = sum(1 for v in by_img.values() if len(v) == 2)
    frac_both = both / max(len(by_img), 1)
    if frac_both < 0.80:
        ok = False
    msgs.append(f"{'PASS' if frac_both >= 0.80 else 'FAIL'}  DIOR images carrying "
                f"BOTH polarities: {100*frac_both:.1f}% (floor 80%) -- this is the "
                f"property that blocks the image-level shortcut")

    negk = collections.Counter(r["kind"] for r in records if r["polarity"] == "negative")
    tot_neg = sum(negk.values())
    if tot_neg:
        got = {k: negk.get(k, 0) / tot_neg for k in NEG_MIX}
        drift = max(abs(got[k] - NEG_MIX[k]) for k in NEG_MIX)
        msgs.append(f"INFO  negative-type mix (target/actual): " + ", ".join(
            f"{k} {100*NEG_MIX[k]:.0f}/{100*got[k]:.0f}" for k in NEG_MIX)
            + f"  max drift {100*drift:.0f}pp")
    return ok, msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=pathlib.Path,
                    default=pathlib.Path("/leonardo_scratch/large/userexternal/"
                                         "adiecidu/waste_vlm/data/rs_sft"))
    ap.add_argument("--img-size", type=int, default=768)
    ap.add_argument("--pixel-shuffle", type=int, default=2)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--per-image", type=int, default=4, help="max DIOR records/image")
    ap.add_argument("--max-vrs-vqa", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    # one visual token covers patch*pixel_shuffle input pixels; DIOR tiles are
    # 800px and get resized to img_size, so convert to source pixels
    token_px = args.patch * args.pixel_shuffle * (800 / args.img_size)
    print(f"[cfg] 1 visual token ~ {token_px:.1f} source px "
          f"({args.img_size}px input, ps{args.pixel_shuffle})")

    print("[dior] loading annotations", flush=True)
    dior = load_dior()
    print(f"[dior] {len(dior)} annotated images with an image file on disk")
    d_rec = build_dior(dior, rng, token_px, args.per_image)
    print(f"[dior] {len(d_rec)} records")

    print("[vrsbench] loading", flush=True)
    v_rec = build_vrsbench(rng, args.max_vrs_vqa)
    print(f"[vrsbench] {len(v_rec)} records")

    allr = d_rec + v_rec
    balanced = balance(allr, rng)
    print(f"[mix] {len(allr)} -> {len(balanced)} after polarity balancing")

    ok, msgs = gates(balanced)
    print("\n=== validation gates")
    for m in msgs:
        print("  " + m)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "rs_sft.jsonl"
    with path.open("w") as fh:
        for r in balanced:
            fh.write(json.dumps(r) + "\n")

    stats = {
        "n_records": len(balanced),
        "n_before_balance": len(allr),
        "by_source": dict(collections.Counter(r["source"] for r in balanced)),
        "by_kind": dict(collections.Counter(r["kind"] for r in balanced)),
        "by_polarity": dict(collections.Counter(r["polarity"] for r in balanced)),
        "n_images": len({r["image"] for r in balanced}),
        "gates_pass": ok,
        "gate_messages": msgs,
        "config": vars(args) | {"out": str(args.out)},
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(f"\n[write] {path}  ({len(balanced)} records over "
          f"{stats['n_images']} images)")
    print(f"[write] {args.out / 'stats.json'}")
    print("\nby kind:")
    for k, v in sorted(stats["by_kind"].items(), key=lambda t: -t[1]):
        print(f"   {k:22s} {v:7d}")
    if not ok:
        raise SystemExit("validation gates FAILED -- do not train on this mix")


if __name__ == "__main__":
    main()
