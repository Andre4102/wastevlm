#!/usr/bin/env python
"""Phase-0: template the external waste datasets (TrashBox, TACO, UAVVaste, SWAD)
into LLaVA single-turn VQA records, then merge into waste_sft/train.json.

Design (see PLAN.md Phase 0 item 4):
- viewpoint curriculum: ground-level (TrashBox, TACO) -> drone (UAVVaste) -> satellite (SWAD)
- >=5 phrasings per question type (avoid template overfitting)
- balanced yes/no within each source (via true-vs-wrong queried class, or
  positive-vs-negative framing on all-positive sources)
- every record carries a `source` tag for Phase-3 dynamic batch loading
- leakage-clean: DroneWaste/AerialWaste never appear (verified separately)

Record schema matches existing train.json:
  {"id", "image": <abs path>, "source", "conversations":[human,gpt]}
"""
import os, json, glob, random, hashlib

DATA = "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data"
EXT = f"{DATA}/external"
SFT = f"{DATA}/waste_sft"
SEED = 1234
rng = random.Random(SEED)

IMG_EXT = (".jpg", ".jpeg", ".png")

# ── phrasing banks ────────────────────────────────────────────────────────
Q_MATERIAL_OPEN = [
    "What material is the discarded object in this image?",
    "Identify the primary material of the waste item shown.",
    "This is a photo of a piece of trash. What is it made of?",
    "What kind of material is this piece of waste?",
    "Classify the material of the object in the picture.",
    "Which material best describes this discarded item?",
]
Q_MATERIAL_BIN = [
    "Is the main material of this waste item {m}?",
    "Is the object in this image made of {m}?",
    "Would you classify this piece of trash as {m}?",
    "Does this discarded item consist mainly of {m}?",
    "Is this {m} waste?",
]
Q_TACO_DESC = [
    "Describe the litter visible in this image and its surroundings.",
    "What waste items can you see in this scene, and where are they?",
    "Summarize the trash present in this photograph.",
    "What kinds of discarded objects appear in this image?",
    "Describe any waste in the picture and the setting it is in.",
]
Q_TACO_BIN = [
    "Is there a {c} visible in this image?",
    "Can you see any {c} among the litter here?",
    "Does this scene contain a {c}?",
    "Is a {c} present in the picture?",
    "Would you say there is a {c} in this image?",
]
Q_DRONE_PRESENCE = [  # location-style (open) — keep answers off yes/no to preserve balance
    "Where in this aerial drone image is the litter located?",
    "Which area of this drone photograph contains the scattered waste?",
    "Point out the rough location of the rubbish in this overhead view.",
    "In which part of this aerial shot can the litter be seen?",
    "Whereabouts in this drone image does the discarded waste appear?",
]
Q_DRONE_POS = [  # answer yes
    "Is there any litter in this drone image?",
    "Does this aerial view contain visible rubbish?",
    "Can any waste be seen from this overhead shot?",
    "Is discarded trash present in this drone photograph?",
    "Are there signs of litter in this aerial image?",
]
Q_DRONE_NEG = [  # answer no
    "Is this drone image completely free of any litter?",
    "Is the area in this aerial shot clean, with no rubbish at all?",
    "Would you say there is no waste whatsoever in this overhead view?",
    "Is this drone photograph clear of any discarded trash?",
    "Is the ground in this aerial image entirely litter-free?",
]
Q_SAT_POS = [  # answer yes
    "Does this satellite image show signs of solid waste?",
    "Is there any solid-waste dumping visible in this satellite tile?",
    "From this overhead satellite view, is waste material present?",
    "Can solid waste be identified in this satellite image?",
    "Does this remote-sensing tile contain a waste site?",
]
Q_SAT_NEG = [  # answer no
    "Is this satellite tile free of any solid waste?",
    "Would you say this satellite scene shows no waste at all?",
    "Is the area in this satellite image clean, without any dumping?",
    "Is this remote-sensing tile clear of solid-waste sites?",
    "Does this satellite view lack any visible waste material?",
]
Q_SAT_LOC = [
    "Where in this satellite image is the solid waste located?",
    "Which part of this satellite tile shows waste material?",
    "Point out the rough location of the waste in this satellite view.",
    "In which region of this overhead image does the dumping appear?",
    "Whereabouts in this satellite tile can waste be seen?",
]

TRASHBOX_READABLE = {
    "cardboard": "cardboard", "e-waste": "electronic waste (e-waste)",
    "glass": "glass", "medical": "medical waste", "metal": "metal",
    "paper": "paper", "plastic": "plastic",
}

def _id(source, key):
    h = hashlib.md5(f"{source}:{key}".encode()).hexdigest()[:10]
    return f"{source}__{h}"

def rec(source, img, q, a):
    return {"id": _id(source, img + q), "image": img, "source": source,
            "conversations": [{"from": "human", "value": "<image>\n" + q},
                              {"from": "gpt", "value": a}]}

def quadrants(centroids):
    """centroids: list of (cx,cy) normalized -> human region phrase."""
    regs = set()
    for cx, cy in centroids:
        v = "top" if cy < 0.4 else ("bottom" if cy > 0.6 else "middle")
        h = "left" if cx < 0.4 else ("right" if cx > 0.6 else "centre")
        regs.add(f"{v}-{h}" if not (v == "middle" and h == "centre") else "centre")
    order = ["top-left", "top-centre", "top-right", "middle-left", "centre",
             "middle-right", "bottom-left", "bottom-centre", "bottom-right"]
    regs = [r for r in order if r in regs]
    if not regs:
        return "the image"
    if len(regs) == 1:
        return f"the {regs[0]}"
    return "the " + ", ".join(regs[:-1]) + " and " + regs[-1] + " areas"

# ── builders ──────────────────────────────────────────────────────────────
def build_trashbox():
    root = f"{EXT}/TrashBox/TrashBox_train_dataset_subfolders"
    classes = sorted(d for d in os.listdir(root) if os.path.isdir(f"{root}/{d}"))
    out = []
    for cls in classes:
        readable = TRASHBOX_READABLE.get(cls, cls)
        imgs = [p for p in glob.glob(f"{root}/{cls}/**/*", recursive=True)
                if p.lower().endswith(IMG_EXT)]
        for p in imgs:
            # open material-ID
            q = rng.choice(Q_MATERIAL_OPEN)
            out.append(rec("trashbox", p, q, f"This is {readable}."))
            # balanced binary
            if rng.random() < 0.5:
                m = readable
                a = f"Yes, the item is {readable}."
            else:
                wrong = rng.choice([c for c in classes if c != cls])
                m = TRASHBOX_READABLE.get(wrong, wrong)
                a = f"No, it is {readable}, not {m}."
            q = rng.choice(Q_MATERIAL_BIN).format(m=m)
            out.append(rec("trashbox", p, q, a))
    return out

def build_taco():
    d = json.load(open(f"{EXT}/TACO/data/annotations.json"))
    cats = {c["id"]: c.get("supercategory") or c["name"] for c in d["categories"]}
    all_sup = sorted(set(cats.values()))
    imgmeta = {im["id"]: im for im in d["images"]}
    from collections import defaultdict
    present = defaultdict(set)
    for a in d["annotations"]:
        present[a["image_id"]].add(cats[a["category_id"]])
    out = []
    for iid, sup in present.items():
        fn = imgmeta[iid]["file_name"]
        p = f"{EXT}/TACO/data/images/{fn}"
        if not os.path.isfile(p):
            continue
        types = sorted(sup)
        # describe (open)
        q = rng.choice(Q_TACO_DESC)
        listing = types[0].lower() if len(types) == 1 else \
            ", ".join(t.lower() for t in types[:-1]) + " and " + types[-1].lower()
        out.append(rec("taco", p, q, f"The visible litter includes {listing}."))
        # balanced binary
        if rng.random() < 0.5 or len(all_sup) == len(sup):
            c = rng.choice(types)
            a = f"Yes, there is {c.lower()} visible."
        else:
            absent = [s for s in all_sup if s not in sup]
            c = rng.choice(absent)
            a = f"No, there is no {c.lower()} visible in this image."
        q = rng.choice(Q_TACO_BIN).format(c=c.lower())
        out.append(rec("taco", p, q, a))
    return out

def build_uavvaste():
    d = json.load(open(f"{EXT}/UAVVaste/annotations/annotations.json"))
    imgmeta = {im["id"]: im for im in d["images"]}
    from collections import defaultdict
    boxes = defaultdict(list)
    for a in d["annotations"]:
        boxes[a["image_id"]].append(a["bbox"])  # [x,y,w,h] absolute
    out = []
    for iid, meta in imgmeta.items():
        p = f"{EXT}/UAVVaste/images/{meta['file_name']}"
        if not os.path.isfile(p):
            continue
        W, H = meta["width"], meta["height"]
        cents = [((x + w / 2) / W, (y + h / 2) / H) for x, y, w, h in boxes.get(iid, [])]
        loc = quadrants(cents)
        n = len(cents)
        # location (open) — descriptive answer, not yes/no, to keep balance
        q = rng.choice(Q_DRONE_PRESENCE)
        piece = "A piece of litter" if n == 1 else f"{n} pieces of litter"
        out.append(rec("uavvaste", p, q,
                       f"{piece} can be seen in {loc}."))
        # balanced binary via framing
        if rng.random() < 0.5:
            q = rng.choice(Q_DRONE_POS)
            a = "Yes, there is litter visible in this aerial image."
        else:
            q = rng.choice(Q_DRONE_NEG)
            a = "No, litter is in fact present in this aerial image."
        out.append(rec("uavvaste", p, q, a))
    return out

def build_swad():
    out = []
    for split in ("train", "val", "test"):
        idir = f"{EXT}/swad/images/{split}"
        ldir = f"{EXT}/swad/labels/{split}"
        for ip in glob.glob(f"{idir}/*"):
            if not ip.lower().endswith(IMG_EXT):
                continue
            lp = f"{ldir}/{os.path.splitext(os.path.basename(ip))[0]}.txt"
            cents = []
            if os.path.isfile(lp):
                for line in open(lp):
                    parts = line.split()
                    if len(parts) >= 5:
                        cents.append((float(parts[1]), float(parts[2])))
            loc = quadrants(cents)
            # scene-level yes/no via framing (all tiles positive)
            if rng.random() < 0.5:
                q = rng.choice(Q_SAT_POS)
                a = f"Yes, solid waste is visible, in {loc}."
            else:
                q = rng.choice(Q_SAT_NEG)
                a = f"No, solid waste is in fact present, in {loc}."
            out.append(rec("swad", ip, q, a))
            # location (open)
            q = rng.choice(Q_SAT_LOC)
            a = f"The solid waste appears in {loc}."
            out.append(rec("swad", ip, q, a))
    return out

def yn_balance(recs, source):
    yes = sum(1 for r in recs if r["conversations"][1]["value"].lower().startswith("yes"))
    no = sum(1 for r in recs if r["conversations"][1]["value"].lower().startswith("no"))
    return f"{source}: {len(recs)} recs (yes={yes}, no={no}, other={len(recs)-yes-no})"

def main():
    builders = {"trashbox": build_trashbox, "taco": build_taco,
                "uavvaste": build_uavvaste, "swad": build_swad}
    external, summary = [], []
    for name, fn in builders.items():
        recs = fn()
        external += recs
        summary.append(yn_balance(recs, name))
        print("[build]", summary[-1], flush=True)
    json.dump(external, open(f"{SFT}/external_vqa.json", "w"))
    print(f"[write] external_vqa.json: {len(external)} records", flush=True)

    # merge into train.json. Read from a stable base snapshot so re-runs are
    # idempotent (never double-append external). Base = wastebench + anchors.
    base = f"{SFT}/train_base.json"
    if not os.path.isfile(base):
        json.dump(json.load(open(f"{SFT}/train.json")), open(base, "w"))
    existing = json.load(open(base))
    merged = existing + external
    rng.shuffle(merged)
    json.dump(merged, open(f"{SFT}/train.json", "w"))
    print(f"[merge] train.json: {len(existing)} existing + {len(external)} external "
          f"= {len(merged)} total", flush=True)

    # per-source tally of final train.json
    from collections import Counter
    c = Counter(r.get("source", "?") for r in merged)
    print("[final] per-source:", dict(c), flush=True)
    # write a machine-readable build stat
    stat = {"external_records": len(external), "train_total": len(merged),
            "per_source": dict(c), "yn_balance": summary, "seed": SEED}
    json.dump(stat, open(f"{SFT}/external_build_stat.json", "w"), indent=2)

if __name__ == "__main__":
    main()
