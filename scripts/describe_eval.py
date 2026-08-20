"""Template vs LLM: does composing the answer with a model beat filling a slot?

Both arms are handed the SAME structured pipeline output (a scene graph) and must
turn it into a description for a user. The template is the floor -- it can only
restate the graph, so it is 0% unsupported by construction. The question is
whether the LLM's fluency and aggregation are worth the claims it invents.

Run on GROUND-TRUTH scene graphs, so anything outside the graph is unsupported by
construction and this measures rendering, not perception.

Metrics are programmatic, not judged:
  unsupported_cat   description names a waste category the graph does not contain
  coverage          fraction of the graph's categories that the description names
  count_error       the stated total object count disagrees with the graph
  region_unsupported  a location word with no object in that region
  parse_rate        non-empty output

The empty-graph control is the sharpest cell: hand each arm a scene with no waste
and count how often it invents some. The template scores 0 by construction; the
VLM's own false-mention rate on negatives is 32.6%, so there is a real number to
beat.
"""
from __future__ import annotations

import argparse, json, pathlib, random, re, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.scene_render import render, graph_facts, region  # noqa: E402

SYN = {
    "Plastic packaging": ["plastic packaging", "packaging", "wrapping"],
    "Plastic": ["plastic"],
    "Mixed items": ["mixed", "assorted", "miscellaneous"],
    "Construction and demolition materials": ["construction and demolition", "demolition", "c&d"],
    "Excavation materials": ["excavation", "soil", "earth"],
    "Metal barrels": ["barrel", "drum"],
    "Tyres": ["tyre", "tire"],
    "Vehicles": ["vehicle", "car", "truck"],
    "Appliances": ["appliance", "fridge", "washing machine"],
    "Electronic equipment": ["electronic", "e-waste"],
    "Furniture": ["furniture", "sofa", "chair", "mattress"],
    "Wood": ["wood", "timber", "lumber"],
    "Paper": ["paper", "cardboard"],
    "Textile": ["textile", "fabric", "cloth"],
    "Scrap": ["scrap"],
    "Rubble": ["rubble"],
    "Asbestos": ["asbestos", "corrugated sheet"],
    "Asphalt milling": ["asphalt"],
    "Foundry": ["foundry", "slag", "ash"],
    "Pallets": ["pallet"],
}
WASTE_WORDS = ["waste", "garbage", "debris", "dump", "litter", "rubbish", "trash", "landfill"]
REGIONS = ["top-left", "top-centre", "top-right", "bottom-left", "bottom-centre",
           "bottom-right", "left", "right", "centre"]

PROMPT = (
    "You are describing an aerial image for a waste-inspection report. "
    "A detector has produced the following structured findings:\n\n{graph}\n\n"
    "Write one or two sentences describing what is in the image, for the "
    "inspector. Use only the findings above. Do not add anything not listed."
)


def graph_to_text(scene: dict) -> str:
    w, h = scene.get("size", [640, 640])
    objs = scene.get("objs", [])
    if not objs:
        return "detections: none"
    lines = []
    for o in objs[:40]:
        c = o.get("category") or o.get("cat")
        lines.append(f"- {c}: at {region(o['cx'], o['cy'], w, h)}, "
                     f"area {o['area']/(w*h)*100:.1f}% of image")
    return f"image {w}x{h}, {len(objs)} detections\n" + "\n".join(lines)


def mentioned_cats(text: str) -> set:
    """Longest surface form wins, and its span is consumed.

    Plain substring tests double-count: "plastic packaging" also contains
    "plastic", so a correct description of Plastic packaging was being scored as
    also claiming Plastic -- which made the template, which cannot invent
    anything, look 16% unsupported. Match longest-first and blank out each hit.
    """
    t = text.lower()
    pairs = sorted(((s, cat) for cat, syns in SYN.items()
                    for s in list(syns) + [cat.lower()]),
                   key=lambda kv: -len(kv[0]))
    out = set()
    for surface, cat in pairs:
        pat = r"\b" + re.escape(surface) + r"\b"
        if re.search(pat, t):
            out.add(cat)
            t = re.sub(pat, lambda m: " " * (m.end() - m.start()), t)
    return out


NEG = ("no ", "not ", "none", "without", "free of", "absence", "n't", "cannot")
COUNT_NOUNS = r"(?:areas?|objects?|piles?|items?|regions?|detections?|instances?|sites?)"
WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _count_claim(text: str):
    """An explicit object count, or None if the arm never states one.

    Taking the first integer in the string counted "35" out of "35.3% of the
    image" as a claim that there are 35 objects, which scored the LLM arms at an
    85% count error when in fact they mostly state no count at all and describe
    area instead. Only a number bound to a counting noun is a count.
    """
    t = text.lower()
    m = re.search(r"\b(\d+)\s+(?:\w+\s+){0,2}?" + COUNT_NOUNS + r"\b", t)
    if m and "%" not in t[m.start():m.end()]:
        return int(m.group(1))
    m = re.search(r"\b(" + "|".join(WORD_NUM) + r")\s+(?:\w+\s+){0,2}?" + COUNT_NOUNS + r"\b", t)
    if m:
        return WORD_NUM[m.group(1)]
    return None


def _asserts_waste(text: str, said_cats: set) -> int:
    if said_cats:
        return 1
    for sent in re.split(r"[.;]", text.lower()):
        if any(w in sent for w in WASTE_WORDS) and not any(n in sent for n in NEG):
            return 1
    return 0


def _regions_in(text: str) -> set:
    """Compound regions consume their span before the bare ones are tested, so
    "bottom-left" does not also register as "left"."""
    t = text.lower()
    out = set()
    for r in sorted(REGIONS, key=lambda r: -len(r)):
        pat = r"\b" + re.escape(r) + r"\b"
        if re.search(pat, t):
            out.add(r)
            # blank EVERY occurrence: one consumed span still left a second
            # "bottom-right" in the text for the bare "right" test to hit
            t = re.sub(pat, lambda m: " " * (m.end() - m.start()), t)
    return out


def score_one(text: str, facts: dict) -> dict:
    t = (text or "").strip()
    if not t:
        return {"parse": 0}
    said = mentioned_cats(t)
    have = facts["categories"]
    r_said = _regions_in(t)
    cnt = _count_claim(t)
    return {
        "parse": 1,
        "unsupported_cat": int(bool(said - have)),
        "n_unsupported": len(said - have),
        "coverage": (len(said & have) / len(have)) if have else float("nan"),
        "states_count": int(cnt is not None),
        "count_error": (int(cnt != facts["total"]) if cnt is not None else np.nan),
        "region_unsupported": int(bool(r_said - facts["regions"] - {"centre"})),
        # The empty-graph metric that matters is whether the arm ASSERTS waste,
        # not whether the string "waste" occurs -- "No waste is visible" contains
        # it. Count an assertion when a category is named, or a waste word appears
        # with no negation cue in the sentence carrying it.
        "asserts_waste": _asserts_waste(t, said),
        "len_words": len(t.split()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["template", "qwen-instruct", "ckpt-text"])
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--n-empty", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", default=None)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    R = pathlib.Path("/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/results")
    scenes = json.loads(pathlib.Path(args.scenes or (R / "scenes_gt_dev.json")).read_text())
    rng = random.Random(args.seed)
    pos = [s for s in scenes if s.get("objs")]
    rng.shuffle(pos)
    pos = pos[: args.n]
    empty = [{"image": f"synthetic_empty_{i}", "size": [640, 640], "objs": []}
             for i in range(args.n_empty)]
    work = [("pos", s) for s in pos] + [("empty", s) for s in empty]
    print(f"[data] {len(pos)} populated + {len(empty)} empty graphs, arm={args.arm}")

    gen = None
    if args.arm != "template":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        path = args.llm
        print(f"[model] {path}", flush=True)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, device_map="cuda").eval()

        @torch.no_grad()
        def gen(prompt: str) -> str:
            msgs = [{"role": "user", "content": prompt}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(text, return_tensors="pt").to(model.device)
            out = model.generate(**ids, max_new_tokens=96, do_sample=False)
            return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    rows = {"pos": [], "empty": []}
    dumps = []
    for i, (kind, s) in enumerate(work):
        facts = graph_facts(s)
        txt = render(s) if args.arm == "template" else gen(PROMPT.format(graph=graph_to_text(s)))
        rows[kind].append(score_one(txt, facts))
        dumps.append({"kind": kind, "image": s.get("image"), "text": txt})
        if i and i % 50 == 0:
            print(f"  {i}/{len(work)}", flush=True)

    def agg(rs):
        if not rs: return {}
        ok = [r for r in rs if r.get("parse")]
        f = lambda k: float(np.nanmean([r[k] for r in ok])) if ok else float("nan")
        return {"n": len(rs), "parse_rate": float(np.mean([r["parse"] for r in rs])),
                "unsupported_cat": f("unsupported_cat"), "n_unsupported": f("n_unsupported"),
                "coverage": f("coverage"), "states_count": f("states_count"), "count_error": f("count_error"),
                "region_unsupported": f("region_unsupported"),
                "asserts_waste": f("asserts_waste"), "len_words": f("len_words")}

    rep = {"arm": args.arm, "populated": agg(rows["pos"]), "empty": agg(rows["empty"]),
           "generations": dumps}
    print(f"\n=== arm={args.arm}")
    for k in ("populated", "empty"):
        a = rep[k]
        if not a: continue
        print(f"  {k:10s} n={a['n']:4d} parse {a['parse_rate']:.3f}  "
              f"unsupported-cat {a['unsupported_cat']:.3f} ({a['n_unsupported']:.2f}/desc)  "
              f"coverage {a['coverage']:.3f}  states-cnt {a['states_count']:.3f}  "
              f"cnt-err {a['count_error']:.3f}  "
              f"region-unsup {a['region_unsupported']:.3f}  "
              f"asserts-waste {a['asserts_waste']:.3f}  {a['len_words']:.0f}w")
    print("\n  samples:")
    for d in dumps[:4]:
        print(f"   [{d['kind']}] {d['text'][:200]}")
    if args.out_json:
        pathlib.Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out_json).write_text(json.dumps(rep, indent=2))
        print(f"\n[write] {args.out_json}")


if __name__ == "__main__":
    main()
