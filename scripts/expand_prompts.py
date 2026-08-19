"""Broaden the prompts for classes that are precise but blind, using the EWC codes.

The per-class table separates two failures that a recall-only view hides, and they
call for opposite fixes. Textile fires at precision 0.883 and recall 0.179;
Furniture at 0.846 and 0.094; Wood at 0.717 and 0.241. When these classes fire they
are right -- the prompt names one form of the material and the class covers many.
That is a breadth problem. Plastic packaging and Plastic, by contrast, are absorbed
by their neighbours and need separating, not widening; broadening them would make
things worse.

So this widens only the conservative ones. Each class is handed its EWC-Stat code
and official wording -- "07.6 Textiles wastes", "10.11 Household wastes" -- and the
decoder writes the forms that entry actually covers as seen from a drone: bales,
loose heaps, scattered items, the colours and arrangements each takes. The code is
the seed because the class NAME is often narrower than the class: "Wood" is 07.53
"other wood wastes", which is anything wooden that is not a pallet.

Generation is per class and cached, twenty calls total, so it composes with the
router's per-candidate-set generation rather than duplicating it.

Written and evaluated on development sites. The held-out sites stay untouched.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# classes whose failure is breadth, not confusion: high precision, low recall
CONSERVATIVE = ["Textile", "Furniture", "Wood", "Rubble", "Scrap",
                "Construction and demolition materials", "Excavation materials",
                "Mixed items", "Appliances", "Paper", "Electronic equipment",
                "Foundry", "Asphalt milling", "Vehicles", "Asbestos"]

TEMPLATE = """You are writing text prompts for a vision model that matches aerial \
drone photographs (looking straight down from about 50 metres) to categories of waste.

Category name: {name}
Official EWC-Stat entry: {code}

IMPORTANT: the category NAME can be misleading and the official entry is what counts. \
For example "Rubble" is the entry "12.61 Soils", meaning earth and soil spoil, not \
broken masonry. Write prompts for what the ENTRY describes.

The model currently recognises only one form of this material and misses the rest. \
Write 8 prompts, each under 14 words, describing how the entry's material looks from \
directly above in DIFFERENT situations. Cover a range: a compacted heap, material \
spread thinly over ground, a few isolated items, a large mound, weathered or \
discoloured material, freshly deposited material. Use different opening words for \
every prompt — do not begin them all the same way. Include ordinary synonyms for \
the material itself.

Describe only colour, texture, shape and arrangement. Never name a different kind \
of waste.

Reply with ONLY a JSON list of 8 strings, no other text."""


BAD_WORDS = None   # filled from the class list at run time


def sane(prompt: str, own: str, others) -> bool:
    """Drop a prompt that names someone else's material.

    The first run produced "Baled fabric scraps form neat rectangles" for Rubble,
    which is 12.61 Soils. A generated prompt is unverified text and a hallucinated
    one does not merely fail to help, it actively pulls the class toward its
    neighbour -- which is the failure this whole expansion is meant to repair.
    """
    low = prompt.lower()
    own_l = own.lower()
    for o in others:
        ol = o.lower()
        if ol == own_l:
            continue
        head = ol.split()[0]
        if len(head) > 3 and head in low and head not in own_l:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder",
                    default="/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/weights/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="dronewaste")
    ap.add_argument("--only-conservative", action="store_true", default=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import re

    from scripts.roi_material import _ewc_descriptions, cue_prompts, load_rois
    from scripts.two_stage_router import load_decoder

    cats, _rois = load_rois(args.dataset, "test")
    ewc = _ewc_descriptions(args.dataset)
    base = cue_prompts(args.dataset, cats)
    gen = load_decoder(args.decoder)

    out = {}
    targets = [c for c in cats if (c in CONSERVATIVE or not args.only_conservative)]
    print(f"[expand] {len(targets)} of {len(cats)} classes widened "
          f"(the collapsed ones are left alone; widening them makes them worse)")
    for c in targets:
        code = ewc.get(c) or c
        got = []
        for attempt in range(3):     # 5 of 15 failed to parse on a single try
            reply = gen(TEMPLATE.format(name=c, code=f"{code}"), max_new_tokens=520)
            m = re.search(r"\[.*?\]", reply, re.S)
            if m:
                try:
                    got = [str(x).strip() for x in json.loads(m.group(0)) if str(x).strip()]
                except json.JSONDecodeError:
                    got = []
            if not got:
                continue
            kept = [g for g in got if sane(g, c, cats)]
            if len(kept) < len(got):
                print(f"   {c:38s} dropped {len(got)-len(kept)} naming another class")
            got = kept
            if got:
                break
        if not got:   # a class that fails keeps its existing prompts rather than losing them
            print(f"   {c:38s} no usable output, keeping base")
            continue
        out[c] = got
        print(f"   {c:38s} +{len(got)}  e.g. {got[0][:58]!r}")

    merged = {c: list(base.get(c, [])) + out.get(c, []) for c in cats}
    pathlib.Path(args.out).write_text(json.dumps(merged, indent=2))
    print(f"\n[write] {args.out}  ({sum(len(v) for v in merged.values())} prompts total)")


if __name__ == "__main__":
    main()
