"""Text prompt variants for the open-vocabulary naming arm.

The zero-shot confusion is not blindness, it is sibling collision inside the
EWC-Stat parents. Plastic packaging (07.41) goes to Plastic (07.42) 48% of the
time and is itself predicted 27 times in 2090; Excavation materials (12.31) goes
to Rubble (12.61) and C&D (12.11) in 88% of cases; Wood (07.53) goes to Pallets
(07.51), which is literally "wood packaging". The encoder is finding the object
and the vocabulary cannot separate two names for it.

Several class names are also misleading read literally, which no amount of
ensembling fixes: "Rubble" is 12.61 *soils*, not broken masonry, and "Foundry" is
12.42 slags and ashes. A prompt built from the name asks for the wrong thing.

`contrastive` therefore says what each class is AND what it is not, for the groups
that actually collide. This is prompt engineering informed by a confusion matrix,
which is legitimate exactly as long as the confusion matrix came from development
data -- see `--dev-sites`. Tuning these on the held-out sites and then reporting
on them would be fitting the evaluation set, which is the failure mode this
pipeline's whole no-fixed-label-vector premise is supposed to avoid.
"""
from __future__ import annotations

# What each class is, and the sibling it is being confused with.
CONTRASTIVE = {
    "Plastic packaging": [
        "baled and wrapped plastic packaging waste",
        "stacked bales of plastic film and bags",
        "plastic wrapping and packaging material bundled together",
        "white and coloured plastic bales, not loose plastic debris",
    ],
    "Plastic": [
        "loose plastic waste scattered on the ground",
        "broken plastic pipes, sheets and containers",
        "plastic debris, not baled packaging",
    ],
    "Pallets": [
        "stacked wooden pallets",
        "a stack of wood packaging pallets in rows",
        "regular wooden pallets, not loose timber",
    ],
    "Wood": [
        "loose timber, planks and branches",
        "a heap of scrap wood and offcuts",
        "waste wood, not stacked pallets",
    ],
    "Rubble": [
        "bare soil and earth spoil",
        "a mound of excavated soil",
        "soil waste, not concrete or brick",
    ],
    "Construction and demolition materials": [
        "broken concrete, bricks and gypsum from demolition",
        "demolition rubble of concrete and masonry",
        "concrete and brick waste, not bare soil",
    ],
    "Excavation materials": [
        "naturally occurring mineral spoil from excavation",
        "quarried stone and gravel heaps",
        "mineral excavation spoil, not demolition debris",
    ],
    "Furniture": [
        "discarded household furniture dumped outdoors",
        "old sofas, mattresses, chairs and cabinets",
        "bulky household furniture, not an undifferentiated mixed heap",
    ],
    "Mixed items": [
        "an undifferentiated heap of mixed rubbish",
        "a jumbled pile of assorted waste of no single type",
        "mixed unsorted waste, not one identifiable material",
    ],
    "Textile": [
        "discarded fabric, clothing and textile waste",
        "bales and heaps of cloth",
        "textile waste, not plastic sheeting",
    ],
}


def build(cats, base: dict, variant: str = "base") -> dict:
    """-> {class: [prompt, ...]}. `base` is the generic/EWC set already in use."""
    if variant == "base":
        return base
    if variant != "contrastive":
        raise ValueError(variant)
    out = {}
    for c in cats:
        # keep the generic and EWC phrasings; the contrastive lines are additions,
        # so a class with no listed sibling is unchanged rather than degraded
        out[c] = list(base.get(c, [])) + CONTRASTIVE.get(c, [])
    return out
