"""Canonical waste-type label space for VQA generation.

Union of DroneWaste paper-10 + AerialWaste m2 taxonomies, preserved verbatim
without merging. The two datasets use different vocabularies (DW is finer, AW
is coarser) and merging would inject judgment calls that risk obscuring real
distinctions. Each label is verifiable only on its source dataset; type-ID
questions for a given image draw from that dataset's verifiable subset so we
never ask "is asbestos visible?" on an AerialWaste image whose GT cannot
confirm or deny it.

Cross-dataset semantic overlaps are documented (not auto-merged) for later
benchmark notes.
"""
from __future__ import annotations

# DroneWaste paper-10 vocabulary (10 fine-grained classes).
DW_LABELS: list[str] = [
    "Mixed items",
    "Scrap",
    "Construction and demolition materials",
    "Pallets",
    "Plastic packaging",
    "Asbestos",
    "Textile",
    "Vehicles",
    "Tyres",
    "Metal barrels",
]

# AerialWaste MCML m2 vocabulary (5 coarser classes).
AW_M2_LABELS: list[str] = [
    "Bulky items",
    "Unknown material",
    "Containers",
    "Rubble",
    "Plastic",
]

# Union: 15 distinct strings; sort-by-source order for stability.
UNION: list[str] = DW_LABELS + AW_M2_LABELS
assert len(UNION) == len(set(UNION)), "label collision between DW and AW vocabularies"

# Which labels each dataset's GT can confirm/deny. Type-ID question generation
# MUST restrict the candidate label set to its source dataset to avoid
# spurious false negatives.
VERIFIABLE_BY_DATASET: dict[str, set[str]] = {
    "dronewaste_paper10": set(DW_LABELS),
    "aerialwaste_m2": set(AW_M2_LABELS),
}

# Documented soft overlaps — informational only, NOT used to rewrite labels.
# Useful for later analysis ("the model said 'Plastic' on a DW image whose GT
# is 'Plastic packaging'" should not be scored as a hard miss).
SEMANTIC_OVERLAPS: dict[str, list[str]] = {
    "Plastic packaging": ["Plastic"],
    "Plastic": ["Plastic packaging"],
    "Construction and demolition materials": ["Rubble"],
    "Rubble": ["Construction and demolition materials"],
    "Metal barrels": ["Containers"],
    "Containers": ["Metal barrels"],
}


def verifiable(dataset: str) -> list[str]:
    """Sorted list of labels whose presence/absence is verifiable from `dataset`'s GT."""
    return sorted(VERIFIABLE_BY_DATASET[dataset])
