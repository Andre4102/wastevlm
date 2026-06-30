"""Dataset loaders for AerialWaste v2 and DroneWaste.

Yields a uniform `Sample` record so downstream code is dataset-agnostic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class Sample:
    dataset: str
    image_id: str
    image_path: Path
    label: int
    width: int
    height: int
    image_source: str
    extra: dict = field(default_factory=dict)


def _aerialwaste_image_source(width: int, height: int) -> str:
    """Map image dimensions to the three AerialWaste sources.

    The dataset's JSON has no explicit source field, but dimensions cluster:
    1000x1000 -> AGEA orthophoto; ~700x700 -> Google Earth crop;
    ~1040-1060 -> WorldView-3 reprojection.
    """
    if width == 1000 and height == 1000:
        return "agea"
    if width < 800:
        return "google_earth"
    return "worldview3"


def _coerce_int(value, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_aerialwaste(root: str | Path, split: str = "testing") -> list[Sample]:
    root = Path(root)
    json_path = root / f"{split}.json"
    with json_path.open() as f:
        data = json.load(f)

    images_dir = root / "images"
    samples: list[Sample] = []
    for img in data["images"]:
        width = _coerce_int(img["width"])
        height = _coerce_int(img["height"])
        label = _coerce_int(img["is_candidate_location"])
        image_path = images_dir / img["file_name"]
        samples.append(
            Sample(
                dataset="aerialwaste",
                image_id=str(img["id"]),
                image_path=image_path,
                label=label,
                width=width,
                height=height,
                image_source=_aerialwaste_image_source(width, height),
                extra={
                    "severity": _coerce_int(img.get("severity")),
                    "evidence": _coerce_int(img.get("evidence")),
                    "site_type": img.get("site_type"),
                    "valid_fine_grain": _coerce_int(img.get("valid_fine_grain")),
                    "split": split,
                },
            )
        )
    return samples


def load_dronewaste(root: str | Path) -> list[Sample]:
    root = Path(root)
    with (root / "dronewaste_v1.0.json").open() as f:
        data = json.load(f)

    has_ann: dict[int, list[dict]] = {}
    for ann in data.get("annotations", []):
        has_ann.setdefault(ann["image_id"], []).append(ann)

    cat_lookup = {c["id"]: c for c in data.get("categories", [])}
    images_dir = root / "images"

    samples: list[Sample] = []
    for img in data["images"]:
        anns = has_ann.get(img["id"], [])
        label = 1 if anns else 0
        cats_present = sorted({cat_lookup[a["category_id"]]["name"] for a in anns})
        samples.append(
            Sample(
                dataset="dronewaste",
                image_id=str(img["id"]),
                image_path=images_dir / img["file_name"],
                label=label,
                width=img["width"],
                height=img["height"],
                image_source=img["site"],
                extra={
                    "site": img["site"],
                    "categories_present": cats_present,
                    "n_annotations": len(anns),
                },
            )
        )
    return samples


def iter_samples(samples: list[Sample]) -> Iterator[Sample]:
    yield from samples


# ---------------------------------------------------------------------------
# Multi-label loaders: return (categories_list, samples) where each sample's
# `extra["gt_categories"]` is a set[str] of canonical category names present.
# ---------------------------------------------------------------------------


def load_aerialwaste_multilabel(
    root: str | Path, split: str = "testing"
) -> tuple[list[str], list[Sample]]:
    """Multi-label-eval-ready samples from AerialWaste.

    Includes:
      * all images with valid fine-grain annotations (`valid_fine_grain=1`)
        with GT = set of category names present,
      * all explicit negatives (`is_candidate_location=0`) with GT = ∅.

    Excludes the candidates without fine-grain annotations — they're
    unverified positives and would inject label noise.

    Multi-label GT is stored differently in the two splits: training has a
    per-image `categories` list of category IDs; testing has both that AND a
    COCO-style `annotations` array. We union both sources for safety.
    """
    root = Path(root)
    with (root / f"{split}.json").open() as f:
        data = json.load(f)

    cat_by_id = {c["id"]: c["name"] for c in data["categories"]}
    categories = [c["name"] for c in data["categories"]]

    img_to_cats: dict[int, set[str]] = {}
    # COCO-annotations path (exists in testing split)
    for ann in data.get("annotations", []):
        img_to_cats.setdefault(ann["image_id"], set()).add(cat_by_id[ann["category_id"]])
    # Per-image `categories` list path (exists in both splits)
    for img in data["images"]:
        cat_field = img.get("categories")
        if not cat_field:
            continue
        ids: list[int] = []
        if isinstance(cat_field, list):
            ids = [int(x) for x in cat_field]
        elif isinstance(cat_field, str) and cat_field not in ("", "[]"):
            try:
                parsed = json.loads(cat_field)
                if isinstance(parsed, list):
                    ids = [int(x) for x in parsed]
            except json.JSONDecodeError:
                pass
        for cid in ids:
            if cid in cat_by_id:
                img_to_cats.setdefault(int(img["id"]), set()).add(cat_by_id[cid])

    images_dir = root / "images"
    samples: list[Sample] = []
    for img in data["images"]:
        valid_fg = _coerce_int(img.get("valid_fine_grain"))
        is_cand = _coerce_int(img.get("is_candidate_location"))
        width = _coerce_int(img["width"])
        height = _coerce_int(img["height"])

        if valid_fg == 1:
            label = 1
            gt = img_to_cats.get(int(img["id"]), set())
        elif is_cand == 0:
            label = 0
            gt = set()
        else:
            continue

        samples.append(
            Sample(
                dataset="aerialwaste",
                image_id=str(img["id"]),
                image_path=images_dir / img["file_name"],
                label=label,
                width=width,
                height=height,
                image_source=_aerialwaste_image_source(width, height),
                extra={
                    "gt_categories": sorted(gt),
                    "split": split,
                    "valid_fine_grain": valid_fg,
                },
            )
        )
    return categories, samples


# The DroneWaste paper (Mora et al.) reports detection mAP on these 10 classes.
DRONEWASTE_PAPER_10 = [
    "Construction and demolition materials",
    "Metal barrels",
    "Plastic packaging",
    "Pallets",
    "Scrap",
    "Vehicles",
    "Tyres",
    "Asbestos",
    "Textile",
    "Mixed items",
]


# Mapping versions distributed by the AW team for multi-class multi-label
# (MCML) evaluation. m2 is the 5-class version used in the paper; m4 is a
# 6-class variant for ablations.
AERIALWASTE_MCML_VERSIONS = {
    "m2": "mcml_split_dataset_1",  # 5 classes: Rubble, Bulky items, Plastic, Containers, Unknown material
    "m4": "mcml_split_dataset_2",  # 6 classes: Rubble/excavated, Scrap, Sludge, Wood, Tires, Other waste
}


def load_aerialwaste_mcml(
    root: str | Path,
    split: str = "test",
    version: str = "m2",
    only_pos: bool = False,
) -> tuple[list[str], list[Sample]]:
    """Load the AW multi-class multi-label split (m2 = 5 cats, m4 = 6 cats).

    Args:
        split: one of 'train', 'val', 'test'.
        version: 'm2' or 'm4'.
        only_pos: if True, load the positives-only variant.
    """
    if version not in AERIALWASTE_MCML_VERSIONS:
        raise ValueError(f"unknown mcml version {version!r}")
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}")
    root = Path(root)
    sub = AERIALWASTE_MCML_VERSIONS[version] + ("/only_pos" if only_pos else "")
    with (root / sub / f"{split}.json").open() as f:
        data = json.load(f)

    cat_by_id = {c["id"]: c["name"] for c in data["categories"]}
    categories = [c["name"] for c in data["categories"]]

    images_dir = root / "images"
    samples: list[Sample] = []
    for img in data["images"]:
        cat_ids = img.get("categories") or []
        gt = {cat_by_id[cid] for cid in cat_ids if cid in cat_by_id}
        width = _coerce_int(img["width"])
        height = _coerce_int(img["height"])
        # Prefer explicit `source` if present (the new mcml splits include it);
        # fall back to dimension proxy otherwise.
        src = img.get("source")
        source = {"GE": "google_earth", "AGEA": "agea", "WV3": "worldview3"}.get(
            src, _aerialwaste_image_source(width, height)
        )
        samples.append(
            Sample(
                dataset="aerialwaste",
                image_id=str(img["id"]),
                image_path=images_dir / img["file_name"],
                label=1 if gt else 0,
                width=width,
                height=height,
                image_source=source,
                extra={
                    "gt_categories": sorted(gt),
                    "split": split,
                    "version": version,
                    "valid_fine_grain": _coerce_int(img.get("valid_fine_grain")),
                    "severity": _coerce_int(img.get("severity")),
                    "evidence": _coerce_int(img.get("evidence")),
                    "site_type": img.get("site_type"),
                    "raw_source": src,
                },
            )
        )
    return categories, samples


def load_dronewaste_multilabel(
    root: str | Path,
    categories_filter: list[str] | None = None,
) -> tuple[list[str], list[Sample]]:
    """All DroneWaste samples with multi-label GT (∅ for negatives).

    If `categories_filter` is given, restrict both the returned categories list
    and each sample's gt_categories to that subset. `label` becomes 1 iff the
    image has at least one annotation in the filtered set.
    """
    root = Path(root)
    with (root / "dronewaste_v1.0.json").open() as f:
        data = json.load(f)

    cat_by_id = {c["id"]: c["name"] for c in data["categories"]}
    all_categories = [c["name"] for c in data["categories"]]
    if categories_filter is not None:
        for c in categories_filter:
            if c not in all_categories:
                raise ValueError(f"category {c!r} not in DroneWaste taxonomy")
        keep = set(categories_filter)
        categories = list(categories_filter)
    else:
        keep = set(all_categories)
        categories = all_categories

    img_to_cats: dict[int, set[str]] = {}
    for ann in data.get("annotations", []):
        name = cat_by_id[ann["category_id"]]
        if name in keep:
            img_to_cats.setdefault(ann["image_id"], set()).add(name)

    images_dir = root / "images"
    samples: list[Sample] = []
    for img in data["images"]:
        gt = img_to_cats.get(int(img["id"]), set())
        samples.append(
            Sample(
                dataset="dronewaste",
                image_id=str(img["id"]),
                image_path=images_dir / img["file_name"],
                label=1 if gt else 0,
                width=img["width"],
                height=img["height"],
                image_source=img["site"],
                extra={
                    "site": img["site"],
                    "gt_categories": sorted(gt),
                    "n_annotations": len(gt),
                },
            )
        )
    return categories, samples
