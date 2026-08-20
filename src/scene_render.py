"""Deterministic renderer: scene graph -> English, with no model in the loop.

This is the floor for the description experiment. It can only say things the
graph contains, so its unsupported-claim rate is 0 by construction, and any
value the LLM arm adds has to show up as something this cannot do -- fluency,
aggregation, answering a question -- rather than as extra facts.

Kept deliberately plain. A florid template would confound "the LLM writes better
prose" with "the template was written badly".
"""
from __future__ import annotations

from collections import Counter

# Nine-box spatial grid, the same binning the metric uses to decide whether a
# location word is supported.
def region(cx: float, cy: float, w: int, h: int) -> str:
    col = "left" if cx < w / 3 else ("centre" if cx < 2 * w / 3 else "right")
    row = "top" if cy < h / 3 else ("middle" if cy < 2 * h / 3 else "bottom")
    if row == "middle" and col == "centre":
        return "centre"
    return f"{row}-{col}" if row != "middle" else col


def size_word(area: float, img_area: float) -> str:
    f = area / max(img_area, 1.0)
    return "small" if f < 0.01 else ("large" if f > 0.10 else "medium-sized")


def render(scene: dict, min_score: float = 0.0) -> str:
    w, h = scene.get("size", [640, 640])
    objs = [o for o in scene.get("objs", []) if o.get("score", 1.0) >= min_score]
    if not objs:
        return "No waste is visible in this image."
    by_cat: dict[str, list] = {}
    for o in objs:
        by_cat.setdefault(o.get("category") or o.get("cat"), []).append(o)
    parts = []
    for cat, items in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        n = len(items)
        regions = Counter(region(o["cx"], o["cy"], w, h) for o in items)
        sizes = Counter(size_word(o["area"], w * h) for o in items)
        where = ", ".join(r for r, _ in regions.most_common(2))
        sz = sizes.most_common(1)[0][0]
        noun = cat.lower()
        if n == 1:
            parts.append(f"one {sz} area of {noun} in the {where}")
        else:
            parts.append(f"{n} areas of {noun} ({sz}, mostly {where})")
    total = len(objs)
    head = (f"This image contains {total} detected waste "
            f"{'object' if total == 1 else 'objects'}.")
    return head + " It shows " + "; ".join(parts) + "."


def graph_facts(scene: dict, min_score: float = 0.0) -> dict:
    """The supported set: what a description may assert without inventing."""
    w, h = scene.get("size", [640, 640])
    objs = [o for o in scene.get("objs", []) if o.get("score", 1.0) >= min_score]
    cats = Counter((o.get("category") or o.get("cat")) for o in objs)
    return {
        "categories": set(cats),
        "counts": dict(cats),
        "total": len(objs),
        "regions": {region(o["cx"], o["cy"], w, h) for o in objs},
        "sizes": {size_word(o["area"], w * h) for o in objs},
    }
