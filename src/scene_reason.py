"""Answer the compositional queries symbolically from a scene graph.

This is the arm that makes the evaluation interpretable. Run over GROUND-TRUTH
detections it is the pure reasoning ceiling and must score ~1.0 by construction --
the queries were generated from these same fields, so anything below 1.0 is a bug
in the solver or a genuine ambiguity in the question, and both are worth knowing
before any model is scored. Run over PREDICTED detections it isolates exactly what
perception error costs, with the reasoning held fixed and perfect.

A scene is a list of objects:
    {'category': str, 'box': [x, y, w, h], 'area': float, 'cx': float, 'cy': float}

The thresholds here mirror `scripts/compositional_queries.py` exactly -- a tenth
of the frame for a direction to count, 20% for one quantity to exceed another.
If the two drift apart the ceiling silently stops being a ceiling.
"""
from __future__ import annotations

from collections import Counter, defaultdict

DIR_MARGIN = 10.0      # a direction counts past a tenth of the frame
RATIO = 1.2            # a comparison counts past 20%
NEAR = 0.25            # "close to" is within a quarter of the diagonal


def cat(o) -> str:
    """Objects reach here from two producers -- the query generator, which writes
    'cat', and the detector, which writes 'category'. Accepting both keeps a
    schema mismatch from being mistaken for a reasoning failure."""
    return o["category"] if "category" in o else o["cat"]


def _present(scene):
    return sorted({cat(o) for o in scene})


def _areas(scene):
    a = defaultdict(float)
    for o in scene:
        a[cat(o)] += o["area"]
    return a


def solve(rec: dict, scene: list, size=(640, 640)) -> str | None:
    """-> the answer string, or None when this family is not handled."""
    fam, meta = rec["family"], rec.get("meta", {})
    W, H = size
    present = set(_present(scene))
    n_by = Counter(cat(o) for o in scene)
    areas = _areas(scene)

    if fam == "presence":
        return "yes" if meta.get("cat") in present else "no"

    if fam == "count":
        n = n_by.get(meta.get("cat"), 0)
        return str(n if n < 4 else "4+")

    if fam == "count_compare":
        return "yes" if n_by.get(meta.get("a"), 0) > n_by.get(meta.get("b"), 0) else "no"

    if fam == "area_compare":
        return "yes" if areas.get(meta.get("a"), 0.0) > areas.get(meta.get("b"), 0.0) else "no"

    if fam == "superlative":
        return max(present, key=lambda c: areas[c]) if present else None

    if fam == "spatial":
        a, b, d = meta.get("a"), meta.get("b"), meta.get("dir")
        A = [o for o in scene if cat(o) == a]
        B = [o for o in scene if cat(o) == b]
        holds = set()
        for o1 in A:
            for o2 in B:
                dx, dy = o1["cx"] - o2["cx"], o1["cy"] - o2["cy"]
                if abs(dx) > W / DIR_MARGIN and abs(dx) > abs(dy):
                    holds.add("to the right of" if dx > 0 else "to the left of")
                elif abs(dy) > H / DIR_MARGIN and abs(dy) > abs(dx):
                    holds.add("below" if dy > 0 else "above")
        return "yes" if d in holds else "no"

    if fam == "negation_spatial":
        c = meta.get("cat")
        diag = (W ** 2 + H ** 2) ** 0.5
        for o1 in (o for o in scene if cat(o) == c):
            for o2 in (o for o in scene if cat(o) != c):
                if ((o1["cx"] - o2["cx"]) ** 2 + (o1["cy"] - o2["cy"]) ** 2) ** 0.5 < NEAR * diag:
                    return "yes"
        return "no"

    return None


def render(scene: list, size=(640, 640)) -> str:
    """The scene as text for a decoder prompt.

    Positions are given in ninths rather than pixels because a decoder reading
    "centre-left" reasons about it better than about "cx=203.4", and the query
    set's own notion of direction is coarse anyway.
    """
    W, H = size
    rows = []
    for i, o in enumerate(sorted(scene, key=lambda o: -o["area"])):
        col = ["left", "centre", "right"][min(2, int(o["cx"] / W * 3))]
        row = ["top", "middle", "bottom"][min(2, int(o["cy"] / H * 3))]
        pct = 100.0 * o["area"] / (W * H)
        rows.append(f"  {i+1}. {cat(o)} — {row}-{col}, covering {pct:.1f}% of the image")
    return "\n".join(rows) if rows else "  (no waste detected)"
