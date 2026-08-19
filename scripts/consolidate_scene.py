"""Merge fragment detections into regions, the way the annotations are drawn.

SAM3 segments concepts instance by instance; DroneWaste's annotations mark waste
AREAS. That is a granularity mismatch, not a detection failure, and the evidence is
that the median predicted box covers 4% of the annotated object it sits inside while
each covered object carries a median of 4 predictions. The detector is finding parts
of a pile and the ground truth is the pile.

So: link boxes that touch, take each connected group as one region, and use the
group's bounding box -- which fills the gaps between fragments by construction.
`--dilate` grows boxes before testing overlap, so fragments that are near but not
quite touching still join. Groups whose merged area is small are dropped, which is
the pruning half: an isolated fragment with nothing around it is noise, while the
same fragment inside a cluster becomes part of a region.

The merged region inherits the category of its largest member rather than a vote,
since a big fragment is the one most likely to have been named on enough pixels to
mean something.

Note this is not NMS. NMS keeps one box and suppresses its neighbours, which throws
away the extent the neighbours were describing; here the neighbours define it.
"""
from __future__ import annotations

import argparse
import json
import pathlib


def overlap(a, b) -> float:
    """Intersection over the SMALLER box.

    Plain IoU is the wrong test between a fragment and the region it belongs to:
    a small box sitting entirely inside a large one scores near zero on IoU and
    1.0 here, which is what "this fragment is part of that pile" should mean.
    """
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    return inter / max(1e-9, min(aw * ah, bw * bh))


def touches(a, b, dilate: float, min_ov: float = 0.0) -> bool:
    if min_ov > 0:
        # Requiring real overlap stops single-linkage from CHAINING. At 86.7 boxes
        # an image is dense enough that mere adjacency links A to B to C and
        # collapses distinct piles into one blob -- which is what the first version
        # did, taking 86.7 boxes to 6.5 regions and the score from 0.479 to 0.442.
        return overlap(a, b) >= min_ov
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ad, bd = dilate * max(aw, ah), dilate * max(bw, bh)
    return not (ax0 - ad > bx0 + bw + bd or bx0 - bd > ax0 + aw + ad
                or ay0 - ad > by0 + bh + bd or by0 - bd > ay0 + ah + ad)


def consolidate(objs, dilate: float, min_frac: float, size, min_members: int = 1,
                min_ov: float = 0.0):
    W, H = size
    n = len(objs)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if touches(objs[i]["box"], objs[j]["box"], dilate, min_ov):
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(objs[i])

    out = []
    for members in groups.values():
        x0 = min(o["box"][0] for o in members)
        y0 = min(o["box"][1] for o in members)
        x1 = max(o["box"][0] + o["box"][2] for o in members)
        y1 = max(o["box"][1] + o["box"][3] for o in members)
        area = (x1 - x0) * (y1 - y0)
        if area < min_frac * W * H or len(members) < min_members:
            continue
        lead = max(members, key=lambda o: o["box"][2] * o["box"][3])
        out.append({"category": lead["category"], "box": [x0, y0, x1 - x0, y1 - y0],
                    "area": float(area), "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2,
                    "score": max(o.get("score", 0.0) for o in members),
                    "margin": lead.get("margin", 0.0), "n_parts": len(members)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dilate", type=float, default=0.0)
    ap.add_argument("--min-frac", type=float, default=0.0)
    ap.add_argument("--min-members", type=int, default=1)
    ap.add_argument("--min-overlap", type=float, default=0.0,
                    help="link two boxes only if their intersection covers this "
                         "fraction of the smaller one; 0 falls back to adjacency")
    args = ap.parse_args()

    scenes = json.loads(pathlib.Path(args.scenes).read_text())
    out = []
    for s in scenes:
        out.append({**s, "objs": consolidate(s["objs"], args.dilate, args.min_frac,
                                             tuple(s["size"]), args.min_members,
                                             args.min_overlap)})
    pathlib.Path(args.out).write_text(json.dumps(out))
    before = sum(len(s["objs"]) for s in scenes) / len(scenes)
    after = sum(len(s["objs"]) for s in out) / len(out)
    print(f"  overlap>={args.min_overlap:.2f} min-frac {args.min_frac:.4f} "
          f"members>={args.min_members}: {before:.1f} -> {after:.1f} regions/img")


if __name__ == "__main__":
    main()
