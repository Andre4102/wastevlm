"""Why is AerialWaste bad? Decompose a run's micro-F1 into detection vs naming.

The constant-baseline control in vlm_eval says *that* an AW number is worthless.
This says *why*, by splitting the score three ways:

  detection   -- did the model emit any label, vs did the image have any GT label.
                 ~70% of the AW test split has empty GT, so every label emitted on
                 one of those is a pure false positive.
  oracle-gate -- our own predictions with every false alarm deleted. Identical to
                 scoring on the positives-only split AW ships and nothing uses.
  naming      -- micro-F1 over the GT-positive images the model actually answered.

Each is compared against the best *constant* (image-independent) predictor under
the same regime, because on AW the taxonomy is prior-dominated and beating the
prior is the only thing that counts.

    python scripts/aw_diagnose.py --gt            # split structure only
    python scripts/aw_diagnose.py --runs A B C    # decompose named eval dirs
    python scripts/aw_diagnose.py --noise         # are the "negatives" real?
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import pathlib

import numpy as np

R = pathlib.Path(
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/results/vlm_eval"
)
DATA = pathlib.Path("/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data")
AW = DATA / "aerialwaste"
# mcml_split_dataset_1 is m2 (5 cats), _2 is m4 (6 cats)
AW_SUB = {"m2": "mcml_split_dataset_1", "m4": "mcml_split_dataset_2"}

DEFAULT_RUNS = [
    "vlm_radio-l_aw_m4_open_cot",
    "vlm_cradiov4-so_r1024ps2_finetune_aw_m4_open_cot",
    "vlm_cradiov4-so_r768ps2_finetune_aw_m4_open_cot",
    "vlm_cradiov4-so_r768ps2_finetune_aw_m2_closed_vocab",
    "vlm_cradiov4-so_r768ps2_finetune_next_dw_paper10_closed_vocab",
]


def read_jsonl(path: pathlib.Path) -> list[dict]:
    # iterate the handle rather than splitlines(): at least one raw_responses file
    # contains a literal U+2028, which str.splitlines() treats as a line break.
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def micro(pairs: list[tuple[set, set]]) -> tuple[float, int, int, int]:
    tp = sum(len(p & g) for p, g in pairs)
    fp = sum(len(p - g) for p, g in pairs)
    fn = sum(len(g - p) for p, g in pairs)
    denom = 2 * tp + fp + fn
    return (2 * tp / denom if denom else 0.0), tp, fp, fn


def best_constant(gts: list[set], cats: list[str], mask: list[bool] | None = None):
    """Best fixed label set, emitted on every image (or only where mask is True).

    Exhaustive over 2^n_classes -- fine at 5/6 classes (AW) and 20 (DW) is the
    one case where this is slow, so DW callers should pass a trimmed cats list.
    """
    keep = mask if mask is not None else [True] * len(gts)
    best = (0.0, ())
    for k in range(1, len(cats) + 1):
        for combo in itertools.combinations(cats, k):
            s = set(combo)
            preds = [s if m else set() for m in keep]
            val = micro(list(zip(preds, gts)))[0]
            if val > best[0]:
                best = (val, combo)
    return best


def gt_structure() -> None:
    """Label prior of each split -- the reason AW's constant floor is so high."""
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    from src.datasets import load_aerialwaste_mcml, load_dronewaste_multilabel

    def show(name, cats, gts):
        n = len(gts)
        empty = sum(1 for g in gts if not g)
        card = [len(g) for g in gts]
        pos = [c for c in card if c]
        prev = collections.Counter(c for g in gts for c in g)
        total = sum(card)
        print(f"\n=== {name}: {n} images, {len(cats)} classes")
        print(f"  empty GT      : {empty} ({100 * empty / n:.1f}%)")
        print(f"  labels/image  : {np.mean(card):.3f} overall, "
              f"{np.mean(pos) if pos else 0:.3f} among positives")
        for c, k in prev.most_common():
            print(f"    {c:50s} {k:5d} imgs "
                  f"({100 * k / n:5.1f}% of imgs, {100 * k / total:5.1f}% of GT mass)")

    for v in ("m2", "m4"):
        cats, s = load_aerialwaste_mcml(str(AW), split="test", version=v)
        s = [x for x in s if x.image_path.exists()]
        show(f"aw_{v}", cats, [set(x.extra["gt_categories"]) for x in s])

    cats, samples = load_dronewaste_multilabel(str(DATA / "dronewaste"))
    site = collections.defaultdict(list)
    for i, x in enumerate(samples):
        site[x.image_source].append(i)
    rng = np.random.default_rng(0)  # same 70/30 site-stratified split as the eval
    idx: list[int] = []
    for _s, ii in site.items():
        ii = list(ii)
        rng.shuffle(ii)
        idx.extend(ii[int(len(ii) * 0.7):])
    show("dw_paper10", cats, [set(samples[i].extra["gt_categories"]) for i in idx])


def decompose(run: str) -> None:
    d = R / run
    if not (d / "raw_responses.jsonl").exists():
        print(f"\n### {run}\n  -- no raw_responses.jsonl --")
        return
    recs = read_jsonl(d / "raw_responses.jsonl")
    pairs = [(set(r.get("parsed") or []), set(r.get("gt") or [])) for r in recs]
    gts = [g for _, g in pairs]
    cats = sorted({c for g in gts for c in g})
    posmask = [bool(g) for g in gts]

    f1_all, tp, fp, fn = micro(pairs)
    tp_d = sum(1 for p, g in pairs if p and g)
    fp_d = sum(1 for p, g in pairs if p and not g)
    fn_d = sum(1 for p, g in pairs if not p and g)
    prec = tp_d / (tp_d + fp_d) if tp_d + fp_d else 0.0
    rec = tp_d / (tp_d + fn_d) if tp_d + fn_d else 0.0
    det = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    gated = micro([(p if g else set(), g) for p, g in pairs])[0]
    answered_pos = [(p, g) for p, g in pairs if g and p]
    naming = micro(answered_pos)[0] if answered_pos else 0.0
    cb_all = best_constant(gts, cats)
    cb_gate = best_constant(gts, cats, mask=posmask)

    n_neg = len(gts) - sum(posmask)
    print(f"\n### {run}")
    print(f"  micro-F1 all            {f1_all:.4f}  (tp {tp}, fp {fp}, fn {fn})")
    print(f"  best CONSTANT           {cb_all[0]:.4f}  {cb_all[1]}")
    print(f"  detection any-label?    P {prec:.3f} R {rec:.3f} F1 {det:.3f}"
          f"   [gt+ {sum(posmask)}/{len(gts)}, false alarms {fp_d}/{n_neg}]")
    print(f"  ours, ORACLE-gated      {gated:.4f}  (= positives-only split)")
    print(f"  constant, ORACLE-gated  {cb_gate[0]:.4f}  {cb_gate[1]}")
    print(f"  naming on answered gt+  {naming:.4f}  over {len(answered_pos)} images")


def label_noise(version: str = "m4") -> None:
    """Are the empty-GT images true negatives, or sites with unusable labels?"""
    # filter to images on disk, like the eval does -- AW ships a PNEO subset whose
    # files aren't in the image zips, and counting them here would not match the runs.
    meta = {
        str(i["id"]): i
        for i in json.load((AW / AW_SUB[version] / "test.json").open())["images"]
        if (AW / "images" / i["file_name"]).exists()
    }
    empty = [i for i in meta.values() if not (i.get("categories") or [])]
    real_site = [i for i in empty if i.get("site_type") not in (None, "n/a")]
    print(f"\naw_{version}: {len(empty)} empty-GT images, of which {len(real_site)} "
          f"carry a real site_type/severity/evidence (waste sites with "
          f"valid_fine_grain=0, scored as negatives)")

    for run in DEFAULT_RUNS:
        d = R / run
        if f"aw_{version}" not in run or not (d / "raw_responses.jsonl").exists():
            continue
        buckets: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
        for r in read_jsonl(d / "raw_responses.jsonl"):
            if r.get("gt"):
                continue
            m = meta.get(str(r["image_id"]))
            if m is None:
                continue
            key = ("background (site_type n/a)"
                   if m.get("site_type") in (None, "n/a")
                   else "REAL SITE, no fine-grain labels")
            buckets[key][1] += 1
            if r.get("parsed"):
                buckets[key][0] += 1
        print(f"  {run}")
        for key, (ans, tot) in sorted(buckets.items()):
            print(f"    answered {ans:4d}/{tot:4d} ({100 * ans / tot:5.1f}%)  {key}")


CAPTION_PROBES = ["pile", "debris", "scattered", "various materials", "dirt",
                  "abandoned", "waste", "construction"]


def caption_conditioning(runs: list[str]) -> None:
    """Do the open_cot turn-1 captions depend on the image at all?

    For each probe term, its rate on GT-positive vs GT-negative images. A term
    that fires equally on both carries zero information about waste presence --
    and since the keyword parser reads exactly these terms, a flat profile means
    the label decision is unconditioned on the image. This is the measurement
    that identifies caption-template collapse; see EXPERIMENTS.md.
    """
    for run in runs:
        d = R / run
        if not (d / "raw_responses.jsonl").exists():
            print(f"\n### {run}\n  -- no raw_responses.jsonl --")
            continue
        recs = read_jsonl(d / "raw_responses.jsonl")
        caps = [(r.get("raw_turn1") or "").strip().lower() for r in recs]
        if not any(caps):
            print(f"\n### {run}\n  -- no raw_turn1 (not an open_cot run) --")
            continue
        pos = [c for c, r in zip(caps, recs) if r.get("gt")]
        neg = [c for c, r in zip(caps, recs) if not r.get("gt")]
        heads = collections.Counter(" ".join(c.split()[:12]) for c in caps)
        head, k = heads.most_common(1)[0]
        print(f"\n### {run}   ({len(pos)} gt+, {len(neg)} gt-)")
        print(f"  distinct captions {len(set(caps))}/{len(caps)}")
        print(f"  modal 12-word opening on {k}/{len(caps)} ({100 * k / len(caps):.0f}%): {head[:80]!r}")
        print(f"  {'term':20s} {'gt+ %':>7s} {'gt- %':>7s} {'diff':>7s}")
        for t in CAPTION_PROBES:
            a = 100 * sum(t in c for c in pos) / max(len(pos), 1)
            b = 100 * sum(t in c for c in neg) / max(len(neg), 1)
            print(f"  {t:20s} {a:7.1f} {b:7.1f} {a - b:+7.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", action="store_true", help="split structure / label prior")
    ap.add_argument("--runs", nargs="*", help="eval dir names (default: the arms in the writeup)")
    ap.add_argument("--noise", action="store_true", help="are the negatives real?")
    ap.add_argument("--captions", nargs="*", metavar="RUN",
                    help="caption-conditioning probe on open_cot runs")
    args = ap.parse_args()

    if args.captions is not None:
        caption_conditioning(args.captions or [r for r in DEFAULT_RUNS if "open_cot" in r])
        if not (args.gt or args.noise or args.runs):
            return

    if args.gt:
        gt_structure()
    if args.noise:
        label_noise()
    if args.runs is not None or not (args.gt or args.noise):
        for run in (args.runs or DEFAULT_RUNS):
            decompose(run)


if __name__ == "__main__":
    main()
