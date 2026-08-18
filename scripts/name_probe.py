"""Describe first, then ASK about each material -- one calibrated question per class.

The intended product shape is describe-then-question. The version of it currently
in the eval (`open_cot`) asks one open turn -- "based on what you described, what
types of waste are visible?" -- and it is the worst-performing readout in the
project: micro-F1 0.004 on aw_m2, an empty parse on 540 of 581 images, `none`
answered on tiles the model has just described as holding piles of solid waste.

The failure is not that the model cannot see the material. It is that the open
turn asks the decoder to *generate* a commitment, and a free-text commitment has
to survive both the model's hedging and a keyword parser. The same model answers
a *closed* Yes/No question about waste presence at AUC 0.94.

So keep the user's structure and change the question type: after the description,
ask one Yes/No question per category and read the first-token margin, exactly the
readout the binary gate already validates. Naming becomes five calibrated binary
decisions with per-class thresholds instead of one generation the parser must
guess at.

Two conditions are recorded per category so the description's contribution is
measurable rather than assumed:

  bare  -- the question against the image alone
  ctx   -- the question with the model's own description prepended, framed exactly
           as `VLMAdapter.generate_cot` frames it

If `ctx` does not beat `bare`, the describe turn is not helping the naming and is
only worth keeping for the human-readable output.

    python scripts/name_probe.py --generate --ckpt <dir> --encoder cradiov4-so \
        --image-size 768 --pixel-shuffle 2 --dataset aw_m2 \
        --train-limit 1200 --out names.json
    python scripts/name_probe.py --report names.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

CTX_FRAME = "Based on this aerial image analysis:\n{desc}\n\n{q}"

# Injected reference: what each category LOOKS LIKE from above, hand-written for
# aerial identification and already in the repo (src/aw_m2_descriptions.json).
# The whole block is 172 tokens, so it is nearly free -- the binding question is
# not budget but whether the model can read the properties the cues appeal to.
# Every cue leans on colour and texture ("grey/beige, dusty", "bright colours,
# smooth/glossy", "rusty-brown"), and the ladder's appearance rung came back as a
# template with colour mentioned once in six images. If the model cannot report
# those properties unprompted, telling it which property maps to which label
# cannot help, and this condition is what shows that rather than assuming it.
CUE_FRAME = "Use these descriptions of how each material looks from above:\n{cues}\n\n{q}"


def cue_block(dataset: str) -> str:
    """The `aerial_cue` line for each category, as one reference block."""
    import json as _json
    name = {"aw_m2": "aw_m2", "aw_m4": "aw_m4", "dw_paper10": "paper10"}[dataset]
    path = pathlib.Path(__file__).resolve().parents[1] / "src" / f"{name}_descriptions.json"
    d = _json.loads(path.read_text())
    return "\n".join(f"- {k}: {v['aerial_cue']}"
                     for k, v in d.items()
                     if isinstance(v, dict) and "aerial_cue" in v)


def questions(dataset: str) -> dict[str, str]:
    """One Yes/No question per category, built from the same cue file the
    open-vocabulary prompt draws its examples from, so the two readouts are
    asking about the same things in different grammatical moods."""
    from src.vlm_eval import AW_M2_CLIP_TAGS, AW_M4_CLIP_TAGS, PAPER10_CLIP_TAGS
    tags = {"aw_m2": AW_M2_CLIP_TAGS, "aw_m4": AW_M4_CLIP_TAGS,
            "dw_paper10": PAPER10_CLIP_TAGS}[dataset]
    out = {}
    for cat, syn in tags.items():
        alts = [s for s in syn[:3]] or [cat.lower()]
        phrase = alts[0] if len(alts) == 1 else ", ".join(alts[:-1]) + " or " + alts[-1]
        out[cat] = (f"Is there {phrase} visible in this image? "
                    f"Answer Yes or No.")
    return out


def generate(args) -> None:
    import random

    from PIL import Image

    from src import vlm_calib
    from src.vlm_eval import PROMPT_DESCRIBE, WasteVLMAdapter
    from scripts.make_convo import load_samples

    qs = questions(args.dataset)
    cues = cue_block(args.dataset)
    print(f"[names] {len(qs)} category questions:")
    for c, q in qs.items():
        print(f"   {c:22s} {q}")

    _cats, samples = load_samples(args.dataset)
    have = [s for s in samples if pathlib.Path(s.image_path).exists()]

    if args.split == "train":
        from src.datasets import load_aerialwaste_mcml
        from src.vlm_eval import WASTE_DATA_ROOT
        _c, tr = load_aerialwaste_mcml(
            str(WASTE_DATA_ROOT / "aerialwaste"), split="train",
            version="m2" if args.dataset == "aw_m2" else "m4")
        have = [s for s in tr if pathlib.Path(s.image_path).exists()]
        if args.train_limit:
            random.Random(0).shuffle(have)
            have = have[: args.train_limit]

    print(f"[names] {len(have)} images x ({len(qs)} cats x 2 conditions + 1 gate)",
          flush=True)

    adapter = WasteVLMAdapter(args.ckpt, encoder=args.encoder,
                              image_size=args.image_size,
                              pixel_shuffle=args.pixel_shuffle,
                              max_new_tokens=args.max_new_tokens)
    adapter.load()

    out = []
    for n, s in enumerate(have):
        img = Image.open(s.image_path).convert("RGB")
        desc = adapter.generate(img, PROMPT_DESCRIBE)
        rec = {"image_id": s.image_id, "gt": sorted(s.extra["gt_categories"]),
               "desc": desc,
               "gate": adapter.decision_margin(img, vlm_calib.QUESTION),
               "bare": {}, "ctx": {}, "cued": {}}
        for cat, q in qs.items():
            rec["bare"][cat] = adapter.decision_margin(img, q)
            rec["ctx"][cat] = adapter.decision_margin(
                img, CTX_FRAME.format(desc=desc, q=q))
            rec["cued"][cat] = adapter.decision_margin(
                img, CUE_FRAME.format(cues=cues, q=q))
        out.append(rec)
        if n % 25 == 0:
            print(f"[names] {n}/{len(have)}", flush=True)

    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[write] {args.out}  ({len(out)} images)")


# ------------------------------------------------------------------ reporting
def fit_threshold(scores: list[float], labels: list[int]) -> tuple[float, float]:
    """Threshold maximising F1 for one category. F1 rather than Youden's J
    because the downstream metric is micro-F1 and J on a rare class buys recall
    at a precision the F1 table then charges for."""
    best = (0.0, 0.0)
    for t in sorted(set(scores)):
        tp = sum(1 for s, y in zip(scores, labels) if s >= t and y)
        fp = sum(1 for s, y in zip(scores, labels) if s >= t and not y)
        fn = sum(1 for s, y in zip(scores, labels) if s < t and y)
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best[1]:
            best = (t, f1)
    return best


def auc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    from scripts.check_grounding import mannwhitney
    return mannwhitney(pos, neg)[0]


def report(args) -> None:
    test = json.loads(pathlib.Path(args.report).read_text())
    fit = json.loads(pathlib.Path(args.fit).read_text()) if args.fit else test
    cats = sorted({c for r in test for c in r["gt"]} | set(test[0]["bare"]))
    if args.fit is None:
        print("  [warn] no --fit: thresholds fitted on the test set itself, "
              "so these numbers are an upper bound, not an estimate")

    conds = [c for c in ("bare", "ctx", "cued") if c in test[0]]
    for cond in conds:
        print(f"\n=== {cond}: one Yes/No question per category "
              f"({len(fit)} fit / {len(test)} test images)")
        print(f"  {'category':24s} {'prev':>6s} {'AUC':>6s} {'thr':>7s} "
              f"{'P':>6s} {'lift':>7s} {'R':>6s} {'F1':>6s}")
        TP = FP = FN = 0
        for c in cats:
            fs = [r[cond][c] for r in fit]
            fy = [int(c in r["gt"]) for r in fit]
            thr, _ = fit_threshold(fs, fy)
            ts = [r[cond][c] for r in test]
            ty = [int(c in r["gt"]) for r in test]
            tp = sum(1 for s, y in zip(ts, ty) if s >= thr and y)
            fp = sum(1 for s, y in zip(ts, ty) if s >= thr and not y)
            fn = sum(1 for s, y in zip(ts, ty) if s < thr and y)
            TP += tp; FP += fp; FN += fn
            prev = sum(ty) / len(ty)
            p = tp / (tp + fp) if tp + fp else 0.0
            r_ = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * p * r_ / (p + r_) if p + r_ else 0.0
            print(f"  {c[:24]:24s} {prev:6.3f} {auc(ts, ty):6.3f} {thr:+7.2f} "
                  f"{p:6.3f} {p-prev:+7.3f} {r_:6.3f} {f1:6.3f}")
        mp = TP / (TP + FP) if TP + FP else 0.0
        mr = TP / (TP + FN) if TP + FN else 0.0
        print(f"  micro  P {mp:.3f}  R {mr:.3f}  "
              f"F1 {2*mp*mr/(mp+mr) if mp+mr else 0:.3f}   "
              f"(TP {TP} FP {FP} FN {FN})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--report")
    ap.add_argument("--fit", help="a second JSON (train split) to fit thresholds on")
    ap.add_argument("--ckpt")
    ap.add_argument("--encoder", default="cradiov4-so")
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--pixel-shuffle", type=int, default=2)
    ap.add_argument("--dataset", default="aw_m2")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--train-limit", type=int, default=1200)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", default="names.json")
    args = ap.parse_args()
    if args.generate:
        generate(args)
    if args.report:
        report(args)


if __name__ == "__main__":
    main()
