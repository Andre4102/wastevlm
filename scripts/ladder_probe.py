"""Lead the model down to the material name instead of asking for it cold.

The naming turn currently fails in a specific way: asked "what types of waste are
visible?", the model that has just written "piles of solid waste in the bottom-left"
answers `none`. It is not that the material is invisible to it -- it is that one
open turn asks the decoder to jump from pixels to a taxonomy label in a single step,
and the step is too long.

So scaffold it. Ask for presence first, then appearance, then the name, with each
answer carried into the next turn:

  1. presence   -- is anything here abandoned, dumped, piled, out of place?
  2. appearance -- for what you just found: colour, texture, shape, size, layout
  3. name       -- given that appearance, what material is it?

The hypothesis is that colour and texture are things the model can report reliably
(they are visual properties, not taxonomy), and that a name conditioned on its own
appearance description is easier to produce than a name conditioned on pixels alone.

Both readouts are recorded at rung 3, because they fail differently:

  open   -- free text through the eval's keyword parser, the readout that scores
            0.004 today
  askyn  -- one Yes/No question per category, first-token margin, the readout the
            binary gate already validates at AUC 0.94

Every rung is stored, so a failure can be localised to the rung that lost it rather
than blamed on the chain as a whole. Rung-2 text is also worth reading on its own:
if the appearance descriptions are generic ("various materials, mixed colours"), the
ladder cannot help and the ceiling is perceptual, not linguistic.

    python scripts/ladder_probe.py --generate --ckpt <dir> --encoder cradiov4-so \
        --image-size 768 --pixel-shuffle 2 --dataset aw_m2 --out ladder.json
    python scripts/ladder_probe.py --report ladder.json --fit ladder_train.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

Q1 = ("This is an aerial photograph of an outdoor area. Is there anything in it "
      "that looks abandoned, dumped, or out of place -- piles, heaps, mounds, "
      "scattered objects, or accumulations of material that do not belong to a "
      "building, a field, or a road? Describe each one and say where it is.")

Q2 = ("For each pile or object you just identified, describe how it LOOKS, "
      "without naming what it is: its colour, its texture (smooth, rough, dusty, "
      "shiny, fibrous), its shape, its rough size, and how the parts are arranged "
      "(heaped, scattered, stacked, flattened).")

Q3_OPEN = ("Given the appearance you just described -- those colours, textures and "
           "shapes -- what material is each pile or object most likely made of? "
           "Name the materials. If there is genuinely nothing dumped here, reply "
           "exactly \"none\".")

# Same framing `VLMAdapter.generate_cot` uses, extended one rung, so the ladder is
# not accidentally testing a new prompt dialect as well as a new question order.
CTX2 = "Based on this aerial image analysis:\n{a1}\n\n{q}"
CTX3 = ("Based on this aerial image analysis:\n{a1}\n\n"
        "And this description of how the materials look:\n{a2}\n\n{q}")


def generate(args) -> None:
    import random

    from PIL import Image

    from src import vlm_calib
    from src.vlm_eval import WasteVLMAdapter
    from scripts.name_probe import questions
    from scripts.make_convo import load_samples

    qs = questions(args.dataset)
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
    if args.limit:
        have = have[: args.limit]
    print(f"[ladder] {len(have)} images x (3 generations + {len(qs)} margins)",
          flush=True)

    adapter = WasteVLMAdapter(args.ckpt, encoder=args.encoder,
                              image_size=args.image_size,
                              pixel_shuffle=args.pixel_shuffle,
                              max_new_tokens=args.max_new_tokens)
    adapter.load()

    out = []
    for n, s in enumerate(have):
        img = Image.open(s.image_path).convert("RGB")
        a1 = adapter.generate(img, Q1)
        a2 = adapter.generate(img, CTX2.format(a1=a1, q=Q2))
        a3 = adapter.generate(img, CTX3.format(a1=a1, a2=a2, q=Q3_OPEN))
        rec = {"image_id": s.image_id, "gt": sorted(s.extra["gt_categories"]),
               "presence": a1, "appearance": a2, "open_name": a3,
               "gate": adapter.decision_margin(img, vlm_calib.QUESTION),
               "askyn": {}}
        for cat, q in qs.items():
            rec["askyn"][cat] = adapter.decision_margin(
                img, CTX3.format(a1=a1, a2=a2, q=q))
        out.append(rec)
        if n % 25 == 0:
            print(f"[ladder] {n}/{len(have)}", flush=True)

    pathlib.Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[write] {args.out}  ({len(out)} images)")


def report(args) -> None:
    from src.vlm_eval import AW_M2_KEYWORDS, AW_M4_KEYWORDS, PAPER10_KEYWORDS, parse_keywords
    from scripts.name_probe import auc, fit_threshold

    bags = {"aw_m2": AW_M2_KEYWORDS, "aw_m4": AW_M4_KEYWORDS,
            "dw_paper10": PAPER10_KEYWORDS}[args.dataset]
    test = json.loads(pathlib.Path(args.report).read_text())
    fit = json.loads(pathlib.Path(args.fit).read_text()) if args.fit else test
    cats = sorted(bags)
    pos = [r for r in test if r["gt"]]

    # How much text each rung actually produced. A rung that collapses to a stock
    # sentence cannot be conditioning the next one, whatever the F1 says.
    print(f"\n=== rungs ({len(test)} images)")
    for k in ("presence", "appearance", "open_name"):
        n_words = sum(len(r[k].split()) for r in test) / len(test)
        distinct = len({r[k].strip() for r in test})
        print(f"  {k:11s} {n_words:5.1f} words   {distinct:4d}/{len(test)} distinct")

    print(f"\n=== open naming through the keyword parser ({len(pos)} positives)")
    TP = FP = FN = 0
    for c in cats:
        tp = sum(1 for r in pos if c in parse_keywords(r["open_name"], bags) and c in r["gt"])
        fp = sum(1 for r in pos if c in parse_keywords(r["open_name"], bags) and c not in r["gt"])
        fn = sum(1 for r in pos if c not in parse_keywords(r["open_name"], bags) and c in r["gt"])
        TP += tp; FP += fp; FN += fn
        prev = sum(1 for r in pos if c in r["gt"]) / len(pos)
        p = tp / (tp + fp) if tp + fp else 0.0
        rr = tp / (tp + fn) if tp + fn else 0.0
        print(f"  {c[:24]:24s} prev {prev:.3f}  P {p:6.3f} ({p-prev:+.3f})  "
              f"R {rr:6.3f}  said {tp+fp:4d}")
    mp = TP / (TP + FP) if TP + FP else 0.0
    mr = TP / (TP + FN) if TP + FN else 0.0
    print(f"  micro F1 {2*mp*mr/(mp+mr) if mp+mr else 0:.3f}  (TP {TP} FP {FP} FN {FN})")
    empty = sum(1 for r in pos if not parse_keywords(r["open_name"], bags))
    print(f"  parsed to nothing on {empty}/{len(pos)} positives ({empty/len(pos):.0%})")

    print(f"\n=== ask-Yes/No per category at the bottom of the ladder")
    if args.fit is None:
        print("  [warn] no --fit: thresholds fitted on test, an upper bound not an estimate")
    print(f"  {'category':24s} {'prev':>6s} {'AUC':>6s} {'thr':>7s} {'P':>6s} "
          f"{'lift':>7s} {'R':>6s} {'F1':>6s}")
    TP = FP = FN = 0
    for c in cats:
        thr, _ = fit_threshold([r["askyn"][c] for r in fit],
                               [int(c in r["gt"]) for r in fit])
        ts = [r["askyn"][c] for r in pos]
        ty = [int(c in r["gt"]) for r in pos]
        tp = sum(1 for s, y in zip(ts, ty) if s >= thr and y)
        fp = sum(1 for s, y in zip(ts, ty) if s >= thr and not y)
        fn = sum(1 for s, y in zip(ts, ty) if s < thr and y)
        TP += tp; FP += fp; FN += fn
        prev = sum(ty) / len(ty)
        p = tp / (tp + fp) if tp + fp else 0.0
        rr = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * rr / (p + rr) if p + rr else 0.0
        print(f"  {c[:24]:24s} {prev:6.3f} {auc(ts, ty):6.3f} {thr:+7.2f} {p:6.3f} "
              f"{p-prev:+7.3f} {rr:6.3f} {f1:6.3f}")
    mp = TP / (TP + FP) if TP + FP else 0.0
    mr = TP / (TP + FN) if TP + FN else 0.0
    print(f"  micro  P {mp:.3f}  R {mr:.3f}  "
          f"F1 {2*mp*mr/(mp+mr) if mp+mr else 0:.3f}   (TP {TP} FP {FP} FN {FN})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--report")
    ap.add_argument("--fit")
    ap.add_argument("--ckpt")
    ap.add_argument("--encoder", default="cradiov4-so")
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--pixel-shuffle", type=int, default=2)
    ap.add_argument("--dataset", default="aw_m2")
    ap.add_argument("--split", default="test", choices=["test", "train"])
    ap.add_argument("--train-limit", type=int, default=1200)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=160)
    ap.add_argument("--out", default="ladder.json")
    args = ap.parse_args()
    if args.generate:
        generate(args)
    if args.report:
        report(args)


if __name__ == "__main__":
    main()
