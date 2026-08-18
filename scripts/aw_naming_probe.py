"""Are the material CATEGORIES in the frozen features, or not in the imagery at all?

`aw_feature_probe.py` settled the detection half: waste/no-waste is linearly
separable in C-RADIOv4-SO400M at AUC 0.970, against J=0.234 for the best of 13
full-VLM runs. Its own caveat was explicit -- "it bounds the detection half only,
not fine-grained naming".

Naming is where every readout now fails. Forced-choice emits its two commonest
labels at their base rate; the open turn answers `none` on tiles it has just
described as holding debris; the appearance rung of the scaffolded ladder comes
back as a template. All of that is consistent with two very different worlds, and
the difference decides what to build next:

  A. the categories ARE in the features and the decoder path loses them
     -> the VLM framing is salvageable; the readout is the thing to change
  B. the categories are NOT recoverable from a 768px aerial tile
     -> no prompt, no scaffold and no decoder change can help, and the honest
        thesis result is a statement about the task, not about the method

One logistic regression per category on frozen features answers it. Same encoder,
same resolution, same train/test images, same 5-way multilabel target as the VLM's
naming eval -- only the readout differs.

The column that decides is precision minus prevalence, not F1. On a split where
71% of positives contain bulky items, a probe that always says "bulky items"
scores 0.71 precision and 1.0 recall while knowing nothing.

Positives only, matching how the VLM naming tables are computed: on negatives the
target is empty for every class and including them inflates every rate.

    python scripts/aw_naming_probe.py --version m2 --image-size 768
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aw_feature_probe import encode_split  # noqa: E402
from src.datasets import load_aerialwaste_mcml  # noqa: E402
from src.vision_encoder import VisionEncoder  # noqa: E402

DATA = pathlib.Path(os.environ.get(
    "WASTE_DATA_ROOT",
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data"))


def fit_category(X_tr, y_tr, X_te, y_te, head: str = "linear") -> dict:
    """Per-class probe with the operating point chosen on TRAIN, never on test."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import normalize

    if head == "mlp":
        # The linear probe is only an upper bound on what a prompt can extract if
        # the representation is linearly readable. An MLP tests whether there is
        # nonlinear signal a cleverer readout could in principle reach; if it does
        # not beat the linear head, the ceiling argument holds.
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(hidden_layer_sizes=(512,), max_iter=400,
                            early_stopping=True, random_state=0)
    else:
        clf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced", n_jobs=-1)
    clf.fit(normalize(X_tr), y_tr)
    s_tr = clf.predict_proba(normalize(X_tr))[:, 1]
    s_te = clf.predict_proba(normalize(X_te))[:, 1]

    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(np.round(s_tr, 3)):
        tp = int(((s_tr >= t) & (y_tr == 1)).sum())
        fp = int(((s_tr >= t) & (y_tr == 0)).sum())
        fn = int(((s_tr < t) & (y_tr == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1

    tp = int(((s_te >= best_t) & (y_te == 1)).sum())
    fp = int(((s_te >= best_t) & (y_te == 0)).sum())
    fn = int(((s_te < best_t) & (y_te == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {
        "auc": float(roc_auc_score(y_te, s_te)) if 0 < y_te.mean() < 1 else float("nan"),
        "threshold": best_t, "tp": tp, "fp": fp, "fn": fn,
        "precision": p, "recall": r,
        "f1": 2 * p * r / (p + r) if p + r else 0.0,
        "prevalence": float(y_te.mean()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="cradiov4-so")
    ap.add_argument("--version", default="m2", choices=["m2", "m4"])
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--heads", nargs="+", default=["linear"],
                    choices=["linear", "mlp"])
    ap.add_argument("--out-json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    aw = str(DATA / "aerialwaste")
    cats, train = load_aerialwaste_mcml(aw, split="train", version=args.version)
    _c, test = load_aerialwaste_mcml(aw, split="test", version=args.version)
    # Positives only, and the same on-disk filter the eval applies.
    train = [s for s in train
             if s.image_path.exists() and s.extra["gt_categories"]]
    test = [s for s in test
            if s.image_path.exists() and s.extra["gt_categories"]]
    print(f"[data] train {len(train)} positives  test {len(test)} positives  "
          f"cats {cats}", flush=True)

    enc = VisionEncoder(args.encoder, device="cuda", image_size=args.image_size)
    print("[encode] train", flush=True)
    F_tr = encode_split(enc, train, args.batch_size)
    print("[encode] test", flush=True)
    F_te = encode_split(enc, test, args.batch_size)

    rep = {"encoder": args.encoder, "version": args.version,
           "image_size": args.image_size, "n_train": len(train),
           "n_test": len(test), "pooling": {}}

    for pool, head in [(p, h) for p in ("cls", "mean", "max") for h in args.heads]:
        print(f"\n=== AW {args.version} naming, frozen {args.encoder} "
              f"@{args.image_size}px, {pool} pooling, {head} head, "
              f"{len(test)} test positives")
        print(f"  {'category':24s} {'prev':>6s} {'AUC':>6s} {'P':>6s} {'lift':>7s} "
              f"{'R':>6s} {'F1':>6s}")
        TP = FP = FN = 0
        per = {}
        for c in cats:
            y_tr = np.array([1 if c in s.extra["gt_categories"] else 0 for s in train])
            y_te = np.array([1 if c in s.extra["gt_categories"] else 0 for s in test])
            r = fit_category(F_tr[pool], y_tr, F_te[pool], y_te, head)
            per[c] = r
            TP += r["tp"]; FP += r["fp"]; FN += r["fn"]
            print(f"  {c[:24]:24s} {r['prevalence']:6.3f} {r['auc']:6.3f} "
                  f"{r['precision']:6.3f} {r['precision']-r['prevalence']:+7.3f} "
                  f"{r['recall']:6.3f} {r['f1']:6.3f}")
        mp = TP / (TP + FP) if TP + FP else 0.0
        mr = TP / (TP + FN) if TP + FN else 0.0
        mf = 2 * mp * mr / (mp + mr) if mp + mr else 0.0
        print(f"  micro  P {mp:.3f}  R {mr:.3f}  F1 {mf:.3f}   "
              f"(TP {TP} FP {FP} FN {FN})")
        rep["pooling"][f"{pool}/{head}"] = {"per_class": per, "micro_f1": mf,
                                "micro_p": mp, "micro_r": mr}

    # The constant predictor is the bar every naming readout has to clear.
    best_const, best_set = 0.0, ()
    import itertools
    for k in range(1, len(cats) + 1):
        for combo in itertools.combinations(cats, k):
            tp = sum(len(set(combo) & set(s.extra["gt_categories"])) for s in test)
            fp = sum(len(set(combo) - set(s.extra["gt_categories"])) for s in test)
            fn = sum(len(set(s.extra["gt_categories"]) - set(combo)) for s in test)
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
            if f1 > best_const:
                best_const, best_set = f1, combo
    print(f"\n  best constant predictor: micro F1 {best_const:.3f} emitting {list(best_set)}")
    print(f"  VLM reference on the same 5-way target (positives only):")
    print(f"    closed_vocab n1  micro F1 0.391   (Bulky +0.026, Containers +0.022 lift)")
    print(f"    open_cot     n1  micro F1 0.004   (540/581 empty parses)")
    rep["best_constant_f1"] = best_const

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(rep, indent=2))
        print(f"[write] {args.out_json}")


if __name__ == "__main__":
    main()
