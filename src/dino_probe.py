"""Train + evaluate linear probes on cached DINOv2 features.

Two modes:
  * `--task binary`  — AW is_candidate_location → 0/1; train on AW training,
                       test on AW testing.
  * `--task multilabel` — multi-label over the 22 (AW) or 20 (DroneWaste)
                          categories. For AW we restrict to the fine-grain-
                          annotated subset on both train and test; for
                          DroneWaste we use the full 4993 with a 70/30 split
                          stratified by site.

Reports overall + per-image-source / per-site metrics, in the same shape as
the VLM reports so they're directly comparable.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import (  # noqa: E402
    load_aerialwaste_mcml,
    load_aerialwaste_multilabel,
    load_dronewaste_multilabel,
)

CACHE_DIR = Path("/home/ids/diecidue/results/waste_vlm/dinov2_features")


def load_features(name: str, variant: str = "cls") -> dict:
    """Load a feature cache.

    variant ∈ {"cls", "patch_mean", "patch_max", "cls_mean", "cls_mean_max"}.
    Older caches only have "cls" (stored as `features`); the new format also
    has features_cls / features_patch_mean / features_patch_max.
    """
    npz = np.load(CACHE_DIR / f"{name}.npz", allow_pickle=True)
    if variant == "cls":
        feats = npz["features"]
    elif variant == "patch_mean":
        feats = npz["features_patch_mean"]
    elif variant == "patch_max":
        feats = npz["features_patch_max"]
    elif variant == "cls_mean":
        feats = np.concatenate([npz["features_cls"], npz["features_patch_mean"]], axis=1)
    elif variant == "cls_mean_max":
        feats = np.concatenate(
            [npz["features_cls"], npz["features_patch_mean"], npz["features_patch_max"]],
            axis=1,
        )
    else:
        raise ValueError(f"unknown feature variant {variant!r}")
    return {
        "features": feats,
        "labels": npz["labels"],
        "image_ids": npz["image_ids"],
        "image_sources": npz["image_sources"],
        "gt_categories": npz["gt_categories"],
    }


def expected_calibration_error(y_true: np.ndarray, p_pos: np.ndarray, n_bins: int = 10) -> float:
    y_pred = (p_pos >= 0.5).astype(int)
    confidence = np.where(y_pred == 1, p_pos, 1 - p_pos)
    correct = (y_pred == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(confidence[mask].mean() - correct[mask].mean())
    return float(ece)


def binary_probe(variant: str = "cls") -> dict:
    train = load_features("aerialwaste_training", variant=variant)
    test = load_features("aerialwaste_testing", variant=variant)

    X_train = normalize(train["features"])  # L2-normalize for stable LR
    y_train = train["labels"].astype(int)
    X_test = normalize(test["features"])
    y_test = test["labels"].astype(int)

    print(f"train={len(y_train)} ({y_train.sum()} pos), test={len(y_test)} ({y_test.sum()} pos)")

    clf = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", n_jobs=-1
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = float((pred == y_test).mean())
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, pred, average="binary", zero_division=0
    )
    brier = brier_score_loss(y_test, proba)
    ece = expected_calibration_error(y_test, proba)
    ap = average_precision_score(y_test, proba)

    overall = {
        "n": int(len(y_test)),
        "accuracy": acc,
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "brier": float(brier),
        "ece": float(ece),
        "average_precision": float(ap),
        "base_rate": float(y_test.mean()),
    }

    by_split: dict[str, dict] = {}
    srcs = test["image_sources"]
    for src in sorted(set(srcs)):
        idx = np.where(srcs == src)[0]
        if len(idx) == 0:
            continue
        yt = y_test[idx]
        pd = pred[idx]
        pp = proba[idx]
        sp, sr, sf, _ = precision_recall_fscore_support(
            yt, pd, average="binary", zero_division=0
        )
        by_split[str(src)] = {
            "n": int(len(idx)),
            "accuracy": float((pd == yt).mean()),
            "precision": float(sp),
            "recall": float(sr),
            "f1": float(sf),
            "brier": float(brier_score_loss(yt, pp)) if len(set(yt)) > 1 else float("nan"),
            "ece": expected_calibration_error(yt, pp),
            "average_precision": float(average_precision_score(yt, pp)) if len(set(yt)) > 1 else float("nan"),
        }

    return {"task": "binary", "overall": overall, "by_split": by_split}


def _load_aw_multilabel_gt() -> tuple[list[str], dict[str, set[str]], dict[str, set[str]]]:
    """Returns (categories, gt_train, gt_test). gt_* includes negatives with GT=∅,
    so the multi-label classifier sees "not waste" examples in addition to
    fine-grain-annotated positives."""
    cats_tr, samples_tr = load_aerialwaste_multilabel("/home/ids/diecidue/data/aerialwaste", "training")
    cats_te, samples_te = load_aerialwaste_multilabel("/home/ids/diecidue/data/aerialwaste", "testing")
    assert cats_tr == cats_te, "category order differs between splits"
    gt_tr = {s.image_id: set(s.extra["gt_categories"]) for s in samples_tr}
    gt_te = {s.image_id: set(s.extra["gt_categories"]) for s in samples_te}
    return cats_tr, gt_tr, gt_te


def _per_class_thresholds(Y_train: np.ndarray, scores_train: np.ndarray) -> np.ndarray:
    """Pick per-class threshold that maximises F1 on the training scores."""
    n_classes = Y_train.shape[1]
    thresholds = np.full(n_classes, 0.5, dtype=float)
    candidates = np.linspace(0.05, 0.95, 19)
    for c in range(n_classes):
        if Y_train[:, c].sum() == 0:
            continue
        best_f1 = -1.0
        best_t = 0.5
        for t in candidates:
            pred = (scores_train[:, c] >= t).astype(int)
            f1 = f1_score(Y_train[:, c], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[c] = best_t
    return thresholds


def multilabel_probe_aw(variant: str = "cls", head: str = "linear") -> dict:
    cats, gt_tr, gt_te = _load_aw_multilabel_gt()
    n_cats = len(cats)

    train = load_features("aerialwaste_training", variant=variant)
    test = load_features("aerialwaste_testing", variant=variant)

    def restrict(feat_cache: dict, gt_map: dict[str, set[str]]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        ids = list(feat_cache["image_ids"])
        keep = [i for i, x in enumerate(ids) if str(x) in gt_map]
        X = normalize(feat_cache["features"][keep])
        kept_ids = [str(ids[i]) for i in keep]
        Y = np.zeros((len(keep), n_cats), dtype=np.int32)
        for r, iid in enumerate(kept_ids):
            for c in gt_map[iid]:
                if c in cats:
                    Y[r, cats.index(c)] = 1
        return X, Y, kept_ids

    X_train, Y_train, _ = restrict(train, gt_tr)
    X_test, Y_test, _ = restrict(test, gt_te)
    print(f"AW multi-label probe: train={len(X_train)}, test={len(X_test)}, classes={n_cats}, head={head}")

    if head == "linear":
        clf = OneVsRestClassifier(
            LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"),
            n_jobs=-1,
        )
    elif head == "mlp":
        # Single hidden layer of width=512 — small enough to need real
        # feature signal, large enough to find non-linear combinations.
        clf = MLPClassifier(
            hidden_layer_sizes=(512,),
            max_iter=200,
            early_stopping=True,
            random_state=0,
        )
    else:
        raise ValueError(head)

    clf.fit(X_train, Y_train)
    scores_test = clf.predict_proba(X_test)
    scores_train = clf.predict_proba(X_train)
    if head == "linear":
        P_default = clf.predict(X_test)
    else:
        P_default = (scores_test >= 0.5).astype(int)

    thresholds = _per_class_thresholds(Y_train, scores_train)
    P_tuned = (scores_test >= thresholds).astype(int)

    rep = _ml_metrics(cats, Y_test, P_tuned)
    rep["thresholding"] = "per-class F1-tuned on train"
    rep["head"] = head
    rep["per_class_ap"] = {
        cats[c]: float(average_precision_score(Y_test[:, c], scores_test[:, c]))
        if Y_test[:, c].sum() > 0
        else None
        for c in range(n_cats)
    }
    rep["per_class_threshold"] = {cats[c]: float(thresholds[c]) for c in range(n_cats)}
    rep["default_threshold_micro_f1"] = float(
        precision_recall_fscore_support(Y_test, P_default, average="micro", zero_division=0)[2]
    )
    return rep


def _add_per_class_ap(rep: dict, cats: list[str], Y_test: np.ndarray, scores: np.ndarray) -> None:
    rep["per_class_ap"] = {
        cats[c]: float(average_precision_score(Y_test[:, c], scores[:, c]))
        if Y_test[:, c].sum() > 0
        else None
        for c in range(len(cats))
    }


def _aw_feature_lookup(variant: str) -> dict[str, np.ndarray]:
    """Build an image_id -> feature dict by pooling both AW feature caches."""
    out: dict[str, np.ndarray] = {}
    for split in ("training", "testing"):
        d = load_features(f"aerialwaste_{split}", variant=variant)
        for i, iid in enumerate(d["image_ids"]):
            out[str(iid)] = d["features"][i]
    return out


def multilabel_probe_aw_mcml(version: str = "m2", variant: str = "cls", head: str = "linear") -> dict:
    cats_tr, train_samples = load_aerialwaste_mcml("/home/ids/diecidue/data/aerialwaste", "train", version)
    cats_te, test_samples = load_aerialwaste_mcml("/home/ids/diecidue/data/aerialwaste", "test", version)
    assert cats_tr == cats_te
    cats = cats_tr
    n_cats = len(cats)

    feat_by_id = _aw_feature_lookup(variant)

    def to_arrays(samples):
        rows = []
        Y = []
        kept = []
        for s in samples:
            if s.image_id not in feat_by_id:
                continue
            rows.append(feat_by_id[s.image_id])
            kept.append(s)
            y = np.zeros(n_cats, dtype=np.int32)
            for c in s.extra["gt_categories"]:
                if c in cats:
                    y[cats.index(c)] = 1
            Y.append(y)
        return normalize(np.stack(rows)), np.stack(Y), kept

    X_train, Y_train, _ = to_arrays(train_samples)
    X_test, Y_test, test_kept = to_arrays(test_samples)
    print(f"AW mcml/{version} probe: train={len(X_train)}, test={len(X_test)}, classes={n_cats}, head={head}")

    if head == "linear":
        clf = OneVsRestClassifier(
            LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"), n_jobs=-1
        )
    elif head == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(512,), max_iter=200, early_stopping=True, random_state=0
        )
    else:
        raise ValueError(head)
    clf.fit(X_train, Y_train)
    scores_test = clf.predict_proba(X_test)
    scores_train = clf.predict_proba(X_train)
    thresholds = _per_class_thresholds(Y_train, scores_train)
    P_tuned = (scores_test >= thresholds).astype(int)
    P_default = (scores_test >= 0.5).astype(int)

    rep = _ml_metrics(cats, Y_test, P_tuned)
    rep["thresholding"] = "per-class F1-tuned on train"
    rep["head"] = head
    rep["version"] = version
    _add_per_class_ap(rep, cats, Y_test, scores_test)
    rep["per_class_threshold"] = {cats[c]: float(thresholds[c]) for c in range(n_cats)}
    rep["default_threshold_micro_f1"] = float(
        precision_recall_fscore_support(Y_test, P_default, average="micro", zero_division=0)[2]
    )

    # By image source
    by_src: dict[str, dict] = {}
    sources = np.array([s.image_source for s in test_kept])
    for src in sorted(set(sources)):
        idx = np.where(sources == src)[0]
        if len(idx) == 0:
            continue
        by_src[src] = _ml_metrics(cats, Y_test[idx], P_tuned[idx])
    rep["by_source"] = by_src
    return rep


def multilabel_probe_dw(variant: str = "cls", paper_10: bool = False) -> dict:
    from src.datasets import DRONEWASTE_PAPER_10
    cats_filter = DRONEWASTE_PAPER_10 if paper_10 else None
    cats, samples = load_dronewaste_multilabel(
        "/home/ids/diecidue/data/dronewaste", categories_filter=cats_filter
    )
    n_cats = len(cats)
    feat = load_features("dronewaste", variant=variant)

    # Site-stratified 70/30 split (so test sites are also seen during training -
    # this measures the per-image generalization, not cross-site).
    site_to_idx: dict[str, list[int]] = defaultdict(list)
    ids = list(feat["image_ids"])
    id_to_idx = {str(x): i for i, x in enumerate(ids)}
    for s in samples:
        if s.image_id in id_to_idx:
            site_to_idx[s.image_source].append(id_to_idx[s.image_id])

    rng = np.random.default_rng(0)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for site, idxs in site_to_idx.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        cut = int(len(idxs) * 0.7)
        train_idx.extend(idxs[:cut])
        test_idx.extend(idxs[cut:])

    X_train = normalize(feat["features"][train_idx])
    X_test = normalize(feat["features"][test_idx])
    Y_train = np.zeros((len(train_idx), n_cats), dtype=np.int32)
    Y_test = np.zeros((len(test_idx), n_cats), dtype=np.int32)

    id_to_sample = {s.image_id: s for s in samples}
    for r, fi in enumerate(train_idx):
        s = id_to_sample[str(ids[fi])]
        for c in s.extra["gt_categories"]:
            if c in cats:
                Y_train[r, cats.index(c)] = 1
    for r, fi in enumerate(test_idx):
        s = id_to_sample[str(ids[fi])]
        for c in s.extra["gt_categories"]:
            if c in cats:
                Y_test[r, cats.index(c)] = 1

    print(f"DroneWaste multi-label probe: train={len(X_train)}, test={len(X_test)}, classes={n_cats}")

    clf = OneVsRestClassifier(
        LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"),
        n_jobs=-1,
    )
    clf.fit(X_train, Y_train)
    scores_test = clf.predict_proba(X_test)
    scores_train = clf.predict_proba(X_train)
    thresholds = _per_class_thresholds(Y_train, scores_train)
    P_tuned = (scores_test >= thresholds).astype(int)

    rep = _ml_metrics(cats, Y_test, P_tuned)
    rep["thresholding"] = "per-class F1-tuned on train"
    _add_per_class_ap(rep, cats, Y_test, scores_test)
    rep["per_class_threshold"] = {cats[c]: float(thresholds[c]) for c in range(len(cats))}
    return rep


def _ml_metrics(categories: list[str], Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    # Micro
    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="micro", zero_division=0
    )
    # Macro (only over classes with support)
    support = Y_true.sum(axis=0)
    classes_with_support = np.where(support > 0)[0]
    per_class: dict = {}
    for ci, name in enumerate(categories):
        sup = int(support[ci])
        if sup == 0 and Y_pred[:, ci].sum() == 0:
            per_class[name] = {"support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue
        p, r, f, _ = precision_recall_fscore_support(
            Y_true[:, ci], Y_pred[:, ci], average="binary", zero_division=0
        )
        per_class[name] = {
            "support": sup,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
        }
    macro_p = float(np.mean([per_class[c]["precision"] for ci, c in enumerate(categories) if support[ci] > 0]))
    macro_r = float(np.mean([per_class[c]["recall"] for ci, c in enumerate(categories) if support[ci] > 0]))
    macro_f = float(np.mean([per_class[c]["f1"] for ci, c in enumerate(categories) if support[ci] > 0]))

    # Image-level exact / jaccard
    exact = float((Y_true == Y_pred).all(axis=1).mean())
    inter = (Y_true & Y_pred).sum(axis=1)
    union = (Y_true | Y_pred).sum(axis=1)
    jaccard = float(np.where(union == 0, 1.0, inter / np.maximum(1, union)).mean())

    return {
        "task": "multilabel",
        "n": int(Y_true.shape[0]),
        "n_categories": int(Y_true.shape[1]),
        "micro": {"precision": float(micro_p), "recall": float(micro_r), "f1": float(micro_f)},
        "macro": {
            "precision": macro_p,
            "recall": macro_r,
            "f1": macro_f,
            "n_classes_with_support": int(len(classes_with_support)),
        },
        "image_level": {
            "exact_match": exact,
            "jaccard_mean": jaccard,
            "avg_predicted_per_image": float(Y_pred.sum(axis=1).mean()),
        },
        "per_class": per_class,
    }


def print_binary(rep: dict) -> None:
    o = rep["overall"]
    print(f"=== AW binary linear probe ===")
    print(f"n={o['n']}, base_rate={o['base_rate']:.3f}")
    print(f"  acc={o['accuracy']:.3f}  P={o['precision']:.3f}  R={o['recall']:.3f}  F1={o['f1']:.3f}")
    print(f"  AP={o['average_precision']:.3f}  Brier={o['brier']:.3f}  ECE={o['ece']:.3f}")
    print()
    print("by image source:")
    for src, m in rep["by_split"].items():
        print(f"  {src:14s} n={m['n']:4d}  acc={m['accuracy']:.3f}  F1={m['f1']:.3f}  AP={m['average_precision']:.3f}")


def print_multilabel(name: str, rep: dict) -> None:
    print(f"=== {name} multi-label linear probe ===")
    print(f"n={rep['n']}, categories={rep['n_categories']}, thresholding={rep.get('thresholding','default 0.5')}")
    if "default_threshold_micro_f1" in rep:
        print(f"  (default-threshold micro F1 = {rep['default_threshold_micro_f1']:.3f}, F1-tuned below)")
    print(f"  micro  P={rep['micro']['precision']:.3f}  R={rep['micro']['recall']:.3f}  F1={rep['micro']['f1']:.3f}")
    print(f"  macro  P={rep['macro']['precision']:.3f}  R={rep['macro']['recall']:.3f}  F1={rep['macro']['f1']:.3f}  "
          f"(over {rep['macro']['n_classes_with_support']} classes with support)")
    img = rep["image_level"]
    print(f"  image  exact={img['exact_match']:.3f}  jaccard={img['jaccard_mean']:.3f}  "
          f"avg_pred/img={img['avg_predicted_per_image']:.2f}")
    print()
    items = sorted(rep["per_class"].items(), key=lambda kv: -kv[1]["support"])
    aps = rep.get("per_class_ap", {})
    print(f"  {'class':40s}  supp   P     R     F1    AP")
    for c, m in items:
        if m["support"] == 0 and m.get("precision", 0) == 0:
            continue
        ap = aps.get(c)
        ap_s = f"{ap:.2f}" if ap is not None else "  — "
        print(f"  {c[:40]:40s}  {m['support']:>4d}  {m['precision']:.2f}  "
              f"{m['recall']:.2f}  {m['f1']:.2f}  {ap_s}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--task",
        choices=["binary", "multilabel_aw", "multilabel_aw_mcml", "multilabel_dw", "all"],
        default="all",
    )
    p.add_argument("--aw-mcml-version", choices=["m2", "m4"], default="m2")
    p.add_argument(
        "--features",
        choices=["cls", "patch_mean", "patch_max", "cls_mean", "cls_mean_max"],
        default="cls",
    )
    p.add_argument("--head", choices=["linear", "mlp"], default="linear")
    p.add_argument("--dw-paper-10", action="store_true", help="DroneWaste: restrict to the 10 classes the paper reports")
    p.add_argument("--out-json", type=Path, default=Path("/home/ids/diecidue/results/waste_vlm/dino_probe_report.json"))
    args = p.parse_args()

    print(f"[probe] feature variant = {args.features}")
    out: dict = {"feature_variant": args.features}
    if args.task in ("binary", "all"):
        out["aerialwaste_binary"] = binary_probe(args.features)
        print_binary(out["aerialwaste_binary"])
        print()
    if args.task in ("multilabel_aw", "all"):
        out["aerialwaste_multilabel"] = multilabel_probe_aw(args.features, head=args.head)
        print_multilabel("AW", out["aerialwaste_multilabel"])
        print()
    if args.task == "multilabel_aw_mcml":
        key = f"aerialwaste_mcml_{args.aw_mcml_version}"
        out[key] = multilabel_probe_aw_mcml(
            version=args.aw_mcml_version, variant=args.features, head=args.head
        )
        print_multilabel(f"AW mcml/{args.aw_mcml_version}", out[key])
        print()
    if args.task in ("multilabel_dw", "all"):
        out["dronewaste_multilabel"] = multilabel_probe_dw(args.features, paper_10=args.dw_paper_10)
        tag = "DroneWaste (paper-10)" if args.dw_paper_10 else "DroneWaste"
        print_multilabel(tag, out["dronewaste_multilabel"])
        print()

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as f:
            json.dump(out, f, indent=2)
        print(f"[saved] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
