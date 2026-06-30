"""Linear probe on the joint DW + AW VQA split.

Reads `data/vqa/split_index.jsonl` (produced by src.vqa_split, owned by the
ANNOTATION track — read-only here). Trains a multi-label OneVsRest
LogisticRegression head over the 15-class union (10 DW paper-10 + 5 AW m2)
and reports per-class F1 + macro-F1 on the val split.

Features are extracted with src.vision_encoder.VisionEncoder and cached to
disk under `results/waste_vlm/encoder_probe/features/` so repeated runs with
the same encoder are fast.

Usage
-----
    python -m src.encoder_probe --encoder radio-l --device cuda
    python -m src.encoder_probe --encoder dinov3-b --no-cache
    python -m src.encoder_probe --encoder radio-l --encoder dinov3-b  # compare both
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vision_encoder import VisionEncoder  # noqa: E402
from src.vqa_labels import UNION as ALL_LABELS  # noqa: E402

SPLIT_INDEX = Path("/home/ids/diecidue/data/vqa/split_index.jsonl")
FEATURE_CACHE = Path("/home/ids/diecidue/results/waste_vlm/encoder_probe/features")
RESULTS_DIR = Path("/home/ids/diecidue/results/waste_vlm/encoder_probe")

_BATCH = 32


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

class SplitRecord(NamedTuple):
    image_path: str
    dataset: str
    image_id: str
    split: str            # "train" | "val" | "test"
    gt_categories: list[str]


def load_split_index() -> list[SplitRecord]:
    if not SPLIT_INDEX.exists():
        raise FileNotFoundError(
            f"{SPLIT_INDEX} not found. "
            "This file is produced by the ANNOTATION track (src.vqa_split). "
            "Wait for it to be generated before running this probe."
        )
    records: list[SplitRecord] = []
    for line in SPLIT_INDEX.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        records.append(SplitRecord(
            image_path=r["image_path"],
            dataset=r["dataset"],
            image_id=r["image_id"],
            split=r["split"],
            gt_categories=r["gt_categories"],
        ))
    return records


# ---------------------------------------------------------------------------
# Feature extraction + caching
# ---------------------------------------------------------------------------

def _cache_path(encoder_id: str, image_size: int) -> Path:
    return FEATURE_CACHE / f"{encoder_id}_{image_size}.npz"


def extract_and_cache(
    encoder: VisionEncoder,
    records: list[SplitRecord],
    batch_size: int = _BATCH,
    force: bool = False,
) -> dict:
    """Extract CLS features for all records and cache to disk.

    Returns a dict with keys: features [N,D], image_ids [N], splits [N],
    gt_categories [N] (object array of lists).
    """
    from PIL import Image

    cache_file = _cache_path(encoder.encoder_id, encoder.image_size)
    if cache_file.exists() and not force:
        print(f"[cache] loading {cache_file}")
        npz = np.load(cache_file, allow_pickle=True)
        return {k: npz[k] for k in npz.files}

    FEATURE_CACHE.mkdir(parents=True, exist_ok=True)
    feats: list[np.ndarray] = []
    ids: list[str] = []
    splits: list[str] = []
    cats: list = []

    buf_imgs: list[Image.Image] = []
    buf_meta: list[SplitRecord] = []

    def flush() -> None:
        out = encoder.encode(buf_imgs)
        feats.append(out.cls.cpu().numpy())
        for r in buf_meta:
            ids.append(r.image_id)
            splits.append(r.split)
            cats.append(r.gt_categories)
        buf_imgs.clear()
        buf_meta.clear()

    for rec in tqdm(records, desc=f"extract {encoder.encoder_id}"):
        try:
            img = Image.open(rec.image_path).convert("RGB")
        except Exception as e:
            print(f"  [skip] {rec.image_path}: {e}", file=sys.stderr)
            continue
        buf_imgs.append(img)
        buf_meta.append(rec)
        if len(buf_imgs) >= batch_size:
            flush()
    if buf_imgs:
        flush()

    features_arr = np.concatenate(feats, axis=0)
    ids_arr = np.array(ids)
    splits_arr = np.array(splits)
    cats_arr = np.array(cats, dtype=object)

    np.savez(cache_file, features=features_arr, image_ids=ids_arr,
             splits=splits_arr, gt_categories=cats_arr)
    print(f"[cache] saved {cache_file}  shape={features_arr.shape}")
    return {"features": features_arr, "image_ids": ids_arr,
            "splits": splits_arr, "gt_categories": cats_arr}


# ---------------------------------------------------------------------------
# Probe training and evaluation
# ---------------------------------------------------------------------------

def build_label_matrix(gt_categories: np.ndarray, label_list: list[str]) -> np.ndarray:
    """Convert object array of lists into [N, C] binary matrix."""
    n_cls = len(label_list)
    label_index = {lb: i for i, lb in enumerate(label_list)}
    Y = np.zeros((len(gt_categories), n_cls), dtype=np.int32)
    for row, cats in enumerate(gt_categories):
        for c in cats:
            if c in label_index:
                Y[row, label_index[c]] = 1
    return Y


def _per_class_thresholds(Y_train: np.ndarray, scores_train: np.ndarray) -> np.ndarray:
    """Per-class threshold maximising F1 on training scores."""
    n_cls = Y_train.shape[1]
    thresholds = np.full(n_cls, 0.5)
    for c in range(n_cls):
        if Y_train[:, c].sum() == 0:
            continue
        best_f1, best_t = -1.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            f1 = f1_score(Y_train[:, c], (scores_train[:, c] >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        thresholds[c] = best_t
    return thresholds


def run_probe(cache: dict, label_list: list[str], eval_split: str = "val") -> dict:
    """Train on 'train' split, eval on eval_split.

    Returns a metrics dict.
    """
    splits = cache["splits"]
    features = cache["features"]
    gt_cats = cache["gt_categories"]

    tr_mask = splits == "train"
    ev_mask = splits == eval_split

    X_train = normalize(features[tr_mask])
    X_eval = normalize(features[ev_mask])
    Y_train = build_label_matrix(gt_cats[tr_mask], label_list)
    Y_eval = build_label_matrix(gt_cats[ev_mask], label_list)

    n_tr_pos = int(Y_train.any(axis=1).sum())
    n_ev_pos = int(Y_eval.any(axis=1).sum())
    print(f"  train={len(X_train)} ({n_tr_pos} pos)  {eval_split}={len(X_eval)} ({n_ev_pos} pos)  classes={len(label_list)}")

    if len(X_train) == 0 or len(X_eval) == 0:
        raise RuntimeError(f"Empty split: train={len(X_train)} {eval_split}={len(X_eval)}")

    clf = OneVsRestClassifier(
        LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"),
        n_jobs=-1,
    )
    clf.fit(X_train, Y_train)
    scores_tr = clf.predict_proba(X_train)
    scores_ev = clf.predict_proba(X_eval)

    thresholds = _per_class_thresholds(Y_train, scores_tr)
    Y_pred = (scores_ev >= thresholds).astype(int)

    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
        Y_eval, Y_pred, average="micro", zero_division=0
    )
    support = Y_eval.sum(axis=0)

    per_class: dict = {}
    macro_f_vals: list[float] = []
    for ci, name in enumerate(label_list):
        sup = int(support[ci])
        p, r, f, _ = precision_recall_fscore_support(
            Y_eval[:, ci], Y_pred[:, ci], average="binary", zero_division=0
        )
        ap = (
            float(average_precision_score(Y_eval[:, ci], scores_ev[:, ci]))
            if sup > 0 else None
        )
        per_class[name] = {
            "support": sup,
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "ap": ap,
            "threshold": float(thresholds[ci]),
        }
        if sup > 0:
            macro_f_vals.append(float(f))

    macro_f = float(np.mean(macro_f_vals)) if macro_f_vals else 0.0

    return {
        "eval_split": eval_split,
        "n_train": int(len(X_train)),
        "n_eval": int(len(X_eval)),
        "n_classes": len(label_list),
        "micro": {"precision": float(micro_p), "recall": float(micro_r), "f1": float(micro_f)},
        "macro_f1": macro_f,
        "per_class": per_class,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(encoder_id: str, rep: dict) -> None:
    print(f"\n=== {encoder_id} — multi-label probe on {rep['eval_split']} split ===")
    print(f"  n_train={rep['n_train']}  n_eval={rep['n_eval']}  classes={rep['n_classes']}")
    mi = rep["micro"]
    print(f"  micro   P={mi['precision']:.3f}  R={mi['recall']:.3f}  F1={mi['f1']:.3f}")
    print(f"  macro F1 (over classes with support) = {rep['macro_f1']:.3f}")
    print()
    items = sorted(rep["per_class"].items(), key=lambda kv: -kv[1]["support"])
    print(f"  {'class':42s}  supp   P     R     F1    AP    thr")
    for c, m in items:
        if m["support"] == 0:
            continue
        ap_s = f"{m['ap']:.2f}" if m["ap"] is not None else "  — "
        print(f"  {c[:42]:42s}  {m['support']:>4d}  {m['precision']:.2f}  "
              f"{m['recall']:.2f}  {m['f1']:.2f}  {ap_s}  {m['threshold']:.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", nargs="+", default=["radio-l"],
                   help="One or more encoder_id values to probe")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--image-size", type=int, default=None,
                   help="Override default image_size for the encoder")
    p.add_argument("--eval-split", default="val", choices=["val", "test"])
    p.add_argument("--batch-size", type=int, default=_BATCH)
    p.add_argument("--no-cache", action="store_true", help="Re-extract features even if cache exists")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args()

    records = load_split_index()
    print(f"[split_index] {len(records):,} records")

    all_results: dict = {}
    for encoder_id in args.encoder:
        print(f"\n[encoder] {encoder_id}")
        enc = VisionEncoder(encoder_id, device=args.device, image_size=args.image_size)
        cache = extract_and_cache(enc, records, batch_size=args.batch_size, force=args.no_cache)
        rep = run_probe(cache, ALL_LABELS, eval_split=args.eval_split)
        rep["encoder_id"] = encoder_id
        print_report(encoder_id, rep)
        all_results[encoder_id] = rep

    # Summary comparison if multiple encoders
    if len(args.encoder) > 1:
        print("\n=== Summary ===")
        print(f"  {'encoder':<14s}  micro_F1  macro_F1")
        for eid, rep in all_results.items():
            print(f"  {eid:<14s}  {rep['micro']['f1']:.3f}     {rep['macro_f1']:.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.out_dir / f"probe_{args.eval_split}.json"
    with out_file.open("w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[saved] {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
