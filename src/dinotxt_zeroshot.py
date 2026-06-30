"""dino.txt zero-shot multi-label classification on AerialWaste + DroneWaste.

For each (dataset, class-set) combo we:
 1. Encode all images with dino.txt -> [N, 2048] L2-normed image features.
 2. Encode each class name with `templates` -> mean -> [C, 2048] L2-normed.
 3. Compute cosine sim -> [N, C].
 4. Tune per-class F1-maximising thresholds on the train scores.
 5. Apply thresholds on the test scores -> binary predictions.
 6. Report micro / macro F1 + per-class F1 + per-class AP.

No model parameters are trained — only operating points are chosen on
train data, matching the regime of the supervised DINOv2 probe baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import (  # noqa: E402
    DRONEWASTE_PAPER_10,
    load_aerialwaste_mcml,
    load_dronewaste_multilabel,
)
from src.dinotxt_runner import AERIAL_TEMPLATES, DinoTxtRunner  # noqa: E402


# Some class names are ambiguous out-of-context; expand the prompt term.
# Anything not in this dict falls back to the class name lowercased.
CLASS_NAME_TO_PROMPT_TERM = {
    "Bulky items": "bulky waste items",
    "Containers": "waste containers",
    "Unknown material": "unidentified waste material",
    "Construction and demolition materials": "construction and demolition waste",
    "Metal barrels": "metal barrels",
    "Plastic packaging": "plastic packaging waste",
    "Pallets": "wooden pallets",
    "Scrap": "scrap metal pile",
    "Vehicles": "abandoned vehicles",
    "Tyres": "pile of tyres",
    "Asbestos": "asbestos roofing sheets",
    "Textile": "textile waste",
    "Mixed items": "mixed waste pile",
    "Rubble": "rubble pile",
    "Plastic": "plastic waste",
    "Rubble/excavated earth and rocks": "rubble and excavated earth",
    "Sludge-Zootechnical waste-Manure": "sludge and manure waste",
    "Wood": "wood waste",
    "Tires": "pile of tires",
    "Other waste": "miscellaneous waste",
    "Excavation materials": "excavated soil and rocks",
    "Furniture": "discarded furniture",
    "Appliances": "discarded household appliances",
    "Electronic equipment": "electronic waste",
    "Paper": "paper waste",
    "Foundry": "foundry waste",
    "Asphalt milling": "asphalt milling waste",
    "Wood": "wood waste",
}


def prompt_term(class_name: str) -> str:
    return CLASS_NAME_TO_PROMPT_TERM.get(class_name, class_name.lower())


class _ImageList(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), idx


def encode_image_set(
    runner: DinoTxtRunner,
    paths: list[Path],
    batch_size: int = 32,
    num_workers: int = 4,
) -> np.ndarray:
    ds = _ImageList(paths, runner.transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    feats = np.zeros((len(paths), 2048), dtype=np.float32)
    seen = 0
    for px, idx in loader:
        f = runner.encode_image_batch(px).cpu().numpy()
        feats[idx.numpy()] = f
        seen += len(idx)
        if seen % (batch_size * 8) == 0 or seen == len(paths):
            print(f"  encoded {seen}/{len(paths)}", flush=True)
    return feats


def per_class_thresholds(Y_train: np.ndarray, scores_train: np.ndarray) -> np.ndarray:
    """Pick per-class threshold that maximises F1 on the training scores."""
    n_classes = Y_train.shape[1]
    thresholds = np.full(n_classes, 0.5, dtype=float)
    # dino.txt cosine scores live in roughly [0.0, 0.4] for aerial waste, so we
    # sweep a finer grid in that range.
    candidates = np.linspace(0.10, 0.40, 31)
    for c in range(n_classes):
        if Y_train[:, c].sum() == 0:
            continue
        best_f1 = -1.0
        best_t = float(candidates[0])
        for t in candidates:
            pred = (scores_train[:, c] >= t).astype(int)
            f1 = f1_score(Y_train[:, c], pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)
        thresholds[c] = best_t
    return thresholds


def ml_metrics(
    cats: list[str], Y_true: np.ndarray, Y_pred: np.ndarray, scores: np.ndarray
) -> dict:
    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="micro", zero_division=0
    )
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        Y_true, Y_pred, average="macro", zero_division=0
    )
    per_class = {}
    for c, name in enumerate(cats):
        if Y_true[:, c].sum() == 0:
            per_class[name] = {"support": 0, "f1": None, "precision": None, "recall": None, "ap": None}
            continue
        p, r, f, _ = precision_recall_fscore_support(
            Y_true[:, c], Y_pred[:, c], average="binary", zero_division=0
        )
        ap = float(average_precision_score(Y_true[:, c], scores[:, c]))
        per_class[name] = {
            "support": int(Y_true[:, c].sum()),
            "f1": float(f),
            "precision": float(p),
            "recall": float(r),
            "ap": ap,
        }
    return {
        "micro": {"f1": float(micro_f), "precision": float(micro_p), "recall": float(micro_r)},
        "macro": {"f1": float(macro_f), "precision": float(macro_p), "recall": float(macro_r)},
        "per_class": per_class,
        "n_test": int(Y_true.shape[0]),
        "n_classes": int(len(cats)),
    }


def run_aw_mcml(runner: DinoTxtRunner, root: str, version: str, out_dir: Path) -> dict:
    print(f"\n=== AW mcml/{version} ===")
    cats_tr, train = load_aerialwaste_mcml(root, split="train", version=version)
    cats_te, test = load_aerialwaste_mcml(root, split="test", version=version)
    assert cats_tr == cats_te
    cats = cats_tr
    # AW distributes a 486-image PNEO subset whose image files aren't shipped
    # with the main images0-6 zips; filter to images present on disk so we match
    # the regime the rest of the baselines used.
    before_tr, before_te = len(train), len(test)
    train = [s for s in train if s.image_path.exists()]
    test = [s for s in test if s.image_path.exists()]
    if (before_tr - len(train)) or (before_te - len(test)):
        print(f"  filtered missing-on-disk: train -{before_tr-len(train)}, test -{before_te-len(test)}")
    print(f"  train={len(train)}  test={len(test)}  classes={len(cats)}")

    train_paths = [s.image_path for s in train]
    test_paths = [s.image_path for s in test]
    print("  encoding train images...")
    X_train = encode_image_set(runner, train_paths)
    print("  encoding test images...")
    X_test = encode_image_set(runner, test_paths)

    Y_train = np.zeros((len(train), len(cats)), dtype=np.int32)
    for r, s in enumerate(train):
        for c in s.extra["gt_categories"]:
            if c in cats:
                Y_train[r, cats.index(c)] = 1
    Y_test = np.zeros((len(test), len(cats)), dtype=np.int32)
    for r, s in enumerate(test):
        for c in s.extra["gt_categories"]:
            if c in cats:
                Y_test[r, cats.index(c)] = 1

    text_feats = runner.encode_text_classes(
        [prompt_term(c) for c in cats], templates=AERIAL_TEMPLATES,
    ).cpu().numpy()  # [C, 2048]
    scores_train = X_train @ text_feats.T
    scores_test = X_test @ text_feats.T

    thr = per_class_thresholds(Y_train, scores_train)
    P_test = (scores_test >= thr).astype(int)
    rep = ml_metrics(cats, Y_test, P_test, scores_test)
    rep["thresholding"] = "per-class F1-tuned on train cosine sims"
    rep["per_class_threshold"] = {cats[c]: float(thr[c]) for c in range(len(cats))}
    rep["prompt_terms"] = {c: prompt_term(c) for c in cats}
    rep["task"] = f"aw_mcml_{version}"

    out = out_dir / f"dinotxt_zeroshot_aw_{version}.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"  micro F1 = {rep['micro']['f1']:.4f}  macro F1 = {rep['macro']['f1']:.4f}")
    print(f"  saved -> {out}")
    return rep


def run_dw_paper10(runner: DinoTxtRunner, root: str, out_dir: Path) -> dict:
    print("\n=== DroneWaste paper-10 ===")
    cats, samples = load_dronewaste_multilabel(root, categories_filter=DRONEWASTE_PAPER_10)
    n_cats = len(cats)

    # Reproduce the existing DW probe's 70/30 site-stratified split (seed=0).
    site_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        site_to_idx[s.image_source].append(i)
    rng = np.random.default_rng(0)
    train_idx: list[int] = []
    test_idx: list[int] = []
    for site, idxs in site_to_idx.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        cut = int(len(idxs) * 0.7)
        train_idx.extend(idxs[:cut])
        test_idx.extend(idxs[cut:])
    print(f"  train={len(train_idx)}  test={len(test_idx)}  classes={n_cats}")

    paths = [s.image_path for s in samples]
    print(f"  encoding {len(paths)} images...")
    X_all = encode_image_set(runner, paths)
    X_train = X_all[train_idx]
    X_test = X_all[test_idx]

    Y_train = np.zeros((len(train_idx), n_cats), dtype=np.int32)
    Y_test = np.zeros((len(test_idx), n_cats), dtype=np.int32)
    for r, i in enumerate(train_idx):
        for c in samples[i].extra["gt_categories"]:
            if c in cats:
                Y_train[r, cats.index(c)] = 1
    for r, i in enumerate(test_idx):
        for c in samples[i].extra["gt_categories"]:
            if c in cats:
                Y_test[r, cats.index(c)] = 1

    text_feats = runner.encode_text_classes(
        [prompt_term(c) for c in cats], templates=AERIAL_TEMPLATES,
    ).cpu().numpy()
    scores_train = X_train @ text_feats.T
    scores_test = X_test @ text_feats.T

    thr = per_class_thresholds(Y_train, scores_train)
    P_test = (scores_test >= thr).astype(int)
    rep = ml_metrics(cats, Y_test, P_test, scores_test)
    rep["thresholding"] = "per-class F1-tuned on train cosine sims"
    rep["per_class_threshold"] = {cats[c]: float(thr[c]) for c in range(len(cats))}
    rep["prompt_terms"] = {c: prompt_term(c) for c in cats}
    rep["task"] = "dw_paper_10"

    out = out_dir / "dinotxt_zeroshot_dw_paper10.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"  micro F1 = {rep['micro']['f1']:.4f}  macro F1 = {rep['macro']['f1']:.4f}")
    print(f"  saved -> {out}")
    return rep


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--aw-root", type=str, default="/home/ids/diecidue/data/aerialwaste")
    p.add_argument("--dw-root", type=str, default="/home/ids/diecidue/data/dronewaste")
    p.add_argument("--out-dir", type=Path, default=Path("/home/ids/diecidue/results/waste_vlm"))
    p.add_argument("--image-size", type=int, default=518)
    p.add_argument("--datasets", type=str, default="aw_m2,aw_m4,dw_paper10",
                   help="comma-separated subset to run")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runner = DinoTxtRunner(image_size=args.image_size)
    print(f"[model] dino.txt loaded; image_size={args.image_size}")

    sel = set(args.datasets.split(","))
    if "aw_m2" in sel:
        run_aw_mcml(runner, args.aw_root, "m2", args.out_dir)
    if "aw_m4" in sel:
        run_aw_mcml(runner, args.aw_root, "m4", args.out_dir)
    if "dw_paper10" in sel:
        run_dw_paper10(runner, args.dw_root, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
