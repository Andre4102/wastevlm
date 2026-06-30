"""Aggregate multi-label JSONL into per-class + micro/macro metrics."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _safe(n: int, d: int) -> float:
    return n / d if d else 0.0


def per_class_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {}
    categories: list[str] = rows[0]["categories"]
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)
    tn: dict[str, int] = defaultdict(int)

    for r in rows:
        gt = set(r["sample"]["extra"]["gt_categories"])
        pred = set(r["predicted"])
        for c in categories:
            in_gt = c in gt
            in_pred = c in pred
            if in_gt and in_pred:
                tp[c] += 1
            elif in_gt and not in_pred:
                fn[c] += 1
            elif not in_gt and in_pred:
                fp[c] += 1
            else:
                tn[c] += 1

    per_class = {}
    for c in categories:
        precision = _safe(tp[c], tp[c] + fp[c])
        recall = _safe(tp[c], tp[c] + fn[c])
        f1 = _safe(2 * precision * recall, precision + recall)
        per_class[c] = {
            "tp": tp[c], "fp": fp[c], "fn": fn[c], "tn": tn[c],
            "precision": precision, "recall": recall, "f1": f1,
            "support": tp[c] + fn[c],
        }

    # Micro: pool tp/fp/fn across classes
    tp_sum = sum(tp.values())
    fp_sum = sum(fp.values())
    fn_sum = sum(fn.values())
    micro_p = _safe(tp_sum, tp_sum + fp_sum)
    micro_r = _safe(tp_sum, tp_sum + fn_sum)
    micro_f = _safe(2 * micro_p * micro_r, micro_p + micro_r)

    # Macro: arithmetic mean of per-class F1
    classes_with_support = [c for c in categories if per_class[c]["support"] > 0]
    macro_p = sum(per_class[c]["precision"] for c in classes_with_support) / max(1, len(classes_with_support))
    macro_r = sum(per_class[c]["recall"] for c in classes_with_support) / max(1, len(classes_with_support))
    macro_f = sum(per_class[c]["f1"] for c in classes_with_support) / max(1, len(classes_with_support))

    # Image-level exact match + jaccard
    exact_match = 0
    jaccard_sum = 0.0
    n_pred_per_image = []
    for r in rows:
        gt = set(r["sample"]["extra"]["gt_categories"])
        pred = set(r["predicted"])
        if gt == pred:
            exact_match += 1
        union = gt | pred
        jaccard_sum += (len(gt & pred) / len(union)) if union else 1.0
        n_pred_per_image.append(len(pred))

    n = len(rows)
    return {
        "n": n,
        "n_categories": len(categories),
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f},
        "macro": {
            "precision": macro_p, "recall": macro_r, "f1": macro_f,
            "n_classes_with_support": len(classes_with_support),
        },
        "image_level": {
            "exact_match": exact_match / n,
            "jaccard_mean": jaccard_sum / n,
            "avg_predicted_per_image": sum(n_pred_per_image) / n,
        },
        "per_class": per_class,
    }


def print_report(rep: dict) -> None:
    print(f"n={rep['n']}, categories={rep['n_categories']}")
    print(f"  micro  P={rep['micro']['precision']:.3f}  R={rep['micro']['recall']:.3f}  F1={rep['micro']['f1']:.3f}")
    print(f"  macro  P={rep['macro']['precision']:.3f}  R={rep['macro']['recall']:.3f}  F1={rep['macro']['f1']:.3f}  "
          f"(over {rep['macro']['n_classes_with_support']} classes with support)")
    img = rep["image_level"]
    print(f"  image  exact={img['exact_match']:.3f}  jaccard={img['jaccard_mean']:.3f}  "
          f"avg_pred/img={img['avg_predicted_per_image']:.2f}")
    print()
    print("per-class (sorted by support):")
    print(f"  {'class':40s}  supp   P     R     F1")
    items = sorted(rep["per_class"].items(), key=lambda kv: -kv[1]["support"])
    for c, m in items:
        if m["support"] == 0 and m["tp"] + m["fp"] == 0:
            continue
        print(f"  {c[:40]:40s}  {m['support']:>4d}  {m['precision']:.2f}  "
              f"{m['recall']:.2f}  {m['f1']:.2f}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()
    rows = load(args.jsonl)
    rep = per_class_metrics(rows)
    print_report(rep)
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        print(f"[saved] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
