"""Aggregate baseline JSONL into a metrics report.

Computes Q1 / Q3 binary metrics overall and split by image_source. Open-ended
Q2/Q4 responses are not auto-graded — sampled for human review.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics import binary_metrics, calibration_metrics  # noqa: E402
from src.prompts import LETTER_TO_BINARY, parse_letter, parse_yes_no  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _q1_predictions(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return y_true, y_text_pred, p_yes, mask_known."""
    y_true, y_pred, p_yes, mask = [], [], [], []
    for r in rows:
        q1 = r["responses"].get("Q1") or {}
        text = q1.get("text", "")
        parsed = parse_yes_no(text)
        # calibrated probability
        py, pn = q1.get("p_yes"), q1.get("p_no")
        if py is not None and pn is not None and (py + pn) > 0:
            prob = py / (py + pn)
        elif parsed is not None:
            prob = float(parsed)
        else:
            prob = 0.5
        y_true.append(r["sample"]["label"])
        y_pred.append(parsed if parsed is not None else int(prob >= 0.5))
        p_yes.append(prob)
        mask.append(parsed is not None)
    return (
        np.array(y_true, dtype=int),
        np.array(y_pred, dtype=int),
        np.array(p_yes, dtype=float),
        np.array(mask, dtype=bool),
    )


def _q3_predictions(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true, y_pred, p_pos = [], [], []
    for r in rows:
        q3 = r["responses"].get("Q3") or {}
        letter = parse_letter(q3.get("text", ""))
        lp = q3.get("letter_probs") or {}
        if lp:
            total = sum(lp.values()) or 1.0
            prob_pos = lp.get("b", 0.0) / total
        else:
            prob_pos = float(letter == "b") if letter else 0.5
        y_true.append(r["sample"]["label"])
        y_pred.append(LETTER_TO_BINARY.get(letter or "", 0))
        p_pos.append(prob_pos)
    return (
        np.array(y_true, dtype=int),
        np.array(y_pred, dtype=int),
        np.array(p_pos, dtype=float),
    )


def report(rows: list[dict]) -> dict:
    out: dict = {"n_total": len(rows), "by_split": {}, "overall": {}}

    # Q1
    y_true, y_pred, p_yes, parsed_mask = _q1_predictions(rows)
    overall_q1 = {
        **binary_metrics(y_true, y_pred),
        **calibration_metrics(y_true, p_yes),
        "parse_rate": float(parsed_mask.mean()) if len(parsed_mask) else 0.0,
    }
    out["overall"]["Q1"] = overall_q1

    # Q3
    y_true3, y_pred3, p_pos3 = _q3_predictions(rows)
    out["overall"]["Q3"] = {
        **binary_metrics(y_true3, y_pred3),
        **calibration_metrics(y_true3, p_pos3),
    }

    # By image source
    src_groups: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        src_groups[r["sample"]["image_source"]].append(i)

    for src, idx in sorted(src_groups.items()):
        idx_arr = np.array(idx)
        out["by_split"][src] = {
            "n": int(len(idx)),
            "Q1": {
                **binary_metrics(y_true[idx_arr], y_pred[idx_arr]),
                **calibration_metrics(y_true[idx_arr], p_yes[idx_arr]),
            },
            "Q3": {
                **binary_metrics(y_true3[idx_arr], y_pred3[idx_arr]),
                **calibration_metrics(y_true3[idx_arr], p_pos3[idx_arr]),
            },
        }

    # Q2 / Q4 sample (text only)
    out["q2_sample"] = [
        {
            "image_path": r["sample"]["image_path"],
            "label": r["sample"]["label"],
            "text": (r["responses"].get("Q2") or {}).get("text", ""),
        }
        for r in rows[:8]
    ]
    out["q4_sample"] = [
        {
            "image_path": r["sample"]["image_path"],
            "label": r["sample"]["label"],
            "text": (r["responses"].get("Q4") or {}).get("text", ""),
        }
        for r in rows[:8]
    ]
    return out


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def print_report(rep: dict) -> None:
    print(f"=== n={rep['n_total']} ===\n")
    for tag in ("Q1", "Q3"):
        m = rep["overall"][tag]
        print(f"[{tag} overall]")
        for k in ("accuracy", "precision", "recall", "f1", "brier", "ece", "base_rate", "mean_p_pos"):
            print(f"  {k:14s} {_fmt(m[k])}")
        if tag == "Q1":
            print(f"  parse_rate     {_fmt(m['parse_rate'])}")
        print()

    for src, m in rep["by_split"].items():
        print(f"--- split: {src} (n={m['n']}) ---")
        for tag in ("Q1", "Q3"):
            mt = m[tag]
            print(
                f"  {tag}: acc={_fmt(mt['accuracy'])} p={_fmt(mt['precision'])} "
                f"r={_fmt(mt['recall'])} f1={_fmt(mt['f1'])} "
                f"brier={_fmt(mt['brier'])} ece={_fmt(mt['ece'])}"
            )
        print()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", type=Path)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    rows = load_jsonl(args.jsonl)
    rep = report(rows)
    print_report(rep)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        print(f"[saved] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
