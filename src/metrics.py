"""Binary classification metrics, including calibration (Brier, ECE)."""
from __future__ import annotations

import numpy as np


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cc = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = cc["tp"], cc["tn"], cc["fp"], cc["fn"]
    n = max(1, len(y_true))
    acc = (tp + tn) / n
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-12, prec + rec)
    return {
        "n": int(n),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        **cc,
    }


def brier_score(y_true: np.ndarray, p_pos: np.ndarray) -> float:
    return float(np.mean((p_pos - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, p_pos: np.ndarray, n_bins: int = 10
) -> float:
    """Standard ECE with equal-width bins on max-confidence."""
    y_pred = (p_pos >= 0.5).astype(int)
    confidence = np.where(y_pred == 1, p_pos, 1 - p_pos)
    correct = (y_pred == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        avg_conf = confidence[mask].mean()
        avg_acc = correct[mask].mean()
        ece += (mask.sum() / n) * abs(avg_conf - avg_acc)
    return float(ece)


def calibration_metrics(y_true: np.ndarray, p_pos: np.ndarray) -> dict:
    return {
        "brier": brier_score(y_true, p_pos),
        "ece": expected_calibration_error(y_true, p_pos),
        "mean_p_pos": float(p_pos.mean()),
        "base_rate": float(y_true.mean()),
    }
