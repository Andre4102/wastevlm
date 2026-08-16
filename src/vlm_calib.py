"""Calibrated binary readout: score the Yes/No margin, threshold it.

Single source of truth for the decision gate, imported by BOTH the calibration
step (`scripts/vlm_binary_auc.py`) and the eval that applies it
(`src/vlm_eval.py`). If the question wording, the token set, or the position that
gets scored ever differed between fitting and applying, the threshold would be
measured against one quantity and used on another -- and it would fail silently,
looking merely like a bad number.

Why this exists at all: sampled-token evals fix ONE operating point, and we
measured how much that hides. On AerialWaste the model ranks images at AUC 0.837
but speaks them at J 0.200, because its positives sit at margin -1.93 while
DroneWaste's sit at +1.48. Reading the margin instead of the sampled string
recovers ~half the gap to a frozen-feature probe, for 50 binary labels and no
retraining. See EXPERIMENTS.md, "The decision is calibration, not perception".
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

# The question is deliberately plain and symmetric: no label menu, no chain of
# thought, nothing pushing toward either answer. Anything richer would put the
# prompt back inside the measurement, which is the thing we are escaping.
QUESTION = (
    "Is there any waste, garbage, debris, or dumped material visible in this "
    "image? Answer Yes or No."
)

YES_WORDS = ["Yes", " Yes", "yes", " yes", "YES"]
NO_WORDS = ["No", " No", "no", " no", "NO"]


def decision_token_ids(tokenizer) -> tuple[list[int], list[int]]:
    """First-token ids of the Yes / No surface forms, deduplicated.

    Only the FIRST token is scored: "Yes" and "No" tokenize to different lengths,
    so summing whole-word logprobs would compare sequences of unequal length and
    bake a length bias into the decision.
    """
    def ids(words):
        out = []
        for w in words:
            enc = tokenizer(w, add_special_tokens=False).input_ids
            if enc:
                out.append(enc[0])
        return sorted(set(out))
    return ids(YES_WORDS), ids(NO_WORDS)


@torch.no_grad()
def score_margin(model, pixel_values, question: str,
                 yes_ids: list[int], no_ids: list[int]) -> float:
    """logsumexp(Yes) - logsumexp(No) at the first assistant token.

    Prompt assembly mirrors `WasteVLM.generate` exactly -- same system prompt,
    same ChatML framing, same IMAGE_TOKEN_INDEX marker. A different framing here
    would measure a model the training never produced.
    """
    from src.vlm_model import IMAGE_TOKEN_INDEX

    def tok(t: str) -> list[int]:
        return model.tokenizer(t, add_special_tokens=False).input_ids

    pre, post = ("<image>\n" + question).split("<image>", 1)
    ids = tok(f"<|im_start|>system\n{model.system_prompt}<|im_end|>\n")
    ids += tok(f"<|im_start|>user\n{pre}") + [IMAGE_TOKEN_INDEX] + tok(f"{post}<|im_end|>\n")
    ids += tok("<|im_start|>assistant\n")

    device = model.llm.device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    inputs_embeds, attn, _ = model.prepare_multimodal(
        input_ids, attn, labels, model.encode_images(pixel_values))

    logits = model.llm(inputs_embeds=inputs_embeds, attention_mask=attn).logits[0, -1]
    lp = torch.log_softmax(logits.float(), dim=-1)
    return float(torch.logsumexp(lp[yes_ids], dim=0)
                 - torch.logsumexp(lp[no_ids], dim=0))


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), ties averaged."""
    pos, neg = score[y_true == 1], score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[y_true == 1].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def youden(y_true: np.ndarray, score: np.ndarray, thr: float) -> tuple[float, float, float]:
    """(J, TPR, FPR) at a given threshold."""
    pred = score >= thr
    npos, nneg = (y_true == 1).sum(), (y_true == 0).sum()
    if npos == 0 or nneg == 0:
        return float("nan"), float("nan"), float("nan")
    tpr = float((pred & (y_true == 1)).sum() / npos)
    fpr = float((pred & (y_true == 0)).sum() / nneg)
    return tpr - fpr, tpr, fpr


def best_threshold(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """(threshold, J) maximising Youden's J.

    Candidates are MIDPOINTS between observed scores, so a cut fitted on a
    calibration set does not sit exactly on one of its own points -- which would
    make it fragile to the next image landing a hair either side.
    """
    uniq = np.unique(score)
    if len(uniq) < 2:
        return (float(uniq[0]) if len(uniq) else 0.0), 0.0
    cands = np.concatenate([[uniq[0] - 1.0], (uniq[:-1] + uniq[1:]) / 2.0,
                            [uniq[-1] + 1.0]])
    best = (0.0, -2.0)
    for t in cands:
        j, _, _ = youden(y_true, score, float(t))
        if j > best[1]:
            best = (float(t), j)
    return best


def write_calibration(path: pathlib.Path, *, dataset: str, ckpt: str,
                      threshold: float, n_calib: int, n_calib_pos: int,
                      question: str, extra: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"dataset": dataset, "ckpt": ckpt, "threshold": float(threshold),
               "n_calib": int(n_calib), "n_calib_pos": int(n_calib_pos),
               "question": question}
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2))


def load_threshold(spec: str) -> tuple[float, dict]:
    """Accept either a bare float or a path to a calibration JSON.

    Returns (threshold, metadata). The metadata is echoed into the eval report so
    a result always carries the provenance of the cut it was scored at -- a J
    without its threshold's origin is not interpretable.
    """
    try:
        return float(spec), {"source": "literal"}
    except ValueError:
        pass
    p = pathlib.Path(spec)
    if not p.exists():
        raise SystemExit(f"--calib: not a float and not a file: {spec}")
    d = json.loads(p.read_text())
    if "threshold" not in d:
        raise SystemExit(f"--calib: {p} has no 'threshold' key")
    d["source"] = str(p)
    return float(d["threshold"]), d
