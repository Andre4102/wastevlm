"""Threshold-free readout of the VLM's binary waste decision.

Every number we have on AerialWaste comes from *sampled tokens*: the model emits
a sentence, a parser turns it into labels, and one operating point falls out. That
conflates two very different failures -- "the decision is not represented" and "it
is represented but verbalised at the wrong threshold" -- and the DW split makes
the conflation concrete: identical weights give recall 0.83 / FPR 0.16 under
closed_vocab and 0.33 / 0.007 under open_cot. A single J cannot tell those apart.

So ask the model one yes/no question and read the *logits* at the first answer
token instead of decoding: score = logsumexp(yes variants) - logsumexp(no
variants). That is monotone in the model's internal belief and needs no parser,
so sweeping it gives an AUC -- a property of the representation rather than of a
threshold.

The comparison that matters is against the frozen-feature linear probe on the
SAME encoder output (EXPERIMENTS.md: aw_m2 AUC 0.970 / aw_m4 0.966). The probe
proves the direction exists in what the LLM is handed. This measures whether the
LLM has any graded access to it:

    AUC ~ 0.5   -> no access; the projector/LLM path must change
    AUC >> 0.5  -> access is fine, only the verbalisation threshold is broken

Run DW too: it is the positive control. If DW scores high and AW is at chance,
the failure is specific to nadir imagery, not to the readout method.

    python scripts/vlm_binary_auc.py --ckpt <dir> --encoder cradiov4-so \
        --image-size 768 --pixel-shuffle 2 --dataset aw_m2
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# The question is deliberately plain and symmetric: no label menu, no chain of
# thought, nothing that pushes toward either answer. Anything fancier would put
# the prompt back inside the measurement, which is what we are trying to escape.
QUESTION = (
    "Is there any waste, garbage, debris, or dumped material visible in this "
    "image? Answer Yes or No."
)

YES = ["Yes", " Yes", "yes", " yes", "YES"]
NO = ["No", " No", "no", " no", "NO"]


def first_token_ids(tokenizer, words: list[str]) -> list[int]:
    """First-token id of each surface form, deduplicated.

    Scoring only the FIRST token is what keeps this a fair contest: "Yes" and
    "No" may tokenize to different lengths, and summing whole-word logprobs
    would then compare sequences of unequal length.
    """
    ids = []
    for w in words:
        enc = tokenizer(w, add_special_tokens=False).input_ids
        if enc:
            ids.append(enc[0])
    return sorted(set(ids))


@torch.no_grad()
def yes_minus_no(model, pixel_values, question: str,
                 yes_ids: list[int], no_ids: list[int]) -> float:
    """Logit margin for Yes over No at the first assistant token.

    Prompt assembly mirrors `WasteVLM.generate` exactly (same system prompt,
    same ChatML framing, same IMAGE_TOKEN_INDEX marker) -- a different framing
    here would measure a model the training never produced.
    """
    from src.vlm_model import IMAGE_TOKEN_INDEX

    def tok(t: str) -> list[int]:
        return model.tokenizer(t, add_special_tokens=False).input_ids

    user = "<image>\n" + question
    ids = tok(f"<|im_start|>system\n{model.system_prompt}<|im_end|>\n")
    pre, post = user.split("<image>", 1)
    ids += tok(f"<|im_start|>user\n{pre}") + [IMAGE_TOKEN_INDEX] + tok(f"{post}<|im_end|>\n")
    ids += tok("<|im_start|>assistant\n")

    device = model.llm.device
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    image_embeds = model.encode_images(pixel_values)
    inputs_embeds, attn, _ = model.prepare_multimodal(input_ids, attn, labels, image_embeds)

    logits = model.llm(inputs_embeds=inputs_embeds, attention_mask=attn).logits[0, -1]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    y = torch.logsumexp(logprobs[yes_ids], dim=0)
    n = torch.logsumexp(logprobs[no_ids], dim=0)
    return float(y - n)


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney U), ties averaged."""
    pos, neg = score[y_true == 1], score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within tie groups, else ties bias the statistic
    s_sorted = score[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    r_pos = ranks[y_true == 1].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def best_youden(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float, float, float]:
    """Best achievable J over all thresholds, and where it sits."""
    best = (-1.0, 0.0, 0.0, 0.0)
    for t in np.unique(score):
        pred = score >= t
        tpr = float((pred & (y_true == 1)).sum() / max((y_true == 1).sum(), 1))
        fpr = float((pred & (y_true == 0)).sum() / max((y_true == 0).sum(), 1))
        if tpr - fpr > best[0]:
            best = (tpr - fpr, float(t), tpr, fpr)
    return best


def load_train_samples(dataset: str, spec: dict, limit: int):
    """The fitting split, mirroring `_load_classification_samples`'s test logic.

    A threshold chosen on the test set is an oracle, not a result, so the cut has
    to come from data the reported number never touches. AW ships a real train
    split; DW is the same 70/30 site-stratified split (seed=0) used for test,
    taken from the other side of the cut so no *site* spans both.
    """
    from collections import defaultdict

    from src.vlm_eval import WASTE_DATA_ROOT

    if dataset in ("aw_m2", "aw_m4"):
        from src.datasets import load_aerialwaste_mcml
        cats, samples = load_aerialwaste_mcml(
            str(WASTE_DATA_ROOT / "aerialwaste"), split="train",
            version="m2" if dataset == "aw_m2" else "m4",
        )
        before = len(samples)
        samples = [s for s in samples if s.image_path.exists()]
        if before != len(samples):
            print(f"[data] train filtered missing-on-disk: -{before-len(samples)}", flush=True)
    elif dataset == "dw_paper10":
        from src.datasets import load_dronewaste_multilabel
        cats, samples_all = load_dronewaste_multilabel(
            str(WASTE_DATA_ROOT / "dronewaste"), categories_filter=spec["cats"],
        )
        site_to_idx: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(samples_all):
            site_to_idx[s.image_source].append(i)
        rng = np.random.default_rng(0)
        train_idx: list[int] = []
        for _site, idxs in site_to_idx.items():
            idxs = list(idxs); rng.shuffle(idxs)
            train_idx.extend(idxs[:int(len(idxs) * 0.7)])
        samples = [samples_all[i] for i in train_idx]
    else:
        raise ValueError(dataset)

    if limit > 0:
        samples = samples[:limit]
    return cats, samples


def score_samples(adapter, model, samples, cats, question, yes_ids, no_ids, tag=""):
    scores, labels_bin = [], []
    for i, s in enumerate(samples):
        try:
            img = Image.open(s.image_path).convert("RGB")
        except Exception as e:
            print(f"[warn] cannot open {s.image_path}: {e}", flush=True)
            continue
        px = adapter._transform(img).unsqueeze(0)
        scores.append(yes_minus_no(model, px, question, yes_ids, no_ids))
        gt = set(s.extra["gt_categories"]) & set(cats)
        labels_bin.append(int(bool(gt)))
        if (i + 1) % 250 == 0:
            print(f"  {tag}{i+1}/{len(samples)}", flush=True)
    return np.array(labels_bin), np.array(scores)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=pathlib.Path, required=True)
    ap.add_argument("--encoder", default="cradiov4-so")
    ap.add_argument("--image-size", type=int, default=768)
    ap.add_argument("--pixel-shuffle", type=int, default=2)
    ap.add_argument("--dataset", default="aw_m2", choices=["aw_m2", "aw_m4", "dw_paper10"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--question", default=QUESTION)
    ap.add_argument("--out-json", type=pathlib.Path, default=None)
    ap.add_argument("--fit-on-train", action="store_true",
                    help="also score the train split, pick the threshold there, "
                         "and report it on test (the only honest number)")
    ap.add_argument("--train-limit", type=int, default=0,
                    help="cap train images used to fit the threshold")
    args = ap.parse_args()

    from src.vlm_eval import DATASETS, WasteVLMAdapter, _load_classification_samples

    spec = DATASETS[args.dataset]
    cats, samples = _load_classification_samples(args.dataset, spec, args.limit)
    print(f"[data] {args.dataset}: {len(samples)} test images, {len(cats)} classes", flush=True)

    adapter = WasteVLMAdapter(args.ckpt, encoder=args.encoder,
                              image_size=args.image_size,
                              pixel_shuffle=args.pixel_shuffle)
    adapter.load("cuda")
    model = adapter.model

    yes_ids = first_token_ids(model.tokenizer, YES)
    no_ids = first_token_ids(model.tokenizer, NO)
    if not yes_ids or not no_ids:
        raise SystemExit("could not resolve Yes/No token ids")
    print(f"[tok] yes={yes_ids} no={no_ids}", flush=True)

    # binary target = "any of the dataset's classes present", filtered to the
    # registry `cats` exactly as run_classify does -- this is the same gt+
    # population as the `detection any-label?` row in aw_diagnose, so the AUC
    # and those J values describe one decision.
    y, sc = score_samples(adapter, model, samples, cats, args.question,
                          yes_ids, no_ids, tag="test ")
    auc = roc_auc(y, sc)
    j, thr, tpr, fpr = best_youden(y, sc)
    # the model's OWN threshold: it would say Yes wherever the margin is positive
    pred0 = sc >= 0.0
    tpr0 = float((pred0 & (y == 1)).sum() / max((y == 1).sum(), 1))
    fpr0 = float((pred0 & (y == 0)).sum() / max((y == 0).sum(), 1))

    print(f"\n=== {args.dataset}  ckpt={args.ckpt.name}")
    print(f"  images            {len(y)}  ({int(y.sum())} pos / {int((y==0).sum())} neg)")
    print(f"  AUC               {auc:.4f}")
    print(f"  best Youden J     {j:.4f}   at margin {thr:+.3f}  (TPR {tpr:.3f}, FPR {fpr:.3f})")
    print(f"  J at margin 0     {tpr0-fpr0:.4f}   (TPR {tpr0:.3f}, FPR {fpr0:.3f})  <- what it would SAY")
    print(f"  margin  pos mean  {sc[y==1].mean():+.3f}   neg mean {sc[y==0].mean():+.3f}")

    fit = None
    if args.fit_on_train:
        _cats_tr, tr_samples = load_train_samples(args.dataset, spec, args.train_limit)
        print(f"\n[fit] scoring {len(tr_samples)} TRAIN images to pick the threshold", flush=True)
        y_tr, sc_tr = score_samples(adapter, model, tr_samples, cats, args.question,
                                    yes_ids, no_ids, tag="train ")
        j_tr, thr_tr, _, _ = best_youden(y_tr, sc_tr)
        pred = sc >= thr_tr
        tpr_f = float((pred & (y == 1)).sum() / max((y == 1).sum(), 1))
        fpr_f = float((pred & (y == 0)).sum() / max((y == 0).sum(), 1))
        fit = {"n_train": int(len(y_tr)), "n_train_pos": int(y_tr.sum()),
               "train_auc": roc_auc(y_tr, sc_tr), "train_best_j": j_tr,
               "threshold": thr_tr, "test_j": tpr_f - fpr_f,
               "test_tpr": tpr_f, "test_fpr": fpr_f}
        print(f"\n--- threshold fitted on TRAIN, applied to TEST "
              f"({len(y_tr)} train imgs, {int(y_tr.sum())} pos)")
        print(f"  train AUC         {fit['train_auc']:.4f}   train best J {j_tr:.4f}")
        print(f"  threshold         {thr_tr:+.3f}")
        print(f"  TEST J (honest)   {fit['test_j']:.4f}   (TPR {tpr_f:.3f}, FPR {fpr_f:.3f})")
        print(f"  vs oracle-on-test {j:.4f}   -> generalisation gap "
              f"{j - fit['test_j']:+.4f}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps({
            "dataset": args.dataset, "ckpt": str(args.ckpt), "question": args.question,
            "n": len(y), "n_pos": int(y.sum()), "auc": auc,
            "best_j": j, "best_thr": thr, "tpr_at_best": tpr, "fpr_at_best": fpr,
            "j_at_zero": tpr0 - fpr0, "tpr_at_zero": tpr0, "fpr_at_zero": fpr0,
            "fit_on_train": fit,
            "scores": sc.tolist(), "labels": y.tolist(),
        }, indent=2))
        print(f"[write] {args.out_json}")


if __name__ == "__main__":
    main()
