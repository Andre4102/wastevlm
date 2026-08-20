"""Margin readout for the external VLM baselines, on our splits.

Qwen2.5-VL and InternVL3 have only ever been scored here through a parser --
micro-F1 on sampled text -- which fixes one operating point and cannot separate
"not represented" from "represented, verbalised at the wrong threshold". Our own
models are reported under a threshold-free margin readout, so the comparison that
most favours us is the one that has never been run on the baselines.

Same question, same yes/no surface forms, same AUC and Youden code as
`vlm_binary_auc.py`. The margin is read at the FIRST generated token via
`output_scores`, which goes through each model's own generate path rather than a
hand-built forward, so each model's image handling stays exactly as its authors
wrote it.

    python scripts/baseline_binary_auc.py --model qwen2_5vl --dataset aw_m2
    python scripts/baseline_binary_auc.py --model internvl3 --dataset dw_paper10 \
        --sites site2,site5,site9,site13,site17 --sites-all-images
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from scripts.vlm_binary_auc import QUESTION, best_youden, roc_auc  # noqa: E402
from src import vlm_calib  # noqa: E402


@torch.no_grad()
def margin_qwen(ad, image, question, yes_ids, no_ids) -> float:
    img_in, _w, _h = ad._resize_for_qwen(image)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img_in}, {"type": "text", "text": question}]}]
    text = ad.processor.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)
    inputs = ad.processor(text=[text], images=[img_in], padding=True,
                          return_tensors="pt").to(ad.model.device)
    out = ad.model.generate(**inputs, max_new_tokens=1, do_sample=False,
                            output_scores=True, return_dict_in_generate=True)
    return _margin_from(out.scores[0][0], yes_ids, no_ids)


@torch.no_grad()
def margin_internvl(ad, image, question, yes_ids, no_ids) -> float:
    """Replicates InternVLChatModel.chat's prompt build, then reads scores.

    chat() decodes its own output, so it cannot return logits; the prompt
    construction is reproduced here verbatim (conv template + IMG_CONTEXT
    expansion) so the input is byte-identical to what chat() would have built.
    """
    pixel_values, n_tiles = ad._prep_pixels(image)
    m = ad.model
    # modeling_internvl_chat is loaded dynamically by trust_remote_code, so the
    # template helper is reachable only through the model class's own module.
    get_conv_template = sys.modules[type(m).__module__].get_conv_template
    tok = ad.tokenizer
    m.img_context_token_id = tok.convert_tokens_to_ids("<IMG_CONTEXT>")
    template = get_conv_template(m.template)
    template.system_message = m.system_message
    template.append_message(template.roles[0], "<image>\n" + question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    img_tokens = "<img>" + "<IMG_CONTEXT>" * m.num_image_token * n_tiles + "</img>"
    query = query.replace("<image>", img_tokens, 1)
    enc = tok(query, return_tensors="pt")
    out = m.generate(pixel_values=pixel_values,
                     input_ids=enc["input_ids"].to(m.device),
                     attention_mask=enc["attention_mask"].to(m.device),
                     max_new_tokens=1, do_sample=False,
                     output_scores=True, return_dict_in_generate=True)
    return _margin_from(out.scores[0][0], yes_ids, no_ids)


def _margin_from(logits, yes_ids, no_ids) -> float:
    lp = torch.log_softmax(logits.float(), dim=-1)
    return float(torch.logsumexp(lp[yes_ids], 0) - torch.logsumexp(lp[no_ids], 0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["qwen2_5vl", "internvl3"], required=True)
    ap.add_argument("--dataset", default="aw_m2",
                    choices=["aw_m2", "aw_m4", "dw_paper10"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sites", default=None)
    ap.add_argument("--sites-all-images", action="store_true")
    ap.add_argument("--out-json", type=pathlib.Path, default=None)
    args = ap.parse_args()

    from src.vlm_eval import ADAPTERS, DATASETS, _load_classification_samples
    spec = DATASETS[args.dataset]
    cats, samples = _load_classification_samples(args.dataset, spec, args.limit)
    if args.sites:
        want = {s.strip() for s in args.sites.split(",")}
        before = len(samples)
        if args.sites_all_images:
            from src.datasets import load_dronewaste_multilabel
            _c, all_s = load_dronewaste_multilabel(
                str(pathlib.Path(os.environ["WASTE_DATA_ROOT"]) / "dronewaste"),
                categories_filter=spec["cats"])
            samples = [s for s in all_s if s.image_source in want]
        else:
            samples = [s for s in samples if s.image_source in want]
        print(f"[data] sites {sorted(want)}: {before} -> {len(samples)} images")

    ad = ADAPTERS[args.model]()
    print(f"[model] loading {args.model} from {ad.path}", flush=True)
    ad.load("cuda")
    tok = getattr(ad, "tokenizer", None) or ad.processor.tokenizer
    yes_ids, no_ids = vlm_calib.decision_token_ids(tok)
    print(f"[tok] yes={yes_ids} no={no_ids}", flush=True)
    fn = margin_qwen if args.model == "qwen2_5vl" else margin_internvl

    from PIL import Image
    scores, labels = [], []
    for i, s in enumerate(samples):
        img = Image.open(s.image_path).convert("RGB")
        scores.append(fn(ad, img, QUESTION, yes_ids, no_ids))
        # Positive exactly as vlm_binary_auc defines it: the image carries at
        # least one ground-truth category from THIS dataset's label set. Using
        # Sample.label instead would silently score a different question.
        labels.append(int(bool(set(s.extra["gt_categories"]) & set(cats))))
        if i and i % 100 == 0:
            print(f"  {i}/{len(samples)}", flush=True)
    score = np.array(scores, float); y = np.array(labels, int)
    auc = roc_auc(y, score)
    j, thr, tpr, fpr = best_youden(y, score)
    pred0 = score >= 0.0
    j0 = (float((pred0 & (y == 1)).sum() / max((y == 1).sum(), 1))
          - float((pred0 & (y == 0)).sum() / max((y == 0).sum(), 1)))
    print(f"\n=== {args.dataset}  model={args.model}")
    print(f"  images            {len(y)}  ({int(y.sum())} pos / {int((1-y).sum())} neg)")
    print(f"  AUC               {auc:.4f}")
    print(f"  best Youden J     {j:.4f}   at margin {thr:.3f}  (TPR {tpr:.3f}, FPR {fpr:.3f})")
    print(f"  J at margin 0     {j0:.4f}   <- what it would SAY")
    print(f"  margin  pos mean  {score[y==1].mean():+.3f}   neg mean {score[y==0].mean():+.3f}")
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(
            {"model": args.model, "dataset": args.dataset, "n": len(y),
             "n_pos": int(y.sum()), "auc": auc, "best_j": j, "thr": thr,
             "j_at_zero": j0, "scores": score.tolist(), "y": y.tolist()}, indent=2))
        print(f"[write] {args.out_json}")


if __name__ == "__main__":
    main()
