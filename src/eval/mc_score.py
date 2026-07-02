"""Score causal-LM checkpoints on a multiple-choice waste benchmark.

Scoring matches the pruning repo's convention: for each choice, sum the
continuation token log-probs given the prompt; the prediction is the choice with
the highest score. Two metrics are reported per the pruning harness:

    acc      = argmax over Σ log p(continuation)              (length-biased)
    acc_norm = argmax over Σ log p(continuation) / n_bytes    (PRIMARY)

Pass several ``--models`` to compare (dense base vs CPT vs pruned) in one run.
Needs torch + transformers (e.g. the ``gausdino`` env or the pruning env)::

    python -m src.eval.mc_score \\
        --bench /leonardo_scratch/.../waste_vlm/data/waste_eval/low_qa.jsonl \\
        --models meta-llama/Llama-3.1-8B /path/to/waste-llm-cpt /path/to/waste-llm-pruned
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from ..corpus.common import read_jsonl


def _load(model_path: str, dtype: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch_dtype,
    ).to(device).eval()
    return model, tok


def _score_choice(model, tok, device, prompt: str, continuation: str
                  ) -> Tuple[float, int]:
    """Return (Σ log p(continuation | prompt), n_bytes)."""
    import torch
    ids_prompt = tok(prompt, add_special_tokens=True)["input_ids"]
    ids_full = tok(prompt + continuation, add_special_tokens=True)["input_ids"]
    n_ctx = len(ids_prompt)
    n_bytes = max(1, len(continuation.encode("utf-8")))
    if len(ids_full) <= n_ctx:
        return -1e9, n_bytes
    input_ids = torch.tensor([ids_full], device=device)
    with torch.no_grad():
        logits = model(input_ids).logits[0]          # (T, V)
    logp = logits.log_softmax(dim=-1)
    tgt = torch.tensor(ids_full[n_ctx:], device=device)
    pos = torch.arange(n_ctx - 1, len(ids_full) - 1, device=device)
    total = logp[pos].gather(1, tgt.unsqueeze(1)).sum().item()
    return total, n_bytes


def evaluate(model, tok, device, questions: List[Dict]) -> Dict:
    hit = defaultdict(int)
    hit_norm = defaultdict(int)
    n = defaultdict(int)
    for q in questions:
        scores = [_score_choice(model, tok, device, q["prompt"], " " + c)
                  for c in q["choices"]]
        pred = max(range(len(scores)), key=lambda i: scores[i][0])
        pred_norm = max(range(len(scores)), key=lambda i: scores[i][0] / scores[i][1])
        for key in ("all", q["type"]):
            n[key] += 1
            hit[key] += int(pred == q["gold"])
            hit_norm[key] += int(pred_norm == q["gold"])
    return {
        k: {"acc": hit[k] / n[k], "acc_norm": hit_norm[k] / n[k], "n": n[k]}
        for k in n
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", type=Path, required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=0, help="0 = all questions")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    questions = list(read_jsonl(args.bench))
    if args.limit:
        questions = questions[: args.limit]
    print(f"{len(questions)} questions from {args.bench}")

    results: Dict[str, Dict] = {}
    for mp in args.models:
        print(f"\n=== {mp} ===", flush=True)
        model, tok = _load(mp, args.dtype, args.device)
        res = evaluate(model, tok, args.device, questions)
        results[mp] = res
        for k in sorted(res):
            r = res[k]
            print(f"  {k:10s}  acc_norm={r['acc_norm']:.3f}  "
                  f"acc={r['acc']:.3f}  (n={r['n']})")
        del model

    # comparison table (acc_norm, primary)
    print("\n=== acc_norm (primary) ===")
    keys = sorted({k for r in results.values() for k in r})
    print(f"{'model':40s} " + " ".join(f"{k:>10s}" for k in keys))
    for mp, res in results.items():
        row = " ".join(f"{res[k]['acc_norm']:10.3f}" if k in res else " " * 10
                       for k in keys)
        print(f"{Path(mp).name:40s} {row}")
    print("random baseline = 0.250")

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
