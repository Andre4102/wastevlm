"""Held-out waste-domain perplexity — the generalisation-retention metric.

Measures PPL of each model on the **held-out** corpus split (`corpus_eval.jsonl`,
docs excluded from CPT). Unlike the List-of-Waste MC probe (memorisation of
trained text), this is leakage-free: lower held-out PPL vs the base model means
CPT learned *transferable* waste knowledge; pruned PPL staying low means pruning
retained it.

Model-free, deterministic, no agent. Non-overlapping `--seq-len` windows,
EOS-joined docs; PPL = exp(Σ NLL / Σ tokens). Compare dense/CPT/pruned in one run.
Needs torch + transformers (`gausdino` env; GPU for 8B)::

    python -m src.eval.ppl_eval \\
      --eval /leonardo_scratch/.../waste_vlm/data/waste_corpus_web/corpus_eval.jsonl \\
      --models meta-llama/Llama-3.1-8B <waste-llm-cpt> <waste-llm-pruned>
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

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


def _token_stream(tok, texts: List[str]) -> List[int]:
    eos = tok.eos_token_id
    ids: List[int] = []
    for t in texts:
        ids.extend(tok(t, add_special_tokens=False)["input_ids"])
        ids.append(eos)
    return ids


def perplexity(model, tok, device, texts: List[str], seq_len: int) -> dict:
    import math
    import torch
    ids = _token_stream(tok, texts)
    n_windows = len(ids) // seq_len
    total_nll, total_tok = 0.0, 0
    for w in range(n_windows):
        chunk = ids[w * seq_len:(w + 1) * seq_len]
        x = torch.tensor([chunk], device=device)
        with torch.no_grad():
            logits = model(x).logits[0]                 # (L, V)
        logp = logits[:-1].log_softmax(dim=-1)          # predict tokens 1..L-1
        tgt = x[0, 1:]
        nll = -logp.gather(1, tgt.unsqueeze(1)).sum().item()
        total_nll += nll
        total_tok += tgt.numel()
    ppl = math.exp(total_nll / max(1, total_tok))
    return {"ppl": ppl, "tokens": total_tok, "windows": n_windows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, required=True, help="corpus_eval.jsonl")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    texts = [d["text"] for d in read_jsonl(args.eval)]
    print(f"{len(texts)} held-out docs from {args.eval}")

    rows = {}
    for mp in args.models:
        print(f"\n=== {mp} ===", flush=True)
        model, tok = _load(mp, args.dtype, args.device)
        r = perplexity(model, tok, args.device, texts, args.seq_len)
        rows[mp] = r
        print(f"  held-out PPL = {r['ppl']:.3f}  "
              f"(tokens={r['tokens']:,}, windows={r['windows']})")
        del model

    print("\n=== held-out waste-domain PPL (lower = better) ===")
    for mp, r in rows.items():
        print(f"{Path(mp).name:40s} {r['ppl']:8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
