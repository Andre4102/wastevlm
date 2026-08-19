"""Defer the confused objects to a decoder-written discriminator, and check it helps.

Stage 1 scores every class and is right 39% of the time, but it knows when it is
not: entropy separates its right answers from its wrong ones at AUROC 0.824, and
the least-confident 30% sits at 0.155 accuracy. Within that 30% the true class is
in the top 5 candidates 63% of the time. So there is 0.155 -> 0.630 of headroom
available to anything that can tell five specific classes apart.

Stage 2 spends the decoder there and only there. It sees the five candidate names,
writes visual descriptions aimed at separating *those five from each other*, and
SigLIP2 re-scores the object against them. This is the one job in the pipeline with
no non-LLM substitute: the candidate set does not exist until inference, so no
fixed prompt bank can have been written for it.

Three things make the result mean something.

**The control.** Restricting the candidate set cannot by itself change an argmax --
the winner of 20 is still the winner of the 5 it belongs to -- so any gain must come
from the new prompts. The `fixed` arm re-scores the same five with a different but
non-generated prompt set, which separates "different prompts help" from "the
decoder's prompts help".

**The ceiling.** `oracle` reports top-5 containment, the most any re-ranker could get.

**The cache.** Prompts are generated per distinct candidate set, not per object.
A few hundred sets cover thousands of objects, which is what makes this affordable.

Everything runs on development sites. The routing threshold is a hyperparameter and
choosing it on the held-out sites would be fitting the evaluation set.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GEN_PROMPT = """You are writing text prompts for a vision model matching aerial \
drone photographs (straight down, about 50 metres) to categories of waste.

It must choose between exactly these, which it currently confuses:
{cands}

Official EWC-Stat entry for each, which is what the category actually means — the \
NAME is often narrower or plainly misleading, so go by the entry:
{codes}

Here is the style and level of detail wanted, for categories not in this list:
  Pallets: "stacked wooden pallets in regular rows, uniform rectangles"
  Tyres: "black rings and dark circular stacks, sharply round"
  Metal barrels: "cylindrical drums, bright tops, arranged upright in clusters"

For EACH category above write 3 prompts under 14 words. Say what THAT material \
looks like from above — colour, texture, shape, arrangement — and choose details \
that separate it from the others listed. State the colour you would actually see, \
not a guess. Never name another category.

Reply with JSON only: {{"Category name": ["...", "...", "..."], ...}}"""


def parse_json_block(text: str) -> dict:
    """Pull the first JSON object out of a decoder reply, tolerating stray prose."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return {k: [str(x) for x in v] for k, v in d.items() if isinstance(v, list)}


def load_decoder(path: str, device: str = "cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map=device).eval()

    def generate(prompt: str, max_new_tokens: int = 700) -> str:
        msgs = [{"role": "user", "content": prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    return generate


def score(emb, prompt_map, cand, encode_text, base_bank=None):
    """-> index within `cand` of the best-matching class for one object."""
    import torch

    sims = []
    for c in cand:
        ps = prompt_map.get(c) or (base_bank or {}).get(c)
        if not ps:
            sims.append(-1e9)
            continue
        t = encode_text(ps)
        t = torch.nn.functional.normalize(t.mean(0), dim=-1)
        sims.append(float(t.cpu().numpy() @ emb))
    return int(np.argmax(sims))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, help="stage-1 json (with sims)")
    ap.add_argument("--emb", help="stage-1 .emb.npy (defaults beside --result)")
    ap.add_argument("--decoder",
                    default="/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/weights/Qwen2.5-7B-Instruct")
    ap.add_argument("--defer", type=float, default=0.30,
                    help="fraction of least-confident objects sent to stage 2")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--arms", nargs="+", default=["llm", "fixed"])
    ap.add_argument("--dump-prompts", action="store_true",
                    help="print the first few generated banks, to see what they say")
    ap.add_argument("--out-json")
    args = ap.parse_args()

    import torch

    from src.prompt_sets import CONTRASTIVE
    from src.radio_adaptors import siglip2_text
    from scripts.roi_material import cue_prompts

    d = json.loads(pathlib.Path(args.result).read_text())
    cats = d["cats"]
    sim = np.array(d["sims"], np.float32)
    y = np.array(d["y_true"])
    emb = np.load(args.emb or str(pathlib.Path(args.result).with_suffix(".emb.npy")))
    assert len(emb) == len(y), f"{len(emb)} embeddings for {len(y)} objects"

    rank = (-sim).argsort(1)
    pred1 = rank[:, 0]
    p = np.exp((sim - sim.max(1, keepdims=True)) / 0.01)
    p /= p.sum(1, keepdims=True)
    entropy = -(p * np.log(p + 1e-12)).sum(1)

    n_def = int(args.defer * len(y))
    deferred = np.argsort(-entropy)[:n_def]          # most uncertain first
    kept = np.setdiff1d(np.arange(len(y)), deferred)

    base = correct1 = (pred1 == y)
    print(f"\nstage 1: {correct1.mean():.3f} over {len(y)} objects")
    print(f"  keeping   {len(kept):5d} ({1-args.defer:.0%})  accuracy {correct1[kept].mean():.3f}")
    print(f"  deferring {len(deferred):5d} ({args.defer:.0%})  accuracy {correct1[deferred].mean():.3f}")
    cont = np.mean([y[i] in rank[i, :args.topk] for i in deferred])
    print(f"  oracle ceiling on the deferred set (top-{args.topk} containment): {cont:.3f}")

    # candidate sets, cached: a few hundred distinct 5-tuples cover thousands of objects
    sets = defaultdict(list)
    for i in deferred:
        sets[tuple(sorted(cats[j] for j in rank[i, :args.topk]))].append(i)
    print(f"  {len(sets)} distinct candidate sets over {len(deferred)} objects")

    encode_text = siglip2_text(device="cuda")
    base_bank = cue_prompts("dronewaste", cats)
    from scripts.roi_material import _ewc_descriptions
    ewc = _ewc_descriptions("dronewaste")
    banks = {}
    if "llm" in args.arms:
        gen = load_decoder(args.decoder)
        for n, cand in enumerate(sets):
            reply = gen(GEN_PROMPT.format(
                cands="\n".join(f"- {c}" for c in cand),
                codes="\n".join(f"- {c}: {ewc.get(c, c)}" for c in cand)))
            got = parse_json_block(reply)
            # only keep names that are actually in the candidate set; a decoder that
            # invents a category must not silently define one
            keep = {}
            for c in cand:
                if c not in got or not got[c]:
                    continue
                # A generated prompt naming another candidate pulls the object
                # toward that rival, which is the opposite of disambiguating. The
                # per-class generator already needed this guard; this template
                # never had it, and this arm scored 0.212 -> 0.097.
                others = [o for o in cand if o != c]
                keep[c] = [g for g in got[c]
                           if not any(o.split()[0].lower() in g.lower()
                                      and o.split()[0].lower() not in c.lower()
                                      for o in others if len(o.split()[0]) > 3)]
                keep[c] = keep[c] or None
            banks[cand] = {k: v for k, v in keep.items() if v}
            if args.dump_prompts and n < 3:
                print(f"   [sample] {cand}\n     {json.dumps(banks[cand], indent=6)[:700]}",
                      flush=True)
            if n % 20 == 0:
                print(f"   generated {n}/{len(sets)}  "
                      f"({len(banks[cand])}/{len(cand)} parsed)", flush=True)

    out = {"stage1": float(correct1.mean()), "defer": args.defer,
           "deferred_stage1": float(correct1[deferred].mean()),
           "oracle_topk": float(cont), "n_sets": len(sets), "arms": {}}

    for arm in args.arms:
        pred2 = pred1.copy()
        for cand, idxs in sets.items():
            pm = banks.get(cand, {}) if arm == "llm" else {
                c: (CONTRASTIVE.get(c) or base_bank.get(c)) for c in cand}
            if not pm:
                continue
            for i in idxs:
                j = score(emb[i], pm, list(cand), encode_text, base_bank)
                pred2[i] = cats.index(list(cand)[j])
        c2 = (pred2 == y)
        print(f"\n  arm={arm}")
        print(f"    deferred set  {correct1[deferred].mean():.3f} -> {c2[deferred].mean():.3f}"
              f"   (ceiling {cont:.3f})")
        print(f"    overall       {correct1.mean():.3f} -> {c2.mean():.3f}")
        out["arms"][arm] = {"deferred": float(c2[deferred].mean()),
                            "overall": float(c2.mean())}

    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(out, indent=2))
        print(f"\n[write] {args.out_json}")


if __name__ == "__main__":
    main()
