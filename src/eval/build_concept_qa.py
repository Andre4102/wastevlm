"""Build the CONCEPT eval: synonym-paraphrased List-of-Waste MC (accuracy).

Reads the authored paraphrases (`data/concept_paraphrases.json`) — plain-language
rewrites of the official LoW descriptions with the discriminative keywords removed
— and builds `desc2code` questions whose PROMPT uses the paraphrase but whose
answer is the real code, distractors drawn from the same 2-digit chapter.

Because the wording differs from the trained string, accuracy here reflects
*concept* knowledge, not verbatim memorisation — the complement to `low_qa`
(memorisation) and `ppl_eval` (held-out generalisation). Consumed by `mc_score`.

    python -m src.eval.build_concept_qa
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from ..corpus.common import CORPUS_ROOT, read_jsonl
from .build_low_qa import EVAL_ROOT, LOW_CELEX, _sample_distractors, parse_list_of_waste

PARAPHRASES = Path(__file__).parent / "data" / "concept_paraphrases.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eurlex", type=Path, default=CORPUS_ROOT / "eurlex.jsonl")
    ap.add_argument("--paraphrases", type=Path, default=PARAPHRASES)
    ap.add_argument("--out", type=Path, default=EVAL_ROOT / "concept_qa.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    para = json.loads(args.paraphrases.read_text())["paraphrases"]
    doc = next(d for d in read_jsonl(args.eurlex) if d["id"] == f"eurlex:{LOW_CELEX}")
    entries = parse_list_of_waste(doc["text"])
    code2desc = {c: d for c, _h, d in entries}
    by_chapter_code = defaultdict(list)
    all_code = []
    for c, _h, _d in entries:
        by_chapter_code[c[:2]].append(c)
        all_code.append(c)

    rng = random.Random(args.seed)
    qs, missing = [], []
    for code, paraphrase in para.items():
        if code not in code2desc:
            missing.append(code)
            continue
        distr = _sample_distractors(by_chapter_code[code[:2]], code, 3, rng)
        if len(distr) < 3:
            distr += _sample_distractors(all_code, code, 3 - len(distr), rng)
        choices = [code] + distr[:3]
        rng.shuffle(choices)
        qs.append({
            "id": f"concept_{code.replace(' ', '')}",
            "type": "concept_desc2code",
            "prompt": f"European List of Waste — the code for waste described as "
                      f"'{paraphrase}' is:",
            "choices": choices, "gold": choices.index(code),
            "source": f"eurlex:{LOW_CELEX}", "gold_desc": code2desc[code],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for q in qs:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"Wrote {len(qs)} concept questions → {args.out}")
    if missing:
        print(f"⚠  {len(missing)} paraphrase codes not found in LoW: {missing}")
    print("Random baseline = 0.250")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
