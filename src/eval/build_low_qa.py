"""Build a multiple-choice waste-knowledge benchmark from the EU List of Waste.

Parses the European List of Waste (Commission Decision 2014/955/EU, collected in
``eurlex.jsonl``) into ~840 six-digit code↔description pairs and generates two MC
question types:

  code2desc : "European List of Waste — code 01 01 01 refers to:" → pick the
              description among 4 (distractors drawn from the same chapter).
  desc2code : "…the code for '<description>' is:" → pick the code among 4.

Distractors are sampled from the *same 2-digit chapter* where possible, so a
model must know the specific entry, not just the topic. Output is a JSONL the
`mc_score.py` scorer consumes; scoring matches the pruning repo's `acc_norm`
(byte-length-normalised continuation log-prob).

This is a *retention / memorisation* probe: the List of Waste is in the CPT
corpus, so it measures how much learned regulatory knowledge survives pruning
(random baseline = 25%). A generalisation split (conceptual questions via
agent-dispatch) is future work.

    python -m src.eval.build_low_qa --max-per-type 400
"""
from __future__ import annotations

import argparse
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from ..corpus.common import CORPUS_ROOT, read_jsonl, write_jsonl

EVAL_ROOT = Path(
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data/waste_eval"
)
LOW_CELEX = "32014D0955"

_CODE6 = re.compile(r"^(\d{2} \d{2} \d{2})(\*?)$")
_CODE_ANY = re.compile(r"^\d{2}( \d{2}){0,2}\*?$")
_GENERIC = re.compile(r"not otherwise specified|other ", re.I)


def parse_list_of_waste(text: str) -> List[Tuple[str, bool, str]]:
    """Return [(code, hazardous, description)] for six-digit LoW entries."""
    lines = [ln.strip() for ln in text.split("\n")]
    entries: List[Tuple[str, bool, str]] = []
    i = 0
    while i < len(lines):
        m = _CODE6.match(lines[i])
        if not m:
            i += 1
            continue
        code, haz = m.group(1), (m.group(2) == "*")
        # description = first following non-empty line that is not a code header
        desc = ""
        j = i + 1
        while j < len(lines):
            if not lines[j]:
                j += 1
                continue
            if _CODE_ANY.match(lines[j]):
                break  # next entry — this code had no description
            desc = lines[j]
            break
        if 8 <= len(desc) <= 200:
            entries.append((code, haz, desc))
        i = j if j > i else i + 1
    return entries


def _sample_distractors(pool: List[str], gold: str, k: int,
                        rng: random.Random) -> List[str]:
    cands = [x for x in dict.fromkeys(pool) if x != gold]
    rng.shuffle(cands)
    return cands[:k]


def build_questions(entries: List[Tuple[str, bool, str]], *,
                    max_per_type: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    by_chapter_desc: Dict[str, List[str]] = defaultdict(list)
    by_chapter_code: Dict[str, List[str]] = defaultdict(list)
    all_desc, all_code = [], []
    desc_count: Dict[str, int] = defaultdict(int)
    for code, _haz, desc in entries:
        ch = code[:2]
        by_chapter_desc[ch].append(desc)
        by_chapter_code[ch].append(code)
        all_desc.append(desc)
        all_code.append(code)
        desc_count[desc] += 1

    def distractors(pool_chapter, pool_all, gold, k):
        d = _sample_distractors(pool_chapter, gold, k, rng)
        if len(d) < k:  # top up from the global pool
            d += _sample_distractors(pool_all, gold, k - len(d), rng)
        return d[:k]

    qs: List[Dict] = []
    # code2desc — every entry with a usable description is eligible
    pool = list(entries)
    rng.shuffle(pool)
    for code, _haz, desc in pool:
        if sum(1 for q in qs if q["type"] == "code2desc") >= max_per_type:
            break
        distr = distractors(by_chapter_desc[code[:2]], all_desc, desc, 3)
        if len(distr) < 3:
            continue
        choices = [desc] + distr
        rng.shuffle(choices)
        qs.append({
            "id": f"low_c2d_{code.replace(' ', '')}",
            "type": "code2desc",
            "prompt": f"European List of Waste — code {code} refers to:",
            "choices": choices, "gold": choices.index(desc),
            "source": f"eurlex:{LOW_CELEX}",
        })

    # desc2code — only descriptions that map to a UNIQUE code (else ambiguous)
    uniq = [(c, d) for c, _h, d in entries
            if desc_count[d] == 1 and not _GENERIC.search(d)]
    rng.shuffle(uniq)
    for code, desc in uniq:
        if sum(1 for q in qs if q["type"] == "desc2code") >= max_per_type:
            break
        distr = distractors(by_chapter_code[code[:2]], all_code, code, 3)
        if len(distr) < 3:
            continue
        choices = [code] + distr
        rng.shuffle(choices)
        qs.append({
            "id": f"low_d2c_{code.replace(' ', '')}",
            "type": "desc2code",
            "prompt": f"European List of Waste — the code for waste described as "
                      f"'{desc}' is:",
            "choices": choices, "gold": choices.index(code),
            "source": f"eurlex:{LOW_CELEX}",
        })
    return qs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eurlex", type=Path, default=CORPUS_ROOT / "eurlex.jsonl")
    ap.add_argument("--out", type=Path, default=EVAL_ROOT / "low_qa.jsonl")
    ap.add_argument("--max-per-type", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    doc = next(d for d in read_jsonl(args.eurlex)
               if d["id"] == f"eurlex:{LOW_CELEX}")
    entries = parse_list_of_waste(doc["text"])
    haz = sum(1 for _c, h, _d in entries if h)
    print(f"Parsed {len(entries)} LoW entries ({haz} hazardous-marked)")

    qs = build_questions(entries, max_per_type=args.max_per_type, seed=args.seed)
    n2 = sum(1 for q in qs if q["type"] == "code2desc")
    n1 = sum(1 for q in qs if q["type"] == "desc2code")
    write_jsonl(args.out, qs)
    print(f"Wrote {len(qs)} questions (code2desc={n2}, desc2code={n1}) → {args.out}")
    print(f"Random baseline = 25% (4-choice).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
