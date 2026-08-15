"""Build a stratified sample of model outputs for hand-labelling.

The reported F1s are produced by a deterministic parser (`parse_label_list` /
`parse_keywords` in src/vlm_eval.py) — substring matching, no model in the loop.
That parser is still a measurement instrument, and on the open_cot cells it
discards a quarter to two fifths of all answers (see scripts/output_stats.py),
so the question "how much of the reported number is the parser" is open.

This writes a sample stratified over (run x parser outcome) with the labelling
columns left blank, plus a CSV for review outside Python. Fill `human_labels`
(semicolon-separated, empty string = no waste asserted) and run
scripts/agreement.py to get parser-vs-human agreement per benchmark.

Quotas deliberately over-sample the cells where the parser can be wrong: a
closed_vocab answer that copies a label string back verbatim agrees with the
parser by construction, an open_cot answer in free vocabulary does not.

  python scripts/build_agreement_sample.py --out-dir <dir> --n 150
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVAL_ROOT = Path("/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/"
                 "results/vlm_eval")

# The cells that carry the reported numbers, with sampling weights. open_cot
# cells get the larger share because that is where the parser does real work.
CELLS = [
    ("vlm_cradiov4-so_r768ps2_finetune_next_dw_paper10_closed_vocab", "819K", 2),
    ("vlm_cradiov4-so_r768ps2_finetune_next_dw_paper10_open_cot",     "819K", 3),
    ("vlm_cradiov4-so_r768ps2_dw_paper10_closed_vocab",               "150K", 2),
    ("vlm_cradiov4-so_r768ps2_dw_paper10_open_cot",                   "150K", 4),
    ("vlm_cradiov4-so_r768ps2_finetune_next_aw_m2_closed_vocab",      "819K", 2),
    ("vlm_cradiov4-so_r768ps2_finetune_next_aw_m2_open_cot",          "819K", 2),
    ("vlm_cradiov4-so_r768ps2_finetune_aw_m2_closed_vocab",           "150K", 2),
    ("vlm_cradiov4-so_r768ps2_finetune_next_aw_m4_closed_vocab",      "819K", 2),
    ("vlm_cradiov4-so_r768ps2_finetune_next_aw_m4_open_cot",          "819K", 2),
    ("vlm_cradiov4-so_r768ps2_finetune_aw_m4_closed_vocab",           "150K", 2),
]


def read_jsonl(path: Path) -> list[dict]:
    """Newline-only split — str.splitlines() breaks on U+2028, which outputs contain."""
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def parser_outcome(rec: dict) -> str:
    """Strata: the parser's verdict, so every failure mode gets represented."""
    raw = (rec.get("raw") or "").strip()
    parsed, gt = rec.get("parsed") or [], rec.get("gt") or []
    if not raw:
        return "empty_generation"
    if not parsed and raw.lower().rstrip(".") in {"none", "no waste", "nothing"}:
        return "explicit_none"
    if not parsed:
        return "spoke_but_unparsed"     # the cell where the parser can lose a hit
    return "parsed_with_gt" if gt else "parsed_no_gt"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=150)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    total_weight = sum(w for _, _, w in CELLS)
    items: list[dict] = []
    for run, stage, weight in CELLS:
        d = args.eval_root / run
        if not (d / "raw_responses.jsonl").exists():
            print(f"[warn] missing run, skipped: {run}")
            continue
        report = json.loads((d / "test_eval.json").read_text())
        recs = read_jsonl(d / "raw_responses.jsonl")
        quota = max(4, round(args.n * weight / total_weight))

        by_outcome: dict[str, list[dict]] = {}
        for r in recs:
            by_outcome.setdefault(parser_outcome(r), []).append(r)
        for v in by_outcome.values():
            rng.shuffle(v)

        # round-robin across outcomes so rare-but-decisive strata appear
        picked, i = [], 0
        keys = sorted(by_outcome)
        while len(picked) < quota and any(len(by_outcome[k]) > i for k in keys):
            for k in keys:
                if len(by_outcome[k]) > i and len(picked) < quota:
                    picked.append((k, by_outcome[k][i]))
            i += 1

        for outcome, rec in picked:
            items.append({
                "sample_id": f"{len(items):03d}",
                "run": run,
                "stage": stage,
                "dataset": report["dataset"],
                "prompt_style": report["prompt_style"],
                "stratum": outcome,
                "file": rec["file"],
                "gt": rec.get("gt", []),
                "parser_labels": rec.get("parsed", []),
                "answer": rec.get("raw", ""),
                "describe": rec.get("raw_turn1", ""),
                "label_set": report["per_class"] and list(report["per_class"].keys()),
                "human_labels": "",     # <- fill: semicolon-separated, "" = nothing asserted
                "human_note": "",
            })

    out_jsonl = args.out_dir / "agreement_sample.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    out_csv = args.out_dir / "agreement_sample.csv"
    cols = ["sample_id", "dataset", "stage", "prompt_style", "stratum", "file",
            "gt", "parser_labels", "answer", "describe", "human_labels", "human_note"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for it in items:
            w.writerow([it["sample_id"], it["dataset"], it["stage"], it["prompt_style"],
                        it["stratum"], it["file"], "; ".join(it["gt"]),
                        "; ".join(it["parser_labels"]),
                        (it["answer"] or "").replace("\n", " ")[:400],
                        (it["describe"] or "").replace("\n", " ")[:400], "", ""])

    print(f"[sample] {len(items)} items -> {out_jsonl}")
    print(f"[sample] review copy      -> {out_csv}")
    print("\nby dataset :", dict(Counter(i["dataset"] for i in items)))
    print("by stage   :", dict(Counter(i["stage"] for i in items)))
    print("by prompt  :", dict(Counter(i["prompt_style"] for i in items)))
    print("by stratum :", dict(Counter(i["stratum"] for i in items)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
