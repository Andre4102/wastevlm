"""Per-cell output statistics for the VLM eval runs.

For every run under results/vlm_eval: output length, parse-failure rate, and
refusal/hedge rate — the three quantities that separate "the model cannot see
it" from "the harness could not read the answer" from "the model declined to
commit". Reuses src.vlm_eval's own normalisation so the counts line up with the
scored numbers rather than approximating them.

Definitions (each is a rate over the run's images):
  refusal    answer normalises into the eval's _NONE_SET ("none", "no waste", …)
             — the model explicitly asserts nothing is there.
  hedge      answer contains speculative language ("might", "appears to be", …)
             while still naming something.
  parse_fail answer is neither empty nor a refusal, yet the parser extracts no
             label — the model spoke and the harness heard nothing. This is
             harness loss, not model blindness.

  python scripts/output_stats.py [--csv out.csv]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.vlm_eval import _NONE_SET, _strip_response  # noqa: E402

EVAL_ROOT = Path("/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/"
                 "results/vlm_eval")

# Hedge markers, matched on the normalised answer. Kept deliberately narrow:
# these are phrases that qualify an assertion, not merely soft words.
HEDGE_PATTERNS = [
    r"\bmight\b", r"\bmay be\b", r"\bcould be\b", r"\bcould include\b",
    r"\bpossibly\b", r"\bperhaps\b", r"\bappears? to be\b", r"\bseems? to\b",
    r"\blikely\b", r"\bprobably\b", r"\bunclear\b", r"\bhard to tell\b",
    r"\bdifficult to (?:tell|determine|identify)\b", r"\bcannot (?:tell|determine)\b",
    r"\bnot (?:clear|certain|sure)\b", r"\bsuch as\b", r"\btypical(?:ly)?\b",
    r"\bcommonly\b", r"\bsome kind of\b", r"\bunidentifiable\b",
]
HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS))


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL splitting on newlines only.

    `str.splitlines()` also splits on U+2028/U+0085/form-feed, which VLM outputs
    do contain (a legacy pruned-Qwen run emits raw U+2028), and json.dumps does
    not escape them — so the splitlines() idiom tears records in half.
    """
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:                      # iteration splits on "\n" only
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def is_refusal(raw: str) -> bool:
    return _strip_response(raw) in _NONE_SET


def is_hedged(raw: str) -> bool:
    return bool(HEDGE_RE.search(_strip_response(raw)))


def run_stats(run_dir: Path) -> dict | None:
    rep_path, raw_path = run_dir / "test_eval.json", run_dir / "raw_responses.jsonl"
    if not (rep_path.exists() and raw_path.exists()):
        return None
    report = json.loads(rep_path.read_text())
    recs = read_jsonl(raw_path)
    n = len(recs)
    if n == 0:
        return None

    answers = [r.get("raw", "") or "" for r in recs]
    refusals = [a for a in answers if is_refusal(a)]
    hedged = [a for a in answers if not is_refusal(a) and is_hedged(a)]
    parse_fail = [r for r in recs
                  if not is_refusal(r.get("raw", "") or "")
                  and (r.get("raw", "") or "").strip()
                  and not r.get("parsed")]

    # answer length in words, and separately for the answers that assert
    # something — a run can look verbose purely because it says "none" a lot,
    # or terse for the same reason.
    words = [len(a.split()) for a in answers]
    words_assert = [len(a.split()) for a in answers if not is_refusal(a)] or [0]
    t1 = [len((r.get("raw_turn1", "") or "").split()) for r in recs if "raw_turn1" in r]

    # The dir name is the only record of which arm produced a run. Legacy dirs
    # (pre stage-tag naming) are all cradiov4-so 150K *or* a different encoder /
    # baseline model entirely, so identify the arm, never assume the stage.
    name = run_dir.name
    if "cradiov4-so" in name:
        arm = ("cradio 819K" if "finetune_next" in name
               else "cradio 150K")          # incl. legacy dirs: only 150K existed then
    elif "radio-l" in name:      arm = "radio-l A1"
    elif "dinov3-b" in name:     arm = "dinov3-b A1"
    elif "qwenvl" in name or "qvl" in name: arm = "qwen-vl base/pruned"
    elif "masked" in name:       arm = "qwen-vl masked"
    else:                        arm = name.split("_")[1] if "_" in name else name
    return {
        "run": run_dir.name,
        "arm": arm,
        "dataset": report["dataset"],
        "prompt": report["prompt_style"],
        "n": n,
        "micro_f1": round(report["micro"]["f1"], 4),
        "words_mean": round(sum(words) / n, 2),
        "words_mean_asserting": round(sum(words_assert) / len(words_assert), 2),
        "turn1_words_mean": round(sum(t1) / len(t1), 2) if t1 else None,
        "refusal_rate": round(len(refusals) / n, 4),
        "hedge_rate": round(len(hedged) / n, 4),
        "parse_fail_rate": round(len(parse_fail) / n, 4),
        "parse_fail_n": len(parse_fail),
        "labels_per_image": report.get("labels_per_image", {}).get("pred_mean"),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--eval-root", type=Path, default=EVAL_ROOT)
    p.add_argument("--csv", type=Path, default=None)
    p.add_argument("--examples", type=int, default=4,
                   help="parse-failure examples to print per run (0 = none)")
    args = p.parse_args()

    rows = [s for d in sorted(args.eval_root.glob("vlm_*")) if d.is_dir()
            if (s := run_stats(d))]
    rows.sort(key=lambda r: (r["dataset"], r["prompt"], r["arm"]))

    hdr = (f"{'arm':<22}{'dataset':<12}{'prompt':<20}{'n':>5}{'micF1':>7}"
           f"{'words':>7}{'w|assert':>9}{'t1_w':>6}{'refuse':>8}{'hedge':>7}"
           f"{'parse_fail':>11}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        t1 = "  —  " if r["turn1_words_mean"] is None else f"{r['turn1_words_mean']:>6.1f}"
        print(f"{r['arm']:<22}{r['dataset']:<12}{r['prompt']:<20}{r['n']:>5}"
              f"{r['micro_f1']:>7.3f}{r['words_mean']:>7.1f}"
              f"{r['words_mean_asserting']:>9.1f}{t1}"
              f"{r['refusal_rate']:>8.3f}{r['hedge_rate']:>7.3f}"
              f"{r['parse_fail_rate']:>11.3f}")

    if args.examples:
        print("\nparse failures — model spoke, harness extracted nothing:")
        for r in rows:
            if not r["parse_fail_n"]:
                continue
            d = args.eval_root / r["run"]
            recs = read_jsonl(d / "raw_responses.jsonl")
            bad = [x for x in recs if not is_refusal(x.get("raw", "") or "")
                   and (x.get("raw", "") or "").strip() and not x.get("parsed")]
            print(f"\n  {r['run']}  ({r['parse_fail_n']} of {r['n']})")
            for x in bad[:args.examples]:
                print(f"    gt={x['gt']}\n      answer: {x['raw'][:150]!r}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\n[saved] {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
