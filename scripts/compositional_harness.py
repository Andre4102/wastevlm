"""Score the compositional queries four ways, so perception and reasoning come apart.

  symbolic / GT         the reasoning ceiling. Validated at 2721/2721 over ground
                        truth, so anything below it later is model error rather
                        than a solver bug or an ambiguous question.
  symbolic / predicted  the same solver over the pipeline's own scene graph. The
                        drop from the row above is exactly what perception costs,
                        with reasoning held perfect.
  decoder / predicted   the decoder reads the same scene graph as text and answers.
                        The drop from the row above is exactly what the decoder
                        costs, with perception held fixed.
  decoder / GT          the decoder over perfect perception, which separates "it
                        cannot reason" from "it was given bad facts".

Most work of this kind reports only the third row and cannot say which half failed.

The controls carried through from the query generator matter as much as the arms.
Every family reports the majority answer and the presence oracle -- an agent told
which categories are present and nothing else, which is an upper bound on what a
text-head pipeline recovers. A family the oracle solves is a control where a tie is
the expected result, not a win.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scene_reason import render, solve  # noqa: E402

ASK = """You are reading an automatic survey of one aerial photograph of a waste site.

Objects detected in the image:
{scene}

Question: {question}

Answer with ONLY the answer itself — for a yes/no question reply exactly "yes" or \
"no"; for "how many" reply with a number; for "which kind" reply with the category \
name exactly as written above. No explanation."""


def normalise(ans: str, family: str) -> str:
    a = ans.strip().strip('".').lower()
    if family in ("presence", "spatial", "count_compare", "area_compare",
                  "negation_spatial"):
        if re.search(r"\byes\b", a):
            return "yes"
        if re.search(r"\bno\b", a):
            return "no"
        return a[:12]
    if family == "count":
        m = re.search(r"\d+", a)
        if not m:
            return a[:6]
        n = int(m.group(0))
        return str(n if n < 4 else "4+")
    return ans.strip().strip('".')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--gt-scenes", required=True)
    ap.add_argument("--pred-scenes")
    ap.add_argument("--decoder",
                    default="/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/weights/Qwen2.5-7B-Instruct")
    ap.add_argument("--arms", nargs="+",
                    default=["symbolic-gt", "symbolic-pred"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.queries)]
    gt = {s["image"]: s for s in json.loads(pathlib.Path(args.gt_scenes).read_text())}
    pred = ({s["image"]: s for s in json.loads(pathlib.Path(args.pred_scenes).read_text())}
            if args.pred_scenes else {})
    # only questions whose image the predicted pass actually covered, so every arm
    # answers the same set
    keep = set(gt) & (set(pred) if pred else set(gt))
    rows = [r for r in rows if r["image"] in keep]
    if args.limit:
        rows = rows[: args.limit]
    fams = sorted({r["family"] for r in rows})
    print(f"[harness] {len(rows)} questions over {len({r['image'] for r in rows})} "
          f"images, {len(fams)} families")

    gen = None
    if any(a.startswith("decoder") for a in args.arms):
        from scripts.two_stage_router import load_decoder
        gen = load_decoder(args.decoder)

    results = {}
    for arm in args.arms:
        src = gt if arm.endswith("gt") else pred
        got = defaultdict(lambda: [0, 0])
        for n, r in enumerate(rows):
            sc = src.get(r["image"])
            if sc is None:
                continue
            if arm.startswith("symbolic"):
                a = solve(r, sc["objs"], tuple(sc["size"]))
            else:
                a = normalise(gen(ASK.format(scene=render(sc["objs"], tuple(sc["size"])),
                                             question=r["question"]),
                                  max_new_tokens=24), r["family"])
            g = got[r["family"]]
            g[0] += (a is not None and str(a).lower() == str(r["answer"]).lower())
            g[1] += 1
            if gen is not None and n % 200 == 0:
                print(f"   {arm} {n}/{len(rows)}", flush=True)
        results[arm] = {f: (v[0] / v[1], v[1]) for f, v in got.items()}

    # references
    ref = {}
    for f in fams:
        sub = [r for r in rows if r["family"] == f]
        cnt = Counter(r["answer"] for r in sub)
        maj = max(cnt.values()) / len(sub)
        known = [r for r in sub if r["presence_answer"] is not None]
        orc = (sum(r["presence_answer"] == r["answer"] for r in known) / len(sub)
               if known else (1.0 / len(cnt) if len(cnt) > 2 else maj))
        ref[f] = {"majority": maj, "oracle": orc, "bar": max(maj, orc)}

    hdr = ["family", "n", "bar"] + args.arms
    print("\n{:18s} {:>5s} {:>6s} ".format(*hdr[:3])
          + " ".join(f"{a:>15s}" for a in args.arms))
    for f in fams:
        n = results[args.arms[0]].get(f, (0, 0))[1]
        line = "{:18s} {:5d} {:6.3f} ".format(f, n, ref[f]["bar"])
        for a in args.arms:
            acc = results[a].get(f, (float('nan'), 0))[0]
            line += f" {acc:14.3f}"
        print(line + ("   CONTROL" if ref[f]["oracle"] > 0.9 else ""))
    print()
    for a in args.arms:
        tot = sum(v[0] * v[1] for v in results[a].values())
        n = sum(v[1] for v in results[a].values())
        print(f"  {a:16s} overall {tot/max(1,n):.3f}")

    if args.out_json:
        pathlib.Path(args.out_json).write_text(
            json.dumps({"results": results, "reference": ref}, indent=2))
        print(f"\n[write] {args.out_json}")


if __name__ == "__main__":
    main()
