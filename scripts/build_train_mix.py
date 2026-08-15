"""Compose a stage-2 training mix from components, with upsampling.

The pieces have incompatible natural scales -- the general replay is 778k
multi-turn records averaging 185 answer tokens, the RS injection is 54k
single-turn records averaging 26 -- so "just concatenate" silently makes the
in-domain data 1% of the loss. This makes the composition explicit and reports
both units, because they answer different questions:

  record share  -> how often the model sees an (image, question, decision) triple
                   and, since every record is one image forward pass, the share of
                   optimizer steps
  token share   -> the weight in the loss, which is masked to answer tokens

For teaching a decision policy the first matters more; the second is why a given
injection is gentler than the record count suggests.

Paths: general replay records carry paths relative to --image-root, RS records
carry absolute paths. Both work unchanged -- src/vlm_data.py only prepends
image_root to relative paths -- so no rewriting is needed.

    python scripts/build_train_mix.py --arm a1 --out <dir>
    python scripts/build_train_mix.py --list
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
import random

DATA = pathlib.Path("/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data")

COMPONENTS = {
    "general_819k": DATA / "alignment/normalized/sft_mix.jsonl",
    "general_150k": DATA / "llava_instruct/llava_instruct_150k.json",
    "rs_sft": DATA / "rs_sft_p2/rs_sft.jsonl",
    "waste_sft": DATA / "waste_sft/train.json",
}

# Each component resolves its relative image paths against a DIFFERENT root:
# the 819K mix against the flat normalized/ tree, LLaVA-150K against COCO, while
# rs_sft and waste_sft already carry absolute paths. A single --image-root cannot
# serve two of them at once -- composing 150K with anything else left 50% of the
# images unresolvable -- so paths are absolutised here and --image-root becomes
# irrelevant to the merged mix.
COMPONENT_ROOTS = {
    "general_819k": DATA / "alignment/normalized",
    "general_150k": DATA / "coco/train2017",
    "rs_sft": None,
    "waste_sft": None,
}

# arm -> {component: repeat}. AerialWaste and DroneWaste appear nowhere, in any
# arm: every reported number stays zero-shot.
ARMS = {
    # main test: does teaching describe/detail/abstain at nadir, with no waste
    # data at all, move the binary decision on AerialWaste?
    "a1": {"general_819k": 1, "rs_sft": 3},
    # cheap pilot on the small general mix. The 150K arm is the one whose captions
    # collapsed to a template (`pile` on 100% of positives and 99.8% of negatives),
    # so it is the most sensitive probe of whether the injection restores
    # image-conditioning -- and it fits in one wall-clock job.
    "pilot": {"general_150k": 1, "rs_sft": 3},
    # ablation: same as a1 plus the non-AW/DW waste tiers, to separate "behaviour
    # was missing" from "waste domain knowledge was missing"
    "b": {"general_819k": 1, "rs_sft": 3, "waste_sft": 3},
}


def read_any(path: pathlib.Path) -> list[dict]:
    if path.suffix == ".jsonl":
        out = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
        return out
    return json.load(path.open(encoding="utf-8"))


def answer_tokens(rec: dict) -> float:
    """Approximate tokens the loss is computed over (assistant turns only)."""
    if "n_text_tokens" in rec:          # precomputed for the general mix
        return float(rec["n_text_tokens"])
    return sum(len(t["value"].split()) for t in rec.get("conversations", [])
               if t.get("from") in ("gpt", "assistant")) * 1.3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), help="named arm from ARMS")
    ap.add_argument("--component", action="append", metavar="NAME=REPEAT",
                    help="ad-hoc composition, repeatable; overrides --arm")
    ap.add_argument("--out", type=pathlib.Path, default=DATA / "train_mixes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="report composition only")
    args = ap.parse_args()

    if args.list:
        for name, comp in ARMS.items():
            print(f"  {name:8s} " + ", ".join(f"{k}x{v}" for k, v in comp.items()))
        return

    if args.component:
        spec = {}
        for c in args.component:
            k, _, v = c.partition("=")
            spec[k] = int(v or 1)
        arm_name = "custom"
    elif args.arm:
        spec, arm_name = ARMS[args.arm], args.arm
    else:
        raise SystemExit("need --arm or --component (or --list)")

    unknown = [k for k in spec if k not in COMPONENTS]
    if unknown:
        raise SystemExit(f"unknown components {unknown}; known: {sorted(COMPONENTS)}")

    rng = random.Random(args.seed)
    merged: list[dict] = []
    summary = []
    for name, repeat in spec.items():
        path = COMPONENTS[name]
        if not path.exists():
            raise SystemExit(f"missing component file: {path}")
        recs = read_any(path)
        tok = sum(answer_tokens(r) for r in recs)
        root = COMPONENT_ROOTS.get(name)
        n_missing = 0
        for r in recs:
            r.setdefault("component", name)
            img = r.get("image") or r.get("image_path")
            if img and not pathlib.Path(img).is_absolute() and root is not None:
                r["image"] = str(root / img)
        # verify on a sample rather than stat()-ing a million files on Lustre
        for r in rng.sample(recs, min(300, len(recs))):
            if not pathlib.Path(r["image"]).exists():
                n_missing += 1
        if n_missing:
            raise SystemExit(
                f"{name}: {n_missing}/300 sampled images do not exist "
                f"(root={root}); fix COMPONENT_ROOTS before composing a mix")
        merged.extend(recs * repeat)
        summary.append({"component": name, "repeat": repeat, "records_1x": len(recs),
                        "records": len(recs) * repeat,
                        "answer_tokens": tok * repeat})
        print(f"[load] {name:14s} x{repeat}  {len(recs):7d} -> {len(recs)*repeat:7d} records, "
              f"{tok*repeat/1e6:6.2f}M answer tokens", flush=True)

    tot_rec = sum(s["records"] for s in summary)
    tot_tok = sum(s["answer_tokens"] for s in summary)
    print(f"\n=== arm '{arm_name}': {tot_rec} records, {tot_tok/1e6:.1f}M answer tokens")
    print(f"  {'component':16s} {'records':>9s} {'rec %':>7s} {'tok %':>7s}")
    for s in summary:
        print(f"  {s['component']:16s} {s['records']:9d} "
              f"{100*s['records']/tot_rec:6.1f}% {100*s['answer_tokens']/tot_tok:6.1f}%")

    if args.dry_run:
        return

    rng.shuffle(merged)
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"mix_{arm_name}.jsonl"
    with path.open("w") as fh:
        for r in merged:
            fh.write(json.dumps(r) + "\n")
    meta = {
        "arm": arm_name, "spec": spec, "n_records": tot_rec,
        "answer_tokens": tot_tok, "components": summary,
        "by_source": dict(collections.Counter(r.get("source", "?") for r in merged)),
        "seed": args.seed,
        "held_out": ["aerialwaste", "dronewaste"],
    }
    (args.out / f"mix_{arm_name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[write] {path}")
    print(f"[write] {args.out / f'mix_{arm_name}.meta.json'}")


if __name__ == "__main__":
    main()
