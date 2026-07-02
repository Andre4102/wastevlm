"""Build the mixed calibration set for learned-mask structural pruning.

The pruning repo's ``pruning_vicuna_masked.py`` loads ``--training_set_override``
via ``datasets.load_from_disk`` and expects the same schema as its RedPajama
calib (``input_ids`` / ``attention_mask`` / ``labels``, each a length-``seq_len``
list). We want the mask to specialise toward waste while keeping a little general
language / reasoning, so the calib is **mostly waste + a small general slice**:

    all packed ``waste`` sequences  +  ``general_frac`` worth of RedPajama calib

Run in an env with ``datasets`` (e.g. ``gausdino``), any node (local reads)::

    python -m src.corpus.build_prune_calib --general-frac 0.25

Output → ``<data>/prune_calib_waste_mostly_seq2048`` (a ``save_to_disk`` dir).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk

from .common import CORPUS_ROOT

# RedPajama calib produced by the pruning repo (schema reference + general slice).
DEFAULT_GENERAL = Path(
    "/leonardo_scratch/large/userexternal/adiecidu/pruning/data/"
    "redpajama_calib_llama3_20000_seq2048"
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--waste-shards", type=Path,
                    default=CORPUS_ROOT / "shards" / "waste",
                    help="packed waste domain dir (holds shard_*.arrow subdirs)")
    ap.add_argument("--general", type=Path, default=DEFAULT_GENERAL,
                    help="RedPajama calib dataset (save_to_disk dir)")
    ap.add_argument("--general-frac", type=float, default=0.25,
                    help="fraction of the FINAL mix that is general text")
    ap.add_argument("--out", type=Path,
                    default=CORPUS_ROOT.parent / "prune_calib_waste_mostly_seq2048")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Waste: concatenate all packed shards (each is a save_to_disk dir).
    shard_dirs = sorted(p for p in args.waste_shards.iterdir() if p.is_dir())
    if not shard_dirs:
        raise SystemExit(f"No shard_*.arrow dirs under {args.waste_shards} — "
                         f"run `python -m src.corpus.tokenize_domain` first.")
    waste = concatenate_datasets([load_from_disk(str(d)) for d in shard_dirs])
    n_waste = len(waste)

    general_full = load_from_disk(str(args.general))

    # n_general so that general_frac = n_general / (n_waste + n_general).
    f = args.general_frac
    n_general = min(len(general_full), round(f * n_waste / (1.0 - f)) if f < 1 else len(general_full))
    general = general_full.shuffle(seed=args.seed).select(range(n_general))

    # Match the calib schema: waste has only input_ids → add attention_mask/labels
    # and cast to the general dataset's features so concatenation is exact.
    def add_fields(ex):
        ids = ex["input_ids"]
        return {"attention_mask": [1] * len(ids), "labels": list(ids)}

    waste = waste.map(add_fields, desc="waste → calib schema")
    waste = waste.cast(general.features)

    mixed = concatenate_datasets([waste, general]).shuffle(seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mixed.save_to_disk(str(args.out))

    print(f"waste={n_waste}  general={n_general}  "
          f"(general_frac={n_general / len(mixed):.2f})  total={len(mixed)}")
    print(f"cols={mixed.column_names}  seq_len={len(mixed[0]['input_ids'])}")
    print(f"saved → {args.out}")
    print(f"\nUse in the pruning launcher:\n"
          f"  --training_set_override {args.out}\n"
          f"  --num_train_samples {len(mixed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
