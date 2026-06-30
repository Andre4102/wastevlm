"""Stratified sampling across (label, image_source) for balanced baselines."""
from __future__ import annotations

import random
from collections import defaultdict

from .datasets import Sample


def stratified_sample(
    samples: list[Sample],
    n: int,
    seed: int = 0,
    strata_keys: tuple[str, ...] = ("label", "image_source"),
) -> list[Sample]:
    """Stratified-by-strata_keys subsample of size ~n.

    Allocates samples proportionally to stratum size, with at least 1 per stratum.
    """
    rng = random.Random(seed)
    buckets: dict[tuple, list[Sample]] = defaultdict(list)
    for s in samples:
        key = tuple(getattr(s, k) if k != "label" else s.label for k in strata_keys)
        buckets[key].append(s)

    total = sum(len(v) for v in buckets.values())
    chosen: list[Sample] = []
    for key, items in buckets.items():
        share = max(1, round(n * len(items) / total))
        rng.shuffle(items)
        chosen.extend(items[: min(share, len(items))])

    rng.shuffle(chosen)
    return chosen


def balanced_binary_sample(
    samples: list[Sample],
    n: int,
    seed: int = 0,
) -> list[Sample]:
    """Equal positive/negative count, stratified by image_source within each class."""
    rng = random.Random(seed)
    pos = [s for s in samples if s.label == 1]
    neg = [s for s in samples if s.label == 0]

    half = n // 2

    def pick(pool: list[Sample], k: int) -> list[Sample]:
        by_src: dict[str, list[Sample]] = defaultdict(list)
        for s in pool:
            by_src[s.image_source].append(s)
        out: list[Sample] = []
        per_src = max(1, k // max(1, len(by_src)))
        for src, items in by_src.items():
            rng.shuffle(items)
            out.extend(items[: min(per_src, len(items))])
        rng.shuffle(out)
        return out[:k]

    chosen = pick(pos, half) + pick(neg, n - half)
    rng.shuffle(chosen)
    return chosen
