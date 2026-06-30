"""Diagnostic: verify image loading + prompt format for each runner.

Checks:
1. PIL loads the right file at the right size/mode.
2. Each runner's `_build_prompt` produces a sensible prompt string.
3. The model actually conditions on the image (random-noise control).

Usage:
    python -m src.diagnose --model llava-next
    python -m src.diagnose --model geochat
    python -m src.diagnose --model geollava8k
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import load_aerialwaste, load_dronewaste  # noqa: E402
from src.prompts import Q1  # noqa: E402


def inspect_samples() -> None:
    print("=" * 60)
    print("1. Dataset loaders — sample inspection")
    print("=" * 60)
    for name, samples in (
        ("aerialwaste", load_aerialwaste("/home/ids/diecidue/data/aerialwaste", "testing")),
        ("dronewaste", load_dronewaste("/home/ids/diecidue/data/dronewaste")),
    ):
        pos = next(s for s in samples if s.label == 1)
        neg = next(s for s in samples if s.label == 0)
        for s in (pos, neg):
            img = Image.open(s.image_path)
            arr = np.asarray(img.convert("RGB"))
            print(
                f"  {name}: id={s.image_id} label={s.label} src={s.image_source} "
                f"json_dims=({s.width}x{s.height}) PIL_dims={img.size} mode={img.mode} "
                f"pixel_mean={arr.mean():.1f} pixel_std={arr.std():.1f}"
            )
        print()


def inspect_prompts(model: str) -> tuple[object, callable]:
    """Build the runner and print the literal prompt it would feed the LLM."""
    print("=" * 60)
    print(f"2. Prompt format for model={model}")
    print("=" * 60)
    if model == "llava-next":
        from src.runner import LlavaNextRunner

        runner = LlavaNextRunner(max_new_tokens=32)
        prompt = runner._build_prompt(Q1)
    elif model == "geochat":
        from src.geochat_runner import GeoChatRunner

        runner = GeoChatRunner(max_new_tokens=32)
        prompt = runner._build_prompt(Q1)
    elif model == "geollava8k":
        from src.geollava8k_runner import GeoLlava8KRunner

        runner = GeoLlava8KRunner(max_new_tokens=32)
        prompt = runner._build_prompt(Q1)
    else:
        raise ValueError(model)

    print("--- raw prompt ---")
    print(repr(prompt))
    print()
    print("--- rendered ---")
    print(prompt)
    print()
    return runner


def control_random_image(runner) -> None:
    """Run the same Q1 on three real images and on uniform-random noise.

    If all responses are identical (or p_yes is constant) the model isn't
    actually using vision.
    """
    print("=" * 60)
    print("3. Image-conditioning control")
    print("=" * 60)
    drone = load_dronewaste("/home/ids/diecidue/data/dronewaste")
    aerial = load_aerialwaste("/home/ids/diecidue/data/aerialwaste", "testing")

    def first(samples, label):
        return next(s for s in samples if s.label == label)

    probes: list[tuple[str, Image.Image]] = [
        ("drone_positive", Image.open(first(drone, 1).image_path).convert("RGB")),
        ("drone_negative", Image.open(first(drone, 0).image_path).convert("RGB")),
        ("aerial_positive", Image.open(first(aerial, 1).image_path).convert("RGB")),
        ("aerial_negative", Image.open(first(aerial, 0).image_path).convert("RGB")),
    ]

    rng = np.random.default_rng(0)
    noise_arr = rng.integers(0, 255, size=(640, 640, 3), dtype=np.uint8)
    probes.append(("random_noise_640", Image.fromarray(noise_arr)))
    probes.append(("solid_black_640", Image.new("RGB", (640, 640), (0, 0, 0))))
    probes.append(("solid_white_640", Image.new("RGB", (640, 640), (255, 255, 255))))

    for tag, img in probes:
        resp = runner.ask(img, Q1, compute_yes_no=True)
        text = resp.text.replace("\n", " ")[:100]
        py = resp.p_yes if resp.p_yes is not None else float("nan")
        pn = resp.p_no if resp.p_no is not None else float("nan")
        print(f"  [{tag:18s}] p_yes={py:.3f} p_no={pn:.3f}  → {text}")
    print()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["llava-next", "geochat", "geollava8k"], required=True)
    p.add_argument("--skip-model", action="store_true", help="only inspect samples + prompts, don't run inference")
    args = p.parse_args()

    inspect_samples()
    runner = inspect_prompts(args.model)
    if not args.skip_model:
        control_random_image(runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
