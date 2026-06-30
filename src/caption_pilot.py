"""Generate per-image captions using Claude Opus 4.7 for VLM distillation.

The teacher is given the image + ground-truth label list; its job is to
*explain* the labels with visual reasoning (where each labeled waste type
is in the image, what colour/shape/texture cues identify it). The output
JSONL is suitable as the target side of a Qwen2.5-VL / InternVL3 LoRA
fine-tune.

Per-image cost ≈ $0.03–0.05 with Opus 4.7 at the prompt sizes below.
The default 100-image pilot is therefore ~$3–5. Set ANTHROPIC_API_KEY
before running.

Usage:
    pip install anthropic pillow
    export ANTHROPIC_API_KEY=sk-ant-...
    python -m src.caption_pilot --n 100 --out /home/ids/diecidue/data/captions/pilot.jsonl

After the pilot looks good, scale up:
    python -m src.caption_pilot --full --out /home/ids/diecidue/data/captions/full.jsonl
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import (  # noqa: E402
    DRONEWASTE_PAPER_10,
    load_aerialwaste_mcml,
    load_dronewaste_multilabel,
)


MODEL = "claude-opus-4-7"
# Anthropic pricing (USD per 1M tokens) — keep in sync with current rate sheet.
PRICE_IN_PER_M = 15.00
PRICE_OUT_PER_M = 75.00

MAX_IMAGE_SIDE = 1024  # downscale long edge to keep image-token cost predictable

SYSTEM_PROMPT = (
    "You are an expert annotator for an aerial / drone illegal-waste-dump "
    "detection dataset. You are looking at top-down photographs of outdoor "
    "areas (rural land, roadsides, abandoned lots, construction sites). Your "
    "job is to caption an image given a list of waste categories the image "
    "is known to contain. The caption is a training target for a vision-"
    "language model, so it must be (a) FACTUAL — describe only what is "
    "visible, never invent objects, (b) SPECIFIC — name colours, shapes, "
    "textures, sizes, and rough positions (top-left, centre, scattered, etc.), "
    "and (c) CONCISE — 2-3 sentences. Do not include a preamble like "
    "'In this image' — start directly with the visual description."
)

USER_TEMPLATE_POS = (
    "Ground-truth waste categories visible in this image:\n{labels}\n\n"
    "Write a 2-3 sentence factual caption describing where each category "
    "is located and what visual cues identify it. Be specific about colours, "
    "shapes, textures, and positions. End the caption with a single line:\n"
    "  Labels: {labels_csv}"
)

USER_TEMPLATE_NEG = (
    "This image is labeled as NEGATIVE — no illegally dumped waste is "
    "visible. Write a 2-3 sentence factual caption describing what the "
    "scene actually contains (e.g. grass field, road, building, farmland) "
    "and why it is not classified as waste. End the caption with a single "
    "line:\n  Labels: none"
)


def _b64_image(path: Path) -> tuple[str, str]:
    """Resize long-edge to MAX_IMAGE_SIDE if needed; return (media_type, b64)."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        long_edge = max(w, h)
        if long_edge > MAX_IMAGE_SIDE:
            scale = MAX_IMAGE_SIDE / long_edge
            img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
    return "image/jpeg", base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _build_messages(image_path: Path, gt_labels: list[str]) -> list[dict]:
    media_type, b64 = _b64_image(image_path)
    if gt_labels:
        labels_pretty = "\n".join(f"  - {c}" for c in gt_labels)
        labels_csv = ", ".join(gt_labels)
        user_text = USER_TEMPLATE_POS.format(labels=labels_pretty, labels_csv=labels_csv)
    else:
        user_text = USER_TEMPLATE_NEG
    return [{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": media_type, "data": b64}},
            {"type": "text", "text": user_text},
        ],
    }]


def _stratified_pilot_sample(
    n_dw_pos: int, n_aw_pos: int, n_aw_neg: int, seed: int = 0,
) -> list[dict]:
    """Return a sample list — each item has {dataset, image_path, gt_categories, image_id}."""
    rng = np.random.default_rng(seed)
    out: list[dict] = []

    # DW paper-10 positives (every DW image has at least one annotation;
    # filter to those with at least one paper-10 label).
    _, dw_samples = load_dronewaste_multilabel(
        "/home/ids/diecidue/data/dronewaste",
        categories_filter=DRONEWASTE_PAPER_10,
    )
    dw_pos = [s for s in dw_samples if s.extra["gt_categories"]]
    rng.shuffle(dw_pos)
    for s in dw_pos[:n_dw_pos]:
        out.append({
            "dataset": "dronewaste_paper10",
            "image_id": s.image_id,
            "image_path": str(s.image_path),
            "gt_categories": sorted(s.extra["gt_categories"]),
        })

    # AW mcml-m2 train split → positives + negatives. (m4 has the same images
    # but a different label set; the pilot uses the m2 split for breadth.)
    _, aw_samples = load_aerialwaste_mcml(
        "/home/ids/diecidue/data/aerialwaste", split="train", version="m2",
    )
    aw_samples = [s for s in aw_samples if s.image_path.exists()]
    aw_pos = [s for s in aw_samples if s.extra["gt_categories"]]
    aw_neg = [s for s in aw_samples if not s.extra["gt_categories"]]
    rng.shuffle(aw_pos); rng.shuffle(aw_neg)
    for s in aw_pos[:n_aw_pos]:
        out.append({
            "dataset": "aerialwaste_m2",
            "image_id": s.image_id,
            "image_path": str(s.image_path),
            "gt_categories": sorted(s.extra["gt_categories"]),
        })
    for s in aw_neg[:n_aw_neg]:
        out.append({
            "dataset": "aerialwaste_m2",
            "image_id": s.image_id,
            "image_path": str(s.image_path),
            "gt_categories": [],
        })
    return out


def caption_one(client, sample: dict, max_tokens: int = 300) -> dict:
    """Call Claude once for a single sample. Returns the full result dict."""
    messages = _build_messages(Path(sample["image_path"]), sample["gt_categories"])
    t0 = time.time()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    elapsed = time.time() - t0
    text = "".join(block.text for block in resp.content if block.type == "text")
    in_tok = int(resp.usage.input_tokens)
    out_tok = int(resp.usage.output_tokens)
    return {
        **sample,
        "caption": text.strip(),
        "model": MODEL,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "cost_usd": in_tok / 1e6 * PRICE_IN_PER_M + out_tok / 1e6 * PRICE_OUT_PER_M,
        "elapsed_s": round(elapsed, 2),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100, help="pilot total sample size")
    p.add_argument("--full", action="store_true",
                   help="caption the full DW + AW train sets instead of a pilot")
    p.add_argument("--out", type=Path, required=True,
                   help="output JSONL path; resume-safe (skips image_ids already in the file)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--retries", type=int, default=3)
    args = p.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("[error] ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1
    try:
        import anthropic  # noqa: E402
    except ImportError:
        print("[error] pip install anthropic", file=sys.stderr)
        return 1
    client = anthropic.Anthropic()

    if args.full:
        # ~3000 AW train + ~500 DW positives ≈ $100-160 at Opus 4.7
        sample = _stratified_pilot_sample(
            n_dw_pos=10_000, n_aw_pos=10_000, n_aw_neg=10_000, seed=args.seed,
        )
    else:
        # Pilot split: 50 DW + 30 AW pos + 20 AW neg (defaults to args.n=100).
        ratio = args.n / 100
        sample = _stratified_pilot_sample(
            n_dw_pos=int(50 * ratio), n_aw_pos=int(30 * ratio),
            n_aw_neg=int(20 * ratio), seed=args.seed,
        )
    print(f"[sample] {len(sample)} images "
          f"({sum(1 for s in sample if s['dataset']=='dronewaste_paper10')} DW, "
          f"{sum(1 for s in sample if s['dataset']=='aerialwaste_m2' and s['gt_categories'])} AW-pos, "
          f"{sum(1 for s in sample if s['dataset']=='aerialwaste_m2' and not s['gt_categories'])} AW-neg)",
          flush=True)

    # Resume: skip image_ids already in args.out.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[tuple[str, str]] = set()
    if args.out.exists():
        with args.out.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add((rec["dataset"], str(rec["image_id"])))
                except Exception:
                    pass
        print(f"[resume] skipping {len(done_ids)} already-captioned images")

    total_cost = 0.0
    n_ok = n_fail = 0
    with args.out.open("a") as f_out:
        for k, s in enumerate(sample):
            key = (s["dataset"], str(s["image_id"]))
            if key in done_ids:
                continue
            for attempt in range(1, args.retries + 1):
                try:
                    result = caption_one(client, s)
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f_out.flush()
                    total_cost += result["cost_usd"]
                    n_ok += 1
                    break
                except Exception as e:
                    print(f"  [warn] {s['image_id']} attempt {attempt}/{args.retries}: {e}",
                          flush=True)
                    if attempt == args.retries:
                        n_fail += 1
                    else:
                        time.sleep(2 ** attempt)
            if (k + 1) % 10 == 0 or (k + 1) == len(sample):
                print(f"  [{k+1:>4d}/{len(sample)}] ok={n_ok} fail={n_fail} "
                      f"cost=${total_cost:.2f}", flush=True)

    print()
    print(f"[done] ok={n_ok} fail={n_fail} total_cost=${total_cost:.2f}")
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
