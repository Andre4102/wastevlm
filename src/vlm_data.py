"""Dataset + LLaVA-style collator for Waste-VLM training.

Supports the LLaVA-1.5 two-stage recipe and the project's own VQA data:

  * Stage 1 (connector): LLaVA-Pretrain LCS-558K — single-turn caption pairs.
  * Stage 2 (instruction): LLaVA-Instruct-150K — multi-turn conversations over
    COCO images.
  * Custom VQA (later): same schema.

Input files may be a JSON array (LLaVA's *.json) or JSON-lines. Two record
schemas are accepted:

  1. LLaVA conversations:
     {"image": "<path>", "conversations": [
         {"from": "human", "value": "<image>\\nQ"},
         {"from": "gpt",   "value": "A"}, ...]}    # may be multi-turn
  2. Flat QA:
     {"image": "<path>", "question": "Q", "answer": "A", "q_type": "presence"}

The collator renders Qwen ChatML manually so it can (a) mask the loss to the
assistant turns only, and (b) splice a single IMAGE_TOKEN_INDEX marker where the
`<image>` placeholder sits — the marker is later expanded into patch tokens by
`WasteVLM`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.vlm_model import IMAGE_TOKEN_INDEX

IMAGE_PLACEHOLDER = "<image>"


def _record_to_messages(rec: dict) -> tuple[Optional[str], list[tuple[str, str]]]:
    """Return (image_path, [(role, content), ...])."""
    image = rec.get("image") or rec.get("image_path")
    if "conversations" in rec:
        msgs = []
        for t in rec["conversations"]:
            role = "user" if t["from"] in ("human", "user") else "assistant"
            msgs.append((role, t["value"].strip()))
    else:
        q, a = rec["question"].strip(), rec["answer"].strip()
        if IMAGE_PLACEHOLDER not in q:
            q = IMAGE_PLACEHOLDER + "\n" + q
        msgs = [("user", q), ("assistant", a)]
    return image, msgs


def _load_records(path: str) -> list[dict]:
    with open(path) as f:
        head = f.read(64).lstrip()
        f.seek(0)
        if head.startswith("["):
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


class VQADataset(Dataset):
    def __init__(self, path: str, image_root: Optional[str] = None) -> None:
        self.records = _load_records(path)
        self.image_root = Path(image_root) if image_root else None

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, image: str) -> Path:
        p = Path(image)
        if not p.is_absolute() and self.image_root is not None:
            p = self.image_root / p
        return p

    def __getitem__(self, idx: int) -> dict:
        image, messages = _record_to_messages(self.records[idx])
        pil = Image.open(self._resolve(image)).convert("RGB") if image else None
        return {"image": pil, "messages": messages}


# ---------------------------------------------------------------------------
# Pre-tokenized cache (offline tokenization; images still decoded per item)
# ---------------------------------------------------------------------------
# A cache is a directory written by `src/pretokenize_vlm.py`:
#   tokens.npy   int32   concatenated input_ids of every record
#   labels.npy   int32   concatenated labels (same per-record lengths)
#   offsets.npy  int64   length N+1 prefix sums delimiting each record
#   images.txt   text    N lines, the raw image field per record ("" if none)
#   meta.json    dict    {count, max_len, llm_path, system_prompt, image_root}
# Only the text tokenization is cached — pixel_values can't be (558K decoded
# images is ~440 GB), so images are still opened lazily in __getitem__.

def save_token_cache(out_dir: str, encoded: list[tuple[list[int], list[int]]],
                     images: list[str], meta: dict) -> None:
    """Write the pre-tokenized cache to `out_dir` as flat ragged arrays."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    offsets = np.zeros(len(encoded) + 1, dtype=np.int64)
    for i, (ids, _) in enumerate(encoded):
        offsets[i + 1] = offsets[i] + len(ids)
    tokens = np.empty(int(offsets[-1]), dtype=np.int32)
    labels = np.empty(int(offsets[-1]), dtype=np.int32)
    for i, (ids, labs) in enumerate(encoded):
        tokens[offsets[i]:offsets[i + 1]] = ids
        labels[offsets[i]:offsets[i + 1]] = labs
    np.save(out / "tokens.npy", tokens)
    np.save(out / "labels.npy", labels)
    np.save(out / "offsets.npy", offsets)
    (out / "images.txt").write_text("\n".join(images))
    (out / "meta.json").write_text(json.dumps({**meta, "count": len(encoded)}, indent=2))


class PretokenizedVQADataset(Dataset):
    """Map-style dataset over a `save_token_cache` directory.

    Token ids/labels are memory-mapped (near-zero RAM, no 558K-record JSON parse
    per rank at startup); images are still decoded lazily per item.
    """

    def __init__(self, cache_dir: str, image_root: Optional[str] = None) -> None:
        d = Path(cache_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        self.tokens = np.load(d / "tokens.npy", mmap_mode="r")
        self.labels = np.load(d / "labels.npy", mmap_mode="r")
        self.offsets = np.load(d / "offsets.npy")  # small; keep resident
        raw = (d / "images.txt").read_text()
        self.images = raw.split("\n") if raw else []
        # image_root arg overrides the one baked into the cache, if given
        root = image_root or self.meta.get("image_root")
        self.image_root = Path(root) if root else None
        n = int(self.meta["count"])
        assert len(self.images) == n == len(self.offsets) - 1, "corrupt token cache"

    def __len__(self) -> int:
        return int(self.meta["count"])

    def _resolve(self, image: str) -> Path:
        p = Path(image)
        if not p.is_absolute() and self.image_root is not None:
            p = self.image_root / p
        return p

    def __getitem__(self, idx: int) -> dict:
        a, b = int(self.offsets[idx]), int(self.offsets[idx + 1])
        input_ids = self.tokens[a:b].tolist()
        labels = self.labels[a:b].tolist()
        image = self.images[idx]
        pil = Image.open(self._resolve(image)).convert("RGB") if image else None
        return {"input_ids": input_ids, "labels": labels, "image": pil}


def encode_messages(
    tokenizer,
    system_prompt: str,
    messages: list[tuple[str, str]],
    max_len: int = 2048,
) -> tuple[list[int], list[int]]:
    """Render a conversation to Qwen ChatML token ids + assistant-only label mask.

    A single `<image>` placeholder is replaced with the IMAGE_TOKEN_INDEX marker
    (expanded into patch tokens later by `WasteVLM`). This is the single source of
    truth for tokenization — used both by the on-the-fly collator and the
    pre-tokenizer, so a cache built offline is byte-identical to the live path.
    """
    def _tok(text: str) -> list[int]:
        if IMAGE_PLACEHOLDER in text:
            pre, post = text.split(IMAGE_PLACEHOLDER, 1)
            return (
                tokenizer(pre, add_special_tokens=False).input_ids
                + [IMAGE_TOKEN_INDEX]
                + tokenizer(post, add_special_tokens=False).input_ids
            )
        return tokenizer(text, add_special_tokens=False).input_ids

    input_ids: list[int] = []
    labels: list[int] = []

    def add(text: str, target: bool) -> None:
        ids = _tok(text)
        input_ids.extend(ids)
        labels.extend(ids if target else [-100] * len(ids))

    add(f"<|im_start|>system\n{system_prompt}<|im_end|>\n", False)
    for role, content in messages:
        if role == "user":
            add(f"<|im_start|>user\n{content}<|im_end|>\n", False)
        else:  # assistant: header is prompt, content+terminator are targets
            add("<|im_start|>assistant\n", False)
            add(content, True)
            add("<|im_end|>\n", True)

    if len(input_ids) > max_len:  # marker sits early; trimming the tail keeps it
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    return input_ids, labels


def _pad_batch(enc: list[tuple[list[int], list[int]]], pad_id: int) -> dict:
    """Right-pad a list of (input_ids, labels) into batched tensors + attn mask."""
    max_l = max(len(ids) for ids, _ in enc)
    B = len(enc)
    input_ids = torch.full((B, max_l), pad_id, dtype=torch.long)
    labels = torch.full((B, max_l), -100, dtype=torch.long)
    attn = torch.zeros((B, max_l), dtype=torch.long)
    for i, (ids, labs) in enumerate(enc):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
        labels[i, :n] = torch.tensor(labs, dtype=torch.long)
        attn[i, :n] = 1
    return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}


def build_collator(
    tokenizer,
    image_transform: Callable,
    system_prompt: str,
    max_len: int = 2048,
) -> Callable:
    """Collate_fn that tokenizes messages on the fly and decodes images per batch."""
    pad_id = tokenizer.pad_token_id

    def collate(samples: list[dict]) -> dict:
        enc = [encode_messages(tokenizer, system_prompt, s["messages"], max_len)
               for s in samples]
        batch = _pad_batch(enc, pad_id)
        batch["pixel_values"] = torch.stack(
            [image_transform(s["image"].convert("RGB")) for s in samples]
        )
        return batch

    return collate


def build_cached_collator(pad_id: int, image_transform: Callable) -> Callable:
    """Collate_fn for `PretokenizedVQADataset`: ids/labels are precomputed, so we
    only pad them and decode the images (the one cost that can't be cached)."""

    def collate(samples: list[dict]) -> dict:
        enc = [(s["input_ids"], s["labels"]) for s in samples]
        batch = _pad_batch(enc, pad_id)
        batch["pixel_values"] = torch.stack(
            [image_transform(s["image"].convert("RGB")) for s in samples]
        )
        return batch

    return collate


def synthetic_samples(n: int = 2, image_size: int = 512) -> list[dict]:
    """Random-image conversations for smoke tests (no dataset on disk needed)."""
    import numpy as np

    qa = [
        ("<image>\nIs there any illegal waste visible in this image?",
         "Yes, there is a small dumping site."),
        ("<image>\nWhat types of waste are present? List them.",
         "Construction debris, tyres."),
        ("<image>\nGive the waste bounding box as [x_min, y_min, x_max, y_max].",
         "[0.31, 0.44, 0.58, 0.72]"),
        ("<image>\nDescribe the visible scenery.",
         "Open cropland bordered by a dirt track and sparse trees."),
    ]
    out = []
    for i in range(n):
        arr = (np.random.rand(image_size, image_size, 3) * 255).astype("uint8")
        q, a = qa[i % len(qa)]
        out.append({"image": Image.fromarray(arr),
                    "messages": [("user", q), ("assistant", a)]})
    return out
