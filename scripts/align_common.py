"""Shared helpers for the alignment-budget dataset pipeline (Act 2).

One place for: the target-LLM tokenizer (Qwen2.5-7B — must match the frozen VLM
decoder so `n_text_tokens` is what token-budget matching subsamples against), the
flat source-prefixed image tree, image verification, and the common
verify -> link -> tokenize -> emit loop every `convert_<name>.py` shares.

Output schema (one JSON object per line, `$DATA_ROOT/normalized/<name>.jsonl`):

    {"id": "sharegpt4v_000123",
     "image": "images/sharegpt4v_coco_train2017_000000123.jpg",
     "conversations": [{"from": "human", "value": "<image>\\nDescribe ..."},
                       {"from": "gpt",   "value": "..."}],
     "source": "sharegpt4v", "task_type": "dense_caption", "n_text_tokens": 137}

`image` is relative to `$DATA_ROOT/normalized/`, so training sets
`--image-root $DATA_ROOT/normalized`. This matches `src/vlm_data.py`'s reader
exactly (LLaVA `conversations` records, single `<image>` placeholder).
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from PIL import Image, ImageFile

# Some corpora ship truncated JPEGs; allow PIL to load them rather than raising
# (we still verify() headers below and drop the genuinely broken ones).
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_PLACEHOLDER = "<image>"

QWEN_PATH = os.environ.get(
    "WASTE_VLM_QWEN",
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/weights/Qwen2.5-7B-Instruct",
)


def data_root() -> Path:
    r = os.environ.get("DATA_ROOT")
    if not r:
        # fall back to the pointer file written by step 0
        ptr = Path(__file__).resolve().parent.parent / ".align_data_root"
        if ptr.exists():
            r = ptr.read_text().strip()
    if not r:
        raise SystemExit("Set DATA_ROOT (see vlm_alignment_setup.md step 0).")
    return Path(r)


@lru_cache(maxsize=1)
def get_tokenizer():
    """Target-LLM tokenizer used for `n_text_tokens`. Qwen2.5-7B to match the
    frozen VLM decoder — see the Act-2 alignment probe design.

    Loaded via the pure-Rust `tokenizers` library straight from tokenizer.json
    when available, so token counting never imports transformers/torch. That
    matters on the shared login node: pulling in torch (~2 GB) mid-run got the
    downloader OOM-killed under node-wide memory pressure. Falls back to the
    transformers AutoTokenizer only if tokenizer.json is absent.
    """
    tj = Path(QWEN_PATH) / "tokenizer.json"
    if tj.exists():
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tj))
        return lambda text: len(tok.encode(text).ids)
    from transformers import AutoTokenizer

    hf = AutoTokenizer.from_pretrained(QWEN_PATH)
    return lambda text: len(hf(text, add_special_tokens=False).input_ids)


def count_text_tokens(conversations: list[dict]) -> int:
    """Tokens of the conversation *content* (both roles), excluding the single
    `<image>` placeholder and any ChatML scaffolding. This is the quantity the
    matched-budget subsampling in build_arms.py equalizes across arms."""
    count = get_tokenizer()
    total = 0
    for turn in conversations:
        total += count(turn["value"].replace(IMAGE_PLACEHOLDER, ""))
    return total


def sanitize(rel: str) -> str:
    return rel.strip("/").replace("/", "_").replace(" ", "_")


def verify_image(path: str | Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def link_image(src_abs: str | Path, images_dir: Path, dest_name: str) -> str:
    """Symlink `src_abs` into the flat `images_dir` as `dest_name` (idempotent,
    no copy). Returns the record-relative path `images/<dest_name>`."""
    dest = images_dir / dest_name
    if not dest.exists():
        try:
            dest.symlink_to(os.path.abspath(src_abs))
        except FileExistsError:
            pass
    return f"images/{dest_name}"


class Sample:
    """One pre-normalization item yielded by a converter's source iterator."""

    __slots__ = ("id", "image_src", "image_rel", "conversations", "task_type")

    def __init__(self, id: str, image_src: Optional[str], image_rel: str,
                 conversations: list[dict], task_type: str) -> None:
        self.id = id
        self.image_src = image_src          # abs path of the image already on disk
        self.image_rel = image_rel          # source-relative name -> flat filename
        self.conversations = conversations  # LLaVA [{from,value}], human has <image>
        self.task_type = task_type


def emit_records(
    source: str,
    samples: Iterable[Sample],
    out_jsonl: Path,
    images_dir: Path,
    bad_log: Path,
    limit: Optional[int] = None,
    verify: bool = True,
    progress_every: int = 2000,
) -> dict:
    """Shared verify -> link -> tokenize -> write loop.

    Drops samples whose image is missing or fails `verify()`, logging the offending
    path to `bad_log` rather than crashing (mirrors src/vlm_data._safe_open_rgb's
    defensive posture). Returns a stats dict for the download/convert log.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    emitted = missing = corrupt = 0
    with open(out_jsonl, "w") as out, open(bad_log, "w") as bad:
        for s in samples:
            if limit is not None and emitted >= limit:
                break
            if s.image_src is not None:
                if not os.path.exists(s.image_src):
                    missing += 1
                    bad.write(f"MISSING\t{s.image_src}\n")
                    continue
                if verify and not verify_image(s.image_src):
                    corrupt += 1
                    bad.write(f"CORRUPT\t{s.image_src}\n")
                    continue
                dest_name = f"{source}_{sanitize(s.image_rel)}"
                image_field = link_image(s.image_src, images_dir, dest_name)
            else:
                image_field = ""
            rec = {
                "id": s.id,
                "image": image_field,
                "conversations": s.conversations,
                "source": source,
                "task_type": s.task_type,
                "n_text_tokens": count_text_tokens(s.conversations),
            }
            out.write(json.dumps(rec) + "\n")
            emitted += 1
            if emitted % progress_every == 0:
                print(f"[{source}] emitted={emitted} missing={missing} "
                      f"corrupt={corrupt}", flush=True)
    stats = {"source": source, "emitted": emitted, "missing": missing,
             "corrupt": corrupt, "out": str(out_jsonl)}
    print(f"[{source}] DONE {stats}", flush=True)
    return stats
