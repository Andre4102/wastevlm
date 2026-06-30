"""Zero-shot eval of open VLMs on DroneWaste paper-10.

Two task modes (the detection path is left in place but classification is
where the open VLMs actually show signal — grounded-detection prompts ask
recognition+localisation+JSON-format all at once, which both Qwen2.5-VL
and InternVL3 collapse on; cf. the smoke-run logs from May 23).

`--task classify` is the apples-to-apples comparison with dino.txt:
    prompt VLM -> text response -> parse paper-10 label set
    -> binary multi-label vector
    -> micro / macro / per-class F1 via dinotxt_zeroshot.ml_metrics
    -> save report with the same JSON shape as dinotxt_zeroshot_dw_paper10.json

    Two prompt styles, both scored against the same DW paper-10 GT and
    using dinotxt's 70/30 site-stratified split (seed=0):
      closed_vocab: 10-label menu, exact-match parsing
      open_caption: 1-2 sentence description, per-class keyword matching

`--task detect` (legacy) prompts grounded JSON and scores via pycocotools
COCOeval. Useful as a sanity ceiling but disabled by default — see §3.5.11
in project.MD for the reasoning.

Adapters expose `generate(image, prompt) -> str`; task layers consume it.

Usage:
    python -m src.vlm_eval --model qwen2_5vl --task classify --prompt-style closed_vocab --limit 20
    python -m src.vlm_eval --model internvl3 --task classify --prompt-style open_caption \\
        --out-json /home/ids/diecidue/results/waste_vlm/vlm_internvl3_classify_open/test_eval.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import DRONEWASTE_PAPER_10  # noqa: E402
from src.det_eval import cocoeval  # noqa: E402
from src.seg_dataset import DroneWasteSegmentation  # noqa: E402

WEIGHTS_DIR = Path("/home/ids/diecidue/results/waste_vlm/weights")
QWEN_PATH = WEIGHTS_DIR / "Qwen2.5-VL-7B-Instruct"
INTERNVL3_PATH = WEIGHTS_DIR / "InternVL3-8B"
# CLIP ViT-B/32 lives in the HF disk cache (two snapshot hashes exist; use the
# one that has pytorch_model.bin — verified 3d74acf9).
CLIP_CACHE = Path(
    "/home/ids/diecidue/.cache/huggingface/hub"
    "/models--openai--clip-vit-base-patch32/snapshots"
    "/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
)


# --- prompts ---------------------------------------------------------------
# Per-class visual cues come from src/<dataset>_descriptions.json files,
# each combining Wikipedia / EU LoW material identity with a hand-layered
# aerial_cue describing how the class appears from a ~50 m drone altitude.
# The aerial_cue field is what flows into the VLM prompts below; the
# wikipedia / ewc fields are kept for provenance and downstream
# documentation.


def _load_cues(json_path: Path, cat_order: list[str]) -> dict[str, str]:
    with json_path.open() as f:
        desc = json.load(f)
    return {c: desc[c]["aerial_cue"] for c in cat_order}


def _load_clip_tags(json_path: Path, cat_order: list[str]) -> dict[str, list[str]]:
    """Load per-class synonym lists for CLIP ensemble scoring (clip_tags field)."""
    with json_path.open() as f:
        desc = json.load(f)
    return {c: desc[c].get("clip_tags", []) for c in cat_order}


PAPER10_CUES: dict[str, str] = _load_cues(
    ROOT / "src" / "paper10_descriptions.json", DRONEWASTE_PAPER_10,
)

# AerialWaste MCML categories (m2 = 5-class, m4 = 6-class) — see AW v2.0.
AW_M2_CATS: list[str] = [
    "Rubble", "Bulky items", "Plastic", "Containers", "Unknown material",
]
AW_M4_CATS: list[str] = [
    "Rubble/excavated earth and rocks", "Scrap",
    "Sludge-Zootechnical waste-Manure", "Wood", "Tires", "Other waste",
]
AW_M2_CUES: dict[str, str] = _load_cues(
    ROOT / "src" / "aw_m2_descriptions.json", AW_M2_CATS,
)
AW_M4_CUES: dict[str, str] = _load_cues(
    ROOT / "src" / "aw_m4_descriptions.json", AW_M4_CATS,
)

PAPER10_CLIP_TAGS: dict[str, list[str]] = _load_clip_tags(
    ROOT / "src" / "paper10_descriptions.json", DRONEWASTE_PAPER_10,
)
AW_M2_CLIP_TAGS: dict[str, list[str]] = _load_clip_tags(
    ROOT / "src" / "aw_m2_descriptions.json", AW_M2_CATS,
)
AW_M4_CLIP_TAGS: dict[str, list[str]] = _load_clip_tags(
    ROOT / "src" / "aw_m4_descriptions.json", AW_M4_CATS,
)

# Keyword bags for parsing free-form open_caption responses. Lowercased,
# substring-matched against the normalised response. One bag per dataset.
PAPER10_KEYWORDS: dict[str, list[str]] = {
    "Construction and demolition materials":
        ["construction", "demolition", "rubble", "concrete", "brick", "plaster", "masonry", "cinder"],
    "Metal barrels":
        ["barrel", "drum", "oil drum"],
    "Plastic packaging":
        ["plastic", "packaging", "bottle", "polystyrene", "styrofoam", "wrap", "plastic bag"],
    "Pallets":
        ["pallet"],
    "Scrap":
        ["scrap", "twisted metal", "metal piece", "rusty metal", "iron"],
    "Vehicles":
        ["vehicle", "car ", "truck", "van ", "trailer", "automobile", "wreck"],
    "Tyres":
        ["tyre", "tire", "wheel"],
    "Asbestos":
        ["asbestos", "roofing sheet", "corrugated", "cement sheet"],
    "Textile":
        ["textile", "fabric", "cloth", "rag", "clothing"],
    "Mixed items":
        ["mixed", "miscellaneous", "assorted", "heterogeneous", "refuse heap", "various waste"],
}

AW_M2_KEYWORDS: dict[str, list[str]] = {
    "Rubble":
        ["rubble", "concrete", "brick", "stone", "masonry", "construction", "demolition"],
    "Bulky items":
        ["bulky", "furniture", "mattress", "appliance", "sofa", "cabinet", "fridge", "washing machine"],
    "Plastic":
        ["plastic", "film", "tarp", "polystyrene", "styrofoam", "bag", "wrap"],
    "Containers":
        ["container", "drum", "barrel", "tank", "ibc", "tote", "bin"],
    "Unknown material":
        ["unknown", "unidentified", "unclear", "indistinguishable"],
}

AW_M4_KEYWORDS: dict[str, list[str]] = {
    "Rubble/excavated earth and rocks":
        ["rubble", "concrete", "brick", "soil", "earth", "rock", "excavated", "sand"],
    "Scrap":
        ["scrap", "twisted metal", "rusty metal", "iron", "metal piece"],
    "Sludge-Zootechnical waste-Manure":
        ["sludge", "manure", "zootechnical", "animal waste", "slurry", "liquid waste"],
    "Wood":
        ["wood", "wooden", "timber", "lumber", "plank", "board", "branch"],
    "Tires":
        ["tire", "tyre", "wheel", "rubber"],
    "Other waste":
        # removed broad single-words "other"/"various" that fire on negations;
        # added "debris" (model's go-to for unidentifiable material) and
        # hedging phrases the model uses when it sees waste but can't classify it
        ["mixed waste", "miscellaneous", "heterogeneous", "refuse", "garbage", "trash",
         "debris", "construction waste", "could be waste", "potentially waste",
         "possible waste", "unidentified material", "unclear material"],
}


def _build_prompts(cats: list[str], cues: dict[str, str],
                    domain_summary: str) -> tuple[str, str, str]:
    """Return (detect_prompt, classify_closed, classify_open) for a dataset.

    Same template as before; the class list and domain preamble are derived
    from the dataset-specific cats + cues.
    """
    cued_list = "\n".join(f"  - {c}  —  {cues[c]}" for c in cats)
    detect = (
        "This is an aerial / top-down drone photograph of an outdoor area "
        "(rural land, roadside, construction yard, etc.) suspected of "
        "containing illegally dumped waste. Looking straight down, identify "
        "every visible patch, pile, or object that fits one of the following "
        "waste categories:\n"
        f"{cued_list}\n\n"
        "Many tiles contain at least one such object. For each instance you "
        "see, produce one entry in a JSON array, where each item is an "
        "object with:\n"
        "  \"label\": one of the categories listed above (exact string match)\n"
        "  \"box\":   [x1, y1, x2, y2] in absolute pixel coordinates of the image\n"
        "Return ONLY the JSON array — no prose, no markdown fence. Return [] "
        "only if the tile genuinely contains no waste."
    )
    classify_closed = (
        "This is an aerial / top-down drone photograph of an outdoor area "
        "(rural land, roadside, construction yard, etc.) suspected of "
        "containing illegally dumped waste.\n\n"
        "Which of the following waste categories are visible in this photo? "
        "Reply with a comma-separated list of the labels that apply, copied "
        "exactly as written. Reply \"none\" if no waste is visible.\n\n"
        f"{cued_list}"
    )
    classify_open = (
        "This is an aerial / top-down drone photograph of an outdoor area "
        "(rural land, roadside, construction yard, etc.) suspected of "
        "containing illegally dumped waste.\n\n"
        f"{domain_summary}\n\n"
        "Describe in 1-2 sentences what waste, if any, is visible. Be "
        "concrete: name the materials and shapes you see (e.g. \"a pile of "
        "broken concrete\" rather than just \"debris\"). If you see no waste, "
        "reply exactly \"none\"."
    )
    return detect, classify_closed, classify_open


_DW_DOMAIN = (
    "Typical waste in such tiles includes construction debris (broken "
    "concrete, bricks, plaster), metal barrels and scrap, abandoned "
    "vehicles, plastic packaging, wooden pallets, tyres, asbestos roofing "
    "sheets, textile rags, and heterogeneous mixed-refuse heaps."
)
_AW_M2_DOMAIN = (
    "Typical waste in such tiles is grouped into 5 broad classes: rubble "
    "(broken concrete, bricks, stones), bulky items (furniture, appliances, "
    "mattresses), plastic (sheets, films, packaging), containers (drums, "
    "tanks, IBCs), and unidentifiable material."
)
_AW_M4_DOMAIN = (
    "Typical waste in such tiles is grouped into 6 broad classes: rubble + "
    "excavated earth/rocks, scrap metal, sludge / manure / zootechnical "
    "waste, wood (pallets, planks, branches), tyres, and other heterogeneous "
    "waste."
)

# Dataset specs — registry consumed by run_classify.
# Each entry: cats list, cues dict, keywords bag, detect prompt,
# classify-closed prompt, classify-open prompt, sample loader callable.
DATASETS: dict[str, dict] = {}
for _name, _cats, _cues, _tags, _kw, _domain in [
    ("dw_paper10", DRONEWASTE_PAPER_10, PAPER10_CUES, PAPER10_CLIP_TAGS, PAPER10_KEYWORDS, _DW_DOMAIN),
    ("aw_m2",      AW_M2_CATS,          AW_M2_CUES,   AW_M2_CLIP_TAGS,   AW_M2_KEYWORDS,   _AW_M2_DOMAIN),
    ("aw_m4",      AW_M4_CATS,          AW_M4_CUES,   AW_M4_CLIP_TAGS,   AW_M4_KEYWORDS,   _AW_M4_DOMAIN),
]:
    _det, _cl, _op = _build_prompts(_cats, _cues, _domain)
    DATASETS[_name] = {
        "cats": _cats,
        "cues": _cues,
        "clip_tags": _tags,
        "keywords": _kw,
        "prompt_detect": _det,
        "prompt_classify": {"closed_vocab": _cl, "open_caption": _op},
    }

# Back-compat names for the existing detection adapters (they hard-code
# the DW paper-10 prompt because grounded detection only ever ran on DW).
PROMPT_DETECT = DATASETS["dw_paper10"]["prompt_detect"]
PROMPT_CLASSIFY_CLOSED = DATASETS["dw_paper10"]["prompt_classify"]["closed_vocab"]
PROMPT_CLASSIFY_OPEN = DATASETS["dw_paper10"]["prompt_classify"]["open_caption"]
PROMPTS_CLASSIFY = DATASETS["dw_paper10"]["prompt_classify"]
PROMPT = PROMPT_DETECT

# CoT prompts for the open_cot prompt style.
# Turn 1: ask for a free-form visual description (no label menu).
# Turn 2: ask the model to name waste from its own description.
# Parsed with keyword bags — same as open_caption but grounded by the CoT.
PROMPT_DESCRIBE = (
    "This is an aerial drone photograph of an outdoor area "
    "(rural land, roadside, construction site, etc.). "
    "Describe in 2-3 sentences what you observe: focus on visible materials, "
    "objects, piles, or accumulations. Include colors, textures, and any signs "
    "of discarded or dumped materials."
)
def make_cot_classify_prompt(clip_tags: dict[str, list[str]]) -> str:
    """Build Turn-2 CoT prompt with examples drawn from this dataset's clip_tags.

    Using dataset-specific synonyms avoids anchoring the model on the hardcoded
    DroneWaste examples ('broken concrete', 'old tyres', …) which bias responses
    away from classes like 'Bulky items' or 'Containers'.
    """
    examples = [tags[0] for tags in clip_tags.values() if tags]
    ex_str = ", ".join(f"'{e}'" for e in examples)
    return (
        "Based on what you described above, what types of waste or discarded "
        "materials are visible, if any? Name specific materials or objects "
        f"(e.g., {ex_str}). "
        "If there is genuinely no waste of any kind, reply exactly \"none\"."
    )


# --- shared utilities ------------------------------------------------------

def _extract_json_array(text: str) -> list:
    """Pull the first top-level JSON array from a possibly noisy response."""
    if not text:
        return []
    # Strip common fenced-code wrappers.
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")
    # Find the first '[' ... matched ']'.
    start = text.find("[")
    if start < 0:
        return []
    depth = 0
    end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        return []
    try:
        return json.loads(text[start:end])
    except Exception:
        return []


def _norm_label(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


_LABEL_LOOKUP = {_norm_label(c): c for c in DRONEWASTE_PAPER_10}


def _match_label(raw: str) -> str | None:
    """Map a free-form label string to one of the paper-10 names."""
    if not isinstance(raw, str):
        return None
    n = _norm_label(raw)
    if n in _LABEL_LOOKUP:
        return _LABEL_LOOKUP[n]
    # Substring fallback: useful for "Plastic packaging waste" -> "Plastic packaging"
    for key, full in _LABEL_LOOKUP.items():
        if key in n or n in key:
            return full
    return None


def _coerce_box(raw, scale_x: float, scale_y: float, w_orig: int, h_orig: int):
    """Accept [x1,y1,x2,y2], clip, scale, return floats in original-image pixels."""
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in raw]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    x1, x2 = sorted((x1 * scale_x, x2 * scale_x))
    y1, y2 = sorted((y1 * scale_y, y2 * scale_y))
    x1 = max(0.0, min(float(w_orig - 1), x1))
    y1 = max(0.0, min(float(h_orig - 1), y1))
    x2 = max(0.0, min(float(w_orig), x2))
    y2 = max(0.0, min(float(h_orig), y2))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    return [x1, y1, x2 - x1, y2 - y1]  # COCO xywh


# --- adapters --------------------------------------------------------------

class VLMAdapter:
    name: str = "abstract"

    def load(self, device: str) -> None:
        raise NotImplementedError

    def generate(self, image: Image.Image, prompt: str) -> str:
        """Single-image VQA: returns the raw model response text."""
        raise NotImplementedError

    def generate_cot(self, image: Image.Image, prompt1: str, prompt2: str) -> tuple[str, str]:
        """Chain-of-thought two-turn: describe first, then classify from that description.

        Turn 1: `prompt1` → `description`.
        Turn 2: `prompt2` prepended with the description → classification text.
        The image is passed in both turns.  Returns (turn1_description, turn2_raw).
        """
        description = self.generate(image, prompt1)
        combined = f"Based on this aerial image analysis:\n{description}\n\n{prompt2}"
        return description, self.generate(image, combined)

    def detect_one(self, image: Image.Image) -> tuple[list[dict], str]:
        """Returns (boxes_in_original_pixels_xywh, raw_response)."""
        raise NotImplementedError


# --- classification post-processors -----------------------------------------

_NONE_SET = {"none", "no waste", "no waste visible", "nothing", "n/a", "na", ""}


def _strip_response(raw: str) -> str:
    """Normalise common output noise (code fences, JSON brackets, quoting)."""
    if not raw:
        return ""
    s = re.sub(r"```[a-z]*\s*", "", raw).replace("```", "")
    s = s.replace("[", " ").replace("]", " ").replace("\"", " ").replace("'", " ")
    return _norm_label(s)


def parse_label_list(raw: str, cats: list[str]) -> set[str]:
    """closed_vocab parser: substring-match each dataset label in the
    normalised response. Tolerant of bullets, JSON arrays, prose preambles."""
    s = _strip_response(raw)
    if s in _NONE_SET:
        return set()
    return {label for label in cats if _norm_label(label) in s}


def parse_keywords(raw: str, keyword_bags: dict[str, list[str]]) -> set[str]:
    """open_caption parser: per-class keyword bags, substring-matched in the
    normalised free-form response."""
    s = _strip_response(raw)
    if s in _NONE_SET:
        return set()
    out: set[str] = set()
    for label, keywords in keyword_bags.items():
        if any(kw in s for kw in keywords):
            out.add(label)
    return out


def parse_classification(raw: str, prompt_style: str, dataset_spec: dict) -> set[str]:
    if prompt_style == "closed_vocab":
        return parse_label_list(raw, dataset_spec["cats"])
    if prompt_style == "open_caption":
        return parse_keywords(raw, dataset_spec["keywords"])
    raise ValueError(f"unknown prompt_style {prompt_style!r}")


class QwenAdapter(VLMAdapter):
    """Qwen2.5-VL-7B-Instruct — absolute-pixel boxes in the processed image."""
    name = "qwen2_5vl"

    def __init__(self, path: Path = QWEN_PATH, max_new_tokens: int = 512):
        self.path = path
        self.max_new_tokens = max_new_tokens

    def load(self, device: str = "cuda") -> None:
        from transformers import (
            Qwen2_5_VLForConditionalGeneration,
            AutoProcessor,
        )
        self.processor = AutoProcessor.from_pretrained(str(self.path))
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(self.path), torch_dtype=torch.bfloat16, device_map=device,
        ).eval()
        # Qwen-VL uses absolute pixel coords in the post-processor-resized image;
        # smart_resize snaps dims to a multiple of (patch*merge)=28.
        try:
            from qwen_vl_utils import smart_resize  # noqa: F401
            self._smart_resize = smart_resize
        except Exception:
            self._smart_resize = None

    def _resize_for_qwen(self, image: Image.Image) -> tuple[Image.Image, int, int]:
        w_orig, h_orig = image.size
        if self._smart_resize is not None:
            # Qwen2.5-VL: patch=14, merge=2 -> dims must be multiples of 28.
            new_h, new_w = self._smart_resize(h_orig, w_orig, factor=28)
            return image.resize((new_w, new_h), Image.BICUBIC), new_w, new_h
        return image, w_orig, h_orig

    @torch.no_grad()
    def generate(self, image: Image.Image, prompt: str) -> str:
        img_in, _new_w, _new_h = self._resize_for_qwen(image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_in},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[img_in], padding=True, return_tensors="pt"
        ).to(self.model.device)
        out_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                      do_sample=False)
        trimmed = out_ids[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0]

    @torch.no_grad()
    def detect_one(self, image: Image.Image) -> tuple[list[dict], str]:
        img_in, new_w, new_h = self._resize_for_qwen(image)
        w_orig, h_orig = image.size
        raw = self.generate(image, PROMPT_DETECT)
        sx = w_orig / max(1, new_w)
        sy = h_orig / max(1, new_h)
        dets = self._parse(raw, sx, sy, w_orig, h_orig)
        return dets, raw

    @staticmethod
    def _parse(raw: str, sx: float, sy: float, w: int, h: int) -> list[dict]:
        items = _extract_json_array(raw)
        out = []
        for it in items:
            if not isinstance(it, dict):
                continue
            label = _match_label(it.get("label") or it.get("name") or "")
            # Qwen often emits "bbox_2d" — accept either.
            box_raw = it.get("box") or it.get("bbox") or it.get("bbox_2d")
            box = _coerce_box(box_raw, sx, sy, w, h) if box_raw else None
            if label and box:
                out.append({"label": label, "box": box})
        return out


class InternVL3Adapter(VLMAdapter):
    """InternVL3-8B — boxes typically in [0,1000] normalised image coords."""
    name = "internvl3"

    def __init__(self, path: Path = INTERNVL3_PATH, max_new_tokens: int = 512):
        self.path = path
        self.max_new_tokens = max_new_tokens

    def load(self, device: str = "cuda") -> None:
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.path), trust_remote_code=True, use_fast=False,
        )
        self.model = AutoModel.from_pretrained(
            str(self.path), torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        ).eval().to(device)
        self.device = device

    @staticmethod
    def _find_closest_aspect_ratio(aspect, target_ratios, w, h, tile):
        best_diff = float("inf"); best = (1, 1)
        area = w * h
        for ratio in target_ratios:
            ta = ratio[0] / ratio[1]
            diff = abs(aspect - ta)
            if diff < best_diff:
                best_diff = diff; best = ratio
            elif diff == best_diff and area > 0.5 * tile * tile * ratio[0] * ratio[1]:
                best = ratio
        return best

    @classmethod
    def _dynamic_preprocess(cls, image: Image.Image, tile: int = 448,
                            max_tiles: int = 6, use_thumbnail: bool = True):
        """Official InternVL multi-tile preprocessing.

        Returns a list of `tile×tile` PIL crops covering the image at the
        closest aspect-ratio (≤ max_tiles), plus a thumbnail. For a 640²
        square DW tile with max_tiles=6 → (2,2) split → 4 crops + thumbnail
        = 5 tiles; InternVL3's chat helper inserts num_image_tokens × 5
        placeholders, matching what we feed.
        """
        w, h = image.size
        aspect = w / h
        target_ratios = sorted({
            (i, j) for n in range(1, max_tiles + 1)
            for i in range(1, n + 1) for j in range(1, n + 1)
            if 1 <= i * j <= max_tiles
        }, key=lambda x: x[0] * x[1])
        tr = cls._find_closest_aspect_ratio(aspect, target_ratios, w, h, tile)
        tw, th = tile * tr[0], tile * tr[1]
        blocks = tr[0] * tr[1]
        resized = image.resize((tw, th), Image.BICUBIC)
        out: list[Image.Image] = []
        cols = tw // tile
        for i in range(blocks):
            box = ((i % cols) * tile, (i // cols) * tile,
                   ((i % cols) + 1) * tile, ((i // cols) + 1) * tile)
            out.append(resized.crop(box))
        if use_thumbnail and blocks != 1:
            out.append(image.resize((tile, tile), Image.BICUBIC))
        return out

    def _prep_pixels(self, image: Image.Image):
        from torchvision import transforms
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        # Tile to 448² blocks + thumbnail. For 640² DW input this yields
        # 5 tiles (2×2 split + thumbnail); the chat() helper then inserts
        # num_image_token × 5 placeholders, which matches the pixel_values
        # we feed. Single-tile 448² loses too much context; single-tile 896²
        # mismatches the placeholder count.
        tile_size = 448
        tiles = self._dynamic_preprocess(image.convert("RGB"), tile=tile_size, max_tiles=6)
        tx = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        pixel_values = torch.stack([tx(t) for t in tiles]).to(
            dtype=torch.bfloat16, device=self.device,
        )
        return pixel_values, len(tiles)

    @torch.no_grad()
    def generate(self, image: Image.Image, prompt: str) -> str:
        pixel_values, n_tiles = self._prep_pixels(image)
        gen_cfg = dict(max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.model.chat(
            self.tokenizer, pixel_values, "<image>\n" + prompt, gen_cfg,
            num_patches_list=[n_tiles],
        )

    @torch.no_grad()
    def detect_one(self, image: Image.Image) -> tuple[list[dict], str]:
        raw = self.generate(image, PROMPT_DETECT)

        # InternVL3 grounded output formats observed in the wild:
        #   1) JSON list of {label, box: [x,y,x,y]}     (what our prompt asks)
        #   2) <ref>cls</ref><box>[[x1,y1,x2,y2]]</box>  (native grounding)
        # Coords commonly in [0,1000]. Heuristic: if ANY coord > 1.5 * max(w,h)
        # of the original image then they're probably normalised.
        w, h = image.size
        items = _extract_json_array(raw)
        boxes = []
        for it in items:
            if not isinstance(it, dict):
                continue
            label = _match_label(it.get("label") or it.get("name") or "")
            box_raw = it.get("box") or it.get("bbox") or it.get("bbox_2d")
            if not (label and box_raw and len(box_raw) == 4):
                continue
            try:
                vals = [float(v) for v in box_raw]
            except Exception:
                continue
            maxv = max(vals)
            if maxv <= 1.01:                       # 0..1 normalised
                sx, sy = w, h
            elif maxv > 1.05 * max(w, h):          # 0..1000 normalised
                sx, sy = w / 1000.0, h / 1000.0
            else:                                  # already in pixels
                sx, sy = 1.0, 1.0
            box = _coerce_box(vals, sx, sy, w, h)
            if box:
                boxes.append({"label": label, "box": box})

        if not boxes:
            # Fallback: parse native <ref>..</ref><box>[[..]]</box> emissions.
            pat = re.compile(
                r"<ref>(?P<label>[^<]+)</ref>\s*<box>\s*\[\[\s*"
                r"(?P<x1>-?\d+)\s*,\s*(?P<y1>-?\d+)\s*,\s*"
                r"(?P<x2>-?\d+)\s*,\s*(?P<y2>-?\d+)\s*\]\]\s*</box>",
                flags=re.IGNORECASE,
            )
            for m in pat.finditer(raw):
                label = _match_label(m.group("label"))
                if not label:
                    continue
                vals = [int(m.group(k)) for k in ("x1", "y1", "x2", "y2")]
                box = _coerce_box(vals, w / 1000.0, h / 1000.0, w, h)
                if box:
                    boxes.append({"label": label, "box": box})

        return boxes, raw


class CLIPAdapter(VLMAdapter):
    """CLIP ViT-B/32 zero-shot multi-label classifier.

    Per-class log-ratio scoring: predict a label iff the cosine similarity of
    the image embedding to the class positive-text ensemble exceeds its
    similarity to the shared negative-text ensemble (margin = threshold).
    No threshold calibration — fully zero-shot.

    Bypasses the generate() → parse_classification() pipeline.  run_classify
    checks for classify_image() via duck-typing and calls it directly.
    """
    name = "clip"

    _POS_TEMPLATES = [
        "an aerial drone photograph of {}",
        "aerial view of {} as illegally dumped waste",
        "a drone image showing {} in an outdoor area",
    ]
    _NEG_TEMPLATES = [
        "an aerial drone photograph with no waste",
        "aerial view of clean land, no dumped material",
        "aerial drone photograph of a clean outdoor area",
    ]

    def __init__(self, path: Path = CLIP_CACHE, threshold: float = 0.0):
        self.path = path
        self.threshold = threshold  # log-ratio margin; 0.0 = equal-prob boundary

    def load(self, device: str = "cuda") -> None:
        from transformers import CLIPModel, CLIPProcessor
        self.processor = CLIPProcessor.from_pretrained(str(self.path))
        self.model = CLIPModel.from_pretrained(
            str(self.path), torch_dtype=torch.float32,
        ).eval().to(device)
        self.device = device
        # Pre-compute the negative text embedding (shared across all images).
        with torch.no_grad():
            neg_inp = self.processor(
                text=self._NEG_TEMPLATES, return_tensors="pt", padding=True,
            ).to(device)
            neg_embs = self.model.get_text_features(**neg_inp)
            neg_embs = neg_embs / neg_embs.norm(dim=-1, keepdim=True)
        # Mean of L2-normalised embeddings (unnormalised mean is fine for
        # comparison; we never need unit-norm on this side).
        self._neg_emb = neg_embs.mean(dim=0, keepdim=True)  # [1, d]

    @torch.no_grad()
    def _img_emb(self, image: Image.Image):
        inp = self.processor(images=image, return_tensors="pt").to(self.device)
        emb = self.model.get_image_features(**inp)
        return emb / emb.norm(dim=-1, keepdim=True)  # [1, d]

    @torch.no_grad()
    def _text_emb(self, texts: list[str]):
        inp = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        emb = self.model.get_text_features(**inp)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.mean(dim=0, keepdim=True)  # [1, d], ensemble mean

    def classify_image(self, image: Image.Image,
                       cats: list[str], cues: dict[str, str],
                       clip_tags: dict[str, list[str]] | None = None) -> set[str]:
        """Zero-shot per-class prediction via log-ratio scoring.

        Positive text ensemble = aerial_cue templates + one template per
        clip_tag synonym (if provided).  Averaging over more texts reduces
        variance and lets CLIP's text encoder match via either the visual
        description *or* the common synonym.
        """
        img = self._img_emb(image)                  # [1, d]
        neg_score = float(img @ self._neg_emb.T)    # scalar
        pred: set[str] = set()
        for cat in cats:
            cue = cues.get(cat, cat).lower()
            pos_texts = [t.format(cue) for t in self._POS_TEMPLATES]
            # Append one short template per synonym tag.
            for tag in (clip_tags or {}).get(cat, []):
                pos_texts.append(f"aerial drone photograph of {tag}")
                pos_texts.append(f"aerial view of {tag} illegally dumped")
            pos_emb = self._text_emb(pos_texts)     # [1, d]
            pos_score = float(img @ pos_emb.T)      # scalar
            if pos_score > neg_score + self.threshold:
                pred.add(cat)
        return pred

    def generate(self, image: Image.Image, prompt: str) -> str:
        raise NotImplementedError(
            "CLIPAdapter does not generate text; "
            "run_classify calls classify_image() directly."
        )


ADAPTERS = {
    "clip": CLIPAdapter,
    "qwen2_5vl": QwenAdapter,
    "internvl3": InternVL3Adapter,
}


# --- paper-10-restricted ground truth --------------------------------------

def build_paper10_gt(ds: DroneWasteSegmentation) -> tuple[dict, set[int]]:
    """COCO-format GT restricted to DRONEWASTE_PAPER_10 categories."""
    cat_by_id: dict[int, dict] = {}
    with (Path("/home/ids/diecidue/data/dronewaste") / "dronewaste_v1.0.json").open() as f:
        full = json.load(f)
    for c in full["categories"]:
        cat_by_id[c["id"]] = c
    keep_ids = {cid for cid, c in cat_by_id.items() if c["name"] in DRONEWASTE_PAPER_10}

    images, annotations = [], []
    ann_id = 1
    for s in ds.samples:
        images.append({
            "id": s["id"], "file_name": s["file_name"],
            "width": s["width"], "height": s["height"],
        })
        for a in s["annotations"]:
            if a["category_id"] not in keep_ids:
                continue
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            annotations.append({
                "id": ann_id, "image_id": s["id"],
                "category_id": a["category_id"],
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h), "iscrowd": 0,
            })
            ann_id += 1
    return {
        "images": images,
        "annotations": annotations,
        "categories": [cat_by_id[cid] for cid in sorted(keep_ids)],
    }, keep_ids


# --- main ------------------------------------------------------------------

def run_detect(adapter: VLMAdapter, args) -> dict:
    """Legacy grounded-detection path; scored via pycocotools paper-10 mAP."""
    ds = DroneWasteSegmentation(split=args.split)
    print(f"[data] split={args.split} n={len(ds)} fg-categories={len(ds.categories)}")
    name_to_cat_id = {ds.categories[i]: cid for cid, i in ds.cat_id_to_idx.items()}
    paper10_cat_ids = {name_to_cat_id[n] for n in DRONEWASTE_PAPER_10}

    samples = ds.samples if args.limit <= 0 else ds.samples[: args.limit]
    detections: list[dict] = []
    n_parsed_empty = n_parsed_ok = 0
    raw_dump: list[dict] = []
    t0 = time.time()
    for k, s in enumerate(samples):
        img_path = ds.images_dir / s["file_name"]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[warn] cannot open {img_path}: {e}", flush=True); continue
        try:
            dets, raw = adapter.detect_one(img)
        except Exception as e:
            print(f"[warn] inference failed on {s['file_name']}: {e}", flush=True); continue
        if args.save_raw is not None:
            raw_dump.append({"image_id": s["id"], "file": s["file_name"],
                             "raw": raw, "parsed": dets})
        if dets: n_parsed_ok += 1
        else:    n_parsed_empty += 1
        for d in dets:
            cid = name_to_cat_id.get(d["label"])
            if cid is None or cid not in paper10_cat_ids: continue
            detections.append({"image_id": s["id"], "category_id": cid,
                               "bbox": d["box"], "score": 1.0})
        if (k + 1) % 20 == 0 or (k + 1) == len(samples):
            dt = time.time() - t0
            print(f"  [{k+1:>4d}/{len(samples)}] dets={len(detections)} "
                  f"empty={n_parsed_empty} ok={n_parsed_ok} elapsed={dt/60:.1f}min", flush=True)

    if args.save_raw is not None:
        args.save_raw.parent.mkdir(parents=True, exist_ok=True)
        with args.save_raw.open("w", encoding="utf-8") as f:
            for r in raw_dump:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[saved-raw] {args.save_raw}")

    gt, _ = build_paper10_gt(ds)
    seen_image_ids = {s["id"] for s in samples}
    gt["images"] = [im for im in gt["images"] if im["id"] in seen_image_ids]
    gt["annotations"] = [a for a in gt["annotations"] if a["image_id"] in seen_image_ids]

    metrics = cocoeval(gt, detections) if detections else {
        "overall": {"mAP": 0.0, "AP50": 0.0, "AP75": 0.0,
                    "APs": 0.0, "APm": 0.0, "APl": 0.0},
        "per_class": {n: 0.0 for n in DRONEWASTE_PAPER_10},
    }
    pc_aps = [v for v in metrics["per_class"].values()
              if isinstance(v, float) and not np.isnan(v)]
    paper10_mean = float(np.mean(pc_aps)) if pc_aps else 0.0

    print()
    print(f"=== {args.model} — DroneWaste {args.split} DETECT (paper-10) ===")
    print(f"  imgs processed   = {len(samples)}")
    print(f"  empty-parse imgs = {n_parsed_empty}")
    print(f"  total detections = {len(detections)}")
    print(f"  mAP @ [.5:.95]   = {metrics['overall']['mAP']:.3f}")
    print(f"  AP@0.5           = {metrics['overall']['AP50']:.3f}")
    print(f"  paper-10 mean AP = {paper10_mean:.3f}")
    return {
        "task": "detect",
        "model": args.model,
        "split": args.split,
        "n_images": len(samples),
        "n_detections": len(detections),
        "n_empty_parse": n_parsed_empty,
        "detection": metrics,
        "paper10_mean_AP": paper10_mean,
    }


def _load_classification_samples(dataset: str, dataset_spec: dict, limit: int):
    """Return (cats, test_samples) for the requested dataset.

    DW paper-10 uses the 70/30 site-stratified split (seed=0) from
    dinotxt_zeroshot.run_dw_paper10 so VLM numbers plug into the §3.5.1
    comparison row directly. AW m2 / m4 use their published test split
    (filter to images present on disk, matching dinotxt's regime).
    """
    from collections import defaultdict
    if dataset == "dw_paper10":
        from src.datasets import load_dronewaste_multilabel  # noqa: E402
        cats, samples_all = load_dronewaste_multilabel(
            "/home/ids/diecidue/data/dronewaste",
            categories_filter=dataset_spec["cats"],
        )
        site_to_idx: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(samples_all):
            site_to_idx[s.image_source].append(i)
        rng = np.random.default_rng(0)
        test_idx: list[int] = []
        for _site, idxs in site_to_idx.items():
            idxs = list(idxs); rng.shuffle(idxs)
            cut = int(len(idxs) * 0.7)
            test_idx.extend(idxs[cut:])
        test_samples = [samples_all[i] for i in test_idx]
    elif dataset in ("aw_m2", "aw_m4"):
        from src.datasets import load_aerialwaste_mcml  # noqa: E402
        version = "m2" if dataset == "aw_m2" else "m4"
        cats, test_samples = load_aerialwaste_mcml(
            "/home/ids/diecidue/data/aerialwaste",
            split="test", version=version,
        )
        # Filter to images actually on disk (PNEO subset isn't shipped).
        before = len(test_samples)
        test_samples = [s for s in test_samples if s.image_path.exists()]
        if before != len(test_samples):
            print(f"[data] filtered missing-on-disk: -{before-len(test_samples)}", flush=True)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")

    if limit > 0:
        test_samples = test_samples[:limit]
    # Sanity check class lists match the registry.
    if list(cats) != dataset_spec["cats"]:
        raise RuntimeError(
            f"dataset cats {cats!r} != registry {dataset_spec['cats']!r}"
        )
    return cats, test_samples


def run_classify(adapter: VLMAdapter, args) -> dict:
    """Closed-vocab / open-caption multi-label classification.

    Dataset is selected via --dataset (dw_paper10 / aw_m2 / aw_m4); each
    dataset uses the same split + scoring as the existing dino.txt
    zero-shot eval so the resulting micro / macro F1 plug into the
    §3.5.1 comparison row directly.
    """
    from src.dinotxt_zeroshot import ml_metrics  # noqa: E402

    dataset_spec = DATASETS[args.dataset]
    cats, test_samples = _load_classification_samples(
        args.dataset, dataset_spec, args.limit,
    )
    n_cats = len(cats)
    print(f"[data] dataset={args.dataset} test={len(test_samples)} classes={n_cats}")

    # open_cot uses two shared CoT prompts; other styles pull from the dataset registry.
    if args.prompt_style == "open_cot":
        prompt = None
        cot_classify_prompt = make_cot_classify_prompt(dataset_spec["clip_tags"])
        print(f"[task] classify  prompt_style=open_cot  "
              f"describe-len={len(PROMPT_DESCRIBE)}  classify-len={len(cot_classify_prompt)} chars")
        print(f"[task] cot_classify_prompt: {cot_classify_prompt}")
    else:
        prompt = dataset_spec["prompt_classify"][args.prompt_style]
        print(f"[task] classify  prompt_style={args.prompt_style}  prompt-len={len(prompt)} chars")

    Y_true = np.zeros((len(test_samples), n_cats), dtype=np.int32)
    Y_pred = np.zeros((len(test_samples), n_cats), dtype=np.int32)
    n_empty = n_nonempty = 0
    raw_dump: list[dict] = []
    t0 = time.time()
    for k, s in enumerate(test_samples):
        try:
            img = Image.open(s.image_path).convert("RGB")
        except Exception as e:
            print(f"[warn] cannot open {s.image_path}: {e}", flush=True); continue
        try:
            if hasattr(adapter, "classify_image"):
                # CLIP zero-shot path
                pred_labels = adapter.classify_image(
                    img, cats, dataset_spec["cues"],
                    clip_tags=dataset_spec.get("clip_tags"),
                )
                raw = json.dumps({"clip_preds": sorted(pred_labels)})
            elif args.prompt_style == "open_cot":
                # Chain-of-thought: describe → classify (free vocab, keyword-bag parse)
                raw_turn1, raw = adapter.generate_cot(img, PROMPT_DESCRIBE, cot_classify_prompt)
                pred_labels = parse_keywords(raw, dataset_spec["keywords"])
            else:
                raw = adapter.generate(img, prompt)
                pred_labels = parse_classification(raw, args.prompt_style, dataset_spec)
        except Exception as e:
            print(f"[warn] inference failed on {s.image_path.name}: {e}", flush=True); continue

        gt_labels = set(s.extra["gt_categories"])
        for c in gt_labels:
            if c in cats:
                Y_true[k, cats.index(c)] = 1
        for c in pred_labels:
            if c in cats:
                Y_pred[k, cats.index(c)] = 1
        if pred_labels: n_nonempty += 1
        else:           n_empty += 1

        if args.save_raw is not None:
            rec = {"image_id": s.image_id, "file": s.image_path.name,
                   "raw": raw, "parsed": sorted(pred_labels), "gt": sorted(gt_labels)}
            if args.prompt_style == "open_cot":
                rec["raw_turn1"] = raw_turn1
            raw_dump.append(rec)

        if (k + 1) % 20 == 0 or (k + 1) == len(test_samples):
            dt = time.time() - t0
            print(f"  [{k+1:>4d}/{len(test_samples)}] nonempty={n_nonempty} "
                  f"empty={n_empty} elapsed={dt/60:.1f}min", flush=True)

    if args.save_raw is not None:
        args.save_raw.parent.mkdir(parents=True, exist_ok=True)
        with args.save_raw.open("w", encoding="utf-8") as f:
            for r in raw_dump:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[saved-raw] {args.save_raw}")

    # ml_metrics signature: (cats, Y_true, Y_pred, scores). VLMs have no
    # ranking score, so reuse Y_pred as a degenerate score (per-class AP
    # then collapses to precision); we report it but the headline numbers
    # are F1, not AP.
    rep = ml_metrics(cats, Y_true, Y_pred, Y_pred.astype(np.float32))
    rep["task"] = f"{args.dataset}_classify"
    rep["dataset"] = args.dataset
    rep["model"] = args.model
    rep["prompt_style"] = args.prompt_style
    rep["n_test"] = len(test_samples)
    rep["n_empty_parse"] = n_empty
    rep["n_nonempty_parse"] = n_nonempty
    rep["prompt"] = prompt

    print()
    print(f"=== {args.model} {args.prompt_style} — {args.dataset} ({len(test_samples)} imgs) ===")
    print(f"  micro F1 = {rep['micro']['f1']:.4f}   "
          f"macro F1 = {rep['macro']['f1']:.4f}")
    print(f"  nonempty preds = {n_nonempty}   empty = {n_empty}")
    print()
    print("per-class F1:")
    for name in cats:
        d = rep["per_class"].get(name, {})
        sup = d.get("support", 0)
        f1 = d.get("f1")
        f1_str = "n/a" if f1 is None else f"{f1:.3f}"
        print(f"  {name[:40]:40s}  F1={f1_str}  support={sup}")
    return rep


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=sorted(ADAPTERS.keys()), required=True,
                   help="clip = zero-shot CLIP ViT-B/32 (classify only, "
                        "ignores --prompt-style); qwen2_5vl / internvl3 = generative VLMs")
    p.add_argument("--task", choices=["detect", "classify"], default="classify",
                   help="classify = closed/open prompt → multi-label F1 (default); "
                        "detect = grounded JSON → paper-10 mAP (legacy, DW only)")
    p.add_argument("--dataset", choices=sorted(DATASETS.keys()),
                   default="dw_paper10",
                   help="only used when --task classify")
    p.add_argument("--prompt-style",
                   choices=sorted(PROMPTS_CLASSIFY.keys()) + ["open_cot"],
                   default="closed_vocab",
                   help="only used when --task classify. open_cot: two-turn CoT "
                        "(describe then classify, free vocab, keyword-bag parse); "
                        "ignored for clip model")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--limit", type=int, default=0,
                   help="if >0, only process the first N test images (smoke test)")
    p.add_argument("--save-raw", type=Path, default=None,
                   help="optional path to dump per-image raw responses (jsonl)")
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    adapter_cls = ADAPTERS[args.model]
    adapter = adapter_cls()
    print(f"[model] loading {args.model} from {adapter.path} ...")
    adapter.load("cuda")
    print("[model] ready", flush=True)

    if args.task == "detect":
        out = run_detect(adapter, args)
    else:
        out = run_classify(adapter, args)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with args.out_json.open("w") as f:
            json.dump(out, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))
        print(f"[saved] {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
