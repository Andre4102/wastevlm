"""GeoChat runner.

Loads `MBZUAI/geochat-7B` via the vendored upstream code in
`vendored/GeoChat/`. Mirrors the `VLMRunner` protocol: returns a `VLMResponse`
with generated text plus calibrated p(yes)/p(no) and per-letter probs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
_GEOCHAT_PATH = REPO_ROOT / "vendored" / "GeoChat"
if str(_GEOCHAT_PATH) not in sys.path:
    sys.path.insert(0, str(_GEOCHAT_PATH))

from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # noqa: E402
from geochat.conversation import SeparatorStyle, conv_templates  # noqa: E402
from geochat.mm_utils import tokenizer_image_token  # noqa: E402
from geochat.model.builder import load_pretrained_model  # noqa: E402

from src.runner import (  # noqa: E402
    VLMResponse,
    build_letter_ids,
    build_yes_no_ids,
    first_token_probs,
)


class GeoChatRunner:
    def __init__(
        self,
        model_id: str = "MBZUAI/geochat-7B",
        device: str = "cuda",
        max_new_tokens: int = 96,
        image_size: int = 504,
        conv_mode: str = "llava_v1",
    ) -> None:
        # `load_pretrained_model` dispatches by substring on model_name, so we
        # pass a name that contains "geochat" to take the correct code path.
        model_name = "geochat-7B"
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_id, model_base=None, model_name=model_name, device=device
        )
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.image_size = image_size
        self.conv_mode = conv_mode

        self._yes_ids, self._no_ids = build_yes_no_ids(self.tokenizer)
        self._letter_ids = build_letter_ids(self.tokenizer)

    def _build_prompt(self, question: str) -> str:
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        out = self.image_processor.preprocess(
            [image],
            crop_size={"height": self.image_size, "width": self.image_size},
            size={"shortest_edge": self.image_size},
            return_tensors="pt",
        )
        return out["pixel_values"].to(self.device, dtype=torch.float16)

    @torch.inference_mode()
    def ask(
        self,
        image: Image.Image,
        question: str,
        *,
        compute_yes_no: bool = False,
        compute_letters: bool = False,
    ) -> VLMResponse:
        prompt = self._build_prompt(question)
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.device)
        )
        images = self._preprocess_image(image)

        want_scores = compute_yes_no or compute_letters
        gen = self.model.generate(
            input_ids,
            images=images,
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            length_penalty=2.0,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=want_scores,
        )

        # Generate echoes the input_ids prefix (containing the -200 image placeholder
        # which would crash sentencepiece) — slice it off before decoding.
        prompt_len = input_ids.shape[1]
        out_ids = gen.sequences[0, prompt_len:]
        text = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        # Strip the conversation separator if it leaked through.
        conv = conv_templates[self.conv_mode]
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        if stop_str and text.endswith(stop_str):
            text = text[: -len(stop_str)].strip()

        p_yes = p_no = None
        letter_probs = None
        if want_scores and gen.scores:
            p_yes, p_no, letter_probs = first_token_probs(
                gen.scores[0][0],
                self._yes_ids,
                self._no_ids,
                self._letter_ids,
                compute_yes_no,
                compute_letters,
            )

        return VLMResponse(text=text, p_yes=p_yes, p_no=p_no, letter_probs=letter_probs)
