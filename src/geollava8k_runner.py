"""GeoLLaVA-8K runner.

Loads `initiacms/GeoLLaVA-8K` via the vendored LongVA-based code in
`vendored/GeoLLaVA-8K/longva/`. Mirrors the `VLMRunner` protocol.

Notes:
- Backbone: Qwen2-7B + CLIP-L/14-336, "anyres" tiling for UHR imagery.
- We force `attn_implementation="eager"` because flash-attn isn't installed in
  the project env.
- `LlavaQwenForCausalLM.generate` uses `inputs_embeds` under the hood, so the
  returned sequences contain only newly-generated tokens (no prompt echo).
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
_LONGVA_PATH = REPO_ROOT / "vendored" / "GeoLLaVA-8K" / "longva"
if str(_LONGVA_PATH) not in sys.path:
    sys.path.insert(0, str(_LONGVA_PATH))

from longva.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN  # noqa: E402
from longva.conversation import SeparatorStyle, conv_templates  # noqa: E402
from longva.mm_utils import process_images, tokenizer_image_token  # noqa: E402
from longva.model.builder import load_pretrained_model  # noqa: E402

from src.runner import (  # noqa: E402
    VLMResponse,
    build_letter_ids,
    build_yes_no_ids,
    first_token_probs,
)


class GeoLlava8KRunner:
    def __init__(
        self,
        model_id: str = "initiacms/GeoLLaVA-8K",
        device: str = "cuda",
        max_new_tokens: int = 96,
        conv_mode: str = "qwen_1_5",
        attn_implementation: str = "eager",
    ) -> None:
        # `load_pretrained_model` dispatches on substring of model_name; must
        # contain "qwen" (for the Qwen2 branch) and "llava" or "longva".
        model_name = "llava-qwen-geollava8k"
        self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
            model_id,
            model_base=None,
            model_name=model_name,
            attn_implementation=attn_implementation,
            device_map=device,
        )
        # Unify dtype across all submodules (vision_tower, projector, reghead,
        # llm) to avoid the Half/BFloat16 mismatch their custom code triggers
        # in the GeoLLaVA-8K Anchored Token Selection head.
        self.model = self.model.to(torch.bfloat16)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.conv_mode = conv_mode

        self._yes_ids, self._no_ids = build_yes_no_ids(self.tokenizer)
        self._letter_ids = build_letter_ids(self.tokenizer)

    def _build_prompt(self, question: str) -> str:
        qs = f"{DEFAULT_IMAGE_TOKEN}\n{question}"
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

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

        processed = process_images([image], self.image_processor, self.model.config)
        if isinstance(processed, list):
            images_input = [p.to(self.device, dtype=torch.bfloat16) for p in processed]
        else:
            images_input = processed.to(self.device, dtype=torch.bfloat16)
        image_sizes = [image.size]  # (W, H)

        want_scores = compute_yes_no or compute_letters
        gen = self.model.generate(
            input_ids,
            images=images_input,
            image_sizes=image_sizes,
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_scores=want_scores,
        )

        # LlavaQwen uses inputs_embeds in generate, so gen.sequences contains
        # only the newly-generated tokens.
        out_ids = gen.sequences[0]
        text = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
        # Strip trailing chat-end marker if any.
        conv = conv_templates[self.conv_mode]
        if conv.sep and text.endswith(conv.sep):
            text = text[: -len(conv.sep)].strip()

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
