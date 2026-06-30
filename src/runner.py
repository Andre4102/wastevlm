"""Inference runner protocol + LLaVA-NeXT implementation.

Each runner answers a (image, question) pair and returns a `VLMResponse`
with the generated text plus optional first-token logprob-derived calibrated
probabilities (p_yes / p_no for Q1, per-letter probs for Q3).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from PIL import Image
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor


@dataclass
class VLMResponse:
    text: str
    p_yes: float | None = None
    p_no: float | None = None
    letter_probs: dict[str, float] | None = None


# Back-compat alias for old name used in earlier checkpoints / scripts.
LlavaResponse = VLMResponse


class VLMRunner(Protocol):
    def ask(
        self,
        image: Image.Image,
        question: str,
        *,
        compute_yes_no: bool = False,
        compute_letters: bool = False,
    ) -> VLMResponse: ...


def _collect_token_ids(tokenizer, words: list[str]) -> list[int]:
    ids: set[int] = set()
    for w in words:
        for tok_id in tokenizer.encode(w, add_special_tokens=False):
            ids.add(int(tok_id))
    return sorted(ids)


def build_yes_no_ids(tokenizer) -> tuple[list[int], list[int]]:
    yes = _collect_token_ids(tokenizer, ["yes", "Yes", "YES", " yes", " Yes"])
    no = _collect_token_ids(tokenizer, ["no", "No", "NO", " no", " No"])
    return yes, no


def build_letter_ids(tokenizer) -> dict[str, list[int]]:
    return {
        ch: _collect_token_ids(tokenizer, [ch, ch.upper(), f" {ch}", f" {ch.upper()}"])
        for ch in "abcde"
    }


def first_token_probs(
    first_logits: torch.Tensor,
    yes_ids: list[int],
    no_ids: list[int],
    letter_ids: dict[str, list[int]] | None,
    want_yes_no: bool,
    want_letters: bool,
) -> tuple[float | None, float | None, dict[str, float] | None]:
    probs = torch.softmax(first_logits.float(), dim=-1)
    p_yes = p_no = None
    letter_probs = None
    if want_yes_no:
        p_yes = float(probs[yes_ids].sum().item())
        p_no = float(probs[no_ids].sum().item())
    if want_letters and letter_ids:
        letter_probs = {ch: float(probs[ids].sum().item()) for ch, ids in letter_ids.items()}
    return p_yes, p_no, letter_probs


class LlavaNextRunner:
    def __init__(
        self,
        model_id: str = "llava-hf/llava-v1.6-mistral-7b-hf",
        dtype: str = "bf16",
        device: str = "cuda",
        max_new_tokens: int = 96,
    ) -> None:
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
        ).to(device)
        self.model.eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

        tok = self.processor.tokenizer
        self._yes_ids, self._no_ids = build_yes_no_ids(tok)
        self._letter_ids = build_letter_ids(tok)

    def _build_prompt(self, question: str) -> str:
        conv = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        return self.processor.apply_chat_template(conv, add_generation_prompt=True)

    @torch.inference_mode()
    def ask(
        self,
        image: Image.Image,
        question: str,
        *,
        compute_yes_no: bool = False,
        compute_letters: bool = False,
    ) -> LlavaResponse:
        prompt = self._build_prompt(question)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            self.device
        )

        gen = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=compute_yes_no or compute_letters,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        prompt_len = inputs["input_ids"].shape[1]
        out_ids = gen.sequences[0, prompt_len:]
        text = self.processor.tokenizer.decode(out_ids, skip_special_tokens=True).strip()

        p_yes = p_no = None
        letter_probs = None
        if (compute_yes_no or compute_letters) and gen.scores:
            p_yes, p_no, letter_probs = first_token_probs(
                gen.scores[0][0],
                self._yes_ids,
                self._no_ids,
                self._letter_ids,
                compute_yes_no,
                compute_letters,
            )

        return VLMResponse(text=text, p_yes=p_yes, p_no=p_no, letter_probs=letter_probs)


def open_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")
