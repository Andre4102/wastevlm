"""Confirm GeoLLaVA-8K's Q1 collapse is task-framing, not precision/pipeline.

The training data (initiacms/GeoLLaVA-Data) is overwhelmingly multiple-choice
"Select the best answer (A/B/C/D)" — there are no "Does this image contain X?
Yes/no" prompts. So we rephrase Q1 as multiple-choice and see whether
GeoLLaVA-8K can in fact discriminate waste vs no-waste when asked in its
training-data style.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import load_aerialwaste, load_dronewaste  # noqa: E402
from src.geollava8k_runner import GeoLlava8KRunner  # noqa: E402


Q1_YESNO = (
    "Does this image contain illegal waste dumping or accumulation? "
    "Answer yes or no, then briefly justify."
)

Q1_MC_YES_FIRST = (
    "Is there visible waste, debris, or illegal dumping in this aerial image? "
    "The choices are listed below:\n"
    "(A) Yes, waste or debris is visible.\n"
    "(B) No, the image shows no visible waste.\n"
    "Select the best answer for the multiple-choice question based on the "
    "image. Only respond with the letter corresponding to the correct answer (A, B)."
)

Q1_MC_NO_FIRST = (
    "Is there visible waste, debris, or illegal dumping in this aerial image? "
    "The choices are listed below:\n"
    "(A) No, the image shows no visible waste.\n"
    "(B) Yes, waste or debris is visible.\n"
    "Select the best answer for the multiple-choice question based on the "
    "image. Only respond with the letter corresponding to the correct answer (A, B)."
)


def main() -> int:
    drone = load_dronewaste("/home/ids/diecidue/data/dronewaste")
    aerial = load_aerialwaste("/home/ids/diecidue/data/aerialwaste", "testing")

    def first(samples, label):
        return next(s for s in samples if s.label == label)

    probes = [
        ("drone_pos", first(drone, 1)),
        ("drone_neg", first(drone, 0)),
        ("aerial_pos", first(aerial, 1)),
        ("aerial_neg", first(aerial, 0)),
    ]

    runner = GeoLlava8KRunner(max_new_tokens=64)
    print("\n" + "=" * 70)
    print("GeoLLaVA-8K — Q1 yes/no vs Q1 multiple-choice")
    print("=" * 70)
    for tag, sample in probes:
        img = Image.open(sample.image_path).convert("RGB")
        print(f"\n--- {tag} (label={sample.label}, id={sample.image_id}) ---")
        for label, q in (
            ("yes/no", Q1_YESNO),
            ("MC yes=A", Q1_MC_YES_FIRST),
            ("MC no=A", Q1_MC_NO_FIRST),
        ):
            resp = runner.ask(
                img, q,
                compute_yes_no=(label == "yes/no"),
                compute_letters=(label.startswith("MC")),
            )
            text = resp.text.replace("\n", " ")[:120]
            extras = ""
            if resp.p_yes is not None:
                extras = f"  [py={resp.p_yes:.3f} pn={resp.p_no:.3f}]"
            if resp.letter_probs:
                lp = resp.letter_probs
                extras = f"  [a={lp['a']:.3f} b={lp['b']:.3f}]"
            print(f"  {label:14s} →{extras}  {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
