"""Q1-Q4 prompts from project.MD section 3, plus parsers."""
from __future__ import annotations

import re

Q1 = (
    "Does this image contain illegal waste dumping or accumulation? "
    "Answer yes or no, then briefly justify."
)

Q2 = "Describe what you see in this aerial image."

Q3 = (
    "Is this image showing: (a) legitimate industrial site, (b) illegal waste dump, "
    "(c) construction site, (d) agricultural land, (e) other? Answer with a single letter."
)

Q4 = (
    "How many distinct waste piles or accumulations are visible, and where in the image "
    "are they located? Use coordinates relative to the image (top-left, center, etc.)."
)

ALL_QUESTIONS: dict[str, str] = {"Q1": Q1, "Q2": Q2, "Q3": Q3, "Q4": Q4}


_YES_RE = re.compile(r"\b(yes|si|sì|yeah|affirmative|present|detected)\b", re.IGNORECASE)
_NO_RE = re.compile(r"\b(no|nope|negative|absent|none)\b", re.IGNORECASE)


def parse_yes_no(text: str) -> int | None:
    """Return 1 for yes, 0 for no, None if undetermined."""
    head = text.strip().split("\n", 1)[0].strip().lower()
    head = head.lstrip("*-#:. \t\"'")
    first_word = head.split()[0] if head else ""
    if first_word.startswith(("yes", "sì", "si")):
        return 1
    if first_word.startswith(("no", "nope")):
        return 0
    if _YES_RE.search(text) and not _NO_RE.search(text):
        return 1
    if _NO_RE.search(text) and not _YES_RE.search(text):
        return 0
    return None


_LETTER_RE = re.compile(r"\b([abcde])\b", re.IGNORECASE)


def parse_letter(text: str) -> str | None:
    """Return the first a-e letter found in the response, lowercase."""
    cleaned = text.strip().lower().lstrip("*-#:. \t\"'(")
    if cleaned and cleaned[0] in "abcde":
        return cleaned[0]
    m = _LETTER_RE.search(text)
    return m.group(1).lower() if m else None


# project.MD Q3: option (b) is "illegal waste dump" -> positive class
LETTER_TO_BINARY: dict[str, int] = {"a": 0, "b": 1, "c": 0, "d": 0, "e": 0}


# ---------------------------------------------------------------------------
# Q5 — combined-prompt multi-label classification.
# ---------------------------------------------------------------------------


def build_q5_multilabel(categories: list[str]) -> str:
    """Single prompt asking which of the listed waste types are visible."""
    bullet = "\n".join(f"- {c}" for c in categories)
    return (
        "Below is a fixed list of waste material types. Which of these types "
        "are visible in this aerial image? Reply with ONLY the names from the "
        "list, separated by commas. Use the exact names as written. If none "
        "of these are visible, reply with the single word: none.\n\n"
        f"Waste types:\n{bullet}"
    )


def parse_q5_multilabel(text: str, categories: list[str]) -> set[str]:
    """Parse a model response into the set of categories it claims are present.

    Strategy: case-insensitive substring match against each canonical name,
    longest-name-first to avoid 'Plastic' eating 'Plastic packaging'. Returns
    an empty set if the response is 'none' or no name matches.
    """
    haystack = text.lower()
    stripped = haystack.strip().lstrip("*-#:. \t\"'")
    if stripped.startswith("none") or stripped == "":
        return set()
    out: set[str] = set()
    consumed = [False] * len(haystack)
    for name in sorted(categories, key=len, reverse=True):
        needle = name.lower()
        start = 0
        while True:
            idx = haystack.find(needle, start)
            if idx == -1:
                break
            # Skip if any of these characters were already consumed by a
            # longer canonical name (e.g. avoid double-counting "Plastic"
            # inside "Plastic packaging").
            if not any(consumed[idx : idx + len(needle)]):
                out.add(name)
                for i in range(idx, idx + len(needle)):
                    consumed[i] = True
            start = idx + len(needle)
    return out
