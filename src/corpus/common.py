"""Shared helpers for the waste-corpus web-collection pipeline.

Design notes
------------
- The *raw* corpus is model-agnostic JSONL (one document per line). Tokenisation
  into per-domain Arrow shards is a separate step (``tokenize_domain.py``) so the
  same corpus can be re-tokenised for a different base model (LLaMA now, Qwen
  later) without re-scraping.
- Collectors depend only on the stdlib + ``requests`` so they run in the ``base``
  env on a Leonardo *login* node (compute nodes have no internet).

Document schema (one JSON object per line)::

    {
      "id":         "wikipedia:Municipal_solid_waste",  # source-prefixed, stable
      "source":     "wikipedia" | "eurlex" | "lea",
      "url":        "https://...",
      "license":    "CC-BY-SA-4.0" | "EUR-Lex-reuse-2011/833/EU" | "verify",
      "title":      "Municipal solid waste",
      "lang":       "en",
      "text":       "…plain text…",
      "n_chars":    12345,
      "collected_at": "2026-07-02"
    }
"""
from __future__ import annotations

import json
import re
import time
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import requests

USER_AGENT = "WasteVLM-corpus/0.1 (academic research; diecidue.andrea@gmail.com)"

# Corpus root on scratch (home has a 50 GB quota; see SETUP_LEONARDO.md).
CORPUS_ROOT = Path(
    "/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data/waste_corpus_web"
)


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
def http_get(
    url: str,
    *,
    params: Optional[dict] = None,
    accept: str = "text/html,application/json",
    extra_headers: Optional[Dict[str, str]] = None,
    timeout: int = 30,
    retries: int = 4,
    backoff: float = 2.0,
) -> requests.Response:
    """GET with a descriptive UA and exponential backoff.

    Wikipedia and EUR-Lex both require a real User-Agent; anonymous scraping is
    rate-limited or blocked. Raises on the final failure.
    """
    headers = {"User-Agent": USER_AGENT, "Accept": accept}
    if extra_headers:
        headers.update(extra_headers)
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            # 202/503 from EUR-Lex mean "generating, retry"; treat as soft failure.
            if r.status_code in (200,):
                return r
            if r.status_code in (202, 429, 500, 502, 503):
                last_exc = RuntimeError(f"HTTP {r.status_code} for {url}")
            else:
                r.raise_for_status()
                return r
        except requests.RequestException as exc:  # noqa: PERF203
            last_exc = exc
        sleep = backoff ** attempt
        time.sleep(sleep)
    raise RuntimeError(f"GET failed after {retries} tries: {url}") from last_exc


# --------------------------------------------------------------------------- #
# HTML -> text
# --------------------------------------------------------------------------- #
class _TextExtractor(HTMLParser):
    """Minimal HTML->text: drops script/style/head, keeps block breaks."""

    _SKIP = {"script", "style", "head", "noscript", "nav", "footer"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
              "article", "section", "table"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: List[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth > 0:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self._chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._chunks.append(data)

    def text(self) -> str:
        return "".join(self._chunks)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    return clean_text(parser.text())


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
_WS_RUN = re.compile(r"[ \t ]+")
_NL_RUN = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Collapse whitespace, normalise newlines, strip control chars."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if ch >= " " or ch == "\n")
    text = _WS_RUN.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _NL_RUN.sub("\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
# Document construction + JSONL IO
# --------------------------------------------------------------------------- #
def make_doc(
    *,
    source: str,
    doc_key: str,
    url: str,
    license: str,
    title: str,
    text: str,
    lang: str = "en",
) -> Optional[Dict]:
    """Build a schema document, or None if the text is too short to keep."""
    text = clean_text(text)
    if len(text) < 200:  # skip stubs / redirect pages / empty extracts
        return None
    return {
        "id": f"{source}:{doc_key}",
        "source": source,
        "url": url,
        "license": license,
        "title": title,
        "lang": lang,
        "text": text,
        "n_chars": len(text),
        "collected_at": date.today().isoformat(),
    }


def write_jsonl(path: Path, docs: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> Iterator[Dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
