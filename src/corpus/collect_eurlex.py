"""Collect EU waste-law texts from EUR-Lex into JSONL.

For each CELEX id in ``sources.py`` fetches the English XHTML from the
Publications Office **Cellar** REST API (content negotiation), strips markup,
and writes one document. Output: ``<out>/eurlex.jsonl``.

Note: the EUR-Lex web front-end (legal-content/…) returns HTTP 202 to
non-browser clients (bot mitigation), so we use the official machine endpoint
``http://publications.europa.eu/resource/celex/<CELEX>`` with
``Accept: application/xhtml+xml`` instead.

Reuse of EUR-Lex content is authorised under Commission Decision 2011/833/EU
with source acknowledgement (carried in each doc's ``url``/``license``).

Run in the ``base`` env on a login node::

    python -m src.corpus.collect_eurlex
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .common import CORPUS_ROOT, html_to_text, http_get, make_doc, write_jsonl
from .sources import EURLEX_CELEX

# Official machine-access endpoint (Cellar), content-negotiated.
CELLAR = "http://publications.europa.eu/resource/celex/{celex}"
# Human-facing permalink recorded for attribution.
PUBLIC_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"
# Newer acts expose XHTML; older/repealed ones only text/html — try in order.
_ACCEPTS = ["application/xhtml+xml", "text/html"]
_HEADERS = {"Accept-Language": "eng"}
LICENSE = "EUR-Lex-reuse-2011/833/EU"


def _fetch_one(celex: str, title: str):
    for accept in _ACCEPTS:
        try:
            html = http_get(
                CELLAR.format(celex=celex), accept=accept, extra_headers=_HEADERS,
            ).text
        except Exception:
            continue
        text = html_to_text(html)
        doc = make_doc(
            source="eurlex", doc_key=celex, url=PUBLIC_URL.format(celex=celex),
            license=LICENSE, title=title, text=text,
        )
        if doc:
            return doc
    print(f"  ! {celex}: no text via {_ACCEPTS}", flush=True)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CORPUS_ROOT / "eurlex.jsonl")
    args = ap.parse_args()

    docs = []
    for celex, title in EURLEX_CELEX:
        print(f"  {celex}  {title}", flush=True)
        doc = _fetch_one(celex, title)
        if doc:
            docs.append(doc)
            print(f"    kept {doc['n_chars']/1e3:.0f}K chars", flush=True)
        else:
            print("    ! no text extracted", flush=True)
        time.sleep(1.0)  # be polite to EUR-Lex

    n = write_jsonl(args.out, docs)
    total_chars = sum(d["n_chars"] for d in docs)
    print(f"Wrote {n}/{len(EURLEX_CELEX)} docs "
          f"({total_chars/1e6:.1f}M chars) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
