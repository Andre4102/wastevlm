"""Collect public law-enforcement / agency waste-crime reports into JSONL.

Reads the curated URL list in ``sources.py`` (PDF or HTML). PDFs are parsed with
pypdf. LICENSING IS NOT ASSUMED — each entry keeps its declared license marker
(default "verify"); review provenance before training on it.
Output: ``<out>/lea.jsonl``.

Run in an env with pypdf + requests on a login node (e.g. after
``pip install pypdf`` into the tokenize env)::

    python -m src.corpus.collect_lea
"""
from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

from .common import CORPUS_ROOT, html_to_text, http_get, make_doc, write_jsonl
from .sources import LEA_SOURCES


def _pdf_to_text(raw: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # gated: pypdf not in every env
        raise RuntimeError(
            "pypdf required for PDF sources — `pip install pypdf`"
        ) from exc
    reader = PdfReader(io.BytesIO(raw))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def _fetch_one(doc_key: str, url: str, kind: str, license: str, title: str):
    resp = http_get(url, accept="application/pdf,text/html")
    if kind == "pdf":
        text = _pdf_to_text(resp.content)
    else:
        text = html_to_text(resp.text)
    return make_doc(
        source="lea", doc_key=doc_key, url=url, license=license,
        title=title, text=text,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CORPUS_ROOT / "lea.jsonl")
    args = ap.parse_args()

    if not LEA_SOURCES:
        print("LEA_SOURCES is empty — add verified public URLs in sources.py "
              "before running. Skipping.")
        write_jsonl(args.out, [])
        return 0

    docs = []
    for doc_key, url, kind, license, title in LEA_SOURCES:
        print(f"  {doc_key}  [{license}]  {title}", flush=True)
        try:
            doc = _fetch_one(doc_key, url, kind, license, title)
        except Exception as exc:
            print(f"    ! {exc}", flush=True)
            continue
        if doc:
            docs.append(doc)
            print(f"    kept {doc['n_chars']/1e3:.0f}K chars", flush=True)
        time.sleep(1.0)

    n = write_jsonl(args.out, docs)
    total_chars = sum(d["n_chars"] for d in docs)
    print(f"Wrote {n}/{len(LEA_SOURCES)} docs "
          f"({total_chars/1e6:.1f}M chars) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
