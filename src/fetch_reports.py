"""Append curated public reports on illegal landfills / waste crime to the corpus.

Hand-curated, openly-accessible, authoritative sources only (EU / EEA / Europol /
ISPRA / open-access reviews) — no news or paywalled content. Handles both HTML
(bs4) and PDF (pypdf). Appends to data/waste_corpus/corpus.jsonl (dedup by url)
and writes reports_manifest.json with fetch status.

    python -m src.fetch_reports
Then re-tokenize:  python -m src.build_cpt_data --tokenizer <qwen-7b-instruct>
"""
from __future__ import annotations

import io
import json
import time
import urllib.request
from pathlib import Path

from src.build_waste_corpus import CORPUS, UA, _norm, fetch_html_text

MANIFEST = Path("/home/ids/diecidue/data/waste_corpus/reports_manifest.json")

# (name, url, kind, license)
REPORTS = [
    ("EC — Landfill waste (topic)",
     "https://environment.ec.europa.eu/topics/waste-and-recycling/landfill-waste_en",
     "html", "© European Union"),
    ("EEA — Diversion of waste from landfill",
     "https://www.eea.europa.eu/en/analysis/indicators/diversion-of-waste-from-landfill",
     "html", "© EEA"),
    ("EU Parliament — Mapping/remediation of illegal Italian toxic waste sites (E-000093/2021)",
     "https://www.europarl.europa.eu/doceo/document/E-9-2021-000093_EN.html",
     "html", "© European Union"),
    ("Europol — Waste and pollution crime",
     "https://www.europol.europa.eu/crime-areas/environmental-crime/waste-and-pollution-crime",
     "html", "© Europol"),
    ("Europol — Environmental Crime in the Age of Climate Change (threat assessment 2022)",
     "https://www.europol.europa.eu/cms/sites/default/files/documents/Environmental_Crime_in_the_Age_of_Climate_Change_threat_assessment_2022.pdf",
     "pdf", "© Europol"),
    ("PMC — GIS waste-risk indicator: health impact of landfills & uncontrolled dumping",
     "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7459911/",
     "html", "open access (CC)"),
    ("PMC — Satellite data in solid-waste landfill monitoring: review & case studies",
     "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10146526/",
     "html", "open access (CC)"),
    ("arXiv 2502.06607 — Illegal Waste Detection in Remote Sensing Images",
     "https://arxiv.org/pdf/2502.06607",
     "pdf", "arXiv (author license)"),
]


def fetch_pdf_text(name: str, url: str, license_: str) -> dict | None:
    try:
        from pypdf import PdfReader
        req = urllib.request.Request(url, headers=UA)
        raw = urllib.request.urlopen(req, timeout=60).read()
        reader = PdfReader(io.BytesIO(raw))
        text = _norm("\n".join((p.extract_text() or "") for p in reader.pages))
        if len(text) < 1000:
            print(f"  [warn] pdf '{name}': only {len(text)} chars (image-only?)")
            return None
        return {"source": "report", "title": name, "url": url,
                "license": license_, "text": text}
    except Exception as e:
        print(f"  [warn] pdf '{name}': {e}")
        return None


def main() -> int:
    done = {json.loads(l)["url"] for l in CORPUS.read_text().splitlines() if l.strip()}
    manifest, n_new = [], 0
    with CORPUS.open("a") as f:
        for name, url, kind, lic in REPORTS:
            if url in done:
                manifest.append({"name": name, "url": url, "status": "already-have"})
                continue
            doc = fetch_pdf_text(name, url, lic) if kind == "pdf" else fetch_html_text(name, url, lic)
            if doc:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n"); f.flush()
                n_new += 1
            manifest.append({"name": name, "url": url, "kind": kind,
                             "status": "ok" if doc else "failed",
                             "chars": len(doc["text"]) if doc else 0})
            print(f"  {'ok ' if doc else 'FAIL'} [{len(doc['text']) if doc else 0:>7} ch] {name}")
            time.sleep(1.0)

    MANIFEST.write_text(json.dumps(manifest, indent=2))
    docs = [json.loads(l) for l in CORPUS.read_text().splitlines() if l.strip()]
    tot = sum(len(d["text"]) for d in docs)
    bysrc: dict = {}
    for d in docs:
        bysrc[d["source"]] = bysrc.get(d["source"], 0) + len(d["text"])
    print(f"\n[corpus] {len(docs)} docs, +{n_new} reports this run")
    print(f"[corpus] {tot:,} chars (~{tot//4:,} tok); by source: {bysrc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
