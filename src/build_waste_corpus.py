"""Build a curated, licensed waste-domain text corpus for LLM domain adaptation.

Sources (authoritative, openly licensed — NOT a broad web scrape):
  - Wikipedia (CC BY-SA) waste/recycling/landfill article tree, via the action
    API plain-text extract endpoint (clean prose, no markup).
  - Regulatory / agency pages (EU List of Waste, EEA, EPA) via requests + bs4.
  - Our own teacher captions (full.jsonl) — already a waste text corpus.

Output:
  data/waste_corpus/corpus.jsonl   one line per document: {source, title, url, license, text, n_chars}
  data/waste_corpus/sources.json   provenance manifest with fetch status

Run on a node with internet (the login shell has it):
    python -m src.build_waste_corpus
Re-run is idempotent: documents already in corpus.jsonl (keyed by url/title) are skipped.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

OUT_DIR = Path("/home/ids/diecidue/data/waste_corpus")
CORPUS = OUT_DIR / "corpus.jsonl"
MANIFEST = OUT_DIR / "sources.json"
UA = {"User-Agent": "waste-vlm-research/1.0 (academic; contact diecidue)"}

# Curated Wikipedia titles — waste / recycling / landfill / illegal-dumping tree.
WIKI_TITLES = [
    "Waste", "Waste management", "Waste hierarchy", "List of waste types",
    "Municipal solid waste", "Hazardous waste", "Industrial waste",
    "Construction and demolition waste", "Demolition", "Inert waste",
    "Biodegradable waste", "Recycling", "Plastic recycling", "Metal recycling",
    "Glass recycling", "Paper recycling", "Tire recycling", "Landfill",
    "Leachate", "Illegal dumping", "Fly-tipping", "Litter", "Open dump",
    "Electronic waste", "Plastic pollution", "Marine debris", "Incineration",
    "Waste-to-energy", "Composting", "Sewage sludge", "Waste collection",
    "Waste sorting", "Scrap", "Tire", "Asbestos", "Asbestos abatement",
    "Circular economy", "Bulky waste", "Waste container", "Intermodal container",
    "Pallet", "Waste picker", "Waste characterisation",
    "List of Waste (European Union)",
]

# Regulatory / agency pages (HTML → text via bs4).
DOC_URLS = [
    ("EU List of Waste (Decision 2014/955/EU)",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex:32014D0955",
     "EUR-Lex / © European Union"),
    ("EU Waste Framework Directive 2008/98/EC",
     "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=celex:32008L0098",
     "EUR-Lex / © European Union"),
]


def _norm(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def fetch_wikipedia(title: str) -> dict | None:
    q = urllib.parse.urlencode({
        "action": "query", "format": "json", "prop": "extracts",
        "explaintext": "1", "redirects": "1", "titles": title,
    })
    url = f"https://en.wikipedia.org/w/api.php?{q}"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=UA)
            data = json.load(urllib.request.urlopen(req, timeout=20))
            pages = data["query"]["pages"]
            page = next(iter(pages.values()))
            if "extract" not in page or not page["extract"].strip():
                return None
            canonical = page.get("title", title)
            return {
                "source": "wikipedia", "title": canonical,
                "url": f"https://en.wikipedia.org/wiki/{urllib.parse.quote(canonical.replace(' ', '_'))}",
                "license": "CC BY-SA 4.0", "text": _norm(page["extract"]),
            }
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 4:
                time.sleep(3 * (attempt + 1))  # backoff on rate limit
                continue
            print(f"  [warn] wiki '{title}': {e}")
            return None
        except Exception as e:
            print(f"  [warn] wiki '{title}': {e}")
            return None
    return None


def fetch_html_text(title: str, url: str, license_: str) -> dict | None:
    try:
        from bs4 import BeautifulSoup
        req = urllib.request.Request(url, headers=UA)
        html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "ignore")
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = _norm(soup.get_text("\n"))
        if len(text) < 500:
            return None
        return {"source": "regulatory", "title": title, "url": url,
                "license": license_, "text": text}
    except Exception as e:
        print(f"  [warn] doc '{title}': {e}")
        return None


def captions_as_docs() -> list[dict]:
    """Fold the teacher captions into the corpus as one concatenated doc per dataset."""
    cap = Path("/home/ids/diecidue/data/captions/full.jsonl")
    if not cap.exists():
        return []
    buckets: dict[str, list[str]] = {}
    for line in cap.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        buckets.setdefault(r["dataset"], []).append(r["caption"].strip())
    docs = []
    for ds, caps in buckets.items():
        docs.append({"source": "captions", "title": f"teacher_captions_{ds}",
                     "url": f"local:full.jsonl#{ds}", "license": "project-internal",
                     "text": "\n\n".join(caps)})
    return docs


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done_keys = set()
    if CORPUS.exists():
        for line in CORPUS.read_text().splitlines():
            if line.strip():
                done_keys.add(json.loads(line)["url"])

    manifest = []
    n_new = 0
    with CORPUS.open("a") as f:
        for title in WIKI_TITLES:
            doc = fetch_wikipedia(title)
            status = "ok" if doc else "failed/empty"
            if doc and doc["url"] not in done_keys:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n"); f.flush()
                n_new += 1
            manifest.append({"requested": title, "source": "wikipedia", "status": status,
                             "chars": len(doc["text"]) if doc else 0})
            time.sleep(1.5)
        for title, url, lic in DOC_URLS:
            doc = fetch_html_text(title, url, lic)
            status = "ok" if doc else "failed/short"
            if doc and doc["url"] not in done_keys:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n"); f.flush()
                n_new += 1
            manifest.append({"requested": title, "source": "regulatory", "status": status,
                             "chars": len(doc["text"]) if doc else 0})
            time.sleep(0.5)
        for doc in captions_as_docs():
            if doc["url"] not in done_keys:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n"); f.flush()
                n_new += 1
            manifest.append({"requested": doc["title"], "source": "captions",
                             "status": "ok", "chars": len(doc["text"])})

    MANIFEST.write_text(json.dumps(manifest, indent=2))
    # corpus stats
    total_chars = ok = 0
    by_source: dict[str, int] = {}
    for line in CORPUS.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line); ok += 1; total_chars += len(r["text"])
        by_source[r["source"]] = by_source.get(r["source"], 0) + len(r["text"])
    print(f"\n[corpus] {ok} docs, +{n_new} new this run")
    print(f"[corpus] {total_chars:,} chars  (~{total_chars//4:,} tokens est.)")
    print(f"[corpus] by source (chars): {by_source}")
    print(f"[manifest] {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
