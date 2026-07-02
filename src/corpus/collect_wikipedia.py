"""Collect waste-related Wikipedia articles into JSONL.

Walks the seed categories in ``sources.py`` (recursing into subcategories up to
``--depth``), then fetches a plain-text extract for each unique article via the
MediaWiki API. Output: ``<out>/wikipedia.jsonl``.

Run in the ``base`` env on a Leonardo *login* node (needs internet)::

    python -m src.corpus.collect_wikipedia --depth 2 --max-pages 800
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .common import CORPUS_ROOT, http_get, make_doc, write_jsonl
from .sources import WIKI_SEED_CATEGORIES

API = "https://en.wikipedia.org/w/api.php"


def _category_members(title: str) -> Tuple[List[str], List[str]]:
    """Return (article_titles, subcategory_titles) for one category."""
    pages: List[str] = []
    subcats: List[str] = []
    cmcontinue = None
    while True:
        params = {
            "action": "query", "format": "json", "list": "categorymembers",
            "cmtitle": title, "cmlimit": "500",
            "cmtype": "page|subcat",
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = http_get(API, params=params, accept="application/json").json()
        for m in data.get("query", {}).get("categorymembers", []):
            if m["ns"] == 14:  # Category namespace
                subcats.append(m["title"])
            elif m["ns"] == 0:  # Article namespace
                pages.append(m["title"])
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
        time.sleep(0.1)
    return pages, subcats


def gather_titles(seeds: List[str], max_depth: int) -> Set[str]:
    """BFS over categories; collect article titles up to max_depth."""
    seen_cat: Set[str] = set()
    titles: Set[str] = set()
    frontier: List[Tuple[str, int]] = [(c, 0) for c in seeds]
    while frontier:
        cat, depth = frontier.pop(0)
        if cat in seen_cat:
            continue
        seen_cat.add(cat)
        pages, subcats = _category_members(cat)
        titles.update(pages)
        print(f"  [{cat}] +{len(pages)} pages, {len(subcats)} subcats "
              f"(depth {depth}, total {len(titles)})", flush=True)
        if depth < max_depth:
            frontier.extend((s, depth + 1) for s in subcats)
    return titles


def _fetch_extract(title: str) -> Dict | None:
    params = {
        "action": "query", "format": "json", "prop": "extracts|info",
        "explaintext": "1", "redirects": "1", "inprop": "url",
        "titles": title,
    }
    data = http_get(API, params=params, accept="application/json").json()
    for _, page in data.get("query", {}).get("pages", {}).items():
        if "missing" in page:
            return None
        return make_doc(
            source="wikipedia",
            doc_key=page["title"].replace(" ", "_"),
            url=page.get("fullurl", f"https://en.wikipedia.org/wiki/{title}"),
            license="CC-BY-SA-4.0",
            title=page["title"],
            text=page.get("extract", ""),
        )
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=2, help="category recursion depth")
    ap.add_argument("--max-pages", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=CORPUS_ROOT / "wikipedia.jsonl")
    args = ap.parse_args()

    print(f"Gathering titles from {len(WIKI_SEED_CATEGORIES)} seed categories …")
    titles = sorted(gather_titles(WIKI_SEED_CATEGORIES, args.depth))
    if len(titles) > args.max_pages:
        titles = titles[: args.max_pages]
    print(f"Fetching extracts for {len(titles)} articles …", flush=True)

    docs = []
    for i, t in enumerate(titles):
        try:
            d = _fetch_extract(t)
        except Exception as exc:  # keep going on individual failures
            print(f"  ! {t}: {exc}", flush=True)
            continue
        if d:
            docs.append(d)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(titles)} kept={len(docs)}", flush=True)
        time.sleep(0.1)

    n = write_jsonl(args.out, docs)
    total_chars = sum(d["n_chars"] for d in docs)
    print(f"Wrote {n} docs ({total_chars/1e6:.1f}M chars) → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
