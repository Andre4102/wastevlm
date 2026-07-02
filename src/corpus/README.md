# Waste-domain web corpus → pruning-ready domain shards

Collect a web-scale waste-domain text corpus (Wikipedia + EU law + public
enforcement reports), merge/dedup it, and pack it into a `waste/` domain of
Arrow shards that drops directly into the pruning repo's domain-sampled
(Sheared-LLaMA-style DBL) continued-pretraining + structural-pruning loop.

This supersedes the small `src/build_waste_corpus.py` (a fixed ~40-title list on
the retired IDS path — the 49-doc/955K-token corpus that `project.md` records as
too small for CPT). Same idea, scaled: category recursion, full CELEX law texts,
enforcement docs, dedup, and domain-shard output.

## Why this exists

The pruning repo (`../pruning`) already implements domain-weighted CPT with a PID
controller that balances a `waste` domain against general text to avoid
catastrophic forgetting, and runs learned-mask structural pruning **in the same
loop**. Point it at these shards to get the case study: *domain-specialize +
structurally prune, and measure how much waste knowledge the pruned model keeps.*

Base model is **LLaMA family** (reuses the pruning code's `LlamaMaskAdapter`
as-is). The raw corpus is model-agnostic JSONL, so it can be re-tokenised for
Qwen later if this feeds back into the VLM.

## Pipeline

```
collect_wikipedia.py ─┐
collect_eurlex.py     ├─→ *.jsonl ─→ build_corpus.py ─→ corpus.jsonl ─→ tokenize_domain.py ─→ shards/waste/*.arrow
collect_lea.py        ─┘   (per-source)   (merge+dedup)   (+stats.json)   (pruning DomainPacker)
```

All paths default to
`/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data/waste_corpus_web/`
(scratch — home has a 50 GB quota).

## Runbook

**1. Collect** — needs internet, so run on a Leonardo **login node** in `base`
(only `requests` required):

```bash
cd /leonardo/home/userexternal/adiecidu/scripts/wastevlm
python -m src.corpus.collect_eurlex
python -m src.corpus.collect_wikipedia --depth 2 --max-pages 1000
python -m src.corpus.collect_lea            # only after adding verified URLs
```

**2. Merge + dedup** (`base`, any node):

```bash
python -m src.corpus.build_corpus --min-chars 400
cat .../waste_corpus_web/stats.json
```

**3. Tokenize into the `waste/` domain** — needs `datasets` (`gausdino` env) and
the base-model tokenizer:

```bash
conda activate gausdino
python -m src.corpus.tokenize_domain \
  --tokenizer <path-or-HF-id of the LLaMA base you'll prune> \
  --pruning-repo /leonardo/home/userexternal/adiecidu/scripts/pruning
```

**4. Build the general domains into the same `--out-root`** with the pruning
repo's `tokenize_slimpajama.py`, then run `pruning_llama3_pretrain.py` with the
`waste` domain added to the mix.

## Sources & licensing

| Source | How | License | Notes |
|---|---|---|---|
| Wikipedia | MediaWiki API plain-text extracts, category recursion | CC-BY-SA-4.0 | Attribution via per-doc `url`. Clean prose, no markup. |
| EU law | Publications Office **Cellar** API (`resource/celex/<CELEX>`, `Accept: application/xhtml+xml`) | Reuse authorised, Decision 2011/833/EU | The `legal-content/…` web UI returns HTTP 202 to bots — Cellar is the machine route. Curated CELEX list in `sources.py`. |
| LEA / agency | Curated public PDF/HTML URLs | **VARIES — `verify`** | `LEA_SOURCES` starts empty. Add only URLs whose reuse terms you've checked; `build_corpus.py` warns if any `verify` docs remain. PDFs need `pip install pypdf`. |

## Scale note

The pruning `total_tokens` default (50B) is a placeholder. This corpus is far
smaller; for a *specialization + compression demo* the honest budget is the
domain tokens (see `stats.json`) mixed with a general-domain sample — the DBL
controller upweights the small `waste` domain rather than training on 50B tokens
of it.

## Not done yet

- Domain **eval benchmark** (waste-regulation QA) to quantify retention — the
  metric that makes the pruning result meaningful. Build from the EUR-Lex corpus;
  doubles as the VLM's deferred "regulatory-reasoning" eval.
- `LEA_SOURCES` is empty — needs verified public URLs.
- Optionally seed `collect_wikipedia` with the curated titles from the old
  `src/build_waste_corpus.py` to guarantee core-article coverage.
