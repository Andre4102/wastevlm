# Waste knowledge-retention eval

A text multiple-choice benchmark that measures how much waste-domain knowledge a
causal LM holds — the **go/no-go gate** before spending compute on visual
training. Compare dense base vs CPT vs CPT+pruned: if pruning erases too much
learned regulatory knowledge, dial back sparsity / waste-weight and re-CPT.

Scoring matches the pruning repo's convention (`acc_norm` = byte-length-
normalised continuation log-prob, argmax over choices), so numbers are directly
comparable to its zero-shot MC harness. Works on **base/CPT models** — no
instruction-following required.

## Benchmark: EU List of Waste (v1)

`build_low_qa.py` parses the European List of Waste (Commission Decision
2014/955/EU, from `eurlex.jsonl`) into ~825 code↔description entries and builds
two 4-choice question types, distractors drawn from the **same 2-digit chapter**
(so the model must know the specific entry, not just the topic):

- `code2desc` — "code 01 01 01 refers to:" → pick the description
- `desc2code` — "the code for '<description>' is:" → pick the code (unique
  descriptions only)

766 questions at `--max-per-type 400`. **Random baseline = 25%.**

This is a *retention / memorisation* probe: the List of Waste is in the CPT
corpus, so it measures survival of learned facts through pruning.

## Three knowledge axes (each reuses one scorer)

| Axis | File / builder | Probe | Scorer / metric | Leakage |
|---|---|---|---|---|
| **Memorisation** | `build_low_qa.py` → `low_qa.jsonl` | code↔desc MC (verbatim) | `mc_score` / acc_norm | in-train (by design) |
| **Concept** | `build_concept_qa.py` → `concept_qa.jsonl` | **synonym-paraphrased** desc→code MC | `mc_score` / acc_norm | wording unseen |
| **Appearance** | `data/appearance.jsonl` | "what waste looks like" prose | `ppl_eval` / PPL | authored, held-out |
| **Generalisation** | `split_corpus.py` → `corpus_eval.jsonl` | held-out waste prose | `ppl_eval` / PPL | docs held out of CPT |

**Concept** (`concept_qa.jsonl`, 55 Q) rewrites the official LoW descriptions with
plain-language synonyms and strips the discriminative keywords, so accuracy
reflects *concept* knowledge rather than string memorisation. Paraphrases are
authored in `data/concept_paraphrases.json` (grounded on real code↔desc pairs).

**Appearance** (`data/appearance.jsonl`, 39 waste types) describes what each waste
*looks like* (materials, colours, forms, outdoor/aerial context) — the visual
world-knowledge the text backbone lends the VLM. Lower PPL = the model finds
correct appearance descriptions more likely.

    python -m src.eval.build_concept_qa
    python -m src.eval.mc_score  --bench .../concept_qa.jsonl --models <base> <cpt> <pruned>
    python -m src.eval.ppl_eval  --eval  src/eval/data/appearance.jsonl --models <base> <cpt> <pruned>

⚠ Under **prune+FT**, FT can re-memorise `low_qa` if it re-sees the corpus — so
put the leakage-controlled axes (concept, appearance, held-out) in the headline
and treat `low_qa` as a capacity/sanity number. The authored paraphrases/
appearance text are subscription-generated (project's no-paid-API constraint).

## Run

Build (no model; `base` env, any node):

```bash
python -m src.eval.build_low_qa --max-per-type 400
```

Score / compare (needs torch+transformers — `gausdino` env; a GPU node for 8B):

```bash
BENCH=/leonardo_scratch/large/userexternal/adiecidu/waste_vlm/data/waste_eval/low_qa.jsonl
python -m src.eval.mc_score --bench $BENCH \
  --models meta-llama/Llama-3.1-8B <waste-llm-cpt> <waste-llm-pruned> \
  --out $BENCH.results.json
```

Prints per-type `acc_norm`/`acc` for each model and an `acc_norm` comparison
table. Expected shape: dense base ≈ chance on codes; CPT lifts it; pruned should
retain most of the CPT gain.

## Schema (one JSON object per line)

```json
{"id": "...", "type": "code2desc|desc2code",
 "prompt": "...", "choices": ["...","...","...","..."],
 "gold": 2, "source": "eurlex:32014D0955"}
```
