# LLM Ops SDK

A learned-router LLM optimization SDK: fine-tuned complexity classifier + semantic cache +
context compressor + guardrails, wrapped in a FastAPI-based SDK.

> Built an LLM Ops SDK that reduces LLM API cost by routing queries through a fine-tuned
> DistilBERT complexity classifier (67% accuracy vs. a 57% rule-based baseline), a semantic
> cache (75% hit rate at a tuned 0.80 similarity threshold), and query-aware context
> compression — measured end-to-end with real API calls at a 65% cost reduction and 2.7x
> lower latency versus a naive always-use-the-strongest-model baseline.

## Status

**All 5 phases complete** — repo scaffold, rule-based baseline router, data labeling/fine-tuning pipeline, semantic cache, context compressor, guardrails, and the wired-together FastAPI service + benchmark, all run end-to-end on real data with real API calls.

- `src/llmops_sdk/router/baseline.py` — rule-based simple/moderate/complex classifier
  (query length, question-word patterns, reasoning/multi-step keyword list). This is the
  comparison baseline the fine-tuned DistilBERT router must beat.
- `scripts/run_baseline_eval.py` — CLI to evaluate the baseline against a labeled
  `.jsonl`/`.csv` dataset (`query`, `true_tier` columns), reporting accuracy and per-class F1.
- `scripts/generate_labels.py` — samples from Dolly-15k and labels them simple/moderate/complex
  using Gemini as a labeling assistant against a fixed rubric (few-shot prompted). Has a
  `--dry-run` mode and a `--resume` mode (writes incrementally so a partial run/quota limit
  doesn't lose progress). Requires `GEMINI_API_KEY` (env var or `.env`, see `.env.example`).
- `src/llmops_sdk/router/train.py` — fine-tunes `distilbert-base-uncased` as a 3-class
  classifier via Hugging Face `Trainer`, with a stratified 80/10/10 train/val/test split
  (test set written to `data/labeled/test.jsonl` so it can be re-scored with
  `run_baseline_eval.py` for a fair comparison), logging results to
  `reports/training_log.md`.
- `src/llmops_sdk/cache/semantic_cache.py` — `SemanticCache`: in-memory `all-MiniLM-L6-v2`
  embeddings + cosine similarity lookup, no vector DB needed at this scale.
- `scripts/sweep_cache_threshold.py` — sweeps similarity thresholds against
  `data/cache_eval_pairs.jsonl` (hand-labeled near-duplicate/distinct query pairs),
  reporting hit rate vs. false-positive rate and recommending a threshold under an
  accepted FP budget (default 5%) — see `reports/cache_threshold_sweep.md`.
- `src/llmops_sdk/compressor/context_compressor.py` — `ContextCompressor`: per-sentence
  embedding similarity to the query for importance scoring, greedy top-scoring-first
  selection under a token budget, re-ordered back to original document order.
- `scripts/eval_compression_quality.py` — measures fact retention (expected fact-strings
  surviving into the compressed context) at a couple of compression ratios on
  `data/compression_eval.jsonl` — see `reports/compression_quality.md`.
- `src/llmops_sdk/guardrails/input_checks.py` — regex PII detection (email/phone/SSN) +
  curated prompt-injection phrase list.
- `src/llmops_sdk/guardrails/output_checks.py` — Pydantic-based JSON schema validation +
  a word-overlap faithfulness heuristic for RAG-style responses (described as a
  heuristic, not a hallucination-detection system).
- `src/llmops_sdk/router/model.py` — `FineTunedRouter`: loads the saved DistilBERT
  model for real-time single-query inference (`predict(query) -> (tier, confidence)`).
- `src/llmops_sdk/api/main.py` — FastAPI `POST /query`: cache check → input guardrails
  (blocks PII/injection before reaching the model or cache) → router → compressor (if
  context given) → model call (mocked — no real API spend per request) → faithfulness
  check → cache write.
- `src/llmops_sdk/api/pricing.py` — router-tier → model mapping and per-model USD/1M-token
  pricing (published rates, not guessed).
- `scripts/run_full_benchmark.py` — naive baseline (always the strongest available
  model, no cache/compression) vs. the full pipeline, on a 20-query realistic mix, using
  **real Gemini API calls** (not mocked) so cost/latency are genuinely measured — see
  `reports/benchmark_results.md`.
- `tests/test_router.py`, `tests/test_cache.py`, `tests/test_compressor.py`,
  `tests/test_guardrails.py`, `tests/test_router_model.py`, `tests/test_api.py` — edge
  case and integration coverage for all components, 49 tests total.

**Real results:**
- Router (486-example labeled subset, scaled down from the original ~2,500 for a faster
  learning-focused pass): fine-tuned **67.4% accuracy / 0.46 macro-F1** vs. rule-based
  baseline **57.1% accuracy / 0.40 macro-F1** on the same held-out test set — see
  `reports/training_log.md` for the full breakdown, confusion matrix, and an honest note
  on the "complex" class being underrepresented (only ~10% of the data) and not yet
  reliably learned by either model.
- Semantic cache: threshold **0.80** gives **75% hit rate at 5% false positives** on the
  40-pair eval set — a stricter zero-FP threshold (0.92) was available but collapses hit
  rate to 35%, not a useful cache in practice. See `reports/cache_threshold_sweep.md` for
  the full table and reasoning.
- Context compressor: **82% fact retention at 50% target compression**, dropping to
  **64% at 33% target compression** — a real quality-vs-compression tradeoff curve, not
  a single cherry-picked number. See `reports/compression_quality.md`.
- Full pipeline benchmark (real Gemini calls, 20-query mix): **65% cost reduction**
  ($0.0227 naive vs. $0.0080 full pipeline), latency **6121ms → 2271ms**, cache hit rate
  **25%** (all 5 planted duplicates correctly caught). Fact retention on the
  context-bearing queries dropped from 82% (naive, uncompressed) to 55% (full pipeline,
  compressed) — a real tradeoff, reported honestly rather than omitted. The originally
  planned "strong" tier (`gemini-3.1-pro`) turned out to have **zero free-tier quota**
  on this API key, so the benchmark's strong/naive tier uses `gemini-3-flash-preview`
  instead — see `reports/benchmark_results.md` for the full reasoning.

## Setup

```
python -m venv .venv
.venv/Scripts/activate      # Windows
pip install -e ".[dev]"
pytest
```

## Usage

```
# Baseline eval
python scripts/run_baseline_eval.py data/labeled/test.jsonl --output reports/baseline_eval.md

# Data labeling (requires GEMINI_API_KEY)
python scripts/generate_labels.py --dry-run
python scripts/generate_labels.py --n 2500 --output data/labeled/dolly_labeled.jsonl

# Fine-tune the router
python -m llmops_sdk.router.train --input data/labeled/dolly_labeled.jsonl

# Semantic cache threshold sweep
python scripts/sweep_cache_threshold.py

# Context compression quality eval
python scripts/eval_compression_quality.py

# Run the API locally
uvicorn llmops_sdk.api.main:app --reload

# Full pipeline benchmark (requires GEMINI_API_KEY, makes real API calls)
python scripts/run_full_benchmark.py
```
