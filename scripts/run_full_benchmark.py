"""CLI: compare the full SDK pipeline (cache + guardrails + router + compressor +
tier-routed model calls) against a naive baseline (always the strongest model, no
cache, no compression) on a small (~20 query) realistic mix — using REAL Gemini API
calls so cost/latency are genuinely measured, not estimated. Cost is computed from
the actual token usage each call returns, times published per-model pricing.

Requires GEMINI_API_KEY (env var or .env, see .env.example).

Usage:
    python scripts/run_full_benchmark.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from llmops_sdk.api.pricing import NAIVE_MODEL, TIER_TO_MODEL, estimate_cost  # noqa: E402
from llmops_sdk.cache.semantic_cache import SemanticCache  # noqa: E402
from llmops_sdk.compressor.context_compressor import ContextCompressor, count_tokens  # noqa: E402
from llmops_sdk.router.baseline import BaselineRouter  # noqa: E402
from llmops_sdk.router.model import DEFAULT_MODEL_DIR, FineTunedRouter  # noqa: E402

# 10 unique queries spanning simple/moderate/complex, plus 5 duplicates of the first
# five (placed at the end) to exercise the cache in the full-pipeline run — the naive
# baseline has no cache, so it re-calls the model for every duplicate too.
PLAIN_QUERIES = [
    "What is the capital of Canada?",
    "Who invented the telephone?",
    "How does a refrigerator keep food cold?",
    "Explain the difference between HTTP and HTTPS.",
    "Write a Python function that checks if a number is prime.",
    "Design a caching strategy for a high-traffic API.",
    "What is the boiling point of ethanol?",
    "Compare the pros and cons of renewable energy versus fossil fuels.",
    "Debug this: a recursive function that causes a stack overflow on large inputs.",
    "What year did the Berlin Wall fall?",
]
DUPLICATE_QUERIES = [
    "What is the capital of Canada?",
    "Who invented the telephone?",
    "What is the boiling point of ethanol?",
    "What year did the Berlin Wall fall?",
    "How does a refrigerator keep food cold?",
]


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_query_stream() -> list[dict]:
    stream = [{"query": q, "context": None, "expected_facts": None} for q in PLAIN_QUERIES]
    with Path("data/compression_eval.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                stream.append(
                    {"query": row["query"], "context": row["context"], "expected_facts": row["expected_facts"]}
                )
    stream += [{"query": q, "context": None, "expected_facts": None} for q in DUPLICATE_QUERIES]
    return stream


def real_llm_call(client, model: str, query: str, context: str | None, max_retries: int = 3):
    prompt = query if not context else f"Context:\n{context}\n\nQuestion: {query}"
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt)
            usage = response.usage_metadata
            return response.text or "", usage.prompt_token_count or 0, usage.candidates_token_count or 0
        except Exception as e:  # noqa: BLE001 - transient API errors, retry
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(5 * (2**attempt))
    raise RuntimeError(f"Failed after {max_retries} attempts") from last_error


def fact_retention(text: str, expected_facts: list[str] | None) -> tuple[int, int] | None:
    if not expected_facts:
        return None
    lowered = text.lower()
    hits = sum(1 for fact in expected_facts if fact.lower() in lowered)
    return hits, len(expected_facts)


def run_naive(client, stream: list[dict], sleep: float) -> list[dict]:
    results = []
    for item in stream:
        t0 = time.perf_counter()
        try:
            text, in_tok, out_tok = real_llm_call(client, NAIVE_MODEL, item["query"], item["context"])
        except RuntimeError as e:
            print(f"WARNING: naive call failed for {item['query']!r}: {e}", file=sys.stderr)
            continue
        results.append(
            {
                "query": item["query"],
                "cost": estimate_cost(NAIVE_MODEL, in_tok, out_tok),
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "cache_hit": False,
                "fact_hits": fact_retention(text, item["expected_facts"]),
            }
        )
        if sleep:
            time.sleep(sleep)
    return results


def run_full_pipeline(client, router, compressor, stream: list[dict], sleep: float) -> list[dict]:
    cache = SemanticCache()
    results = []
    for item in stream:
        t0 = time.perf_counter()
        cached_response, _ = cache.get(item["query"])
        if cached_response is not None:
            results.append(
                {
                    "query": item["query"],
                    "cost": 0.0,
                    "latency_ms": (time.perf_counter() - t0) * 1000,
                    "cache_hit": True,
                    "fact_hits": fact_retention(cached_response, item["expected_facts"]),
                }
            )
            continue

        tier, _ = router.predict(item["query"])
        model = TIER_TO_MODEL[tier]

        context_used = item["context"]
        if item["context"]:
            budget = max(1, round(count_tokens(item["context"]) * 0.5))
            context_used = compressor.compress(item["query"], item["context"], token_budget=budget).compressed_text

        try:
            text, in_tok, out_tok = real_llm_call(client, model, item["query"], context_used)
        except RuntimeError as e:
            print(f"WARNING: pipeline call failed for {item['query']!r}: {e}", file=sys.stderr)
            continue
        cache.set(item["query"], text)
        results.append(
            {
                "query": item["query"],
                "cost": estimate_cost(model, in_tok, out_tok),
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "cache_hit": False,
                "fact_hits": fact_retention(text, item["expected_facts"]),
            }
        )
        if sleep:
            time.sleep(sleep)
    return results


def summarize(results: list[dict]) -> dict:
    fact_results = [r["fact_hits"] for r in results if r["fact_hits"] is not None]
    total_facts = sum(t for _, t in fact_results)
    hit_facts = sum(h for h, _ in fact_results)
    return {
        "n": len(results),
        "total_cost": sum(r["cost"] for r in results),
        "avg_latency_ms": sum(r["latency_ms"] for r in results) / len(results) if results else 0.0,
        "cache_hit_rate": sum(1 for r in results if r["cache_hit"]) / len(results) if results else 0.0,
        "fact_retention": hit_facts / total_facts if total_facts else None,
    }


def main() -> None:
    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set (checked environment and .env).", file=sys.stderr)
        sys.exit(1)

    from google import genai

    client = genai.Client()
    stream = build_query_stream()
    router = FineTunedRouter() if DEFAULT_MODEL_DIR.exists() else BaselineRouter()
    compressor = ContextCompressor()
    sleep = 0.5

    print(f"Running naive baseline ({len(stream)} queries, always {NAIVE_MODEL}, no cache/compression)...")
    naive_results = run_naive(client, stream, sleep)

    print(f"Running full pipeline ({len(stream)} queries, cache + router + compressor)...")
    full_results = run_full_pipeline(client, router, compressor, stream, sleep)

    naive = summarize(naive_results)
    full = summarize(full_results)

    def fmt_retention(s: dict) -> str:
        return f"{s['fact_retention']:.0%}" if s["fact_retention"] is not None else "n/a"

    lines = [
        "# Full pipeline benchmark",
        "",
        f"n = {len(stream)} queries in the stream (10 unique + 5 duplicates for cache + 5 with "
        "attached context). Real Gemini API calls — cost computed from actual returned token "
        "usage x published pricing (see src/llmops_sdk/api/pricing.py).",
        "",
        "| | naive baseline | full pipeline |",
        "|---|---|---|",
        f"| completed calls | {naive['n']} | {full['n']} |",
        f"| total cost (est.) | ${naive['total_cost']:.6f} | ${full['total_cost']:.6f} |",
        f"| avg latency | {naive['avg_latency_ms']:.0f}ms | {full['avg_latency_ms']:.0f}ms |",
        f"| cache hit rate | {naive['cache_hit_rate']:.0%} | {full['cache_hit_rate']:.0%} |",
        f"| fact retention (context queries) | {fmt_retention(naive)} | {fmt_retention(full)} |",
    ]
    if naive["total_cost"] > 0:
        lines.append(f"\nCost reduction: {(1 - full['total_cost'] / naive['total_cost']):.0%}")

    output_text = "\n".join(lines)
    print(output_text)
    Path("reports/benchmark_results.md").write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()
