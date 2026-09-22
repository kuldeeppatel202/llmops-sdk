"""FastAPI service wiring the full pipeline: semantic cache -> input guardrails ->
router -> context compressor (if context given) -> model call -> output guardrails
-> cache write.

The model call here is a mock (no real API spend/quota on every request) — see
scripts/run_full_benchmark.py for a comparison against real Gemini calls on a small
query set, which is where genuine latency/cost numbers come from.
"""

from __future__ import annotations

import time
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel

from llmops_sdk.api.pricing import TIER_TO_MODEL, estimate_cost
from llmops_sdk.cache.semantic_cache import SemanticCache
from llmops_sdk.compressor.context_compressor import ContextCompressor, count_tokens
from llmops_sdk.guardrails.input_checks import check_input
from llmops_sdk.guardrails.output_checks import check_faithfulness
from llmops_sdk.router.baseline import BaselineRouter
from llmops_sdk.router.model import DEFAULT_MODEL_DIR, FineTunedRouter

app = FastAPI(title="LLM Ops SDK")


class _BaselineRouterAdapter:
    """Matches FineTunedRouter's predict(query) -> (tier, confidence) interface."""

    def __init__(self) -> None:
        self._router = BaselineRouter()

    def predict(self, query: str) -> tuple[str, float]:
        return self._router.predict(query), 1.0


@lru_cache
def get_cache() -> SemanticCache:
    return SemanticCache()


@lru_cache
def get_compressor() -> ContextCompressor:
    return ContextCompressor()


@lru_cache
def get_router():
    if DEFAULT_MODEL_DIR.exists():
        return FineTunedRouter()
    return _BaselineRouterAdapter()


def mock_llm_call(model: str, query: str) -> tuple[str, int]:
    """Stand-in for a real model call — no real API spend/quota per request."""
    response_text = f"[mock {model} response] Placeholder answer for: {query[:80]}"
    return response_text, count_tokens(response_text)


class QueryRequest(BaseModel):
    query: str
    context: str | None = None
    compression_ratio: float = 0.5


class QueryResponse(BaseModel):
    response: str
    tier: str
    tier_confidence: float
    cache_hit: bool
    cache_similarity: float
    tokens_sent: int
    tokens_saved_by_compression: int
    model_used: str
    estimated_cost_usd: float
    latency_ms: float
    input_flagged_pii: bool
    input_flagged_injection: bool
    output_faithful: bool | None


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()
    cache = get_cache()

    cached_response, similarity = cache.get(req.query)
    if cached_response is not None:
        return QueryResponse(
            response=cached_response,
            tier="cached",
            tier_confidence=1.0,
            cache_hit=True,
            cache_similarity=similarity,
            tokens_sent=0,
            tokens_saved_by_compression=0,
            model_used="cache",
            estimated_cost_usd=0.0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            input_flagged_pii=False,
            input_flagged_injection=False,
            output_faithful=None,
        )

    input_check = check_input(req.query)
    if not input_check.passed:
        # Flagged queries are blocked before reaching the model or the cache —
        # PII/injection attempts shouldn't be answered or persisted.
        return QueryResponse(
            response="",
            tier="blocked",
            tier_confidence=0.0,
            cache_hit=False,
            cache_similarity=similarity,
            tokens_sent=0,
            tokens_saved_by_compression=0,
            model_used="none",
            estimated_cost_usd=0.0,
            latency_ms=(time.perf_counter() - t0) * 1000,
            input_flagged_pii=input_check.has_pii,
            input_flagged_injection=input_check.has_injection,
            output_faithful=None,
        )

    tier, confidence = get_router().predict(req.query)

    tokens_saved = 0
    context_used = ""
    if req.context:
        original_tokens = count_tokens(req.context)
        budget = max(1, round(original_tokens * req.compression_ratio))
        compression = get_compressor().compress(req.query, req.context, token_budget=budget)
        context_used = compression.compressed_text
        tokens_saved = compression.original_tokens - compression.compressed_tokens

    model_name = TIER_TO_MODEL[tier]
    input_tokens = count_tokens(req.query) + count_tokens(context_used)
    response_text, output_tokens = mock_llm_call(model_name, req.query)
    cost = estimate_cost(model_name, input_tokens, output_tokens)

    output_faithful = check_faithfulness(response_text, context_used).passed if context_used else None

    cache.set(req.query, response_text)

    return QueryResponse(
        response=response_text,
        tier=tier,
        tier_confidence=confidence,
        cache_hit=False,
        cache_similarity=similarity,
        tokens_sent=input_tokens,
        tokens_saved_by_compression=tokens_saved,
        model_used=model_name,
        estimated_cost_usd=cost,
        latency_ms=(time.perf_counter() - t0) * 1000,
        input_flagged_pii=input_check.has_pii,
        input_flagged_injection=input_check.has_injection,
        output_faithful=output_faithful,
    )
