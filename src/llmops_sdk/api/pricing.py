"""Per-model pricing and the router-tier -> model mapping.

Prices are Gemini API published paid-tier rates, USD per 1M tokens, fetched from
https://ai.google.dev/gemini-api/docs/pricing on 2026-09-22 — not guessed. Used to
estimate cost even when running on the free tier (which is $0 in practice but
quota-limited), so the benchmark can report what these queries would cost at scale.

Model choice note: gemini-3.1-pro (the originally-intended "strong" tier) turned out
to have **zero free-tier quota** on this API key (confirmed via a direct 429
RESOURCE_EXHAUSTED response naming `limit: 0` — Pro access requires billing enabled,
not just higher usage). gemini-3.5-flash and gemini-3.5-flash-lite were also
quota-exhausted from earlier phases' real usage today. Router tiers collapse to two
real cost points here (simple+moderate share the cheap tier) rather than three,
because a third distinct free-tier-available model wasn't reliably available at
benchmark time — this is an honest scoping constraint, not a design ideal.
"""

from __future__ import annotations

TIER_TO_MODEL = {
    "simple": "gemini-3.1-flash-lite",
    "moderate": "gemini-3.1-flash-lite",
    "complex": "gemini-3-flash-preview",
}

NAIVE_MODEL = "gemini-3-flash-preview"

MODEL_PRICING_PER_1M_TOKENS = {
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-3-flash-preview": {"input": 0.50, "output": 3.00},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING_PER_1M_TOKENS[model]
    return (input_tokens / 1_000_000) * pricing["input"] + (output_tokens / 1_000_000) * pricing["output"]
