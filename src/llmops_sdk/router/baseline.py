"""Rule-based complexity classifier — the baseline the fine-tuned router (Phase 2) must beat."""

from __future__ import annotations

TIERS = ("simple", "moderate", "complex")

# Signals that a query needs multi-step reasoning, synthesis, or generation —
# strong enough on their own to call it "complex" regardless of length.
COMPLEX_KEYWORDS = (
    "compare",
    "explain why",
    "step by step",
    "step-by-step",
    "analyze",
    "analyse",
    "write a function",
    "write code",
    "implement",
    "design a",
    "optimize",
    "prove that",
    "derive",
    "debug",
    "algorithm",
    "trade-off",
    "tradeoff",
    "pros and cons",
    "root cause",
)

# "why"/"how" ask for an explanation or mechanism, not just a fact lookup —
# treated as a moderate signal unless a complex keyword above also fires.
REASONING_QUESTION_WORDS = ("why", "how")

# "what/who/when/where/which" are typically single-fact lookups.
LOOKUP_QUESTION_WORDS = ("what", "who", "when", "where", "which")


def _first_word_lower(query: str) -> str:
    stripped = query.strip().lower()
    return stripped.split(" ", 1)[0].rstrip("?,.!") if stripped else ""


def classify(query: str) -> str:
    """Tag a query as simple/moderate/complex using length + keyword heuristics."""
    text = query.strip()
    if not text:
        return "simple"

    lowered = text.lower()
    word_count = len(text.split())

    if any(keyword in lowered for keyword in COMPLEX_KEYWORDS):
        return "complex"

    first_word = _first_word_lower(text)
    has_reasoning_word = first_word in REASONING_QUESTION_WORDS or any(
        f" {w} " in f" {lowered} " for w in REASONING_QUESTION_WORDS
    )
    has_lookup_word = first_word in LOOKUP_QUESTION_WORDS

    if word_count > 40:
        return "complex"

    if word_count > 15:
        return "moderate" if not has_reasoning_word else "complex"

    if has_reasoning_word:
        return "moderate"

    if has_lookup_word or word_count <= 6:
        return "simple"

    return "moderate"


class BaselineRouter:
    """Same `predict` interface the Phase 2 fine-tuned router will expose, for drop-in comparison."""

    def predict(self, query: str) -> str:
        return classify(query)

    def predict_batch(self, queries: list[str]) -> list[str]:
        return [self.predict(q) for q in queries]
