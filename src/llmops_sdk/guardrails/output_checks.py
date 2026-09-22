"""Output guardrails: Pydantic-based schema validation for structured responses,
and a lightweight word-overlap faithfulness heuristic for RAG-style responses.
The faithfulness check is a heuristic signal ("is this grounded in what we gave
it"), not a hallucination-detection system — described honestly, not overclaimed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

# Common short words excluded so overlap isn't inflated by function words that
# appear in almost any two texts regardless of topic.
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "and", "or", "but", "with", "as",
    "it", "its", "this", "that", "these", "those", "by", "from", "not",
}


def validate_schema(response_text: str, schema: type[BaseModel]) -> tuple[bool, str | None, BaseModel | None]:
    """Returns (is_valid, error_message, parsed_instance)."""
    try:
        parsed = schema.model_validate_json(response_text)
        return True, None, parsed
    except ValidationError as e:
        return False, str(e), None


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


@dataclass
class FaithfulnessResult:
    overlap_ratio: float
    passed: bool


def check_faithfulness(response_text: str, context: str, min_overlap: float = 0.3) -> FaithfulnessResult:
    """Fraction of the response's content words that also appear in the context."""
    response_words = _content_words(response_text)
    context_words = _content_words(context)
    if not response_words:
        return FaithfulnessResult(overlap_ratio=0.0, passed=False)
    overlap = len(response_words & context_words) / len(response_words)
    return FaithfulnessResult(overlap_ratio=overlap, passed=overlap >= min_overlap)
