"""Context compressor: score each sentence in a context block by its embedding
similarity to the query, then greedily keep the highest-scoring sentences until a
token budget is filled — instead of naive head/tail truncation, which isn't
correlated with which sentences actually answer the query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter — doesn't handle abbreviations (e.g. "Dr. Smith")
    perfectly, an accepted simplification for a v1 (per the project's own scoping)."""
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def count_tokens(text: str) -> int:
    """Whitespace word count, used as an approximate token count — no tokenizer
    dependency; good enough for budget trimming, not meant to match any specific
    model's exact tokenizer."""
    return len(text.split())


@dataclass
class CompressionResult:
    compressed_text: str
    kept_sentences: list[str]
    original_tokens: int
    compressed_tokens: int

    @property
    def compression_ratio(self) -> float:
        return self.compressed_tokens / self.original_tokens if self.original_tokens else 0.0


class ContextCompressor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def compress(self, query: str, context: str, token_budget: int) -> CompressionResult:
        sentences = split_sentences(context)
        original_tokens = count_tokens(context)
        if not sentences:
            return CompressionResult("", [], original_tokens, 0)

        sentence_embs = self._model.encode(sentences, normalize_embeddings=True)
        query_emb = self._model.encode([query], normalize_embeddings=True)[0]
        scores = sentence_embs @ query_emb

        ranked_idx = sorted(range(len(sentences)), key=lambda i: -scores[i])
        kept_idx: list[int] = []
        used_tokens = 0
        for i in ranked_idx:
            t = count_tokens(sentences[i])
            if used_tokens + t > token_budget:
                break
            kept_idx.append(i)
            used_tokens += t

        # Re-order kept sentences back into original document order for coherence,
        # per the technical deep-dive's recommendation (models are sensitive to order).
        kept_idx.sort()
        kept_sentences = [sentences[i] for i in kept_idx]
        compressed_text = " ".join(kept_sentences)

        return CompressionResult(compressed_text, kept_sentences, original_tokens, count_tokens(compressed_text))
