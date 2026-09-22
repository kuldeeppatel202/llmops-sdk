"""In-memory semantic cache: embed queries with a sentence-transformer and serve a
cached response when a new query's cosine similarity to a stored query clears a
threshold.

Storage is a plain in-memory array — sufficient at portfolio scale (per the project's
technical deep-dive). At production scale this would move to a proper vector index
(e.g. FAISS) without changing this class's interface.
"""

from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "all-MiniLM-L6-v2"

# Chosen via scripts/sweep_cache_threshold.py — see reports/cache_threshold_sweep.md
# for the hit-rate/false-positive-rate tradeoff that justifies this value.
DEFAULT_THRESHOLD = 0.80


class SemanticCache:
    def __init__(self, threshold: float = DEFAULT_THRESHOLD, model_name: str = DEFAULT_MODEL):
        from sentence_transformers import SentenceTransformer

        self.threshold = threshold
        self._model = SentenceTransformer(model_name)
        self._queries: list[str] = []
        self._responses: list[str] = []
        self._embeddings: np.ndarray | None = None  # (n, dim), L2-normalized rows

    def _embed(self, text: str) -> np.ndarray:
        return self._model.encode([text], normalize_embeddings=True)[0]

    def get(self, query: str) -> tuple[str | None, float]:
        """Returns (cached_response, similarity) on a hit, or (None, best_similarity) on a miss."""
        if self._embeddings is None:
            return None, 0.0
        q_emb = self._embed(query)
        similarities = self._embeddings @ q_emb
        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])
        if best_similarity >= self.threshold:
            return self._responses[best_idx], best_similarity
        return None, best_similarity

    def set(self, query: str, response: str) -> None:
        emb = self._embed(query).reshape(1, -1)
        self._queries.append(query)
        self._responses.append(response)
        self._embeddings = emb if self._embeddings is None else np.vstack([self._embeddings, emb])

    def __len__(self) -> int:
        return len(self._queries)
