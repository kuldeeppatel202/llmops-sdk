import pytest

from llmops_sdk.cache.semantic_cache import DEFAULT_THRESHOLD, SemanticCache


@pytest.fixture(scope="module")
def cache():
    return SemanticCache(threshold=DEFAULT_THRESHOLD)


def test_empty_cache_is_a_miss(cache):
    response, similarity = cache.get("What is the capital of France?")
    assert response is None
    assert similarity == 0.0


def test_exact_repeat_query_is_a_hit(cache):
    cache.set("What is the capital of France?", "Paris.")
    response, similarity = cache.get("What is the capital of France?")
    assert response == "Paris."
    assert similarity > 0.99


def test_paraphrased_query_is_a_hit(cache):
    # Verified similarity ~0.945 against the cached entry above — well above threshold.
    response, similarity = cache.get("Which city is the capital of France?")
    assert response == "Paris."
    assert similarity >= cache.threshold


def test_distinct_query_is_a_miss(cache):
    # Verified similarity ~0.48 — topically related (both "capital of a country")
    # but a genuinely different question, so it must not hit the cache.
    response, similarity = cache.get("What is the capital of Mongolia?")
    assert response is None
    assert similarity < cache.threshold


def test_len_tracks_stored_entries(cache):
    n_before = len(cache)
    cache.set("What is the weather today?", "Sunny, 72F.")
    assert len(cache) == n_before + 1


def test_higher_threshold_rejects_borderline_match():
    # Same paraphrase pair (~0.945 similarity) but a near-1.0 threshold should reject it.
    strict_cache = SemanticCache(threshold=0.999)
    strict_cache.set("What is the capital of France?", "Paris.")
    response, _ = strict_cache.get("Which city is the capital of France?")
    assert response is None
