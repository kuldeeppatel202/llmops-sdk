from llmops_sdk.router.baseline import BaselineRouter, classify


def test_empty_string_is_simple():
    assert classify("") == "simple"
    assert classify("   ") == "simple"


def test_short_lookup_query_is_simple():
    assert classify("What is the capital of France?") == "simple"


def test_short_query_without_question_word_is_simple():
    assert classify("Define entropy") == "simple"


def test_reasoning_question_is_moderate():
    assert classify("How does photosynthesis work?") == "moderate"
    assert classify("Why is the sky blue?") == "moderate"


def test_complex_keyword_triggers_complex_regardless_of_length():
    assert classify("Compare A and B.") == "complex"
    assert classify("Write a function to reverse a list") == "complex"
    assert classify("debug this") == "complex"


def test_very_long_query_is_complex():
    long_query = " ".join(["word"] * 45)
    assert classify(long_query) == "complex"


def test_long_query_without_reasoning_word_is_moderate():
    # Long enough to rule out "simple", but no complex keyword and no why/how —
    # ambiguous case that should land in the middle tier, not get bumped to complex.
    query = " ".join(["banana"] * 20)
    assert classify(query) == "moderate"


def test_long_reasoning_query_is_complex():
    query = "why " + " ".join(["reason"] * 20)
    assert classify(query) == "complex"


def test_ambiguous_medium_length_query_defaults_to_moderate():
    assert classify("Summarize the plot of this book for me please") == "moderate"


def test_baseline_router_predict_matches_classify():
    router = BaselineRouter()
    assert router.predict("What is 2+2?") == classify("What is 2+2?")


def test_baseline_router_predict_batch():
    router = BaselineRouter()
    queries = ["What is 2+2?", "Why is the sky blue?", "Compare Python and Go"]
    assert router.predict_batch(queries) == [router.predict(q) for q in queries]
