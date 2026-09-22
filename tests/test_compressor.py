import pytest

from llmops_sdk.compressor.context_compressor import ContextCompressor, count_tokens, split_sentences


def test_split_sentences_basic():
    text = "The sky is blue. Water boils at 100C. Paris is the capital of France."
    sentences = split_sentences(text)
    assert sentences == [
        "The sky is blue.",
        "Water boils at 100C.",
        "Paris is the capital of France.",
    ]


def test_split_sentences_empty_string():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_split_sentences_single_sentence_no_trailing_punctuation():
    assert split_sentences("just some text") == ["just some text"]


def test_count_tokens():
    assert count_tokens("one two three") == 3
    assert count_tokens("") == 0


@pytest.fixture(scope="module")
def compressor():
    return ContextCompressor()


def test_compress_keeps_relevant_sentence_and_drops_irrelevant(compressor):
    context = (
        "The company was founded in 1998 in Seattle. "
        "Employees receive 15 days of paid time off per year. "
        "The office has a rooftop garden and a cafeteria."
    )
    result = compressor.compress("How much PTO do employees get?", context, token_budget=10)
    assert "15 days of paid time off" in result.compressed_text
    assert "rooftop garden" not in result.compressed_text


def test_compress_respects_token_budget(compressor):
    context = (
        "The company was founded in 1998 in Seattle. "
        "Employees receive 15 days of paid time off per year. "
        "The office has a rooftop garden and a cafeteria. "
        "The company was acquired by a larger firm in 2015."
    )
    result = compressor.compress("How much PTO do employees get?", context, token_budget=8)
    assert result.compressed_tokens <= 8


def test_compress_reduces_token_count(compressor):
    context = (
        "The company was founded in 1998 in Seattle. "
        "Employees receive 15 days of paid time off per year. "
        "The office has a rooftop garden and a cafeteria. "
        "The company was acquired by a larger firm in 2015."
    )
    result = compressor.compress("How much PTO do employees get?", context, token_budget=8)
    assert result.compressed_tokens < result.original_tokens
    assert result.compression_ratio < 1.0


def test_compress_empty_context_returns_empty_result(compressor):
    result = compressor.compress("Any query", "", token_budget=50)
    assert result.compressed_text == ""
    assert result.kept_sentences == []
    assert result.compressed_tokens == 0


def test_compress_generous_budget_keeps_everything(compressor):
    context = "The sky is blue. Water boils at 100C."
    result = compressor.compress("What color is the sky?", context, token_budget=1000)
    assert result.compressed_tokens == result.original_tokens
