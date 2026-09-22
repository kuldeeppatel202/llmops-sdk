import pytest

from llmops_sdk.router.baseline import TIERS
from llmops_sdk.router.model import DEFAULT_MODEL_DIR, FineTunedRouter

pytestmark = pytest.mark.skipif(
    not DEFAULT_MODEL_DIR.exists(), reason="fine-tuned model not present (run router/train.py first)"
)


@pytest.fixture(scope="module")
def router():
    return FineTunedRouter()


def test_predict_returns_valid_tier_and_confidence(router):
    tier, confidence = router.predict("What is the capital of France?")
    assert tier in TIERS
    assert 0.0 <= confidence <= 1.0


def test_predict_is_deterministic(router):
    query = "Write a function to merge two sorted lists."
    tier1, conf1 = router.predict(query)
    tier2, conf2 = router.predict(query)
    assert tier1 == tier2
    assert conf1 == conf2
