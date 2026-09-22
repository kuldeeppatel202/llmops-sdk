"""Load the fine-tuned DistilBERT router (saved by router/train.py) for real-time
single-query inference — the predict(query) -> (tier, confidence) interface the
API layer uses."""

from __future__ import annotations

from pathlib import Path

from llmops_sdk.router.baseline import TIERS

DEFAULT_MODEL_DIR = Path("models/router-distilbert")


class FineTunedRouter:
    def __init__(self, model_dir: Path | str = DEFAULT_MODEL_DIR):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_dir = Path(model_dir)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self._model.eval()

    def predict(self, query: str) -> tuple[str, float]:
        inputs = self._tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        probs = self._torch.softmax(logits, dim=-1)[0]
        idx = int(self._torch.argmax(probs))
        return TIERS[idx], float(probs[idx])
