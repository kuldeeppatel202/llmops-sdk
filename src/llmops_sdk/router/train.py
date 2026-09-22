"""Fine-tune distilbert-base-uncased as a 3-class (simple/moderate/complex) query router.

Usage:
    python -m llmops_sdk.router.train --input data/labeled/dolly_labeled.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from llmops_sdk.router.baseline import TIERS

TIER_TO_ID = {tier: i for i, tier in enumerate(TIERS)}
ID_TO_TIER = {i: tier for tier, i in TIER_TO_ID.items()}


def load_labeled_examples(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def split_examples(rows: list[dict], seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    labels = [row["true_tier"] for row in rows]
    train_rows, temp_rows = train_test_split(
        rows, test_size=0.2, random_state=seed, stratify=labels
    )
    temp_labels = [row["true_tier"] for row in temp_rows]
    val_rows, test_rows = train_test_split(
        temp_rows, test_size=0.5, random_state=seed, stratify=temp_labels
    )
    return train_rows, val_rows, test_rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def to_hf_dataset(rows: list[dict], tokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_list(
        [{"text": row["query"], "label": TIER_TO_ID[row["true_tier"]]} for row in rows]
    )
    return ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
    )


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", labels=list(range(len(TIERS)))),
    }


def format_confusion_matrix(cm: np.ndarray) -> str:
    header = "| true \\ pred | " + " | ".join(TIERS) + " |"
    divider = "|---|" + "---|" * len(TIERS)
    lines = [header, divider]
    for i, tier in enumerate(TIERS):
        lines.append(f"| {tier} | " + " | ".join(str(x) for x in cm[i]) + " |")
    return "\n".join(lines)


def write_training_log(
    log_path: Path,
    args: argparse.Namespace,
    n_train: int,
    n_val: int,
    n_test: int,
    val_metrics: dict,
    test_metrics: dict,
    confusion: np.ndarray,
) -> None:
    lines = [
        "# Router fine-tuning log",
        "",
        "## Hyperparameters",
        f"- base model: distilbert-base-uncased",
        f"- epochs: {args.epochs}",
        f"- batch size: {args.batch_size}",
        f"- learning rate: {args.lr}",
        f"- max sequence length: {args.max_length}",
        f"- seed: {args.seed}",
        "",
        "## Data split",
        f"- train: {n_train}",
        f"- val: {n_val}",
        f"- test: {n_test}",
        "",
        "## Validation metrics (best checkpoint)",
        f"- accuracy: {val_metrics['eval_accuracy']:.4f}",
        f"- macro-F1: {val_metrics['eval_f1_macro']:.4f}",
        "",
        "## Test metrics (held-out, final)",
        f"- accuracy: {test_metrics['eval_accuracy']:.4f}",
        f"- macro-F1: {test_metrics['eval_f1_macro']:.4f}",
        "",
        "## Confusion matrix (test set)",
        format_confusion_matrix(confusion),
        "",
        "## Baseline comparison",
        "Run `python scripts/run_baseline_eval.py data/labeled/test.jsonl` "
        "(the same held-out test set written by this script) to get the rule-based "
        "baseline's accuracy/F1 for an apples-to-apples comparison.",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/labeled/dolly_labeled.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/router-distilbert"))
    parser.add_argument("--log-path", type=Path, default=Path("reports/training_log.md"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_labeled_examples(args.input)
    train_rows, val_rows, test_rows = split_examples(rows, args.seed)

    # Test set is written out so run_baseline_eval.py can be scored on the exact
    # same held-out examples for a fair baseline-vs-fine-tuned comparison.
    write_jsonl(train_rows, Path("data/labeled/train.jsonl"))
    write_jsonl(val_rows, Path("data/labeled/val.jsonl"))
    write_jsonl(test_rows, Path("data/labeled/test.jsonl"))

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_ds = to_hf_dataset(train_rows, tokenizer, args.max_length)
    val_ds = to_hf_dataset(val_rows, tokenizer, args.max_length)
    test_ds = to_hf_dataset(test_rows, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=len(TIERS), id2label=ID_TO_TIER, label2id=TIER_TO_ID
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=args.seed,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    val_metrics = trainer.evaluate(val_ds)
    test_metrics = trainer.evaluate(test_ds)

    test_logits = trainer.predict(test_ds).predictions
    test_preds = np.argmax(test_logits, axis=-1)
    test_labels = [TIER_TO_ID[row["true_tier"]] for row in test_rows]
    confusion = confusion_matrix(test_labels, test_preds, labels=list(range(len(TIERS))))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))

    write_training_log(args.log_path, args, len(train_rows), len(val_rows), len(test_rows), val_metrics, test_metrics, confusion)
    print(f"Model saved to {args.output_dir}")
    print(f"Training log written to {args.log_path}")
    print(f"Test accuracy: {test_metrics['eval_accuracy']:.4f}, macro-F1: {test_metrics['eval_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
