# Router fine-tuning log

## Hyperparameters
- base model: distilbert-base-uncased
- epochs: 4
- batch size: 16
- learning rate: 2e-05
- max sequence length: 128
- seed: 42

## Data split
- train: 388
- val: 49
- test: 49

## Validation metrics (best checkpoint)
- accuracy: 0.7347
- macro-F1: 0.5054

## Test metrics (held-out, final)
- accuracy: 0.6735
- macro-F1: 0.4597

## Confusion matrix (test set)
| true \ pred | simple | moderate | complex |
|---|---|---|---|
| simple | 10 | 8 | 0 |
| moderate | 3 | 23 | 0 |
| complex | 1 | 4 | 0 |

## Baseline comparison (same held-out test.jsonl, n=49)

| model | accuracy | macro-F1 |
|---|---|---|
| rule-based baseline | 0.5714 | 0.40 |
| fine-tuned DistilBERT | 0.6735 | 0.4597 |

Fine-tuned router beats the baseline by +10.2pp accuracy and +6pp macro-F1 on the same
test set.

## Known limitation

Neither model predicts "complex" correctly on this test set (0 recall for both) — the
labeled dataset used here is a **486-example subset** (not the full ~2,500 the original
plan called for; scaled down deliberately for a faster iteration/learning pass), and
"complex" is only ~10% of it (49/486 total, 5/49 in this test split). That's too few
examples for either model to learn or reliably evaluate that class. With the full-size
dataset this would need re-running — flagging this honestly rather than hiding it, per
the project's own "spot-check and report agreement rate" principle.