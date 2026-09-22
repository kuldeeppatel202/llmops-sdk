"""CLI: evaluate the rule-based baseline router against a labeled dataset.

Usage:
    python scripts/run_baseline_eval.py data/labeled/test.jsonl
    python scripts/run_baseline_eval.py data/labeled/test.csv --output reports/baseline_eval.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from llmops_sdk.router.baseline import TIERS, BaselineRouter  # noqa: E402


def load_examples(path: Path) -> tuple[list[str], list[str]]:
    queries: list[str] = []
    true_tiers: list[str] = []

    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                queries.append(row["query"])
                true_tiers.append(row["true_tier"])
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                queries.append(row["query"])
                true_tiers.append(row["true_tier"])
    else:
        raise ValueError(f"Unsupported file type: {path.suffix} (expected .jsonl or .csv)")

    return queries, true_tiers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Labeled dataset (.jsonl or .csv) with query, true_tier columns")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to write the report as markdown")
    args = parser.parse_args()

    queries, true_tiers = load_examples(args.input)
    router = BaselineRouter()
    predicted_tiers = router.predict_batch(queries)

    accuracy = accuracy_score(true_tiers, predicted_tiers)
    report = classification_report(true_tiers, predicted_tiers, labels=list(TIERS), zero_division=0)

    output_lines = [
        f"# Baseline router evaluation - {args.input.name}",
        "",
        f"n = {len(queries)}",
        f"accuracy = {accuracy:.4f}",
        "",
        "```",
        report,
        "```",
    ]
    output_text = "\n".join(output_lines)

    print(output_text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()
