"""CLI: measure whether expected facts survive context compression, at a couple of
compression ratios, on a small hand-labeled set of (query, context, expected_facts)
triples.

No LLM call is wired up yet (that's Phase 5), so this checks fact-string presence in
the *compressed context* itself rather than in a generated answer — the honest proxy
available at this phase: if a fact doesn't survive into the compressed context, no
downstream model could use it either.

Usage:
    python scripts/eval_compression_quality.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmops_sdk.compressor.context_compressor import ContextCompressor, count_tokens


def load_eval_set(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/compression_eval.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/compression_quality.md"))
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.5, 0.33])
    args = parser.parse_args()

    examples = load_eval_set(args.input)
    compressor = ContextCompressor()

    lines = [
        "# Context compression quality eval",
        "",
        f"n = {len(examples)} (query, context, expected_facts) triples",
        "",
        "Fact retention = fraction of expected fact-strings still present (case-insensitive "
        "substring match) in the compressed context, at each compression ratio.",
        "",
        "| ratio | avg compression achieved | facts retained |",
        "|---|---|---|",
    ]

    for ratio in args.ratios:
        total_facts = 0
        retained_facts = 0
        achieved_ratios = []
        for ex in examples:
            original_tokens = count_tokens(ex["context"])
            budget = max(1, round(original_tokens * ratio))
            result = compressor.compress(ex["query"], ex["context"], token_budget=budget)
            achieved_ratios.append(result.compression_ratio)
            lowered = result.compressed_text.lower()
            for fact in ex["expected_facts"]:
                total_facts += 1
                if fact.lower() in lowered:
                    retained_facts += 1

        avg_ratio = sum(achieved_ratios) / len(achieved_ratios) if achieved_ratios else 0.0
        retention = retained_facts / total_facts if total_facts else 0.0
        lines.append(f"| target {ratio:.0%} | {avg_ratio:.0%} | {retention:.0%} ({retained_facts}/{total_facts}) |")

    output_text = "\n".join(lines)
    print(output_text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()
