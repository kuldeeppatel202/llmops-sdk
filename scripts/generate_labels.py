"""CLI: sample queries from a public instruction dataset and label them by complexity
tier (simple/moderate/complex) using Gemini as a labeling assistant.

Requires GEMINI_API_KEY, either as an environment variable or in a .env file at the
project root (see .env.example).

Usage:
    # Sanity-check the rubric on 10 examples first (writes data/labeled/dry_run_sample.jsonl)
    python scripts/generate_labels.py --dry-run

    # Full batch
    python scripts/generate_labels.py --n 2500 --output data/labeled/dolly_labeled.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from llmops_sdk.router.baseline import TIERS  # noqa: E402

# Rubric from the technical deep-dive doc — this is the ground-truth definition an
# interviewer can be shown; the LLM is only a labeling assistant applying it.
RUBRIC = """\
Classify the COMPLEXITY of answering the following user query into exactly one tier:

- simple: a single-fact lookup or definition. Answerable directly from one piece of
  knowledge, no synthesis or multi-step reasoning required.
- moderate: requires synthesizing 2-3 pieces of information, or light reasoning
  (e.g. summarizing, comparing two things, explaining a mechanism).
- complex: requires multi-step reasoning, code generation, or open-ended analysis
  (e.g. multi-hop reasoning, writing/debugging code, designing something, weighing
  trade-offs).

Judge the complexity of the QUERY itself, not how long or short it is phrased."""

FEW_SHOT_EXAMPLES = [
    ("What is the boiling point of water at sea level in Celsius?", "simple",
     "Single fact lookup, no synthesis needed."),
    ("Summarize the main differences between TCP and UDP.", "moderate",
     "Requires synthesizing a few distinct facts into a comparison."),
    ("Write a Python function that returns the nth Fibonacci number using memoization, "
     "and explain its time complexity.", "complex",
     "Requires code generation plus algorithmic analysis."),
]


class LabelResponse(BaseModel):
    tier: Literal["simple", "moderate", "complex"]
    rationale: str


def load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader — avoids adding python-dotenv as a dependency for one env var."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def build_prompt(query: str) -> str:
    examples_text = "\n\n".join(
        f'Query: "{q}"\nTier: {tier}\nRationale: {rationale}'
        for q, tier, rationale in FEW_SHOT_EXAMPLES
    )
    return (
        f"{RUBRIC}\n\n"
        f"Worked examples:\n\n{examples_text}\n\n"
        f'Now classify this query:\n\nQuery: "{query}"\n\n'
        "Respond with the tier and a one-sentence rationale."
    )


def load_sample(dataset_name: str, n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split="train")
    ds = ds.shuffle(seed=seed)
    n = min(n, len(ds))
    rows = ds.select(range(n))
    return [
        {"id": i, "query": row["instruction"], "category": row.get("category")}
        for i, row in enumerate(rows)
        if row["instruction"] and row["instruction"].strip()
    ]


def label_query(client, model: str, query: str, max_retries: int) -> LabelResponse:
    from google.genai import types

    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=build_prompt(query),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=LabelResponse,
                ),
            )
            parsed = response.parsed
            if isinstance(parsed, LabelResponse):
                return parsed
            return LabelResponse.model_validate_json(response.text)
        except Exception as e:  # noqa: BLE001 - transient API/parse errors, retry
            last_error = e
            if attempt < max_retries - 1:
                # Gemini flash models intermittently return 503 "high demand" under
                # real load (observed in practice) — back off longer than a typical
                # transient-error retry to actually ride it out across a 2500-call batch.
                time.sleep(min(60, 5 * (2**attempt)))
    raise RuntimeError(f"Failed to label query after {max_retries} attempts: {query!r}") from last_error


def load_already_labeled_ids(output_path: Path) -> set[int]:
    if not output_path.exists():
        return set()
    ids = set()
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line)["id"])
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    parser.add_argument("--n", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="gemini-3.5-flash-lite")
    parser.add_argument("--output", type=Path, default=Path("data/labeled/dolly_labeled.jsonl"))
    parser.add_argument("--dry-run", action="store_true", help="Label only 10 examples for rubric sanity-check")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between API calls")
    parser.add_argument("--resume", action="store_true", help="Skip ids already present in --output (full runs only)")
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set (checked environment and .env).", file=sys.stderr)
        sys.exit(1)

    from google import genai

    client = genai.Client()

    sample_size = 10 if args.dry_run else args.n
    rows = load_sample(args.dataset, sample_size, args.seed)
    output_path = Path("data/labeled/dry_run_sample.jsonl") if args.dry_run else args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already_labeled = load_already_labeled_ids(output_path) if (args.resume and not args.dry_run) else set()
    if already_labeled:
        print(f"Resuming: {len(already_labeled)} examples already labeled in {output_path}, skipping those ids.")
        rows = [r for r in rows if r["id"] not in already_labeled]

    # Full batches (2500 calls) hit occasional transient API errors in practice, so
    # results are appended and flushed per-row: a mid-run failure loses at most the
    # one in-flight example, not everything labeled so far, and --resume can pick up
    # where it left off.
    mode = "a" if already_labeled else "w"
    tier_counts = {tier: 0 for tier in TIERS}
    n_labeled = 0
    failed_queries: list[dict] = []

    with output_path.open(mode, encoding="utf-8") as f:
        for i, row in enumerate(rows):
            try:
                result = label_query(client, args.model, row["query"], args.max_retries)
            except RuntimeError as e:
                print(f"WARNING: skipping id={row['id']} after repeated failures: {e}", file=sys.stderr)
                failed_queries.append(row)
                continue

            labeled_row = {
                "id": row["id"],
                "query": row["query"],
                "true_tier": result.tier,
                "rationale": result.rationale,
                "category": row["category"],
            }
            f.write(json.dumps(labeled_row) + "\n")
            f.flush()
            n_labeled += 1
            tier_counts[result.tier] += 1

            if args.dry_run:
                print(f"[{i + 1}/{len(rows)}] {result.tier:9s} | {row['query'][:80]}")
                print(f"           rationale: {result.rationale}")
            elif (i + 1) % 50 == 0:
                print(f"labeled {i + 1}/{len(rows)}")

            if args.sleep:
                time.sleep(args.sleep)

    print(f"\nWrote {n_labeled} labeled examples to {output_path}")
    print(f"Tier distribution: {tier_counts}")
    if failed_queries:
        print(f"{len(failed_queries)} examples failed after {args.max_retries} retries each and were skipped.")
        print("Re-run with --resume to retry only the missing ids.")
    if args.dry_run:
        print("\nDry run complete. Review the rationale for each label above, then re-run without --dry-run.")


if __name__ == "__main__":
    main()
