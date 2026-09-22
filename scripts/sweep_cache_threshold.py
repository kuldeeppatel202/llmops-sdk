"""CLI: sweep semantic cache similarity thresholds against a labeled near-duplicate/
distinct query-pair set, reporting hit rate vs. false-positive rate per threshold.

Usage:
    python scripts/sweep_cache_threshold.py data/cache_eval_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_pairs(path: Path) -> list[dict]:
    pairs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, nargs="?", default=Path("data/cache_eval_pairs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/cache_threshold_sweep.md"))
    parser.add_argument(
        "--thresholds", type=float, nargs="+", default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]
    )
    parser.add_argument(
        "--max-fp-rate",
        type=float,
        default=0.05,
        help="Recommend the threshold with the best hit rate among those at or below this false positive rate",
    )
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer

    pairs = load_pairs(args.input)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    queries_a = [p["query_a"] for p in pairs]
    queries_b = [p["query_b"] for p in pairs]
    emb_a = model.encode(queries_a, normalize_embeddings=True)
    emb_b = model.encode(queries_b, normalize_embeddings=True)
    similarities = (emb_a * emb_b).sum(axis=1)

    duplicates = [(sim, p["is_duplicate"]) for sim, p in zip(similarities, pairs)]
    n_dup = sum(1 for _, is_dup in duplicates if is_dup)
    n_distinct = len(duplicates) - n_dup

    rows = []
    for t in sorted(args.thresholds):
        hits = sum(1 for sim, is_dup in duplicates if is_dup and sim >= t)
        false_positives = sum(1 for sim, is_dup in duplicates if not is_dup and sim >= t)
        hit_rate = hits / n_dup if n_dup else 0.0
        fp_rate = false_positives / n_distinct if n_distinct else 0.0
        rows.append((t, hit_rate, fp_rate))

    lines = [
        "# Semantic cache threshold sweep",
        "",
        f"n_duplicate_pairs = {n_dup}, n_distinct_pairs = {n_distinct}",
        "",
        "| threshold | hit rate | false positive rate |",
        "|---|---|---|",
    ]
    for t, hit_rate, fp_rate in rows:
        lines.append(f"| {t:.2f} | {hit_rate:.2%} | {fp_rate:.2%} |")

    # A strict zero-FP requirement collapses hit rate (a cache that almost never fires
    # isn't useful — see the technical deep-dive's own warning on this). Instead,
    # recommend the best hit rate within an accepted false-positive budget.
    eligible = [r for r in rows if r[2] <= args.max_fp_rate]
    recommended = max(eligible, key=lambda r: r[1]) if eligible else min(rows, key=lambda r: r[2])
    zero_fp_rows = [r for r in rows if r[2] == 0.0]
    strictest = min(zero_fp_rows, key=lambda r: r[0]) if zero_fp_rows else None

    lines += [
        "",
        f"## Recommended threshold: {recommended[0]:.2f}",
        f"Hit rate {recommended[1]:.2%}, false positive rate {recommended[2]:.2%} — "
        f"the best hit rate among thresholds at or below a {args.max_fp_rate:.0%} false "
        "positive budget.",
    ]
    if strictest and strictest[0] != recommended[0]:
        lines.append(
            f"A stricter zero-FP threshold was available ({strictest[0]:.2f}, "
            f"{strictest[2]:.2%} FP) but collapses hit rate to {strictest[1]:.2%}, which "
            "isn't a useful cache in practice — this tradeoff is a judgment call, not an "
            "arbitrary pick; see the full table above."
        )

    output_text = "\n".join(lines)
    print(output_text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output_text, encoding="utf-8")


if __name__ == "__main__":
    main()
