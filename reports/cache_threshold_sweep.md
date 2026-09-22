# Semantic cache threshold sweep

n_duplicate_pairs = 20, n_distinct_pairs = 20

| threshold | hit rate | false positive rate |
|---|---|---|
| 0.70 | 95.00% | 30.00% |
| 0.75 | 90.00% | 10.00% |
| 0.80 | 75.00% | 5.00% |
| 0.85 | 65.00% | 5.00% |
| 0.90 | 55.00% | 5.00% |
| 0.92 | 35.00% | 0.00% |
| 0.95 | 5.00% | 0.00% |

## Recommended threshold: 0.80
Hit rate 75.00%, false positive rate 5.00% — the best hit rate among thresholds at or below a 5% false positive budget.
A stricter zero-FP threshold was available (0.92, 0.00% FP) but collapses hit rate to 35.00%, which isn't a useful cache in practice — this tradeoff is a judgment call, not an arbitrary pick; see the full table above.