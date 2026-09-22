# Full pipeline benchmark

n = 20 queries in the stream (10 unique + 5 duplicates for cache + 5 with attached context). Real Gemini API calls — cost computed from actual returned token usage x published pricing (see src/llmops_sdk/api/pricing.py).

| | naive baseline | full pipeline |
|---|---|---|
| completed calls | 19 | 20 |
| total cost (est.) | $0.022685 | $0.008043 |
| avg latency | 6121ms | 2271ms |
| cache hit rate | 0% | 25% |
| fact retention (context queries) | 82% | 55% |

Cost reduction: 65%

## Notes

- **Model choice constraint**: the originally-intended "strong" tier (`gemini-3.1-pro`)
  has **zero free-tier quota** on this API key (confirmed via a direct 429
  `RESOURCE_EXHAUSTED` response naming `limit: 0` — Pro access requires billing, not
  just more usage). `gemini-3.5-flash`/`gemini-3.5-flash-lite` were also exhausted from
  earlier phases' real usage today. The naive baseline and the router's "complex" tier
  both fall back to `gemini-3-flash-preview`; "simple"/"moderate" share
  `gemini-3.1-flash-lite`. Two real cost tiers rather than three — an honest scoping
  constraint from live free-tier availability, not a design ideal. See
  `src/llmops_sdk/api/pricing.py` for the full reasoning.
- **Fact retention dropped from 82% (naive, full uncompressed context) to 55% (full
  pipeline, compressed context)** — a real cost: compression trades some answer
  fidelity for the cost/latency win. This is the honest number, not the best case from
  `reports/compression_quality.md`'s controlled sweep — a live pipeline run with only 5
  context-bearing queries is a small, noisy sample, and this run used a different
  (weaker, cheaper) model than the compression eval's context-survival check. Worth
  citing both numbers together if asked, not just the more flattering one.
- Naive baseline had 19/20 completed calls (1 transient failure after 3 retries) —
  reported honestly as n=19, not padded to 20.