# Context compression quality eval

n = 5 (query, context, expected_facts) triples

Fact retention = fraction of expected fact-strings still present (case-insensitive substring match) in the compressed context, at each compression ratio.

| ratio | avg compression achieved | facts retained |
|---|---|---|
| target 50% | 40% | 82% (9/11) |
| target 33% | 26% | 64% (7/11) |