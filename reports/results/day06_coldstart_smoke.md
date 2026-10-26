# Day 6 — Cold start (onboarding seeds and/or text)

Split fingerprint `a99079d4a40a…`. Relevant = rating ≥ 4. Model: pool = M most-liked movies in the W days before onboarding, ranked by cos(profile, movie) + λ·prior. W, M, λ tuned on pseudo-cold validation.

## Cold test — 1,000 users never seen in training (fit on train+val), window 30 days

| setting | model | config | ndcg@5 | ndcg@10 | ndcg@20 | recall@10 | mrr@10 | hit_rate@10 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| seeds_0 | popularity | all-time (Day 3 baseline) | 0.0284 | 0.0305 | 0.0428 | 0.0326 | 0.0666 | 0.1710 |
| seeds_0 | time-aware popularity | 30 days | 0.0232 | 0.0304 | 0.0464 | 0.0375 | 0.0559 | 0.1750 |
| seeds_1 | popularity | all-time (Day 3 baseline) | 0.0294 | 0.0314 | 0.0437 | 0.0333 | 0.0689 | 0.1730 |
| seeds_1 | time-aware popularity | 30 days | 0.0247 | 0.0315 | 0.0479 | 0.0383 | 0.0586 | 0.1780 |
| seeds_1 | embeddings only | pool=all, λ=0 | 0.0046 | 0.0044 | 0.0057 | 0.0044 | 0.0115 | 0.0270 |
| seeds_1 | semantic rerank (ours) | pool=50, λ=0.5 | 0.0267 | 0.0310 | 0.0458 | 0.0367 | 0.0620 | 0.1870 |
| seeds_3 | popularity | all-time (Day 3 baseline) | 0.0304 | 0.0325 | 0.0448 | 0.0342 | 0.0715 | 0.1740 |
| seeds_3 | time-aware popularity | 30 days | 0.0278 | 0.0332 | 0.0510 | 0.0398 | 0.0622 | 0.1800 |
| seeds_3 | embeddings only | pool=all, λ=0 | 0.0054 | 0.0058 | 0.0081 | 0.0066 | 0.0130 | 0.0380 |
| seeds_3 | semantic rerank (ours) | pool=50, λ=1 | 0.0267 | 0.0313 | 0.0491 | 0.0366 | 0.0633 | 0.1780 |
| seeds_5 | popularity | all-time (Day 3 baseline) | 0.0314 | 0.0336 | 0.0459 | 0.0352 | 0.0737 | 0.1760 |
| seeds_5 | time-aware popularity | 30 days | 0.0293 | 0.0359 | 0.0535 | 0.0429 | 0.0664 | 0.1880 |
| seeds_5 | embeddings only | pool=all, λ=0 | 0.0089 | 0.0096 | 0.0115 | 0.0109 | 0.0213 | 0.0560 |
| seeds_5 | semantic rerank (ours) | pool=50, λ=0.5 | 0.0349 | 0.0377 | 0.0547 | 0.0417 | 0.0807 | 0.2070 |
| text | text only | pool=all, λ=0 | 0.0006 | 0.0005 | 0.0011 | 0.0006 | 0.0012 | 0.0050 |
| text | semantic rerank (ours) | pool=50, λ=0.1 | 0.0258 | 0.0293 | 0.0445 | 0.0349 | 0.0576 | 0.1610 |
| text | popular in your genres | pool=50, λ=0 | 0.0283 | 0.0317 | 0.0446 | 0.0347 | 0.0682 | 0.1820 |
| text+seeds_1 | semantic rerank (ours) | pool=50, λ=0.2 | 0.0273 | 0.0312 | 0.0473 | 0.0358 | 0.0632 | 0.1770 |
| text+seeds_3 | semantic rerank (ours) | pool=50, λ=0.5 | 0.0267 | 0.0329 | 0.0483 | 0.0395 | 0.0648 | 0.1930 |
| text+seeds_5 | semantic rerank (ours) | pool=50, λ=0.2 | 0.0343 | 0.0371 | 0.0542 | 0.0407 | 0.0800 | 0.2030 |

| setting | model | coverage@10 | long_tail@10 | diversity@10 |
|---:|---|---:|---:|---:|
| seeds_0 | popularity | 0.0004 | 0.0000 | 0.7406 |
| seeds_0 | time-aware popularity | 0.0108 | 0.0021 | 0.6800 |
| seeds_1 | popularity | 0.0005 | 0.0000 | 0.7431 |
| seeds_1 | time-aware popularity | 0.0109 | 0.0020 | 0.6832 |
| seeds_1 | embeddings only | 0.1433 | 0.7528 | 0.5326 |
| seeds_1 | semantic rerank (ours) | 0.0197 | 0.0008 | 0.6040 |
| seeds_3 | popularity | 0.0006 | 0.0000 | 0.7410 |
| seeds_3 | time-aware popularity | 0.0112 | 0.0025 | 0.6881 |
| seeds_3 | embeddings only | 0.1840 | 0.6302 | 0.5044 |
| seeds_3 | semantic rerank (ours) | 0.0186 | 0.0012 | 0.6070 |
| seeds_5 | popularity | 0.0006 | 0.0000 | 0.7375 |
| seeds_5 | time-aware popularity | 0.0118 | 0.0028 | 0.6898 |
| seeds_5 | embeddings only | 0.1762 | 0.5812 | 0.4960 |
| seeds_5 | semantic rerank (ours) | 0.0197 | 0.0013 | 0.5948 |
| text | text only | 0.0367 | 0.8516 | 0.4782 |
| text | semantic rerank (ours) | 0.0180 | 0.0007 | 0.5190 |
| text | popular in your genres | 0.0160 | 0.0006 | 0.4549 |
| text+seeds_1 | semantic rerank (ours) | 0.0193 | 0.0007 | 0.5519 |
| text+seeds_3 | semantic rerank (ours) | 0.0192 | 0.0010 | 0.5486 |
| text+seeds_5 | semantic rerank (ours) | 0.0198 | 0.0012 | 0.5475 |

TF-IDF (Day 3, same users and seeds), NDCG@10: 0 seeds 0.0279, 1 seeds 0.0038, 3 seeds 0.0063, 5 seeds 0.0084

### Paired bootstrap (NDCG@10, 2,000 resamples of users)

| model | setting | vs | NDCG@10 | baseline | change | 95% CI of diff | p(≤0) |
|---|---|---|---:|---:|---:|---|---:|
| semantic rerank (ours) | seeds_1 | popularity (seeds_1) | 0.0310 | 0.0314 | -1.2% | [-0.0060, +0.0050] | 0.558 |
| semantic rerank (ours) | seeds_1 | time-aware popularity (seeds_1) | 0.0310 | 0.0315 | -1.5% | [-0.0059, +0.0044] | 0.562 |
| semantic rerank (ours) | seeds_3 | popularity (seeds_3) | 0.0313 | 0.0325 | -3.8% | [-0.0074, +0.0047] | 0.651 |
| semantic rerank (ours) | seeds_3 | time-aware popularity (seeds_3) | 0.0313 | 0.0332 | -5.6% | [-0.0069, +0.0032] | 0.761 |
| semantic rerank (ours) | seeds_5 | popularity (seeds_5) | 0.0377 | 0.0336 | +12.0% | [-0.0024, +0.0105] | 0.112 |
| semantic rerank (ours) | seeds_5 | time-aware popularity (seeds_5) | 0.0377 | 0.0359 | +4.9% | [-0.0036, +0.0073] | 0.269 |
| semantic rerank (ours) | text | popularity (seeds_0) | 0.0293 | 0.0305 | -3.8% | [-0.0075, +0.0051] | 0.655 |
| semantic rerank (ours) | text | time-aware popularity (seeds_0) | 0.0293 | 0.0304 | -3.5% | [-0.0069, +0.0049] | 0.635 |
| popular in your genres | text | popularity (seeds_0) | 0.0317 | 0.0305 | +4.0% | [-0.0050, +0.0073] | 0.345 |
| popular in your genres | text | time-aware popularity (seeds_0) | 0.0317 | 0.0304 | +4.3% | [-0.0044, +0.0070] | 0.320 |
| semantic rerank (ours) | text+seeds_1 | popularity (seeds_1) | 0.0312 | 0.0314 | -0.6% | [-0.0060, +0.0054] | 0.514 |
| semantic rerank (ours) | text+seeds_1 | time-aware popularity (seeds_1) | 0.0312 | 0.0315 | -0.9% | [-0.0057, +0.0050] | 0.525 |
| semantic rerank (ours) | text+seeds_3 | popularity (seeds_3) | 0.0329 | 0.0325 | +1.2% | [-0.0058, +0.0065] | 0.438 |
| semantic rerank (ours) | text+seeds_3 | time-aware popularity (seeds_3) | 0.0329 | 0.0332 | -0.7% | [-0.0056, +0.0051] | 0.527 |
| semantic rerank (ours) | text+seeds_5 | popularity (seeds_5) | 0.0371 | 0.0336 | +10.2% | [-0.0032, +0.0102] | 0.150 |
| semantic rerank (ours) | text+seeds_5 | time-aware popularity (seeds_5) | 0.0371 | 0.0359 | +3.2% | [-0.0049, +0.0070] | 0.355 |

## One onboarding request (1 at a time, CPU)

| path | p50 ms | p95 ms |
|---|---:|---:|
| text (encode + pool + rerank) | 39.15 | 57.27 |
| 5 seeds (profile + pool + rerank) | 4.88 | 6.96 |

## Tuning — pseudo-cold validation (warm val users, first n liked train movies only; fit on train), 2,000 users

Time window: **30 days**

| model | config | ndcg@10 | recall@10 |
|---|---|---:|---:|
| popularity (all-time) | train, incl. ratings after onboarding | 0.0275 | 0.0332 |
| time-aware popularity | 30 days | 0.0270 | 0.0342 |
| time-aware popularity | 90 days | 0.0270 | 0.0341 |
| time-aware popularity | 180 days | 0.0267 | 0.0330 |
| time-aware popularity | 365 days | 0.0259 | 0.0325 |
| time-aware popularity | 730 days | 0.0259 | 0.0322 |
| time-aware popularity | 1825 days | 0.0259 | 0.0326 |
| time-aware popularity | all before onboarding | 0.0244 | 0.0289 |

| setting | best | NDCG@10 |
|---|---|---:|
| seeds_1 | pool=50, λ=0.5 | 0.0331 |
| seeds_3 | pool=50, λ=1 | 0.0374 |
| seeds_5 | pool=50, λ=0.5 | 0.0405 |
| text | prefix, pool=50, λ=0.1 | 0.0322 |
| genre_pop | pool=50, λ=0 | 0.0318 |
| text+seeds_1 | pool=50, λ=0.2 | 0.0349 |
| text+seeds_3 | pool=50, λ=0.5 | 0.0385 |
| text+seeds_5 | pool=50, λ=0.2 | 0.0417 |

Example questionnaire texts: *Action, adventure and animation movies*; *Action, adventure and comedy movies*; *Action, adventure and crime movies*; *Action, adventure and drama movies*; *Action, adventure and fantasy movies*

Selected: `{"model": "bge-small", "center": true, "weighting": "graded", "half_life": 5, "query_prefix": true, "time_window_days": 30, "blend": {"seeds_1": {"pool": 50, "lambda": 0.5}, "seeds_3": {"pool": 50, "lambda": 1.0}, "seeds_5": {"pool": 50, "lambda": 0.5}, "text": {"pool": 50, "lambda": 0.1}, "text+seeds_1": {"pool": 50, "lambda": 0.2}, "text+seeds_3": {"pool": 50, "lambda": 0.5}, "text+seeds_5": {"pool": 50, "lambda": 0.2}}, "genre_pop": {"pool": 50, "lambda": 0}}` (full grid in the json)
