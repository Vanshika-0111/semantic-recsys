# Day 6 — Cold start (onboarding seeds and/or text)

Split fingerprint `a99079d4a40a…`. Relevant = rating ≥ 4. Model: pool = M most-liked movies in the W days before onboarding, ranked by cos(profile, movie) + λ·prior. W, M, λ tuned on pseudo-cold validation.

## Cold test — 5,000 users never seen in training (fit on train+val), window all-time

| setting | model | config | ndcg@5 | ndcg@10 | ndcg@20 | recall@10 | mrr@10 | hit_rate@10 |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| seeds_0 | popularity | all-time (Day 3 baseline) | 0.0247 | 0.0279 | 0.0414 | 0.0313 | 0.0578 | 0.1590 |
| seeds_1 | popularity | all-time (Day 3 baseline) | 0.0254 | 0.0288 | 0.0422 | 0.0321 | 0.0603 | 0.1622 |
| seeds_1 | embeddings only | pool=all, λ=0 | 0.0047 | 0.0045 | 0.0055 | 0.0046 | 0.0106 | 0.0238 |
| seeds_1 | semantic rerank (ours) | pool=20, λ=2 | 0.0242 | 0.0295 | 0.0420 | 0.0344 | 0.0584 | 0.1668 |
| seeds_3 | popularity | all-time (Day 3 baseline) | 0.0265 | 0.0297 | 0.0433 | 0.0330 | 0.0619 | 0.1644 |
| seeds_3 | embeddings only | pool=all, λ=0 | 0.0065 | 0.0066 | 0.0085 | 0.0073 | 0.0152 | 0.0410 |
| seeds_3 | semantic rerank (ours) | pool=20, λ=5 | 0.0268 | 0.0311 | 0.0437 | 0.0355 | 0.0630 | 0.1736 |
| seeds_5 | popularity | all-time (Day 3 baseline) | 0.0274 | 0.0305 | 0.0443 | 0.0337 | 0.0636 | 0.1658 |
| seeds_5 | embeddings only | pool=all, λ=0 | 0.0082 | 0.0087 | 0.0109 | 0.0092 | 0.0206 | 0.0544 |
| seeds_5 | semantic rerank (ours) | pool=20, λ=5 | 0.0278 | 0.0322 | 0.0448 | 0.0369 | 0.0654 | 0.1780 |
| text | text only | pool=all, λ=0 | 0.0006 | 0.0009 | 0.0014 | 0.0012 | 0.0017 | 0.0070 |
| text | semantic rerank (ours) | pool=20, λ=1000 | 0.0247 | 0.0279 | 0.0414 | 0.0313 | 0.0578 | 0.1590 |
| text | popular in your genres | pool=20, λ=1000 | 0.0247 | 0.0279 | 0.0414 | 0.0313 | 0.0578 | 0.1590 |
| text+seeds_1 | semantic rerank (ours) | pool=20, λ=5 | 0.0252 | 0.0295 | 0.0422 | 0.0339 | 0.0593 | 0.1660 |
| text+seeds_3 | semantic rerank (ours) | pool=20, λ=5 | 0.0265 | 0.0308 | 0.0436 | 0.0353 | 0.0626 | 0.1730 |
| text+seeds_5 | semantic rerank (ours) | pool=20, λ=5 | 0.0279 | 0.0323 | 0.0449 | 0.0367 | 0.0660 | 0.1782 |

| setting | model | coverage@10 | long_tail@10 | diversity@10 |
|---:|---|---:|---:|---:|
| seeds_0 | popularity | 0.0004 | 0.0000 | 0.7406 |
| seeds_1 | popularity | 0.0005 | 0.0000 | 0.7429 |
| seeds_1 | embeddings only | 0.3158 | 0.7570 | 0.5304 |
| seeds_1 | semantic rerank (ours) | 0.0009 | 0.0000 | 0.6850 |
| seeds_3 | popularity | 0.0006 | 0.0000 | 0.7406 |
| seeds_3 | embeddings only | 0.4245 | 0.6220 | 0.5042 |
| seeds_3 | semantic rerank (ours) | 0.0010 | 0.0000 | 0.6903 |
| seeds_5 | popularity | 0.0006 | 0.0000 | 0.7376 |
| seeds_5 | embeddings only | 0.4041 | 0.5737 | 0.4926 |
| seeds_5 | semantic rerank (ours) | 0.0011 | 0.0000 | 0.6905 |
| text | text only | 0.0530 | 0.8489 | 0.4779 |
| text | semantic rerank (ours) | 0.0004 | 0.0000 | 0.7406 |
| text | popular in your genres | 0.0005 | 0.0000 | 0.7405 |
| text+seeds_1 | semantic rerank (ours) | 0.0009 | 0.0000 | 0.6706 |
| text+seeds_3 | semantic rerank (ours) | 0.0010 | 0.0000 | 0.6716 |
| text+seeds_5 | semantic rerank (ours) | 0.0011 | 0.0000 | 0.6728 |

TF-IDF (Day 3, same users and seeds), NDCG@10: 0 seeds 0.0279, 1 seeds 0.0038, 3 seeds 0.0063, 5 seeds 0.0084

### Paired bootstrap (NDCG@10, 2,000 resamples of users)

| model | setting | vs | NDCG@10 | baseline | change | 95% CI of diff | p(≤0) |
|---|---|---|---:|---:|---:|---|---:|
| semantic rerank (ours) | seeds_1 | popularity (seeds_1) | 0.0295 | 0.0288 | +2.2% | [-0.0011, +0.0024] | 0.230 |
| semantic rerank (ours) | seeds_3 | popularity (seeds_3) | 0.0311 | 0.0297 | +4.5% | [+0.0000, +0.0026] | 0.024 |
| semantic rerank (ours) | seeds_5 | popularity (seeds_5) | 0.0322 | 0.0305 | +5.7% | [+0.0004, +0.0031] | 0.005 |
| semantic rerank (ours) | text | popularity (seeds_0) | 0.0279 | 0.0279 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| popular in your genres | text | popularity (seeds_0) | 0.0279 | 0.0279 | +0.0% | [+0.0000, +0.0000] | 1.000 |
| semantic rerank (ours) | text+seeds_1 | popularity (seeds_1) | 0.0295 | 0.0288 | +2.1% | [-0.0006, +0.0020] | 0.174 |
| semantic rerank (ours) | text+seeds_3 | popularity (seeds_3) | 0.0308 | 0.0297 | +3.5% | [-0.0004, +0.0024] | 0.072 |
| semantic rerank (ours) | text+seeds_5 | popularity (seeds_5) | 0.0323 | 0.0305 | +5.7% | [+0.0003, +0.0032] | 0.008 |

## One onboarding request (1 at a time, CPU)

| path | p50 ms | p95 ms |
|---|---:|---:|
| text (encode + pool + rerank) | 41.82 | 67.92 |
| 5 seeds (profile + pool + rerank) | 4.03 | 6.36 |

## Tuning — pseudo-cold validation (warm val users, first n liked train movies only; fit on train), 10,000 users

Time window: **all-time**

| model | config | ndcg@10 | recall@10 |
|---|---|---:|---:|
| popularity (all-time) | train, incl. ratings after onboarding | 0.0273 | 0.0318 |
| time-aware popularity | all-time | 0.0273 | 0.0318 |
| time-aware popularity | 30 days | 0.0268 | 0.0330 |
| time-aware popularity | 180 days | 0.0266 | 0.0330 |
| time-aware popularity | 730 days | 0.0266 | 0.0324 |
| time-aware popularity | all before onboarding | 0.0237 | 0.0283 |

| setting | best | NDCG@10 |
|---|---|---:|
| seeds_1 | pool=20, λ=2 | 0.0286 |
| seeds_3 | pool=20, λ=5 | 0.0303 |
| seeds_5 | pool=20, λ=5 | 0.0316 |
| text | prefix, pool=20, λ=1000 | 0.0273 |
| genre_pop | pool=20, λ=1000 | 0.0273 |
| text+seeds_1 | pool=20, λ=5 | 0.0282 |
| text+seeds_3 | pool=20, λ=5 | 0.0301 |
| text+seeds_5 | pool=20, λ=5 | 0.0311 |

Example questionnaire texts: *Action, adventure and animation movies*; *Action, adventure and children's movies*; *Action, adventure and comedy movies*; *Action, adventure and crime movies*; *Action, adventure and drama movies*

Selected: `{"model": "bge-small", "center": true, "weighting": "graded", "half_life": 5, "query_prefix": true, "time_window_days": "all-time", "blend": {"seeds_1": {"pool": 20, "lambda": 2.0}, "seeds_3": {"pool": 20, "lambda": 5.0}, "seeds_5": {"pool": 20, "lambda": 5.0}, "text": {"pool": 20, "lambda": 1000.0}, "text+seeds_1": {"pool": 20, "lambda": 5.0}, "text+seeds_3": {"pool": 20, "lambda": 5.0}, "text+seeds_5": {"pool": 20, "lambda": 5.0}}, "genre_pop": {"pool": 20, "lambda": 1000.0}}` (full grid in the json)
