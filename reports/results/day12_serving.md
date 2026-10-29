# Day 12 — Serving: FastAPI + bundle + latency

## Release filter — validation (5,000 Day 9 ranker-valid users)

Rule (written before the run): on iff ranker+filter > ranker on val NDCG@10, bootstrap CI lower bound > 0.

| model | ndcg@10 | recall@10 | mrr@10 | late_share@10 |
|---|---:|---:|---:|---:|
| ranker | 0.0909 | 0.1001 | 0.1765 | 0.0453 |
| ranker + release filter | 0.0930 | 0.1025 | 0.1803 | 0.0000 |
| ranker + MMR (served) | 0.0892 | 0.0981 | 0.1751 | 0.0431 |
| ranker + MMR + release filter | 0.0917 | 0.1015 | 0.1785 | 0.0000 |

Filter vs ranker: +2.3% (CI [+0.0017, +0.0026], p = 0.000); with the served MMR +2.8% (p = 0.000).

**Decision: release filter ON** for demo users. Checks: ranker = Day 9 val True, MMR = Day 11 val True.

`late_share@10` = share of top-10 slots released after the user's cutoff year.

## Bundle — `data/serving` (46.9 MB, 23,109 movies, 1,000 demo users)

History: train+val (28,932,244 ratings) — the Day 10 test setting. Settings: MMR {'space': 'genre', 'lambda': 0.7, 'pool': 50}, policy warm ranker from 3 picks, release filter ON, explore {'pool': 1000, 'lambda': 0.3}.

| file | MB |
|---|---:|
| arrays.npz | 37.82 |
| bundle.json | 0.0 |
| cold_ranker.txt | 0.85 |
| demo_history.parquet | 1.12 |
| demo_truth.parquet | 0.04 |
| items.parquet | 5.96 |
| warm_ranker.txt | 1.13 |

## Golden check — API engine (bundle only) vs the evaluated batch code

- Warm users: identical top-10 for 200/200 (ranker) and 200/200 (ranker + MMR)
- Real cold users, 1 pick(s) (cold ranker): identical for 200/200
- Real cold users, 3 pick(s) (warm ranker): identical for 200/200
- Real cold users, 5 pick(s) (warm ranker): identical for 200/200
- **All identical: ✓**

Demo users (sanity only — 1,000 users, noisy; the CV numbers are Day 10/11):

| model | ndcg@10 | recall@10 | n_users |
|---|---:|---:|---:|
| popularity | 0.0554 | 0.0582 | 1,000 |
| ranker | 0.0892 | 0.0938 | 1,000 |
| served (MMR + release filter) | 0.0912 | 0.0980 | 1,000 |

## Latency — 300 sequential requests per endpoint, after 10 warm-up

Target: in-process (FastAPI TestClient: routing + validation + JSON, no network).

| endpoint | p50 | p95 | p99 | mean | engine_p50 |
|---|---:|---:|---:|---:|---:|
| warm user — served (ranker + MMR + explanations) | 88.9 | 103.7 | 109.8 | 88.1 | 77.2 |
| warm user — ranker only | 43.2 | 50.9 | 55.6 | 42.5 | 30.3 |
| cold start — 0 picks (popularity) | 14.4 | 19.7 | 24.3 | 14.8 | 3.1 |
| cold start — 1 pick (cold ranker) | 76.7 | 102.2 | 159.1 | 78.3 | 63.4 |
| cold start — 3 picks (warm ranker) | 88.5 | 105.7 | 115.8 | 87.9 | 75.3 |
| cold start — 5 picks (warm ranker) | 91.7 | 102.2 | 127.3 | 90.2 | 77.1 |
| free text — explore mode | 93.0 | 110.5 | 132.4 | 92.3 | 80.7 |
| title search | 18.0 | 25.3 | 29.3 | 18.6 | 7.8 |

All times in ms. `engine_p50` = time inside the engine (the rest is HTTP, validation, JSON).

Warm served request, mean ms per step: stage1 8.13, features 12.31, ranker 4.65, mmr 10.92, explain 35.8, format 0.6.

engine load 0.71 s; engine adds ≈ 236.4 MB of memory (process 379.4 MB); first text request (loads the encoder) 67.94 s.

one request at a time on this machine; latencies vary run to run — quote as approximate.
## Latency — 300 sequential requests per endpoint, after 10 warm-up

Target: http://localhost:8000.

| endpoint | p50 | p95 | p99 | mean | engine_p50 |
|---|---:|---:|---:|---:|---:|
| warm user — served (ranker + MMR + explanations) | 44.4 | 72.8 | 112.8 | 49.8 | 41.4 |
| warm user — ranker only | 19.0 | 24.6 | 40.2 | 20.2 | 16.0 |
| cold start — 0 picks (popularity) | 4.2 | 7.3 | 10.0 | 4.6 | 1.3 |
| cold start — 1 pick (cold ranker) | 38.1 | 46.4 | 60.0 | 38.9 | 34.6 |
| cold start — 3 picks (warm ranker) | 43.6 | 56.6 | 88.0 | 49.0 | 40.2 |
| cold start — 5 picks (warm ranker) | 43.6 | 55.8 | 102.8 | 46.6 | 40.2 |
| free text — explore mode | 35.2 | 45.5 | 133.4 | 38.2 | 32.3 |
| title search | 5.6 | 6.8 | 13.8 | 6.0 | 3.1 |

All times in ms. `engine_p50` = time inside the engine (the rest is HTTP, validation, JSON).

first text request (loads the encoder) 0.52 s.

one request at a time on this machine; latencies vary run to run — quote as approximate.
