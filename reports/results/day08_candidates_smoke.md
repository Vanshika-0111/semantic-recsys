# Day 8 — Stage-1 candidates (popular ∪ FAISS) + ranker features

Split fingerprint `a99079d4a40a…`. FAISS = exact Flat search over `bge-small` mean-centered vectors with the Day 5 profile ({'weighting': 'graded', 'half_life': 5}); popularity mode `liked`. Seen movies are removed from both sources. Relevant = rating ≥ 4. Recall@size = share of a user's relevant held-out movies that are in the candidate set.

**Selected for Day 9:** popular top-400 ∪ FAISS top-100 (highest val recall with FAISS in the union, size <= 500).

## Test — fit on train+val, 5,000 users

| candidates | recall | hit_rate |
|---|---:|---:|
| union: pop 400 + faiss 100 | 0.5216 | 0.8678 |
| popularity top-500 (same size) | 0.5355 | 0.8598 |
| popularity top-400 (union without FAISS) | 0.4871 | 0.8280 |
| FAISS top-100 alone | 0.0696 | 0.3020 |

Retrieval (both sources + union): 1.16 ms per user, batched.

## Candidate recall — validation, 3,000 users (fit on train)

| candidates | size | avg_unique | recall | hit_rate | recall_gain_from_faiss | pop_same_size_recall | vs_pop_same_size |
|---|---:|---:|---:|---:|---:|---:|---:|
| pop 100 | 100 | 100.0000 | 0.2498 | 0.6070 | 0.0000 | 0.2498 | 0.0000 |
| pop 100 + faiss 50 | 150 | 148.6000 | 0.2862 | 0.6747 | 0.0364 | 0.3068 | -0.0206 |
| pop 100 + faiss 100 | 200 | 197.6000 | 0.3039 | 0.7087 | 0.0541 | 0.3605 | -0.0567 |
| pop 100 + faiss 200 | 300 | 295.8000 | 0.3318 | 0.7453 | 0.0820 | 0.4411 | -0.1093 |
| pop 100 + faiss 300 | 400 | 394.1000 | 0.3524 | 0.7733 | 0.1027 | 0.5017 | -0.1492 |
| pop 200 | 200 | 200.0000 | 0.3605 | 0.7293 | 0.0000 | 0.3605 | 0.0000 |
| pop 200 + faiss 50 | 250 | 247.6000 | 0.3906 | 0.7717 | 0.0300 | 0.4014 | -0.0108 |
| pop 200 + faiss 100 | 300 | 295.8000 | 0.4053 | 0.7963 | 0.0448 | 0.4411 | -0.0358 |
| pop 200 + faiss 200 | 400 | 392.6000 | 0.4285 | 0.8193 | 0.0680 | 0.5017 | -0.0732 |
| pop 200 + faiss 300 | 500 | 489.7000 | 0.4457 | 0.8353 | 0.0851 | 0.5491 | -0.1034 |
| pop 300 | 300 | 300.0000 | 0.4411 | 0.7967 | 0.0000 | 0.4411 | 0.0000 |
| pop 300 + faiss 50 | 350 | 346.6000 | 0.4671 | 0.8270 | 0.0260 | 0.4730 | -0.0059 |
| pop 300 + faiss 100 | 400 | 394.2000 | 0.4804 | 0.8450 | 0.0393 | 0.5017 | -0.0213 |
| pop 300 + faiss 200 | 500 | 489.8000 | 0.5014 | 0.8623 | 0.0603 | 0.5491 | -0.0477 |
| pop 300 + faiss 300 | 600 | 585.9000 | 0.5164 | 0.8757 | 0.0753 | 0.5902 | -0.0739 |
| pop 400 | 400 | 400.0000 | 0.5017 | 0.8357 | 0.0000 | 0.5017 | 0.0000 |
| pop 400 + faiss 50 | 450 | 445.8000 | 0.5252 | 0.8587 | 0.0235 | 0.5277 | -0.0025 |
| pop 400 + faiss 100 | 500 | 492.6000 | 0.5371 | 0.8727 | 0.0355 | 0.5491 | -0.0120 |
| pop 400 + faiss 200 | 600 | 587.0000 | 0.5560 | 0.8850 | 0.0543 | 0.5902 | -0.0342 |
| pop 400 + faiss 300 | 700 | 682.0000 | 0.5689 | 0.8960 | 0.0672 | 0.6255 | -0.0566 |
| pop 500 | 500 | 500.0000 | 0.5491 | 0.8617 | 0.0000 | 0.5491 | 0.0000 |
| pop 500 + faiss 50 | 550 | 545.0000 | 0.5706 | 0.8810 | 0.0215 | 0.5709 | -0.0002 |
| pop 500 + faiss 100 | 600 | 591.2000 | 0.5818 | 0.8930 | 0.0327 | 0.5902 | -0.0085 |
| pop 500 + faiss 200 | 700 | 684.6000 | 0.5995 | 0.9033 | 0.0503 | 0.6255 | -0.0260 |
| pop 500 + faiss 300 | 800 | 778.5000 | 0.6112 | 0.9127 | 0.0621 | 0.6532 | -0.0420 |

FAISS alone: top-50 0.0491, top-100 0.0725, top-200 0.1084, top-300 0.1362.

Of the relevant movies inside the selected union, 6.6% were found only by FAISS. Best popularity-only set ≤ 500: pop 500 (recall 0.5491).

## Ranker features — 2,000 validation users, history = train, labels = val

985,100 (user, candidate) rows, 6,695 relevant (0.68%), 492.6 candidates per user; `data\runs\day08_features_val_smoke.parquet` (29.8 MB); retrieval + features 2.24 ms per user.

Check: ranking the candidates by `pop_rank` reproduces the popularity baseline for 100.00% of users.

### Each feature alone as the ranker (NDCG@10 on these users)

| feature | ndcg@10 | recall@10 | mrr@10 |
|---|---:|---:|---:|
| (oracle) | 0.6264 | 0.5450 | 0.8700 |
| in_pop | 0.0483 | 0.0541 | 0.0924 |
| pop_rank | 0.0483 | 0.0541 | 0.0924 |
| item_liked_log | 0.0483 | 0.0541 | 0.0924 |
| (union order) | 0.0483 | 0.0541 | 0.0924 |
| item_count_log | 0.0479 | 0.0537 | 0.0918 |
| in_faiss | 0.0381 | 0.0381 | 0.0908 |
| item_recent_log | 0.0368 | 0.0397 | 0.0757 |
| year_gap | 0.0309 | 0.0335 | 0.0671 |
| item_avg_rating | 0.0287 | 0.0305 | 0.0658 |
| max_cos_liked | 0.0256 | 0.0273 | 0.0566 |
| item_liked_share | 0.0245 | 0.0264 | 0.0559 |
| cos_longterm | 0.0166 | 0.0183 | 0.0381 |
| genre_affinity | 0.0157 | 0.0168 | 0.0364 |
| faiss_rank | 0.0150 | 0.0159 | 0.0333 |
| cos_profile | 0.0150 | 0.0159 | 0.0333 |
| item_year | 0.0017 | 0.0020 | 0.0031 |
| item_trend | 0.0015 | 0.0025 | 0.0021 |

`(oracle)` = relevant movies first: the ceiling for any ranker on these candidates. `(union order)` = popular first, then FAISS-only.

### Feature means: relevant vs other rows

| feature | mean_relevant | mean_other | missing_share | meaning |
|---|---:|---:|---:|---|
| in_pop | 0.9428 | 0.8112 | 0.0000 | retrieved by the popularity source (1/0) |
| pop_rank | 128.8706 | 200.0617 | 0.1879 | position in the user's popularity list (NaN = not retrieved by it) |
| in_faiss | 0.1207 | 0.2033 | 0.0000 | retrieved by FAISS (1/0) |
| faiss_rank | 38.1052 | 49.5463 | 0.7973 | position in the user's FAISS list (NaN = not retrieved by it) |
| item_liked_log | 9.6099 | 8.4866 | 0.0000 | log(1 + #ratings >= 4) in the history |
| item_count_log | 10.1085 | 9.1562 | 0.0000 | log(1 + #ratings) in the history |
| item_avg_rating | 3.8324 | 3.6727 | 0.0000 | Bayesian average rating (20 pseudo-votes at the global mean) |
| item_liked_share | 0.6246 | 0.5543 | 0.0000 | #ratings >= 4 / #ratings (smoothed like the average) |
| item_recent_log | 5.7115 | 4.7124 | 0.0000 | log(1 + #ratings >= 4 in the last 365 days of the history) |
| item_trend | 0.0260 | 0.0307 | 0.0000 | share of the movie's liked ratings that fall in those 365 days |
| item_year | 1994.6109 | 1994.4443 | 0.0010 | release year (newer first in the one-feature check) |
| item_runtime | 123.3652 | 118.5411 | 0.0001 | runtime in minutes (TMDB; NaN if missing) |
| user_n_ratings_log | 4.0303 | 4.3215 | 0.0000 | log(1 + #ratings of the user) |
| user_mean_rating | 3.8615 | 3.7270 | 0.0000 | the user's mean rating |
| user_liked_share | 0.6482 | 0.5890 | 0.0000 | share of the user's ratings that are >= 4 |
| cos_profile | 0.1376 | 0.1024 | 0.0012 | cos(Day 5 profile = graded, half-life 5, mean-centered; movie) |
| cos_longterm | 0.1567 | 0.1085 | 0.0012 | cos(all liked movies, no decay; movie) — long-term taste |
| max_cos_liked | 0.2755 | 0.2447 | 0.0012 | max cos between the movie and any of the user's last 50 liked |
| genre_affinity | 0.3111 | 0.2819 | 0.0000 | Σ_g (share of the user's liked movies in g) over the movie's genres, / #genres |
| year_gap | 8.9285 | 13.9757 | 0.0022 | |release year - mean year of the user's liked movies| |
| pop_gap | 0.4365 | -0.4890 | 0.0012 | item_liked_log - mean item_liked_log of the user's liked movies |
