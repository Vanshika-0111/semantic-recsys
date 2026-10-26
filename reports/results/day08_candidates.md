# Day 8 — Stage-1 candidates (popular ∪ FAISS) + ranker features

Split fingerprint `a99079d4a40a…`. FAISS = exact Flat search over `bge-small` mean-centered vectors with the Day 5 profile ({'weighting': 'graded', 'half_life': 5}); popularity mode `liked`. Seen movies are removed from both sources. Relevant = rating ≥ 4. Recall@size = share of a user's relevant held-out movies that are in the candidate set.

**Selected for Day 9:** popular top-400 ∪ FAISS top-100 (highest val recall with FAISS in the union, size <= 500).

## Test — fit on train+val, 189,197 users

| candidates | recall | hit_rate |
|---|---:|---:|
| union: pop 400 + faiss 100 | 0.5188 | 0.8663 |
| popularity top-500 (same size) | 0.5306 | 0.8565 |
| popularity top-400 (union without FAISS) | 0.4852 | 0.8284 |
| FAISS top-100 alone | 0.0682 | 0.2977 |

Retrieval (both sources + union): 0.91 ms per user, batched.

## Candidate recall — validation, 30,000 users (fit on train)

| candidates | size | avg_unique | recall | hit_rate | recall_gain_from_faiss | pop_same_size_recall | vs_pop_same_size |
|---|---:|---:|---:|---:|---:|---:|---:|
| pop 100 | 100 | 100.0000 | 0.2497 | 0.5999 | 0.0000 | 0.2497 | 0.0000 |
| pop 100 + faiss 50 | 150 | 148.7000 | 0.2841 | 0.6733 | 0.0344 | 0.3093 | -0.0253 |
| pop 100 + faiss 100 | 200 | 197.7000 | 0.3025 | 0.7056 | 0.0529 | 0.3608 | -0.0583 |
| pop 100 + faiss 200 | 300 | 296.1000 | 0.3302 | 0.7467 | 0.0805 | 0.4414 | -0.1112 |
| pop 100 + faiss 300 | 400 | 394.6000 | 0.3519 | 0.7727 | 0.1022 | 0.5005 | -0.1486 |
| pop 200 | 200 | 200.0000 | 0.3608 | 0.7233 | 0.0000 | 0.3608 | 0.0000 |
| pop 200 + faiss 50 | 250 | 247.7000 | 0.3890 | 0.7695 | 0.0282 | 0.4023 | -0.0133 |
| pop 200 + faiss 100 | 300 | 295.9000 | 0.4045 | 0.7907 | 0.0437 | 0.4414 | -0.0368 |
| pop 200 + faiss 200 | 400 | 393.0000 | 0.4280 | 0.8186 | 0.0672 | 0.5005 | -0.0725 |
| pop 200 + faiss 300 | 500 | 490.3000 | 0.4468 | 0.8358 | 0.0860 | 0.5475 | -0.1007 |
| pop 300 | 300 | 300.0000 | 0.4414 | 0.7860 | 0.0000 | 0.4414 | 0.0000 |
| pop 300 + faiss 50 | 350 | 346.7000 | 0.4655 | 0.8203 | 0.0241 | 0.4737 | -0.0082 |
| pop 300 + faiss 100 | 400 | 394.3000 | 0.4791 | 0.8365 | 0.0377 | 0.5005 | -0.0214 |
| pop 300 + faiss 200 | 500 | 490.1000 | 0.4999 | 0.8577 | 0.0585 | 0.5475 | -0.0476 |
| pop 300 + faiss 300 | 600 | 586.3000 | 0.5165 | 0.8708 | 0.0751 | 0.5872 | -0.0707 |
| pop 400 | 400 | 400.0000 | 0.5005 | 0.8277 | 0.0000 | 0.5005 | 0.0000 |
| pop 400 + faiss 50 | 450 | 445.9000 | 0.5221 | 0.8530 | 0.0216 | 0.5262 | -0.0041 |
| pop 400 + faiss 100 | 500 | 492.8000 | 0.5344 | 0.8655 | 0.0339 | 0.5475 | -0.0131 |
| pop 400 + faiss 200 | 600 | 587.4000 | 0.5530 | 0.8820 | 0.0524 | 0.5872 | -0.0342 |
| pop 400 + faiss 300 | 700 | 682.5000 | 0.5680 | 0.8925 | 0.0675 | 0.6202 | -0.0521 |
| pop 500 | 500 | 500.0000 | 0.5475 | 0.8557 | 0.0000 | 0.5475 | 0.0000 |
| pop 500 + faiss 50 | 550 | 545.1000 | 0.5667 | 0.8762 | 0.0193 | 0.5680 | -0.0012 |
| pop 500 + faiss 100 | 600 | 591.4000 | 0.5778 | 0.8867 | 0.0304 | 0.5872 | -0.0094 |
| pop 500 + faiss 200 | 700 | 684.9000 | 0.5947 | 0.9004 | 0.0472 | 0.6202 | -0.0255 |
| pop 500 + faiss 300 | 800 | 779.1000 | 0.6083 | 0.9089 | 0.0608 | 0.6491 | -0.0408 |

FAISS alone: top-50 0.0458, top-100 0.0702, top-200 0.1064, top-300 0.1349.

Of the relevant movies inside the selected union, 6.3% were found only by FAISS. Best popularity-only set ≤ 500: pop 500 (recall 0.5475).

## Ranker features — 20,000 validation users, history = train, labels = val

9,854,146 (user, candidate) rows, 64,946 relevant (0.66%), 492.7 candidates per user; `data\runs\day08_features_val.parquet` (299.4 MB); retrieval + features 1.87 ms per user.

Check: ranking the candidates by `pop_rank` reproduces the popularity baseline for 100.00% of users.

### Each feature alone as the ranker (NDCG@10 on these users)

| feature | ndcg@10 | recall@10 | mrr@10 |
|---|---:|---:|---:|
| (oracle) | 0.6181 | 0.5349 | 0.8693 |
| in_pop | 0.0482 | 0.0544 | 0.0945 |
| pop_rank | 0.0482 | 0.0544 | 0.0945 |
| item_liked_log | 0.0482 | 0.0544 | 0.0945 |
| (union order) | 0.0482 | 0.0544 | 0.0945 |
| item_count_log | 0.0477 | 0.0529 | 0.0953 |
| item_recent_log | 0.0393 | 0.0436 | 0.0800 |
| in_faiss | 0.0392 | 0.0388 | 0.0914 |
| max_cos_liked | 0.0289 | 0.0318 | 0.0622 |
| item_avg_rating | 0.0284 | 0.0313 | 0.0610 |
| year_gap | 0.0277 | 0.0293 | 0.0628 |
| item_liked_share | 0.0249 | 0.0273 | 0.0539 |
| cos_longterm | 0.0175 | 0.0199 | 0.0387 |
| faiss_rank | 0.0170 | 0.0175 | 0.0395 |
| cos_profile | 0.0170 | 0.0175 | 0.0395 |
| genre_affinity | 0.0165 | 0.0181 | 0.0381 |
| item_year | 0.0012 | 0.0019 | 0.0020 |
| item_trend | 0.0007 | 0.0012 | 0.0011 |

`(oracle)` = relevant movies first: the ceiling for any ranker on these candidates. `(union order)` = popular first, then FAISS-only.

### Feature means: relevant vs other rows

| feature | mean_relevant | mean_other | missing_share | meaning |
|---|---:|---:|---:|---|
| in_pop | 0.9391 | 0.8110 | 0.0000 | retrieved by the popularity source (1/0) |
| pop_rank | 130.2037 | 200.0323 | 0.1882 | position in the user's popularity list (NaN = not retrieved by it) |
| in_faiss | 0.1318 | 0.2031 | 0.0000 | retrieved by FAISS (1/0) |
| faiss_rank | 36.9191 | 49.5542 | 0.7974 | position in the user's FAISS list (NaN = not retrieved by it) |
| item_liked_log | 9.5978 | 8.4829 | 0.0000 | log(1 + #ratings >= 4) in the history |
| item_count_log | 10.0916 | 9.1518 | 0.0000 | log(1 + #ratings) in the history |
| item_avg_rating | 3.8392 | 3.6731 | 0.0000 | Bayesian average rating (20 pseudo-votes at the global mean) |
| item_liked_share | 0.6269 | 0.5543 | 0.0000 | #ratings >= 4 / #ratings (smoothed like the average) |
| item_recent_log | 5.7361 | 4.7085 | 0.0000 | log(1 + #ratings >= 4 in the last 365 days of the history) |
| item_trend | 0.0268 | 0.0310 | 0.0000 | share of the movie's liked ratings that fall in those 365 days |
| item_year | 1994.9537 | 1994.4669 | 0.0010 | release year (newer first in the one-feature check) |
| item_runtime | 124.0971 | 118.5257 | 0.0000 | runtime in minutes (TMDB; NaN if missing) |
| user_n_ratings_log | 4.0606 | 4.3488 | 0.0000 | log(1 + #ratings of the user) |
| user_mean_rating | 3.8445 | 3.7179 | 0.0000 | the user's mean rating |
| user_liked_share | 0.6388 | 0.5831 | 0.0000 | share of the user's ratings that are >= 4 |
| cos_profile | 0.1392 | 0.1012 | 0.0015 | cos(Day 5 profile = graded, half-life 5, mean-centered; movie) |
| cos_longterm | 0.1577 | 0.1084 | 0.0015 | cos(all liked movies, no decay; movie) — long-term taste |
| max_cos_liked | 0.2781 | 0.2445 | 0.0015 | max cos between the movie and any of the user's last 50 liked |
| genre_affinity | 0.3109 | 0.2813 | 0.0000 | Σ_g (share of the user's liked movies in g) over the movie's genres, / #genres |
| year_gap | 9.2618 | 14.0144 | 0.0025 | |release year - mean year of the user's liked movies| |
| pop_gap | 0.4457 | -0.4698 | 0.0015 | item_liked_log - mean item_liked_log of the user's liked movies |
