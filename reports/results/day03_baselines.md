# Day 3 — Baselines

Split fingerprint `a99079d4a40a…`. Relevant = rating ≥ 4. Already-rated movies are never recommended.

## Test — fit on train+val, 189,197 users

| model | config | ndcg@5 | ndcg@10 | ndcg@20 | recall@10 | precision@10 | mrr@10 | hit_rate@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | mode=liked | 0.0461 | 0.0501 | 0.0674 | 0.0548 | 0.0345 | 0.1013 | 0.2351 |
| tfidf_knn | weighting=liked, history=50 | 0.0176 | 0.0178 | 0.0215 | 0.0181 | 0.0111 | 0.0410 | 0.0913 |

| model | coverage@10 | long_tail@10 | diversity@10 | ms_per_user |
|---|---:|---:|---:|---:|
| popularity | 0.0166 | 0.0000 | 0.7135 | 0.3895 |
| tfidf_knn | 0.6248 | 0.5922 | 0.5645 | 0.6412 |

long_tail@10 = share of recommended slots outside the 2,311 most-rated movies; diversity@10 = mean pairwise (1 − cosine) of genre vectors.

## Cold-start users — 5,000 users never seen in training

| model | n_seeds | ndcg@10 | recall@10 | mrr@10 | hit_rate@10 |
|---|---:|---:|---:|---:|---:|
| popularity | 0 | 0.0279 | 0.0313 | 0.0578 | 0.1590 |
| tfidf_knn | 0 | 0.0279 | 0.0313 | 0.0578 | 0.1590 |
| popularity | 1 | 0.0288 | 0.0321 | 0.0603 | 0.1622 |
| tfidf_knn | 1 | 0.0038 | 0.0040 | 0.0089 | 0.0206 |
| popularity | 3 | 0.0297 | 0.0330 | 0.0619 | 0.1644 |
| tfidf_knn | 3 | 0.0063 | 0.0064 | 0.0149 | 0.0344 |
| popularity | 5 | 0.0305 | 0.0337 | 0.0636 | 0.1658 |
| tfidf_knn | 5 | 0.0084 | 0.0085 | 0.0206 | 0.0464 |

0 seeds: TF-IDF has no profile and falls back to popularity.

## Tuning — fit on train, scored on val (30,000 sampled users)

| model | config | ndcg@10 | recall@10 | mrr@10 |
|---|---|---:|---:|---:|
| popularity | mode=count | 0.0480 | 0.0529 | 0.0967 |
| popularity | mode=liked | 0.0485 | 0.0543 | 0.0961 |
| tfidf_knn | weighting=liked, history=all | 0.0168 | 0.0176 | 0.0378 |
| tfidf_knn | weighting=liked, history=50 | 0.0175 | 0.0182 | 0.0395 |
| tfidf_knn | weighting=centered, history=all | 0.0156 | 0.0161 | 0.0361 |
| tfidf_knn | weighting=centered, history=50 | 0.0157 | 0.0159 | 0.0367 |

Selected: `{"popularity": {"mode": "liked"}, "tfidf_knn": {"weighting": "liked", "max_history": 50}}`
