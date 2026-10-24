# Day 3 — Baselines

Split fingerprint `a99079d4a40a…`. Relevant = rating ≥ 4. Already-rated movies are never recommended.

## Test — fit on train+val, 5,000 users

| model | config | ndcg@5 | ndcg@10 | ndcg@20 | recall@10 | precision@10 | mrr@10 | hit_rate@10 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| popularity | mode=liked | 0.0457 | 0.0502 | 0.0683 | 0.0555 | 0.0350 | 0.1003 | 0.2374 |
| tfidf_knn | weighting=liked, history=50 | 0.0183 | 0.0186 | 0.0219 | 0.0188 | 0.0114 | 0.0430 | 0.0950 |

| model | coverage@10 | long_tail@10 | diversity@10 | ms_per_user |
|---|---:|---:|---:|---:|
| popularity | 0.0077 | 0.0001 | 0.7132 | 0.1622 |
| tfidf_knn | 0.2233 | 0.5914 | 0.5638 | 0.3314 |

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

## Tuning — fit on train, scored on val (3,000 sampled users)

| model | config | ndcg@10 | recall@10 | mrr@10 |
|---|---|---:|---:|---:|
| popularity | mode=count | 0.0496 | 0.0543 | 0.1004 |
| popularity | mode=liked | 0.0504 | 0.0565 | 0.0996 |
| tfidf_knn | weighting=liked, history=all | 0.0165 | 0.0176 | 0.0374 |
| tfidf_knn | weighting=liked, history=50 | 0.0180 | 0.0183 | 0.0405 |
| tfidf_knn | weighting=centered, history=all | 0.0154 | 0.0153 | 0.0369 |
| tfidf_knn | weighting=centered, history=50 | 0.0155 | 0.0150 | 0.0385 |

Selected: `{"popularity": {"mode": "liked"}, "tfidf_knn": {"weighting": "liked", "max_history": 50}}`
