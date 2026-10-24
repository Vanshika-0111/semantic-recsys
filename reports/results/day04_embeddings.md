# Day 4 — Sentence embeddings + FAISS Flat

Split fingerprint `a99079d4a40a…`. Item documents identical to Day 3.

## Encoding cost (one-off, whole catalog)

| model | dim | max_seq_length | tokens_median | tokens_p95 | tokens_max | truncated | seconds | docs_per_sec | device |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minilm | 384 | 256 | 107 | 178 | 297 | 0.11% | 1,184 | 20 | cpu |
| bge-small | 384 | 512 | 107 | 178 | 297 | 0.00% | 2,216 | 10 | cpu |

## Item neighbours (top-10) — metadata agreement and FAISS Flat latency

| model | genre_jaccard@10 | year_gap_median@10 | flat_p50_ms | flat_p95_ms |
|---|---:|---:|---:|---:|
| tfidf | 0.3875 | 11 |  |  |
| minilm | 0.3784 | 10 | 2.3601 | 4.9915 |
| bge-small | 0.4358 | 10 | 2.8250 | 5.2039 |

genre_jaccard = mean genre overlap between a movie and its neighbours (higher = more on-topic); year_gap = median |year difference|. Latency: one top-100 query at a time, 1,000 queries. Examples: `day04_similar_movies.md`.

## Validation — fit on train, 30,000 sampled users (same as Day 3)

| model | config | ndcg@10 | recall@10 | mrr@10 | coverage@10 | long_tail@10 | diversity@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| popularity | mode=liked | 0.0485 | 0.0543 | 0.0961 | 0.0117 | 0.0000 | 0.7084 |
| tfidf_knn | weighting=liked, history=50 | 0.0175 | 0.0182 | 0.0395 | 0.3916 | 0.5861 | 0.5628 |
| minilm | centroid, raw, history=all | 0.0043 | 0.0052 | 0.0091 | 0.1996 | 0.8179 | 0.6631 |
| minilm | centroid, raw, history=50 | 0.0045 | 0.0055 | 0.0095 | 0.2018 | 0.8174 | 0.6552 |
| minilm | centroid, mean-centered, history=all | 0.0117 | 0.0130 | 0.0264 | 0.5319 | 0.6530 | 0.5344 |
| minilm | centroid, mean-centered, history=50 | 0.0122 | 0.0136 | 0.0274 | 0.5927 | 0.6686 | 0.5353 |
| minilm | knn, raw, history=all | 0.0059 | 0.0071 | 0.0128 | 0.3011 | 0.7606 | 0.6021 |
| minilm | knn, raw, history=50 | 0.0061 | 0.0073 | 0.0134 | 0.3132 | 0.7579 | 0.6006 |
| minilm | knn, mean-centered, history=all | 0.0103 | 0.0113 | 0.0230 | 0.4510 | 0.6832 | 0.5403 |
| minilm | knn, mean-centered, history=50 | 0.0109 | 0.0119 | 0.0245 | 0.4945 | 0.6901 | 0.5469 |
| bge-small | centroid, raw, history=all | 0.0047 | 0.0056 | 0.0102 | 0.1862 | 0.7494 | 0.5921 |
| bge-small | centroid, raw, history=50 | 0.0049 | 0.0058 | 0.0106 | 0.1889 | 0.7523 | 0.5843 |
| bge-small | centroid, mean-centered, history=all | 0.0145 | 0.0158 | 0.0334 | 0.4699 | 0.5565 | 0.4838 |
| bge-small | centroid, mean-centered, history=50 | 0.0144 | 0.0158 | 0.0329 | 0.5234 | 0.5748 | 0.4843 |
| bge-small | knn, raw, history=all | 0.0055 | 0.0071 | 0.0109 | 0.2808 | 0.7326 | 0.5698 |
| bge-small | knn, raw, history=50 | 0.0059 | 0.0076 | 0.0120 | 0.2974 | 0.7364 | 0.5627 |
| bge-small | knn, mean-centered, history=all | 0.0124 | 0.0141 | 0.0270 | 0.4083 | 0.5885 | 0.5125 |
| bge-small | knn, mean-centered, history=50 | 0.0125 | 0.0143 | 0.0270 | 0.4530 | 0.6067 | 0.5152 |

Selected for Day 5: `{"model": "bge-small", "weighting": "liked", "method": "centroid", "center": true, "max_history": null}`
