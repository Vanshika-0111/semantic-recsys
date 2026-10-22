# Semantic Recommendation & Ranking Engine

Two-stage movie recommender: dense semantic retrieval (Sentence Transformers + FAISS)
followed by a LightGBM learning-to-rank model, with zero-shot cold-start support.
Data: MovieLens 32M joined with TMDB metadata.

> Demo link, results table and charts go here (Day 14).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_data.py    # MovieLens 32M + Kaggle TMDB (needs Kaggle API token)
python scripts/prepare_data.py     # join + k-core filter -> data/processed/
python scripts/eda.py              # figures + reports/eda_summary.md
python scripts/make_split.py       # Day 2: create + freeze the evaluation split
pytest -q
```

## Repo layout

```
src/recsys/      library code (config, data loading, later: metrics, models)
scripts/         runnable pipeline steps
tests/           unit tests
reports/         EDA summary and figures
data/            raw + processed data (git-ignored)
```

## Data decisions

- Items are joined to TMDB on `tmdbId` from `links.csv`, falling back to the IMDb id.
- Movies without a TMDB overview are dropped (the model is text-based).
- Iterative k-core: every user and item keeps >= 20 ratings after filtering.
- `items.parquet` holds metadata only; popularity / average-rating features are computed
  later from the training split to avoid leakage.

## Evaluation protocol (frozen on Day 2)

- **Warm users** — per user, ratings ordered by (timestamp, movieId):
  last 10 → **test**; the 10 before → **validation** (only if ≥ 10 training ratings remain);
  the rest → **train**. Tuning: fit on train, score on val. Final numbers: fit on train + val, score on test.
- **Cold-start users** — 5,000 users (seeded sample) removed from *all* training data.
  Their first 5 liked movies are onboarding seeds; models are scored on their last 10 ratings
  with 0 / 1 / 3 / 5 seeds.
- **Relevance**: rating ≥ 4. Users with no relevant held-out item are skipped.
- **Metrics** (`src/recsys/metrics.py`, hand-checked in `tests/test_metrics.py`):
  NDCG@K, Recall@K, Precision@K, MRR@K, hit rate, catalog coverage, long-tail share,
  intra-list diversity, at K = 5 / 10 / 20.
- The split is fingerprinted (SHA-256 of user, item, split) in
  `data/processed/splits/split_manifest.json`; `python scripts/make_split.py --verify` proves
  every model used the same split.
- Known limitation: a per-user time split is not a global time split, so some training
  ratings of other users are later than a user's test ratings (standard for MovieLens; noted).

## Data attribution

MovieLens: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM TiiS 5, 4. This product uses the TMDB API but is not endorsed
or certified by TMDB.
