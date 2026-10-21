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

## Data attribution

MovieLens: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM TiiS 5, 4. This product uses the TMDB API but is not endorsed
or certified by TMDB.
