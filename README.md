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
python scripts/run_baselines.py    # Day 3: popularity + TF-IDF baselines -> reports/results/
python scripts/run_embeddings.py   # Day 4: MiniLM vs bge-small item embeddings + FAISS Flat
pytest -q
```

## Repo layout

```
src/recsys/      library code (config, data, split, metrics, text, baselines, embeddings)
scripts/         runnable pipeline steps
tests/           unit tests
reports/         EDA summary, figures, results/ (metric tables per day)
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

## Baselines (Day 3)

Both baselines share one scoring path (dense batch scores → remove already-rated movies →
top-K), so every later model is compared under identical rules.

- **Popularity** — rank movies by number of training ratings (`count`) or by number of
  ratings ≥ 4 (`liked`); the variant is picked on validation.
- **TF-IDF content-based (item-kNN)** — TF-IDF (1–2-grams) over the *same* item document the
  embedding model uses on Day 4 (title, year, genres, keywords, tagline, overview). Each movie
  keeps its 200 most cosine-similar movies; a user's score for movie *j* is
  Σᵢ w(u,i)·sim(i,j) over their history. With all neighbours kept this ranks exactly like
  cosine(user TF-IDF centroid, movie) (unit-tested); keeping 200 makes it fast and gives
  "because you liked X" explanations for free. Weights: `liked` (1 for rating ≥ 4) or
  `centered` (rating − user mean, so dislikes push similar movies down), optionally only the
  last 50 items; chosen on validation.
- Cold start: profiles come from the 1 / 3 / 5 onboarding seeds; with 0 seeds the content
  model has nothing to go on and falls back to popularity.
- Beyond accuracy: catalog coverage, long-tail share (outside the top 10% most-rated movies)
  and genre intra-list diversity @10.

Results: `reports/results/day03_baselines.md`.

## Semantic item embeddings (Day 4)

- The same item document as TF-IDF is encoded by two small sentence encoders:
  `all-MiniLM-L6-v2` (6 layers) and `bge-small-en-v1.5` (12 layers, retrieval-tuned), both 384-d.
  Vectors are L2-normalised, so a dot product is a cosine similarity. bge's query instruction
  prefix is used only for short text queries (cold-start onboarding), never for documents.
- Vectors are cached in `data/embeddings/` together with a SHA-256 of the documents, so a change
  to the document template can never silently reuse stale vectors.
- **FAISS `IndexFlatIP`** = exact nearest-neighbour search (every item is compared). It is the
  ground truth for the approximate HNSW index (Day 5) and the candidate generator (Day 8).
  A unit test checks it returns exactly the brute-force numpy top-K.
- **Embedding recommender**: profile = normalised sum of the user's history vectors, with the
  same weights as TF-IDF (`liked` / `centered`, all or last 50); score = cosine(profile, movie);
  same tie-break, seen-item masking and top-K as every other model. Encoder and profile setting
  are chosen on validation only.

Results: `reports/results/day04_embeddings.md`, examples: `reports/results/day04_similar_movies.md`.

## Data attribution

MovieLens: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
History and Context. ACM TiiS 5, 4. This product uses the TMDB API but is not endorsed
or certified by TMDB.
