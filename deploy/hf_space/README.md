---
title: {{TITLE}}
emoji: 🎬
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
license: other
short_description: Two-stage movie recommender — embeddings, FAISS, LightGBM
---

# {{TITLE}}

Live demo of the **Semantic Recommendation & Ranking Engine** (MovieLens 32M + TMDB).

- Pick a few movies → a two-stage recommender answers: popular ∪ FAISS candidates → 21 leak-free
  features → LightGBM LambdaMART → MMR diversity → "Because you liked X" explanations.
- 1–2 picks use a dedicated cold-start ranker, 3+ the full ranker (+27% NDCG@10 over popularity at
  5 picks for users never seen in training; +73% for warm users on the held-out test split).
- Or describe what you want in words: semantic search with bge-small embeddings.

Code, evaluation and results: {{CODE_URL}} (this Space runs commit `{{COMMIT}}`).

This Space contains only code. At startup it downloads the serving bundle (movie vectors, metadata,
aggregate item statistics, the two rankers) from the model repo in the `BUNDLE_REPO` variable.
No individual MovieLens ratings are published (licence). Ratings: MovieLens 32M, GroupLens
Research (no endorsement implied, non-commercial). Metadata/posters: TMDB, not endorsed by TMDB.
