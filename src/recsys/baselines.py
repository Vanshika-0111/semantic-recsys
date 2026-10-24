"""Day 3 baselines: popularity and TF-IDF content-based (item-kNN).

Both models share the same machinery so the comparison is fair:
  * an ItemIndex mapping movieId <-> column (the 23k-movie catalog),
  * scores computed for a batch of users as a dense (batch x items) matrix,
  * items the user already rated in the training data are removed,
  * top-K taken with argpartition.  Batches keep memory flat on an 8 GB laptop.

Popularity
  score(j) = number of training ratings of j   ("count")
          or number of training ratings >= 4  ("liked")
  Ties are broken by movieId so results are deterministic.

TF-IDF content-based (item-kNN)
  1. TF-IDF vectors of the item documents (recsys.text), L2-normalised.
  2. For every item keep its N most similar items (cosine)  ->  sparse S (items x items).
  3. score(u, j) = sum_i w(u, i) * S[i, j]   over the user's history items i.
     With a full S this ranks items exactly like cosine(user TF-IDF centroid, item),
     because the centroid's norm is the same for every j. Keeping N neighbours makes it
     fast and gives a free explanation ("because you liked i").
  4. Items with no content score are filled by popularity (also the 0-seed cold-start case);
     popularity is added with a tiny weight so it only breaks ties.

Weights w(u, i):
  "liked"    : 1 if rating >= threshold, else not used
  "centered" : rating - user's mean rating (dislikes push similar items down)
max_history : only the user's last N weighted items (None = all).
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .metrics import PAD

TIE_EPS = 1e-6   # popularity weight inside content scores (tie-break / fill only)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
class ItemIndex:
    """movieId <-> column position in the catalog (sorted movieIds)."""

    def __init__(self, movie_ids) -> None:
        ids = np.unique(np.asarray(movie_ids, dtype=np.int64))
        self.ids = ids

    def __len__(self) -> int:
        return len(self.ids)

    def cols(self, movie_ids) -> np.ndarray:
        m = np.asarray(movie_ids, dtype=np.int64)
        pos = np.searchsorted(self.ids, m)
        ok = (pos < len(self.ids)) & (self.ids[np.minimum(pos, len(self.ids) - 1)] == m)
        if not ok.all():
            bad = np.unique(m[~ok])[:5]
            raise KeyError(f"{(~ok).sum()} ratings refer to movies outside the catalog, e.g. {bad}")
        return pos


def user_item_matrix(df: pd.DataFrame, users: np.ndarray, index: ItemIndex,
                     values: np.ndarray | None = None, dtype=np.float32) -> sp.csr_matrix:
    """CSR (len(users) x n_items). Rows of `df` whose userId is not in `users` are ignored.
    `users` must be sorted and unique."""
    users = np.asarray(users, dtype=np.int64)
    u = df["userId"].to_numpy(np.int64)
    rows = np.searchsorted(users, u)
    keep = (rows < len(users)) & (users[np.minimum(rows, len(users) - 1)] == u)
    cols = index.cols(df["movieId"].to_numpy()[keep])
    vals = np.ones(keep.sum(), dtype=dtype) if values is None else np.asarray(values, dtype)[keep]
    mat = sp.csr_matrix((vals, (rows[keep], cols)), shape=(len(users), len(index)), dtype=dtype)
    mat.sum_duplicates()
    return mat


def batches(n: int, size: int) -> Iterator[slice]:
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def mask_seen(scores: np.ndarray, seen: sp.csr_matrix) -> None:
    """Set scores of already-rated items to -inf (in place). Rows of seen == rows of scores."""
    if seen.nnz:
        r = np.repeat(np.arange(seen.shape[0]), np.diff(seen.indptr))
        scores[r, seen.indices] = -np.inf


def topk(scores: np.ndarray, k: int, index: ItemIndex) -> np.ndarray:
    """Top-k movieIds per row (descending score); -inf slots become PAD."""
    n = scores.shape[1]
    k_eff = min(k, n)
    part = np.argpartition(-scores, k_eff - 1, axis=1)[:, :k_eff]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    cols = np.take_along_axis(part, order, axis=1)
    best = np.take_along_axis(part_scores, order, axis=1)
    out = np.full((scores.shape[0], k), PAD, dtype=np.int64)
    out[:, :k_eff] = np.where(np.isfinite(best), index.ids[cols], PAD)
    return out


def popularity_scores(train: pd.DataFrame, index: ItemIndex, mode: str = "count",
                      threshold: float = 4.0) -> np.ndarray:
    """Unique popularity score per item: higher = more popular, ties -> smaller movieId first.
    Returned values are ranks scaled to (0, 1], so they can also serve as a tie-breaker."""
    if mode == "count":
        movies = train["movieId"].to_numpy()
    elif mode == "liked":
        movies = train.loc[train["rating"].to_numpy() >= threshold, "movieId"].to_numpy()
    else:
        raise ValueError(f"unknown popularity mode {mode!r}")
    counts = np.bincount(index.cols(movies), minlength=len(index)).astype(np.float64)
    order = np.lexsort((index.ids, -counts))           # most popular first
    score = np.empty(len(index), dtype=np.float64)
    score[order] = (len(index) - np.arange(len(index))) / len(index)
    return score


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
@dataclass
class PopularityRecommender:
    mode: str = "count"
    threshold: float = 4.0
    index: ItemIndex | None = None
    score: np.ndarray | None = None

    def fit(self, train: pd.DataFrame, index: ItemIndex) -> "PopularityRecommender":
        self.index = index
        self.score = popularity_scores(train, index, self.mode, self.threshold)
        return self

    def recommend(self, seen: sp.csr_matrix, k: int, batch_size: int = 2000) -> np.ndarray:
        out = np.empty((seen.shape[0], k), dtype=np.int64)
        row = self.score.astype(np.float32)
        for b in batches(seen.shape[0], batch_size):
            s = np.tile(row, (b.stop - b.start, 1))
            mask_seen(s, seen[b])
            out[b] = topk(s, k, self.index)
        return out


@dataclass
class TfidfKNNRecommender:
    n_neighbors: int = 200
    weighting: str = "liked"           # "liked" | "centered"
    max_history: int | None = None
    threshold: float = 4.0
    half_life: float | None = None     # Day 5 recency decay (see history_weights)
    ngram_max: int = 2
    min_df: int = 2
    max_df: float = 0.5
    max_features: int | None = 200_000
    index: ItemIndex | None = None
    sim: sp.csr_matrix | None = None    # items x items, top-N neighbours per row
    vocab_size: int = 0

    # --- item side (independent of ratings, so it can be cached) ---
    def fit_items(self, documents: pd.Series, index: ItemIndex,
                  chunk: int = 1000) -> "TfidfKNNRecommender":
        """documents: Series indexed by movieId. Builds TF-IDF + top-N cosine neighbours."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.index = index
        docs = documents.reindex(index.ids)
        if docs.isna().any():
            raise ValueError(f"{docs.isna().sum()} catalog movies have no document")
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, self.ngram_max),
                              min_df=self.min_df, max_df=self.max_df,
                              max_features=self.max_features, sublinear_tf=True,
                              dtype=np.float32)
        X = vec.fit_transform(docs.tolist()).tocsr()     # rows are L2-normalised
        self.vocab_size = len(vec.vocabulary_)
        self.sim = top_neighbors(X, self.n_neighbors, chunk)
        return self

    def set_similarity(self, sim: sp.csr_matrix, index: ItemIndex) -> "TfidfKNNRecommender":
        self.sim, self.index = sim.tocsr(), index
        return self

    # --- user side ---
    def user_weights(self, history: pd.DataFrame, users: np.ndarray) -> sp.csr_matrix:
        """history: rows (userId, movieId, rating) in TIME ORDER per user."""
        return history_weights(history, users, self.index, self.weighting,
                               self.max_history, self.threshold, self.half_life)

    def recommend(self, weights: sp.csr_matrix, seen: sp.csr_matrix, k: int,
                  fallback: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """fallback: popularity score per item (0..1] used for ties / empty profiles."""
        out = np.empty((weights.shape[0], k), dtype=np.int64)
        tie = (TIE_EPS * fallback).astype(np.float32)
        for b in batches(weights.shape[0], batch_size):
            s = (weights[b] @ self.sim).toarray().astype(np.float32, copy=False)
            s += tie
            mask_seen(s, seen[b])
            out[b] = topk(s, k, self.index)
        return out

    def explain(self, history_cols: np.ndarray, item_col: int) -> int | None:
        """Column of the history item that contributes most to `item_col` (Day 11 explanations)."""
        col = self.sim[:, item_col].toarray().ravel()[history_cols]
        return int(history_cols[np.argmax(col)]) if col.size and col.max() > 0 else None


# The item-kNN scoring above works for ANY item-item similarity matrix (TF-IDF on Day 3,
# sentence embeddings on Day 4), so it is also exposed under a representation-neutral name.
ItemKNNRecommender = TfidfKNNRecommender


def history_weights(history: pd.DataFrame, users: np.ndarray, index: ItemIndex,
                    weighting: str = "liked", max_history: int | None = None,
                    threshold: float = 4.0, half_life: float | None = None) -> sp.csr_matrix:
    """User x item weight matrix W used by every content model (TF-IDF Day 3, embeddings Day 4-5).

    history: rows (userId, movieId, rating) in TIME ORDER per user.
      "liked"    : w = 1 if rating >= threshold (other rows dropped)
      "graded"   : w = rating - threshold + 1 for liked rows  (4★ -> 1, 4.5★ -> 1.5, 5★ -> 2;
                   same gain as graded NDCG) — a 5★ counts twice as much as a 4★
      "centered" : w = rating - user's mean rating
    half_life (Day 5) multiplies each weight by 0.5 ** (p / half_life), where p = 0 for the
      user's most recent weighted row, 1 for the one before ... So with half_life = 10 the 10th
      most recent liked movie counts half, the 20th a quarter. Positions, not days: EDA showed
      most users rate in one bulk session, so clock time carries almost no order information.
    max_history keeps only each user's last N weighted rows.
    """
    h = history
    if weighting in ("liked", "graded"):
        r = h["rating"].to_numpy()
        h = h[r >= threshold]
        r = r[r >= threshold]
        w = (np.ones(len(h)) if weighting == "liked" else r - threshold + 1.0).astype(np.float32)
    elif weighting == "centered":
        mean = h.groupby("userId")["rating"].transform("mean").to_numpy()
        w = (h["rating"].to_numpy() - mean).astype(np.float32)
    else:
        raise ValueError(f"unknown weighting {weighting!r}")
    if max_history or half_life:
        pos = _pos_from_end(h["userId"].to_numpy())
        if half_life:
            if half_life <= 0:
                raise ValueError("half_life must be positive")
            w = (w * np.power(0.5, pos / float(half_life))).astype(np.float32)
        if max_history:
            keep = pos < max_history
            h, w = h[keep], w[keep]
    return user_item_matrix(h, users, index, values=w)


def _pos_from_end(user_ids: np.ndarray) -> np.ndarray:
    """0 = user's last row, 1 = second last ... (rows grouped by user, in time order)."""
    n = len(user_ids)
    if n == 0:
        return np.empty(0, np.int64)
    starts = np.flatnonzero(np.r_[True, user_ids[1:] != user_ids[:-1]])
    sizes = np.diff(np.r_[starts, n])
    pos = np.arange(n) - np.repeat(starts, sizes)
    return np.repeat(sizes, sizes) - 1 - pos


def top_neighbors(X: sp.csr_matrix, n_neighbors: int, chunk: int = 1000) -> sp.csr_matrix:
    """Keep the n most cosine-similar OTHER items per row (rows of X are L2-normalised).
    Only positive similarities are kept. Returns CSR (items x items), float32."""
    n = X.shape[0]
    k = min(n_neighbors, n - 1)
    XT = X.T.tocsc()
    rows, cols, vals = [], [], []
    for b in batches(n, chunk):
        s = (X[b] @ XT).toarray().astype(np.float32, copy=False)
        s[np.arange(b.stop - b.start), np.arange(b.start, b.stop)] = -np.inf   # no self-match
        part = np.argpartition(-s, k - 1, axis=1)[:, :k]
        v = np.take_along_axis(s, part, axis=1)
        keep = v > 0
        r = np.repeat(np.arange(b.start, b.stop), k).reshape(-1, k)
        rows.append(r[keep]); cols.append(part[keep]); vals.append(v[keep])
    rows, cols, vals = (np.concatenate(a) for a in (rows, cols, vals))
    return sp.csr_matrix((vals.astype(np.float32), (rows, cols)), shape=(n, n))
