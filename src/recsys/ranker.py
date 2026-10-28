"""Day 9: stage 2 — a LightGBM LambdaMART ranker over the Day 8 candidates.

Stage 1 (candidates.candidate_set) decides WHAT can be recommended; this module decides the ORDER.
One row per (user, candidate) with the Day 8 features (features.FEATURES); a "query" in LightGBM
terms is one user's candidate list. LambdaMART (objective = lambdarank) builds gradient-boosted
trees whose gradients come from how much swapping two movies would change NDCG — it optimises the
ranking metric directly, instead of predicting a rating.

Labels (from the held-out window AFTER the feature history — no leakage, see features.py):
  binary  rating >= 4 -> 1, else 0                      (the CV metric: binary NDCG@10)
  graded  4★ -> 1, 4.5★ -> 2, 5★ -> 3, else 0 with label_gain [0, 1, 1.5, 2]
          = the project's graded NDCG gain (rating - 3): 4★ counts 1, 5★ counts 2
Which one trains a better BINARY-NDCG ranker is a tuning question, answered on validation.

Users are split ONCE into ranker-train / ranker-valid (split_users, seeded): every row of a user
is on one side, so the ranker is never evaluated on a user it was fitted on.

TwoStage is the serving object used by Day 9 (pop-500 features, cold users), Day 10 (test) and
Day 12 (API): history -> candidates -> features -> booster scores -> top-k.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baselines import PopularityRecommender, user_item_matrix
from .candidates import candidate_set, faiss_candidates
from .features import FEATURE_NAMES, FeatureBatch, FeatureBuilder
from .metrics import PAD

LABEL_SCHEMES = ("binary", "graded")
GRADED_GAIN = [0.0, 1.0, 1.5, 2.0]          # label 0..3 -> gain = rating - 3 (4★ 1, 4.5★ 1.5, 5★ 2)


# --------------------------------------------------------------------------- #
# Labels and the user split
# --------------------------------------------------------------------------- #
def make_labels(rating: np.ndarray, scheme: str, threshold: float = 4.0) -> np.ndarray:
    """Integer relevance labels from held-out ratings (0 = not rated in the label window)."""
    r = np.asarray(rating, dtype=np.float64)
    if scheme == "binary":
        return (r >= threshold).astype(np.int32)
    if scheme == "graded":
        lab = np.where(r >= threshold, np.round((r - threshold) * 2) + 1, 0)
        return np.clip(lab, 0, 3).astype(np.int32)
    raise ValueError(f"unknown label scheme {scheme!r}")


def label_gain(scheme: str) -> list[float]:
    return [0.0, 1.0] if scheme == "binary" else list(GRADED_GAIN)


def split_users(users: np.ndarray, valid_share: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(train_users, valid_users), both sorted, disjoint, covering `users`; same split every run."""
    users = np.unique(np.asarray(users, dtype=np.int64))
    n_valid = int(round(valid_share * len(users)))
    if not 0 < n_valid < len(users):
        raise ValueError("valid_share must leave users on both sides")
    valid = np.sort(np.random.default_rng(seed).choice(users, size=n_valid, replace=False))
    return np.setdiff1d(users, valid), valid


# --------------------------------------------------------------------------- #
# Grouped rows: one block of consecutive rows per user
# --------------------------------------------------------------------------- #
def group_bounds(user_col: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(users, starts, sizes) of the user blocks. Rows must be grouped by user with users in
    increasing order (FeatureBatch.to_frame writes them like that)."""
    u = np.asarray(user_col, dtype=np.int64)
    if u.size == 0:
        return u, u, u
    if np.any(np.diff(u) < 0):
        raise ValueError("rows must be grouped by user, users in increasing order")
    users, starts, sizes = np.unique(u, return_index=True, return_counts=True)
    return users, starts, sizes


def pad_by_user(user_col: np.ndarray, values: np.ndarray, fill) -> tuple[np.ndarray, np.ndarray]:
    """Long rows -> (U, longest list) matrix, keeping each user's row order; `fill` pads."""
    users, starts, sizes = group_bounds(user_col)
    values = np.asarray(values)
    out = np.full((len(users), int(sizes.max(initial=0))), fill, dtype=values.dtype)
    r = np.repeat(np.arange(len(users)), sizes)
    c = np.arange(len(r)) - np.repeat(starts, sizes)
    out[r, c] = values
    return users, out


def topk_by_score(user_col: np.ndarray, movie_col: np.ndarray, score: np.ndarray,
                  k: int) -> tuple[np.ndarray, np.ndarray]:
    """(users, (U, k) movieIds) — each user's candidates sorted by score, highest first.
    Ties keep the row order (= union order: popular first), so equal scores resolve like the
    popularity baseline. NaN scores go last; short lists are PAD-padded."""
    s = np.where(np.isnan(score), -np.inf, np.asarray(score, dtype=np.float64))
    users, S = pad_by_user(user_col, s, -np.inf)
    _, M = pad_by_user(user_col, np.asarray(movie_col, dtype=np.int64), PAD)
    order = np.argsort(-S, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(M, order, axis=1)
    if top.shape[1] < k:
        top = np.pad(top, ((0, 0), (0, k - top.shape[1])), constant_values=PAD)
    return users, top


# --------------------------------------------------------------------------- #
# Training data for LightGBM
# --------------------------------------------------------------------------- #
@dataclass
class RankData:
    X: np.ndarray            # (n, F) float32 features, NaN = missing (LightGBM handles it)
    rating: np.ndarray       # (n,) held-out rating, 0 = not rated in the label window
    users: np.ndarray        # (n,) userId, grouped
    movies: np.ndarray       # (n,) movieId
    features: list[str]

    def __len__(self) -> int:
        return len(self.users)

    @classmethod
    def from_frame(cls, df: pd.DataFrame, features: list[str] | None = None) -> "RankData":
        f = list(features or FEATURE_NAMES)
        X = np.empty((len(df), len(f)), dtype=np.float32)
        for j, name in enumerate(f):                 # column by column: no float64 copy of all
            X[:, j] = df[name].to_numpy(np.float32)
        rating = (df["rating"].to_numpy(np.float32, copy=True) if "rating" in df
                  else np.zeros(len(df), np.float32))
        return cls(X, rating, df["userId"].to_numpy(np.int64, copy=True),
                   df["movieId"].to_numpy(np.int64, copy=True), f)

    def subset(self, users: np.ndarray) -> "RankData":
        m = np.isin(self.users, users)
        return RankData(self.X[m], self.rating[m], self.users[m], self.movies[m], self.features)

    def drop_users_without_positives(self, threshold: float = 4.0) -> "RankData":
        """Training only: a query with no relevant candidate has no pair to learn from."""
        pos = np.unique(self.users[self.rating >= threshold])
        return self.subset(pos)

    def columns(self, names: list[str]) -> "RankData":
        idx = [self.features.index(n) for n in names]
        return RankData(self.X[:, idx], self.rating, self.users, self.movies, list(names))


def lgb_dataset(d: RankData, scheme: str, threshold: float = 4.0, reference=None):
    import lightgbm as lgb

    _, _, sizes = group_bounds(d.users)
    return lgb.Dataset(d.X, label=make_labels(d.rating, scheme, threshold), group=sizes,
                       feature_name=d.features, reference=reference, free_raw_data=True)


def lgb_params(cfg: dict, seed: int, threads: int = 0) -> dict:
    """LightGBM parameters for one grid point. deterministic + row-wise histograms -> the same
    trees on every run with the same data, threads and version (Day 7 reproducibility)."""
    scheme = cfg.get("labels", "binary")
    return {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [10],
        "label_gain": label_gain(scheme),
        "learning_rate": cfg.get("learning_rate", 0.05),
        "num_leaves": cfg.get("num_leaves", 63),
        "min_data_in_leaf": cfg.get("min_data_in_leaf", 200),
        "feature_fraction": cfg.get("feature_fraction", 0.8),
        "lambda_l2": cfg.get("lambda_l2", 1.0),
        "lambdarank_truncation_level": cfg.get("truncation_level", 30),
        "seed": seed, "deterministic": True, "force_row_wise": True,
        "num_threads": threads, "verbose": -1,
    }


def train_ranker(train: RankData, valid: RankData, cfg: dict, seed: int, max_rounds: int,
                 early_stopping: int, threads: int = 0, log=None):
    """Fit one LambdaMART model; early stopping on LightGBM's NDCG@10 of the valid users.
    Returns (booster, info). info['best_iter'] = rounds kept."""
    import lightgbm as lgb

    scheme = cfg.get("labels", "binary")
    t0 = time.time()
    dtr = lgb_dataset(train, scheme)
    dva = lgb_dataset(valid, scheme, reference=dtr)
    hist: dict = {}
    callbacks = [lgb.early_stopping(early_stopping, verbose=False), lgb.record_evaluation(hist)]
    if log:
        callbacks.append(lgb.log_evaluation(100) if log is True else _log_every(log, 100))
    booster = lgb.train(lgb_params(cfg, seed, threads), dtr, num_boost_round=max_rounds,
                        valid_sets=[dva], valid_names=["valid"], callbacks=callbacks)
    curve = hist.get("valid", {}).get("ndcg@10", [])
    best = booster.best_iteration or len(curve)
    return booster, {"best_iter": int(best), "seconds": round(time.time() - t0, 1),
                     "lgb_valid_ndcg@10": round(float(curve[best - 1]), 5) if curve else None}


def _log_every(fn, n: int):
    def cb(env):
        if (env.iteration + 1) % n == 0 and env.evaluation_result_list:
            _, name, val, _ = env.evaluation_result_list[0]
            fn(f"    round {env.iteration + 1}: lightgbm {name} {val:.4f}")
    cb.order = 30
    return cb


def predict(booster, X: np.ndarray, batch: int = 500_000) -> np.ndarray:
    """Scores with the early-stopped number of trees, in row batches (flat memory)."""
    out = np.empty(len(X), dtype=np.float64)
    it = booster.best_iteration or None
    for s in range(0, len(X), batch):
        out[s:s + batch] = booster.predict(X[s:s + batch], num_iteration=it)
    return out


def importance(booster, names: list[str]) -> list[dict]:
    """Gain importance (share of total loss reduction) and split counts, largest first."""
    gain = booster.feature_importance("gain")
    split = booster.feature_importance("split")
    tot = gain.sum() or 1.0
    rows = [{"feature": n, "gain_share": round(float(g / tot), 4), "splits": int(s)}
            for n, g, s in zip(names, gain, split)]
    return sorted(rows, key=lambda r: -r["gain_share"])


# --------------------------------------------------------------------------- #
# Serving: history -> candidates -> features -> scores -> top-k
# --------------------------------------------------------------------------- #
@dataclass
class TwoStage:
    """The whole two-stage recommender for a batch of users.

    pop, emb, flat : stage-1 sources (popularity fitted on the history; Day 5 profile; FAISS Flat
                     over the SAME vectors as the profiles)
    builder        : FeatureBuilder already fit() on the SAME history (Day 8 leakage rule)
    n_pop, m_faiss : candidates = popular top-n_pop ∪ FAISS top-m_faiss
    faiss_feature_m: FAISS list length behind in_faiss / faiss_rank (Day 8: 100)
    booster        : LightGBM model (None -> candidates in union order = popularity first)
    """
    pop: PopularityRecommender
    emb: object
    flat: object
    builder: FeatureBuilder
    n_pop: int
    m_faiss: int
    faiss_feature_m: int
    booster: object | None = None
    features: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    faiss_batch: int = 500

    def stage1(self, hist_u: pd.DataFrame, users: np.ndarray):
        """(C, ranks) for `users` (sorted) given their history rows (grouped, time order).
        Already-rated movies (everything in hist_u) are never candidates."""
        seen = user_item_matrix(hist_u, users, self.pop.index, dtype=np.int8)
        P = self.emb.user_profiles(hist_u, users)
        m = max(self.m_faiss, self.faiss_feature_m)
        pop_l = self.pop.recommend(seen, self.n_pop)
        faiss_l = faiss_candidates(self.flat, P, seen, m, self.faiss_batch)
        return candidate_set(pop_l, faiss_l, self.n_pop, self.m_faiss, self.faiss_feature_m)

    def feature_batch(self, hist_u: pd.DataFrame, users: np.ndarray) -> FeatureBatch:
        C, ranks = self.stage1(hist_u, users)
        return self.builder.transform(users, hist_u, C, ranks)

    def feature_batches(self, hist_u: pd.DataFrame, users: np.ndarray,
                        variants: dict[str, tuple[int, int, int]]) -> dict[str, FeatureBatch]:
        """Day 10: features for several stage-1 variants {name: (n_pop, m_faiss, faiss_feature_m)}
        from ONE profile + popularity + FAISS pass (the expensive part). Each batch equals
        feature_batch() with that variant's sizes (tested)."""
        seen = user_item_matrix(hist_u, users, self.pop.index, dtype=np.int8)
        P = self.emb.user_profiles(hist_u, users)
        n_max = max(v[0] for v in variants.values())
        m_max = max(max(v[1], v[2]) for v in variants.values())
        pop_l = self.pop.recommend(seen, n_max)
        faiss_l = faiss_candidates(self.flat, P, seen, m_max, self.faiss_batch)
        out = {}
        for name, (n, m, f) in variants.items():
            C, ranks = candidate_set(pop_l, faiss_l, n, m, f)
            out[name] = self.builder.transform(users, hist_u, C, ranks)
        return out

    def scores(self, fb: FeatureBatch) -> np.ndarray:
        """(U, K) booster scores, -inf on PAD; without a booster: union order (popular first)."""
        return self.score_with(fb, self.booster, self.features)

    @staticmethod
    def score_with(fb: FeatureBatch, booster, features: list[str]) -> np.ndarray:
        """scores() for any booster and its feature list (Day 10 ablation models)."""
        U, K = fb.candidates.shape
        if booster is None:
            S = -np.tile(np.arange(K, dtype=np.float64), (U, 1))
        else:
            X = np.stack([fb.X[n].ravel() for n in features], axis=1)
            S = predict(booster, X).reshape(U, K)
        return np.where(fb.candidates == PAD, -np.inf, S)

    def recommend(self, hist_u: pd.DataFrame, users: np.ndarray, k: int) -> np.ndarray:
        """(U, k) movieIds, best first (ties -> union order)."""
        fb = self.feature_batch(hist_u, users)
        return top_from_scores(fb.candidates, self.scores(fb), k)


def top_from_scores(C: np.ndarray, S: np.ndarray, k: int) -> np.ndarray:
    """(U, k) candidates by score, best first; ties keep candidate order; -inf (PAD or filtered
    out) never shown -> PAD."""
    order = np.argsort(-S, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(C, order, axis=1)
    return np.where(np.isfinite(np.take_along_axis(S, order, axis=1)), top, PAD)


def build_two_stage(catalog, raw_vectors: np.ndarray, chosen: dict, history: pd.DataFrame,
                    n_pop: int, m_faiss: int, faiss_feature_m: int, booster=None,
                    features: list[str] | None = None) -> TwoStage:
    """Fit every history-dependent part on `history` (train -> val labels; train+val -> test)."""
    from . import config
    from .embeddings import EmbeddingRecommender, FlatIndex

    index = catalog.index
    pop = PopularityRecommender(chosen["popularity_mode"], config.RELEVANCE_THRESHOLD).fit(history, index)
    emb = EmbeddingRecommender(raw_vectors, index, weighting=chosen["profile"]["weighting"],
                               threshold=config.RELEVANCE_THRESHOLD, center=chosen["center"],
                               half_life=chosen["profile"]["half_life"])
    flat = FlatIndex(emb.item_vectors, index.ids)
    items = catalog.items
    years = pd.to_numeric(items["year"], errors="coerce").to_numpy(np.float64)
    runtime = (pd.to_numeric(items["runtime"], errors="coerce").replace(0, np.nan).to_numpy(np.float64)
               if "runtime" in items else None)
    builder = FeatureBuilder(raw_vectors, index, catalog.genres, years, runtime,
                             center=chosen["center"], weighting=chosen["profile"]["weighting"],
                             half_life=chosen["profile"]["half_life"],
                             threshold=config.RELEVANCE_THRESHOLD, maxcos_history=config.MAXCOS_HISTORY,
                             recent_days=config.RECENT_DAYS,
                             prior_votes=config.AVG_RATING_PRIOR).fit(history)
    return TwoStage(pop, emb, flat, builder, n_pop, m_faiss, faiss_feature_m, booster,
                    list(features or FEATURE_NAMES))
