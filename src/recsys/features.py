"""Day 8: ranker features for (user, candidate movie) pairs — computed WITHOUT leakage.

The LightGBM ranker (Day 9) reorders the stage-1 candidates (candidates.union_candidates). It sees
one row per (user, candidate) with the features below and learns how to mix them, instead of the
single hand-set λ of Day 6.

The leakage rule (project plan): every feature comes from ONE ratings table, the "history", and
the labels come from ratings AFTER it:
    tune / train the ranker : history = train       labels = val    (fit on train, score on val)
    final test              : history = train+val   labels = test
FeatureBuilder.fit(history) is the only place ratings enter; transform() never sees labels.
Candidates never contain a movie the user already rated, so the user's own label can never be
inside an item statistic either. (Known limitation, Day 2 finding 3: the split is per user, so
OTHER users' later ratings are in the history — standard for MovieLens, stated openly.)

TMDB vote_count / vote_average / popularity are NOT used: the Kaggle dump was taken years after
most MovieLens ratings, so they would carry future popularity into the past. Only static
metadata (release year, runtime, genres) and the item text embeddings are used.

Feature groups (FEATURES lists every column, its direction for the one-feature check, and meaning)
  source     where the candidate came from: in_pop, pop_rank, in_faiss, faiss_rank
  item       popularity and quality from the history: liked / rating counts, Bayesian average
             rating, recent popularity (last 365 days before the history ends), trend, year, runtime
  user       activity and taste level: #ratings, mean rating, liked share (constant within a user;
             useful to the trees only in combination with item features)
  user-item  semantic: cos to the Day 5 profile (recent taste), cos to the long-term profile, max
             cos to any of the last 50 liked movies ("because you liked X"); metadata: genre
             affinity, year gap, popularity gap (mainstream vs niche users)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .baselines import ItemIndex, history_weights
from .embeddings import center_vectors, l2_normalize
from .metrics import PAD

DAY = 86_400

# (name, direction, meaning). direction: +1 higher = better, -1 lower = better, 0 = not a
# within-user ranking signal (user-level constants) -> skipped by the one-feature check.
FEATURES: list[tuple[str, int, str]] = [
    ("in_pop", +1, "retrieved by the popularity source (1/0)"),
    ("pop_rank", -1, "position in the user's popularity list (NaN = not retrieved by it)"),
    ("in_faiss", +1, "retrieved by FAISS (1/0)"),
    ("faiss_rank", -1, "position in the user's FAISS list (NaN = not retrieved by it)"),
    ("item_liked_log", +1, "log(1 + #ratings >= 4) in the history"),
    ("item_count_log", +1, "log(1 + #ratings) in the history"),
    ("item_avg_rating", +1, "Bayesian average rating (20 pseudo-votes at the global mean)"),
    ("item_liked_share", +1, "#ratings >= 4 / #ratings (smoothed like the average)"),
    ("item_recent_log", +1, "log(1 + #ratings >= 4 in the last 365 days of the history)"),
    ("item_trend", +1, "share of the movie's liked ratings that fall in those 365 days"),
    ("item_year", +1, "release year (newer first in the one-feature check)"),
    ("item_runtime", 0, "runtime in minutes (TMDB; NaN if missing)"),
    ("user_n_ratings_log", 0, "log(1 + #ratings of the user)"),
    ("user_mean_rating", 0, "the user's mean rating"),
    ("user_liked_share", 0, "share of the user's ratings that are >= 4"),
    ("cos_profile", +1, "cos(Day 5 profile = graded, half-life 5, mean-centered; movie)"),
    ("cos_longterm", +1, "cos(all liked movies, no decay; movie) — long-term taste"),
    ("max_cos_liked", +1, "max cos between the movie and any of the user's last 50 liked"),
    ("genre_affinity", +1, "Σ_g (share of the user's liked movies in g) over the movie's genres, / #genres"),
    ("year_gap", -1, "|release year - mean year of the user's liked movies|"),
    ("pop_gap", 0, "item_liked_log - mean item_liked_log of the user's liked movies"),
]
FEATURE_NAMES = [f[0] for f in FEATURES]
DIRECTION = {f[0]: f[1] for f in FEATURES}


# --------------------------------------------------------------------------- #
# Item statistics (from the history only)
# --------------------------------------------------------------------------- #
def item_statistics(history: pd.DataFrame, index: ItemIndex, threshold: float = 4.0,
                    recent_days: int = 365, prior_votes: float = 20.0) -> dict[str, np.ndarray]:
    """Per catalog movie (index order), using ONLY the rows of `history`.

    avg_rating   = (sum of ratings + prior_votes · global mean) / (count + prior_votes)
                   — a movie with 2 ratings of 5★ is not ranked above one with 10,000 of 4.4★
    liked_share  = (liked + prior_votes · global liked share) / (count + prior_votes)
    recent       = liked ratings with timestamp in (cutoff - recent_days, cutoff], cutoff = the
                   last timestamp in the history (nothing after it is known)
    """
    cols = index.cols(history["movieId"].to_numpy())
    r = history["rating"].to_numpy(np.float64)
    n = len(index)
    count = np.bincount(cols, minlength=n).astype(np.float64)
    total = np.bincount(cols, weights=r, minlength=n)
    liked_mask = r >= threshold
    liked = np.bincount(cols[liked_mask], minlength=n).astype(np.float64)
    g_mean = float(r.mean()) if len(r) else 0.0
    g_liked = float(liked_mask.mean()) if len(r) else 0.0
    out = {
        "count": count, "liked": liked,
        "liked_log": np.log1p(liked), "count_log": np.log1p(count),
        "avg_rating": (total + prior_votes * g_mean) / (count + prior_votes),
        "liked_share": (liked + prior_votes * g_liked) / (count + prior_votes),
    }
    if "timestamp" in history:
        t = history["timestamp"].to_numpy(np.int64)
        cutoff = int(t.max()) if len(t) else 0
        recent_mask = liked_mask & (t > cutoff - recent_days * DAY)
        recent = np.bincount(cols[recent_mask], minlength=n).astype(np.float64)
    else:
        cutoff, recent = None, np.zeros(n)
    out["recent_log"] = np.log1p(recent)
    out["trend"] = np.divide(recent, liked, out=np.zeros(n), where=liked > 0)
    out["global_mean"], out["cutoff"] = g_mean, cutoff
    return out


# --------------------------------------------------------------------------- #
# Features for a batch of users
# --------------------------------------------------------------------------- #
@dataclass
class FeatureBatch:
    users: np.ndarray                    # (U,)
    candidates: np.ndarray               # (U, K) movieIds, PAD-padded
    X: dict[str, np.ndarray]             # name -> (U, K) float32, NaN where unknown

    def to_frame(self) -> pd.DataFrame:
        """Long format, one row per real (non-PAD) candidate: userId, movieId, features..."""
        valid = self.candidates != PAD
        r, c = np.nonzero(valid)
        df = pd.DataFrame({"userId": self.users[r].astype(np.int32),
                           "movieId": self.candidates[r, c].astype(np.int32)})
        for name in FEATURE_NAMES:
            df[name] = self.X[name][r, c]
        return df


@dataclass
class FeatureBuilder:
    item_vectors: np.ndarray             # RAW unit vectors from Day 4, rows = index order
    index: ItemIndex
    genres: np.ndarray                   # (n_items, G) multi-hot
    years: np.ndarray                    # (n_items,) float, NaN if unknown
    runtime: np.ndarray | None = None    # (n_items,) float minutes, NaN if unknown
    center: bool = True                  # Day 4 choice: mean-centered vectors
    weighting: str = "graded"            # Day 5 profile
    half_life: float | None = 5
    threshold: float = 4.0
    maxcos_history: int = 50
    recent_days: int = 365
    prior_votes: float = 20.0
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.item_vectors.shape[0] != len(self.index):
            raise ValueError("item_vectors must have one row per catalog movie")
        V = self.item_vectors
        self.V = center_vectors(V) if self.center else l2_normalize(V)
        g = np.asarray(self.genres, dtype=np.float32)
        self.genre_norm = g / np.maximum(g.sum(axis=1, keepdims=True), 1.0)   # rows sum to 1 (or 0)
        self.years = np.asarray(self.years, dtype=np.float64)
        self.runtime = (np.full(len(self.index), np.nan) if self.runtime is None
                        else np.asarray(self.runtime, dtype=np.float64))

    # --- the only place ratings come in ---
    def fit(self, history: pd.DataFrame) -> "FeatureBuilder":
        """history: every rating the features may use (train to predict val; train+val to
        predict test). Item statistics are computed from it once."""
        self.stats = item_statistics(history, self.index, self.threshold, self.recent_days,
                                     self.prior_votes)
        return self

    def transform(self, users: np.ndarray, user_history: pd.DataFrame,
                  candidates: np.ndarray, ranks: dict[str, np.ndarray]) -> FeatureBatch:
        """Features for `users` (sorted, unique) and their candidate matrix (U, K).

        user_history : these users' rows of the SAME history passed to fit(), in time order
        ranks        : from candidates.union_candidates ({"pop": (U,K), "faiss": (U,K)})
        """
        if not self.stats:
            raise RuntimeError("call fit(history) first")
        users = np.asarray(users, dtype=np.int64)
        C = np.asarray(candidates, dtype=np.int64)
        U, K = C.shape
        valid = C != PAD
        cols = np.where(valid, self.index.cols(np.where(valid, C, self.index.ids[0])), 0)
        st = self.stats
        X: dict[str, np.ndarray] = {}

        def item(arr):                      # per-item array -> (U, K) gathered at the candidates
            return np.asarray(arr, dtype=np.float32)[cols]

        # source
        for src in ("pop", "faiss"):
            rk = ranks.get(src, np.full((U, K), np.nan, dtype=np.float32))
            X[f"in_{src}"] = (~np.isnan(rk)).astype(np.float32)
            X[f"{src}_rank"] = rk.astype(np.float32)

        # item
        X["item_liked_log"] = item(st["liked_log"])
        X["item_count_log"] = item(st["count_log"])
        X["item_avg_rating"] = item(st["avg_rating"])
        X["item_liked_share"] = item(st["liked_share"])
        X["item_recent_log"] = item(st["recent_log"])
        X["item_trend"] = item(st["trend"])
        X["item_year"] = item(self.years)
        X["item_runtime"] = item(self.runtime)

        # user
        h = user_history
        hu = np.searchsorted(users, h["userId"].to_numpy(np.int64))
        ok = (hu < U) & (users[np.minimum(hu, U - 1)] == h["userId"].to_numpy(np.int64))
        h, hu = h[ok], hu[ok]
        r = h["rating"].to_numpy(np.float64)
        n_r = np.bincount(hu, minlength=U).astype(np.float64)
        mean_r = np.divide(np.bincount(hu, weights=r, minlength=U), n_r,
                           out=np.full(U, np.nan), where=n_r > 0)
        liked = r >= self.threshold
        n_l = np.bincount(hu[liked], minlength=U).astype(np.float64)
        per_user = {"user_n_ratings_log": np.log1p(n_r), "user_mean_rating": mean_r,
                    "user_liked_share": np.divide(n_l, n_r, out=np.full(U, np.nan), where=n_r > 0)}
        for name, v in per_user.items():
            X[name] = np.broadcast_to(v.astype(np.float32)[:, None], (U, K)).copy()

        # user-item: semantic
        P = l2_normalize(np.asarray(history_weights(h, users, self.index, self.weighting, None,
                                                    self.threshold, self.half_life) @ self.V))
        L = l2_normalize(np.asarray(history_weights(h, users, self.index, "liked", None,
                                                    self.threshold, None) @ self.V))
        X["cos_profile"] = self._gather_cos(P, cols)
        X["cos_longterm"] = self._gather_cos(L, cols)
        X["max_cos_liked"] = self._max_cos(h[liked], hu[liked], cols, U)

        # user-item: metadata
        lcols = self.index.cols(h.loc[liked, "movieId"].to_numpy())
        lu = hu[liked]
        Gu = np.zeros((U, self.genre_norm.shape[1]), dtype=np.float32)
        np.add.at(Gu, lu, (self.genres[lcols] > 0).astype(np.float32))
        Gu /= np.maximum(n_l, 1.0)[:, None].astype(np.float32)          # share of liked in genre g
        X["genre_affinity"] = np.einsum("ug,ukg->uk", Gu, self.genre_norm[cols]).astype(np.float32)
        yr = self.years[lcols]
        yk = ~np.isnan(yr)
        ny = np.bincount(lu[yk], minlength=U).astype(np.float64)
        mean_year = np.divide(np.bincount(lu[yk], weights=yr[yk], minlength=U), ny,
                              out=np.full(U, np.nan), where=ny > 0)
        X["year_gap"] = np.abs(item(self.years) - mean_year[:, None].astype(np.float32))
        lp = st["liked_log"][lcols]
        mean_pop = np.divide(np.bincount(lu, weights=lp, minlength=U), n_l,
                             out=np.full(U, np.nan), where=n_l > 0)
        X["pop_gap"] = X["item_liked_log"] - mean_pop[:, None].astype(np.float32)

        for name in FEATURE_NAMES:                                      # PAD slots -> NaN
            X[name] = np.where(valid, X[name], np.nan).astype(np.float32)
        return FeatureBatch(users, C, X)

    # ------------------------------------------------------------------ #
    def _gather_cos(self, P: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """cos(profile_u, movie) at the candidate columns; NaN for users with no profile.
        Scores the whole catalog (U x n_items, ~46 MB for 500 users) and gathers — simpler and
        faster than gathering U x K x d candidate vectors first."""
        S = (P @ self.V.T)[np.arange(P.shape[0])[:, None], cols].astype(np.float32)
        S[np.linalg.norm(P, axis=1) == 0] = np.nan
        return S

    def _max_cos(self, liked_rows: pd.DataFrame, lu: np.ndarray, cols: np.ndarray,
                 U: int) -> np.ndarray:
        """max over the user's last `maxcos_history` liked movies of cos(liked, candidate)."""
        out = np.full(cols.shape, np.nan, dtype=np.float32)
        if len(liked_rows) == 0:
            return out
        lcol = self.index.cols(liked_rows["movieId"].to_numpy())
        if np.any(np.diff(lu) < 0):
            raise ValueError("user_history must be grouped by user (sorted userId, then time)")
        starts = np.searchsorted(lu, np.arange(U))           # rows are grouped by user (time order)
        ends = np.searchsorted(lu, np.arange(U), side="right")
        for u in range(U):
            a, b = starts[u], ends[u]
            if b == a:
                continue
            Hv = self.V[lcol[max(a, b - self.maxcos_history):b]]     # last N liked
            out[u] = (self.V[cols[u]] @ Hv.T).max(axis=1)
        return out


# --------------------------------------------------------------------------- #
# Labels, leakage guard, one-feature check
# --------------------------------------------------------------------------- #
def attach_labels(df: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Add `rating` = the user's held-out rating of the candidate (0 if not rated in the label
    window). Day 9 turns it into a graded label (4★ -> 1, 4.5★ -> 2, 5★ -> 3, else 0)."""
    t = truth[["userId", "movieId", "rating"]].astype({"userId": np.int32, "movieId": np.int32})
    out = df.merge(t, on=["userId", "movieId"], how="left")
    out["rating"] = out["rating"].fillna(0.0).astype(np.float32)
    return out


def assert_no_label_in_history(history: pd.DataFrame, truth: pd.DataFrame) -> None:
    """The label window must not overlap the history: no (user, movie) pair in both."""
    k = lambda d: d["userId"].to_numpy(np.int64) * 10_000_000 + d["movieId"].to_numpy(np.int64)  # noqa: E731
    both = np.intersect1d(k(history), k(truth))
    if both.size:
        raise ValueError(f"{both.size} (user, movie) pairs are in both the history and the labels")


def rank_by_feature(batch: FeatureBatch, name: str, k: int,
                    direction: int | None = None) -> np.ndarray:
    """Top-k movieIds per user when the candidates are sorted by ONE feature (NaN last; ties keep
    the union order, i.e. popularity first). Used to sanity-check every feature on its own: a
    single feature that scores far above everything else would point to leakage."""
    d = DIRECTION[name] if direction is None else direction
    x = batch.X[name].astype(np.float64) * d
    x = np.where(np.isnan(x) | (batch.candidates == PAD), -np.inf, x)
    order = np.argsort(-x, axis=1, kind="stable")[:, :k]
    top = np.take_along_axis(batch.candidates, order, axis=1)
    top_x = np.take_along_axis(x, order, axis=1)
    return np.where(np.isfinite(top_x), top, PAD)


def oracle_topk(batch: FeatureBatch, truth: pd.DataFrame, k: int) -> np.ndarray:
    """Best possible top-k from these candidates (relevant held-out movies first): the ceiling a
    perfect ranker could reach with this stage 1. Uses labels — an upper bound, not a model."""
    df = attach_labels(pd.DataFrame({"userId": np.repeat(batch.users, batch.candidates.shape[1]),
                                     "movieId": batch.candidates.ravel()}), truth)
    lab = df["rating"].to_numpy().reshape(batch.candidates.shape)
    lab = np.where(batch.candidates == PAD, -1.0, lab)
    order = np.argsort(-lab, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(batch.candidates, order, axis=1)

