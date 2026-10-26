"""Day 6: cold-start users — no training history, only what onboarding tells us.

Onboarding signals
  seeds : the first n liked movies (n = 0 / 1 / 3 / 5). Profile = the warm-user profile from Day 5
          (graded weights, half-life) built from those n movies only.
  text  : a short preference text. In the evaluation it is a simulated genre questionnaire
          ("Science fiction, action and thriller movies"); in the demo it is free text. It is
          encoded by the SAME sentence encoder as the movie documents and moved into the same
          mean-centered space (subtract the SAME catalog mean).

Model — two stages, nothing trained for the new user (zero-shot)
  1. pool  : the M movies with the highest TIME-AWARE popularity prior at the user's onboarding
             moment t (liked ratings in the window [t - W, t), log-scaled). Only ratings from before
             t are counted, so no future information is used.
  2. rerank: score(u, j) = cos(profile_u, v_j) + λ · prior_u(j)   for j in the pool.
  λ -> ∞ (or no onboarding) = time-aware popularity;  M = all, λ = 0 = pure semantic search.

Why (Day 6 smoke run, confirmed by Days 3-5): pure content similarity finds plausible but obscure
movies and loses to popularity 3-6x on MovieLens; adding log-popularity to the score over the WHOLE
catalog just collapses back to popularity. Semantics helps where it is asked a narrower question:
"among the movies people are watching right now, which are closest to your taste?"

W, M and λ are chosen on PSEUDO-cold validation users (warm validation users shown only their first
n liked training movies, see first_liked). cold_test is never used for a choice.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .baselines import TIE_EPS, ItemIndex, batches, history_weights, mask_seen, topk
from .embeddings import center_vectors, encode_texts, l2_normalize
from .metrics import PAD

# MovieLens genre names -> words a person would type. "IMAX" is a screen format, not a taste.
GENRE_WORDS = {"Sci-Fi": "science fiction", "Film-Noir": "film noir", "Children": "children's"}
NOT_A_GENRE = ("IMAX",)
DAY = 86_400


# --------------------------------------------------------------------------- #
# Popularity priors
# --------------------------------------------------------------------------- #
def log_scale(counts: np.ndarray) -> np.ndarray:
    """log(1 + c) / log(1 + row max), row-wise, in [0, 1]. Rows of zeros stay zero."""
    c = np.atleast_2d(np.asarray(counts, dtype=np.float32))
    top = np.log1p(c.max(axis=1, keepdims=True))
    out = np.divide(np.log1p(c), top, out=np.zeros_like(c), where=top > 0)
    return out if np.ndim(counts) == 2 else out[0]


def popularity_prior(train: pd.DataFrame, index: ItemIndex, threshold: float = 4.0) -> np.ndarray:
    """All-time: log(1 + number of training ratings >= threshold), most-liked movie = 1."""
    liked = train.loc[train["rating"].to_numpy() >= threshold, "movieId"].to_numpy()
    return log_scale(np.bincount(index.cols(liked), minlength=len(index)).astype(np.float32))


class TimePopularity:
    """Liked counts per movie in time buckets, stored cumulatively, so the count of liked ratings in
    any window [t - W, t) is one subtraction:  C[b(t)] - C[b(t) - W].

    Buckets are `bucket_days` long (14 days: ~30 years -> ~800 buckets x 23k movies x 4 bytes
    = ~75 MB). The bucket that contains t is NOT counted, so only ratings strictly before the start
    of t's bucket are used (never the future, at most 2 weeks of the past are lost)."""

    def __init__(self, liked: pd.DataFrame, index: ItemIndex, bucket_days: int = 14) -> None:
        self.index, self.size = index, bucket_days * DAY
        b = liked["timestamp"].to_numpy(np.int64) // self.size
        self.b0 = int(b.min()) if len(b) else 0
        nb = int(b.max()) - self.b0 + 1 if len(b) else 1
        cols = index.cols(liked["movieId"].to_numpy())
        counts = np.bincount((b - self.b0) * len(index) + cols,
                             minlength=nb * len(index)).reshape(nb, len(index))
        self.cum = np.zeros((nb + 1, len(index)), dtype=np.float32)
        np.cumsum(counts, axis=0, out=self.cum[1:])        # cum[k] = liked in buckets < k
        self.n_buckets = nb

    @classmethod
    def from_ratings(cls, ratings: pd.DataFrame, index: ItemIndex, threshold: float = 4.0,
                     bucket_days: int = 14) -> "TimePopularity":
        liked = ratings.loc[ratings["rating"].to_numpy() >= threshold, ["movieId", "timestamp"]]
        return cls(liked, index, bucket_days)

    def counts(self, t, window_days: int | None) -> np.ndarray:
        """(len(t), n_items) liked counts in [t - window, t); window None = everything before t."""
        k = np.clip(np.asarray(t, np.int64) // self.size - self.b0, 0, self.n_buckets)
        if window_days is None:
            return self.cum[k]
        lo = np.clip(k - max(1, round(window_days * DAY / self.size)), 0, self.n_buckets)
        return self.cum[k] - self.cum[lo]

    def prior(self, t, window_days: int | None) -> np.ndarray:
        return log_scale(self.counts(np.atleast_1d(t), window_days))

    def now(self) -> int:
        """A timestamp just after the last rating (for serving)."""
        return (self.b0 + self.n_buckets) * self.size


def onboarding_time(seeds: pd.DataFrame, users: np.ndarray) -> np.ndarray:
    """When each user finished onboarding = timestamp of their last seed (seed_rank max).
    Used only as 'now' for the time-aware prior — the same t for every seed count."""
    last = seeds.sort_values("seed_rank").groupby("userId")["timestamp"].last()
    t = last.reindex(users).to_numpy()
    if np.isnan(t.astype(float)).any():
        raise ValueError("every user needs at least one seed with a timestamp")
    return t.astype(np.int64)


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #
def first_liked(history: pd.DataFrame, n: int, threshold: float = 4.0) -> pd.DataFrame:
    """Each user's first n liked rows (history must be in time order per user) + seed_rank 1..n.
    Same rule as split.cold_seeds — used to turn warm validation users into pseudo-cold users."""
    h = history[history["rating"].to_numpy() >= threshold]
    rank = h.groupby("userId", sort=False).cumcount().to_numpy() + 1
    out = h[rank <= n].copy()
    out["seed_rank"] = rank[rank <= n].astype(np.int8)
    return out.reset_index(drop=True)


def query_profiles(query_vectors: np.ndarray, raw_item_vectors: np.ndarray,
                   center: bool) -> np.ndarray:
    """Encoded texts -> the item space the recommender scores in.

    If items were mean-centered (Day 4), the query must lose the SAME catalog mean; otherwise
    every query would still carry the shared 'average movie' direction and match the hubs."""
    Q = l2_normalize(query_vectors)
    if not center:
        return Q
    mean = np.asarray(raw_item_vectors, dtype=np.float32).mean(axis=0, keepdims=True)
    return l2_normalize(Q - mean)


def combine_profiles(a: np.ndarray, b: np.ndarray, weight_b: float = 1.0) -> np.ndarray:
    """normalise(a + weight_b · b) for unit (or all-zero) rows: text + seeds onboarding.
    A user with only one signal keeps exactly that profile."""
    return l2_normalize(np.asarray(a, np.float32) + weight_b * np.asarray(b, np.float32))


# --------------------------------------------------------------------------- #
# Simulated genre questionnaire
# --------------------------------------------------------------------------- #
def onboarding_genres(seed_cols: Sequence[int], genres: np.ndarray, vocab: Sequence[str],
                      n_genres: int = 3) -> list[str]:
    """The n genres a user would tick, proxied by their seeds: most frequent first,
    ties -> the genre that appeared in an earlier seed, then alphabetical (vocab order)."""
    counts: dict[int, int] = {}
    first: dict[int, int] = {}
    for pos, c in enumerate(seed_cols):
        for g in np.flatnonzero(genres[c]):
            if vocab[g] in NOT_A_GENRE:
                continue
            counts[g] = counts.get(g, 0) + 1
            first.setdefault(g, pos)
    order = sorted(counts, key=lambda g: (-counts[g], first[g], g))
    return [vocab[g] for g in order[:n_genres]]


def genre_query(chosen: Sequence[str]) -> str:
    """['Sci-Fi', 'Action', 'Thriller'] -> 'Science fiction, action and thriller movies'."""
    words = [GENRE_WORDS.get(g, g.lower()) for g in chosen]
    if not words:
        return ""
    text = words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]
    return text[0].upper() + text[1:] + " movies"


def genre_match(chosen: Sequence[Sequence[str]], genres: np.ndarray,
                vocab: Sequence[str]) -> np.ndarray:
    """(users x items) share of a user's chosen genres that each movie has, in [0, 1].
    The non-semantic baseline for text onboarding: 'popular movies in the genres you picked'."""
    col = {g: j for j, g in enumerate(vocab)}
    C = np.zeros((len(chosen), len(vocab)), dtype=np.float32)
    for r, gs in enumerate(chosen):
        for g in gs:
            C[r, col[g]] = 1.0
    C = np.divide(C, C.sum(axis=1, keepdims=True), out=np.zeros_like(C),
                  where=C.sum(axis=1, keepdims=True) > 0)
    return C @ genres.T.astype(np.float32)


# --------------------------------------------------------------------------- #
# Pool + rerank top-K for a whole grid of (pool size, λ) at once
# --------------------------------------------------------------------------- #
Prior = np.ndarray | Callable[[slice], np.ndarray]


def pool_rerank_topk(content: Callable[[slice], np.ndarray], n_rows: int, prior: Prior,
                     pools: Sequence[int | None], weights: Sequence[float], seen: sp.csr_matrix,
                     k: int, index: ItemIndex, fallback: np.ndarray,
                     batch_size: int = 1000) -> dict[tuple, np.ndarray]:
    """Top-k movieIds for every (M, λ):  rank the pool by content + λ·prior.

    content(rows) -> dense (len(rows) x n_items) scores (cosines, genre-match shares ...)
    prior         -> (n_items,) shared by everyone, or prior(rows) -> (len(rows) x n_items)
    pool M        -> keep each user's M highest-prior UNSEEN movies (None = whole catalog)
    fallback      -> popularity ranks in (0, 1], added with TIE_EPS: exact ties and empty
                     profiles resolve like the popularity baseline.
    Content scores, masking and the pool are computed once per batch and shared by every λ.
    Returns {(M, λ): (n_rows, k) movieIds}."""
    tie = (TIE_EPS * np.asarray(fallback)).astype(np.float32)
    out = {(m, w): np.empty((n_rows, k), dtype=np.int64) for m in pools for w in weights}
    for b in batches(n_rows, batch_size):
        n = b.stop - b.start
        pr = prior(b) if callable(prior) else np.broadcast_to(np.asarray(prior, np.float32),
                                                              (n, len(index)))
        pr = np.asarray(pr, dtype=np.float32) + tie        # tie-break inside the prior, too
        base = np.asarray(content(b), dtype=np.float32) + tie
        mask_seen(base, seen[b])
        ranked_prior = np.where(np.isfinite(base), pr, -np.inf)
        for m in pools:
            if m is None or m >= len(index):
                for w in weights:
                    out[(m, w)][b] = topk(base + np.float32(w) * pr if w else base, k, index)
                continue
            # only the M pool columns are ranked: (n x M) instead of (n x 23k) per λ
            keep = np.argpartition(-ranked_prior, m - 1, axis=1)[:, :m]
            cb = np.take_along_axis(base, keep, axis=1)
            pb = np.take_along_axis(pr, keep, axis=1)
            kk = min(k, m)
            for w in weights:
                s = cb + np.float32(w) * pb if w else cb
                order = np.argsort(-s, axis=1, kind="stable")[:, :kk]
                cols = np.take_along_axis(keep, order, axis=1)
                best = np.take_along_axis(s, order, axis=1)
                res = np.full((n, k), PAD, dtype=np.int64)
                res[:, :kk] = np.where(np.isfinite(best), index.ids[cols], PAD)
                out[(m, w)][b] = res
    return out


def blend_topk(content: Callable[[slice], np.ndarray], n_rows: int, prior: Prior,
               weights: Sequence[float], seen: sp.csr_matrix, k: int, index: ItemIndex,
               fallback: np.ndarray, batch_size: int = 1000) -> dict[float, np.ndarray]:
    """Whole-catalog blend (pool = all): {λ: top-k}. Kept for the ablation 'no pool'."""
    r = pool_rerank_topk(content, n_rows, prior, [None], weights, seen, k, index, fallback,
                         batch_size)
    return {w: r[(None, w)] for w in weights}


# --------------------------------------------------------------------------- #
# The serving object (Day 12 API / Day 13 UI use this; Day 6 evaluates its parts)
# --------------------------------------------------------------------------- #
@dataclass
class ColdStartRecommender:
    """Onboarding recommender: seeds and/or text in, top-k + 'because' out.

    prior : (n_items,) time-aware popularity prior at serving time ('now')
    blend : tuned {"lambda": λ, "pool": M} per setting, keys "seeds_1/3/5", "text",
            "text+seeds_1/3/5" (written by scripts/run_coldstart.py)."""
    raw_vectors: np.ndarray
    index: ItemIndex
    prior: np.ndarray
    fallback: np.ndarray
    blend: dict = field(default_factory=dict)
    center: bool = True
    weighting: str = "graded"
    half_life: float | None = 5
    threshold: float = 4.0
    query_prefix: str = ""
    encoder: object | None = None

    def __post_init__(self) -> None:
        self.vectors = (center_vectors(self.raw_vectors) if self.center
                        else l2_normalize(self.raw_vectors))

    def setting_for(self, n_seeds: int, has_text: bool) -> dict:
        """{lambda, pool} of the closest tuned setting (more than 5 seeds -> the 5-seed value).
        No onboarding at all -> pure prior (time-aware popularity)."""
        if n_seeds == 0:
            return dict(self.blend.get("text", {"lambda": 0.0, "pool": None})) if has_text \
                else {"lambda": 1.0, "pool": None}
        key = "text+seeds_" if has_text else "seeds_"
        tuned = sorted(int(k.rsplit("_", 1)[1]) for k in self.blend if k.startswith(key))
        if not tuned:
            return {"lambda": 0.0, "pool": None}
        n = min(tuned, key=lambda t: (abs(t - n_seeds), -t))
        return dict(self.blend[key + str(n)])

    def seed_profile(self, seed_ids: Sequence[int],
                     ratings: Sequence[float] | None = None) -> np.ndarray:
        """Seeds in the order the user picked them (last = most recent). Default rating 5★."""
        if len(seed_ids) == 0:
            return np.zeros(self.vectors.shape[1], dtype=np.float32)
        r = [5.0] * len(seed_ids) if ratings is None else list(ratings)
        h = pd.DataFrame({"userId": 0, "movieId": list(seed_ids), "rating": r})
        W = history_weights(h, np.array([0]), self.index, self.weighting, None,
                            self.threshold, self.half_life)
        return l2_normalize(np.asarray(W @ self.vectors))[0]

    def text_profile(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            return np.zeros(self.vectors.shape[1], dtype=np.float32)
        if self.encoder is None:
            raise ValueError("text onboarding needs an encoder")
        q = encode_texts(self.encoder, [text.strip()], prefix=self.query_prefix,
                         show_progress=False)
        return query_profiles(q, self.raw_vectors, self.center)[0]

    def recommend(self, seed_ids: Sequence[int] = (), text: str | None = None, k: int = 10,
                  exclude: Sequence[int] = (), seed_ratings: Sequence[float] | None = None,
                  prior: np.ndarray | None = None) -> list[dict]:
        """[{movieId, score, because}] — because = most similar seed (None for text/popularity).
        prior: override the serving prior (e.g. the prior at a past user's onboarding time)."""
        p = combine_profiles(self.seed_profile(seed_ids, seed_ratings)[None],
                             self.text_profile(text or "")[None])
        cfg = self.setting_for(len(seed_ids), bool(text and text.strip()))
        hide = list(seed_ids) + list(exclude)
        seen = sp.csr_matrix((np.ones(len(hide)), ([0] * len(hide), self.index.cols(hide) if hide else [])),
                             shape=(1, len(self.index)))
        pr = self.prior if prior is None else prior
        ids = pool_rerank_topk(lambda b: p[b] @ self.vectors.T, 1, pr, [cfg["pool"]],
                               [cfg["lambda"]], seen, k, self.index, self.fallback)[
            (cfg["pool"], cfg["lambda"])][0]
        seed_cols = self.index.cols(seed_ids) if len(seed_ids) else np.empty(0, np.int64)
        out = []
        for m in ids[ids >= 0]:
            c = int(self.index.cols([m])[0])
            score = float(self.vectors[c] @ p[0] + cfg["lambda"] * np.asarray(pr)[c])
            because = None
            if seed_cols.size:
                sims = self.vectors[seed_cols] @ self.vectors[c]
                because = int(self.index.ids[seed_cols[int(np.argmax(sims))]])
            out.append({"movieId": int(m), "score": score, "because": because})
        return out
