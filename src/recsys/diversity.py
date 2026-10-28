"""Day 11: MMR diversity re-ranking of the ranker's list + per-list diversity measures.

Why: the ranker (Day 9-10) fills a top-10 with whatever scores highest, and the best-scoring
movies for a user are often near-duplicates (same franchise, same genre, same decade). Maximal
Marginal Relevance (Carbonell & Goldstein, 1998) re-ranks the ranker's top-`pool` greedily:

    next = argmax over not-yet-picked i of   λ · rel(i)  −  (1 − λ) · max_{j already picked} sim(i, j)

  rel(i)  the ranker score, min-max scaled to [0, 1] inside the user's pool (LambdaMART scores have
          no fixed scale, so λ must not depend on it)
  sim     cosine between two movies — in the embedding space (mean-centered bge vectors, the same
          ones the features use) or the genre space (multi-hot MovieLens genres)
  λ = 1   exactly the ranker's order (tested);  λ = 0   only "be different from what's on the list"

Only the order inside the ranker's top-`pool` changes, so MMR can never show a movie the ranker
thinks is poor; the cost is a little accuracy, the gain a less repetitive list.

Diversity is reported in TWO spaces on purpose: MMR in the embedding space is also judged by genre
diversity, which it never saw — an improvement there is not "grading your own homework".
"""
from __future__ import annotations

import numpy as np

from .metrics import PAD


def unit_rows(V: np.ndarray) -> np.ndarray:
    """Rows scaled to length 1 (all-zero rows stay zero, e.g. a movie with no genre)."""
    V = np.asarray(V, dtype=np.float64)
    return V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-12, None)


def scale_in_pool(S: np.ndarray) -> np.ndarray:
    """Per row: min-max scale the finite scores to [0, 1]; -inf stays -inf. A row whose finite
    scores are all equal becomes all 0 (nothing to prefer — MMR then only diversifies)."""
    S = np.asarray(S, dtype=np.float64)
    fin = np.isfinite(S)
    lo = np.where(fin, S, np.inf).min(axis=1, keepdims=True)
    hi = np.where(fin, S, -np.inf).max(axis=1, keepdims=True)
    span = np.where(np.isfinite(hi - lo) & (hi > lo), hi - lo, 1.0)
    out = (S - np.where(np.isfinite(lo), lo, 0.0)) / span
    return np.where(fin, out, -np.inf)


def pool_of(C: np.ndarray, S: np.ndarray, pool: int) -> tuple[np.ndarray, np.ndarray]:
    """Each user's `pool` best candidates by score, best first (ties keep candidate order,
    exactly like ranker.top_from_scores); PAD / -inf slots stay at the end with score -inf."""
    S = np.where(np.asarray(C) == PAD, -np.inf, np.asarray(S, dtype=np.float64))
    order = np.argsort(-S, axis=1, kind="stable")[:, :pool]
    return np.take_along_axis(np.asarray(C), order, axis=1), np.take_along_axis(S, order, axis=1)


def mmr_rerank(C: np.ndarray, S: np.ndarray, item_vectors: np.ndarray, index,
               lambdas, k: int, pool: int) -> dict[float, np.ndarray]:
    """{λ: (U, k) movieIds} — MMR re-ranking of each user's top-`pool` candidates.

    C, S          (U, K) candidates and ranker scores (-inf or PAD = not available)
    item_vectors  (n_items, d) rows in index order; cosine is taken after unit_rows()
    One similarity tensor (U, pool, pool) is built per call and shared by every λ.
    The pool is at least k (a top-k needs k movies to choose from). Lists shorter than k
    (fewer valid candidates) are PAD-padded.
    """
    Cp, Sp = pool_of(C, S, max(pool, k))
    U, P = Cp.shape
    valid = np.isfinite(Sp)
    rel = np.where(valid, scale_in_pool(Sp), 0.0)
    V = unit_rows(item_vectors)
    cols = index.cols(np.where(valid, Cp, index.ids[0]))
    X = V[cols] * valid[..., None]                         # (U, P, d), invalid rows = 0
    sim = np.einsum("upd,uqd->upq", X, X)
    rows = np.arange(U)
    out = {}
    for lam in lambdas:
        lam = float(lam)
        picked = np.zeros((U, P), dtype=bool)
        maxsim = np.zeros((U, P))                          # unused until the first pick
        top = np.full((U, k), PAD, dtype=np.int64)
        for step in range(min(k, P)):
            penalty = maxsim if step else 0.0
            obj = np.where(valid & ~picked, lam * rel - (1.0 - lam) * penalty, -np.inf)
            j = np.argmax(obj, axis=1)                     # first max -> ranker order on ties
            ok = np.isfinite(obj[rows, j])
            top[ok, step] = Cp[rows[ok], j[ok]]
            picked[rows[ok], j[ok]] = True
            s_j = sim[rows, :, j]                          # (U, P): similarity to the new pick
            maxsim = np.where(ok[:, None], s_j if step == 0 else np.maximum(maxsim, s_j), maxsim)
        out[lam] = top
    return out


# --------------------------------------------------------------------------- #
# Per-list diversity
# --------------------------------------------------------------------------- #
def per_user_ild(R: np.ndarray, index, item_vectors: np.ndarray, k: int = 10) -> np.ndarray:
    """(U,) intra-list diversity of each top-k list = mean pairwise (1 − cosine); NaN for lists
    with fewer than 2 movies. Same definition as metrics.intra_list_diversity (tested); the
    per-user values let a paired bootstrap test a diversity gain."""
    R = np.asarray(R, dtype=np.int64)[:, :k]
    valid = R != PAD
    V = unit_rows(item_vectors)
    X = V[index.cols(np.where(valid, R, index.ids[0]))] * valid[..., None]
    sim = np.einsum("ukd,ujd->ukj", X, X)
    n = valid.sum(axis=1)
    pair = sim.sum(axis=(1, 2)) - np.einsum("ukk->u", sim)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= 2, 1.0 - pair / (n * (n - 1)), np.nan)


def distinct_genres(R: np.ndarray, index, genres: np.ndarray, k: int = 10) -> np.ndarray:
    """(U,) number of different genres covered by each top-k list (an easy-to-explain diversity)."""
    R = np.asarray(R, dtype=np.int64)[:, :k]
    valid = R != PAD
    G = (np.asarray(genres) > 0)[index.cols(np.where(valid, R, index.ids[0]))] & valid[..., None]
    return G.any(axis=1).sum(axis=1).astype(np.float64)


def pick_lambda(rows: list[dict], max_loss: float, metric: str = "ndcg@10",
                diversity: str = "diversity@10") -> dict:
    """The most diverse λ whose `metric` is within `max_loss` (relative) of λ = 1 (the ranker).
    rows: [{"lambda", metric, diversity}, ...] for ONE similarity space; must contain λ = 1."""
    base = next(r for r in rows if r["lambda"] == 1.0)[metric]
    ok = [r for r in rows if r[metric] >= (1.0 - max_loss) * base]
    return max(ok, key=lambda r: (r[diversity], r["lambda"]))
