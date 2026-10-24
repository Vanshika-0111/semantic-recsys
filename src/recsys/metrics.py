"""Ranking metrics.

Two layers:
1. Reference functions on ONE ranked list (`precision_at_k`, `ndcg_at_k`, ...).
   Small and readable; unit-tested against hand-computed numbers.
2. `evaluate(...)` — a vectorized version for 200k users at once. A test checks it
   gives exactly the same numbers as layer 1 on random data.

Conventions (write these in the README; interviewers ask):
- Relevant = held-out item with rating >= threshold (4.0).
- Users with no relevant held-out item are skipped (their metrics are undefined).
- Precision@K = hits in top-K / K   (K even if the model returned fewer items).
- Recall@K    = hits in top-K / |relevant|.
- MRR@K       = 1 / rank of the first relevant item in top-K, else 0.
- NDCG@K      = DCG@K / IDCG@K, DCG = sum gain_i / log2(i + 1), i = 1..K.
  Binary gains by default; graded=True uses gain = rating - threshold + 1
  (4 -> 1, 4.5 -> 1.5, 5 -> 2).
- A user with no recommendations at all scores 0 (never silently dropped).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

PAD = -1  # padding value for users with fewer than K recommendations


# --------------------------------------------------------------------------- #
# 1) Reference implementations — one user, one ranked list
# --------------------------------------------------------------------------- #
def _gains(relevant: Mapping[int, float] | Iterable[int]) -> dict[int, float]:
    if isinstance(relevant, Mapping):
        return {i: float(g) for i, g in relevant.items() if g > 0}
    return {i: 1.0 for i in relevant}


def _check(recs: Sequence[int], k: int) -> list[int]:
    if k <= 0:
        raise ValueError("k must be positive")
    if len(set(recs)) != len(recs):
        raise ValueError("recommendation list contains duplicates")
    return list(recs)[:k]


def precision_at_k(recs: Sequence[int], relevant, k: int) -> float:
    rel = _gains(relevant)
    return sum(i in rel for i in _check(recs, k)) / k


def recall_at_k(recs: Sequence[int], relevant, k: int) -> float:
    rel = _gains(relevant)
    if not rel:
        return float("nan")
    return sum(i in rel for i in _check(recs, k)) / len(rel)


def mrr_at_k(recs: Sequence[int], relevant, k: int) -> float:
    rel = _gains(relevant)
    for rank, i in enumerate(_check(recs, k), start=1):
        if i in rel:
            return 1.0 / rank
    return 0.0


def dcg_at_k(gains_in_rank_order: Sequence[float], k: int) -> float:
    g = np.asarray(list(gains_in_rank_order)[:k], dtype=float)
    return float(np.sum(g / np.log2(np.arange(2, g.size + 2))))


def ndcg_at_k(recs: Sequence[int], relevant, k: int) -> float:
    rel = _gains(relevant)
    if not rel:
        return float("nan")
    dcg = dcg_at_k([rel.get(i, 0.0) for i in _check(recs, k)], k)
    idcg = dcg_at_k(sorted(rel.values(), reverse=True), k)
    return dcg / idcg


# --------------------------------------------------------------------------- #
# Beyond-accuracy metrics (computed over all users' lists)
# --------------------------------------------------------------------------- #
def catalog_coverage(rec_lists: Iterable[Sequence[int]], catalog_size: int, k: int) -> float:
    """Share of the catalog that appears in at least one top-K list."""
    seen: set[int] = set()
    for recs in rec_lists:
        seen.update(i for i in list(recs)[:k] if i != PAD)
    return len(seen) / catalog_size


def long_tail_share(rec_lists: Iterable[Sequence[int]], head_items: set[int], k: int) -> float:
    """Share of recommended slots filled by NON-head (long-tail) items.
    head_items = most popular items in TRAIN (e.g. the top 10% that hold ~79% of ratings)."""
    total = tail = 0
    for recs in rec_lists:
        for i in list(recs)[:k]:
            if i == PAD:
                continue
            total += 1
            tail += i not in head_items
    return tail / total if total else float("nan")


def intra_list_diversity(recs: Sequence[int], item_vectors: np.ndarray,
                         item_index: Mapping[int, int], k: int) -> float:
    """Mean pairwise (1 - cosine similarity) inside one top-K list.
    item_vectors: one row per item (embeddings from Day 4, or multi-hot genres)."""
    rows = [item_index[i] for i in list(recs)[:k] if i in item_index]
    if len(rows) < 2:
        return float("nan")
    v = item_vectors[rows].astype(float)
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    sim = v @ v.T
    n = len(rows)
    return float(1.0 - (sim.sum() - np.trace(sim)) / (n * (n - 1)))


def mean_intra_list_diversity(rec_lists: Iterable[Sequence[int]], item_vectors: np.ndarray,
                              item_index: Mapping[int, int], k: int) -> float:
    vals = [intra_list_diversity(r, item_vectors, item_index, k) for r in rec_lists]
    vals = [x for x in vals if not np.isnan(x)]
    return float(np.mean(vals)) if vals else float("nan")


def beyond_accuracy(recs: np.ndarray, item_ids: np.ndarray, head_items: np.ndarray,
                    item_vectors: np.ndarray, k: int = 10, chunk: int = 20000) -> dict:
    """Vectorized catalog coverage, long-tail share and mean intra-list diversity @k
    for a (U, Kmax) recommendation matrix padded with PAD. Same definitions as the
    reference functions above (a test checks they agree).

    item_ids     : sorted catalog movieIds; item_vectors rows follow item_ids
    head_items   : movieIds of the popular head (e.g. top 10% by training ratings)
    """
    r = np.asarray(recs, dtype=np.int64)[:, :k]
    valid = r != PAD
    item_ids = np.asarray(item_ids, dtype=np.int64)
    rec_items = np.unique(r[valid])
    out = {f"coverage@{k}": float(rec_items.size / item_ids.size)}
    n_slots = int(valid.sum())
    tail = valid & ~np.isin(r, np.asarray(head_items, dtype=np.int64))
    out[f"long_tail@{k}"] = float(tail.sum() / n_slots) if n_slots else float("nan")

    v = np.asarray(item_vectors, dtype=np.float64)
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-12, None)
    total, count = 0.0, 0
    for s in range(0, r.shape[0], chunk):
        rr, vv = r[s:s + chunk], valid[s:s + chunk]
        pos = np.searchsorted(item_ids, np.where(vv, rr, item_ids[0]))
        vv = vv & (item_ids[np.minimum(pos, item_ids.size - 1)] == rr)
        L = v[pos] * vv[..., None]                          # (u, k, d), padded rows = 0
        sim = np.einsum("ukd,ujd->ukj", L, L)
        n = vv.sum(axis=1)
        pair_sum = sim.sum(axis=(1, 2)) - np.einsum("ukk->u", sim)
        ok = n >= 2
        total += float(np.sum(1.0 - pair_sum[ok] / (n[ok] * (n[ok] - 1))))
        count += int(ok.sum())
    out[f"diversity@{k}"] = total / count if count else float("nan")
    return out


# --------------------------------------------------------------------------- #
# 2) Vectorized evaluation for many users
# --------------------------------------------------------------------------- #
def recs_to_matrix(recs_by_user: Mapping[int, Sequence[int]], k: int) -> tuple[np.ndarray, np.ndarray]:
    """{userId: [movieIds...]} -> (users array, int64 matrix padded with PAD)."""
    users = np.fromiter(recs_by_user.keys(), dtype=np.int64, count=len(recs_by_user))
    mat = np.full((len(users), k), PAD, dtype=np.int64)
    for r, u in enumerate(users):
        items = list(recs_by_user[u])[:k]
        mat[r, :len(items)] = items
    return users, mat


def evaluate(users: np.ndarray, recs: np.ndarray, truth: pd.DataFrame,
             ks: Sequence[int] = (5, 10, 20), threshold: float = 4.0, graded: bool = False,
             return_per_user: bool = False):
    """Score top-K lists for many users at once.

    users : (U,) userIds, one per row of `recs`
    recs  : (U, Kmax) movieIds in rank order, padded with PAD (-1)
    truth : held-out rows with columns userId, movieId, rating
    Returns a dict {"ndcg@10": ..., ..., "n_users": ...} (and a per-user DataFrame
    if return_per_user=True — needed later for NDCG-vs-history and significance tests).
    """
    users = np.asarray(users, dtype=np.int64)
    recs = np.asarray(recs, dtype=np.int64)
    if recs.ndim != 2 or recs.shape[0] != users.shape[0]:
        raise ValueError("recs must be a (len(users), K) matrix")
    if len(np.unique(users)) != len(users):
        raise ValueError("users must be unique")
    kmax = max(ks)
    if recs.shape[1] < kmax:
        recs = np.pad(recs, ((0, 0), (0, kmax - recs.shape[1])), constant_values=PAD)
    recs = recs[:, :kmax]
    srt = np.sort(recs, axis=1)
    if np.any((srt[:, 1:] == srt[:, :-1]) & (srt[:, 1:] != PAD)):
        raise ValueError("a recommendation list contains duplicates")

    # Relevant held-out items and their gains
    rel = truth.loc[truth["rating"] >= threshold, ["userId", "movieId", "rating"]]
    gain = (rel["rating"].to_numpy(float) - threshold + 1.0) if graded else np.ones(len(rel))
    eval_users = np.unique(rel["userId"].to_numpy(np.int64))     # users we can score
    P = len(eval_users)
    upos = np.searchsorted(eval_users, rel["userId"].to_numpy(np.int64))

    # Align the recommendation matrix to eval_users (missing user -> empty list)
    if P == 0:
        raise ValueError("truth has no relevant items — nothing to evaluate")
    R = np.full((P, kmax), PAD, dtype=np.int64)
    found = np.zeros(P, dtype=bool)
    if len(users):
        order = np.argsort(users)
        su = users[order]
        idx = np.clip(np.searchsorted(su, eval_users), 0, len(su) - 1)
        found = su[idx] == eval_users
        R[found] = recs[order[idx[found]]]

    # Gain of every recommended slot via a sorted (user, item) key lookup
    span = int(max(rel["movieId"].max(), recs.max(initial=0))) + 2
    t_keys = upos.astype(np.int64) * span + rel["movieId"].to_numpy(np.int64)
    t_order = np.argsort(t_keys)
    t_keys, t_gain = t_keys[t_order], gain[t_order]
    r_keys = np.arange(P, dtype=np.int64)[:, None] * span + R
    pos = np.clip(np.searchsorted(t_keys, r_keys), 0, len(t_keys) - 1)
    G = np.where((R != PAD) & (t_keys[pos] == r_keys), t_gain[pos], 0.0)

    # Ideal DCG: each user's gains sorted descending
    n_rel = np.bincount(upos, minlength=P).astype(float)
    ideal = pd.DataFrame({"u": upos, "g": gain}).sort_values(["u", "g"], ascending=[True, False])
    ideal_rank = ideal.groupby("u").cumcount().to_numpy()
    disc = 1.0 / np.log2(np.arange(2, kmax + 2))

    out: dict = {"n_users": int(P), "n_users_without_recs": int((~found).sum())}
    per_user = {"userId": eval_users, "n_relevant": n_rel.astype(int)}
    hits = G > 0
    for k in ks:
        Gk, Hk = G[:, :k], hits[:, :k]
        nh = Hk.sum(axis=1)
        sel = ideal_rank < k
        idcg = np.bincount(ideal["u"].to_numpy()[sel],
                           weights=ideal["g"].to_numpy()[sel] * disc[ideal_rank[sel]], minlength=P)
        first = np.where(Hk.any(axis=1), Hk.argmax(axis=1) + 1, 0)
        m = {
            f"ndcg@{k}": (Gk @ disc[:k]) / idcg,
            f"recall@{k}": nh / n_rel,
            f"precision@{k}": nh / k,
            f"mrr@{k}": np.where(first > 0, 1.0 / np.maximum(first, 1), 0.0),
            f"hit_rate@{k}": (nh > 0).astype(float),
        }
        for name, v in m.items():
            out[name] = float(v.mean())
            per_user[name] = v
    if return_per_user:
        return out, pd.DataFrame(per_user)
    return out
