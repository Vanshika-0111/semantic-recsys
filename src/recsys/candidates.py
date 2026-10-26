"""Candidate sets for the two-stage recommender (measured on Day 5, built properly on Day 8).

Stage 1 (retrieval) must return a few hundred movies per user that CONTAIN the movies the user
will like; stage 2 (the LightGBM ranker) can only reorder what it is given. So the stage-1
metric is recall@N of the candidate set, not NDCG.

`mix_candidates` combines two ranked lists under a fixed budget N, e.g. the embedding model's
top-a plus popular movies to fill the rest:
    take primary[:a], then append fill items in order, skipping duplicates, until N items.

Day 8 — the real stage 1 is a UNION, not a fixed split (Day 5: any fixed split of 200 lost recall
to popularity alone, but the two lists overlap only 3.5%):
    candidates(u) = popular top-N(u)  ∪  FAISS top-M(u)       (seen movies removed from both)
  faiss_candidates  exact FAISS search with the user profile as the query; over-fetches so that
                    after removing already-seen movies M are left (works with HNSW too, Day 12)
  union_candidates  the union in a fixed order (popular first, then FAISS-only), plus for every
                    candidate its position in each source -> ranker features in_pop / pop_rank /
                    in_faiss / faiss_rank (Day 8 features.py)
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .metrics import PAD


def dedupe_rows(C: np.ndarray) -> np.ndarray:
    """Replace repeated items in each row by PAD, keeping the FIRST occurrence (vectorized).

    A stable sort puts equal items next to each other in their original order, so every
    element equal to its left neighbour in sorted order is a later duplicate."""
    C = np.asarray(C, dtype=np.int64)
    order = np.argsort(C, axis=1, kind="stable")
    s = np.take_along_axis(C, order, axis=1)
    dup_sorted = np.zeros_like(s, dtype=bool)
    dup_sorted[:, 1:] = (s[:, 1:] == s[:, :-1]) & (s[:, 1:] != PAD)
    dup = np.zeros_like(dup_sorted)
    np.put_along_axis(dup, order, dup_sorted, axis=1)
    return np.where(dup, PAD, C)


def compact_rows(C: np.ndarray) -> np.ndarray:
    """Move PAD slots to the end of each row, keeping the order of the real items."""
    C = np.asarray(C, dtype=np.int64)
    order = np.argsort(C == PAD, axis=1, kind="stable")       # False (real) before True (PAD)
    return np.take_along_axis(C, order, axis=1)


def mix_candidates(primary: np.ndarray, fill: np.ndarray, n_primary: int,
                   budget: int) -> np.ndarray:
    """(U, budget) candidate matrix: primary[:, :n_primary] then fill, no duplicates, PAD-padded.

    primary, fill : (U, >=budget) ranked movieIds per user (PAD allowed), already without
                    movies the user has seen.
    """
    if not 0 <= n_primary <= budget:
        raise ValueError("need 0 <= n_primary <= budget")
    C = np.concatenate([np.asarray(primary)[:, :n_primary], np.asarray(fill)[:, :budget]], axis=1)
    C = compact_rows(dedupe_rows(C))[:, :budget]
    if C.shape[1] < budget:
        C = np.pad(C, ((0, 0), (0, budget - C.shape[1])), constant_values=PAD)
    return C


# --------------------------------------------------------------------------- #
# Day 8: FAISS retrieval with seen-movie filtering, and the union of sources
# --------------------------------------------------------------------------- #
def faiss_candidates(index, profiles: np.ndarray, seen: sp.csr_matrix, m: int,
                     batch_size: int = 500) -> np.ndarray:
    """(U, m) movieIds: each user's FAISS top-m among movies they have NOT rated, best first.

    index    : embeddings.FlatIndex or ann.HNSWIndex over the SAME vectors as the profiles
               (its .ids must be the catalog order, i.e. column j of `seen` <-> index.ids[j])
    profiles : (U, d) user query vectors; an all-zero profile (no liked movie) -> a PAD row
    seen     : (U, n_items) CSR, nonzero = already rated (never recommended again)

    FAISS cannot skip seen movies itself, so we ask for k = m + (#seen) and drop them afterwards.
    Users are processed in order of #seen, so light users are not slowed down by heavy ones
    (one k per batch = m + the largest #seen in that batch).
    """
    U = profiles.shape[0]
    out = np.full((U, m), PAD, dtype=np.int64)
    if m == 0 or U == 0:
        return out
    n_seen = np.diff(seen.indptr)
    has_query = np.linalg.norm(profiles, axis=1) > 0
    order = np.argsort(n_seen, kind="stable")
    order = order[has_query[order]]
    for s in range(0, len(order), batch_size):
        rows = order[s:s + batch_size]
        k = int(min(index.ntotal, m + n_seen[rows].max()))
        _, ids = index.search(profiles[rows], k)
        cols = np.searchsorted(index.ids, np.where(ids == PAD, index.ids[0], ids))
        S = seen[rows].toarray() != 0
        is_seen = np.take_along_axis(S, cols, axis=1)
        keep = (ids != PAD) & ~is_seen
        C = compact_rows(np.where(keep, ids, PAD))[:, :m]
        out[rows, :C.shape[1]] = C
    return out


def union_candidates(sources: dict[str, np.ndarray], sizes: dict[str, int]
                     ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Union of several ranked candidate lists, per user.

    sources : {name: (U, >= size) ranked movieIds, PAD-padded}, e.g. {"pop": ..., "faiss": ...}
    sizes   : {name: how many to take from that source}  (dict order = order in the union)
    Returns
      C     : (U, sum(sizes)) movieIds — source 1's items in its order, then source 2's items
              that source 1 did not have, ... ; duplicates removed, PAD at the end of each row
      ranks : {name: (U, sum(sizes)) float32} 0-based position of each candidate in that source's
              list, NaN if that source did not retrieve it (also NaN on PAD slots)
    """
    names = [n for n in sizes if sizes[n] > 0]
    if not names:
        raise ValueError("take at least one item from one source")
    parts = [np.asarray(sources[n], dtype=np.int64)[:, :sizes[n]] for n in names]
    C = compact_rows(dedupe_rows(np.concatenate(parts, axis=1)))
    ranks = {}
    for n in names:
        src = np.asarray(sources[n], dtype=np.int64)[:, :sizes[n]]
        ranks[n] = positions_in(C, src)
    for n in sizes:                       # a source with size 0 retrieved nothing
        ranks.setdefault(n, np.full(C.shape, np.nan, dtype=np.float32))
    return C, ranks


def positions_in(C: np.ndarray, src: np.ndarray) -> np.ndarray:
    """For each C[u, i], its 0-based position in src[u] (NaN if absent or PAD). Vectorized with
    one sorted (row, movieId) key lookup — the same trick as metrics.evaluate."""
    U, K = C.shape
    span = int(max(C.max(initial=0), src.max(initial=0))) + 2
    rows = np.arange(U, dtype=np.int64)[:, None]
    s_keys = (rows * span + src).ravel()
    s_pos = np.tile(np.arange(src.shape[1], dtype=np.float32), U)
    valid = (src != PAD).ravel()
    s_keys, s_pos = s_keys[valid], s_pos[valid]
    o = np.argsort(s_keys, kind="stable")
    s_keys, s_pos = s_keys[o], s_pos[o]
    out = np.full((U, K), np.nan, dtype=np.float32)
    if s_keys.size == 0:
        return out
    c_keys = rows * span + C
    j = np.clip(np.searchsorted(s_keys, c_keys), 0, s_keys.size - 1)
    hit = (C != PAD) & (s_keys[j] == c_keys)
    out[hit] = s_pos[j[hit]]
    return out
