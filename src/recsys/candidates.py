"""Candidate sets for the two-stage recommender (measured on Day 5, built properly on Day 8).

Stage 1 (retrieval) must return a few hundred movies per user that CONTAIN the movies the user
will like; stage 2 (the LightGBM ranker) can only reorder what it is given. So the stage-1
metric is recall@N of the candidate set, not NDCG.

`mix_candidates` combines two ranked lists under a fixed budget N, e.g. the embedding model's
top-a plus popular movies to fill the rest:
    take primary[:a], then append fill items in order, skipping duplicates, until N items.
"""
from __future__ import annotations

import numpy as np

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
