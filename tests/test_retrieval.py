"""Day 5 tests: graded + recency-decayed profiles, HNSW vs Flat, candidate mixing.

Same toy catalog as test_embeddings.py (unit vectors, cosine = dot product):
    10: ( 1.0,  0.0)      20: ( 0.0, 1.0)      30: ( 0.6, 0.8)
    40: ( 0.8, -0.6)      50: (-1.0, 0.0)
Popularity (count, from the shared toy ratings): 10, 20, 30, 40, 50 (ties -> smaller id).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.ann import HNSWIndex, index_size_mb, query_latency_ms, recall_vs_exact  # noqa: E402
from recsys.baselines import (ItemIndex, history_weights, popularity_scores,  # noqa: E402
                              user_item_matrix)
from recsys.candidates import compact_rows, dedupe_rows, mix_candidates  # noqa: E402
from recsys.embeddings import EmbeddingRecommender, FlatIndex, l2_normalize  # noqa: E402
from recsys.metrics import PAD  # noqa: E402

INDEX = ItemIndex([10, 20, 30, 40, 50])
V = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.8, -0.6], [-1.0, 0.0]], dtype=np.float32)
TRAIN = pd.DataFrame([(1, 10, 5), (1, 20, 2), (2, 10, 4), (2, 30, 4), (3, 10, 3), (3, 40, 5)],
                     columns=["userId", "movieId", "rating"])
POP = popularity_scores(TRAIN, INDEX, "count")


def _df(rows):
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"])


def _w(history, user, **kw):
    """Weights of one user as {movieId: weight}."""
    W = history_weights(history, np.array([user]), INDEX, **kw).toarray()[0]
    return {int(INDEX.ids[c]): float(W[c]) for c in np.flatnonzero(W)}


# --------------------------------------------------------------------------- #
# Profile weights
# --------------------------------------------------------------------------- #
def test_graded_weights_by_hand():
    # graded: w = rating - 4 + 1 for liked rows. user 1: 10 (5★) -> 2 ; 20 (2★) dropped
    assert _w(TRAIN, 1, weighting="graded") == {10: 2.0}
    # user 2: two 4★ -> 1 each (graded == liked when everything is 4★)
    assert _w(TRAIN, 2, weighting="graded") == {10: 1.0, 30: 1.0}
    h = _df([(9, 10, 4.5)])
    assert _w(h, 9, weighting="graded") == {10: 1.5}


def test_half_life_counts_liked_positions_only():
    # time order: 10 (5★), 20 (2★, not liked), 30 (4★)
    # liked rows from the end: 30 -> p=0, 10 -> p=1  (the 2★ row is NOT a position)
    h = _df([(1, 10, 5), (1, 20, 2), (1, 30, 4)])
    # half_life = 1: 30 -> 0.5^0 = 1 ; 10 -> 0.5^1 = 0.5
    assert _w(h, 1, weighting="liked", half_life=1) == {30: 1.0, 10: 0.5}
    # graded x decay: 30 -> 1 * 1 = 1 ; 10 -> 2 * 0.5 = 1
    assert _w(h, 1, weighting="graded", half_life=1) == {30: 1.0, 10: 1.0}
    # half_life = 2: 10 -> 0.5^(1/2) = 0.7071
    assert _w(h, 1, weighting="liked", half_life=2)[10] == pytest.approx(np.sqrt(0.5))
    # max_history = 1 keeps only the newest liked row
    assert _w(h, 1, weighting="liked", half_life=1, max_history=1) == {30: 1.0}
    # no decay -> unchanged Day 3/4 behaviour
    assert _w(h, 1, weighting="liked") == {10: 1.0, 30: 1.0}
    with pytest.raises(ValueError):
        _w(h, 1, half_life=-1)


def test_half_life_is_per_user():
    # two users interleaved by userId; positions restart for each user
    h = _df([(1, 10, 4), (1, 20, 4), (2, 30, 4)])
    assert _w(h, 1, half_life=1) == {20: 1.0, 10: 0.5}
    assert _w(h, 2, half_life=1) == {30: 1.0}


def _recs(history, user, **kw):
    model = EmbeddingRecommender(V, INDEX, threshold=4.0, **kw)
    users = np.array([user])
    P = model.user_profiles(history, users)
    seen = user_item_matrix(history, users, INDEX, dtype=np.int8)
    return P[0], model.recommend(P, seen, 3, POP)[0].tolist()


def test_decay_changes_the_ranking_by_hand():
    # user 5 liked 40 = (0.8, -0.6) first, then 20 = (0, 1) last. Unseen: 10, 30, 50.
    h = _df([(5, 40, 4), (5, 20, 4)])
    # no decay: profile ∝ (0.8, 0.4).  10: 0.8   30: 0.48 + 0.32 = 0.8   <- exact TIE
    #   popularity breaks it (10 more popular) -> 10, 30, 50
    _, recs = _recs(h, 5)
    assert recs == [10, 30, 50]
    # half_life = 1: w(20) = 1, w(40) = 0.5 -> profile ∝ (0.4, -0.3 + 1) = (0.4, 0.7)
    #   10: 0.4    30: 0.24 + 0.56 = 0.80    50: -0.4    -> the recent taste (20) wins: 30 first
    P, recs = _recs(h, 5, half_life=1)
    assert P == pytest.approx(np.array([0.4, 0.7]) / np.sqrt(0.65), abs=1e-6)
    assert recs == [30, 10, 50]


def test_graded_changes_the_ranking_by_hand():
    # same two movies, no decay, but 40 got 4★ and 20 got 5★
    # liked : weights 1, 1 -> the tie above -> 10, 30, 50
    # graded: weights 1, 2 -> profile ∝ (0.8, -0.6 + 2) = (0.8, 1.4)
    #   10: 0.8    30: 0.48 + 1.12 = 1.6    -> 30, 10, 50
    h = _df([(5, 40, 4), (5, 20, 5)])
    assert _recs(h, 5, weighting="liked")[1] == [10, 30, 50]
    assert _recs(h, 5, weighting="graded")[1] == [30, 10, 50]


# --------------------------------------------------------------------------- #
# HNSW vs Flat
# --------------------------------------------------------------------------- #
def test_recall_vs_exact_by_hand():
    exact = np.array([[1, 2, 3], [4, 5, 6]])
    approx = np.array([[3, 9, 1], [4, 5, 6]])
    # row 1: found 1 and 3 of {1, 2, 3} -> 2/3 ; row 2: 3/3 ; mean = (2/3 + 1) / 2 = 5/6
    assert recall_vs_exact(approx, exact, 3) == pytest.approx(5 / 6)
    # order inside the top-k does not matter
    assert recall_vs_exact(np.array([[3, 2, 1]]), np.array([[1, 2, 3]]), 3) == 1.0
    # k = 1 looks at the first column only: 3 vs 1 -> 0 ; 4 vs 4 -> 1
    assert recall_vs_exact(approx, exact, 1) == pytest.approx(0.5)
    # PAD in the exact list is ignored (only 1 true neighbour)
    assert recall_vs_exact(np.array([[1, 7]]), np.array([[1, PAD]]), 2) == 1.0


def test_hnsw_matches_flat_on_toy_vectors():
    # 5 items, every item is linked to every other -> the graph search is exhaustive = exact
    hnsw = HNSWIndex(V, INDEX.ids, M=4, ef_search=16)
    D, I = hnsw.search(np.array([1.0, 0.0]), k=3)
    assert I.tolist() == [[10, 40, 30]] and D[0] == pytest.approx([1.0, 0.8, 0.6])
    _, I = hnsw.similar([10], k=2)
    assert I[0].tolist() == [40, 30]


def test_hnsw_recall_and_knobs():
    rng = np.random.default_rng(0)
    X = l2_normalize(rng.normal(size=(2000, 32)))
    Q = l2_normalize(rng.normal(size=(50, 32)))
    ids = np.arange(2000) + 1
    _, exact = FlatIndex(X, ids).search(Q, 10)
    hnsw = HNSWIndex(X, ids, M=16, ef_construction=100)
    hnsw.ef_search = 2000                        # candidate list = whole catalog -> exact
    assert hnsw.ef_search == 2000
    assert recall_vs_exact(hnsw.search(Q, 10)[1], exact, 10) == 1.0
    hnsw.ef_search = 10                          # small list -> may miss some, never > 1
    r = recall_vs_exact(hnsw.search(Q, 10)[1], exact, 10)
    assert 0.5 < r <= 1.0
    assert hnsw.build_seconds > 0
    # the graph costs memory on top of the raw vectors (2000 x 32 x 4 bytes = 0.256 MB)
    assert index_size_mb(hnsw) > index_size_mb(FlatIndex(X, ids)) >= 0.256


def test_query_latency_ms_one_value_per_query():
    ms = query_latency_ms(FlatIndex(V, INDEX.ids), V, k=2, warmup=2)
    assert ms.shape == (5,) and np.all(ms >= 0)


# --------------------------------------------------------------------------- #
# Candidate mixing
# --------------------------------------------------------------------------- #
def test_dedupe_and_compact_by_hand():
    C = np.array([[5, 3, 5, PAD, 3], [1, 2, 3, 4, 5]])
    # later copies of 5 and 3 become PAD; the second row has no duplicates
    assert dedupe_rows(C).tolist() == [[5, 3, PAD, PAD, PAD], [1, 2, 3, 4, 5]]
    assert compact_rows(np.array([[5, PAD, 3, PAD]])).tolist() == [[5, 3, PAD, PAD]]


def test_mix_candidates_by_hand():
    emb = np.array([[1, 2, 3, 4]])
    pop = np.array([[2, 7, 1, 8, 9]])
    # 2 from embeddings (1, 2), then popular fill skipping 2 and 1 -> 7, 8
    assert mix_candidates(emb, pop, 2, 4).tolist() == [[1, 2, 7, 8]]
    assert mix_candidates(emb, pop, 0, 4).tolist() == [[2, 7, 1, 8]]    # popularity only
    assert mix_candidates(emb, pop, 4, 4).tolist() == [[1, 2, 3, 4]]    # embeddings only
    # not enough distinct items -> PAD at the end
    short = mix_candidates(np.array([[1, PAD]]), np.array([[1, PAD]]), 1, 3)
    assert short.tolist() == [[1, PAD, PAD]]
    with pytest.raises(ValueError):
        mix_candidates(emb, pop, 5, 4)


def test_candidate_recall_equals_union_logic():
    # a mixed set's recall can be checked with plain set logic
    rng = np.random.default_rng(3)
    emb = np.stack([rng.permutation(100) + 1 for _ in range(20)])
    pop = np.tile(np.arange(100) + 1, (20, 1))
    M = mix_candidates(emb, pop, 15, 30)
    for e, m in zip(emb, M):
        want = list(e[:15]) + [x for x in range(1, 101) if x not in set(e[:15])][:15]
        assert m.tolist() == want
