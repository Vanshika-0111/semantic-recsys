"""Baseline tests. Expected values are worked out BY HAND in the comments.

Toy catalog: movieIds 10, 20, 30, 40, 50  ->  columns 0, 1, 2, 3, 4
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.baselines import (ItemIndex, PopularityRecommender,  # noqa: E402
                              TfidfKNNRecommender, mask_seen, popularity_scores, top_neighbors,
                              topk, user_item_matrix)
from recsys.metrics import PAD  # noqa: E402

INDEX = ItemIndex([50, 10, 40, 20, 30])          # unsorted on purpose -> ids sorted


def _df(rows):
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"])


# user 1: 10 (5), 20 (2)   user 2: 10 (4), 30 (4)   user 3: 10 (3), 40 (5)
TRAIN = _df([(1, 10, 5), (1, 20, 2), (2, 10, 4), (2, 30, 4), (3, 10, 3), (3, 40, 5)])


def test_item_index():
    assert list(INDEX.ids) == [10, 20, 30, 40, 50]
    assert list(INDEX.cols([30, 10])) == [2, 0]
    with pytest.raises(KeyError):
        INDEX.cols([99])


def test_popularity_count_and_liked():
    # count : 10 -> 3, 20/30/40 -> 1, 50 -> 0. Ties by movieId: order 10, 20, 30, 40, 50
    s = popularity_scores(TRAIN, INDEX, "count")
    assert list(INDEX.ids[np.argsort(-s)]) == [10, 20, 30, 40, 50]
    assert list(s) == pytest.approx([5 / 5, 4 / 5, 3 / 5, 2 / 5, 1 / 5])
    # liked (rating >= 4): 10 -> 2 (users 1, 2), 30 -> 1, 40 -> 1, 20 -> 0, 50 -> 0
    s = popularity_scores(TRAIN, INDEX, "liked")
    assert list(INDEX.ids[np.argsort(-s)]) == [10, 30, 40, 20, 50]


def test_popularity_excludes_seen_items():
    users = np.array([1, 3])
    seen = user_item_matrix(TRAIN, users, INDEX, dtype=np.int8)
    recs = PopularityRecommender("count").fit(TRAIN, INDEX).recommend(seen, k=3)
    # user 1 saw 10, 20 -> next most popular: 30, 40, 50
    # user 3 saw 10, 40 -> 20, 30, 50
    assert recs.tolist() == [[30, 40, 50], [20, 30, 50]]


def test_topk_pads_when_everything_is_seen():
    s = np.array([[0.1, 0.9, 0.5, 0.0, 0.2]], dtype=np.float32)
    seen = sp.csr_matrix(np.array([[1, 1, 0, 1, 1]], dtype=np.int8))
    mask_seen(s, seen)
    assert topk(s, 3, INDEX).tolist() == [[30, PAD, PAD]]


def test_top_neighbors():
    # unit vectors: a=[1,0,0] b=[.8,.6,0] c=[0,1,0] d=[0,0,1]
    # cos(a,b)=.8  cos(b,c)=.6  cos(a,c)=0  d is orthogonal to everything
    X = sp.csr_matrix(np.array([[1, 0, 0], [.8, .6, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32))
    S = top_neighbors(X, n_neighbors=1).toarray()
    expected = np.zeros((4, 4))
    expected[0, 1] = expected[1, 0] = 0.8      # a <-> b
    expected[2, 1] = 0.6                       # c -> b  (b's best is a, not c)
    assert S == pytest.approx(expected)        # d keeps nothing (only zero similarities)
    assert np.all(np.diag(top_neighbors(X, 3).toarray()) == 0)   # never its own neighbour


# Hand-made similarity graph between the 5 movies (row i -> neighbours of i)
SIM = sp.csr_matrix(np.array([
    #  10   20   30   40   50
    [0.0, 0.5, 0.2, 0.0, 0.0],   # 10
    [0.5, 0.0, 0.0, 0.1, 0.0],   # 20
    [0.2, 0.0, 0.0, 0.9, 0.0],   # 30
    [0.0, 0.1, 0.9, 0.0, 0.0],   # 40
    [0.0, 0.0, 0.0, 0.0, 0.0],   # 50 (no text neighbours)
], dtype=np.float32))
# user 7 rated, in time order: 10 (5), 20 (4), 40 (1)
HIST = _df([(7, 10, 5.0), (7, 20, 4.0), (7, 40, 1.0)])
FALLBACK = np.array([5, 4, 3, 2, 1], dtype=float) / 5         # popularity order 10 > ... > 50


def _model(**kw):
    return TfidfKNNRecommender(**kw).set_similarity(SIM, INDEX)


def test_tfidf_liked_weights_and_scores():
    m = _model(weighting="liked")
    W = m.user_weights(HIST, np.array([7]))
    assert W.toarray().tolist() == [[1, 1, 0, 0, 0]]           # only 10 and 20 are liked
    seen = user_item_matrix(HIST, np.array([7]), INDEX, dtype=np.int8)
    # scores = S[10] + S[20] = [.5, .5, .2, .1, 0]; seen 10, 20, 40 removed
    # -> 30 (0.2), then 50 (0 + popularity tie-break)
    assert m.recommend(W, seen, 3, FALLBACK).tolist() == [[30, 50, PAD]]


def test_tfidf_centered_dislike_pushes_items_down():
    m = _model(weighting="centered")
    W = m.user_weights(HIST, np.array([7]))
    # mean = 10/3 -> weights 10: 5/3, 20: 2/3, 40: -7/3
    assert W.toarray()[0] == pytest.approx([5 / 3, 2 / 3, 0, -7 / 3, 0])
    seen = user_item_matrix(HIST, np.array([7]), INDEX, dtype=np.int8)
    # score(30) = 5/3*.2 + 2/3*0 - 7/3*.9 = -1.767  < score(50) = 0  -> 50 first
    assert m.recommend(W, seen, 2, FALLBACK).tolist() == [[50, 30]]


def test_tfidf_max_history_keeps_latest():
    W = _model(weighting="liked", max_history=1).user_weights(HIST, np.array([7]))
    assert W.toarray().tolist() == [[0, 1, 0, 0, 0]]           # last liked item = 20


def test_empty_profile_falls_back_to_popularity():
    m = _model()
    users = np.array([7, 8])                                    # user 8 has no history
    W = m.user_weights(HIST, users)
    seen = user_item_matrix(HIST, users, INDEX, dtype=np.int8)
    assert m.recommend(W, seen, 5, FALLBACK)[1].tolist() == [10, 20, 30, 40, 50]


def test_explain():
    m = _model()
    # why recommend 30 to someone who liked 10 and 40?  S[40, 30] = .9 > S[10, 30] = .2
    assert m.explain(np.array([0, 3]), item_col=2) == 3


def test_full_knn_equals_centroid_cosine():
    """With all neighbours kept, item-kNN ranks unseen items exactly like
    cosine(user TF-IDF centroid, item) — the claim in the module docstring."""
    rng = np.random.default_rng(0)
    X = rng.random((40, 15)) * (rng.random((40, 15)) < 0.3)
    X = X[np.linalg.norm(X, axis=1) > 0]
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n = X.shape[0]
    S = top_neighbors(sp.csr_matrix(X.astype(np.float32)), n_neighbors=n - 1)
    hist = rng.choice(n, size=6, replace=False)
    w = np.zeros(n); w[hist] = 1
    knn = (sp.csr_matrix(w) @ S).toarray().ravel()
    centroid = X[hist].sum(0)
    cos = X @ centroid / np.linalg.norm(centroid)
    unseen = np.setdiff1d(np.arange(n), hist)
    # same scores up to the constant 1/||centroid||  ->  identical ranking
    assert knn[unseen] / np.linalg.norm(centroid) == pytest.approx(cos[unseen], rel=1e-5, abs=1e-6)
