"""Day 8 tests: union of candidate sources, FAISS retrieval with seen filtering, ranker features.

Toy catalog (same as Days 3-6): movieIds 10, 20, 30, 40, 50 -> columns 0..4
  vectors  10 (1,0)   20 (0,1)   30 (0.6,0.8)   40 (0.8,-0.6)   50 (-1,0)
  genres   10 A       20 B       30 A,B         40 A            50 —
  years    1990       2000       2010           2000            unknown
Expected numbers are worked out by hand in the comments.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.baselines import ItemIndex, user_item_matrix  # noqa: E402
from recsys.candidates import faiss_candidates, positions_in, union_candidates  # noqa: E402
from recsys.embeddings import FlatIndex  # noqa: E402
from recsys.features import (FEATURE_NAMES, FeatureBuilder, assert_no_label_in_history,  # noqa: E402
                             attach_labels, item_statistics, oracle_topk, rank_by_feature)
from recsys.metrics import PAD  # noqa: E402

INDEX = ItemIndex([10, 20, 30, 40, 50])
V = np.array([[1, 0], [0, 1], [0.6, 0.8], [0.8, -0.6], [-1, 0]], dtype=np.float32)
GENRES = np.array([[1, 0], [0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)   # columns A, B
YEARS = np.array([1990, 2000, 2010, 2000, np.nan])
D = 86_400


def nan_eq(a, b):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), equal_nan=True, atol=1e-5)


# --------------------------------------------------------------------------- #
# union of sources
# --------------------------------------------------------------------------- #
def test_union_keeps_popular_first_then_faiss_only_items():
    pop = np.array([[10, 20, 30], [20, 10, 40]])
    faiss = np.array([[30, 50, PAD], [40, 50, 30]])
    C, ranks = union_candidates({"pop": pop, "faiss": faiss}, {"pop": 2, "faiss": 2})
    # row 0: pop top-2 = 10, 20; faiss top-2 = 30, 50 -> all new
    assert C.tolist() == [[10, 20, 30, 50], [20, 10, 40, 50]]
    assert nan_eq(ranks["pop"][0], [0, 1, np.nan, np.nan])
    assert nan_eq(ranks["faiss"][0], [np.nan, np.nan, 0, 1])


def test_union_overlap_is_counted_once_and_ranked_in_both():
    pop = np.array([[20, 10, 40]])
    faiss = np.array([[40, 50, 30]])
    C, ranks = union_candidates({"pop": pop, "faiss": faiss}, {"pop": 3, "faiss": 2})
    # 40 is in both lists: kept once (where popularity put it), PAD fills the freed slot
    assert C.tolist() == [[20, 10, 40, 50, PAD]]
    assert nan_eq(ranks["pop"], [[0, 1, 2, np.nan, np.nan]])
    assert nan_eq(ranks["faiss"], [[np.nan, np.nan, 0, 1, np.nan]])


def test_union_with_zero_from_a_source():
    C, ranks = union_candidates({"pop": np.array([[10, 20]]), "faiss": np.array([[30, 40]])},
                                {"pop": 2, "faiss": 0})
    assert C.tolist() == [[10, 20]] and np.isnan(ranks["faiss"]).all()


def test_positions_in_by_hand():
    C = np.array([[30, 10, PAD]])
    src = np.array([[10, 50, 30]])
    assert nan_eq(positions_in(C, src), [[2, 0, np.nan]])


# --------------------------------------------------------------------------- #
# FAISS retrieval minus seen movies
# --------------------------------------------------------------------------- #
def test_faiss_candidates_skip_seen_and_match_dense_ranking():
    index = FlatIndex(V, INDEX.ids)
    P = np.array([[1, 0], [0.6, 0.8], [0, 0]], dtype=np.float32)
    hist = pd.DataFrame({"userId": [1, 2, 2, 2], "movieId": [10, 30, 20, 10]})
    seen = user_item_matrix(hist, np.array([1, 2, 3]), INDEX, dtype=np.int8)
    out = faiss_candidates(index, P, seen, m=3, batch_size=2)
    # user 1, query (1,0): 10 1.0 (seen), 40 0.8, 30 0.6, 20 0.0, 50 -1.0  -> 40, 30, 20
    # user 2, query (0.6,0.8): 30 1.0, 20 0.8, 10 0.6 (all seen), 40 0.0, 50 -0.6 -> 40, 50, PAD
    #   (needs k = 3 + 3 seen = 6 > 5 movies: capped at 5, the 3rd slot stays empty)
    # user 3: no liked movie -> no query -> all PAD
    assert out.tolist() == [[40, 30, 20], [40, 50, PAD], [PAD, PAD, PAD]]
    # same as scoring all movies densely and masking seen ones
    S = P[:2] @ V.T
    S[seen[:2].toarray() > 0] = -np.inf
    dense = INDEX.ids[np.argsort(-S, axis=1)[:, :3]]
    assert out[0].tolist() == dense[0].tolist()


# --------------------------------------------------------------------------- #
# item statistics
# --------------------------------------------------------------------------- #
HIST = pd.DataFrame({
    "userId":    [1, 1, 2, 2, 3, 3],
    "movieId":   [10, 20, 10, 30, 10, 40],
    "rating":    [5.0, 2.0, 4.0, 4.0, 3.0, 5.0],
    "timestamp": [0, 1 * D, 5 * D, 9 * D, int(9.5 * D), 10 * D],
})


def test_item_statistics_by_hand():
    st = item_statistics(HIST, INDEX, threshold=4.0, recent_days=1, prior_votes=2)
    # global mean = (5+2+4+4+3+5)/6 = 23/6
    g = 23 / 6
    assert st["count"].tolist() == [3, 1, 1, 1, 0]
    assert st["liked"].tolist() == [2, 0, 1, 1, 0]                # 10: 5★ and 4★
    # movie 10: (5+4+3 + 2·23/6) / (3+2) = 3.9333;  movie 50 has no rating -> the global mean
    assert np.isclose(st["avg_rating"][0], (12 + 2 * g) / 5)
    assert np.isclose(st["avg_rating"][4], g)
    # recent = liked in (day 9, day 10]: only 40 (5★ on day 10). 30 is day 9 exactly -> out;
    # 10 on day 9.5 is a 3★ -> not liked
    assert np.isclose(np.expm1(st["recent_log"]), [0, 0, 0, 1, 0]).all()
    assert st["trend"].tolist() == [0, 0, 0, 1, 0]
    assert st["cutoff"] == 10 * D


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def builder():
    return FeatureBuilder(V, INDEX, GENRES, YEARS, runtime=np.array([100, 90, 120, np.nan, 80.0]),
                          center=False, weighting="liked", half_life=None,
                          recent_days=1, prior_votes=2).fit(HIST)


def test_features_for_one_user_by_hand():
    b = builder()
    users = np.array([1])
    C = np.array([[30, 40, 50]])
    C, ranks = union_candidates({"pop": np.array([[30, 40]]), "faiss": np.array([[50, 40]])},
                                {"pop": 2, "faiss": 2})
    assert C.tolist() == [[30, 40, 50, PAD]]
    fb = b.transform(users, HIST[HIST.userId == 1], C, ranks)
    X = {k: v[0, :3] for k, v in fb.X.items()}
    # user 1 rated 10 (5★, liked) and 20 (2★): profile = v10 = (1, 0)
    assert nan_eq(X["cos_profile"], [0.6, 0.8, -1.0])
    assert nan_eq(X["cos_longterm"], [0.6, 0.8, -1.0])
    assert nan_eq(X["max_cos_liked"], [0.6, 0.8, -1.0])
    # liked genres: A in 1 of 1 liked movie, B in 0 -> 30 (A,B): 1·½ + 0·½ = 0.5; 40 (A): 1; 50: 0
    assert nan_eq(X["genre_affinity"], [0.5, 1.0, 0.0])
    # mean year of liked = 1990 -> |2010-1990|, |2000-1990|, unknown
    assert nan_eq(X["year_gap"], [20, 10, np.nan])
    # user: 2 ratings, mean 3.5, half liked
    assert nan_eq(X["user_n_ratings_log"], [np.log1p(2)] * 3)
    assert nan_eq(X["user_mean_rating"], [3.5] * 3)
    assert nan_eq(X["user_liked_share"], [0.5] * 3)
    # sources: 40 is in both lists
    assert nan_eq(X["in_pop"], [1, 1, 0]) and nan_eq(X["pop_rank"], [0, 1, np.nan])
    assert nan_eq(X["in_faiss"], [0, 1, 1]) and nan_eq(X["faiss_rank"], [np.nan, 1, 0])
    # popularity gap = log(1+liked(j)) - log(1+liked(10)) = log2 - log3 for 30 and 40
    assert nan_eq(X["pop_gap"], [np.log(2) - np.log(3)] * 2 + [-np.log(3)])
    assert nan_eq(X["item_runtime"], [120, np.nan, 80])
    # the PAD slot is NaN in every feature
    assert all(np.isnan(v[0, 3]) for v in fb.X.values())
    df = fb.to_frame()
    assert df.columns.tolist() == ["userId", "movieId"] + FEATURE_NAMES and len(df) == 3


def test_max_cos_uses_the_closest_liked_movie_not_the_average():
    b = builder()
    hist = pd.DataFrame({"userId": [7, 7], "movieId": [10, 20], "rating": [5.0, 5.0],
                         "timestamp": [0, 1]})
    C = np.array([[30, 40, 50]])
    fb = b.transform(np.array([7]), hist, C, {})
    # liked 10 (1,0) and 20 (0,1): profile = (1,1)/√2
    # 30: profile 1.4/√2 = 0.9899; max(0.6, 0.8) = 0.8
    # 40: 0.2/√2 = 0.1414;         max(0.8, -0.6) = 0.8
    assert nan_eq(fb.X["cos_profile"][0], [1.4 / np.sqrt(2), 0.2 / np.sqrt(2), -1 / np.sqrt(2)])
    assert nan_eq(fb.X["max_cos_liked"][0], [0.8, 0.8, 0.0])
    # no source ranks given -> not retrieved by either
    assert np.isnan(fb.X["pop_rank"]).all() and (fb.X["in_pop"] == 0).all()


def test_user_with_no_liked_movie_gets_nan_semantic_features():
    b = builder()
    hist = pd.DataFrame({"userId": [3], "movieId": [20], "rating": [2.0], "timestamp": [0]})
    fb = b.transform(np.array([3]), hist, np.array([[10, 30]]), {})
    for f in ("cos_profile", "cos_longterm", "max_cos_liked", "year_gap", "pop_gap"):
        assert np.isnan(fb.X[f]).all(), f
    assert (fb.X["genre_affinity"] == 0).all()


def test_features_do_not_depend_on_the_batch():
    """A user's features are the same alone or in a batch with others (no cross-user mixing)."""
    b = builder()
    C = np.array([[30, 40, 50], [20, 40, 50], [20, 30, 50]])
    batch = b.transform(np.array([1, 2, 3]), HIST, C, {})
    alone = b.transform(np.array([2]), HIST[HIST.userId == 2], C[1:2], {})
    for f in FEATURE_NAMES:
        assert nan_eq(batch.X[f][1], alone.X[f][0]), f


def test_item_features_come_only_from_the_history_passed_to_fit():
    """Leakage guard: adding label-window ratings to the data must not change features unless
    they are passed to fit() — and the script only ever passes the history view."""
    b1 = builder()
    future = pd.DataFrame({"userId": [9, 9], "movieId": [30, 30], "rating": [5.0, 5.0],
                           "timestamp": [20 * D, 21 * D]})
    b2 = FeatureBuilder(V, INDEX, GENRES, YEARS, center=False, weighting="liked", half_life=None,
                        recent_days=1, prior_votes=2).fit(pd.concat([HIST, future]))
    C = np.array([[30]])
    f1 = b1.transform(np.array([1]), HIST[HIST.userId == 1], C, {})
    f2 = b2.transform(np.array([1]), HIST[HIST.userId == 1], C, {})
    assert f1.X["item_liked_log"][0, 0] == pytest.approx(np.log(2))       # 1 liked rating
    assert f2.X["item_liked_log"][0, 0] == pytest.approx(np.log(4))       # 3 with the future
    assert f1.X["cos_profile"][0, 0] == f2.X["cos_profile"][0, 0]          # user side unchanged


def test_assert_no_label_in_history():
    truth = pd.DataFrame({"userId": [1], "movieId": [30], "rating": [5.0]})
    assert_no_label_in_history(HIST, truth)                                  # fine
    with pytest.raises(ValueError):
        assert_no_label_in_history(HIST, pd.DataFrame({"userId": [1], "movieId": [10]}))


# --------------------------------------------------------------------------- #
# labels, one-feature ranking, oracle
# --------------------------------------------------------------------------- #
def test_attach_labels_zero_when_not_rated():
    df = pd.DataFrame({"userId": [1, 1], "movieId": [30, 40]}).astype(np.int32)
    out = attach_labels(df, pd.DataFrame({"userId": [1], "movieId": [40], "rating": [4.5]}))
    assert out["rating"].tolist() == [0.0, 4.5]


def test_rank_by_feature_and_oracle():
    b = builder()
    C = np.array([[30, 40, 50, PAD]])
    fb = b.transform(np.array([1]), HIST[HIST.userId == 1], C, {})
    # cos_profile 30 0.6, 40 0.8, 50 -1.0 -> 40, 30, 50 ; PAD never ranked
    assert rank_by_feature(fb, "cos_profile", 4).tolist() == [[40, 30, 50, PAD]]
    # year_gap: lower is better, NaN last -> 40 (10), 30 (20), 50 (NaN -> PAD in the top list)
    assert rank_by_feature(fb, "year_gap", 3).tolist() == [[40, 30, PAD]]
    # oracle: the held-out 5★ first, then the 4★, then the rest in candidate order
    truth = pd.DataFrame({"userId": [1, 1], "movieId": [50, 30], "rating": [4.0, 5.0]})
    assert oracle_topk(fb, truth, 3).tolist() == [[30, 50, 40]]
