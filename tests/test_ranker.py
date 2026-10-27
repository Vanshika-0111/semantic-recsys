"""Day 9 tests: labels, user split, top-k by score, candidate_set, LambdaMART, TwoStage.

Toy catalog (same as Days 3-8): movieIds 10, 20, 30, 40, 50 -> columns 0..4
  vectors  10 (1,0)   20 (0,1)   30 (0.6,0.8)   40 (0.8,-0.6)   50 (-1,0)
Expected numbers are worked out by hand in the comments.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.baselines import ItemIndex, user_item_matrix  # noqa: E402
from recsys.candidates import candidate_set, union_candidates  # noqa: E402
from recsys.features import FEATURE_NAMES, rank_by_feature  # noqa: E402
from recsys.metrics import PAD, evaluate  # noqa: E402
from recsys.ranker import (GRADED_GAIN, RankData, build_two_stage, group_bounds,  # noqa: E402
                           label_gain, make_labels, pad_by_user, predict, split_users,
                           topk_by_score, train_ranker)

INDEX = ItemIndex([10, 20, 30, 40, 50])
V = np.array([[1, 0], [0, 1], [0.6, 0.8], [0.8, -0.6], [-1, 0]], dtype=np.float32)
GENRES = np.array([[1, 0], [0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
D = 86_400


def nan_eq(a, b):
    return np.allclose(np.asarray(a, float), np.asarray(b, float), equal_nan=True, atol=1e-6)


# --------------------------------------------------------------------------- #
# labels and the user split
# --------------------------------------------------------------------------- #
def test_labels_by_hand():
    r = np.array([0.0, 3.5, 4.0, 4.5, 5.0])        # 0 = not rated in the label window
    assert make_labels(r, "binary").tolist() == [0, 0, 1, 1, 1]
    assert make_labels(r, "graded").tolist() == [0, 0, 1, 2, 3]
    # graded gains = the project's graded NDCG gain (rating - 3) for relevant ratings
    lab = make_labels(r, "graded")
    assert [GRADED_GAIN[x] for x in lab[2:]] == [1.0, 1.5, 2.0]
    assert label_gain("binary") == [0.0, 1.0]
    with pytest.raises(ValueError):
        make_labels(r, "stars")


def test_split_users_disjoint_complete_and_seeded():
    users = np.arange(100, 200)
    tr, va = split_users(users[::-1], 0.25, seed=7)
    assert len(va) == 25 and len(tr) == 75
    assert np.intersect1d(tr, va).size == 0
    assert np.array_equal(np.union1d(tr, va), users)
    assert np.all(np.diff(tr) > 0) and np.all(np.diff(va) > 0)
    tr2, va2 = split_users(users, 0.25, seed=7)
    assert np.array_equal(va, va2)                  # same seed -> same split, any input order
    with pytest.raises(ValueError):
        split_users(users, 0.0, seed=7)


# --------------------------------------------------------------------------- #
# grouped rows -> per-user top-k
# --------------------------------------------------------------------------- #
def test_group_bounds_and_padding():
    u = np.array([3, 3, 3, 7, 7])
    users, starts, sizes = group_bounds(u)
    assert users.tolist() == [3, 7] and starts.tolist() == [0, 3] and sizes.tolist() == [3, 2]
    _, M = pad_by_user(u, np.array([1, 2, 3, 4, 5]), -1)
    assert M.tolist() == [[1, 2, 3], [4, 5, -1]]
    with pytest.raises(ValueError):
        group_bounds(np.array([3, 7, 3]))           # not grouped


def test_topk_by_score_ties_keep_row_order_and_nan_last():
    u = np.array([1, 1, 1, 2, 2])
    m = np.array([10, 20, 30, 40, 50])
    s = np.array([0.5, 0.9, 0.5, np.nan, 0.1])
    users, top = topk_by_score(u, m, s, k=3)
    # user 1: 20 (0.9), then the 0.5 tie in row order: 10 before 30
    # user 2: 50 (0.1), 40 (NaN -> last), then PAD
    assert users.tolist() == [1, 2]
    assert top.tolist() == [[20, 10, 30], [50, 40, PAD]]


# --------------------------------------------------------------------------- #
# candidate_set: union, or popularity with FAISS only as a feature
# --------------------------------------------------------------------------- #
def test_candidate_set_popularity_only_still_records_faiss_rank():
    pop = np.array([[20, 10, 40]])
    faiss = np.array([[40, 50, 30]])
    C, ranks = candidate_set(pop, faiss, n_pop=3, m_faiss=0, faiss_feature_m=2)
    # candidates = popularity only; FAISS top-2 is (40, 50) -> 40 is at FAISS position 0
    assert C.tolist() == [[20, 10, 40]]
    assert nan_eq(ranks["pop"], [[0, 1, 2]])
    assert nan_eq(ranks["faiss"], [[np.nan, np.nan, 0]])


def test_candidate_set_equals_day8_union():
    pop = np.array([[10, 20, 30], [20, 10, 40]])
    faiss = np.array([[30, 50, PAD], [40, 50, 30]])
    C1, r1 = candidate_set(pop, faiss, 2, 2)
    C2, r2 = union_candidates({"pop": pop, "faiss": faiss}, {"pop": 2, "faiss": 2})
    assert np.array_equal(C1, C2)
    assert all(nan_eq(r1[k], r2[k]) for k in ("pop", "faiss"))


# --------------------------------------------------------------------------- #
# LambdaMART on a planted signal
# --------------------------------------------------------------------------- #
def planted(n_users=300, n_cand=20, seed=0):
    """Feature 'good' > 0.8 <=> the user rated the movie >= 4; 'noise' is random.
    Rows grouped by user (like FeatureBatch.to_frame)."""
    rng = np.random.default_rng(seed)
    u = np.repeat(np.arange(1, n_users + 1), n_cand)
    m = np.tile(np.arange(1, n_cand + 1), n_users)
    good = rng.random(len(u)).astype(np.float32)
    rating = np.where(good > 0.8, rng.choice([4.0, 4.5, 5.0], len(u)), 0.0).astype(np.float32)
    df = pd.DataFrame({"userId": u, "movieId": m, "good": good,
                       "noise": rng.random(len(u)).astype(np.float32), "rating": rating})
    return RankData.from_frame(df, ["good", "noise"])


@pytest.mark.parametrize("scheme", ["binary", "graded"])
def test_lambdarank_learns_the_planted_feature_and_is_deterministic(scheme):
    d = planted()
    tr_u, va_u = split_users(np.unique(d.users), 0.3, seed=1)
    tr, va = d.subset(tr_u).drop_users_without_positives(), d.subset(va_u)
    cfg = {"labels": scheme, "num_leaves": 7, "min_data_in_leaf": 5}
    b1, info = train_ranker(tr, va, cfg, seed=3, max_rounds=60, early_stopping=20)
    s1 = predict(b1, va.X)
    users, top = topk_by_score(va.users, va.movies, s1, k=10)
    truth = pd.DataFrame({"userId": va.users, "movieId": va.movies, "rating": va.rating})
    truth = truth[truth["rating"] > 0]
    assert evaluate(users, top, truth, (10,))["ndcg@10"] > 0.95      # a perfect signal is found
    assert info["best_iter"] >= 1
    b2, _ = train_ranker(tr, va, cfg, seed=3, max_rounds=60, early_stopping=20)
    assert np.array_equal(s1, predict(b2, va.X))                      # same data -> same model


def test_rankdata_subset_and_drop_empty_queries():
    d = planted(n_users=4, n_cand=5)
    d.rating[:] = 0
    d.rating[[1, 12]] = 4.0                          # users 1 and 3 have a relevant row
    kept = d.drop_users_without_positives()
    assert np.unique(kept.users).tolist() == [1, 3] and len(kept) == 10
    assert d.columns(["noise"]).X.shape == (20, 1)


# --------------------------------------------------------------------------- #
# TwoStage on the toy catalog
# --------------------------------------------------------------------------- #
def toy_two_stage(booster=None):
    items = pd.DataFrame({"movieId": INDEX.ids, "year": [1990, 2000, 2010, 2000, np.nan],
                          "runtime": [100, 90, 0, 120, 95]})
    cat = SimpleNamespace(index=INDEX, items=items, genres=GENRES)
    chosen = {"popularity_mode": "liked", "center": False,
              "profile": {"weighting": "liked", "half_life": None}}
    # popularity (liked counts in this history): 10: 2, 20: 2, 30: 1, 40: 0, 50: 0
    history = pd.DataFrame({"userId": [1, 1, 2, 2, 2], "movieId": [10, 20, 10, 20, 30],
                            "rating": [5.0, 4.0, 4.0, 4.5, 5.0],
                            "timestamp": [1 * D, 2 * D, 1 * D, 2 * D, 3 * D]})
    ts = build_two_stage(cat, V, chosen, history, n_pop=2, m_faiss=1, faiss_feature_m=1,
                         booster=booster)
    return ts, history


def test_two_stage_without_booster_is_the_popularity_baseline():
    ts, hist = toy_two_stage()
    users = np.array([1])
    # user 1 rated 10, 20 -> popular unseen: 30, then 40 vs 50 tie (0 likes) -> 40 (smaller id)
    # FAISS top-1 for profile (1,1)/√2: 30 cos 0.99 (already popular) -> union [30, 40] + PAD
    rec = ts.recommend(hist[hist.userId == 1], users, k=2)
    seen = user_item_matrix(hist[hist.userId == 1], users, INDEX, dtype=np.int8)
    assert rec.tolist() == ts.pop.recommend(seen, 2).tolist() == [[30, 40]]


class ColumnBooster:
    """Stand-in for a LightGBM booster: the score IS one feature column."""
    best_iteration = 0

    def __init__(self, col: int):
        self.col = col

    def predict(self, X, num_iteration=None):
        return np.nan_to_num(X[:, self.col], nan=-1e9)


def test_two_stage_scores_use_the_right_feature_columns():
    j = FEATURE_NAMES.index("cos_profile")
    ts, hist = toy_two_stage(ColumnBooster(j))
    users = np.array([2])
    h = hist[hist.userId == 2]
    fb = ts.feature_batch(h, users)
    # the batch path (Day 10 / API) must rank exactly like sorting by that one feature
    assert ts.recommend(h, users, k=2).tolist() == rank_by_feature(fb, "cos_profile", 2).tolist()
    # and like the long-format path used in training (to_frame -> RankData -> predict)
    rd = RankData.from_frame(fb.to_frame())
    _, top = topk_by_score(rd.users, rd.movies, predict(ts.booster, rd.X), k=2)
    assert top.tolist() == ts.recommend(h, users, k=2).tolist()
