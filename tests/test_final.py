"""Day 10 tests: ablation feature sets, release-aware popularity, the cold policy, user-aligned
bootstrap, TreeSHAP helpers, multi-variant feature batches.

Toy catalog (same as Days 3-9): movieIds 10, 20, 30, 40, 50 -> columns 0..4
  vectors  10 (1,0)   20 (0,1)   30 (0.6,0.8)   40 (0.8,-0.6)   50 (-1,0)
Expected numbers are worked out by hand in the comments.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys import config  # noqa: E402
from recsys.baselines import ItemIndex, PopularityRecommender, user_item_matrix  # noqa: E402
from recsys.evaluation import (FEATURE_GROUPS, GROUP_OF, SEMANTIC_FEATURES, YEAR_FEATURES,  # noqa: E402
                               compare, group_shares, policy_model, project_file,
                               release_aware_topk, released_after, shap_table, significant,
                               timestamp_year, top_reasons, tree_shap, user_cutoff_year, without)
from recsys.features import FEATURE_NAMES  # noqa: E402
from recsys.metrics import PAD  # noqa: E402
from recsys.ranker import RankData, predict, split_users, top_from_scores, train_ranker  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ranker import ColumnBooster, planted, toy_two_stage  # noqa: E402

INDEX = ItemIndex([10, 20, 30, 40, 50])
Y2000 = 946_684_800                       # 2000-01-01 00:00:00 UTC
YEARS = np.array([1990, 2005, 2010, np.nan, 1980])   # release years of 10, 20, 30, 40, 50


# --------------------------------------------------------------------------- #
# feature sets
# --------------------------------------------------------------------------- #
def test_groups_cover_every_feature_once_and_ablations():
    flat = [f for fs in FEATURE_GROUPS.values() for f in fs]
    assert sorted(flat) == sorted(FEATURE_NAMES) and len(flat) == len(set(flat))
    assert set(SEMANTIC_FEATURES) == {"in_faiss", "faiss_rank", "cos_profile", "cos_longterm", "max_cos_liked"}
    nosem = without(SEMANTIC_FEATURES)
    assert len(nosem) == len(FEATURE_NAMES) - 5 and not set(nosem) & set(SEMANTIC_FEATURES)
    assert nosem == [f for f in FEATURE_NAMES if f in nosem]          # order kept
    assert "year_gap" not in without(YEAR_FEATURES)
    with pytest.raises(ValueError):
        without(["cos_profil"])                                     # typo guard
    assert GROUP_OF["pop_gap"] == "user-item metadata"


# --------------------------------------------------------------------------- #
# release-date-aware popularity
# --------------------------------------------------------------------------- #
def test_timestamp_year_and_user_cutoff():
    assert timestamp_year([0, Y2000 - 1, Y2000]).tolist() == [1970, 1999, 2000]
    h = pd.DataFrame({"userId": [1, 1, 2], "timestamp": [0, Y2000, Y2000 - 1]})
    cut = user_cutoff_year(h, np.array([1, 2, 3]))
    assert cut[:2].tolist() == [2000, 1999] and np.isnan(cut[2])   # user 3: no history -> NaN


def test_released_after_by_hand():
    C = np.array([[30, 50, PAD], [20, 40, 10]])
    # user A cutoff 2000: 30 (2010) late, 50 (1980) ok, PAD never; user B cutoff NaN -> nothing late
    m = released_after(C, INDEX, YEARS, np.array([2000.0, np.nan]))
    assert m.tolist() == [[True, False, False], [False, False, False]]


def popularity_toy():
    # liked counts: 30: 3, 10: 2, 40: 1, 20 and 50: 0 -> order 30, 10, 40, 20, 50 (tie -> smaller id)
    train = pd.DataFrame({"userId": [9, 8, 7, 9, 8, 9], "movieId": [30, 30, 30, 10, 10, 40],
                          "rating": [5.0, 4.0, 4.0, 4.0, 5.0, 4.5]})
    return PopularityRecommender("liked", 4.0).fit(train, INDEX)


@pytest.mark.parametrize("pool", [2, 5])      # pool 2 forces the whole-catalog fallback
def test_release_aware_popularity_by_hand(pool):
    pop = popularity_toy()
    hist = pd.DataFrame({"userId": [1], "movieId": [10], "rating": [4.0]})
    users = np.array([1, 2])
    seen = user_item_matrix(hist, users, INDEX, dtype=np.int8)
    # user 1 (cutoff 2000, saw 10): 30 late, 10 seen, 40 unknown year -> kept, 20 late, 50 ok
    # user 2 (cutoff +inf): plain popularity
    top = release_aware_topk(pop.score, INDEX, seen, YEARS, np.array([2000.0, np.inf]), k=3, pool=pool)
    assert top.tolist() == [[40, 50, PAD], [30, 10, 40]]


def test_release_aware_with_no_cutoff_is_exactly_popularity():
    pop = popularity_toy()
    hist = pd.DataFrame({"userId": [1, 2, 2], "movieId": [30, 40, 50], "rating": [4.0, 2.0, 5.0]})
    users = np.array([1, 2, 3])
    seen = user_item_matrix(hist, users, INDEX, dtype=np.int8)
    top = release_aware_topk(pop.score, INDEX, seen, YEARS, np.full(3, np.inf), k=4, pool=2)
    assert top.tolist() == pop.recommend(seen, 4).tolist()


# --------------------------------------------------------------------------- #
# cold policy, top-k with a filter
# --------------------------------------------------------------------------- #
def test_policy():
    picks = [policy_model(n, 3) for n in (0, 1, 2, 3, 5)]
    assert picks == ["popularity", "cold", "cold", "warm", "warm"]


def test_top_from_scores_filtered_slots_become_pad():
    C = np.array([[10, 20, 30]])
    S = np.array([[0.2, -np.inf, 0.2]])         # 20 filtered out; tie 10/30 keeps candidate order
    assert top_from_scores(C, S, 3).tolist() == [[10, 30, PAD]]


# --------------------------------------------------------------------------- #
# bootstrap on the SAME users
# --------------------------------------------------------------------------- #
def test_compare_matches_users_by_id():
    a = pd.DataFrame({"userId": [1, 2, 3], "ndcg@10": [0.5, 0.2, 0.0]})
    b = pd.DataFrame({"userId": [3, 2, 1, 4], "ndcg@10": [0.0, 0.1, 0.5, 0.9]})   # other order, +1 user
    r = compare(a, b)
    # paired by user: diffs 0, 0.1, 0 -> mean 0.0333; user 4 is not in a -> not compared
    assert r["n_users"] == 3 and r["diff"] == pytest.approx(0.1 / 3, abs=1e-6)
    assert r["mean_b"] == pytest.approx(0.2, abs=1e-6)
    with pytest.raises(ValueError):
        compare(a, pd.DataFrame({"userId": [7], "ndcg@10": [0.1]}))


def test_significant_needs_ci_above_zero():
    assert significant({"p_le_0": 0.01, "ci95": [0.001, 0.01]})
    assert not significant({"p_le_0": 0.01, "ci95": [0.0, 0.01]})
    assert not significant({"p_le_0": 0.2, "ci95": [-0.01, 0.01]})


# --------------------------------------------------------------------------- #
# SHAP
# --------------------------------------------------------------------------- #
def test_tree_shap_rows_sum_to_the_raw_score():
    d = planted(n_users=120, n_cand=10)
    tr_u, va_u = split_users(np.unique(d.users), 0.3, seed=1)
    b, _ = train_ranker(d.subset(tr_u).drop_users_without_positives(), d.subset(va_u),
                        {"labels": "binary", "num_leaves": 7, "min_data_in_leaf": 5}, 3, 30, 10)
    X = d.subset(va_u).X
    c = tree_shap(b, X, batch=100)
    assert c.shape == (len(X), 3)                                     # 2 features + expected value
    assert np.allclose(c.sum(axis=1), predict(b, X), atol=1e-6)       # additivity
    assert np.abs(c[:, 0]).mean() > np.abs(c[:, 1]).mean()            # the planted feature dominates


def test_shap_table_groups_and_reasons_by_hand():
    names = ["cos_profile", "pop_rank", "year_gap"]
    # 2 rows: |shap| sums 0.6, 0.3, 0.1 -> mean 0.3, 0.15, 0.05 -> shares 0.6, 0.3, 0.1
    t = shap_table(np.array([0.6, 0.3, 0.1]), np.array([0.4, -0.3, 0.0]), 2, names)
    assert [r["feature"] for r in t] == names and [r["share"] for r in t] == [0.6, 0.3, 0.1]
    assert t[1]["mean_shap"] == -0.15 and t[0]["group"] == "semantic (embeddings / FAISS)"
    g = group_shares(t)
    assert g[0] == {"group": "semantic (embeddings / FAISS)", "share": 0.6}
    # one candidate: contributions +0.5, -0.2, +0.1 (+ expected value column)
    rs = top_reasons(np.array([0.5, -0.2, 0.1, 9.9]), np.array([0.81, 3.0, np.nan]), names, n=3)
    assert [r["feature"] for r in rs] == ["cos_profile", "year_gap"]   # negative one left out
    assert rs[0]["value"] == 0.81 and rs[1]["value"] is None


def test_project_file_reads_windows_and_posix_paths():
    assert project_file("data\\runs\\day09_ranker.txt") == config.ROOT / "data" / "runs" / "day09_ranker.txt"
    assert project_file("data/runs/day09_ranker.txt") == config.ROOT / "data" / "runs" / "day09_ranker.txt"


# --------------------------------------------------------------------------- #
# one stage-1 pass, several candidate variants
# --------------------------------------------------------------------------- #
def test_feature_batches_equal_single_variant_batches():
    import dataclasses

    ts, hist = toy_two_stage(ColumnBooster(FEATURE_NAMES.index("cos_profile")))
    users = np.array([1, 2])
    variants = {"union": (2, 1, 1), "pop3": (3, 0, 1), "wide": (1, 2, 2)}
    many = ts.feature_batches(hist, users, variants)
    for name, (n, m, f) in variants.items():
        one = dataclasses.replace(ts, n_pop=n, m_faiss=m, faiss_feature_m=f).feature_batch(hist, users)
        assert np.array_equal(many[name].candidates, one.candidates), name
        for feat in FEATURE_NAMES:
            assert np.allclose(many[name].X[feat], one.X[feat], equal_nan=True), (name, feat)
    # the score path with an explicit booster = scores() of the TwoStage
    fb = many["union"]
    assert np.array_equal(ts.score_with(fb, ts.booster, ts.features), ts.scores(fb))
    # no booster -> union order (popular first)
    S0 = ts.score_with(fb, None, [])
    assert top_from_scores(fb.candidates, S0, 3).tolist()[0][:1] == [fb.candidates[0, 0]]


def test_rankdata_columns_used_by_ablation_keep_rows():
    d = planted(n_users=3, n_cand=4)
    sub = d.columns(["noise"])
    assert sub.X.shape == (12, 1) and np.array_equal(sub.users, d.users)
    assert np.array_equal(sub.X[:, 0], d.X[:, 1])
    assert isinstance(sub, RankData)
