"""Day 6 tests: popularity priors (all-time + time-aware), pseudo-cold seeds, text queries,
genre onboarding, pool + rerank.

Same toy catalog as test_embeddings.py / test_retrieval.py (unit vectors, cosine = dot product):
    10: ( 1.0,  0.0)      20: ( 0.0, 1.0)      30: ( 0.6, 0.8)
    40: ( 0.8, -0.6)      50: (-1.0, 0.0)
Toy ratings: u1: 10★5, 20★2 · u2: 10★4, 30★4 · u3: 10★3, 40★5
Popularity (count) order: 10, 20, 30, 40, 50   (ties -> smaller movieId first)
Liked (>= 4) counts:      10 -> 2, 30 -> 1, 40 -> 1, 20 -> 0, 50 -> 0
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.baselines import (ItemIndex, PopularityRecommender, popularity_scores,  # noqa: E402
                              user_item_matrix)
from recsys.coldstart import (DAY, ColdStartRecommender, TimePopularity, blend_topk,  # noqa: E402
                              combine_profiles, first_liked, genre_match, genre_query,
                              log_scale, onboarding_genres, onboarding_time, pool_rerank_topk,
                              popularity_prior, query_profiles)
from recsys.metrics import PAD  # noqa: E402
from recsys.embeddings import center_vectors  # noqa: E402

INDEX = ItemIndex([10, 20, 30, 40, 50])
V = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.8, -0.6], [-1.0, 0.0]], dtype=np.float32)
TRAIN = pd.DataFrame([(1, 10, 5), (1, 20, 2), (2, 10, 4), (2, 30, 4), (3, 10, 3), (3, 40, 5)],
                     columns=["userId", "movieId", "rating"])
POP = popularity_scores(TRAIN, INDEX, "count")
PRIOR = popularity_prior(TRAIN, INDEX)
L23 = np.log(2) / np.log(3)          # log(1+1) / log(1+2) = 0.6309


def _seen(rows):
    """rows: one list of seen movieIds per user (in user order 0, 1, ...)."""
    df = pd.DataFrame([(u, m) for u, ms in enumerate(rows) for m in ms], columns=["userId", "movieId"])
    return user_item_matrix(df, np.arange(len(rows)), INDEX, dtype=np.int8)


def _cos(P):
    return lambda b: P[b] @ V.T


# --------------------------------------------------------------------------- #
# Prior + pseudo-cold seeds
# --------------------------------------------------------------------------- #
def test_popularity_prior_by_hand():
    # log(1 + liked count) / log(1 + 2):  10 -> 1,  30 and 40 -> log 2 / log 3,  20 and 50 -> 0
    np.testing.assert_allclose(PRIOR, [1.0, 0.0, L23, L23, 0.0], rtol=1e-6)
    assert PRIOR.dtype == np.float32


def test_log_scale_rowwise():
    np.testing.assert_allclose(log_scale(np.array([[0, 1, 3], [0, 0, 0]])),
                               [[0, np.log(2) / np.log(4), 1], [0, 0, 0]], rtol=1e-6)


# liked ratings by day (1-day buckets): 10 on days 0, 1, 5 ; 30 on day 4 ; 40 on day 5 is 3★ (not liked)
TIMED = pd.DataFrame([(1, 10, 5, 0), (2, 10, 4, 1), (3, 30, 4, 4), (4, 10, 5, 5), (5, 40, 3, 5)],
                     columns=["userId", "movieId", "rating", "timestamp"])
TIMED["timestamp"] = TIMED["timestamp"] * DAY + 100


def test_time_popularity_counts_only_the_past_window():
    tp = TimePopularity.from_ratings(TIMED, INDEX, bucket_days=1)
    t5, t6 = 5 * DAY + 50_000, 6 * DAY + 1
    # before day 5 (all): 10 -> 2 (days 0, 1), 30 -> 1 ; day 5 itself is NOT counted
    np.testing.assert_array_equal(tp.counts([t5], None)[0], [2, 0, 1, 0, 0])
    # window 2 days before day 5 = days 3-4: only 30
    np.testing.assert_array_equal(tp.counts([t5], 2)[0], [0, 0, 1, 0, 0])
    # before day 6: 10 -> 3 ; 40 never (3★) ; window 1 day = day 5 only
    np.testing.assert_array_equal(tp.counts([t6], None)[0], [3, 0, 1, 0, 0])
    np.testing.assert_array_equal(tp.counts([t6], 1)[0], [1, 0, 0, 0, 0])
    # nothing before the first rating -> zero prior (falls back to the popularity tie-break)
    np.testing.assert_array_equal(tp.prior([0], None)[0], 0)
    # prior at day 5: log(1+2)/log(3) = 1 for 10, log 2 / log 3 for 30 ; two users at once
    np.testing.assert_allclose(tp.prior([t5, t6], None)[0], [1, 0, L23, 0, 0], rtol=1e-6)
    assert tp.prior([t5, t6], None).shape == (2, 5)
    assert tp.now() >= TIMED["timestamp"].max()


def test_onboarding_time_is_the_last_seed():
    seeds = pd.DataFrame([(7, 10, 300, 2), (7, 20, 100, 1), (9, 30, 50, 1)],
                         columns=["userId", "movieId", "timestamp", "seed_rank"])
    assert onboarding_time(seeds, np.array([7, 9])).tolist() == [300, 50]
    with pytest.raises(ValueError):
        onboarding_time(seeds, np.array([7, 8]))


def test_first_liked_keeps_time_order_and_skips_dislikes():
    h = pd.DataFrame([(1, 10, 5), (1, 20, 2), (1, 30, 4), (1, 40, 5), (2, 50, 4)],
                     columns=["userId", "movieId", "rating"])
    s = first_liked(h, 2)
    # user 1: 20 (2★) is not liked -> first two liked are 10, 30 ; user 2 has only one
    assert s[["userId", "movieId", "seed_rank"]].values.tolist() == [[1, 10, 1], [1, 30, 2], [2, 50, 1]]


# --------------------------------------------------------------------------- #
# Text queries in the item space
# --------------------------------------------------------------------------- #
def test_query_is_centered_with_the_catalog_mean():
    # catalog mean = (0.28, 0.24). A query identical to movie 10's vector must land exactly on
    # movie 10's CENTERED vector: (1 - 0.28, 0 - 0.24) / norm = (0.9487, -0.3162)
    q = query_profiles(np.array([[1.0, 0.0]]), V, center=True)
    np.testing.assert_allclose(q[0], center_vectors(V)[0], atol=1e-6)
    np.testing.assert_allclose(q[0], [0.9487, -0.3162], atol=1e-4)
    # no centering: the query is only normalised
    np.testing.assert_allclose(query_profiles(np.array([[3.0, 4.0]]), V, center=False)[0], [0.6, 0.8])


def test_combine_profiles():
    a, b, z = np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]), np.zeros((1, 2))
    np.testing.assert_allclose(combine_profiles(a, b)[0], [0.70711, 0.70711], atol=1e-5)
    np.testing.assert_allclose(combine_profiles(z, b)[0], [0.0, 1.0])     # only one signal
    np.testing.assert_allclose(combine_profiles(z, z)[0], [0.0, 0.0])     # none -> popularity


# --------------------------------------------------------------------------- #
# Genre questionnaire
# --------------------------------------------------------------------------- #
VOCAB = ["Action", "Comedy", "Drama", "IMAX", "Sci-Fi"]
#                 Action Comedy Drama IMAX Sci-Fi
G = np.array([[1, 0, 0, 1, 1],      # 10
              [0, 1, 1, 0, 0],      # 20
              [0, 0, 1, 0, 1],      # 30
              [1, 0, 0, 0, 0],      # 40
              [0, 1, 0, 0, 0]],     # 50
             dtype=np.float32)


def test_onboarding_genres_by_hand():
    # seeds in time order: 10, 30, 20  ->  counts Sci-Fi 2, Drama 2, Action 1, Comedy 1 (IMAX ignored)
    # ties: Sci-Fi first seen at seed 0, Drama at seed 1 ; Action (seed 0) before Comedy (seed 2)
    assert onboarding_genres([0, 2, 1], G, VOCAB, 3) == ["Sci-Fi", "Drama", "Action"]
    assert onboarding_genres([0, 2, 1], G, VOCAB, 10) == ["Sci-Fi", "Drama", "Action", "Comedy"]
    assert onboarding_genres([], G, VOCAB) == []


def test_genre_query_text():
    assert genre_query(["Sci-Fi", "Action", "Thriller"]) == "Science fiction, action and thriller movies"
    assert genre_query(["Comedy"]) == "Comedy movies"
    assert genre_query(["Children", "Animation"]) == "Children's and animation movies"
    assert genre_query([]) == ""


def test_genre_match_share_by_hand():
    # chose Action + Sci-Fi: 10 has both -> 1 ; 30 Sci-Fi -> 0.5 ; 40 Action -> 0.5 ; 20, 50 -> 0
    m = genre_match([["Action", "Sci-Fi"], []], G, VOCAB)
    np.testing.assert_allclose(m[0], [1.0, 0.0, 0.5, 0.5, 0.0])
    np.testing.assert_allclose(m[1], 0.0)                                   # nothing chosen


# --------------------------------------------------------------------------- #
# Blending  score = cos + λ · prior
# --------------------------------------------------------------------------- #
def test_blend_by_hand():
    # one user liked 20 -> profile (0, 1) ; 20 is hidden.
    # cos:   10: 0     30: 0.8     40: -0.6     50: 0
    # λ = 0 : 30 ; then 10 and 50 tie at 0 -> popularity tie-break: 10 before 50 ; then 40
    # λ = 1 : 30: 0.8 + 0.631 = 1.431 | 10: 0 + 1 = 1 | 40: -0.6 + 0.631 = 0.031 | 50: 0
    # λ = 10: 10: 10 | 30: 0.8 + 6.309 = 7.109 | 40: -0.6 + 6.309 = 5.709 | 50: 0
    P = np.array([[0.0, 1.0]], dtype=np.float32)
    out = blend_topk(_cos(P), 1, PRIOR, [0, 1, 10], _seen([[20]]), 4, INDEX, POP)
    assert out[0][0].tolist() == [30, 10, 50, 40]
    assert out[1][0].tolist() == [30, 10, 40, 50]
    assert out[10][0].tolist() == [10, 30, 40, 50]


def test_blend_with_no_onboarding_is_popularity():
    # all-zero profile: every cosine is 0, so any λ ranks by the prior / tie-break = popularity
    P = np.zeros((2, 2), dtype=np.float32)
    seen = _seen([[], [10]])
    pop = PopularityRecommender("count")
    pop.index, pop.score = INDEX, POP
    for lam, recs in blend_topk(_cos(P), 2, PRIOR, [0, 0.5], seen, 3, INDEX, POP).items():
        if lam == 0:
            np.testing.assert_array_equal(recs, pop.recommend(seen, 3))     # [10,20,30],[20,30,40]
        else:
            # prior ties (30 = 40, 20 = 50) are broken by popularity: [10, 30, 40], [30, 40, 20]
            assert recs.tolist() == [[10, 30, 40], [30, 40, 20]]


def test_blend_batches_do_not_change_results():
    rng = np.random.default_rng(0)
    P = rng.normal(size=(7, 2)).astype(np.float32)
    seen = _seen([[10], [], [20, 30], [], [50], [], [40]])
    a = blend_topk(_cos(P), 7, PRIOR, [0, 0.3], seen, 3, INDEX, POP, batch_size=2)
    b = blend_topk(_cos(P), 7, PRIOR, [0, 0.3], seen, 3, INDEX, POP, batch_size=100)
    for lam in a:
        np.testing.assert_array_equal(a[lam], b[lam])


def test_pool_then_rerank_by_hand():
    # profile (0, 1), 20 seen. Pool of 2 by prior among UNSEEN: 10 (1.0), then 30 vs 40 (both
    # 0.631) -> popularity tie-break -> 30. Pool = {10, 30}.
    # λ = 0 : rank the pool by cosine: 30 (0.8), 10 (0)   ;   λ = 10 : 10 (10), 30 (7.1)
    P = np.array([[0.0, 1.0]], dtype=np.float32)
    out = pool_rerank_topk(_cos(P), 1, PRIOR, [2, None], [0, 10], _seen([[20]]), 4, INDEX, POP)
    assert out[(2, 0)][0].tolist() == [30, 10, PAD, PAD]
    assert out[(2, 10)][0].tolist() == [10, 30, PAD, PAD]
    # pool None == the whole-catalog blend
    np.testing.assert_array_equal(out[(None, 10)],
                                  blend_topk(_cos(P), 1, PRIOR, [10], _seen([[20]]), 4, INDEX, POP)[10])


def test_pool_with_per_user_prior():
    # user 0's prior favours 50, user 1's favours 40 ; zero profiles -> pure prior order
    pri = np.array([[0, 0, 0, 0.2, 1.0], [0, 0, 0, 1.0, 0.2]], dtype=np.float32)
    P = np.zeros((2, 2), dtype=np.float32)
    out = pool_rerank_topk(_cos(P), 2, lambda b: pri[b], [2], [1.0], _seen([[], []]), 2,
                           INDEX, POP, batch_size=1)
    assert out[(2, 1.0)].tolist() == [[50, 40], [40, 50]]


# --------------------------------------------------------------------------- #
# Serving object
# --------------------------------------------------------------------------- #
class StubEncoder:
    """Stands in for sentence-transformers: fixed vector per text, records what it was given."""

    def __init__(self, table):
        self.table, self.calls = table, []

    def encode(self, texts, **_):
        self.calls.extend(texts)
        return np.array([self.table[t] for t in texts], dtype=np.float32)


def _rec(**kw):
    return ColdStartRecommender(V, INDEX, PRIOR, POP, center=False, **kw)


def _cfg(lam, pool=None):
    return {"lambda": lam, "pool": pool}


def test_setting_for_picks_the_closest_tuned_setting():
    r = _rec(blend={"seeds_1": _cfg(0.1), "seeds_3": _cfg(0.3), "seeds_5": _cfg(0.5),
                    "text": _cfg(0.7, 100), "text+seeds_1": _cfg(1.1), "text+seeds_5": _cfg(1.5)})
    assert r.setting_for(0, False) == {"lambda": 1.0, "pool": None}   # nothing given -> prior only
    assert r.setting_for(0, True) == _cfg(0.7, 100)
    assert r.setting_for(1, False)["lambda"] == 0.1
    assert r.setting_for(2, False)["lambda"] == 0.3        # tie 1 vs 3 -> the larger
    assert r.setting_for(9, False)["lambda"] == 0.5        # beyond the grid -> 5 seeds
    assert r.setting_for(4, True)["lambda"] == 1.5


def test_recommend_from_one_seed_matches_pool_by_hand():
    # same numbers as test_pool_then_rerank_by_hand (pool 2, λ = 0); explained by seed 20
    r = _rec(blend={"seeds_1": _cfg(0.0, 2)})
    out = r.recommend(seed_ids=[20], k=4)
    assert [o["movieId"] for o in out] == [30, 10]
    assert all(o["because"] == 20 for o in out)
    assert out[0]["score"] == pytest.approx(0.8, abs=1e-5)
    # whole catalog, λ = 1 : 30 (1.431), 10 (1), 40 (0.031), 50 (0)
    r = _rec(blend={"seeds_1": _cfg(1.0)})
    assert [o["movieId"] for o in r.recommend(seed_ids=[20], k=4)] == [30, 10, 40, 50]


def test_recommend_from_text_uses_prefix_and_no_explanation():
    enc = StubEncoder({"q: space adventure": [1.0, 0.0]})
    r = _rec(blend={"text": _cfg(0.0)}, encoder=enc, query_prefix="q: ")
    out = r.recommend(text="  space adventure ", k=3)
    # cos with (1, 0): 10: 1, 40: 0.8, 30: 0.6, 20: 0, 50: -1
    assert [o["movieId"] for o in out] == [10, 40, 30]
    assert enc.calls == ["q: space adventure"]
    assert all(o["because"] is None for o in out)


def test_recommend_hides_seeds_and_excluded_movies_and_takes_a_prior():
    r = _rec(blend={"seeds_1": _cfg(1.0)})
    ids = [o["movieId"] for o in r.recommend(seed_ids=[20], k=5, exclude=[30])]
    assert 20 not in ids and 30 not in ids and len(ids) == 3
    # nothing given at all -> the prior: 10 (1), 30 = 40 (0.631, tie -> popularity), ...
    assert [o["movieId"] for o in r.recommend(k=3)] == [10, 30, 40]
    # a different prior (e.g. the prior at a past onboarding time) changes that order
    assert [o["movieId"] for o in r.recommend(k=2, prior=np.array([0, 0, 0, 0, 1.0]))] == [50, 10]


def test_pool_fast_path_equals_masking_the_catalog():
    # ranking only the M pool columns must give the same lists as scoring the whole catalog with
    # every non-pool movie set to -inf (the slow, obvious version)
    from recsys.baselines import TIE_EPS, mask_seen, topk
    rng = np.random.default_rng(1)
    P = rng.normal(size=(6, 2)).astype(np.float32)
    pri = rng.random((6, 5)).astype(np.float32)
    seen = _seen([[10], [], [20, 30], [], [50], []])
    fast = pool_rerank_topk(_cos(P), 6, lambda b: pri[b], [3], [0, 0.7], seen, 4, INDEX, POP)
    tie = (TIE_EPS * POP).astype(np.float32)
    for w in (0, 0.7):
        base = P @ V.T + tie
        mask_seen(base, seen)
        pr = pri + tie
        rp = np.where(np.isfinite(base), pr, -np.inf)
        pool = np.argsort(-rp, axis=1)[:, :3]
        s = np.full_like(base, -np.inf)
        np.put_along_axis(s, pool, np.take_along_axis(base + w * pr, pool, axis=1), axis=1)
        np.testing.assert_array_equal(fast[(3, w)], topk(s, 4, INDEX))
