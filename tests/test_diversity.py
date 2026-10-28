"""Day 11 tests: MMR re-ranking, per-list diversity, λ selection, "because you liked X"
explanations and the remove-and-rescore faithfulness check.

Toy catalog (same as Days 3-10): movieIds 10, 20, 30, 40, 50 -> columns 0..4
  vectors  10 (1,0)   20 (0,1)   30 (0.6,0.8)   40 (0.8,-0.6)   50 (-1,0)
  genres   10 [A]     20 [B]     30 [A,B]       40 [A]          50 []
Expected numbers are worked out by hand in the comments.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.baselines import ItemIndex  # noqa: E402
from recsys.diversity import (distinct_genres, mmr_rerank, per_user_ild, pick_lambda,  # noqa: E402
                              pool_of, scale_in_pool)
from recsys.explain import (REASONS, Explainer, because_you_liked, counterfactual,  # noqa: E402
                            liked_history, reason_sums)
from recsys.features import FEATURE_NAMES  # noqa: E402
from recsys.metrics import PAD, beyond_accuracy, intra_list_diversity  # noqa: E402
from recsys.ranker import top_from_scores  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_ranker import ColumnBooster, toy_two_stage  # noqa: E402

INDEX = ItemIndex([10, 20, 30, 40, 50])
V = np.array([[1, 0], [0, 1], [0.6, 0.8], [0.8, -0.6], [-1, 0]], dtype=np.float64)
GENRES = np.array([[1, 0], [0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)


# --------------------------------------------------------------------------- #
# MMR
# --------------------------------------------------------------------------- #
def test_scale_in_pool_by_hand():
    S = np.array([[3.0, 1.0, 2.0, -np.inf], [2.0, 2.0, -np.inf, -np.inf]])
    out = scale_in_pool(S)
    assert out[0, :3].tolist() == [1.0, 0.0, 0.5] and out[0, 3] == -np.inf
    assert out[1, :2].tolist() == [0.0, 0.0]          # nothing to prefer -> all 0


def test_pool_keeps_ranker_order_and_ties():
    C = np.array([[10, 20, 30, PAD]])
    S = np.array([[1.0, 2.0, 1.0, 9.0]])               # PAD's score is ignored
    Cp, Sp = pool_of(C, S, 3)
    assert Cp.tolist() == [[20, 10, 30]]               # tie 10 vs 30 -> candidate order


def test_mmr_by_hand():
    # candidates by ranker score 10 > 40 > 30 > 20 > 50 -> rel 1, .75, .5, .25, 0
    C = np.array([[10, 40, 30, 20, 50]])
    S = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    out = mmr_rerank(C, S, V, INDEX, [0.5, 0.8, 1.0], k=3, pool=5)
    # λ .5: pick 10; then obj = .5 rel - .5 maxsim:  40 .375-.4 = -.025, 30 .25-.3 = -.05,
    #       20 .125-0 = .125, 50 0+.5 = .5 -> 50; then max sim to {10, 50}: 40 .8, 30 .6, 20 0
    #       -> 40 -.025, 30 -.05, 20 .125 -> 20
    assert out[0.5].tolist() == [[10, 50, 20]]
    # λ .8: 40 .6-.16 = .44 wins; then sim(30, 40) = 0, so 30: .4-.8·.6... = .4-.12 = .28,
    #       20 .2-0 = .2, 50 0+.16 = .16 -> 30  (= the ranker's order here)
    assert out[0.8].tolist() == [[10, 40, 30]]
    assert out[1.0].tolist() == [[10, 40, 30]]


def test_mmr_lambda_one_is_exactly_the_ranker_order():
    rng = np.random.default_rng(0)
    U, K = 40, 12
    C = np.array([rng.permutation([10, 20, 30, 40, 50] + [PAD] * (K - 5)) for _ in range(U)])
    S = np.round(rng.random((U, K)), 1)                # many ties
    S[C == PAD] = -np.inf
    for k, pool in ((3, 5), (5, 5), (4, 3)):        # pool < k -> pool is raised to k
        got = mmr_rerank(C, S, V, INDEX, [1.0], k=k, pool=pool)[1.0]
        assert got.tolist() == top_from_scores(C, S, k).tolist()


def test_mmr_genre_space_short_lists_and_zero_vectors():
    # λ 0 in genre space: pick 10 [A]; then sims: 20 0, 30 .707, 40 1, 50 0 (no genre)
    #   -> 20 vs 50 tie at 0 -> earlier in the pool = 20; then 50 (max sim 0) before 30 (.707)
    C = np.array([[10, 40, 30, 20, 50], [30, PAD, PAD, PAD, PAD]])
    S = np.array([[5.0, 4.0, 3.0, 2.0, 1.0], [1.0, 0, 0, 0, 0]])
    out = mmr_rerank(C, S, GENRES, INDEX, [0.0], k=4, pool=5)[0.0]
    assert out[0].tolist() == [10, 20, 50, 30]
    assert out[1].tolist() == [30, PAD, PAD, PAD]      # one candidate -> padded


# --------------------------------------------------------------------------- #
# diversity measures + λ choice
# --------------------------------------------------------------------------- #
def test_per_user_ild_matches_reference_and_beyond_accuracy():
    R = np.array([[10, 20, PAD], [10, 40, 30], [50, PAD, PAD]])
    ild = per_user_ild(R, INDEX, V, k=3)
    # [10, 20]: 1 - cos 0 = 1;  [10, 40, 30]: 1 - (.8 + .6 + 0)/3 = .5333;  one movie -> NaN
    assert ild[0] == pytest.approx(1.0) and ild[1] == pytest.approx(1 - 1.4 / 3) and np.isnan(ild[2])
    mapping = {m: i for i, m in enumerate(INDEX.ids)}
    for u in range(2):
        assert ild[u] == pytest.approx(intra_list_diversity(R[u][R[u] != PAD], V, mapping, 3))
    ba = beyond_accuracy(R, INDEX.ids, np.array([10]), V, k=3)
    assert ba["diversity@3"] == pytest.approx(np.nanmean(ild))


def test_distinct_genres():
    R = np.array([[10, 20, 50], [10, 40, PAD], [50, PAD, PAD]])
    assert distinct_genres(R, INDEX, GENRES, k=3).tolist() == [2, 1, 0]


def test_pick_lambda_most_diverse_within_budget():
    rows = [{"lambda": 1.0, "ndcg@10": 0.100, "diversity@10": 0.50},
            {"lambda": 0.8, "ndcg@10": 0.099, "diversity@10": 0.60},
            {"lambda": 0.5, "ndcg@10": 0.095, "diversity@10": 0.70}]
    assert pick_lambda(rows, 0.02)["lambda"] == 0.8   # .095 < .098 -> λ .5 is over budget
    assert pick_lambda(rows, 0.10)["lambda"] == 0.5
    assert pick_lambda(rows, 0.0)["lambda"] == 1.0


def test_two_stage_mmr_lambda_one_changes_nothing():
    ts, hist = toy_two_stage(ColumnBooster(FEATURE_NAMES.index("cos_profile")))
    h, users = hist[hist.userId == 2], np.array([2])
    base = ts.recommend(h, users, k=2)
    assert ts.recommend(h, users, k=2, mmr={"lambda": 1.0, "pool": 5}).tolist() == base.tolist()
    div = ts.recommend(h, users, k=2, mmr={"lambda": 0.5, "pool": 5, "space": "genre"})
    assert sorted(div[0][div[0] != PAD]) == sorted(base[0][base[0] != PAD])   # same pool of 2
    with pytest.raises(ValueError):
        ts.mmr_vectors("colour")


# --------------------------------------------------------------------------- #
# explanations
# --------------------------------------------------------------------------- #
def test_because_you_liked_by_hand():
    liked = [np.array([0, 1, 2]), np.array([], dtype=np.int64)]   # user A liked 10, 20, 30
    rec = np.array([[3, 4], [3, 4]])                              # recommended 40, 50
    src, cos = because_you_liked(V, liked, rec)
    # 40 (.8,-.6): cos to 10 .8, 20 -.6, 30 0 -> 10;  50 (-1,0): -1, 0, -.6 -> 20 (cos 0)
    assert src[0].tolist() == [0, 1] and cos[0] == pytest.approx([0.8, 0.0])
    assert src[1].tolist() == [-1, -1] and np.isnan(cos[1]).all()   # liked nothing


def test_cited_cosine_is_the_max_cos_liked_feature():
    ts, hist = toy_two_stage()
    h, users = hist[hist.userId == 2], np.array([2])
    fb = ts.feature_batch(h, users)
    ok = fb.candidates != PAD
    liked = liked_history(h, users, INDEX, 4.0, ts.builder.maxcos_history)
    _, cos = because_you_liked(ts.builder.V, liked, INDEX.cols(fb.candidates[ok])[None])
    assert cos[0] == pytest.approx(fb.X["max_cos_liked"][ok])


def test_reason_sums_by_hand():
    names = ["cos_profile", "genre_affinity", "user_mean_rating", "pop_rank", "max_cos_liked"]
    c = np.array([[0.2, 0.1, 0.5, -0.3, 0.05]])
    got = dict(zip(REASONS, reason_sums(c, names)[0]))
    # similar .2+.05, genre .1, popular -.3; user_mean_rating is not a user-facing reason
    assert got == pytest.approx({"similar": 0.25, "genre": 0.1, "era": 0.0, "popular": -0.3, "quality": 0.0})


class ContribBooster:
    """Stand-in booster: every row gets SHAP +1 on genre_affinity and +.5 on cos_profile."""
    best_iteration = 0

    def predict(self, X, num_iteration=None, pred_contrib=False):
        out = np.zeros((len(X), X.shape[1] + 1))
        out[:, FEATURE_NAMES.index("genre_affinity")] = 1.0
        out[:, FEATURE_NAMES.index("cos_profile")] = 0.5
        return out if pred_contrib else out.sum(axis=1)


def test_explainer_texts_and_fallbacks():
    ts, hist = toy_two_stage(ContribBooster())
    h, users = hist[hist.userId == 2], np.array([2])            # liked 10 [A], 20 [B], 30 [A,B]
    fb = ts.feature_batch(h, users)
    cat = type("Cat", (), {"genres": GENRES, "vocab": ["Action", "Drama"],
                           "label": {10: "Ten (1990)", 20: "Twenty (2000)"}})()
    ex = Explainer.from_two_stage(ts, cat, min_cos=0.3)
    top = np.array([[40, 50]])
    recs = ex.explain(fb, h, top)[0]
    # 40 [A]: genre wins (SHAP 1) and A is liked -> genre text; cited movie 10 (cos .8)
    assert recs[0]["reason"] == "genre" and recs[0]["text"] == "Matches genres you like: Action"
    assert recs[0]["because"] == 10 and recs[0]["because_cos"] == pytest.approx(0.8)
    # 50 has no genre -> next: similar, but its best cos is 0 < .3 -> popular (0 likes in history)
    assert recs[1]["reason"] == "popular" and recs[1]["because"] is None
    assert recs[1]["text"] == "Widely liked — 0 people rated it 4★ or more"
    with pytest.raises(ValueError):
        ex.explain(fb, h, np.array([[10, PAD]]))                  # 10 is not a candidate


def test_counterfactual_remove_cited_vs_other():
    ts, hist = toy_two_stage(ColumnBooster(FEATURE_NAMES.index("max_cos_liked")))
    h, users = hist[hist.userId == 2], np.array([2])
    C, ranks = ts.stage1(h, users)
    assert 40 in C[0]
    # recommended 40 cites 10 (cos .8). Remove 10 -> max(cos 20, cos 30) = max(-.6, 0) = 0
    # Remove 20 instead (control) -> 10 still there -> .8
    out = counterfactual(ts, h, users, C, ranks, [(0, 40, 10), (0, 40, 20)])
    assert out["score_before"].tolist() == pytest.approx([0.8, 0.8])
    assert out["score_after"].tolist() == pytest.approx([0.0, 0.8], abs=1e-6)
    assert out["rank_before"].tolist() == [1, 1]
    assert counterfactual(ts, h, users, C, ranks, []).empty
