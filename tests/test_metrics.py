"""Metric tests. Expected numbers are worked out BY HAND in the comments.

Running example (used in most tests):
    recs     = [10, 20, 30, 40, 50]      (rank 1 .. 5)
    relevant = {20, 40, 60}              (60 was never recommended)
    hits at ranks 2 and 4.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.metrics import (PAD, beyond_accuracy, catalog_coverage, evaluate,  # noqa: E402
                            intra_list_diversity, long_tail_share, mean_intra_list_diversity, mrr_at_k, ndcg_at_k, precision_at_k,
                            recall_at_k, recs_to_matrix)

RECS = [10, 20, 30, 40, 50]
REL = {20, 40, 60}
L2, L3, L4, L5 = np.log2(2), np.log2(3), np.log2(4), np.log2(5)   # 1, 1.585, 2, 2.322


def test_precision():
    assert precision_at_k(RECS, REL, 5) == pytest.approx(2 / 5)   # 2 hits / 5 slots
    assert precision_at_k(RECS, REL, 2) == pytest.approx(1 / 2)   # rank 2 is a hit
    assert precision_at_k(RECS, REL, 1) == 0.0
    # fewer recs than K: still divide by K
    assert precision_at_k([20], REL, 5) == pytest.approx(1 / 5)


def test_recall():
    assert recall_at_k(RECS, REL, 5) == pytest.approx(2 / 3)      # 2 of 3 relevant found
    assert recall_at_k(RECS, REL, 2) == pytest.approx(1 / 3)
    assert np.isnan(recall_at_k(RECS, set(), 5))                  # undefined, not 0


def test_mrr():
    assert mrr_at_k(RECS, REL, 5) == pytest.approx(1 / 2)         # first hit at rank 2
    assert mrr_at_k(RECS, REL, 1) == 0.0                          # no hit in top-1
    assert mrr_at_k([40, 20], REL, 5) == 1.0


def test_ndcg_binary():
    # DCG@5  = 1/log2(3) + 1/log2(5)                = 0.6309 + 0.4307 = 1.0616
    # IDCG@5 = 1/log2(2) + 1/log2(3) + 1/log2(4)    = 1 + 0.6309 + 0.5 = 2.1309
    expected = (1 / L3 + 1 / L5) / (1 / L2 + 1 / L3 + 1 / L4)
    assert expected == pytest.approx(0.4982, abs=1e-4)
    assert ndcg_at_k(RECS, REL, 5) == pytest.approx(expected)
    # K=2: DCG = 1/log2(3) ; IDCG = 1 + 1/log2(3)  -> 0.3869
    assert ndcg_at_k(RECS, REL, 2) == pytest.approx((1 / L3) / (1 + 1 / L3))
    assert ndcg_at_k(RECS, REL, 2) == pytest.approx(0.3869, abs=1e-4)


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k([20, 40, 60], REL, 3) == pytest.approx(1.0)
    assert ndcg_at_k([60, 20, 40, 1, 2], REL, 5) == pytest.approx(1.0)  # order among relevant irrelevant
    assert ndcg_at_k([1, 2, 3], REL, 3) == 0.0


def test_ndcg_graded():
    # gains: 20 -> 2, 40 -> 1, 60 -> 3 ; recs top-3 = [10, 20, 30]
    # DCG@3  = 2/log2(3)                          = 1.2619
    # IDCG@3 = 3/log2(2) + 2/log2(3) + 1/log2(4)  = 3 + 1.2619 + 0.5 = 4.7619
    gains = {20: 2.0, 40: 1.0, 60: 3.0}
    assert ndcg_at_k(RECS, gains, 3) == pytest.approx((2 / L3) / (3 + 2 / L3 + 1 / L4))
    assert ndcg_at_k(RECS, gains, 3) == pytest.approx(0.2650, abs=1e-4)


def test_duplicates_rejected():
    with pytest.raises(ValueError):
        precision_at_k([20, 20, 30], REL, 3)


def test_coverage_and_long_tail():
    lists = [[1, 2, 3], [2, 3, 4], [3, 4, PAD]]
    assert catalog_coverage(lists, catalog_size=8, k=3) == pytest.approx(4 / 8)  # {1,2,3,4}
    # head = {1, 2}; slots: 1 2 3 | 2 3 4 | 3 4  -> 8 slots, tail ones: 3,3,4,3,4 = 5
    assert long_tail_share(lists, head_items={1, 2}, k=3) == pytest.approx(5 / 8)


def test_intra_list_diversity():
    vecs = np.array([[1, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    index = {100: 0, 200: 1, 300: 2, 400: 3}
    assert intra_list_diversity([100, 200], vecs, index, 2) == pytest.approx(1.0)  # orthogonal
    assert intra_list_diversity([100, 300], vecs, index, 2) == pytest.approx(0.0)  # identical
    # [100, 200, 300]: pairs sims = 0, 1, 0 -> mean sim 1/3 -> diversity 2/3
    assert intra_list_diversity([100, 200, 300], vecs, index, 3) == pytest.approx(2 / 3)


# ------------------------------- vectorized ------------------------------- #
def _truth(rows):
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"])


def test_evaluate_hand_example():
    # user 1: running example (60 relevant, 70 rated 3 -> NOT relevant)
    # user 2: recs [5, 6], relevant {6}  -> hit at rank 2
    # user 3: no relevant held-out item -> skipped
    # user 4: has relevant items but got no recommendations -> scores 0
    truth = _truth([(1, 20, 5), (1, 40, 4), (1, 60, 4.5), (1, 70, 3),
                    (2, 6, 4), (3, 9, 2), (4, 8, 5)])
    users, mat = recs_to_matrix({1: RECS, 2: [5, 6], 3: [9]}, k=5)
    out, per = evaluate(users, mat, truth, ks=(5,), return_per_user=True)
    assert out["n_users"] == 3 and out["n_users_without_recs"] == 1
    u1 = (1 / L3 + 1 / L5) / (1 + 1 / L3 + 1 / L4)
    u2 = (1 / L3) / 1.0
    assert out["ndcg@5"] == pytest.approx((u1 + u2 + 0) / 3)
    assert out["recall@5"] == pytest.approx((2 / 3 + 1 + 0) / 3)
    assert out["precision@5"] == pytest.approx((2 / 5 + 1 / 5 + 0) / 3)
    assert out["mrr@5"] == pytest.approx((1 / 2 + 1 / 2 + 0) / 3)
    assert out["hit_rate@5"] == pytest.approx(2 / 3)
    assert list(per["userId"]) == [1, 2, 4]


def test_evaluate_matches_reference_on_random_data():
    rng = np.random.default_rng(0)
    rows, recs = [], {}
    for u in range(300):
        held = rng.choice(60, size=10, replace=False)
        rows += [(u, int(i), float(rng.choice([1, 2, 3, 3.5, 4, 4.5, 5]))) for i in held]
        recs[u] = list(rng.choice(60, size=int(rng.integers(0, 21)), replace=False))
    truth = _truth(rows)
    users, mat = recs_to_matrix(recs, k=20)
    for graded in (False, True):
        out = evaluate(users, mat, truth, ks=(5, 10, 20), graded=graded)
        for k in (5, 10, 20):
            ref = {"ndcg": [], "recall": [], "precision": [], "mrr": []}
            for u, g in truth[truth.rating >= 4].groupby("userId"):
                gains = dict(zip(g.movieId, g.rating - 3)) if graded else set(g.movieId)
                ref["ndcg"].append(ndcg_at_k(recs[u], gains, k))
                ref["recall"].append(recall_at_k(recs[u], gains, k))
                ref["precision"].append(precision_at_k(recs[u], gains, k))
                ref["mrr"].append(mrr_at_k(recs[u], gains, k))
            for name, vals in ref.items():
                assert out[f"{name}@{k}"] == pytest.approx(np.mean(vals)), (name, k, graded)


def test_evaluate_rejects_duplicates():
    with pytest.raises(ValueError):
        evaluate(np.array([1]), np.array([[20, 20, 30]]), _truth([(1, 20, 5)]), ks=(3,))


def test_beyond_accuracy_matches_reference():
    rng = np.random.default_rng(1)
    item_ids = np.arange(100, 150)
    vecs = (rng.random((50, 6)) < 0.4).astype(float)          # genre-like multi-hot (some all-zero)
    index = {int(i): j for j, i in enumerate(item_ids)}
    lists = [list(rng.choice(item_ids, size=int(rng.integers(0, 11)), replace=False))
             for _ in range(200)]
    _, mat = recs_to_matrix(dict(enumerate(lists)), k=10)
    head = item_ids[:5]
    out = beyond_accuracy(mat, item_ids, head, vecs, k=10, chunk=37)   # odd chunk on purpose
    assert out["coverage@10"] == pytest.approx(catalog_coverage(lists, 50, 10))
    assert out["long_tail@10"] == pytest.approx(long_tail_share(lists, set(head.tolist()), 10))
    assert out["diversity@10"] == pytest.approx(mean_intra_list_diversity(lists, vecs, index, 10))


def test_paired_bootstrap_by_hand():
    from recsys.metrics import paired_bootstrap
    # every user gains exactly 0.1 -> every resample has mean diff 0.1 -> CI [0.1, 0.1], p = 0
    r = paired_bootstrap(np.array([0.3, 0.2, 0.1]), np.array([0.2, 0.1, 0.0]), n_boot=200)
    assert r["diff"] == pytest.approx(0.1) and r["p_le_0"] == 0.0
    assert r["ci95"] == pytest.approx([0.1, 0.1])
    assert r["rel_change"] == pytest.approx(0.1 / 0.1)        # mean_b = 0.1 -> +100%
    # A never better: one user loses, one ties -> every resample mean <= 0 -> p = 1
    r = paired_bootstrap(np.array([0.0, 0.5]), np.array([0.2, 0.5]), n_boot=200)
    assert r["p_le_0"] == 1.0 and r["ci95"][1] <= 0
