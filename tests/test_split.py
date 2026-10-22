import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys import split as S  # noqa: E402


def _user(uid, n, t0=0, rating=4.0):
    return pd.DataFrame({"userId": uid, "movieId": np.arange(n) + 100,
                         "rating": rating, "timestamp": t0 + np.arange(n)})


def test_position_from_end():
    from_end, size = S.position_from_end(np.array([1, 1, 1, 2, 2]))
    assert list(from_end) == [2, 1, 0, 1, 0]
    assert list(size) == [3, 3, 3, 2, 2]


def test_ensure_sorted_tie_break():
    # same timestamp -> movieId decides the order (deterministic tie-break)
    r = pd.DataFrame({"userId": [1, 1, 1], "movieId": [9, 3, 5],
                      "rating": [4.0] * 3, "timestamp": [7, 7, 1]})
    out = S.ensure_sorted(r)
    assert list(out["movieId"]) == [5, 3, 9]


def test_assign_split_counts():
    # user 1: 35 ratings -> 10 test, 10 val, 15 train
    # user 2: 25 ratings -> 10 test, val skipped (would leave only 5 train)
    # user 3: 30 ratings, cold -> 10 cold_test, 20 cold_history
    r = pd.concat([_user(1, 35), _user(2, 25), _user(3, 30)], ignore_index=True)
    code = S.assign_split(r, np.array([3]), n_test=10, n_val=10, min_train_for_val=10)
    c = pd.Series(code).groupby(r["userId"]).value_counts().unstack(fill_value=0)
    assert c.loc[1, S.TEST] == 10 and c.loc[1, S.VAL] == 10 and c.loc[1, S.TRAIN] == 15
    assert c.loc[2, S.TEST] == 10 and c.loc[2, S.VAL] == 0 and c.loc[2, S.TRAIN] == 15
    assert c.loc[3, S.COLD_TEST] == 10 and c.loc[3, S.COLD_HISTORY] == 20
    assert c.loc[3, [S.TRAIN, S.VAL, S.TEST]].sum() == 0


def test_test_is_latest_val_before_train():
    r = _user(1, 35)
    code = S.assign_split(r, np.array([], dtype=int), 10, 10, 10)
    t = r["timestamp"].to_numpy()
    assert t[code == S.TRAIN].max() < t[code == S.VAL].min()
    assert t[code == S.VAL].max() < t[code == S.TEST].min()


def test_cold_seeds_are_first_liked():
    r = _user(3, 30)
    r.loc[[0, 2], "rating"] = 2.0  # first and third not liked -> skipped
    code = S.assign_split(r, np.array([3]), 10, 10, 10)
    seeds = S.cold_seeds(r, code, n_seeds=5, threshold=4.0)
    assert list(seeds["movieId"]) == [101, 103, 104, 105, 106]
    assert list(seeds["seed_rank"]) == [1, 2, 3, 4, 5]


def test_pick_cold_users_deterministic_and_eligible():
    r = pd.concat([_user(u, 30) for u in range(1, 21)] + [_user(99, 30, rating=2.0)],
                  ignore_index=True)
    fe, _ = S.position_from_end(r["userId"].to_numpy())
    a = S.pick_cold_users(r, fe, 5, 10, 5, 4.0, seed=42)
    b = S.pick_cold_users(r, fe, 5, 10, 5, 4.0, seed=42)
    assert list(a) == list(b) and 99 not in a  # user 99 liked nothing -> not eligible


def test_fingerprint_changes_with_split():
    r = _user(1, 35)
    code = S.assign_split(r, np.array([], dtype=int), 10, 10, 10)
    code2 = code.copy()
    code2[0] = S.VAL
    assert S.fingerprint(r, code) == S.fingerprint(r, code.copy())
    assert S.fingerprint(r, code) != S.fingerprint(r, code2)


def test_cold_seeds_long_history_no_overflow():
    # 300 liked history ratings: seed_rank must not wrap around (int8 overflow bug)
    r = _user(3, 310)
    code = S.assign_split(r, np.array([3]), 10, 10, 10)
    seeds = S.cold_seeds(r, code, n_seeds=5, threshold=4.0)
    assert len(seeds) == 5 and list(seeds["seed_rank"]) == [1, 2, 3, 4, 5]
