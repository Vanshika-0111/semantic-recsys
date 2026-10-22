"""Time-based train / validation / test split + held-out cold-start users.

Protocol (frozen on Day 2 — every model is evaluated on exactly this split):

Warm users (everyone except the cold-start sample)
  - TEST  : each user's last N_TEST ratings by (timestamp, movieId)
  - VAL   : the N_VAL ratings just before those, only if the user still keeps
            >= MIN_TRAIN_FOR_VAL training ratings (used for tuning, never for
            final numbers)
  - TRAIN : everything else
  To score on TEST, models train on TRAIN + VAL.  To tune, train on TRAIN and
  score on VAL.

Cold-start users (N_COLD_USERS sampled, removed from ALL training data)
  - COLD_TEST    : their last N_TEST ratings (same rule as warm users)
  - COLD_HISTORY : the rest; their first N_COLD_SEEDS liked movies (rating >=
                   threshold, in time order) become onboarding seeds, so a model
                   can be scored with 0 / 1 / 3 / 5 seeds.

Split codes are stored as one int8 column so the 31M-row table is written once.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN, VAL, TEST, COLD_HISTORY, COLD_TEST = 0, 1, 2, 3, 4
SPLIT_NAMES = {TRAIN: "train", VAL: "val", TEST: "test",
               COLD_HISTORY: "cold_history", COLD_TEST: "cold_test"}

# Which codes each named view loads.
VIEWS = {
    "train_full": [TRAIN, VAL],   # fit here -> evaluate on "test"
    "train": [TRAIN],             # fit here -> evaluate on "val"
    "val": [VAL],
    "test": [TEST],
    "cold_history": [COLD_HISTORY],
    "cold_test": [COLD_TEST],
}

RATINGS_FILE = "ratings_split.parquet"
SEEDS_FILE = "cold_seeds.parquet"
MANIFEST_FILE = "split_manifest.json"


# --------------------------------------------------------------------------- #
# Core logic (pure functions, unit-tested)
# --------------------------------------------------------------------------- #
def ensure_sorted(ratings: pd.DataFrame) -> pd.DataFrame:
    """Sort by (userId, timestamp, movieId) if not already sorted (Day 1 file is)."""
    u = ratings["userId"].to_numpy()
    t = ratings["timestamp"].to_numpy()
    m = ratings["movieId"].to_numpy()
    same_u = u[1:] == u[:-1]
    same_t = same_u & (t[1:] == t[:-1])
    ok = (np.all(u[1:] >= u[:-1]) and np.all(~same_u | (t[1:] >= t[:-1]))
          and np.all(~same_t | (m[1:] > m[:-1])))
    if ok:
        return ratings.reset_index(drop=True)
    return ratings.sort_values(["userId", "timestamp", "movieId"], kind="stable").reset_index(drop=True)


def position_from_end(user_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For rows sorted by user: (0 = user's last row, 1 = second last, ...), user size.

    Pure numpy, int32 — avoids a 31M-row pandas groupby on an 8 GB laptop.
    """
    n = len(user_ids)
    if n == 0:
        return np.empty(0, np.int32), np.empty(0, np.int32)
    starts = np.flatnonzero(np.r_[True, user_ids[1:] != user_ids[:-1]])
    sizes = np.diff(np.r_[starts, n])
    size_per_row = np.repeat(sizes, sizes).astype(np.int32)
    pos = (np.arange(n) - np.repeat(starts, sizes)).astype(np.int32)
    return (size_per_row - 1 - pos).astype(np.int32), size_per_row


def pick_cold_users(ratings: pd.DataFrame, from_end: np.ndarray, n_users: int,
                    n_test: int, n_seeds: int, threshold: float, seed: int) -> np.ndarray:
    """Sample users that have >= n_seeds liked movies before their test window and
    >= 1 liked movie inside it (so every cold user can actually be scored)."""
    liked = ratings["rating"].to_numpy() >= threshold
    in_test = from_end < n_test
    df = pd.DataFrame({"userId": ratings["userId"].to_numpy(),
                       "hist_liked": liked & ~in_test, "test_liked": liked & in_test})
    agg = df.groupby("userId", sort=True)[["hist_liked", "test_liked"]].sum()
    eligible = agg.index[(agg["hist_liked"] >= n_seeds) & (agg["test_liked"] >= 1)].to_numpy()
    if len(eligible) < n_users:
        raise ValueError(f"only {len(eligible)} users are eligible for cold start, need {n_users}")
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(eligible, size=n_users, replace=False))


def assign_split(ratings: pd.DataFrame, cold_users: np.ndarray, n_test: int, n_val: int,
                 min_train_for_val: int) -> np.ndarray:
    """Return an int8 split code per row. `ratings` must be sorted (see ensure_sorted)."""
    from_end, size = position_from_end(ratings["userId"].to_numpy())
    code = np.full(len(ratings), TRAIN, dtype=np.int8)
    has_val = size >= n_test + n_val + min_train_for_val
    code[(from_end >= n_test) & (from_end < n_test + n_val) & has_val] = VAL
    code[from_end < n_test] = TEST
    is_cold = np.isin(ratings["userId"].to_numpy(), cold_users)
    code[is_cold] = np.where(code[is_cold] == TEST, COLD_TEST, COLD_HISTORY).astype(np.int8)
    return code


def cold_seeds(ratings: pd.DataFrame, code: np.ndarray, n_seeds: int,
               threshold: float) -> pd.DataFrame:
    """First n_seeds liked movies (time order) from each cold user's history."""
    mask = (code == COLD_HISTORY) & (ratings["rating"].to_numpy() >= threshold)
    h = ratings.loc[mask, ["userId", "movieId", "rating", "timestamp"]].copy()
    rank = h.groupby("userId").cumcount().to_numpy() + 1
    h = h[rank <= n_seeds].copy()
    h["seed_rank"] = rank[rank <= n_seeds].astype(np.int8)
    return h.reset_index(drop=True)


def fingerprint(ratings: pd.DataFrame, code: np.ndarray) -> str:
    """SHA-256 over (userId, movieId, split) — proves the split never changed."""
    h = hashlib.sha256()
    for arr in (ratings["userId"].to_numpy(np.int64), ratings["movieId"].to_numpy(np.int64),
                np.asarray(code, dtype=np.int8)):
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def summarize(ratings: pd.DataFrame, code: np.ndarray, seeds: pd.DataFrame,
              threshold: float) -> dict:
    users = ratings["userId"].to_numpy()
    movies = ratings["movieId"].to_numpy()
    liked = ratings["rating"].to_numpy() >= threshold
    out: dict = {"splits": {}}
    for c, name in SPLIT_NAMES.items():
        m = code == c
        out["splits"][name] = {
            "ratings": int(m.sum()),
            "users": int(np.unique(users[m]).size),
            "users_with_relevant": int(np.unique(users[m & liked]).size),
            "relevant_pct": round(float(100 * liked[m].mean()), 2) if m.any() else None,
        }
    train_items = np.unique(movies[(code == TRAIN) | (code == VAL)])
    for name, c in (("test", TEST), ("cold_test", COLD_TEST)):
        items = np.unique(movies[code == c])
        out[f"{name}_items_unseen_in_train"] = int((~np.isin(items, train_items)).sum())
    out["catalog_items_in_train"] = int(train_items.size)
    out["cold_seed_users"] = int(seeds["userId"].nunique())
    return out


# --------------------------------------------------------------------------- #
# I/O helpers used by every later day
# --------------------------------------------------------------------------- #
def load_view(view: str, split_dir: Path,
              columns: list[str] | None = None) -> pd.DataFrame:
    """Load one view (see VIEWS) without reading the other splits into memory."""
    if view not in VIEWS:
        raise KeyError(f"unknown view {view!r}; choose from {list(VIEWS)}")
    return pd.read_parquet(split_dir / RATINGS_FILE, columns=columns,
                           filters=[("split", "in", VIEWS[view])])


def load_cold_seeds(split_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(split_dir / SEEDS_FILE)


def verify_frozen(split_dir: Path) -> bool:
    """Recompute the fingerprint and compare it to the manifest written on Day 2."""
    manifest = json.loads((split_dir / MANIFEST_FILE).read_text())
    df = pd.read_parquet(split_dir / RATINGS_FILE, columns=["userId", "movieId", "split"])
    return fingerprint(df, df["split"].to_numpy()) == manifest["fingerprint"]
