"""Day 10: helpers for the final test run — ablation feature sets, a release-date-aware popularity
baseline, the cold-start serving policy, user-aligned significance tests and TreeSHAP summaries.

Nothing here is tuned on test. Every choice it serves was made on validation (Days 3-9); Day 10
only measures:
  * the selected two-stage ranker on ALL warm test users (fit on train+val) at K = 5/10/20,
    against popularity, a release-date-aware popularity, TF-IDF and the embedding recommender;
  * ablations — the same ranker retrained (on validation) without the semantic features, without
    any semantics (popular-500 candidates + no semantic features), without release-year features;
  * the real cold-start users: popularity / Day 6 rerank / cold ranker / warm ranker / the policy
    "cold ranker below 3 seeds, warm ranker from 3" (chosen on pseudo-cold validation, Day 9);
  * SHAP values of the ranker on test candidates (LightGBM's exact TreeSHAP, pred_contrib=True —
    the same numbers shap.TreeExplainer gives, without the extra dependency).

Why the release-aware baseline: the split is per user (Day 2), so plain all-time popularity happily
recommends movies released AFTER a user's last rating — movies they could not have rated. The
ranker can learn to avoid them from item_year / year_gap. A live app never shows unreleased movies,
so this is legitimate, but to say exactly what the ranker adds we also compare it with popularity
that only recommends movies released by the year of the user's last training rating.
"""
from __future__ import annotations

from pathlib import PureWindowsPath

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .features import FEATURE_NAMES
from .metrics import PAD, paired_bootstrap

# --------------------------------------------------------------------------- #
# Feature groups (ablations and grouped SHAP)
# --------------------------------------------------------------------------- #
FEATURE_GROUPS: dict[str, list[str]] = {
    "semantic (embeddings / FAISS)": ["in_faiss", "faiss_rank", "cos_profile", "cos_longterm",
                                      "max_cos_liked"],
    "item popularity & quality": ["in_pop", "pop_rank", "item_liked_log", "item_count_log",
                                  "item_avg_rating", "item_liked_share", "item_recent_log",
                                  "item_trend"],
    "item metadata": ["item_year", "item_runtime"],
    "user": ["user_n_ratings_log", "user_mean_rating", "user_liked_share"],
    "user-item metadata": ["genre_affinity", "year_gap", "pop_gap"],
}
GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
SEMANTIC_FEATURES = FEATURE_GROUPS["semantic (embeddings / FAISS)"]
YEAR_FEATURES = ["item_year", "year_gap"]


def without(drop: list[str], names: list[str] | None = None) -> list[str]:
    """Feature list minus `drop`, original order kept. Unknown names are an error (typo guard)."""
    names = list(names or FEATURE_NAMES)
    unknown = set(drop) - set(names)
    if unknown:
        raise ValueError(f"unknown features {sorted(unknown)}")
    return [n for n in names if n not in set(drop)]


# --------------------------------------------------------------------------- #
# Release-date-aware popularity
# --------------------------------------------------------------------------- #
def timestamp_year(ts) -> np.ndarray:
    """Unix seconds -> calendar year (UTC), as float."""
    t = np.asarray(ts, dtype=np.int64).astype("datetime64[s]")
    return t.astype("datetime64[Y]").astype(np.int64).astype(np.float64) + 1970


def user_cutoff_year(history: pd.DataFrame, users: np.ndarray) -> np.ndarray:
    """Year of each user's LAST rating in the history (NaN for a user with no history).
    A movie released after that year could not have been rated by them at that time."""
    last = history.groupby("userId")["timestamp"].max()
    t = last.reindex(np.asarray(users, dtype=np.int64)).to_numpy(np.float64)
    out = np.full(len(t), np.nan)
    ok = ~np.isnan(t)
    out[ok] = timestamp_year(t[ok].astype(np.int64))
    return out


def released_after(C: np.ndarray, index, item_years: np.ndarray, cutoff: np.ndarray) -> np.ndarray:
    """(U, K) bool: candidate released after the user's cutoff year. Unknown year or unknown
    cutoff -> False (kept). PAD -> False."""
    C = np.asarray(C, dtype=np.int64)
    valid = C != PAD
    cols = index.cols(np.where(valid, C, index.ids[0]))
    yr = np.asarray(item_years, dtype=np.float64)[cols]
    with np.errstate(invalid="ignore"):
        late = yr > np.asarray(cutoff, dtype=np.float64)[:, None]
    return valid & late


def release_aware_topk(score: np.ndarray, index, seen: sp.csr_matrix, item_years: np.ndarray,
                       cutoff: np.ndarray, k: int, pool: int = 2000) -> np.ndarray:
    """(U, k) most popular movies per user that are unseen AND released no later than the user's
    cutoff year (unknown years are kept). With cutoff = +inf this is exactly the popularity
    baseline: same scores, same order (popularity_scores gives every movie a distinct score).

    Works on the `pool` most popular movies first; rows that find fewer than k there are redone
    on the whole catalog."""
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    years = np.asarray(item_years, dtype=np.float64)
    cutoff = np.asarray(cutoff, dtype=np.float64)
    U = seen.shape[0]
    out = np.full((U, k), PAD, dtype=np.int64)

    def run(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        s = seen[rows][:, cols].toarray().astype(bool)
        with np.errstate(invalid="ignore"):
            late = years[cols][None, :] > cutoff[rows][:, None]
        ok = ~s & ~late
        idx = np.argsort(~ok, axis=1, kind="stable")[:, :k]
        got = np.take_along_axis(ok, idx, axis=1)
        res = np.where(got, index.ids[cols][idx], PAD)
        if res.shape[1] < k:
            res = np.pad(res, ((0, 0), (0, k - res.shape[1])), constant_values=PAD)
        return res

    rows = np.arange(U)
    out[:] = run(rows, order[:pool])
    short = np.nonzero((out == PAD).any(axis=1))[0]
    if len(short) and pool < len(order):
        out[short] = run(short, order)
    return out


# --------------------------------------------------------------------------- #
# Cold-start serving policy
# --------------------------------------------------------------------------- #
def policy_model(n_seeds: int, warm_from: int) -> str:
    """Which model serves a new user with n onboarding picks (Day 9 validation finding):
    0 picks -> popularity, fewer than `warm_from` -> the cold ranker, otherwise the warm ranker."""
    if n_seeds <= 0:
        return "popularity"
    return "cold" if n_seeds < warm_from else "warm"


# --------------------------------------------------------------------------- #
# Significance on the SAME users
# --------------------------------------------------------------------------- #
def compare(a: pd.DataFrame, b: pd.DataFrame, metric: str = "ndcg@10", n_boot: int = 2000,
            seed: int = 42) -> dict:
    """Paired bootstrap of model A vs model B from two per-user metric tables (metrics.evaluate
    return_per_user frames), matched on userId. Only users present in both are compared; the
    count is returned so a mismatch is visible."""
    m = a[["userId", metric]].merge(b[["userId", metric]], on="userId", suffixes=("_a", "_b"))
    if m.empty:
        raise ValueError("the two tables share no users")
    r = paired_bootstrap(m[f"{metric}_a"].to_numpy(), m[f"{metric}_b"].to_numpy(), n_boot, seed)
    return {"metric": metric, "n_users": int(len(m)),
            **{k: (round(v, 6) if isinstance(v, float) else [round(x, 6) for x in v])
               for k, v in r.items()}}


def significant(r: dict, alpha: float = 0.05) -> bool:
    return r["p_le_0"] < alpha and r["ci95"][0] > 0


# --------------------------------------------------------------------------- #
# TreeSHAP
# --------------------------------------------------------------------------- #
def tree_shap(booster, X: np.ndarray, batch: int = 200_000) -> np.ndarray:
    """(n, F + 1) exact TreeSHAP values from LightGBM (last column = expected value). Each row
    sums to the model's raw score for that row (tested). Uses the early-stopped tree count."""
    it = booster.best_iteration or None
    parts = [booster.predict(X[s:s + batch], num_iteration=it, pred_contrib=True)
             for s in range(0, len(X), batch)]
    return np.vstack(parts) if parts else np.zeros((0, X.shape[1] + 1))


def shap_table(abs_sum: np.ndarray, signed_sum: np.ndarray, n_rows: int,
               names: list[str]) -> list[dict]:
    """Per feature: mean |SHAP|, its share of the total, mean signed SHAP; largest first."""
    mean_abs = np.asarray(abs_sum, dtype=np.float64) / max(n_rows, 1)
    tot = mean_abs.sum() or 1.0
    rows = [{"feature": n, "group": GROUP_OF.get(n, "other"), "mean_abs_shap": round(float(a), 5),
             "share": round(float(a / tot), 4), "mean_shap": round(float(s / max(n_rows, 1)), 5)}
            for n, a, s in zip(names, mean_abs, signed_sum)]
    return sorted(rows, key=lambda r: -r["mean_abs_shap"])


def group_shares(table: list[dict]) -> list[dict]:
    """Sum of per-feature shares by FEATURE_GROUPS group, largest first."""
    acc: dict[str, float] = {}
    for r in table:
        acc[r["group"]] = acc.get(r["group"], 0.0) + r["share"]
    return [{"group": g, "share": round(s, 4)} for g, s in sorted(acc.items(), key=lambda kv: -kv[1])]


def top_reasons(contrib: np.ndarray, values: np.ndarray, names: list[str], n: int = 3) -> list[dict]:
    """The n features that pushed ONE candidate's score up the most (positive SHAP only)."""
    c = np.asarray(contrib, dtype=np.float64)[:len(names)]
    order = [j for j in np.argsort(-c, kind="stable") if c[j] > 0][:n]
    return [{"feature": names[j], "value": None if np.isnan(values[j]) else round(float(values[j]), 3),
             "shap": round(float(c[j]), 4)} for j in order]


def project_file(rel: str):
    """A path stored in a results json (relative to the repo root) as a local Path. Day 9 wrote
    Windows backslashes ('data\\runs\\x.txt'); this reads either form on any OS."""
    from . import config

    return config.ROOT.joinpath(*PureWindowsPath(rel).parts)
