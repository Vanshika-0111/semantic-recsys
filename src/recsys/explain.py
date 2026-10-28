"""Day 11: "Because you liked X" — explanations for the two-stage recommender.

Two parts, both faithful to what the model actually used:

1. The cited movie X. For a recommended movie m, X = the movie among the user's last 50 liked
   movies with the highest cosine to m (mean-centered bge vectors). That cosine IS the ranker
   feature `max_cos_liked` (tested equal), so the explanation names the exact movie behind a
   feature the model scores with — not a separate, made-up similarity.

2. Which kind of reason to show. TreeSHAP (Day 10) splits the movie's ranker score into one
   contribution per feature. Features are summed into user-facing reason groups:
     similar   semantic features (cos_profile, cos_longterm, max_cos_liked, FAISS)  -> "Because you liked X"
     genre     genre_affinity                                                   -> "Matches genres you like: …"
     era       year_gap, item_year                                              -> "From the 1990s, like most of what you watch"
     popular   popularity features (pop_rank, liked counts, recent, trend, pop_gap) -> "Widely liked — 12,345 fans"
     quality   Bayesian average rating, liked share                             -> "Highly rated — 4.2★ average"
   User-level constants and runtime are not user-facing and are left out. The group with the
   largest positive SHAP sum wins, if its text can be written (e.g. "similar" needs cos >= min_cos).

counterfactual(): does removing the cited movie X from the history really lower the movie's
score? Compared with removing a random other recently liked movie (the control) — Day 11's
faithfulness check.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .metrics import PAD

REASON_GROUPS: dict[str, list[str]] = {
    "similar": ["cos_profile", "cos_longterm", "max_cos_liked", "faiss_rank", "in_faiss"],
    "genre": ["genre_affinity"],
    "era": ["year_gap", "item_year"],
    "popular": ["in_pop", "pop_rank", "item_liked_log", "item_count_log", "item_recent_log",
                "item_trend", "pop_gap"],
    "quality": ["item_avg_rating", "item_liked_share"],
}
REASONS = list(REASON_GROUPS)


def liked_history(hist: pd.DataFrame, users: np.ndarray, index, threshold: float,
                  n_last: int) -> list[np.ndarray]:
    """Per user (in `users` order): catalog columns of their last `n_last` liked movies, oldest
    first — the same rows FeatureBuilder._max_cos uses (history grouped by user, time order)."""
    h = hist[hist["rating"].to_numpy() >= threshold]
    uid = h["userId"].to_numpy(np.int64)
    if np.any(np.diff(uid) < 0):
        raise ValueError("history must be grouped by user (sorted userId, then time)")
    cols = index.cols(h["movieId"].to_numpy())
    a = np.searchsorted(uid, users)
    b = np.searchsorted(uid, users, side="right")
    return [cols[max(s, e - n_last):e] for s, e in zip(a, b)]


def because_you_liked(V: np.ndarray, liked: list[np.ndarray], rec_cols: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
    """(U, k) column of the cited liked movie (-1 if the user liked nothing) and its cosine.
    V: unit item vectors (the feature builder's centered V). Ties -> the earliest of those likes."""
    U, k = rec_cols.shape
    src = np.full((U, k), -1, dtype=np.int64)
    cos = np.full((U, k), np.nan)
    for u in range(U):
        if len(liked[u]) == 0:
            continue
        sims = V[rec_cols[u]] @ V[liked[u]].T                 # (k, n_liked)
        j = np.argmax(sims, axis=1)
        src[u] = liked[u][j]
        cos[u] = sims[np.arange(k), j]
    return src, cos


def reason_sums(contrib: np.ndarray, names: list[str]) -> np.ndarray:
    """(n, len(REASONS)) SHAP summed per reason group (features not in a group are ignored)."""
    c = np.asarray(contrib, dtype=np.float64)
    out = np.zeros((c.shape[0], len(REASONS)))
    for g, feats in enumerate(REASON_GROUPS.values()):
        idx = [names.index(f) for f in feats if f in names]
        if idx:
            out[:, g] = c[:, idx].sum(axis=1)
    return out


@dataclass
class Explainer:
    """Turns ranker output into one short reason per recommendation.

    booster / features : the ranker (None -> no SHAP: 'similar' when possible, else 'popular')
    V                  : unit item vectors used by the features (FeatureBuilder.V)
    stats              : FeatureBuilder.stats (item statistics of the same history)
    """
    booster: object | None
    features: list[str]
    V: np.ndarray
    index: object
    genres: np.ndarray
    vocab: list[str]
    years: np.ndarray
    stats: dict
    label: dict = field(default_factory=dict)
    threshold: float = 4.0
    n_last: int = 50
    min_cos: float = 0.3

    @classmethod
    def from_two_stage(cls, ts, catalog, min_cos: float = 0.3) -> "Explainer":
        b = ts.builder
        return cls(ts.booster, list(ts.features), b.V, b.index, catalog.genres, list(catalog.vocab),
                   b.years, b.stats, dict(catalog.label), b.threshold, b.maxcos_history, min_cos)

    def name(self, col: int) -> str:
        m = int(self.index.ids[col])
        return self.label.get(m, str(m))

    def explain(self, fb, hist_u: pd.DataFrame, top: np.ndarray) -> list[list[dict]]:
        """For each user of the feature batch `fb` and their shown list `top` (U, k movieIds,
        PAD-padded): [{movieId, reason, text, because, because_cos, shap}] per shown movie."""
        from .evaluation import tree_shap

        U, k = top.shape
        users = fb.users
        valid = top != PAD
        # where each shown movie sits in the candidate matrix -> its feature row
        pos = np.full((U, k), -1)
        for u in range(U):
            where = {int(m): j for j, m in enumerate(fb.candidates[u]) if m != PAD}
            pos[u] = [where.get(int(m), -1) for m in top[u]]
        if np.any(valid & (pos < 0)):
            raise ValueError("every shown movie must be one of the user's candidates")
        rows = np.where(valid, pos, 0)
        X = np.stack([np.take_along_axis(fb.X[f], rows, axis=1).ravel() for f in self.features], axis=1)
        if self.booster is not None:
            groups = reason_sums(tree_shap(self.booster, X)[:, :len(self.features)], self.features)
        else:
            groups = np.zeros((U * k, len(REASONS)))
            groups[:, REASONS.index("similar")], groups[:, REASONS.index("popular")] = 2.0, 1.0
        groups = groups.reshape(U, k, len(REASONS))

        cols = self.index.cols(np.where(valid, top, self.index.ids[0]))
        liked = liked_history(hist_u, users, self.index, self.threshold, self.n_last)
        src, cos = because_you_liked(self.V, liked, cols)
        G = np.asarray(self.genres) > 0
        out = []
        for u in range(U):
            share = (G[liked[u]].mean(axis=0) if len(liked[u]) else np.zeros(G.shape[1]))
            ly = self.years[liked[u]] if len(liked[u]) else np.array([])
            ly = ly[~np.isnan(ly)]
            mean_year = float(ly.mean()) if ly.size else np.nan
            recs = []
            for j in range(k):
                if not valid[u, j]:
                    continue
                c = int(cols[u, j])
                has_x = src[u, j] >= 0 and cos[u, j] >= self.min_cos
                order = np.argsort(-groups[u, j], kind="stable")
                reason, text = "popular", self._popular(c)
                for g in order:
                    if groups[u, j, g] <= 0:
                        break
                    t = self._text(REASONS[g], c, has_x, int(src[u, j]), share, mean_year)
                    if t:
                        reason, text = REASONS[g], t
                        break
                recs.append({"movieId": int(top[u, j]), "reason": reason, "text": text,
                             "because": int(self.index.ids[src[u, j]]) if has_x else None,
                             "because_cos": None if np.isnan(cos[u, j]) else round(float(cos[u, j]), 3),
                             "shap": {r: round(float(groups[u, j, g]), 4) for g, r in enumerate(REASONS)}})
            out.append(recs)
        return out

    # --- texts ---
    def _text(self, reason: str, c: int, has_x: bool, src: int, share, mean_year) -> str | None:
        if reason == "similar":
            return f"Because you liked {self.name(src)}" if has_x else None
        if reason == "genre":
            g = np.flatnonzero((np.asarray(self.genres[c]) > 0) & (share > 0))
            if g.size == 0:
                return None
            g = g[np.argsort(-share[g], kind="stable")][:2]
            return "Matches genres you like: " + ", ".join(self.vocab[i] for i in g)
        if reason == "era":
            if np.isnan(mean_year) or np.isnan(self.years[c]):
                return None
            return f"From {int(self.years[c])} — close to the {int(mean_year) // 10 * 10}s movies you like"
        if reason == "quality":
            return f"Highly rated — {self.stats['avg_rating'][c]:.1f}★ average"
        return self._popular(c)

    def _popular(self, c: int) -> str:
        n = int(self.stats["liked"][c])
        return f"Widely liked — {n:,} people rated it 4★ or more"


# --------------------------------------------------------------------------- #
# Faithfulness: remove the cited movie, re-score
# --------------------------------------------------------------------------- #
def counterfactual(ts, hist_u: pd.DataFrame, users: np.ndarray, C: np.ndarray, ranks: dict,
                   pairs: list[tuple[int, int, int]]) -> pd.DataFrame:
    """Re-score after removing one movie from a user's history.

    pairs : (row of the user in `users`, recommended movieId, removed movieId)
    Stage 1 is held fixed (same candidates and source ranks); every stage-2 feature is
    recomputed from the reduced history (profile, max-cos, genre affinity, user stats …).
    Returns per pair: score / rank (1 = best among the user's candidates) before and after.
    """
    if not pairs:
        return pd.DataFrame(columns=["row", "movieId", "removed", "score_before", "score_after",
                                     "rank_before", "rank_after"])
    uid = hist_u["userId"].to_numpy(np.int64)
    a = np.searchsorted(uid, users)
    b = np.searchsorted(uid, users, side="right")
    mids = hist_u["movieId"].to_numpy()
    take, owner = [], []
    for p, (r, _, removed) in enumerate(pairs):
        idx = np.arange(a[r], b[r])
        idx = idx[mids[idx] != removed]
        take.append(idx)
        owner.append(np.full(len(idx), p))
    take = np.concatenate(take)
    h2 = hist_u.iloc[take].copy()
    h2["userId"] = np.concatenate(owner)
    rr = np.array([p[0] for p in pairs])
    pseudo = np.arange(len(pairs), dtype=np.int64)
    C2 = C[rr]
    fb0 = ts.builder.transform(users, hist_u, C, ranks)
    S0 = ts.scores(fb0)[rr]
    fb2 = ts.builder.transform(pseudo, h2, C2, {s: v[rr] for s, v in ranks.items()})
    S2 = ts.scores(fb2)
    col = np.array([int(np.flatnonzero(C2[p] == m)[0]) for p, (_, m, _) in enumerate(pairs)])
    i = np.arange(len(pairs))
    s0, s2 = S0[i, col], S2[i, col]
    return pd.DataFrame({"row": rr, "movieId": [p[1] for p in pairs], "removed": [p[2] for p in pairs],
                         "score_before": s0, "score_after": s2,
                         "rank_before": 1 + (S0 > s0[:, None]).sum(axis=1),
                         "rank_after": 1 + (S2 > s2[:, None]).sum(axis=1)})
