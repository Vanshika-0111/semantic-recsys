"""Day 13: a small synthetic MovieLens-shaped world and a REAL serving bundle built from it.

Used by
  - tests (Day 12 serving tests, Day 13 UI test) — moved here from tests/test_serving.py so the
    tests and CI build the bundle the same way;
  - CI (GitHub Actions): `python scripts/make_synthetic_bundle.py` writes data/serving, then the
    Docker images are built and started with it. The real 47 MB bundle never leaves the laptop /
    Hugging Face, and CI needs no MovieLens download.

The world: 80 movies with 16-d vectors, 4 genres, years 1970–2018 (two unknown), a popularity skew,
60 users whose ratings follow taste · vector + noise. Two real LightGBM LambdaMART rankers are
trained on it with the Day 9 recipe in miniature, then export_bundle writes the same files as the
real pipeline (bundle.json, arrays.npz, items.parquet, rankers, demo users).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import ItemIndex
from .embeddings import l2_normalize
from .experiment import Catalog

DAY = 86_400
YEAR_1996 = 820_454_400                     # 1996-01-01 UTC
MMR = {"space": "genre", "lambda": 0.7, "pool": 15}
CHOSEN = {"popularity_mode": "liked", "center": True,
          "profile": {"weighting": "graded", "half_life": 5}}
GENRES = ["Action", "Comedy", "Drama", "Horror"]


class FakeEncoder:
    """Text -> a fixed 16-d vector seeded by the text length (stand-in for bge-small)."""

    def encode(self, texts, **kw):
        return np.stack([np.random.default_rng(len(t)).normal(size=16) for t in texts]).astype(np.float32)


def make_world(seed: int = 0):
    """-> (Catalog, raw item vectors (80, 16), ratings history with userId/movieId/rating/timestamp)."""
    rng = np.random.default_rng(seed)
    n_items, n_users, d = 80, 60, 16
    ids = np.arange(1, n_items + 1) * 10
    raw = l2_normalize(rng.normal(size=(n_items, d)))
    G = np.zeros((n_items, 4), dtype=np.float32)
    G[np.arange(n_items), np.argmax(raw[:, :4], axis=1)] = 1
    G[rng.random(n_items) < 0.3, rng.integers(0, 4)] = 1
    years = rng.integers(1970, 2019, n_items).astype(float)
    years[[3, 17]] = np.nan
    items = pd.DataFrame({"movieId": ids, "title_clean": [f"Movie {m}" for m in ids], "year": years,
                          "runtime": rng.integers(80, 160, n_items).astype(float),
                          "poster_path": [f"/p{m}.jpg" for m in ids],
                          "overview": [f"Overview of movie {m}" for m in ids], "tmdbId": ids + 1})
    index = ItemIndex(ids)
    label = {m: f"Movie {m} ({'?' if np.isnan(y) else int(y)})" for m, y in zip(ids, years)}
    cat = Catalog(items, index, pd.Series([""] * n_items, index=ids), G, list(GENRES), label)

    pop = np.exp(rng.normal(size=n_items))
    taste = l2_normalize(rng.normal(size=(n_users, d)))
    rows = []
    for u in range(1, n_users + 1):
        w = pop * np.exp(3 * raw @ taste[u - 1])
        picks = rng.choice(n_items, size=25, replace=False, p=w / w.sum())
        t = YEAR_1996 + rng.integers(0, 20 * 365) * DAY + np.arange(25) * DAY * 30
        r = np.clip(np.round((3.2 + 2.5 * (raw[picks] @ taste[u - 1]) + rng.normal(0, .5, 25)) * 2) / 2, .5, 5)
        rows.append(pd.DataFrame({"userId": u, "movieId": ids[picks], "rating": r, "timestamp": t}))
    hist = pd.concat(rows, ignore_index=True)
    return cat, raw, hist


def train_booster(cat, raw, hist, leaves: int, rounds: int, path: Path):
    """Real LightGBM LambdaMART on the synthetic data: features from all but each user's last 5
    ratings, labels = those 5 (the Day 9 recipe in miniature)."""
    from .features import attach_labels
    from .ranker import RankData, build_two_stage, train_ranker

    last5 = hist.groupby("userId").cumcount(ascending=False) < 5
    train, label = hist[~last5], hist[last5]
    ts = build_two_stage(cat, raw, CHOSEN, train, n_pop=30, m_faiss=10, faiss_feature_m=10)
    users = np.unique(train["userId"])
    df = attach_labels(ts.feature_batch(train, users).to_frame(), label)
    d = RankData.from_frame(df)
    cfg = {"labels": "graded", "num_leaves": leaves, "min_data_in_leaf": 5, "learning_rate": 0.1}
    booster, _ = train_ranker(d.drop_users_without_positives(), d, cfg, seed=1, max_rounds=rounds,
                              early_stopping=rounds, threads=1)
    booster.save_model(str(path))
    return booster


def build_synthetic_bundle(out_dir: Path | str, work_dir: Path | str | None = None,
                           demo_users: bool = True, seed: int = 0) -> dict:
    """Train the two rankers on the synthetic world and export a serving bundle to out_dir.
    demo_users=False writes a bundle without demo histories (like the public Hugging Face one).
    Returns {"cat", "raw", "hist", "ts", "warm", "cold", "man", "demo", "dir"}."""
    import lightgbm as lgb

    from .coldstart import popularity_prior
    from .ranker import build_two_stage
    from .serving import export_bundle

    out_dir = Path(out_dir)
    work = Path(work_dir) if work_dir is not None else out_dir.parent / f"{out_dir.name}_models"
    work.mkdir(parents=True, exist_ok=True)
    cat, raw, hist = make_world(seed)
    train_booster(cat, raw, hist, 7, 20, work / "warm.txt")
    train_booster(cat, raw, hist, 3, 10, work / "cold.txt")
    warm = lgb.Booster(model_file=str(work / "warm.txt"))
    cold = lgb.Booster(model_file=str(work / "cold.txt"))
    ts = build_two_stage(cat, raw, CHOSEN, hist, n_pop=30, m_faiss=10, faiss_feature_m=10, booster=warm)
    demo = np.arange(1, 21) if demo_users else np.array([], dtype=np.int64)
    truth = (pd.DataFrame({"userId": [1, 1], "movieId": [10, 20], "rating": [4.0, 2.0]})
             if demo_users else None)
    settings = {"policy": {"warm_from": 3}, "mmr": MMR, "release_filter": {"enabled": False},
                "explain": {"min_cos": 0.3}, "explore": {"pool": 1000, "lambda": 0.3},
                "encoder": {"key": "fake", "hf_name": None, "query_prefix": "q: "}}
    man = export_bundle(out_dir, cat, raw, ts, warm_model=work / "warm.txt",
                        cold_model=work / "cold.txt", prior=popularity_prior(hist, cat.index),
                        demo_history=hist[hist["userId"].isin(demo)], demo_truth=truth,
                        settings=settings, meta={"split_fingerprint": "synthetic"})
    return {"cat": cat, "raw": raw, "hist": hist, "ts": ts, "warm": warm, "cold": cold,
            "man": man, "demo": demo, "dir": out_dir}
