"""Day 12 tests: serving bundle, Engine, FastAPI.

Most tests use a small synthetic world (80 movies, 60 users, real LightGBM rankers trained on it)
so the WHOLE path runs: export_bundle -> Engine.load -> requests. The key test is the round trip:
the Engine (built only from the bundle files, no ratings) must return exactly what the batch code
used in the evaluation (TwoStage.recommend, Days 9-11) returns for the same history.

Small hand-computed tests: release filter, latency percentiles, history parsing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.baselines import ItemIndex  # noqa: E402
from recsys.coldstart import popularity_prior  # noqa: E402
from recsys.embeddings import center_vectors, l2_normalize  # noqa: E402
from recsys.experiment import Catalog  # noqa: E402
from recsys.explain import REASONS  # noqa: E402
from recsys.features import attach_labels  # noqa: E402
from recsys.metrics import PAD  # noqa: E402
from recsys.ranker import RankData, build_two_stage, top_from_scores, train_ranker  # noqa: E402
from recsys.serving import (Engine, UnknownMovie, export_bundle, latency_summary,  # noqa: E402
                            release_filter)

DAY = 86_400
YEAR_1996 = 820_454_400                     # 1996-01-01 UTC
MMR = {"space": "genre", "lambda": 0.7, "pool": 15}
CHOSEN = {"popularity_mode": "liked", "center": True,
          "profile": {"weighting": "graded", "half_life": 5}}


# --------------------------------------------------------------------------- #
# Synthetic world
# --------------------------------------------------------------------------- #
class FakeEncoder:
    """Text -> a fixed 16-d vector seeded by the text length (stand-in for bge-small)."""

    def encode(self, texts, **kw):
        return np.stack([np.random.default_rng(len(t)).normal(size=16) for t in texts]).astype(np.float32)


def make_world(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_items, n_users, d = 80, 60, 16
    ids = np.arange(1, n_items + 1) * 10
    raw = l2_normalize(rng.normal(size=(n_items, d)))
    vocab = ["Action", "Comedy", "Drama", "Horror"]
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
    cat = Catalog(items, index, pd.Series([""] * n_items, index=ids), G, vocab, label)

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


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    import lightgbm as lgb

    tmp = tmp_path_factory.mktemp("serving")
    cat, raw, hist = make_world()
    train_booster(cat, raw, hist, 7, 20, tmp / "warm.txt")
    train_booster(cat, raw, hist, 3, 10, tmp / "cold.txt")
    warm = lgb.Booster(model_file=str(tmp / "warm.txt"))
    cold = lgb.Booster(model_file=str(tmp / "cold.txt"))
    ts = build_two_stage(cat, raw, CHOSEN, hist, n_pop=30, m_faiss=10, faiss_feature_m=10, booster=warm)
    demo = np.arange(1, 21)
    truth = pd.DataFrame({"userId": [1, 1], "movieId": [10, 20], "rating": [4.0, 2.0]})
    settings = {"policy": {"warm_from": 3}, "mmr": MMR, "release_filter": {"enabled": False},
                "explain": {"min_cos": 0.3}, "explore": {"pool": 1000, "lambda": 0.3},
                "encoder": {"key": "fake", "hf_name": None, "query_prefix": "q: "}}
    man = export_bundle(tmp / "bundle", cat, raw, ts, warm_model=tmp / "warm.txt",
                        cold_model=tmp / "cold.txt", prior=popularity_prior(hist, cat.index),
                        demo_history=hist[hist["userId"].isin(demo)], demo_truth=truth,
                        settings=settings, meta={"split_fingerprint": "synthetic"})
    eng = Engine.load(tmp / "bundle", encoder=FakeEncoder())
    return {"cat": cat, "raw": raw, "hist": hist, "ts": ts, "warm": warm, "cold": cold,
            "eng": eng, "man": man, "demo": demo, "dir": tmp / "bundle"}


def hist_of(hist, u):
    return hist[hist["userId"] == u].sort_values(["timestamp", "movieId"], kind="stable")


# --------------------------------------------------------------------------- #
# Bundle round trip: Engine == the evaluated batch path
# --------------------------------------------------------------------------- #
def test_bundle_restores_every_item_statistic(world):
    eng, ts = world["eng"], world["ts"]
    for k in ("liked", "count", "avg_rating", "liked_share", "recent_log", "trend"):
        assert np.allclose(eng.ts.builder.stats[k], ts.builder.stats[k])
    assert np.array_equal(eng.ts.pop.score, ts.pop.score)
    assert eng.ts.builder.stats["cutoff"] == ts.builder.stats["cutoff"]
    assert world["man"]["n_demo_users"] == 20 and world["man"]["size_mb"] >= 0


def test_engine_returns_exactly_the_batch_lists(world):
    eng, ts, hist = world["eng"], world["ts"], world["hist"]
    for u in world["demo"]:
        h = hist_of(hist, u)
        ref = ts.recommend(h, np.array([u]), 10)[0]
        got = eng.recommend_user(int(u), 10, mmr=None, explain=False, release=False)
        assert [r["movieId"] for r in got["items"]] == [int(m) for m in ref if m != PAD]
        ref_mmr = ts.recommend(h, np.array([u]), 10, mmr=MMR)[0]
        got_mmr = eng.recommend_user(int(u), 10, mmr="default", explain=False, release=False)
        assert [r["movieId"] for r in got_mmr["items"]] == [int(m) for m in ref_mmr if m != PAD]


def test_scores_are_the_ranker_scores_and_sorted(world):
    eng, ts, hist = world["eng"], world["ts"], world["hist"]
    h = hist_of(hist, 3)
    fb = ts.feature_batch(h, np.array([3]))
    S = ts.scores(fb)[0]
    got = eng.recommend_user(3, 10, mmr=None, explain=False, release=False)["items"]
    by_id = {int(m): s for m, s in zip(fb.candidates[0], S) if m != PAD}
    assert [r["score"] for r in got] == pytest.approx([by_id[r["movieId"]] for r in got], abs=1e-5)
    assert all(a["score"] >= b["score"] for a, b in zip(got, got[1:]))


def test_explanations_are_well_formed(world):
    eng, hist = world["eng"], world["hist"]
    for u in world["demo"][:8]:
        liked = set(hist_of(hist, u).query("rating >= 4")["movieId"].tail(50))
        for r in eng.recommend_user(int(u), 10)["items"]:
            assert r["reason"] in REASONS and r["explanation"]
            if r["because"] is not None:
                assert r["because"]["movieId"] in liked and r["because"]["cosine"] >= 0.3
            if r["reason"] == "similar":
                assert r["explanation"] == f"Because you liked {r['because']['title']}"


def test_recommendations_never_repeat_the_history(world):
    eng, hist = world["eng"], world["hist"]
    for u in world["demo"]:
        seen = set(hist_of(hist, u)["movieId"])
        assert not seen & {r["movieId"] for r in eng.recommend_user(int(u), 20)["items"]}


# --------------------------------------------------------------------------- #
# Cold start policy, explore, popularity
# --------------------------------------------------------------------------- #
def test_zero_picks_is_popularity(world):
    eng, ts = world["eng"], world["ts"]
    out = eng.cold_start([], k=5)
    order = np.argsort(-ts.pop.score, kind="stable")[:5]
    assert out["model"] == "popularity"
    assert [r["movieId"] for r in out["items"]] == ts.pop.index.ids[order].tolist()
    assert all(r["explanation"].startswith("Widely liked") for r in out["items"])


@pytest.mark.parametrize("picks,model", [([10], "cold"), ([10, 20], "cold"), ([10, 20, 30], "warm"),
                                         ([10, 20, 30, 40, 50], "warm")])
def test_policy_picks_the_day10_model_and_matches_batch(world, picks, model):
    eng, ts = world["eng"], world["ts"]
    booster = world[model]
    h = pd.DataFrame({"userId": 0, "movieId": picks, "rating": 5.0, "timestamp": np.arange(len(picks))})
    fb = ts.feature_batch(h, np.array([0]))
    ref = top_from_scores(fb.candidates, ts.score_with(fb, booster, booster.feature_name()), 10)[0]
    out = eng.cold_start(picks, k=10, mmr=None, explain=False)
    assert out["model"] == f"{model} ranker"
    assert [r["movieId"] for r in out["items"]] == [int(m) for m in ref if m != PAD]
    assert not set(picks) & {r["movieId"] for r in out["items"]}


def test_pick_order_is_time_order(world):
    # graded weights + half-life 5: the LAST pick counts most, so the order can change the list
    eng = world["eng"]
    a = eng.cold_start([10, 20, 30, 40, 50, 60, 70], k=10, mmr=None, explain=False)
    h = eng.history_frame([10, 20, 30, 40, 50, 60, 70])
    assert h["movieId"].tolist() == [10, 20, 30, 40, 50, 60, 70]
    assert a["n_picks"] == 7


def test_explore_ranks_by_cosine_plus_prior_by_hand(world):
    eng, raw, hist = world["eng"], world["raw"], world["hist"]
    text = "folk horror"
    q = FakeEncoder().encode(["q: " + text])                       # bundle prefix "q: "
    q = l2_normalize(l2_normalize(q) - raw.mean(axis=0, keepdims=True))[0]   # same mean as items
    prior = popularity_prior(hist, eng.index)
    s = center_vectors(raw) @ q + 0.3 * prior + 1e-6 * eng.ts.pop.score    # TIE_EPS · popularity
    s[eng.index.cols([10, 20])] = -np.inf                           # picks are excluded
    ref = eng.index.ids[np.argsort(-s, kind="stable")[:8]]
    out = eng.cold_start([10, 20], text=text, k=8, mode="explore")
    assert out["model"].startswith("explore")
    assert [r["movieId"] for r in out["items"]] == ref.tolist()


def test_text_without_picks_goes_to_explore_and_text_with_picks_is_ignored(world):
    eng = world["eng"]
    assert eng.cold_start([], text="space opera", k=5)["model"].startswith("explore")
    with_text = eng.cold_start([10, 20, 30], text="space opera", k=5, mmr=None, explain=False)
    without = eng.cold_start([10, 20, 30], k=5, mmr=None, explain=False)
    assert with_text["items"] == without["items"] and with_text["note"]


def test_unknown_movies_are_rejected(world):
    with pytest.raises(UnknownMovie) as e:
        world["eng"].cold_start([10, 12345])
    assert e.value.ids == [12345]


def test_history_policy_counts_liked_movies(world):
    eng = world["eng"]
    assert eng.recommend_history([10, 20, 30], [5, 2, 1])["model"] == "cold ranker"   # 1 liked
    assert eng.recommend_history([10, 20, 30], [5, 4, 4.5])["model"] == "warm ranker"
    assert eng.recommend_history([])["model"] == "popularity"


# --------------------------------------------------------------------------- #
# Hand-computed helpers
# --------------------------------------------------------------------------- #
def test_history_frame_keeps_last_rating_and_order(world):
    h = world["eng"].history_frame([30, 10, 30, 20], [2.0, 5.0, 4.5, 3.0])
    # 30 appears twice -> its LAST rating (4.5) and its LAST position (after 10)
    assert h["movieId"].tolist() == [10, 30, 20]
    assert h["rating"].tolist() == [5.0, 4.5, 3.0]


def test_release_filter_by_hand():
    index = ItemIndex([10, 20, 30, 40])
    years = np.array([1990, 2005, np.nan, 2000])
    C = np.array([[10, 20, 30, 40, PAD]])
    S = np.array([[1.0, 2.0, 3.0, 4.0, -np.inf]])
    # cutoff 2000: 20 (2005) hidden; 30 (unknown year) kept; 40 (2000, same year) kept
    out = release_filter(S, C, index, years, 2000.0)
    assert out.tolist() == [[1.0, -np.inf, 3.0, 4.0, -np.inf]]
    assert release_filter(S, C, index, years, np.nan).tolist() == S.tolist()   # unknown cutoff


def test_release_filter_in_the_engine(world):
    eng = world["eng"]
    years = dict(zip(eng.index.ids, eng.ts.builder.years))
    for u in world["demo"]:
        out = eng.recommend_user(int(u), 20, release=True, explain=False)
        cut = out["release_cutoff_year"]
        assert all(np.isnan(years[r["movieId"]]) or years[r["movieId"]] <= cut for r in out["items"])


def test_latency_summary_by_hand():
    # numpy linear percentiles of [1, 2, 3, 4, 100]: p50 = 3; p95 at position 0.95·4 = 3.8
    # -> 4 + 0.8·(100 − 4) = 80.8; mean 22
    s = latency_summary([1, 2, 3, 4, 100])
    assert (s["p50"], s["p95"], s["mean"], s["max"], s["n"]) == (3.0, 80.8, 22.0, 100.0, 5)


def test_bundle_version_mismatch_is_refused(world, tmp_path):
    import json
    import shutil
    d = tmp_path / "b"
    shutil.copytree(world["dir"], d)
    man = json.loads((d / "bundle.json").read_text())
    man["bundle_version"] = 999
    (d / "bundle.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="bundle version"):
        Engine.load(d)


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def client(world):
    from fastapi.testclient import TestClient

    from recsys.api import create_app
    with TestClient(create_app(engine=world["eng"])) as c:
        yield c


def test_api_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and r.json()["items"] == 80


def test_api_recommend_demo_user(client, world):
    r = client.get("/recommend/3", params={"k": 7})
    body = r.json()
    assert r.status_code == 200 and len(body["items"]) == 7 and body["model"] == "warm ranker"
    assert float(r.headers["X-Process-Time-ms"]) >= 0 and body["latency_ms"] >= 0
    assert body["mmr"] == MMR
    plain = client.get("/recommend/3", params={"k": 7, "diversify": False, "explain": False}).json()
    ref = world["eng"].recommend_user(3, 7, mmr=None, explain=False)
    assert [i["movieId"] for i in plain["items"]] == [i["movieId"] for i in ref["items"]]


def test_api_errors(client):
    assert client.get("/recommend/99999").status_code == 404
    assert client.get("/recommend/3", params={"k": 0}).status_code == 422
    assert client.get("/recommend/3", params={"k": 51}).status_code == 422
    assert client.post("/recommend/cold-start", json={"picks": [10, 777]}).status_code == 422
    assert client.post("/recommend/cold-start", json={"picks": [10], "ratings": [5, 4]}).status_code == 422
    assert client.post("/recommend/cold-start", json={"mode": "explore"}).status_code == 422
    assert client.get("/movies/123456").status_code == 404
    assert client.get("/movies/123456/similar").status_code == 404


def test_api_cold_start_and_history(client):
    r = client.post("/recommend/cold-start", json={"picks": [10, 20, 30], "k": 5}).json()
    assert r["model"] == "warm ranker" and len(r["items"]) == 5 and r["n_picks"] == 3
    r = client.post("/recommend/cold-start", json={}).json()
    assert r["model"] == "popularity"
    r = client.post("/recommend/cold-start", json={"text": "heist gone wrong", "k": 4}).json()
    assert r["model"].startswith("explore") and len(r["items"]) == 4
    r = client.post("/recommend/history", json={"ratings": [{"movieId": 10, "rating": 4.5},
                                                            {"movieId": 20}], "k": 3}).json()
    assert r["model"] == "cold ranker" and len(r["items"]) == 3


def test_api_movies_and_users(client):
    s = client.get("/movies/search", params={"q": "movie 10"}).json()["items"]
    assert s and all("movie 10" in i["title"].lower() for i in s)
    m = client.get("/movies/10").json()
    assert m["title"] == "Movie 10" and m["poster_url"].endswith("/p10.jpg")
    sim = client.get("/movies/10/similar", params={"k": 4}).json()["items"]
    assert len(sim) == 4 and 10 not in [i["movieId"] for i in sim]
    assert client.get("/users/demo").json()["n"] == 20
    h = client.get("/users/1/history").json()
    assert [x["movieId"] for x in h["next_liked"]] == [10]     # truth: 10 (4★) liked, 20 (2★) not
