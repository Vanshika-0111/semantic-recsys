"""Day 12: the serving side — a compact bundle on disk + an Engine that answers single requests.

Why a bundle: the experiments rebuild everything from 31M ratings (minutes, GBs of RAM). An API
must start in seconds on a small machine. Everything the two-stage recommender needs that comes
from ratings is computed ONCE (scripts/run_serving.py, stage `bundle`, history = train+val, the
same history Day 10 tested with) and saved:

    bundle.json          settings chosen on validation (Days 3-11), provenance, file list
    arrays.npz           raw item vectors (Day 4), genres, years, runtime, popularity score,
                         cold-start prior and the FeatureBuilder item statistics (Day 8)
    items.parquet        movie metadata for display (title, year, genres, poster, overview)
    warm_ranker.txt      Day 9 LightGBM ranker          cold_ranker.txt   Day 9 cold-mode ranker
    demo_history.parquet train+val ratings of a sample of warm test users (the demo's "log in as")
    demo_truth.parquet   their test ratings (the UI can mark what they really watched next)

No ratings are needed to load it: the Engine rebuilds the SAME objects the evaluation used
(PopularityRecommender, EmbeddingRecommender, FlatIndex, FeatureBuilder with its statistics,
TwoStage, Explainer, ColdStartRecommender) from these arrays. run_serving.py's `check` stage
proves it: for sampled users the Engine returns exactly the lists the batch evaluation code
returns (warm users, MMR, and real cold users at 1/3/5 picks).

Serving rules (each chosen on validation earlier — nothing new is tuned here):
  warm user   stage 1 popular 400 ∪ FAISS 100 -> 21 features -> warm LambdaMART ranker
              -> genre MMR λ 0.7 over the top 50 (Day 11) -> SHAP reason + "Because you liked X"
  new user    0 picks -> popularity; 1-2 picks -> cold ranker; >= 3 picks -> warm ranker
              (Day 9 policy, confirmed on the real cold users on Day 10)
  free text   explore mode (Day 6): semantic search over the 1,000 most-liked movies,
              cos(text, movie) + 0.3 · popularity prior — a UX feature, not an accuracy claim
  release     optional filter: hide movies released after the user's last rating year (demo
              users replay the past). Turned on only if the validation check in run_serving.py
              says so (Day 10 note: +2.8% on test, but never checked on validation).
"""
from __future__ import annotations

import dataclasses
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import ItemIndex, PopularityRecommender, user_item_matrix
from .coldstart import ColdStartRecommender
from .features import FeatureBuilder
from .metrics import PAD
from .ranker import TwoStage

BUNDLE_VERSION = 1
STAT_KEYS = ("count", "liked", "liked_log", "count_log", "avg_rating", "liked_share",
             "recent_log", "trend")
HISTORY_COLS = ["userId", "movieId", "rating", "timestamp"]
SOURCE_USER = 0          # userId given to an ad-hoc history inside the engine


class UnknownMovie(KeyError):
    """A request named movieIds that are not in the catalog."""

    def __init__(self, ids) -> None:
        self.ids = [int(i) for i in ids]
        super().__init__(f"unknown movieId(s): {self.ids[:10]}")


# --------------------------------------------------------------------------- #
# Export (run once, from the experiment data)
# --------------------------------------------------------------------------- #
def export_bundle(out_dir: Path, catalog, raw_vectors: np.ndarray, ts: TwoStage, *,
                  warm_model: Path, cold_model: Path | None, prior: np.ndarray,
                  demo_history: pd.DataFrame, demo_truth: pd.DataFrame | None,
                  settings: dict, meta: dict) -> dict:
    """Write the serving bundle for a TwoStage that was fit on the serving history.

    ts           built with build_two_stage(...) on train+val (its pop / builder hold the item
                 statistics that get saved; its booster is not used — the model FILES are copied)
    prior        Day 6 cold-start prior (all-time liked popularity in [0, 1]) for explore mode
    settings     policy / mmr / release_filter / explain / explore / encoder blocks (bundle.json)
    meta         provenance (split fingerprint, git commit, history description ...)
    Returns the bundle.json content."""
    import shutil

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    b = ts.builder
    arrays = {"movie_ids": catalog.index.ids, "raw_vectors": np.asarray(raw_vectors, np.float32),
              "genres": np.asarray(catalog.genres, np.float32), "years": b.years,
              "runtime": b.runtime, "pop_score": ts.pop.score, "prior": np.asarray(prior, np.float64)}
    arrays.update({f"stat_{k}": np.asarray(b.stats[k]) for k in STAT_KEYS})
    np.savez(out_dir / "arrays.npz", **arrays)

    it = catalog.items
    items = pd.DataFrame({
        "movieId": catalog.index.ids,
        "title": it["title_clean"] if "title_clean" in it else it.get("title"),
        "year": pd.to_numeric(it["year"], errors="coerce").astype("Int64"),
        "genres": [" | ".join(catalog.vocab[j] for j in np.flatnonzero(g)) for g in catalog.genres],
        "label": [catalog.label[m] for m in catalog.index.ids],
        "poster_path": it["poster_path"] if "poster_path" in it else None,
        "overview": it["overview"] if "overview" in it else None,
        "tmdbId": pd.to_numeric(it["tmdbId"], errors="coerce").astype("Int64") if "tmdbId" in it else None,
    })
    items.to_parquet(out_dir / "items.parquet", index=False)

    hist = demo_history[HISTORY_COLS].sort_values(["userId", "timestamp", "movieId"], kind="stable")
    if len(hist):                     # no demo users -> no file (Engine.load treats it as empty)
        hist.to_parquet(out_dir / "demo_history.parquet", index=False)
    if demo_truth is not None and len(demo_truth):
        demo_truth[["userId", "movieId", "rating"]].sort_values(["userId", "movieId"]).to_parquet(
            out_dir / "demo_truth.parquet", index=False)

    shutil.copyfile(warm_model, out_dir / "warm_ranker.txt")
    if cold_model is not None:
        shutil.copyfile(cold_model, out_dir / "cold_ranker.txt")

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        **meta,
        "stage1": {"n_pop": ts.n_pop, "m_faiss": ts.m_faiss, "faiss_feature_m": ts.faiss_feature_m},
        "popularity_mode": ts.pop.mode,
        "feature_builder": {"center": b.center, "weighting": b.weighting, "half_life": b.half_life,
                            "threshold": b.threshold, "maxcos_history": b.maxcos_history,
                            "recent_days": b.recent_days, "prior_votes": b.prior_votes,
                            "global_mean": float(b.stats["global_mean"]),
                            "cutoff": None if b.stats["cutoff"] is None else int(b.stats["cutoff"])},
        "genre_vocab": list(catalog.vocab),
        "models": {"warm": "warm_ranker.txt", "cold": "cold_ranker.txt" if cold_model else None},
        "n_items": int(len(catalog.index)),
        "n_demo_users": int(hist["userId"].nunique()),
        **settings,
    }
    (out_dir / "bundle.json").write_text(json.dumps(manifest, indent=2))
    manifest["size_mb"] = round(sum(p.stat().st_size for p in out_dir.iterdir()) / 2**20, 1)
    (out_dir / "bundle.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# --------------------------------------------------------------------------- #
# The engine (load once, answer many requests)
# --------------------------------------------------------------------------- #
@dataclass
class Engine:
    dir: Path
    manifest: dict
    index: ItemIndex
    items: pd.DataFrame                   # index order
    ts: TwoStage                          # warm ranker inside
    cold_booster: object | None
    explainer: object                     # explain.Explainer for the warm booster
    cold_explainer: object | None
    coldstart: ColdStartRecommender       # explore mode (free text)
    demo_history: pd.DataFrame
    demo_truth: pd.DataFrame
    encoder_loader: object = None         # () -> encoder, called on the first text request
    load_seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _demo_users: np.ndarray = field(default=None, repr=False)

    # ---------------- loading ----------------
    @classmethod
    def load(cls, bundle_dir: Path | str, encoder=None) -> "Engine":
        """encoder: an object with .encode() (tests pass a fake); default = sentence-transformers
        model named in the bundle, loaded lazily on the first free-text request."""
        import lightgbm as lgb

        from .embeddings import EmbeddingRecommender, FlatIndex
        from .explain import Explainer

        t0 = time.time()
        d = Path(bundle_dir)
        if not (d / "bundle.json").exists():
            raise FileNotFoundError(f"no serving bundle in {d} — run scripts/run_serving.py --stages bundle")
        man = json.loads((d / "bundle.json").read_text())
        if man.get("bundle_version") != BUNDLE_VERSION:
            raise ValueError(f"bundle version {man.get('bundle_version')} != {BUNDLE_VERSION}; rebuild it")
        A = np.load(d / "arrays.npz")
        index = ItemIndex(A["movie_ids"])
        if not np.array_equal(index.ids, A["movie_ids"]):
            raise ValueError("movie_ids in the bundle are not sorted/unique")
        raw = A["raw_vectors"]
        fbc = man["feature_builder"]
        pop = PopularityRecommender(man["popularity_mode"], fbc["threshold"], index, A["pop_score"])
        emb = EmbeddingRecommender(raw, index, weighting=fbc["weighting"], threshold=fbc["threshold"],
                                   center=fbc["center"], half_life=fbc["half_life"])
        flat = FlatIndex(emb.item_vectors, index.ids)
        runtime = A["runtime"]
        builder = FeatureBuilder(raw, index, A["genres"], A["years"], runtime, center=fbc["center"],
                                 weighting=fbc["weighting"], half_life=fbc["half_life"],
                                 threshold=fbc["threshold"], maxcos_history=fbc["maxcos_history"],
                                 recent_days=fbc["recent_days"], prior_votes=fbc["prior_votes"])
        builder.stats = {k: A[f"stat_{k}"] for k in STAT_KEYS}
        builder.stats["global_mean"], builder.stats["cutoff"] = fbc["global_mean"], fbc["cutoff"]

        warm = lgb.Booster(model_file=str(d / man["models"]["warm"]))
        cold = (lgb.Booster(model_file=str(d / man["models"]["cold"]))
                if man["models"].get("cold") and (d / man["models"]["cold"]).exists() else None)
        s1 = man["stage1"]
        ts = TwoStage(pop, emb, flat, builder, s1["n_pop"], s1["m_faiss"], s1["faiss_feature_m"],
                      warm, list(warm.feature_name()))

        items = pd.read_parquet(d / "items.parquet")
        items = items.set_index("movieId").loc[index.ids].reset_index()
        catalog = _CatalogView(index, A["genres"], list(man["genre_vocab"]),
                               dict(zip(items["movieId"].to_numpy(), items["label"])))
        min_cos = man.get("explain", {}).get("min_cos", 0.3)
        explainer = Explainer.from_two_stage(ts, catalog, min_cos)
        cold_explainer = (dataclasses.replace(explainer, booster=cold, features=list(cold.feature_name()))
                          if cold is not None else None)

        enc = man.get("encoder", {})
        cs = ColdStartRecommender(raw, index, A["prior"], A["pop_score"],
                                  blend={"text": dict(man.get("explore", {"pool": 1000, "lambda": 0.3}))},
                                  center=fbc["center"], weighting=fbc["weighting"],
                                  half_life=fbc["half_life"], threshold=fbc["threshold"],
                                  query_prefix=enc.get("query_prefix", ""), encoder=encoder)
        loader = (_encoder_loader(enc["hf_name"]) if encoder is None and enc.get("hf_name") else None)

        # Day 13: demo users are optional — the public bundle ships without them (the MovieLens
        # licence does not allow redistributing ratings), the laptop bundle has 1,000
        hist = (pd.read_parquet(d / "demo_history.parquet")[HISTORY_COLS]
                if (d / "demo_history.parquet").exists() else
                pd.DataFrame({c: pd.Series(dtype="float64" if c == "rating" else "int64")
                              for c in HISTORY_COLS}))
        hist = hist.sort_values(["userId", "timestamp", "movieId"], kind="stable").reset_index(drop=True)
        truth = (pd.read_parquet(d / "demo_truth.parquet") if (d / "demo_truth.parquet").exists()
                 else pd.DataFrame(columns=["userId", "movieId", "rating"]))
        eng = cls(d, man, index, items, ts, cold, explainer, cold_explainer, cs, hist, truth, loader)
        eng._demo_users = np.unique(hist["userId"].to_numpy(np.int64))
        eng.load_seconds = round(time.time() - t0, 2)
        return eng

    # ---------------- small helpers ----------------
    @property
    def demo_users(self) -> np.ndarray:
        return self._demo_users

    @property
    def mmr_default(self) -> dict | None:
        return self.manifest.get("mmr")

    @property
    def release_filter_default(self) -> bool:
        return bool(self.manifest.get("release_filter", {}).get("enabled", False))

    def check_movies(self, ids) -> np.ndarray:
        ids = np.asarray(list(ids), dtype=np.int64)
        pos = np.searchsorted(self.index.ids, ids)
        ok = (pos < len(self.index)) & (self.index.ids[np.minimum(pos, len(self.index) - 1)] == ids)
        if not ok.all():
            raise UnknownMovie(ids[~ok])
        return ids

    def movie(self, movie_id: int) -> dict:
        c = int(self.index.cols(self.check_movies([movie_id]))[0])
        return self._movie_row(c)

    def _movie_row(self, c: int) -> dict:
        """Display fields of catalog column c (cached: built once per movie, then a dict copy)."""
        cache = self.__dict__.setdefault("_row_cache", {})
        if c not in cache:
            cache[c] = self._build_row(c)
        return dict(cache[c])

    def _build_row(self, c: int) -> dict:
        r = self.items.iloc[c]
        poster = r.get("poster_path")
        base = self.manifest.get("poster_base", "https://image.tmdb.org/t/p/w342")
        year = r.get("year")
        return {"movieId": int(r["movieId"]), "title": str(r["title"]),
                "year": None if pd.isna(year) else int(year),
                "genres": [g for g in str(r["genres"] or "").split(" | ") if g],
                "poster_url": f"{base}{poster}" if isinstance(poster, str) and poster else None,
                "overview": r.get("overview") if isinstance(r.get("overview"), str) else None,
                "liked_count": int(self.ts.builder.stats["liked"][c])}

    def has_user(self, user_id: int) -> bool:
        u = self.demo_users
        p = np.searchsorted(u, user_id)
        return bool(p < len(u) and u[p] == user_id)

    def user_history(self, user_id: int) -> pd.DataFrame:
        if not self.has_user(user_id):
            raise KeyError(user_id)
        uid = self.demo_history["userId"].to_numpy()
        a, b = np.searchsorted(uid, user_id), np.searchsorted(uid, user_id, side="right")
        return self.demo_history.iloc[a:b]

    def history_frame(self, movie_ids, ratings=None, timestamps=None) -> pd.DataFrame:
        """An ad-hoc history (in the order given = time order) as the engine's history frame.
        A movie listed twice keeps its LAST rating (the newest opinion)."""
        ids = self.check_movies(movie_ids)
        n = len(ids)
        r = np.full(n, 5.0) if ratings is None else np.asarray(ratings, dtype=np.float64)
        if len(r) != n:
            raise ValueError("ratings must have one value per movie")
        t = np.arange(n, dtype=np.int64) if timestamps is None else np.asarray(timestamps, dtype=np.int64)
        h = pd.DataFrame({"userId": np.full(n, SOURCE_USER, dtype=np.int64), "movieId": ids,
                          "rating": r, "timestamp": t, "_pos": np.arange(n)})
        h = h.drop_duplicates("movieId", keep="last").sort_values(["timestamp", "_pos"], kind="stable")
        return h.drop(columns="_pos").reset_index(drop=True)

    def cutoff_year(self, hist: pd.DataFrame) -> float:
        from .evaluation import timestamp_year
        if hist.empty:
            return np.nan
        return float(timestamp_year(np.array([int(hist["timestamp"].max())]))[0])

    # ---------------- recommending ----------------
    def rank(self, hist: pd.DataFrame, k: int = 10, model: str = "warm", mmr: dict | None = None,
             explain: bool = True, cutoff_year: float | None = None,
             timings: dict | None = None) -> list[dict]:
        """One user's history (userId column ignored) -> top-k dicts, the full two-stage path.
        model: 'warm' | 'cold' (Day 9 ranker trained on pseudo-cold users).
        timings: if a dict is given, milliseconds per step are added to it (latency breakdown)."""
        clock = _Clock(timings)
        h = hist.assign(userId=SOURCE_USER)
        users = np.array([SOURCE_USER], dtype=np.int64)
        booster, expl = ((self.cold_booster, self.cold_explainer) if model == "cold"
                         else (self.ts.booster, self.explainer))
        if booster is None:
            raise ValueError(f"no {model} ranker in this bundle")
        C, ranks = self.ts.stage1(h, users)
        clock("stage1_ms")
        fb = self.ts.builder.transform(users, h, C, ranks)
        clock("features_ms")
        feats = list(booster.feature_name())
        S = self.ts.score_with(fb, booster, feats)
        if cutoff_year is not None and not np.isnan(cutoff_year):
            S = release_filter(S, fb.candidates, self.index, self.ts.builder.years, cutoff_year)
        clock("ranker_ms")
        top = self.ts.top(fb, S, k, mmr)
        clock("mmr_ms")
        col = {int(m): j for j, m in enumerate(fb.candidates[0]) if m != PAD}
        score = {int(m): float(S[0, col[int(m)]]) for m in top[0] if m != PAD}
        reasons = expl.explain(fb, h, top)[0] if explain else [{"movieId": int(m)} for m in top[0] if m != PAD]
        clock("explain_ms")
        out = [self._item(r, score.get(r["movieId"])) for r in reasons]
        clock("format_ms")
        return out

    def recommend_user(self, user_id: int, k: int = 10, mmr: dict | None | str = "default",
                       explain: bool = True, release: bool | None = None) -> dict:
        """A demo (warm) user from the bundle."""
        hist = self.user_history(user_id)
        mmr = self.mmr_default if mmr == "default" else mmr
        rel = self.release_filter_default if release is None else release
        cut = self.cutoff_year(hist) if rel else None
        recs = self.rank(hist, k, "warm", mmr, explain, cut)
        return {"user_id": int(user_id), "model": "warm ranker", "n_history": int(len(hist)),
                "mmr": mmr, "release_cutoff_year": None if cut is None else int(cut), "items": recs}

    def recommend_history(self, movie_ids, ratings=None, k: int = 10,
                          mmr: dict | None | str = "default", explain: bool = True) -> dict:
        """Any user's own rating history (time order). No release filter: a live user's 'now'."""
        hist = self.history_frame(movie_ids, ratings)
        mmr = self.mmr_default if mmr == "default" else mmr
        if hist.empty:
            return {"model": "popularity", "n_history": 0, "mmr": None, "items": self.popular(k)}
        n_liked = int((hist["rating"].to_numpy() >= self.ts.builder.threshold).sum())
        model = "cold" if n_liked < self.warm_from and self.cold_booster is not None else "warm"
        return {"model": f"{model} ranker", "n_history": int(len(hist)), "mmr": mmr,
                "items": self.rank(hist, k, model, mmr, explain)}

    @property
    def warm_from(self) -> int:
        return int(self.manifest.get("policy", {}).get("warm_from", 3))

    def policy(self, n_picks: int) -> str:
        from .evaluation import policy_model
        p = policy_model(n_picks, self.warm_from)
        return "warm" if p == "cold" and self.cold_booster is None else p

    def cold_start(self, picks=(), ratings=None, text: str | None = None, k: int = 10,
                   mode: str = "auto", mmr: dict | None | str = "default", explain: bool = True) -> dict:
        """Onboarding. mode 'auto': picks -> Day 9/10 policy (popularity / cold / warm ranker);
        no picks but text -> explore; nothing -> popularity. mode 'explore': text search, picks
        are only excluded. The text adds no accuracy (Day 6), so with picks it is not used."""
        picks = [int(p) for p in picks]
        self.check_movies(picks)
        text = (text or "").strip()
        mmr = self.mmr_default if mmr == "default" else mmr
        note = None
        if mode == "explore" or (not picks and text):
            if not text:
                raise ValueError("explore mode needs a text")
            return {"model": "explore (semantic search)", "n_picks": len(picks), "mmr": None,
                    "items": self.explore(text, k, exclude=picks), "note": None}
        if text:
            note = "text ignored: with picks the ranker is used (text adds no accuracy, Day 6)"
        which = self.policy(len(set(picks)))
        if which == "popularity":
            return {"model": "popularity", "n_picks": 0, "mmr": None, "items": self.popular(k),
                    "note": note}
        hist = self.history_frame(picks, ratings)
        return {"model": f"{which} ranker", "n_picks": int(len(hist)), "mmr": mmr,
                "items": self.rank(hist, k, which, mmr, explain), "note": note}

    def popular(self, k: int = 10, exclude=()) -> list[dict]:
        ids = self.check_movies(exclude)
        h = pd.DataFrame({"userId": SOURCE_USER, "movieId": ids})
        seen = user_item_matrix(h, np.array([SOURCE_USER]), self.index, dtype=np.int8)
        top = self.ts.pop.recommend(seen, k)[0]
        return [self._item({"movieId": int(m), "reason": "popular",
                            "text": self.explainer._popular(int(self.index.cols([m])[0]))}, None)
                for m in top if m != PAD]

    def encoder(self):
        with self._lock:
            if self.coldstart.encoder is None:
                if self.encoder_loader is None:
                    raise RuntimeError("no text encoder available in this deployment")
                self.coldstart.encoder = self.encoder_loader()
        return self.coldstart.encoder

    def explore(self, text: str, k: int = 10, exclude=()) -> list[dict]:
        self.encoder()
        recs = self.coldstart.recommend((), text=text, k=k, exclude=list(exclude))
        return [self._item({"movieId": r["movieId"], "reason": "search",
                            "text": "Matches what you described"}, r["score"]) for r in recs]

    def similar(self, movie_id: int, k: int = 10) -> list[dict]:
        sims, ids = self.ts.flat.similar(self.check_movies([movie_id]), k)
        return [self._item({"movieId": int(m), "reason": "similar",
                            "text": f"Similar to {self.explainer.name(int(self.index.cols([movie_id])[0]))}"},
                           float(s)) for m, s in zip(ids[0], sims[0]) if m != PAD]

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Title search for onboarding: case-insensitive substring, most-liked first."""
        q = (query or "").strip().lower()
        if not q:
            return []
        hit = np.flatnonzero(self.items["title"].str.lower().str.contains(q, regex=False).to_numpy())
        order = hit[np.argsort(-self.ts.pop.score[hit], kind="stable")][:limit]
        return [self._movie_row(int(c)) for c in order]

    def _item(self, r: dict, score: float | None) -> dict:
        c = int(self.index.cols([r["movieId"]])[0])
        m = self._movie_row(c)
        because = r.get("because")
        return {**{k: m[k] for k in ("movieId", "title", "year", "genres", "poster_url")},
                "score": None if score is None or not np.isfinite(score) else round(float(score), 5),
                "reason": r.get("reason"), "explanation": r.get("text"),
                "because": None if because is None else
                {"movieId": int(because), "title": self.explainer.name(int(self.index.cols([because])[0])),
                 "cosine": r.get("because_cos")}}

    def info(self) -> dict:
        m = self.manifest
        return {"items": int(len(self.index)), "demo_users": int(len(self.demo_users)),
                "warm_ranker_trees": int(self.ts.booster.num_trees()),
                "cold_ranker": self.cold_booster is not None, "mmr": m.get("mmr"),
                "release_filter": bool(m.get("release_filter", {}).get("enabled", False)),
                "policy_warm_from": self.warm_from, "split_fingerprint": (m.get("split_fingerprint") or "")[:12],
                "git_commit": (m.get("git_commit") or "")[:7], "load_seconds": self.load_seconds,
                "text_search": self.coldstart.encoder is not None or self.encoder_loader is not None}


def _encoder_loader(hf_name: str):
    """() -> the sentence-transformers model; None if the package isn't installed (then free-text
    search answers 503 and everything else still works)."""
    import importlib.util

    if importlib.util.find_spec("sentence_transformers") is None:
        return None

    def load():
        from .embeddings import load_encoder
        return load_encoder(hf_name, device="cpu")
    return load


class _Clock:
    """Adds the milliseconds since the previous call to timings[name] (no-op without a dict)."""

    def __init__(self, timings: dict | None) -> None:
        self.t, self.timings = time.perf_counter(), timings

    def __call__(self, name: str) -> None:
        if self.timings is not None:
            now = time.perf_counter()
            self.timings[name] = self.timings.get(name, 0.0) + 1000 * (now - self.t)
            self.t = now


@dataclass
class _CatalogView:
    """The parts of experiment.Catalog that Explainer.from_two_stage reads."""
    index: ItemIndex
    genres: np.ndarray
    vocab: list[str]
    label: dict


def release_filter(S: np.ndarray, C: np.ndarray, index, years: np.ndarray, cutoff) -> np.ndarray:
    """Scores with every candidate released after the user's cutoff year set to -inf (never
    shown). Unknown release year or unknown cutoff -> kept. cutoff: scalar or (U,)."""
    from .evaluation import released_after

    cut = np.broadcast_to(np.asarray(cutoff, dtype=np.float64), (C.shape[0],))
    return np.where(released_after(C, index, years, cut), -np.inf, S)


# --------------------------------------------------------------------------- #
# Latency summary
# --------------------------------------------------------------------------- #
def latency_summary(ms) -> dict:
    """p50 / p95 / p99 / mean / max of per-request milliseconds (numpy 'linear' percentiles)."""
    x = np.asarray(ms, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    p50, p95, p99 = np.percentile(x, [50, 95, 99])
    return {"n": int(x.size), "p50": round(float(p50), 2), "p95": round(float(p95), 2),
            "p99": round(float(p99), 2), "mean": round(float(x.mean()), 2), "max": round(float(x.max()), 2)}
