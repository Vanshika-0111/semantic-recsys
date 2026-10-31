"""Day 12: FastAPI service around serving.Engine.

    uvicorn recsys.api:app --port 8000          (bundle from $RECSYS_BUNDLE, default data/serving)
    open http://localhost:8000/docs             (interactive Swagger page for every endpoint)

Endpoints
  GET  /health                         bundle info, model sizes, settings in use
  GET  /recommend/{user_id}            a demo (warm) user: two-stage ranker + genre MMR + reasons
  POST /recommend/history              any user's own ratings (time order) -> policy ranker
  POST /recommend/cold-start           onboarding picks and/or free text (auto | explore mode)
  GET  /users/demo                     some demo user ids (for the UI's "log in as")
  GET  /users/{user_id}/history        a demo user's liked movies + what they rated next (test)
  GET  /movies/search?q=               title search for onboarding
  GET  /movies/{movie_id}              metadata
  GET  /movies/{movie_id}/similar      nearest movies in the (mean-centered) embedding space

Every response carries `latency_ms` (time inside the engine) and the header X-Process-Time-ms
(whole request incl. validation + JSON). Bad input -> 422 with a readable message, unknown demo
user or movie -> 404, text search without an encoder -> 503.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, model_validator

from . import config

MAX_K = 50
MAX_HISTORY = 5000


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class Because(BaseModel):
    movieId: int
    title: str
    cosine: float | None = None


class Item(BaseModel):
    movieId: int
    title: str
    year: int | None = None
    genres: list[str] = []
    poster_url: str | None = None
    score: float | None = None
    reason: str | None = Field(None, description="similar | genre | era | popular | quality | search")
    explanation: str | None = None
    because: Because | None = None


class RecResponse(BaseModel):
    model: str
    k: int
    items: list[Item]
    latency_ms: float
    user_id: int | None = None
    n_history: int | None = None
    n_picks: int | None = None
    mmr: dict | None = None
    release_cutoff_year: int | None = None
    note: str | None = None


class Rating(BaseModel):
    movieId: int
    rating: float = Field(5.0, ge=0.5, le=5.0)


class HistoryRequest(BaseModel):
    ratings: list[Rating] = Field(..., max_length=MAX_HISTORY,
                                  description="the user's ratings, oldest first")
    k: int = Field(10, ge=1, le=MAX_K)
    diversify: bool = True
    explain: bool = True


class ColdStartRequest(BaseModel):
    picks: list[int] = Field([], max_length=100, description="movies the user liked, in pick order")
    ratings: list[float] | None = Field(None, description="optional rating per pick (default 5★)")
    text: str | None = Field(None, max_length=500, description="free-text taste description")
    mode: Literal["auto", "explore"] = "auto"
    k: int = Field(10, ge=1, le=MAX_K)
    diversify: bool = True
    explain: bool = True

    @model_validator(mode="after")
    def _check(self):
        if self.ratings is not None and len(self.ratings) != len(self.picks):
            raise ValueError("ratings must have one value per pick")
        if self.ratings is not None and any(not 0.5 <= r <= 5.0 for r in self.ratings):
            raise ValueError("ratings must be between 0.5 and 5")
        if self.mode == "explore" and not (self.text or "").strip():
            raise ValueError("explore mode needs a text")
        return self


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
def create_app(engine=None, bundle_dir: str | Path | None = None) -> FastAPI:
    """engine: an already loaded serving.Engine (tests). Otherwise the bundle is loaded at startup
    from bundle_dir / $RECSYS_BUNDLE / data/serving."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.engine is None:
            from .experiment import setup_runtime
            setup_runtime()
            from .serving import Engine
            d = bundle_dir or os.environ.get("RECSYS_BUNDLE") or config.DATA_DIR / "serving"
            app.state.engine = Engine.load(d)
            # load the text encoder now, not on the first free-text request (~20 s on a laptop
            # the first time); RECSYS_PRELOAD_ENCODER=0 skips it (faster startup, lazy load)
            if os.environ.get("RECSYS_PRELOAD_ENCODER", "1") != "0":
                try:
                    app.state.engine.encoder()
                except RuntimeError:
                    pass                         # no sentence-transformers: text search -> 503
        yield

    app = FastAPI(title="Semantic Recommendation & Ranking Engine",
                  description="MovieLens 32M + TMDB · bge-small embeddings · FAISS · LightGBM LambdaMART",
                  version="1.0.0", lifespan=lifespan)
    app.state.engine = engine

    @app.middleware("http")
    async def timing(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-ms"] = f"{1000 * (time.perf_counter() - t0):.2f}"
        return response

    def eng():
        return app.state.engine

    def run(fn, *args, **kw) -> tuple[dict | list, float]:
        from .serving import UnknownMovie
        t0 = time.perf_counter()
        try:
            out = fn(*args, **kw)
        except UnknownMovie as e:
            raise HTTPException(422, f"unknown movieId(s): {e.ids[:10]}") from e
        except RuntimeError as e:
            raise HTTPException(503, str(e)) from e
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        return out, round(1000 * (time.perf_counter() - t0), 2)

    def mmr(diversify: bool):
        return "default" if diversify else None

    @app.get("/health")
    def health():
        return {"status": "ok", **eng().info()}

    @app.get("/recommend/{user_id}", response_model=RecResponse)
    def recommend_user(user_id: int, k: int = Query(10, ge=1, le=MAX_K), diversify: bool = True,
                       explain: bool = True, release_filter: bool | None = None):
        if not eng().has_user(user_id):
            raise HTTPException(404, f"user {user_id} is not a demo user — GET /users/demo lists some, "
                                     "or POST /recommend/history with your own ratings")
        out, ms = run(eng().recommend_user, user_id, k, mmr(diversify), explain, release_filter)
        return {**out, "k": k, "latency_ms": ms}

    @app.post("/recommend/history", response_model=RecResponse)
    def recommend_history(req: HistoryRequest):
        out, ms = run(eng().recommend_history, [r.movieId for r in req.ratings],
                      [r.rating for r in req.ratings], req.k, mmr(req.diversify), req.explain)
        return {**out, "k": req.k, "latency_ms": ms}

    @app.post("/recommend/cold-start", response_model=RecResponse)
    def cold_start(req: ColdStartRequest):
        out, ms = run(eng().cold_start, req.picks, req.ratings, req.text, req.k, req.mode,
                      mmr(req.diversify), req.explain)
        return {**out, "k": req.k, "latency_ms": ms}

    @app.get("/users/demo")
    def demo_users(limit: int = Query(20, ge=1, le=1000)):
        u = eng().demo_users
        return {"n": int(len(u)), "user_ids": [int(x) for x in u[:limit]]}

    @app.get("/users/{user_id}/history")
    def user_history(user_id: int, limit: int = Query(20, ge=1, le=500)):
        e = eng()
        if not e.has_user(user_id):
            raise HTTPException(404, f"user {user_id} is not a demo user")
        h = e.user_history(user_id)
        liked = h[h["rating"] >= e.ts.builder.threshold].tail(limit).iloc[::-1]
        nxt = e.demo_truth[e.demo_truth["userId"] == user_id]
        row = lambda m, r: {**{k: v for k, v in e.movie(int(m)).items() if k != "overview"}, "rating": float(r)}  # noqa: E731
        return {"user_id": user_id, "n_ratings": int(len(h)), "n_liked": int((h["rating"] >= 4).sum()),
                "recent_liked": [row(m, r) for m, r in zip(liked["movieId"], liked["rating"])],
                "next_liked": [row(m, r) for m, r in zip(nxt["movieId"], nxt["rating"]) if r >= 4]}

    @app.get("/movies/search")
    def search(q: str = Query(..., min_length=1, max_length=100), limit: int = Query(10, ge=1, le=50)):
        out, ms = run(eng().search, q, limit)
        return {"query": q, "items": out, "latency_ms": ms}

    @app.get("/movies/{movie_id}")
    def movie(movie_id: int):
        try:
            return eng().movie(movie_id)
        except KeyError as e:
            raise HTTPException(404, f"movie {movie_id} not in the catalog") from e

    @app.get("/movies/{movie_id}/similar")
    def similar(movie_id: int, k: int = Query(10, ge=1, le=MAX_K)):
        try:
            out, ms = run(eng().similar, movie_id, k)
        except HTTPException as e:
            raise HTTPException(404, f"movie {movie_id} not in the catalog") from e
        return {"movieId": movie_id, "items": out, "latency_ms": ms}

    return app


app = create_app()
