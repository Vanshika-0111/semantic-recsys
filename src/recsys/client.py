"""Day 13: a small HTTP client for the recommendation API (used by the Streamlit UI).

The UI never imports the model code — it only talks HTTP to the FastAPI service, exactly like a
web or mobile front end would. That keeps the UI process light and means the numbers it shows are
the API's (the one with the golden check, Day 12).

    c = ApiClient("http://localhost:8000")
    c.health()                                   # bundle info
    c.cold_start(picks=[318, 2571, 4993])        # onboarding -> policy ranker
    c.explore("heist movie that goes wrong")     # free text -> semantic search
"""
from __future__ import annotations

import os
from pathlib import Path

import httpx

DEFAULT_URL = "http://localhost:8000"


class ApiError(RuntimeError):
    """The API answered with an error (status + its readable `detail`) or could not be reached."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 30.0, transport=None) -> None:
        self.base_url = (base_url or os.environ.get("RECSYS_API_URL") or DEFAULT_URL).rstrip("/")
        self._http = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    # ---------------- plumbing ----------------
    def _call(self, method: str, path: str, **kw) -> dict:
        try:
            r = self._http.request(method, path, **kw)
        except httpx.HTTPError as e:                    # connection refused, timeout ...
            raise ApiError(f"API not reachable at {self.base_url} ({type(e).__name__})") from e
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            if isinstance(detail, list):                # pydantic validation errors
                detail = "; ".join(str(d.get("msg", d)) for d in detail)
            raise ApiError(f"{detail}", r.status_code)
        out = r.json()
        if isinstance(out, dict) and "X-Process-Time-ms" in r.headers:
            out["_http_ms"] = float(r.headers["X-Process-Time-ms"])
        return out

    # ---------------- endpoints ----------------
    def health(self) -> dict:
        return self._call("GET", "/health")

    def popular(self, k: int = 10) -> dict:
        return self.cold_start([], k=k)

    def cold_start(self, picks=(), text: str | None = None, k: int = 10, mode: str = "auto",
                   diversify: bool = True, explain: bool = True, ratings=None) -> dict:
        body = {"picks": [int(p) for p in picks], "k": k, "mode": mode,
                "diversify": diversify, "explain": explain}
        if text:
            body["text"] = text
        if ratings is not None:
            body["ratings"] = [float(r) for r in ratings]
        return self._call("POST", "/recommend/cold-start", json=body)

    def explore(self, text: str, k: int = 10, exclude=()) -> dict:
        return self.cold_start(exclude, text=text, k=k, mode="explore")

    def recommend_user(self, user_id: int, k: int = 10, diversify: bool = True,
                       explain: bool = True) -> dict:
        return self._call("GET", f"/recommend/{int(user_id)}",
                          params={"k": k, "diversify": diversify, "explain": explain})

    def demo_users(self, limit: int = 50) -> list[int]:
        return self._call("GET", "/users/demo", params={"limit": limit})["user_ids"]

    def user_history(self, user_id: int, limit: int = 20) -> dict:
        return self._call("GET", f"/users/{int(user_id)}/history", params={"limit": limit})

    def search(self, q: str, limit: int = 10) -> list[dict]:
        q = (q or "").strip()
        if not q:
            return []
        return self._call("GET", "/movies/search", params={"q": q[:100], "limit": limit})["items"]

    def movie(self, movie_id: int) -> dict:
        return self._call("GET", f"/movies/{int(movie_id)}")

    def similar(self, movie_id: int, k: int = 10) -> dict:
        return self._call("GET", f"/movies/{int(movie_id)}/similar", params={"k": k})

    # ---------------- in-process mode (free hosting without a second server) ----------------
    @classmethod
    def in_process(cls, app, label: str = "in-process") -> "ApiClient":
        """A client that sends every request to a FastAPI app object in this process (FastAPI's
        TestClient), instead of over the network. Same routes, validation, error codes and Engine
        as the HTTP server — only the transport differs."""
        from fastapi.testclient import TestClient

        obj = cls.__new__(cls)
        obj.base_url = label
        obj._http = TestClient(app, base_url="http://in-process", raise_server_exceptions=False)
        return obj


# --------------------------------------------------------------------------- #
# Day 13: where the UI gets its API from
# --------------------------------------------------------------------------- #
EMBEDDED = "embedded"


def choose_target(env_url: str | None, force_embedded: bool = False,
                  probe_url: str = DEFAULT_URL, probe_timeout: float = 1.5) -> str:
    """The API the UI should use:
    RECSYS_API_URL set      -> that server (Docker Space, CI, custom port)
    RECSYS_EMBEDDED=1       -> the API inside the UI process
    otherwise               -> a server on localhost:8000 if one answers (your two-terminal setup),
                               else the API inside the UI process (Streamlit Community Cloud)."""
    if env_url:
        return env_url.rstrip("/")
    if force_embedded:
        return EMBEDDED
    try:
        ApiClient(probe_url, timeout=probe_timeout).health()
        return probe_url
    except ApiError:
        return EMBEDDED


def embedded_client(bundle_dir: Path | str | None = None, bundle_repo: str | None = None,
                    preload_encoder: bool = True) -> ApiClient:
    """Load the serving bundle and wrap the FastAPI app in an in-process client.

    Bundle: bundle_dir / $RECSYS_BUNDLE if it holds one, else data/serving if it holds one,
    else download bundle_repo / $BUNDLE_REPO from Hugging Face into a temp folder (cached there
    across restarts of the same machine)."""
    import tempfile

    from . import config
    from .experiment import setup_runtime

    setup_runtime()
    candidates = [bundle_dir, os.environ.get("RECSYS_BUNDLE"), config.SERVING_DIR]
    found = next((Path(c) for c in candidates if c and (Path(c) / "bundle.json").exists()), None)
    if found is None:
        repo = bundle_repo or os.environ.get("BUNDLE_REPO")
        if not repo:
            raise ApiError("no serving bundle found and no BUNDLE_REPO to download one from")
        from .hub import download_bundle
        found = download_bundle(repo, Path(tempfile.gettempdir()) / "recsys-serving")

    from .api import create_app
    from .serving import Engine
    engine = Engine.load(found)
    if preload_encoder:
        try:
            engine.encoder()
        except RuntimeError:
            pass                                   # no sentence-transformers: text search -> 503
    return ApiClient.in_process(create_app(engine=engine), label=f"in-process ({found.name})")
