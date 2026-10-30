"""Day 13 tests: API client + the Streamlit UI, end to end.

A REAL uvicorn server runs in a background thread on a free local port, serving the synthetic
bundle (recsys.synthetic, real LightGBM rankers). The UI script app/streamlit_app.py is then run
with Streamlit's AppTest (headless, no browser) and clicked through: popular grid -> add picks ->
the recommendations shown must be exactly the API's for those picks (which Day 12 checked equal to
the evaluated batch path). So UI -> HTTP -> Engine -> ranker is covered in one chain.
"""
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recsys.client import ApiClient, ApiError  # noqa: E402
from recsys.serving import Engine  # noqa: E402
from recsys.synthetic import FakeEncoder, build_synthetic_bundle  # noqa: E402

APP = str(ROOT / "app" / "streamlit_app.py")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(engine):
    """Start uvicorn with this engine in a daemon thread -> (base_url, server)."""
    import uvicorn

    from recsys.api import create_app
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(engine=engine), host="127.0.0.1", port=port,
                                           log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            ApiClient(url, timeout=1).health()
            return url, server
        except ApiError:
            time.sleep(0.05)
    raise RuntimeError("test API did not start")


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ui")
    build_synthetic_bundle(tmp / "with_demo", work_dir=tmp / "m1")
    build_synthetic_bundle(tmp / "public", work_dir=tmp / "m2", demo_users=False)
    urls, servers = {}, []
    for name in ("with_demo", "public"):
        url, srv = serve(Engine.load(tmp / name, encoder=FakeEncoder()))
        urls[name] = url
        servers.append(srv)
    yield urls
    for s in servers:
        s.should_exit = True


@pytest.fixture
def app_test(monkeypatch):
    from streamlit.testing.v1 import AppTest

    def make(url: str):
        monkeypatch.setenv("RECSYS_API_URL", url)
        at = AppTest.from_file(APP, default_timeout=30)
        at.run()
        assert not at.exception, at.exception
        return at
    return make


def markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
def test_client_calls_and_errors(api):
    c = ApiClient(api["with_demo"])
    assert c.health()["items"] == 80
    pop = c.popular(5)
    assert pop["model"] == "popularity" and len(pop["items"]) == 5 and pop["_http_ms"] >= 0
    assert c.cold_start([10, 20, 30], k=4)["model"] == "warm ranker"
    assert c.explore("space opera", k=3)["model"].startswith("explore")
    assert [m["title"] for m in c.search("movie 10", 3)][0] == "Movie 10" and c.search("  ") == []
    assert len(c.similar(10, 4)["items"]) == 4 and c.movie(10)["title"] == "Movie 10"
    assert c.demo_users(5) == [1, 2, 3, 4, 5]
    with pytest.raises(ApiError) as e:
        c.cold_start([10, 777])
    assert e.value.status == 422 and "777" in str(e.value)
    with pytest.raises(ApiError) as e:
        c.recommend_user(99999)
    assert e.value.status == 404


def test_client_unreachable_api():
    with pytest.raises(ApiError, match="not reachable") as e:
        ApiClient(f"http://127.0.0.1:{free_port()}", timeout=1).health()
    assert e.value.status is None


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #
def test_ui_starts_with_popular_movies(api, app_test):
    at = app_test(api["with_demo"])
    pop = ApiClient(api["with_demo"]).popular(10)["items"]
    md = markdown(at)
    assert "Popular right now" in [s.value for s in at.subheader]
    assert all(m["title"] in md for m in pop)                 # the 10 popular cards are shown
    assert [t.label for t in at.tabs][-1] == "👤 Demo users"   # bundle has demo users


def test_ui_picks_drive_the_policy_and_match_the_api(api, app_test):
    at = app_test(api["with_demo"])
    c = ApiClient(api["with_demo"])
    pop = c.popular(20)["items"]
    first, second, third = (m["movieId"] for m in pop[:3])

    at.button(key=f"add_p_{first}").click().run()
    assert not at.exception and "Your picks (1)" in [s.value for s in at.subheader]
    assert "cold ranker" in "\n".join(x.value for x in at.caption)

    # the popular grid hides after the first pick -> the next picks come from title search
    for m in (pop[1], pop[2]):
        at.text_input(key="pick_q").input(m["title"]).run()
        at.button(key=f"add_s_{m['movieId']}").click().run()
        assert not at.exception
    assert not at.exception and "Your picks (3)" in [s.value for s in at.subheader]
    caps = "\n".join(x.value for x in at.caption)
    assert "warm ranker" in caps

    # exactly the API's list for these picks, in this order (= time order for the ranker)
    ref = c.cold_start([first, second, third], k=10, diversify=True)["items"]
    md = markdown(at)
    pos = [md.find(f'class="t">{m["title"]}') for m in ref]   # each card's title, in rank order
    assert all(p >= 0 for p in pos) and pos == sorted(pos)
    assert not {first, second, third} & {m["movieId"] for m in ref}

    at.button(key=f"rm_{first}").click().run()
    assert "Your picks (2)" in [s.value for s in at.subheader]
    at.button(key="clear").click().run()
    assert "Popular right now" in [s.value for s in at.subheader]


def test_ui_free_text_and_similar(api, app_test):
    at = app_test(api["with_demo"])
    at.text_input(key="query").input("space opera").run()
    assert not at.exception
    ref = ApiClient(api["with_demo"]).explore("space opera", 10)["items"]
    assert all(m["title"] in markdown(at) for m in ref)
    at.text_input(key="sim_q").input("Movie 10").run()
    assert not at.exception and at.selectbox(key="sim_pick").value == 10


def test_ui_demo_user_marks_later_likes(api, app_test):
    at = app_test(api["with_demo"])
    at.selectbox(key="demo_uid").set_value(1).run()
    assert not at.exception
    assert any("ratings," in m.value and "liked later" in m.value for m in at.markdown)


def test_ui_public_bundle_has_no_demo_tab(api, app_test):
    at = app_test(api["public"])
    assert "👤 Demo users" not in [t.label for t in at.tabs]


def test_ui_shows_a_clear_error_without_api(app_test):
    at = app_test(f"http://127.0.0.1:{free_port()}")
    assert any("isn't reachable" in e.value for e in at.error)


# --------------------------------------------------------------------------- #
# In-process ("embedded") mode — used on Streamlit Community Cloud
# --------------------------------------------------------------------------- #
from recsys.client import EMBEDDED, choose_target, embedded_client  # noqa: E402


def test_choose_target(api):
    dead = f"http://127.0.0.1:{free_port()}"
    assert choose_target("http://x:1/") == "http://x:1"                       # explicit URL wins
    assert choose_target("http://x:1", force_embedded=True) == "http://x:1"
    assert choose_target(None, force_embedded=True) == EMBEDDED
    assert choose_target(None, probe_url=api["public"]) == api["public"]       # a server answers
    assert choose_target(None, probe_url=dead, probe_timeout=0.5) == EMBEDDED   # nothing there


def test_embedded_client_is_the_same_api(api, tmp_path):
    build_synthetic_bundle(tmp_path / "b", work_dir=tmp_path / "m", demo_users=False)
    emb = embedded_client(tmp_path / "b", preload_encoder=False)
    http = ApiClient(api["public"])                       # same synthetic world, served over HTTP
    assert emb.health()["items"] == 80 and emb.base_url.startswith("in-process")
    for picks in ([], [10], [10, 20, 30]):
        a, b = emb.cold_start(picks, k=6), http.cold_start(picks, k=6)
        assert a["model"] == b["model"] and a["items"] == b["items"]
    with pytest.raises(ApiError) as e:                    # same validation + status codes
        emb.cold_start([10, 777])
    assert e.value.status == 422
    with pytest.raises(ApiError) as e:
        emb.movie(123456)
    assert e.value.status == 404
    with pytest.raises(ApiError) as e:                    # no encoder in the synthetic bundle
        emb.explore("space opera")
    assert e.value.status == 503


def test_embedded_client_downloads_when_no_local_bundle(tmp_path, monkeypatch):
    from recsys import config, hub
    build_synthetic_bundle(tmp_path / "remote", work_dir=tmp_path / "m", demo_users=False)
    calls = []

    def fake_download(repo, out, revision=None, token=None):
        calls.append(repo)
        shutil.copytree(tmp_path / "remote", out, dirs_exist_ok=True)
        return Path(out)

    monkeypatch.setattr(hub, "download_bundle", fake_download)
    monkeypatch.setattr(config, "SERVING_DIR", tmp_path / "no-local-bundle")
    monkeypatch.delenv("RECSYS_BUNDLE", raising=False)
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path / "tmp"))
    emb = embedded_client(bundle_repo="alice/semantic-recsys-bundle", preload_encoder=False)
    assert calls == ["alice/semantic-recsys-bundle"] and emb.health()["items"] == 80
    # no local bundle and no repo to download from -> a clear error, not a crash
    monkeypatch.delenv("BUNDLE_REPO", raising=False)
    with pytest.raises(ApiError, match="no serving bundle"):
        embedded_client(tmp_path / "nothing-here", preload_encoder=False)


def test_ui_runs_in_process_without_an_api_server(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    build_synthetic_bundle(tmp_path / "b", work_dir=tmp_path / "m", demo_users=False)
    monkeypatch.delenv("RECSYS_API_URL", raising=False)
    monkeypatch.setenv("RECSYS_EMBEDDED", "1")
    monkeypatch.setenv("RECSYS_BUNDLE", str(tmp_path / "b"))
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, at.exception
    assert any("in-process" in c.value for c in at.caption)
    assert "Popular right now" in [s.value for s in at.subheader]
    pop = ApiClient.in_process(__import__("recsys.api", fromlist=["x"]).create_app(
        engine=Engine.load(tmp_path / "b", encoder=FakeEncoder()))).popular(20)["items"]
    at.button(key=f"add_p_{pop[0]['movieId']}").click().run()
    assert not at.exception and "cold ranker" in "\n".join(c.value for c in at.caption)
