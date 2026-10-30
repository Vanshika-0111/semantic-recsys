"""Day 13 tests: API client + the Streamlit UI, end to end.

A REAL uvicorn server runs in a background thread on a free local port, serving the synthetic
bundle (recsys.synthetic, real LightGBM rankers). The UI script app/streamlit_app.py is then run
with Streamlit's AppTest (headless, no browser) and clicked through: popular grid -> add picks ->
the recommendations shown must be exactly the API's for those picks (which Day 12 checked equal to
the evaluated batch path). So UI -> HTTP -> Engine -> ranker is covered in one chain.
"""
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
