"""Day 13: Streamlit demo UI for the Semantic Recommendation & Ranking Engine.

    uvicorn recsys.api:app --port 8000                 (terminal 1: the API, Day 12)
    streamlit run app/streamlit_app.py                 (terminal 2: this UI -> http://localhost:8501)

The UI only talks HTTP to the API (recsys.client.ApiClient) — no model code runs here. Set
RECSYS_API_URL if the API is not on http://localhost:8000.

Tabs
  Pick movies     onboarding: search / popular grid -> picks -> live recommendations
                  (0 picks popularity · 1–2 cold-start ranker · 3+ full two-stage ranker)
  Describe it     free text -> semantic search over well-known movies (explore mode)
  Demo users      only when the bundle has demo users (laptop): their likes, recs, hits
  Similar movies  nearest movies in the embedding space
"""
from __future__ import annotations

import html
import inspect
import os

import streamlit as st

from recsys.client import ApiClient, ApiError

REPO_URL = os.environ.get("RECSYS_REPO_URL", "https://github.com/Vanshika-0111/semantic-recsys")
MAX_PICKS = 20

# Headline test numbers (Day 10/11, reports/results/summary.md) — shown in the sidebar
RESULTS = [
    ("+73%", "NDCG@10 vs popularity", "189k warm test users, 0.050 → 0.087"),
    ("+24%", "from semantic features", "ablation: same ranker without embeddings"),
    ("+27%", "for brand-new users", "5 onboarding picks, 5,000 users never in training"),
    ("+13%", "more genres per top-10", "MMR re-ranking, −1.4% NDCG"),
]

# One-click sample tastes for visitors who don't want to search. Resolved by title search,
# so they work on any bundle that contains these movies (and quietly disappear otherwise).
SAMPLE_TASTES = {
    "Animation night": ["Toy Story", "Finding Nemo", "Up", "Spirited Away"],
    "Mind-bending sci-fi": ["The Matrix", "Inception", "Interstellar", "Blade Runner"],
    "Crime classics": ["The Godfather", "Pulp Fiction", "Goodfellas", "Heat"],
    "Horror": ["The Shining", "Alien", "The Exorcist", "Hereditary"],
}
EXAMPLE_QUERIES = [
    "a mind-bending science fiction film about dreams and memory",
    "heartwarming animated movie about friendship",
    "gritty crime thriller where a heist goes wrong",
    "epic war movie based on a true story",
    "funny superhero movie",
]
MODEL_NOTES = {
    "popularity": "Most-liked movies — pick a movie to personalise.",
    "cold ranker": "Cold-start ranker (trained for users with only 1–2 picks).",
    "warm ranker": "Full two-stage ranker: popular ∪ FAISS candidates → LightGBM LambdaMART.",
}

# full-width buttons: `width="stretch"` in new Streamlit, `use_container_width` in older ones
WIDE = ({"width": "stretch"} if "width" in inspect.signature(st.button).parameters
        else {"use_container_width": True})

st.set_page_config(page_title="Semantic Movie Recommender", page_icon="🎬", layout="wide")

st.markdown("""
<style>
.rc-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:14px;margin:6px 0 18px}
.rc{border:1px solid rgba(128,128,128,.25);border-radius:10px;overflow:hidden;background:rgba(128,128,128,.04);position:relative}
.rc img,.rc .np{width:100%;aspect-ratio:2/3;object-fit:cover;display:block;background:rgba(128,128,128,.15)}
.rc .np{display:flex;align-items:center;justify-content:center;font-size:2.2rem}
.rc .b{padding:7px 9px 9px}
.rc .t{font-weight:600;font-size:.88rem;line-height:1.2}
.rc .t span{font-weight:400;opacity:.65}
.rc .g{font-size:.72rem;opacity:.65;margin-top:2px}
.rc .why{font-size:.76rem;margin-top:6px;line-height:1.25}
.rc .by{font-size:.72rem;margin-top:3px;opacity:.75;line-height:1.2}
.rc .n{position:absolute;top:6px;left:6px;background:rgba(0,0,0,.7);color:#fff;font-size:.72rem;
       font-weight:600;border-radius:6px;padding:1px 6px}
.rc .hit{position:absolute;top:6px;right:6px;background:#1a7f37;color:#fff;font-size:.68rem;
         border-radius:6px;padding:1px 6px}
.pick-poster{width:100%;aspect-ratio:2/3;object-fit:cover;border-radius:8px;display:block;
             background:rgba(128,128,128,.15)}
.stat{font-size:1.35rem;font-weight:700;line-height:1.1}
.stat-l{font-size:.8rem;opacity:.85}
.stat-s{font-size:.7rem;opacity:.6;margin-bottom:8px}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# API access (cached; errors are not cached)
# --------------------------------------------------------------------------- #
API_URL = os.environ.get("RECSYS_API_URL", "http://localhost:8000")


# every cached call is keyed by the API URL, so a cache never answers for another server
@st.cache_resource
def _client(url: str) -> ApiClient:
    return ApiClient(url)


def client() -> ApiClient:
    return _client(API_URL)


@st.cache_data(ttl=60, show_spinner=False)
def _health(url: str) -> dict:
    return _client(url).health()


@st.cache_data(ttl=3600, show_spinner=False)
def _popular(url: str, k: int) -> list[dict]:
    return _client(url).popular(k)["items"]


@st.cache_data(ttl=3600, show_spinner=False)
def _search(url: str, q: str, limit: int) -> list[dict]:
    return _client(url).search(q, limit)


@st.cache_data(ttl=600, show_spinner=False)
def _recommend_picks(url: str, picks: tuple[int, ...], k: int, diversify: bool) -> dict:
    return _client(url).cold_start(list(picks), k=k, diversify=diversify)


@st.cache_data(ttl=600, show_spinner=False)
def _explore(url: str, text: str, k: int) -> dict:
    return _client(url).explore(text, k=k)


@st.cache_data(ttl=600, show_spinner=False)
def _similar(url: str, movie_id: int, k: int) -> dict:
    return _client(url).similar(movie_id, k=k)


@st.cache_data(ttl=600, show_spinner=False)
def _demo_user(url: str, user_id: int, k: int, diversify: bool) -> tuple[dict, dict]:
    c = _client(url)
    return c.user_history(user_id, limit=12), c.recommend_user(user_id, k=k, diversify=diversify)


@st.cache_data(ttl=3600, show_spinner=False)
def _demo_users(url: str) -> list[int]:
    return _client(url).demo_users(limit=50)


def health() -> dict:
    return _health(API_URL)


def popular(k: int) -> list[dict]:
    return _popular(API_URL, k)


def search(q: str, limit: int = 8) -> list[dict]:
    return _search(API_URL, q, limit)


def recommend_picks(picks: tuple[int, ...], k: int, diversify: bool) -> dict:
    return _recommend_picks(API_URL, picks, k, diversify)


def explore(text: str, k: int) -> dict:
    return _explore(API_URL, text, k)


def similar(movie_id: int, k: int) -> dict:
    return _similar(API_URL, movie_id, k)


def demo_user(user_id: int, k: int, diversify: bool) -> tuple[dict, dict]:
    return _demo_user(API_URL, user_id, k, diversify)


def demo_users() -> list[int]:
    return _demo_users(API_URL)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def esc(x) -> str:
    return html.escape(str(x)) if x is not None else ""


def poster_html(m: dict, cls: str = "") -> str:
    if m.get("poster_url"):
        return f'<img class="{cls}" src="{esc(m["poster_url"])}" loading="lazy" alt="{esc(m["title"])}">'
    return f'<div class="np {cls}">🎬</div>'


def card(m: dict, rank: int | None = None, hit: bool = False) -> str:
    year = f' <span>({m["year"]})</span>' if m.get("year") else ""
    genres = " · ".join(m.get("genres") or [])
    why = m.get("explanation") or ""
    by = ""
    b = m.get("because")
    if b and not why.startswith("Because you liked"):
        by = f'<div class="by">↳ Because you liked {esc(b["title"])}</div>'
    n = f'<div class="n">{rank}</div>' if rank else ""
    h = '<div class="hit">✓ liked it later</div>' if hit else ""
    return (f'<div class="rc">{n}{h}{poster_html(m)}<div class="b"><div class="t">{esc(m["title"])}{year}</div>'
            f'<div class="g">{esc(genres)}</div><div class="why">{esc(why)}</div>{by}</div></div>')


def grid(items: list[dict], hits: set[int] = frozenset(), numbered: bool = True) -> None:
    cards = "".join(card(m, i + 1 if numbered else None, m["movieId"] in hits) for i, m in enumerate(items))
    st.markdown(f'<div class="rc-grid">{cards}</div>', unsafe_allow_html=True)


def caption(out: dict) -> None:
    bits = [f"model: **{out.get('model', '?')}**"]
    if out.get("mmr"):
        bits.append(f"diversity: MMR ({out['mmr'].get('space')}, λ {out['mmr'].get('lambda')})")
    if out.get("latency_ms") is not None:
        bits.append(f"answered in {out['latency_ms']:.0f} ms")
    st.caption(" · ".join(bits))
    if out.get("note"):
        st.caption(out["note"])


def show_error(e: ApiError) -> None:
    if e.status == 503:
        st.info("Free-text search isn't available in this deployment (no text encoder).")
    else:
        st.error(f"API error: {e}")


# --------------------------------------------------------------------------- #
# Picks (session state)
# --------------------------------------------------------------------------- #
if "picks" not in st.session_state:
    st.session_state.picks = []          # movie dicts, in pick order (= time order for the ranker)


def pick_ids() -> list[int]:
    return [m["movieId"] for m in st.session_state.picks]


def add_pick(m: dict) -> None:
    if m["movieId"] not in pick_ids() and len(st.session_state.picks) < MAX_PICKS:
        st.session_state.picks.append(m)


def remove_pick(movie_id: int) -> None:
    st.session_state.picks = [m for m in st.session_state.picks if m["movieId"] != movie_id]


def clear_picks() -> None:
    st.session_state.picks = []


def use_sample(titles: tuple[str, ...]) -> None:
    clear_picks()
    for t in titles:
        hits = search(t, 3)
        exact = [h for h in hits if h["title"].lower() == t.lower()]
        if hits:
            add_pick((exact or hits)[0])


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("🎬 Semantic Movie Recommender")
st.markdown("Two-stage recommender on **MovieLens 32M + TMDB**: sentence embeddings (bge-small) + "
            "FAISS retrieval → LightGBM LambdaMART ranking → MMR diversity, with "
            "*“Because you liked X”* explanations.")

try:
    info = health()
except ApiError as e:
    st.error(f"The recommendation API isn't reachable yet: {e}. If the demo has just started, "
             "wait a few seconds and reload.")
    st.stop()

with st.sidebar:
    st.subheader("Results (held-out test)")
    for big, label, sub in RESULTS:
        st.markdown(f'<div class="stat">{big}</div><div class="stat-l">{label}</div>'
                    f'<div class="stat-s">{sub}</div>', unsafe_allow_html=True)
    st.subheader("How it works")
    st.markdown(
        "1. **Candidates** — 400 most-liked movies ∪ 100 nearest in embedding space (FAISS)\n"
        "2. **Features** — 21 leak-free signals: popularity, era, genres, 3 semantic similarities\n"
        "3. **Ranker** — LightGBM LambdaMART (separate cold-start ranker for 1–2 picks)\n"
        "4. **Diversity** — MMR re-ranking over genres\n"
        "5. **Why** — TreeSHAP reason + the liked movie closest to each pick")
    st.caption(f"API: {info.get('items', '?'):,} movies · ranker {info.get('warm_ranker_trees', '?')} trees · "
               f"code {info.get('git_commit') or 'dev'} · [GitHub]({REPO_URL})")

has_demo = int(info.get("demo_users", 0)) > 0
names = ["🎯 Pick movies", "💬 Describe it", "🔁 Similar movies"] + (["👤 Demo users"] if has_demo else [])
tabs = st.tabs(names)

# ---------------- Tab 1: onboarding ----------------
with tabs[0]:
    picks = st.session_state.picks
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader(f"Your picks ({len(picks)})")
        if not picks:
            st.caption("Add a few movies you like. 1–2 picks use the cold-start ranker; "
                       "from 3 picks the full ranker takes over.")
        for m in picks:
            c1, c2 = st.columns([5, 1])
            c1.markdown(f"**{esc(m['title'])}** {('(' + str(m['year']) + ')') if m.get('year') else ''}")
            c2.button("✕", key=f"rm_{m['movieId']}", on_click=remove_pick, args=(m["movieId"],),
                      help="remove")
        if picks:
            st.button("Clear all", on_click=clear_picks, key="clear")

        st.markdown("**Search a title**")
        q = st.text_input("Search a title", key="pick_q", placeholder="e.g. Amélie",
                          label_visibility="collapsed")
        if q.strip():
            try:
                found = search(q.strip())
            except ApiError as e:
                found = []
                show_error(e)
            if not found:
                st.caption("No match.")
            for m in found:
                c1, c2 = st.columns([5, 1])
                c1.markdown(f"{esc(m['title'])} {('(' + str(m['year']) + ')') if m.get('year') else ''}")
                c2.button("＋", key=f"add_s_{m['movieId']}", on_click=add_pick, args=(m,),
                          disabled=m["movieId"] in pick_ids())

        st.markdown("**Or try a sample taste**")
        sc = st.columns(2)
        for i, (name, titles) in enumerate(SAMPLE_TASTES.items()):
            sc[i % 2].button(name, key=f"sample_{i}", on_click=use_sample, args=(tuple(titles),),
                             **WIDE)

    with right:
        c1, c2 = st.columns(2)
        k = c1.slider("How many recommendations", 5, 20, 10, key="k_picks")
        diversify = c2.toggle("Diversify (MMR)", value=True, key="div_picks",
                              help="Re-rank for more genre variety (−1.4% NDCG, +13% genres on test)")
        try:
            out = recommend_picks(tuple(pick_ids()), k, diversify)
        except ApiError as e:
            show_error(e)
        else:
            st.subheader("Recommended for you" if picks else "Popular right now")
            note = MODEL_NOTES.get(out.get("model", ""))
            if note:
                st.caption(note)
            grid(out["items"])
            caption(out)

    if not picks:
        st.markdown("---")
        st.subheader("Popular movies — click to add")
        try:
            pop = popular(20)
        except ApiError as e:
            pop = []
            show_error(e)
        cols = st.columns(5)
        for i, m in enumerate(pop):
            with cols[i % 5]:
                st.markdown(poster_html(m, "pick-poster"), unsafe_allow_html=True)
                st.button(f"＋ {m['title']}"[:40], key=f"add_p_{m['movieId']}", on_click=add_pick,
                          args=(m,), **WIDE)

# ---------------- Tab 2: free text ----------------
with tabs[1]:
    st.markdown("Describe a movie night in your own words — the query is embedded with the same "
                "encoder as the movies and matched against well-known films (*explore mode*).")
    if "query" not in st.session_state:
        st.session_state.query = ""
    ex = st.columns(len(EXAMPLE_QUERIES))
    for i, eq in enumerate(EXAMPLE_QUERIES):
        if ex[i].button(eq, key=f"ex_{i}", **WIDE):
            st.session_state.query = eq
    text = st.text_input("What are you in the mood for?", key="query", max_chars=300)
    if text.strip():
        try:
            out = explore(text.strip(), 10)
        except ApiError as e:
            show_error(e)
        else:
            grid(out["items"])
            caption(out)

# ---------------- Tab 3: similar movies ----------------
with tabs[2]:
    st.markdown("Nearest neighbours in the (mean-centered) embedding space — pure content similarity.")
    q = st.text_input("Movie", key="sim_q", placeholder="e.g. Alien")
    if q.strip():
        try:
            found = search(q.strip(), 10)
        except ApiError as e:
            found = []
            show_error(e)
        if not found:
            st.caption("No match.")
        else:
            label = {m["movieId"]: f"{m['title']} ({m.get('year') or '?'})" for m in found}
            mid = st.selectbox("Pick one", list(label), format_func=label.get, key="sim_pick")
            try:
                out = similar(int(mid), 10)
            except ApiError as e:
                show_error(e)
            else:
                grid(out["items"])
                st.caption(f"answered in {out['latency_ms']:.0f} ms")

# ---------------- Tab 4: demo users (laptop bundle only) ----------------
if has_demo:
    with tabs[3]:
        st.markdown("Real MovieLens users from the **test** split: the model sees their history up to "
                    "the split, and ✓ marks a recommendation they actually rated 4★+ afterwards.")
        try:
            users = demo_users()
        except ApiError as e:
            users = []
            show_error(e)
        if users:
            c1, c2, c3 = st.columns([2, 1, 1])
            uid = c1.selectbox("Demo user", users, key="demo_uid")
            k_d = c2.slider("Recommendations", 5, 20, 10, key="k_demo")
            div_d = c3.toggle("Diversify (MMR)", value=True, key="div_demo")
            try:
                hist, out = demo_user(int(uid), k_d, div_d)
            except ApiError as e:
                show_error(e)
            else:
                hits = {m["movieId"] for m in hist["next_liked"]}
                n_hit = sum(m["movieId"] in hits for m in out["items"])
                st.markdown(f"**{hist['n_ratings']} ratings, {hist['n_liked']} liked** · "
                            f"{n_hit} of the top-{k_d} were liked later")
                st.subheader("Recommendations")
                grid(out["items"], hits)
                caption(out)
                st.subheader("Their most recent likes")
                grid([{**m, "explanation": f"rated {m['rating']}★"} for m in hist["recent_liked"]],
                     numbered=False)

st.markdown("---")
st.caption("Ratings: MovieLens 32M (GroupLens Research, University of Minnesota) — used for a "
           "non-commercial demo, no endorsement implied; no user ratings are shown in the public demo. "
           "Movie metadata and posters: TMDB (via a public Kaggle dump). This product uses TMDB "
           "images but is not endorsed or certified by TMDB.")
