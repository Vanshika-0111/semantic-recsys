import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.data import clean_title, join_items, kcore_filter  # noqa: E402


def test_clean_title():
    assert clean_title("Matrix, The (1999)") == ("The Matrix", 1999.0)
    assert clean_title("Toy Story (1995)") == ("Toy Story", 1995.0)
    t, y = clean_title("Untitled")
    assert t == "Untitled" and np.isnan(y)


def test_kcore_cascades():
    # item 3 has 1 rating -> removed -> user 2 drops to 1 rating -> removed
    # -> item 2 then has only user 1 -> removed -> user 1 has 1 -> everything goes.
    r = pd.DataFrame({"userId": [1, 1, 2, 2], "movieId": [1, 2, 2, 3]})
    out, _ = kcore_filter(r, min_user=2, min_item=2)
    assert out.empty


def test_kcore_keeps_dense_block():
    users, items = np.meshgrid([1, 2, 3], [10, 20, 30])
    dense = pd.DataFrame({"userId": users.ravel(), "movieId": items.ravel()})
    noise = pd.DataFrame({"userId": [4], "movieId": [99]})
    out, _ = kcore_filter(pd.concat([dense, noise]), min_user=3, min_item=3)
    assert len(out) == 9 and set(out["userId"]) == {1, 2, 3}
    assert out["userId"].map(out["userId"].value_counts()).min() >= 3
    assert out["movieId"].map(out["movieId"].value_counts()).min() >= 3


def test_join_falls_back_to_imdb():
    movies = pd.DataFrame({"movieId": [1, 2], "title": ["A (2000)", "B (2001)"],
                           "genres": ["Drama", "Comedy"]})
    links = pd.DataFrame({"movieId": [1, 2], "imdbId": ["0000001", "0000002"],
                          "tmdbId": pd.array([100, pd.NA], dtype="Int64")})
    tmdb = pd.DataFrame({"id": pd.array([100, 200], dtype="Int64"),
                         "imdb_id": ["tt0000001", "tt0000002"],
                         "title": ["A", "B"], "overview": ["plot a", "plot b"]})
    out = join_items(movies, links, tmdb).set_index("movieId")
    assert out.loc[1, "match_source"] == "tmdbId" and out.loc[1, "overview"] == "plot a"
    assert out.loc[2, "match_source"] == "imdbId" and out.loc[2, "tmdbId"] == 200
    assert out["has_overview"].all()
