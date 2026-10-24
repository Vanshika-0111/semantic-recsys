import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.text import build_item_documents, genre_matrix, split_list  # noqa: E402


def test_split_list_formats():
    assert split_list("Action|Comedy") == ["Action", "Comedy"]
    assert split_list("Action, Science Fiction") == ["Action", "Science Fiction"]
    assert split_list("['time travel', 'robot']") == ["time travel", "robot"]
    assert split_list("[{'id': 1, 'name': 'heist'}]") == ["heist"]
    assert split_list("Drama, drama") == ["Drama"]            # case-insensitive de-dup
    for empty in (None, np.nan, "", "(no genres listed)"):
        assert split_list(empty) == []


ITEMS = pd.DataFrame({
    "movieId": [1, 2],
    "title_clean": ["The Matrix", "Toy Story"],
    "year": pd.array([1999, pd.NA], dtype="Int64"),
    "genres": ["Action|Sci-Fi", "Animation|Comedy"],
    "tmdb_genres": ["Action, Science Fiction", np.nan],       # movie 2 falls back to ML genres
    "keywords": ["simulated reality, hacker", np.nan],
    "tagline": ["Free your mind.", np.nan],
    "overview": ["A hacker learns the truth.", "Toys come alive."],
})


def test_build_item_documents():
    docs = build_item_documents(ITEMS)
    assert list(docs.index) == [1, 2]
    assert docs[1] == ("The Matrix (1999). Genres: Action, Science Fiction. "
                       "Keywords: simulated reality, hacker. Free your mind. A hacker learns the truth.")
    assert docs[2] == "Toy Story. Genres: Animation, Comedy. Toys come alive."


def test_genre_matrix():
    mat, vocab = genre_matrix(ITEMS)
    assert vocab == ["Action", "Animation", "Comedy", "Sci-Fi"]
    assert mat.tolist() == [[1, 0, 0, 1], [0, 1, 1, 0]]
