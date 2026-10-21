"""Loading, joining and filtering MovieLens 32M + TMDB metadata.

Design rules (from the project plan):
- Item metadata only (no rating statistics) goes into items.parquet, so later
  features cannot accidentally use full-period popularity / avg rating.
- Ratings are filtered with an iterative k-core so that every remaining user
  AND item has >= k ratings after filtering.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Columns we want from the Kaggle TMDB dump (asaniczka "Full TMDB Movies Dataset").
# Missing ones are skipped, so a different TMDB dump with fewer columns still works.
TMDB_COLS = [
    "id", "imdb_id", "title", "overview", "tagline", "genres", "keywords",
    "poster_path", "original_language", "release_date", "runtime", "adult",
    "vote_count",
]

_YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
_ARTICLE_RE = re.compile(r"^(.*), (The|A|An|Les|La|Le|L'|Il|Das|Der|Die|El|Los|Las)$")


# --------------------------------------------------------------------------- #
# MovieLens
# --------------------------------------------------------------------------- #
def load_ratings(ml_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(ml_dir / "ratings.csv", engine="pyarrow")
    return df.astype({"userId": "int32", "movieId": "int32",
                      "rating": "float32", "timestamp": "int64"})


def clean_title(raw: str) -> tuple[str, float]:
    """'Matrix, The (1999)' -> ('The Matrix', 1999.0)."""
    raw = (raw or "").strip()
    m = _YEAR_RE.search(raw)
    year = float(m.group(1)) if m else np.nan
    title = _YEAR_RE.sub("", raw).strip()
    a = _ARTICLE_RE.match(title)
    if a:
        sep = "" if a.group(2).endswith("'") else " "
        title = f"{a.group(2)}{sep}{a.group(1)}"
    return title, year


def load_movies(ml_dir: Path) -> pd.DataFrame:
    movies = pd.read_csv(ml_dir / "movies.csv", dtype={"movieId": "int32"})
    parsed = movies["title"].map(clean_title)
    movies["title_clean"] = [p[0] for p in parsed]
    movies["year"] = pd.array([p[1] for p in parsed], dtype="Int64")
    movies["genres"] = movies["genres"].replace("(no genres listed)", "")
    return movies


def load_links(ml_dir: Path) -> pd.DataFrame:
    links = pd.read_csv(ml_dir / "links.csv", dtype={"movieId": "int32", "imdbId": str})
    links["tmdbId"] = pd.to_numeric(links["tmdbId"], errors="coerce").astype("Int64")
    return links


# --------------------------------------------------------------------------- #
# TMDB
# --------------------------------------------------------------------------- #
def find_tmdb_csv(tmdb_dir: Path) -> Path:
    csvs = sorted(tmdb_dir.glob("*.csv"), key=lambda p: p.stat().st_size, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {tmdb_dir}")
    return csvs[0]  # the main movie file is by far the largest


def load_tmdb(tmdb_dir: Path) -> pd.DataFrame:
    path = find_tmdb_csv(tmdb_dir)
    header = pd.read_csv(path, nrows=0).columns
    usecols = [c for c in TMDB_COLS if c in header]
    if "id" not in usecols:
        raise ValueError(f"{path.name} has no 'id' column; columns are {list(header)}")
    df = pd.read_csv(path, usecols=usecols, dtype=str, keep_default_na=True)
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["id"])
    if "vote_count" in df:  # keep the most-voted row when an id is duplicated
        df["vote_count"] = pd.to_numeric(df["vote_count"], errors="coerce")
        df = df.sort_values("vote_count", ascending=False, na_position="last")
    df = df.drop_duplicates("id").drop(columns=["vote_count"], errors="ignore")
    for col in ("overview", "tagline", "keywords", "genres", "poster_path"):
        if col in df:
            df[col] = df[col].str.strip().replace("", np.nan)
    if "runtime" in df:
        df["runtime"] = pd.to_numeric(df["runtime"], errors="coerce")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Join
# --------------------------------------------------------------------------- #
def join_items(movies: pd.DataFrame, links: pd.DataFrame, tmdb: pd.DataFrame) -> pd.DataFrame:
    """Join ML movies to TMDB metadata: first on tmdbId, then fall back to imdb id."""
    items = movies.merge(links, on="movieId", how="left")
    t = tmdb.rename(columns={"id": "tmdbId_meta", "title": "tmdb_title", "genres": "tmdb_genres"})
    meta_cols = [c for c in t.columns if c not in ("tmdbId_meta", "imdb_id")]

    # 1) primary join on tmdbId
    primary = items.merge(t.drop(columns=["imdb_id"], errors="ignore"),
                          left_on="tmdbId", right_on="tmdbId_meta", how="left")
    primary["match_source"] = np.where(primary["tmdbId_meta"].notna(), "tmdbId", "none")

    # 2) fallback on IMDb id for rows the tmdbId join missed
    miss = primary["match_source"].eq("none") & primary["imdbId"].notna()
    if "imdb_id" in t and miss.any():
        t_imdb = t.dropna(subset=["imdb_id"]).drop_duplicates("imdb_id")
        keys = pd.DataFrame({
            "row": primary.index[miss],
            "imdb_id": "tt" + primary.loc[miss, "imdbId"].astype(str).str.zfill(7),
        })
        fb = keys.merge(t_imdb, on="imdb_id", how="inner").set_index("row")
        if len(fb):
            primary.loc[fb.index, meta_cols] = fb[meta_cols].values
            primary.loc[fb.index, "match_source"] = "imdbId"
            primary.loc[fb.index, "tmdbId"] = fb["tmdbId_meta"].astype("Int64").values

    primary = primary.drop(columns=["tmdbId_meta"])
    primary["has_overview"] = primary.get("overview", pd.Series(dtype=str)).notna()
    return primary


# --------------------------------------------------------------------------- #
# Filtering
# --------------------------------------------------------------------------- #
def kcore_filter(ratings: pd.DataFrame, min_user: int = 20, min_item: int = 20,
                 max_iter: int = 50) -> tuple[pd.DataFrame, int]:
    """Iteratively drop users/items below the thresholds until stable."""
    df = ratings
    for it in range(1, max_iter + 1):
        n_before = len(df)
        ic = df["movieId"].map(df["movieId"].value_counts())
        df = df[ic >= min_item]
        uc = df["userId"].map(df["userId"].value_counts())
        df = df[uc >= min_user]
        if len(df) == n_before:
            return df.reset_index(drop=True), it
    raise RuntimeError("k-core filtering did not converge")
