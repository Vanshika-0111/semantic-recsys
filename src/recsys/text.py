"""Item text documents — ONE definition shared by every content model.

Day 3 (TF-IDF) and Day 4 (sentence embeddings) both read `build_item_documents`,
so the comparison "lexical vs semantic" uses exactly the same text. Only the
representation changes, which is what makes that comparison fair.

Document layout (EDA: overviews are short, median 40 words, so nothing is cut):
    "<Title> (<year>). Genres: a, b. Keywords: x, y. <tagline> <overview>"
"""
from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd

_SPLIT_RE = re.compile(r"\s*[|,;]\s*")


def split_list(value) -> list[str]:
    """Parse a list-like metadata cell into clean strings.

    Handles the formats seen across MovieLens / TMDB dumps:
      'Action|Comedy'   'Action, Comedy'   "['Action', 'Comedy']"   NaN / ''
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, (list, tuple, np.ndarray)):
        parts = [str(v) for v in value]
    else:
        s = str(value).strip()
        if not s or s.lower() in ("nan", "none", "(no genres listed)"):
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                parts = [p["name"] if isinstance(p, dict) and "name" in p else str(p)
                         for p in parsed]
            except (ValueError, SyntaxError):
                parts = _SPLIT_RE.split(s.strip("[]"))
        else:
            parts = _SPLIT_RE.split(s)
    out, seen = [], set()
    for p in parts:
        p = p.strip().strip("'\"").strip()
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)
    return out


def _clean(value) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    s = str(value).strip()
    return "" if s.lower() in ("nan", "none") else s


def item_genres(items: pd.DataFrame) -> pd.Series:
    """Genre list per movie: TMDB genres, else MovieLens genres."""
    tmdb = items["tmdb_genres"].map(split_list) if "tmdb_genres" in items else None
    ml = items["genres"].map(split_list) if "genres" in items else None
    if tmdb is None and ml is None:
        return pd.Series([[]] * len(items), index=items.index)
    if tmdb is None:
        return ml
    if ml is None:
        return tmdb
    return pd.Series([t if t else m for t, m in zip(tmdb, ml)], index=items.index)


def build_item_documents(items: pd.DataFrame, max_keywords: int = 20) -> pd.Series:
    """One text document per movie, indexed by movieId (same order as `items`)."""
    title = items["title_clean"] if "title_clean" in items else items["title"]
    year = items["year"] if "year" in items else pd.Series([None] * len(items), index=items.index)
    genres = item_genres(items)
    keywords = (items["keywords"].map(split_list) if "keywords" in items
                else pd.Series([[]] * len(items), index=items.index))
    tagline = items["tagline"] if "tagline" in items else pd.Series([""] * len(items), index=items.index)
    overview = items["overview"] if "overview" in items else pd.Series([""] * len(items), index=items.index)

    docs = []
    for t, y, g, kw, tag, ov in zip(title, year, genres, keywords, tagline, overview):
        head = _clean(t)
        if pd.notna(y):
            head += f" ({int(y)})"
        parts = [head + "."]
        if g:
            parts.append("Genres: " + ", ".join(g) + ".")
        if kw:
            parts.append("Keywords: " + ", ".join(kw[:max_keywords]) + ".")
        for extra in (_clean(tag), _clean(ov)):
            if extra:
                parts.append(extra)
        docs.append(" ".join(parts))
    return pd.Series(docs, index=items["movieId"].to_numpy(), name="document")


def genre_matrix(items: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Multi-hot MovieLens genres (rows follow `items`) — used for intra-list diversity."""
    lists = (items["genres"] if "genres" in items else item_genres(items)).map(split_list)
    vocab = sorted({g for gs in lists for g in gs})
    col = {g: j for j, g in enumerate(vocab)}
    mat = np.zeros((len(items), len(vocab)), dtype=np.float32)
    for i, gs in enumerate(lists):
        for g in gs:
            mat[i, col[g]] = 1.0
    return mat, vocab
