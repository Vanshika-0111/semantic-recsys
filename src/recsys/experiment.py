"""Shared plumbing for the experiment scripts (Day 7 refactor).

Until Day 6 these helpers lived inside scripts/run_baselines.py, and the Day 4-6 scripts imported
them from there (one script importing another script). They are library code now, tested in
tests/test_experiment.py, and every script uses the same definitions:

  setup_runtime()        environment fixes that must run before torch / faiss are imported
  log(msg, t0)           timestamped progress line
  eval_users / only      which users get scored; filter a ratings frame to those users
  rows_of                fast `only` for a batch of consecutive sorted users
  head_items             the 10% most-rated movies (the "head" for the long-tail metric)
  load_catalog           items in index order + item documents + genres + readable labels
  load_item_vectors      cached sentence-embedding vectors (clear error if missing or stale)
  read_results / chosen_settings   settings picked on validation by Days 3-5, from their json
  open_results / save_results / split_fingerprint   the per-day results file
  row / md_table         one result row; a markdown table of rows

Nothing here changes a number: the Day 7 check re-ran every script before and after the move
and compared the result files (see claude/day-07-refactor-pipeline.md in the project).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import split as S

COLS = ["userId", "movieId", "rating"]
TCOLS = ["userId", "movieId", "rating", "timestamp"]
KMAX = max(config.EVAL_KS)
SHOW = ["ndcg@5", "ndcg@10", "ndcg@20", "recall@10", "precision@10", "mrr@10", "hit_rate@10"]


# --------------------------------------------------------------------------- #
# Runtime
# --------------------------------------------------------------------------- #
def setup_runtime() -> None:
    """Call at the top of a script, BEFORE importing torch / faiss / sentence-transformers.

    - KMP_DUPLICATE_LIB_OK: torch and faiss each ship an OpenMP runtime on Windows; loading both
      can abort with "OMP: Error #15 ... libiomp5md.dll already initialized".
    - truststore: use the Windows certificate store for HTTPS (fixes the SSL error when Hugging
      Face downloads or checks a model; same fix as download_data.py on Day 1).
    """
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import truststore
        truststore.inject_into_ssl()
    except ImportError:
        pass


def log(msg: str, t0: float) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Users and items
# --------------------------------------------------------------------------- #
def eval_users(truth: pd.DataFrame, limit: int | None, seed: int) -> np.ndarray:
    """Users with >= 1 relevant held-out item (the only ones metrics are defined for).
    With `limit`, a seeded random sample of them (sorted) — the same sample every run."""
    users = np.unique(truth.loc[truth["rating"] >= config.RELEVANCE_THRESHOLD, "userId"]
                      .to_numpy(np.int64))
    if limit and limit < len(users):
        users = np.sort(np.random.default_rng(seed).choice(users, size=limit, replace=False))
    return users


def only(df: pd.DataFrame, users: np.ndarray) -> pd.DataFrame:
    """Rows of `df` that belong to `users`."""
    return df[np.isin(df["userId"].to_numpy(), users)]


def rows_of(df: pd.DataFrame, users_sorted: np.ndarray) -> pd.DataFrame:
    """Rows of `df` (sorted by userId) for a contiguous run of sorted users — a fast `only()`
    for batches, valid when df holds only users from the same sorted list (Day 8; shared Day 9)."""
    uid = df["userId"].to_numpy()
    a, b = np.searchsorted(uid, users_sorted[0]), np.searchsorted(uid, users_sorted[-1], side="right")
    return df.iloc[a:b]


def head_items(movie_ids: np.ndarray, index, share: float = config.HEAD_SHARE) -> np.ndarray:
    """movieIds of the `share` most-rated movies in `movie_ids` (one entry per training rating).
    Ties -> catalog order (stable sort), so the head is identical every run."""
    counts = np.bincount(index.cols(np.asarray(movie_ids)), minlength=len(index))
    n_head = int(round(share * len(index)))
    return index.ids[np.argsort(-counts, kind="stable")[:n_head]]


@dataclass
class Catalog:
    items: pd.DataFrame          # one row per movie, in index order (sorted movieId)
    index: object                # baselines.ItemIndex
    docs: pd.Series              # item documents (Day 3), indexed by movieId
    genres: np.ndarray           # multi-hot MovieLens genres, rows = index order
    vocab: list[str]             # genre names (columns of `genres`)
    label: dict                  # movieId -> "Title (year)"


def load_catalog(processed_dir: Path | None = None) -> Catalog:
    """items.parquet in index order + everything derived from item metadata alone."""
    from .baselines import ItemIndex
    from .text import build_item_documents, genre_matrix

    items = pd.read_parquet((processed_dir or config.PROCESSED_DIR) / "items.parquet")
    index = ItemIndex(items["movieId"])
    items = items.set_index("movieId").loc[index.ids].reset_index()
    genres, vocab = genre_matrix(items)
    label = dict(zip(items["movieId"], items["title_clean"] + " (" +
                     items["year"].astype("string").fillna("?") + ")"))
    return Catalog(items, index, build_item_documents(items), genres, vocab, label)


def load_item_vectors(catalog: Catalog, model_key: str, emb_dir: Path | None = None) -> np.ndarray:
    """Cached raw (L2-normalised, not centered) item vectors from Day 4, in index order."""
    from .embeddings import documents_hash, load_embeddings

    cached = load_embeddings(emb_dir or config.EMB_DIR, model_key, catalog.index.ids,
                             documents_hash(catalog.docs))
    if cached is None:
        raise SystemExit(f"{model_key}: no cached vectors for these documents — "
                         "run scripts/run_embeddings.py --stages embed first")
    return cached[0]


# --------------------------------------------------------------------------- #
# Results files
# --------------------------------------------------------------------------- #
def read_results(name: str, results_dir: Path | None = None) -> dict:
    """A results json from reports/results ({} if that day has not been run)."""
    p = (results_dir or config.RESULTS_DIR) / name
    return json.loads(p.read_text()) if p.exists() else {}


def chosen_settings(results_dir: Path | None = None) -> dict:
    """Settings picked ON VALIDATION by earlier days, with the defaults used if a day is missing.

    popularity_mode  Day 3   count | liked
    tfidf            Day 3   TF-IDF profile weights + history length
    encoder, center  Day 4   sentence encoder + mean-centering
    profile          Day 5   warm-user profile: weighting + recency half-life
    """
    sel3 = read_results("day03_baselines.json", results_dir).get("selected", {})
    sel4 = read_results("day04_embeddings.json", results_dir).get("selected", {})
    sel5 = read_results("day05_retrieval.json", results_dir).get("selected", {})
    return {
        "popularity_mode": sel3.get("popularity", {}).get("mode", "liked"),
        "tfidf": sel3.get("tfidf_knn", dict(config.TFIDF_DEFAULT)),
        "encoder": sel4.get("model", "bge-small"),
        "center": bool(sel4.get("center", True)),
        "profile": {"weighting": sel5.get("weighting", "graded"),
                    "half_life": sel5.get("half_life", 5)},
    }


def split_fingerprint(split_dir: Path | None = None) -> str:
    """SHA-256 of the frozen Day 2 split, stored in every results file."""
    return json.loads(((split_dir or config.SPLIT_DIR) / S.MANIFEST_FILE).read_text())["fingerprint"]


def open_results(tag: str) -> tuple[Path, dict]:
    """reports/results/<tag>.json and its current content. Stages that are not re-run keep
    their earlier results, so one stage can be repeated without redoing the others."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{tag}.json"
    return path, (json.loads(path.read_text()) if path.exists() else {})


def save_results(path: Path, results: dict) -> None:
    path.write_text(json.dumps(results, indent=2))


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def row(name: str, m: dict, extra: dict | None = None) -> dict:
    """One result row: model name + every float metric (rounded to 5 decimals) + extras."""
    d = {"model": name, **{k: round(v, 5) for k, v in m.items() if isinstance(v, float)},
         "n_users": m.get("n_users")}
    if extra:
        d.update({k: round(v, 5) if isinstance(v, float) else v for k, v in extra.items()})
    return d


def md_table(rows: list[dict], cols: list[str], left: tuple[str, ...] = ("model", "config")) -> str:
    """Markdown table; `left` columns left-aligned, the rest right-aligned (floats 4 decimals)."""
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" if c in left else "---:"
                                                       for c in cols) + "|\n"
    body = ""
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            cells.append(f"{v:.4f}" if isinstance(v, float) else f"{v:,}" if isinstance(v, int) else str(v))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body
