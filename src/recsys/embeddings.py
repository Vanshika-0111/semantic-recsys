"""Day 4: dense semantic item embeddings, a FAISS Flat index, and an embedding recommender.

Pipeline
  1. Item documents come from recsys.text.build_item_documents — the SAME text TF-IDF used
     on Day 3, so "lexical vs semantic" compares representations, not inputs.
  2. A sentence encoder (MiniLM or bge-small, see config.EMB_MODELS) maps every document to a
     384-dim vector. Vectors are L2-normalised, so  dot product == cosine similarity.
  3. FlatIndex wraps faiss.IndexFlatIP: exact (brute-force) inner-product search. It is the
     ground truth that approximate indexes (HNSW, Day 5) are measured against, and the
     candidate generator on Day 8.
  4. Two ways to score a user, both with the same history weights W as TF-IDF:
       centroid : profile = normalised Σ w(u,i)·v_i ;  score(u, j) = cos(profile_u, v_j)
       knn      : FlatIndex.knn_graph -> each movie's top-N neighbours ;
                  score(u, j) = Σ_i w(u,i)·cos(v_i, v_j) over neighbours only  (= Day 3 method)
     Optional mean-centering of the vectors (center_vectors). Then the shared Day 3 path:
     add a tiny popularity tie-break -> hide already-rated movies -> top-K.

Why normalise the profile? Ranking for ONE user does not change, but it puts every user's
scores on the same [-1, 1] scale, so the 1e-6 popularity tie-break means the same thing
for everyone and a user with an empty profile (all zeros) falls back to pure popularity.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .baselines import TIE_EPS, ItemIndex, batches, history_weights, mask_seen, topk
from .metrics import PAD


# --------------------------------------------------------------------------- #
# Vectors
# --------------------------------------------------------------------------- #
def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise unit length (float32 copy). All-zero rows stay zero instead of becoming NaN."""
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, norms, out=np.zeros_like(X), where=norms > 0)


def center_vectors(X: np.ndarray) -> np.ndarray:
    """Subtract the catalog mean vector, then re-normalise.

    Sentence embeddings are anisotropic: all vectors lean in one shared direction, so almost
    every pair has cosine 0.2-0.6. Averaging many liked movies then cancels the personal part
    and keeps that shared direction -> everyone gets the same 'generic' movies (hubs).
    Removing the mean makes cosine measure how two movies differ from an average movie."""
    X = np.asarray(X, dtype=np.float32)
    return l2_normalize(X - X.mean(axis=0, keepdims=True))


def documents_hash(docs: pd.Series) -> str:
    """Fingerprint of (movieId, text) pairs. Cached embeddings are reused only if this matches,
    so changing the document template on a later day can never silently mix old vectors."""
    h = hashlib.sha256()
    h.update(np.asarray(docs.index, dtype=np.int64).tobytes())
    for text in docs.tolist():
        h.update(text.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def load_encoder(hf_name: str, device: str | None = None):
    """Download (first time, ~90-130 MB) and load a sentence-transformers model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(hf_name, device=device)


def token_lengths(encoder, texts: list[str], chunk: int = 2000) -> np.ndarray:
    """Number of word-piece tokens per text INCLUDING [CLS]/[SEP] and WITHOUT truncation,
    so we can report how many documents the model has to cut (max_seq_length)."""
    tok = encoder.tokenizer
    out = []
    for s in range(0, len(texts), chunk):
        ids = tok(texts[s:s + chunk], add_special_tokens=True, truncation=False)["input_ids"]
        out.extend(len(x) for x in ids)
    return np.asarray(out, dtype=np.int64)


def encode_texts(encoder, texts: list[str], batch_size: int = 64, prefix: str = "",
                 show_progress: bool = True) -> np.ndarray:
    """(n, d) float32, unit-length rows. `prefix` is for bge QUERIES only, never documents."""
    E = encoder.encode([prefix + t for t in texts], batch_size=batch_size,
                       normalize_embeddings=True, convert_to_numpy=True,
                       show_progress_bar=show_progress)
    return l2_normalize(E)


# --------------------------------------------------------------------------- #
# Cache on disk:  <dir>/<key>.npy  (vectors)  .ids.npy (movieIds)  .json (meta)
# --------------------------------------------------------------------------- #
def save_embeddings(directory: Path, key: str, vectors: np.ndarray, ids: np.ndarray,
                    meta: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{key}.npy", np.asarray(vectors, dtype=np.float32))
    np.save(directory / f"{key}.ids.npy", np.asarray(ids, dtype=np.int64))
    (directory / f"{key}.json").write_text(json.dumps(meta, indent=2))


def load_embeddings(directory: Path, key: str, expected_ids: np.ndarray | None = None,
                    expected_hash: str | None = None) -> tuple[np.ndarray, dict] | None:
    """Cached vectors, or None if missing or stale (different catalog or documents)."""
    paths = [directory / f"{key}{s}" for s in (".npy", ".ids.npy", ".json")]
    if not all(p.exists() for p in paths):
        return None
    ids = np.load(paths[1])
    meta = json.loads(paths[2].read_text())
    if expected_ids is not None and not np.array_equal(ids, np.asarray(expected_ids, np.int64)):
        return None
    if expected_hash is not None and meta.get("documents_hash") != expected_hash:
        return None
    return np.load(paths[0]), meta


# --------------------------------------------------------------------------- #
# FAISS exact index
# --------------------------------------------------------------------------- #
class FlatIndex:
    """Exact inner-product search over unit vectors (= cosine), returning movieIds.

    IndexFlatIP compares the query with EVERY item: O(n·d) per query, no training, no
    approximation. 23k items x 384 dims is ~9M multiply-adds per query (a few ms on a
    laptop), so exact search is affordable here; HNSW (Day 5) trades a little recall
    for speed and is what you'd use at millions of items.
    """

    def __init__(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        import faiss

        V = np.ascontiguousarray(vectors, dtype=np.float32)
        self.ids = np.asarray(ids, dtype=np.int64)
        if V.shape[0] != len(self.ids):
            raise ValueError("one id per vector required")
        self.vectors = V
        self.index = faiss.IndexFlatIP(V.shape[1])
        self.index.add(V)

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(scores, movieIds), each (n_queries, k), best first; missing slots -> PAD."""
        Q = np.ascontiguousarray(np.atleast_2d(queries), dtype=np.float32)
        k_eff = min(k, self.ntotal)
        D, I = self.index.search(Q, k_eff)
        ids = np.where(I >= 0, self.ids[np.maximum(I, 0)], PAD)
        if k_eff < k:
            pad = k - k_eff
            ids = np.pad(ids, ((0, 0), (0, pad)), constant_values=PAD)
            D = np.pad(D, ((0, 0), (0, pad)), constant_values=-np.inf)
        return D, ids

    def similar(self, movie_ids, k: int) -> tuple[np.ndarray, np.ndarray]:
        """k nearest OTHER movies for each given movie (the movie itself is removed)."""
        movie_ids = np.asarray(movie_ids, dtype=np.int64)
        pos = np.searchsorted(self.ids, movie_ids)
        if np.any(self.ids[np.minimum(pos, len(self.ids) - 1)] != movie_ids):
            raise KeyError("movie not in the index")
        Q = self.vectors[pos]
        D, I = self.search(Q, k + 1)
        out_ids = np.full((len(movie_ids), k), PAD, dtype=np.int64)
        out_d = np.full((len(movie_ids), k), -np.inf, dtype=np.float32)
        for r, m in enumerate(movie_ids):
            keep = I[r] != m                      # usually drops exactly the first column
            ids_r, d_r = I[r][keep][:k], D[r][keep][:k]
            out_ids[r, :len(ids_r)], out_d[r, :len(d_r)] = ids_r, d_r
        return out_d, out_ids

    def knn_graph(self, n_neighbors: int) -> sp.csr_matrix:
        """Item-item graph: each movie's n most similar OTHER movies (positive cosine only),
        as a sparse (items x items) matrix in catalog column order — the same structure as
        the Day 3 TF-IDF kNN graph, so ItemKNNRecommender can use it unchanged."""
        n = self.ntotal
        k = min(n_neighbors, n - 1)
        D, I = self.search(self.vectors, k + 1)
        keep = (I != self.ids[:, None]) & (I != PAD)          # drop the movie itself
        keep &= np.cumsum(keep, axis=1) <= k                  # at most k per row
        keep &= D > 0
        rows = np.repeat(np.arange(n), I.shape[1]).reshape(I.shape)[keep]
        cols = np.searchsorted(self.ids, I[keep])
        return sp.csr_matrix((D[keep].astype(np.float32), (rows, cols)), shape=(n, n))

    def save(self, path: Path) -> None:
        import faiss

        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))


# --------------------------------------------------------------------------- #
# Recommender
# --------------------------------------------------------------------------- #
@dataclass
class EmbeddingRecommender:
    item_vectors: np.ndarray            # rows follow index.ids, unit length
    index: ItemIndex
    weighting: str = "liked"            # "liked" | "centered"   (same meaning as Day 3)
    max_history: int | None = None
    threshold: float = 4.0
    center: bool = False                # subtract the catalog mean first (see center_vectors)

    def __post_init__(self) -> None:
        if self.item_vectors.shape[0] != len(self.index):
            raise ValueError("item_vectors must have one row per catalog movie")
        self.item_vectors = (center_vectors(self.item_vectors) if self.center
                             else l2_normalize(self.item_vectors))

    def user_profiles(self, history: pd.DataFrame, users: np.ndarray) -> np.ndarray:
        """(len(users), d) unit profiles; users with no usable history get a zero row."""
        W = history_weights(history, users, self.index, self.weighting,
                            self.max_history, self.threshold)
        return l2_normalize(np.asarray(W @ self.item_vectors))

    def scores(self, profiles: np.ndarray) -> np.ndarray:
        return profiles @ self.item_vectors.T

    def recommend(self, profiles: np.ndarray, seen: sp.csr_matrix, k: int,
                  fallback: np.ndarray, batch_size: int = 1000) -> np.ndarray:
        """fallback: popularity score per item in (0, 1] — tie-break and empty-profile fill."""
        out = np.empty((profiles.shape[0], k), dtype=np.int64)
        tie = (TIE_EPS * np.asarray(fallback)).astype(np.float32)
        for b in batches(profiles.shape[0], batch_size):
            s = self.scores(profiles[b])
            s += tie
            mask_seen(s, seen[b])
            out[b] = topk(s, k, self.index)
        return out

    def explain(self, history_cols: np.ndarray, item_col: int) -> int | None:
        """Column of the history movie most similar to `item_col` ("Because you liked X")."""
        history_cols = np.asarray(history_cols, dtype=np.int64)
        if history_cols.size == 0:
            return None
        sims = self.item_vectors[history_cols] @ self.item_vectors[item_col]
        return int(history_cols[int(np.argmax(sims))])


# --------------------------------------------------------------------------- #
# Intrinsic checks (metadata only — no ratings, so no leakage questions)
# --------------------------------------------------------------------------- #
def topk_from_sparse(sim: sp.csr_matrix, k: int) -> np.ndarray:
    """Top-k column ids per row of a sparse similarity matrix (TF-IDF kNN graph); PAD if short."""
    sim = sim.tocsr()
    out = np.full((sim.shape[0], k), PAD, dtype=np.int64)
    for r in range(sim.shape[0]):
        a, b = sim.indptr[r], sim.indptr[r + 1]
        if b > a:
            cols, vals = sim.indices[a:b], sim.data[a:b]
            order = np.lexsort((cols, -vals))[:k]
            out[r, :len(order)] = cols[order]
    return out


def neighbor_agreement(neighbor_cols: np.ndarray, genres: np.ndarray,
                       years: np.ndarray) -> dict:
    """How 'sensible' item neighbours are, from metadata alone.

    neighbor_cols : (n_items, k) catalog columns of each item's neighbours (PAD = missing)
    genres        : (n_items, g) multi-hot genres;  years: (n_items,) float, NaN if unknown
    genre_jaccard : mean |G_i ∩ G_j| / |G_i ∪ G_j| over (item, neighbour) pairs
    year_gap      : median |year_i - year_j|
    """
    n, k = neighbor_cols.shape
    valid = neighbor_cols != PAD
    rows = np.repeat(np.arange(n), k)[valid.ravel()]
    cols = neighbor_cols.ravel()[valid.ravel()]
    G = genres.astype(bool)
    inter = (G[rows] & G[cols]).sum(axis=1)
    union = (G[rows] | G[cols]).sum(axis=1)
    jac = np.divide(inter, union, out=np.zeros(len(rows)), where=union > 0)
    gap = np.abs(years[rows] - years[cols])
    gap = gap[~np.isnan(gap)]
    return {"genre_jaccard@%d" % k: float(jac.mean()),
            "year_gap_median@%d" % k: float(np.median(gap)) if gap.size else float("nan")}
