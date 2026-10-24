"""Day 5: approximate nearest-neighbour search (HNSW) and how to measure it against exact search.

Exact search (FlatIndex, Day 4) compares a query with all n items: O(n·d). HNSW (Hierarchical
Navigable Small World) builds a layered graph once — every item linked to ~M near neighbours —
and answers a query by walking that graph greedily from an entry point, keeping a short list
of the `ef_search` best items seen so far. It visits a few thousand items instead of all n,
so it is much faster at scale, but it can MISS true neighbours. Two knobs:

  M               links per item. Larger = better recall, more memory, slower build.
  ef_search       size of the candidate list at query time. Larger = better recall, slower.
                  FAISS always uses at least k (you can't return 200 items from a list of 64).
  ef_construction the same idea while building the graph (quality of the links).

How to judge it: run the SAME queries through Flat (ground truth) and HNSW and report
  recall@k vs exact = |HNSW top-k ∩ Flat top-k| / k     (1.0 = no loss)
next to per-query latency (p50 / p95) and index size. That gives the recall/latency curve.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from .embeddings import FlatIndex
from .metrics import PAD


class HNSWIndex(FlatIndex):
    """Same interface as FlatIndex (search / similar / knn_graph / save, movieIds out);
    only the FAISS structure differs: IndexHNSWFlat with inner product (= cosine on unit vectors)."""

    def __init__(self, vectors: np.ndarray, ids: np.ndarray, M: int = 32,
                 ef_construction: int = 200, ef_search: int = 64) -> None:
        self.M, self.ef_construction = int(M), int(ef_construction)
        t0 = time.perf_counter()
        super().__init__(vectors, ids)               # calls _build_index, then adds vectors
        self.build_seconds = time.perf_counter() - t0
        self.ef_search = ef_search

    def _build_index(self, dim: int):
        import faiss

        index = faiss.IndexHNSWFlat(dim, self.M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = self.ef_construction
        return index

    @property
    def ef_search(self) -> int:
        return int(self.index.hnsw.efSearch)

    @ef_search.setter
    def ef_search(self, value: int) -> None:
        self.index.hnsw.efSearch = int(value)


def recall_vs_exact(approx_ids: np.ndarray, exact_ids: np.ndarray, k: int) -> float:
    """Mean over queries of |approx top-k ∩ exact top-k| / |exact top-k|  (PAD ignored).

    Order inside the top-k does not matter: we only ask whether HNSW FOUND the true neighbours."""
    A = np.asarray(approx_ids)[:, :k]
    E = np.asarray(exact_ids)[:, :k]
    vals = []
    for a, e in zip(A, E):
        e = e[e != PAD]
        if e.size:
            vals.append(np.isin(e, a[a != PAD]).sum() / e.size)
    return float(np.mean(vals)) if vals else float("nan")


@contextmanager
def faiss_threads(n: int | None):
    """Temporarily set FAISS's OpenMP thread count (None = leave unchanged).

    Latency is measured with 1 thread: one API worker serves one request, and with many threads
    a single tiny query mostly pays thread start-up costs, which makes timings noisy."""
    import faiss

    if n is None:
        yield
        return
    old = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(int(n))
    try:
        yield
    finally:
        faiss.omp_set_num_threads(old)


def query_latency_ms(index: FlatIndex, queries: np.ndarray, k: int, warmup: int = 20) -> np.ndarray:
    """Milliseconds per query, ONE query at a time (how an online API calls the index)."""
    Q = np.ascontiguousarray(queries, dtype=np.float32)
    index.search(Q[:warmup], k)
    out = np.empty(len(Q))
    for i in range(len(Q)):
        t0 = time.perf_counter()
        index.search(Q[i], k)
        out[i] = 1000.0 * (time.perf_counter() - t0)
    return out


def index_size_mb(index: FlatIndex) -> float:
    """Size of the serialised FAISS index (vectors + graph links for HNSW)."""
    import faiss

    return float(faiss.serialize_index(index.index).nbytes / 1e6)
