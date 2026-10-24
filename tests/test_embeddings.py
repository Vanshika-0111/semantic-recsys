"""Day 4 tests: embeddings, FAISS Flat index, embedding recommender.

No model download is needed: real encoders are replaced by tiny hand-made 2-d vectors, so
every expected number below can be checked on paper.

Toy catalog (movieId -> unit vector, all have length 1):
    10: ( 1.0,  0.0)      20: ( 0.0, 1.0)      30: ( 0.6, 0.8)
    40: ( 0.8, -0.6)      50: (-1.0, 0.0)
Cosine = dot product, e.g. cos(10, 30) = 1*0.6 + 0*0.8 = 0.6
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.baselines import ItemIndex, popularity_scores, user_item_matrix  # noqa: E402
from recsys.embeddings import (EmbeddingRecommender, FlatIndex, center_vectors,  # noqa: E402
                               documents_hash, encode_texts, l2_normalize, load_embeddings,
                               neighbor_agreement, save_embeddings, token_lengths,
                               topk_from_sparse)
from recsys.metrics import PAD  # noqa: E402

INDEX = ItemIndex([10, 20, 30, 40, 50])
V = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.8, -0.6], [-1.0, 0.0]], dtype=np.float32)


def _df(rows):
    return pd.DataFrame(rows, columns=["userId", "movieId", "rating"])


# Same toy ratings as test_baselines.py
# user 1: 10 (5), 20 (2)   user 2: 10 (4), 30 (4)   user 3: 10 (3), 40 (5)
TRAIN = _df([(1, 10, 5), (1, 20, 2), (2, 10, 4), (2, 30, 4), (3, 10, 3), (3, 40, 5)])
POP = popularity_scores(TRAIN, INDEX, "count")   # order 10, 20, 30, 40, 50 (ties -> smaller id)


# --------------------------------------------------------------------------- #
# Vectors
# --------------------------------------------------------------------------- #
def test_l2_normalize_keeps_zero_rows():
    # (3, 4) has length 5 -> (0.6, 0.8); a zero row must stay zero, not NaN
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0]])
    assert out.dtype == np.float32


class FakeEncoder:
    """Stands in for SentenceTransformer: 'vector' = (len(text), number of 'a's)."""

    class tokenizer:  # noqa: N801  (mimics encoder.tokenizer)
        def __new__(cls, texts, add_special_tokens=True, truncation=False):
            # one token per word, plus [CLS] and [SEP]
            return {"input_ids": [[0] * (len(t.split()) + 2) for t in texts]}

    def __init__(self):
        self.seen = None

    def encode(self, texts, **kw):
        self.seen = list(texts)
        return np.array([[len(t), t.count("a")] for t in texts], dtype=np.float32)


def test_encode_texts_normalises_and_prefixes():
    enc = FakeEncoder()
    E = encode_texts(enc, ["aaaa", "bbb"], prefix="q: ", show_progress=False)
    assert enc.seen == ["q: aaaa", "q: bbb"]
    # "q: aaaa" -> (7, 4) -> /sqrt(65)   "q: bbb" -> (6, 0) -> (1, 0)
    assert E[0] == pytest.approx(np.array([7, 4]) / np.sqrt(65))
    assert E[1] == pytest.approx([1.0, 0.0])


def test_token_lengths_counts_special_tokens():
    # "a b c" = 3 words + [CLS] + [SEP] = 5 ;  "x" -> 3
    assert token_lengths(FakeEncoder(), ["a b c", "x"]).tolist() == [5, 3]


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
def test_documents_hash_and_cache_roundtrip(tmp_path):
    docs = pd.Series(["A film.", "Another film."], index=[1, 2])
    h = documents_hash(docs)
    assert h == documents_hash(docs.copy())                         # deterministic
    assert h != documents_hash(pd.Series(["A film!", "Another film."], index=[1, 2]))
    assert h != documents_hash(pd.Series(["A film.", "Another film."], index=[1, 3]))

    save_embeddings(tmp_path, "m", V[:2], np.array([1, 2]), {"documents_hash": h})
    E, meta = load_embeddings(tmp_path, "m", np.array([1, 2]), h)
    assert np.array_equal(E, V[:2]) and meta["documents_hash"] == h
    assert load_embeddings(tmp_path, "m", np.array([1, 3]), h) is None     # other catalog
    assert load_embeddings(tmp_path, "m", np.array([1, 2]), "stale") is None
    assert load_embeddings(tmp_path, "missing") is None


# --------------------------------------------------------------------------- #
# FAISS Flat index
# --------------------------------------------------------------------------- #
def test_flat_search_by_hand():
    # query (1, 0): scores = first coordinate -> 10: 1.0, 40: 0.8, 30: 0.6, 20: 0.0, 50: -1.0
    idx = FlatIndex(V, INDEX.ids)
    D, I = idx.search(np.array([1.0, 0.0]), k=3)
    assert I.tolist() == [[10, 40, 30]]
    assert D[0] == pytest.approx([1.0, 0.8, 0.6])
    D, I = idx.search(np.array([1.0, 0.0]), k=7)                  # more than the catalog
    assert I[0, 5:].tolist() == [PAD, PAD]


def test_flat_similar_drops_the_movie_itself():
    # neighbours of 10: itself (1.0) removed -> 40 (0.8), 30 (0.6)
    # neighbours of 20 = (0, 1): 30 (0.8), then 10 and 50 BOTH score 0.0 (a tie FAISS may
    # order either way), 40 (-0.6) -> only the first neighbour is well defined: 30
    D, I = FlatIndex(V, INDEX.ids).similar([10, 20], k=2)
    assert I[0].tolist() == [40, 30] and D[0] == pytest.approx([0.8, 0.6])
    assert I[1, 0] == 30
    with pytest.raises(KeyError):
        FlatIndex(V, INDEX.ids).similar([99], k=2)


def test_flat_equals_brute_force():
    # "Flat" = exact: same top-10 as sorting all dot products with numpy
    rng = np.random.default_rng(0)
    X = l2_normalize(rng.normal(size=(300, 16)))
    Q = l2_normalize(rng.normal(size=(20, 16)))
    ids = np.arange(1000, 1300)
    _, I = FlatIndex(X, ids).search(Q, 10)
    brute = ids[np.argsort(-(Q @ X.T), axis=1)[:, :10]]
    assert np.array_equal(I, brute)


# --------------------------------------------------------------------------- #
# Recommender
# --------------------------------------------------------------------------- #
def _recommend(weighting, users, k=3, history=TRAIN):
    model = EmbeddingRecommender(V, INDEX, weighting=weighting)
    P = model.user_profiles(history, users)
    seen = user_item_matrix(history, users, INDEX, dtype=np.int8)
    return model, P, model.recommend(P, seen, k, POP)


def test_liked_profiles_by_hand():
    users = np.array([1, 2, 3])
    _, P, recs = _recommend("liked", users)
    # user 1 liked only 10 -> profile (1, 0).  Unseen: 30 0.6, 40 0.8, 50 -1.0
    assert P[0] == pytest.approx([1.0, 0.0])
    assert recs[0].tolist() == [40, 30, 50]
    # user 2 liked 10 and 30 -> (1.6, 0.8) / sqrt(3.2) = (0.894, 0.447)
    #   20: 0.447    40: 0.8*0.894 - 0.6*0.447 = 0.447    <- an exact TIE
    #   popularity breaks it: 20 is more popular than 40 (ties -> smaller id) -> 20 first
    assert P[1] == pytest.approx([2 / np.sqrt(5), 1 / np.sqrt(5)], abs=1e-6)
    assert recs[1].tolist() == [20, 40, 50]
    # user 3 liked only 40 (10 was 3 stars) -> (0.8, -0.6)
    #   20: -0.6   30: 0.48 - 0.48 = 0.0   50: -0.8   -> 30, 20, 50
    assert recs[2].tolist() == [30, 20, 50]


def test_centered_dislike_pushes_items_down():
    # user 1: mean 3.5 -> w(10) = +1.5, w(20) = -1.5 -> profile ∝ (1.5, -1.5) = (0.707, -0.707)
    #   30: 0.707*(0.6 - 0.8) = -0.141   40: 0.707*(0.8 + 0.6) = 0.990   50: -0.707
    _, P, recs = _recommend("centered", np.array([1]))
    assert P[0] == pytest.approx([np.sqrt(0.5), -np.sqrt(0.5)], abs=1e-6)
    assert recs[0].tolist() == [40, 30, 50]


def test_empty_profile_falls_back_to_popularity():
    # user 4 rated 50 with 2 stars -> no liked items -> zero profile -> popularity: 10, 20, 30
    hist = _df([(4, 50, 2)])
    _, P, recs = _recommend("liked", np.array([4]), history=hist)
    assert P[0].tolist() == [0.0, 0.0]
    assert recs[0].tolist() == [10, 20, 30]


def test_recommend_matches_faiss_search_without_seen_items():
    # Dense scoring (used for evaluation) and FAISS Flat must rank identically
    rng = np.random.default_rng(1)
    X = l2_normalize(rng.normal(size=(200, 8)))
    index = ItemIndex(np.arange(200) + 1)
    model = EmbeddingRecommender(X, index)
    P = l2_normalize(rng.normal(size=(30, 8)))
    no_seen = sp.csr_matrix((30, 200), dtype=np.int8)
    recs = model.recommend(P, no_seen, 10, np.zeros(200))
    _, I = FlatIndex(X, index.ids).search(P, 10)
    assert np.array_equal(recs, I)


def test_explain_picks_most_similar_history_item():
    # history of user 2 = {10, 30}; for candidate 40: cos(10,40) = 0.8, cos(30,40) = 0.0 -> 10
    model = EmbeddingRecommender(V, INDEX)
    hist = INDEX.cols([10, 30])
    assert INDEX.ids[model.explain(hist, int(INDEX.cols([40])[0]))] == 10
    assert model.explain(np.array([], dtype=np.int64), 0) is None


def test_wrong_shape_rejected():
    with pytest.raises(ValueError):
        EmbeddingRecommender(V[:4], INDEX)


# --------------------------------------------------------------------------- #
# Intrinsic neighbour checks
# --------------------------------------------------------------------------- #
def test_topk_from_sparse():
    S = sp.csr_matrix(np.array([[0, 0.2, 0.9], [0.5, 0, 0], [0, 0, 0]], dtype=np.float32))
    assert topk_from_sparse(S, 2).tolist() == [[2, 1], [0, PAD], [PAD, PAD]]


def test_neighbor_agreement_by_hand():
    # genres: item0 {A,B}, item1 {B}, item2 {C}; years 2000, 2004, NaN
    G = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
    years = np.array([2000.0, 2004.0, np.nan])
    nb = np.array([[1], [0], [0]])            # 0->1, 1->0, 2->0
    # Jaccard: 0-1 = 1/2, 1-0 = 1/2, 2-0 = 0/3 -> mean 1/3 ; year gaps 4, 4 (NaN dropped) -> 4
    out = neighbor_agreement(nb, G, years)
    assert out["genre_jaccard@1"] == pytest.approx(1 / 3)
    assert out["year_gap_median@1"] == 4.0


# --------------------------------------------------------------------------- #
# Fixes after run 1: mean-centering and an embedding kNN graph
# --------------------------------------------------------------------------- #
def test_center_vectors_by_hand():
    # (1, 0) and (0, 1): mean (0.5, 0.5) -> (0.5, -0.5), (-0.5, 0.5) -> length 1: (±0.707, ∓0.707)
    # Their cosine goes from 0 to -1: after removing what they share, they are opposites.
    C = center_vectors(np.array([[1.0, 0.0], [0.0, 1.0]]))
    r = np.sqrt(0.5)
    assert np.allclose(C, [[r, -r], [-r, r]])
    assert C[0] @ C[1] == pytest.approx(-1.0)
    assert np.allclose(EmbeddingRecommender(V, INDEX, center=True).item_vectors, center_vectors(V))


def test_knn_graph_by_hand():
    # n = 2 neighbours, positive cosine only, the movie itself excluded
    #   10 = (1, 0)  : 40 0.8, 30 0.6                      -> {40: 0.8, 30: 0.6}
    #   20 = (0, 1)  : 30 0.8, then 10 / 50 at 0.0 (not > 0) -> {30: 0.8}
    #   50 = (-1, 0) : every other movie <= 0                -> empty row
    G = FlatIndex(V, INDEX.ids).knn_graph(2).toarray()
    c = {m: i for i, m in enumerate(INDEX.ids)}
    assert G[c[10], c[40]] == pytest.approx(0.8) and G[c[10], c[30]] == pytest.approx(0.6)
    assert (G[c[10]] > 0).sum() == 2
    assert G[c[20], c[30]] == pytest.approx(0.8) and (G[c[20]] > 0).sum() == 1
    assert (G[c[50]] > 0).sum() == 0
    assert np.all(np.diag(G) == 0)
