"""Day 4: sentence-embedding item vectors (MiniLM vs bge-small) + FAISS Flat index.

Stages (all by default):
  embed      encode every item document with each model -> data/embeddings/<model>.npy (cached;
             re-used unless the documents or catalog change)
  neighbors  FAISS Flat index per model; "similar movies" sanity table (vs TF-IDF);
             metadata agreement of neighbours; single-query latency
  tune       VALIDATION only (fit on train, score val, the same sampled users as Day 3):
             embedding recommender x 4 profile settings x 2 models, next to popularity and
             TF-IDF -> picks the encoder + profile setting carried into Day 5

Outputs
  reports/results/day04_embeddings.json + .md      (commit)
  reports/results/day04_similar_movies.md          (commit)
  data/embeddings/<model>.npy|.ids.npy|.json|_flat.faiss   (git-ignored)

Usage
  python scripts/run_embeddings.py --smoke             # 3k val users (embeds full catalog once)
  python scripts/run_embeddings.py                     # full run
  python scripts/run_embeddings.py --stages tune       # re-use cached vectors
  python scripts/run_embeddings.py --models minilm     # one encoder only
"""
from __future__ import annotations

import os
import sys

# torch and faiss each ship an OpenMP runtime on Windows; loading both in one process can
# abort with "OMP: Error #15 ... libiomp5md.dll already initialized". This is the standard
# workaround (set before either library is imported).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
try:                     # same Windows SSL fix as download_data.py (model download from HF)
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recsys import config  # noqa: E402
from recsys import split as S  # noqa: E402
from recsys.baselines import (ItemIndex, ItemKNNRecommender,  # noqa: E402
                              PopularityRecommender, TfidfKNNRecommender, user_item_matrix)
from recsys.embeddings import (EmbeddingRecommender, FlatIndex, documents_hash,  # noqa: E402
                               encode_texts, load_embeddings, load_encoder,
                               neighbor_agreement, save_embeddings, token_lengths,
                               topk_from_sparse)
from recsys.metrics import PAD, beyond_accuracy, evaluate  # noqa: E402
from recsys.text import build_item_documents, genre_matrix  # noqa: E402
from run_baselines import eval_users, log, md_table, only, row  # noqa: E402  (Day 7: move to src)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402

COLS = ["userId", "movieId", "rating"]
KMAX = max(config.EVAL_KS)
N_NEIGHBORS = 10


def cfg_name(cfg: dict) -> str:
    if "method" in cfg:          # Day 4 embedding grid
        return (f"{cfg['method']}, {'mean-centered' if cfg['center'] else 'raw'}, "
                f"history={cfg['max_history'] or 'all'}")
    return f"weighting={cfg['weighting']}, history={cfg['max_history'] or 'all'}"


def to_cols(ids: np.ndarray, index: ItemIndex) -> np.ndarray:
    """movieId matrix (PAD allowed) -> catalog column matrix (PAD kept)."""
    out = np.full(ids.shape, PAD, dtype=np.int64)
    ok = ids != PAD
    out[ok] = index.cols(ids[ok])
    return out


def load_tfidf_sim(index: ItemIndex) -> sp.csr_matrix | None:
    path = config.RUNS_DIR / f"tfidf_knn_{config.KNN_NEIGHBORS}.npz"
    ids_path = path.with_suffix(".ids.npy")
    if path.exists() and ids_path.exists() and np.array_equal(np.load(ids_path), index.ids):
        return sp.load_npz(path).tocsr()
    return None


def find_movie(items: pd.DataFrame, title: str, year: int) -> int | None:
    t = items["title_clean"].str.lower()
    hit = items[t.str.startswith(title.lower()) & (items["year"] == year)]
    if hit.empty:
        return None
    exact = hit[hit["title_clean"].str.lower() == title.lower()]
    return int((exact if len(exact) else hit)["movieId"].iloc[0])


# --------------------------------------------------------------------------- #
def stage_embed(models, docs, index, results, args, t0) -> None:
    h = documents_hash(docs)
    texts = docs.reindex(index.ids).tolist()
    for key in models:
        spec = config.EMB_MODELS[key]
        cached = None if args.re_embed else load_embeddings(config.EMB_DIR, key, index.ids, h)
        if cached is not None:
            results.setdefault("embed", {})[key] = cached[1]
            log(f"{key}: cached vectors {cached[0].shape} (documents unchanged)", t0)
            continue
        log(f"{key}: loading {spec['hf_name']} (downloads once, ~90-130 MB)", t0)
        enc = load_encoder(spec["hf_name"])
        max_len = int(enc.max_seq_length)
        lengths = token_lengths(enc, texts)
        log(f"{key}: device={enc.device}, max_seq_length={max_len}, "
            f"tokens median={int(np.median(lengths))} p95={int(np.percentile(lengths, 95))} "
            f"max={int(lengths.max())}; truncated={np.mean(lengths > max_len):.2%}", t0)
        t1 = time.time()
        E = encode_texts(enc, texts, batch_size=args.batch_size)   # documents: NO prefix
        secs = time.time() - t1
        meta = {
            "model": key, "hf_name": spec["hf_name"], "dim": int(E.shape[1]),
            "n_items": int(E.shape[0]), "documents_hash": h, "device": str(enc.device),
            "max_seq_length": max_len,
            "tokens_median": float(np.median(lengths)),
            "tokens_p95": float(np.percentile(lengths, 95)),
            "tokens_max": int(lengths.max()),
            "truncated_share": float(np.mean(lengths > max_len)),
            "seconds": round(secs, 1), "docs_per_sec": round(len(texts) / secs, 1),
            "batch_size": args.batch_size,
        }
        save_embeddings(config.EMB_DIR, key, E, index.ids, meta)
        results.setdefault("embed", {})[key] = meta
        log(f"{key}: encoded {E.shape[0]:,} docs -> {E.shape[1]}-d in {secs:.0f}s "
            f"({meta['docs_per_sec']:.0f} docs/s)", t0)
        del enc, E
        gc.collect()


def stage_neighbors(models, docs, items, index, results, args, t0) -> None:
    h = documents_hash(docs)
    genres, _ = genre_matrix(items)
    years = items["year"].astype("float64").to_numpy()
    rng = np.random.default_rng(config.SEED)
    sanity = [(t, y, find_movie(items, t, y)) for t, y in config.SANITY_MOVIES]
    sanity = [(t, y, m) for t, y, m in sanity if m is not None]
    label = dict(zip(items["movieId"], items["title_clean"] + " (" +
                     items["year"].astype("string").fillna("?") + ")"))
    lists: dict[str, dict[int, list[str]]] = {}
    rows = []

    sim = load_tfidf_sim(index)
    if sim is not None:
        cols = topk_from_sparse(sim, N_NEIGHBORS)
        rows.append({"model": "tfidf", **neighbor_agreement(cols, genres, years)})
        ids = np.where(cols == PAD, PAD, index.ids[np.maximum(cols, 0)])
        pos = {m: i for i, m in enumerate(index.ids)}
        lists["tfidf"] = {m: [label[x] for x in ids[pos[m]][:5] if x != PAD] for _, _, m in sanity}
        log(f"tfidf: neighbour agreement {rows[-1]}", t0)
    else:
        log("tfidf: no cached kNN graph (run Day 3 first) — skipped in the comparison", t0)

    for key in models:
        cached = load_embeddings(config.EMB_DIR, key, index.ids, h)
        if cached is None:
            raise SystemExit(f"{key}: no cached vectors for these documents; run --stages embed")
        flat = FlatIndex(cached[0], index.ids)
        flat.save(config.EMB_DIR / f"{key}_flat.faiss")
        _, nb = flat.similar(index.ids, N_NEIGHBORS)
        agree = neighbor_agreement(to_cols(nb, index), genres, years)

        # single-query latency (one user/item at a time, like the API on Day 12)
        q = flat.vectors[rng.choice(flat.ntotal, size=args.latency_queries, replace=False)]
        flat.search(q[:5], 100)                                   # warm-up
        ms = []
        for v in q:
            t1 = time.perf_counter()
            flat.search(v, 100)
            ms.append(1000 * (time.perf_counter() - t1))
        rows.append({"model": key, **agree, "flat_p50_ms": float(np.percentile(ms, 50)),
                     "flat_p95_ms": float(np.percentile(ms, 95))})
        pos = {m: i for i, m in enumerate(index.ids)}
        lists[key] = {m: [label[x] for x in nb[pos[m]][:5] if x != PAD] for _, _, m in sanity}
        log(f"{key}: Flat index ({flat.ntotal:,} x {flat.vectors.shape[1]}) agreement={agree} "
            f"top-100 query p50={rows[-1]['flat_p50_ms']:.2f} ms p95={rows[-1]['flat_p95_ms']:.2f} ms",
            t0)
        del flat, cached
        gc.collect()

    results["neighbors"] = {"k": N_NEIGHBORS, "rows": rows,
                            "latency_queries": args.latency_queries}
    write_similar_md(sanity, lists, label)


def stage_tune(models, docs, items, index, results, args, t0) -> None:
    h = documents_hash(docs)
    day3 = config.RESULTS_DIR / "day03_baselines.json"
    sel3 = json.loads(day3.read_text()).get("selected", {}) if day3.exists() else {}
    pop_mode = sel3.get("popularity", {}).get("mode", "liked")
    tf_cfg = sel3.get("tfidf_knn", dict(config.TFIDF_DEFAULT))

    train = S.load_view("train", config.SPLIT_DIR, columns=COLS)
    val = S.load_view("val", config.SPLIT_DIR, columns=COLS)
    users = eval_users(val, args.tune_users, config.SEED)       # same sample as Day 3
    counts = np.bincount(index.cols(train["movieId"].to_numpy()), minlength=len(index))
    head = index.ids[np.argsort(-counts, kind="stable")[:int(round(config.HEAD_SHARE * len(index)))]]
    pop = PopularityRecommender(pop_mode, config.RELEVANCE_THRESHOLD).fit(train, index)
    del counts
    val, train_u = only(val, users), only(train, users)
    del train
    gc.collect()
    seen = user_item_matrix(train_u, users, index, dtype=np.int8)
    genres, _ = genre_matrix(items)
    log(f"tune: {len(users):,} val users (fit on train)", t0)

    runs = []

    def score(name: str, cfg: str, recs: np.ndarray) -> dict:
        m = evaluate(users, recs, val, config.EVAL_KS)
        ba = beyond_accuracy(recs, index.ids, head, genres, k=10)
        r = row(name, m, {**ba, "config": cfg})
        runs.append(r)
        log(f"  {name:10s} {cfg:38s} ndcg@10={m['ndcg@10']:.4f} recall@10={m['recall@10']:.4f} "
            f"coverage@10={ba['coverage@10']:.3f}", t0)
        return r

    score("popularity", f"mode={pop_mode}", pop.recommend(seen, KMAX))
    sim = load_tfidf_sim(index)
    if sim is not None:
        tf = TfidfKNNRecommender(threshold=config.RELEVANCE_THRESHOLD, **tf_cfg).set_similarity(sim, index)
        score("tfidf_knn", cfg_name(tf_cfg),
              tf.recommend(tf.user_weights(train_u, users), seen, KMAX, pop.score, args.batch_users))
        del sim, tf
        gc.collect()

    for key in models:
        cached = load_embeddings(config.EMB_DIR, key, index.ids, h)
        if cached is None:
            raise SystemExit(f"{key}: no cached vectors for these documents; run --stages embed")
        graphs: dict[bool, sp.csr_matrix] = {}          # kNN graph per centering choice
        for cfg in config.EMB_GRID:
            # weights are always 'liked' (run 1: 'centered' weights lost for both encoders)
            if cfg["method"] == "centroid":
                model = EmbeddingRecommender(cached[0], index, weighting="liked",
                                             max_history=cfg["max_history"],
                                             threshold=config.RELEVANCE_THRESHOLD,
                                             center=cfg["center"])
                P = model.user_profiles(train_u, users)
                recs = model.recommend(P, seen, KMAX, pop.score, args.batch_users)
                del P
            else:
                if cfg["center"] not in graphs:
                    vecs = EmbeddingRecommender(cached[0], index, center=cfg["center"]).item_vectors
                    graphs[cfg["center"]] = FlatIndex(vecs, index.ids).knn_graph(
                        config.EMB_KNN_NEIGHBORS)
                    log(f"  {key}: {config.EMB_KNN_NEIGHBORS}-NN graph "
                        f"({'mean-centered' if cfg['center'] else 'raw'}), "
                        f"nnz={graphs[cfg['center']].nnz:,}", t0)
                    del vecs
                model = ItemKNNRecommender(n_neighbors=config.EMB_KNN_NEIGHBORS, weighting="liked",
                                           max_history=cfg["max_history"],
                                           threshold=config.RELEVANCE_THRESHOLD)
                model.set_similarity(graphs[cfg["center"]], index)
                recs = model.recommend(model.user_weights(train_u, users), seen, KMAX,
                                       pop.score, args.batch_users)
            score(key, cfg_name(cfg), recs)
            del model, recs
            gc.collect()
        del graphs, cached
        gc.collect()

    emb_runs = [r for r in runs if r["model"] in models]
    best = max(emb_runs, key=lambda r: r["ndcg@10"])
    best_cfg = next(c for c in config.EMB_GRID if cfg_name(c) == best["config"])
    results["tuning"] = {"split": "val (fit on train)", "n_users": int(len(users)), "runs": runs}
    results["selected"] = {"model": best["model"], "weighting": "liked", **best_cfg}
    log(f"selected: {results['selected']}  (val ndcg@10={best['ndcg@10']:.4f})", t0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stages", nargs="+", default=["embed", "neighbors", "tune"],
                   choices=["embed", "neighbors", "tune"])
    p.add_argument("--models", nargs="+", default=list(config.EMB_MODELS),
                   choices=list(config.EMB_MODELS))
    p.add_argument("--batch-size", type=int, default=config.EMB_BATCH_SIZE,
                   help="encoder batch size (lower it if you run out of memory)")
    p.add_argument("--batch-users", type=int, default=1000)
    p.add_argument("--tune-users", type=int, default=config.TUNE_USERS)
    p.add_argument("--latency-queries", type=int, default=1000)
    p.add_argument("--re-embed", action="store_true", help="ignore cached vectors")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.tune_users, args.latency_queries = 3000, 200
    tag = "day04_embeddings_smoke" if args.smoke else "day04_embeddings"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = config.RESULTS_DIR / f"{tag}.json"
    results = json.loads(out_json.read_text()) if out_json.exists() else {}
    results["split_fingerprint"] = json.loads(
        (config.SPLIT_DIR / S.MANIFEST_FILE).read_text())["fingerprint"]
    t0 = time.time()

    items = pd.read_parquet(config.PROCESSED_DIR / "items.parquet")
    index = ItemIndex(items["movieId"])
    items = items.set_index("movieId").loc[index.ids].reset_index()
    docs = build_item_documents(items)
    log(f"catalog: {len(index):,} movies; models: {args.models}", t0)

    if "embed" in args.stages:
        stage_embed(args.models, docs, index, results, args, t0)
    if "neighbors" in args.stages:
        stage_neighbors(args.models, docs, items, index, results, args, t0)
    if "tune" in args.stages:
        stage_tune(args.models, docs, items, index, results, args, t0)

    results["seconds_last_run"] = round(time.time() - t0, 1)
    out_json.write_text(json.dumps(results, indent=2))
    write_md(results, out_json.with_suffix(".md"))
    log(f"wrote {out_json.relative_to(config.ROOT)} and .md", t0)


# --------------------------------------------------------------------------- #
def write_similar_md(sanity, lists, label) -> None:
    models = list(lists)
    md = "# Day 4 — Most similar movies (content only, no ratings)\n\n"
    md += ("Top-5 neighbours of well-known movies. TF-IDF = Day 3 lexical cosine; the others are "
           "FAISS Flat search over sentence-embedding vectors of the SAME item documents.\n\n")
    for _, _, m in sanity:
        md += f"### {label[m]}\n\n| # | " + " | ".join(models) + " |\n|---:|" + "---|" * len(models) + "\n"
        for i in range(5):
            md += f"| {i + 1} | " + " | ".join(
                (lists[k][m][i] if i < len(lists[k][m]) else "") for k in models) + " |\n"
        md += "\n"
    (config.RESULTS_DIR / "day04_similar_movies.md").write_text(md, encoding="utf-8")


def write_md(res: dict, path: Path) -> None:
    md = "# Day 4 — Sentence embeddings + FAISS Flat\n\n"
    md += f"Split fingerprint `{res['split_fingerprint'][:12]}…`. Item documents identical to Day 3.\n\n"
    if "embed" in res:
        rows = [{"model": k, **v} for k, v in res["embed"].items()]
        for r in rows:                              # whole numbers read better here
            r["truncated"] = f"{100 * r['truncated_share']:.2f}%"
            for c in ("tokens_median", "tokens_p95", "seconds", "docs_per_sec"):
                r[c] = int(round(r[c]))
        md += "## Encoding cost (one-off, whole catalog)\n\n"
        md += md_table(rows, ["model", "dim", "max_seq_length", "tokens_median", "tokens_p95",
                              "tokens_max", "truncated", "seconds", "docs_per_sec", "device"])
        md += "\n"
    if "neighbors" in res:
        n = res["neighbors"]
        md += f"## Item neighbours (top-{n['k']}) — metadata agreement and FAISS Flat latency\n\n"
        for r in n["rows"]:
            r[f"year_gap_median@{n['k']}"] = int(round(r[f"year_gap_median@{n['k']}"]))
        md += md_table(n["rows"], ["model", f"genre_jaccard@{n['k']}", f"year_gap_median@{n['k']}",
                                   "flat_p50_ms", "flat_p95_ms"])
        md += ("\ngenre_jaccard = mean genre overlap between a movie and its neighbours (higher = "
               "more on-topic); year_gap = median |year difference|. Latency: one top-100 query "
               f"at a time, {n['latency_queries']:,} queries. Examples: `day04_similar_movies.md`.\n\n")
    if "tuning" in res:
        t = res["tuning"]
        md += f"## Validation — fit on train, {t['n_users']:,} sampled users (same as Day 3)\n\n"
        md += md_table(t["runs"], ["model", "config", "ndcg@10", "recall@10", "mrr@10",
                                   "coverage@10", "long_tail@10", "diversity@10"])
        md += f"\nSelected for Day 5: `{json.dumps(res['selected'])}`\n"
    path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
