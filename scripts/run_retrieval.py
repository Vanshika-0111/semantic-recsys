"""Day 5: warm-user profiles (rating-weighted, recency-decayed), stage-1 retrieval eval,
FAISS Flat vs HNSW, and the first TEST numbers of the embedding recommender.

Uses the Day 4 choice (encoder + mean-centering, read from day04_embeddings.json) and the cached
item vectors in data/embeddings/ — nothing is re-encoded.

Stages (all by default):
  tune        VALIDATION (fit on train, same 30k users as Days 3-4): profile weighting
              (liked | graded) x recency half-life -> picks the Day 5 profile
  candidates  VALIDATION: recall@50/100/200/500 of embedding vs popularity retrieval, and of
              mixed sets (embedding top-a + popular fill, 200 in total) -> input for Day 8
  ann         Flat (exact) vs HNSW (M x ef_search) on real user-profile queries:
              recall vs exact @10 / @200, single-query p50/p95 latency, build time, size
              -> reports/figures/day05_flat_vs_hnsw.png
  test        TEST (fit on train+val, all warm test users): embedding recommender with the
              Day 4 profile and the Day 5 profile, next to the Day 3 baselines

Outputs
  reports/results/day05_retrieval.json + .md, day05_examples.md     (commit)
  reports/figures/day05_flat_vs_hnsw.png                             (commit)
  data/runs/day05_<model>_test_per_user.parquet                      (git-ignored)

Usage
  python scripts/run_retrieval.py --smoke                 # ~2-4 min check
  python scripts/run_retrieval.py                         # full run
  python scripts/run_retrieval.py --stages ann            # one stage (others are kept in the json)
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")     # torch/faiss OpenMP clash on Windows

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from recsys import config  # noqa: E402
from recsys import split as S  # noqa: E402
from recsys.ann import (HNSWIndex, faiss_threads, index_size_mb,  # noqa: E402
                        query_latency_ms, recall_vs_exact)
from recsys.baselines import ItemIndex, PopularityRecommender, user_item_matrix  # noqa: E402
from recsys.candidates import mix_candidates  # noqa: E402
from recsys.embeddings import (EmbeddingRecommender, FlatIndex, documents_hash,  # noqa: E402
                               load_embeddings)
from recsys.metrics import PAD, beyond_accuracy, evaluate  # noqa: E402
from recsys.text import build_item_documents, genre_matrix  # noqa: E402
from run_baselines import eval_users, log, md_table, only, row  # noqa: E402  (Day 7: move to src)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLS = ["userId", "movieId", "rating"]
KMAX = max(config.EVAL_KS)
SHOW = ["ndcg@5", "ndcg@10", "ndcg@20", "recall@10", "precision@10", "mrr@10", "hit_rate@10"]


def profile_name(cfg: dict) -> str:
    return f"weights={cfg['weighting']}, half-life={cfg['half_life'] or 'none'}"


class Context:
    """Everything the stages share, loaded once and only when a stage needs it."""

    def __init__(self, args, t0: float) -> None:
        self.args, self.t0 = args, t0
        items = pd.read_parquet(config.PROCESSED_DIR / "items.parquet")
        self.index = ItemIndex(items["movieId"])
        self.items = items.set_index("movieId").loc[self.index.ids].reset_index()
        docs = build_item_documents(self.items)
        self.genres, _ = genre_matrix(self.items)
        self.label = dict(zip(self.items["movieId"], self.items["title_clean"] + " (" +
                              self.items["year"].astype("string").fillna("?") + ")"))

        day4 = config.RESULTS_DIR / "day04_embeddings.json"
        sel4 = json.loads(day4.read_text()).get("selected", {}) if day4.exists() else {}
        self.model_key = sel4.get("model", "bge-small")
        self.center = bool(sel4.get("center", True))
        self.day4_val_ndcg = None
        if day4.exists():
            for r in json.loads(day4.read_text()).get("tuning", {}).get("runs", []):
                if (r["model"] == self.model_key and r["config"].startswith("centroid")
                        and ("mean-centered" in r["config"]) == self.center
                        and r["config"].endswith("history=all")):
                    self.day4_val_ndcg = r["ndcg@10"]
        cached = load_embeddings(config.EMB_DIR, self.model_key, self.index.ids, documents_hash(docs))
        if cached is None:
            raise SystemExit(f"{self.model_key}: no cached vectors for these documents — "
                             "run scripts/run_embeddings.py --stages embed first")
        self.vectors = cached[0]
        day3 = config.RESULTS_DIR / "day03_baselines.json"
        self.day3 = json.loads(day3.read_text()) if day3.exists() else {}
        self.pop_mode = self.day3.get("selected", {}).get("popularity", {}).get("mode", "liked")
        self._val = None
        log(f"catalog {len(self.index):,} movies; encoder {self.model_key} "
            f"({'mean-centered' if self.center else 'raw'}), popularity mode={self.pop_mode}", t0)

    def model(self, cfg: dict) -> EmbeddingRecommender:
        return EmbeddingRecommender(self.vectors, self.index, weighting=cfg["weighting"],
                                    threshold=config.RELEVANCE_THRESHOLD, center=self.center,
                                    half_life=cfg["half_life"])

    def val(self) -> dict:
        """Validation setup: fit on train, the same sampled val users as Days 3-4."""
        if self._val is None:
            train = S.load_view("train", config.SPLIT_DIR, columns=COLS)
            val = S.load_view("val", config.SPLIT_DIR, columns=COLS)
            users = eval_users(val, self.args.tune_users, config.SEED)
            pop = PopularityRecommender(self.pop_mode, config.RELEVANCE_THRESHOLD).fit(train, self.index)
            counts = np.bincount(self.index.cols(train["movieId"].to_numpy()), minlength=len(self.index))
            head = self.index.ids[np.argsort(-counts, kind="stable")
                                  [:int(round(config.HEAD_SHARE * len(self.index)))]]
            train_u = only(train, users)
            del train, counts
            gc.collect()
            self._val = {"users": users, "truth": only(val, users), "history": train_u,
                         "seen": user_item_matrix(train_u, users, self.index, dtype=np.int8),
                         "pop": pop, "head": head}
            log(f"validation: {len(users):,} users (fit on train)", self.t0)
        return self._val


# --------------------------------------------------------------------------- #
def stage_tune(ctx: Context, results: dict) -> None:
    v = ctx.val()
    runs = []

    def add(name: str, cfg: str, recs: np.ndarray) -> dict:
        m = evaluate(v["users"], recs, v["truth"], config.EVAL_KS)
        ba = beyond_accuracy(recs, ctx.index.ids, v["head"], ctx.genres, k=10)
        r = row(name, m, {**ba, "config": cfg})
        runs.append(r)
        log(f"  {name:10s} {cfg:34s} ndcg@10={m['ndcg@10']:.4f} recall@10={m['recall@10']:.4f} "
            f"coverage@10={ba['coverage@10']:.3f}", ctx.t0)
        return r

    add("popularity", f"mode={ctx.pop_mode}", v["pop"].recommend(v["seen"], KMAX))
    for cfg in config.PROFILE_GRID:
        model = ctx.model(cfg)
        P = model.user_profiles(v["history"], v["users"])
        add(ctx.model_key, profile_name(cfg),
            model.recommend(P, v["seen"], KMAX, v["pop"].score, ctx.args.batch_users))
        del model, P
        gc.collect()

    emb = [r for r in runs if r["model"] == ctx.model_key]
    base = next(r for r in emb if r["config"] == profile_name({"weighting": "liked", "half_life": None}))
    if ctx.day4_val_ndcg is not None and not ctx.args.smoke:
        same = abs(base["ndcg@10"] - ctx.day4_val_ndcg) < 5e-4
        log(f"  check: liked/no-decay = {base['ndcg@10']:.4f} vs Day 4 {ctx.day4_val_ndcg:.4f} "
            f"-> {'reproduced' if same else 'DIFFERENT — investigate before trusting Day 5'}", ctx.t0)
    best = max(emb, key=lambda r: r["ndcg@10"])
    best_cfg = next(c for c in config.PROFILE_GRID if profile_name(c) == best["config"])
    results["tuning"] = {"split": "val (fit on train)", "n_users": int(len(v["users"])),
                         "day4_val_ndcg@10": ctx.day4_val_ndcg, "runs": runs}
    results["selected"] = {"model": ctx.model_key, "center": ctx.center, **best_cfg}
    log(f"selected profile: {results['selected']} (val ndcg@10={best['ndcg@10']:.4f}, "
        f"no-decay liked {base['ndcg@10']:.4f})", ctx.t0)
    write_examples(ctx, best_cfg)


def selected_cfg(results: dict) -> dict:
    s = results.get("selected", {})
    return {"weighting": s.get("weighting", "liked"), "half_life": s.get("half_life")}


def stage_candidates(ctx: Context, results: dict) -> None:
    v = ctx.val()
    cfg = selected_cfg(results)
    n = max(max(config.CANDIDATE_BUDGETS), config.CANDIDATE_MIX_BUDGET)
    model = ctx.model(cfg)
    P = model.user_profiles(v["history"], v["users"])
    emb = model.recommend(P, v["seen"], n, v["pop"].score, ctx.args.batch_users)
    pop = v["pop"].recommend(v["seen"], n)
    del P
    gc.collect()
    ks = tuple(config.CANDIDATE_BUDGETS)
    rows = []
    for name, recs in ((f"{ctx.model_key} ({profile_name(cfg)})", emb),
                       (f"popularity ({ctx.pop_mode})", pop)):
        m = evaluate(v["users"], recs, v["truth"], ks)
        rows.append({"retriever": name, **{f"recall@{k}": round(m[f"recall@{k}"], 4) for k in ks},
                     **{f"hit_rate@{k}": round(m[f"hit_rate@{k}"], 4) for k in ks}})
        log(f"  {name:45s} " + "  ".join(f"recall@{k}={m[f'recall@{k}']:.3f}" for k in ks), ctx.t0)
    B = config.CANDIDATE_MIX_BUDGET
    mixes = []
    for share in config.CANDIDATE_MIX_SHARES:
        a = int(round(share * B))
        C = mix_candidates(emb, pop, a, B)
        m = evaluate(v["users"], C, v["truth"], (B,))
        mixes.append({"from_embeddings": a, "from_popularity": B - a,
                      f"recall@{B}": round(m[f"recall@{B}"], 4),
                      f"hit_rate@{B}": round(m[f"hit_rate@{B}"], 4),
                      "avg_unique": round(float((C != PAD).sum(axis=1).mean()), 1)})
        log(f"  mix {a:3d} emb + {B - a:3d} pop: recall@{B}={m[f'recall@{B}']:.4f}", ctx.t0)
    # how different are the two retrievers? share of the embedding top-B that popularity also has
    overlap = np.mean([np.isin(e[:B], p[:B]).mean() for e, p in zip(emb, pop)])
    results["candidates"] = {"split": "val (fit on train)", "n_users": int(len(v["users"])),
                             "profile": cfg, "rows": rows, "mix_budget": B, "mixes": mixes,
                             f"overlap@{B}": round(float(overlap), 4)}


def stage_ann(ctx: Context, results: dict) -> None:
    """Exact vs approximate search, with real user profiles as queries (what the API sends)."""
    import faiss

    a = ctx.args
    v = ctx.val()
    cfg = selected_cfg(results)
    model = ctx.model(cfg)
    rng = np.random.default_rng(config.SEED)
    q_users = np.sort(rng.choice(v["users"], size=min(a.ann_queries, len(v["users"])), replace=False))
    Q = model.user_profiles(only(v["history"], q_users), q_users)
    Q = Q[np.linalg.norm(Q, axis=1) > 0]                    # users with no liked movie: no query
    lat_Q = Q[:a.latency_queries]
    X = model.item_vectors                                  # the same (centered) vectors as scoring
    kmax = max(config.ANN_KS)
    log(f"ann: {len(Q):,} profile queries ({len(lat_Q):,} timed one by one, 1 FAISS thread), "
        f"{X.shape[0]:,} x {X.shape[1]} vectors, faiss {faiss.__version__}", ctx.t0)

    flat = FlatIndex(X, ctx.index.ids)
    _, exact = flat.search(Q, kmax)
    rows = []
    for k in config.ANN_KS:
        with faiss_threads(1):
            ms = query_latency_ms(flat, lat_Q, k)
        rows.append({"index": "Flat", "M": None, "ef_search": None, "k": k,
                     "recall": 1.0, "p50_ms": round(float(np.percentile(ms, 50)), 3),
                     "p95_ms": round(float(np.percentile(ms, 95)), 3),
                     "build_s": 0.0, "size_mb": round(index_size_mb(flat), 1)})
        log(f"  Flat          k={k:3d} recall=1.000 p50={rows[-1]['p50_ms']:.3f} ms "
            f"p95={rows[-1]['p95_ms']:.3f} ms size={rows[-1]['size_mb']} MB", ctx.t0)

    for M in a.hnsw_m:
        hnsw = HNSWIndex(X, ctx.index.ids, M=M, ef_construction=config.HNSW_EF_CONSTRUCTION)
        size = round(index_size_mb(hnsw), 1)
        log(f"  HNSW M={M}: built in {hnsw.build_seconds:.1f}s, {size} MB", ctx.t0)
        for k in config.ANN_KS:
            for ef in sorted({max(e, k) for e in config.HNSW_EF_SEARCH}):   # FAISS uses ef >= k
                hnsw.ef_search = ef
                _, I = hnsw.search(Q, k)
                rec = recall_vs_exact(I, exact, k)
                with faiss_threads(1):
                    ms = query_latency_ms(hnsw, lat_Q, k)
                rows.append({"index": "HNSW", "M": M, "ef_search": ef, "k": k,
                             "recall": round(rec, 4),
                             "p50_ms": round(float(np.percentile(ms, 50)), 3),
                             "p95_ms": round(float(np.percentile(ms, 95)), 3),
                             "build_s": round(hnsw.build_seconds, 1), "size_mb": size})
                log(f"  HNSW M={M:<3d} ef={ef:<4d} k={k:3d} recall={rec:.3f} "
                    f"p50={rows[-1]['p50_ms']:.3f} ms p95={rows[-1]['p95_ms']:.3f} ms", ctx.t0)
        del hnsw
        gc.collect()

    # Operating point per k: the fastest (p95) HNSW setting that still reaches the target recall
    rec_pt = {}
    for k in config.ANN_KS:
        ok = [r for r in rows if r["index"] == "HNSW" and r["k"] == k
              and r["recall"] >= config.ANN_TARGET_RECALL]
        flat_r = next(r for r in rows if r["index"] == "Flat" and r["k"] == k)
        if ok:
            best = min(ok, key=lambda r: r["p95_ms"])
            rec_pt[str(k)] = {**best, "speedup_p95_vs_flat": round(flat_r["p95_ms"] / best["p95_ms"], 1)}
    results["ann"] = {"n_queries": int(len(Q)), "n_timed": int(len(lat_Q)), "threads": 1,
                      "n_items": int(X.shape[0]), "dim": int(X.shape[1]),
                      "ef_construction": config.HNSW_EF_CONSTRUCTION,
                      "target_recall": config.ANN_TARGET_RECALL, "rows": rows,
                      "recommended": rec_pt, "faiss": faiss.__version__}
    for k, r in rec_pt.items():
        log(f"  operating point k={k}: M={r['M']} ef={r['ef_search']} recall={r['recall']:.3f} "
            f"p95={r['p95_ms']:.3f} ms (p95 speed-up vs Flat: {r['speedup_p95_vs_flat']}x)", ctx.t0)
    plot_ann(results["ann"], ctx.model_key, config.FIG_DIR / f"{ctx.tag}_flat_vs_hnsw.png")


def stage_test(ctx: Context, results: dict) -> None:
    a = ctx.args
    train_full = S.load_view("train_full", config.SPLIT_DIR, columns=COLS)
    test = S.load_view("test", config.SPLIT_DIR, columns=COLS)
    users = eval_users(test, a.test_users, config.SEED)
    test = only(test, users)
    pop = PopularityRecommender(ctx.pop_mode, config.RELEVANCE_THRESHOLD).fit(train_full, ctx.index)
    counts = np.bincount(ctx.index.cols(train_full["movieId"].to_numpy()), minlength=len(ctx.index))
    n_head = int(round(config.HEAD_SHARE * len(ctx.index)))
    head = ctx.index.ids[np.argsort(-counts, kind="stable")[:n_head]]
    hist = only(train_full, users)
    del train_full, counts
    gc.collect()
    seen = user_item_matrix(hist, users, ctx.index, dtype=np.int8)
    log(f"test: {len(users):,} users (fit on train+val)", ctx.t0)

    configs = {"day4 profile": {"weighting": "liked", "half_life": None},
               "day5 profile": selected_cfg(results)}
    if configs["day5 profile"] == configs["day4 profile"]:
        del configs["day4 profile"]
    rows = []
    for label, cfg in configs.items():
        t1 = time.time()
        model = ctx.model(cfg)
        P = model.user_profiles(hist, users)
        recs = model.recommend(P, seen, KMAX, pop.score, a.batch_users)
        secs = time.time() - t1
        del P
        m, per_user = evaluate(users, recs, test, config.EVAL_KS, return_per_user=True)
        ba = beyond_accuracy(recs, ctx.index.ids, head, ctx.genres, k=10)
        tag = "day05_" + label.split()[0] + "_" + ctx.model_key
        if not a.smoke:
            config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
            per_user.to_parquet(config.RUNS_DIR / f"{tag}_test_per_user.parquet", index=False)
        rows.append(row(f"{ctx.model_key} ({label})", m,
                        {**ba, "config": profile_name(cfg), "ms_per_user": 1000 * secs / len(users)}))
        log(f"  {label}: ndcg@10={m['ndcg@10']:.4f} recall@10={m['recall@10']:.4f} "
            f"coverage@10={ba['coverage@10']:.3f} ({secs:.0f}s)", ctx.t0)
        del recs, per_user, model
        gc.collect()
    results["test"] = {"split": "test (fit on train+val)", "n_users": int(len(users)),
                       "rows": rows}


# --------------------------------------------------------------------------- #
def write_examples(ctx: Context, cfg: dict, n_users: int = 3) -> None:
    """A few real validation users: what they liked recently, what we recommend, and why."""
    v = ctx.val()
    model = ctx.model(cfg)
    rng = np.random.default_rng(config.SEED + 5)
    liked = v["history"][v["history"]["rating"] >= config.RELEVANCE_THRESHOLD]
    counts = liked.groupby("userId").size()
    pool = counts.index[(counts >= 5) & (counts <= 60)].to_numpy()     # readable histories
    if len(pool) == 0:
        return
    picked = np.sort(rng.choice(pool, size=min(n_users, len(pool)), replace=False))
    pos = np.searchsorted(v["users"], picked)
    P = model.user_profiles(only(v["history"], picked), picked)
    recs = model.recommend(P, v["seen"][pos], 10, v["pop"].score)
    truth = v["truth"]
    md = f"# Day 5 — example users ({ctx.model_key}, {profile_name(cfg)})\n\n"
    md += ("Validation users (fit on train). ✓ = the user rated it ≥ 4 in their next 10 ratings. "
           "'Because you liked' = the liked movie most similar to the recommendation.\n\n")
    for u, r in zip(picked, recs):
        h = liked[liked["userId"] == u]
        nxt = set(truth.loc[(truth["userId"] == u) & (truth["rating"] >= config.RELEVANCE_THRESHOLD),
                            "movieId"].tolist())
        md += f"## User {u} — {len(h)} liked movies in train\n\nMost recent likes: "
        md += "; ".join(ctx.label[m] for m in h["movieId"].tolist()[-5:][::-1]) + "\n\n"
        md += "| # | recommendation | because you liked | hit |\n|---:|---|---|:-:|\n"
        hist_cols = ctx.index.cols(h["movieId"].to_numpy())
        for i, m in enumerate(r):
            if m == PAD:
                continue
            because = model.explain(hist_cols, int(ctx.index.cols([m])[0]))
            md += (f"| {i + 1} | {ctx.label[m]} | "
                   f"{ctx.label[int(ctx.index.ids[because])] if because is not None else '—'} | "
                   f"{'✓' if m in nxt else ''} |\n")
        md += "\n"
    (config.RESULTS_DIR / f"{ctx.tag}_examples.md").write_text(md, encoding="utf-8")


def plot_ann(ann: dict, model_key: str, path: Path) -> None:
    """Recall vs p95 latency: one panel per k, one line per M, Flat as the exact reference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]  # fixed order
    rows = ann["rows"]
    Ms = sorted({r["M"] for r in rows if r["index"] == "HNSW"})
    ks = sorted({r["k"] for r in rows})
    fig, axes = plt.subplots(1, len(ks), figsize=(6.2 * len(ks), 4.6), facecolor=surface)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, ks):
        ax.set_facecolor(surface)
        label_M = Ms[len(Ms) // 2] if Ms else None          # label ef values on one line only
        for color, M in zip(series, Ms):
            pts = sorted((r for r in rows if r["index"] == "HNSW" and r["k"] == k and r["M"] == M),
                         key=lambda r: r["ef_search"])
            x, y = [r["p95_ms"] for r in pts], [r["recall"] for r in pts]
            ax.plot(x, y, color=color, lw=2, marker="o", ms=6, label=f"HNSW M={M}",
                    markeredgecolor=surface, markeredgewidth=1.5, zorder=3)
            if M == label_M:                                 # skip labels that would collide
                last_x = None
                for r in pts:
                    near_flat = r["p95_ms"] > 0.6 * next(
                        x["p95_ms"] for x in rows if x["index"] == "Flat" and x["k"] == k)
                    if (last_x is None or r["p95_ms"] / last_x > 1.3) and not near_flat:
                        ax.annotate(f"ef={r['ef_search']}", (r["p95_ms"], r["recall"]),
                                    textcoords="offset points", xytext=(5, -13), fontsize=8,
                                    color=ink2)
                        last_x = r["p95_ms"]
        f = next(r for r in rows if r["index"] == "Flat" and r["k"] == k)
        ax.scatter([f["p95_ms"]], [1.0], marker="*", s=220, color=ink, zorder=4,
                   label="Flat (exact)", edgecolors=surface, linewidths=1.5)
        ax.annotate(f"Flat (exact): {f['p95_ms']:.2f} ms", (f["p95_ms"], 1.0),
                    textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=ink)
        ax.axhline(ann["target_recall"], color=ink2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.text(0.01, ann["target_recall"], f" target recall {ann['target_recall']:.2f}",
                transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=8, color=ink2)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        xs = [r["p95_ms"] for r in rows if r["k"] == k]
        ax.set_xlim(min(xs) / 1.5, max(xs) * 1.8)
        ax.set_xlabel("p95 latency per query, ms (log scale, 1 thread)", color=ink2)
        ax.set_ylabel(f"recall@{k} vs exact search", color=ink2)
        ax.set_title(f"top-{k} {'(a recommendation list)' if k <= 20 else '(stage-1 candidates)'}",
                     color=ink, fontsize=11, loc="left")
        ax.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=ink2)
        lo = min([r["recall"] for r in rows if r["k"] == k] + [ann["target_recall"]])
        ax.set_ylim(max(0.0, lo - 0.03), 1.0 + 0.09 * (1.0 - max(0.0, lo - 0.03)))   # room for the Flat label
    axes[0].legend(frameon=False, fontsize=9, loc="lower right", labelcolor=ink)
    fig.suptitle(f"FAISS Flat vs HNSW — {ann['n_items']:,} movies, {model_key} {ann['dim']}-d, "
                 f"{ann['n_queries']:,} user-profile queries", color=ink, x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def write_md(res: dict, path: Path, ctx: Context) -> None:
    md = "# Day 5 — Warm-user profiles, retrieval, Flat vs HNSW\n\n"
    md += (f"Split fingerprint `{res['split_fingerprint'][:12]}…`. Encoder `{ctx.model_key}`, "
           f"{'mean-centered' if ctx.center else 'raw'} vectors (Day 4 choice). Relevant = rating ≥ 4.\n\n")
    if "test" in res:
        t = res["test"]
        rows = list(t["rows"])
        for r in ctx.day3.get("test", {}).get("rows", []):        # same users, same split
            rows.append({**r, "model": f"{r['model']} (Day 3)"})
        md += f"## Test — fit on train+val, {t['n_users']:,} users\n\n"
        md += md_table(rows, ["model", "config"] + SHOW)
        md += "\n" + md_table(rows, ["model", "coverage@10", "long_tail@10", "diversity@10", "ms_per_user"])
        md += "\n"
    if "tuning" in res:
        tu = res["tuning"]
        md += f"## Profile tuning — validation, {tu['n_users']:,} users (same sample as Days 3–4)\n\n"
        md += md_table(tu["runs"], ["model", "config", "ndcg@10", "recall@10", "mrr@10",
                                    "coverage@10", "long_tail@10", "diversity@10"])
        md += (f"\nDay 4 (liked, no decay) on validation: {tu['day4_val_ndcg@10']}. "
               f"Selected: `{json.dumps(res['selected'])}`. Examples: `{ctx.tag}_examples.md`.\n\n")
    if "candidates" in res:
        c = res["candidates"]
        md += f"## Stage-1 retrieval — validation, {c['n_users']:,} users\n\n"
        ks = config.CANDIDATE_BUDGETS
        md += md_table(c["rows"], ["retriever"] + [f"recall@{k}" for k in ks] + [f"hit_rate@{k}" for k in ks])
        B = c["mix_budget"]
        md += f"\nMixed candidate sets of {B} (embedding top-a, then popular movies not already in):\n\n"
        md += md_table(c["mixes"], ["from_embeddings", "from_popularity", f"recall@{B}",
                                    f"hit_rate@{B}", "avg_unique"])
        md += (f"\nOverlap: {100 * c[f'overlap@{B}']:.1f}% of the embedding top-{B} is also in the "
               f"popularity top-{B}.\n\n")
    if "ann" in res:
        a = res["ann"]
        md += (f"## FAISS Flat vs HNSW — {a['n_items']:,} × {a['dim']}-d, {a['n_queries']:,} user-profile "
               f"queries (latency: {a['n_timed']:,} single queries, {a['threads']} thread)\n\n")
        md += f"![Flat vs HNSW](../figures/{ctx.tag}_flat_vs_hnsw.png)\n\n"
        md += md_table([{**r, "M": r["M"] or "", "ef_search": r["ef_search"] or ""} for r in a["rows"]],
                       ["index", "M", "ef_search", "k", "recall", "p50_ms", "p95_ms", "build_s", "size_mb"])
        for k, r in a["recommended"].items():
            md += (f"\nOperating point top-{k} (fastest with recall ≥ {a['target_recall']}): "
                   f"M={r['M']}, ef_search={r['ef_search']} → recall {r['recall']:.3f}, "
                   f"p95 {r['p95_ms']:.3f} ms, {r['speedup_p95_vs_flat']}× faster than Flat.")
        md += f"\n\nefConstruction {a['ef_construction']}, faiss {a['faiss']}.\n"
    path.write_text(md, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stages", nargs="+", default=["tune", "candidates", "ann", "test"],
                   choices=["tune", "candidates", "ann", "test"])
    p.add_argument("--tune-users", type=int, default=config.TUNE_USERS)
    p.add_argument("--test-users", type=int, default=0, help="0 = all test users")
    p.add_argument("--batch-users", type=int, default=1000)
    p.add_argument("--ann-queries", type=int, default=config.ANN_QUERIES)
    p.add_argument("--latency-queries", type=int, default=1000)
    p.add_argument("--hnsw-m", type=int, nargs="+", default=list(config.HNSW_M))
    p.add_argument("--smoke", action="store_true", help="small run to check everything works")
    p.add_argument("--plot-only", action="store_true",
                   help="redraw the Flat vs HNSW chart from the saved json (no data loading)")
    args = p.parse_args()
    if args.smoke:
        args.tune_users, args.test_users = 3000, 5000
        args.ann_queries, args.latency_queries, args.hnsw_m = 300, 200, [16, 32]
    tag = "day05_retrieval_smoke" if args.smoke else "day05_retrieval"
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json = config.RESULTS_DIR / f"{tag}.json"
    results = json.loads(out_json.read_text()) if out_json.exists() else {}
    t0 = time.time()
    if args.plot_only:
        if "ann" not in results:
            raise SystemExit(f"no 'ann' results in {out_json.name}; run --stages ann first")
        key = results.get("selected", {}).get("model", "bge-small")
        plot_ann(results["ann"], key, config.FIG_DIR / f"{tag}_flat_vs_hnsw.png")
        log(f"redrew reports/figures/{tag}_flat_vs_hnsw.png", t0)
        return
    results["split_fingerprint"] = json.loads(
        (config.SPLIT_DIR / S.MANIFEST_FILE).read_text())["fingerprint"]
    ctx = Context(args, t0)
    ctx.tag = tag

    for stage, fn in (("tune", stage_tune), ("candidates", stage_candidates),
                      ("ann", stage_ann), ("test", stage_test)):
        if stage in args.stages:
            log(f"--- stage {stage} ---", t0)
            fn(ctx, results)
            out_json.write_text(json.dumps(results, indent=2))       # save after every stage

    results["seconds_last_run"] = round(time.time() - t0, 1)
    out_json.write_text(json.dumps(results, indent=2))
    write_md(results, out_json.with_suffix(".md"), ctx)
    log(f"wrote {out_json.relative_to(config.ROOT)} and .md", t0)


if __name__ == "__main__":
    main()
