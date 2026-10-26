"""Day 6: cold-start users — onboarding seeds and/or a preference text, zero-shot.

Model (recsys.coldstart): pool = the M movies most liked in the W days before the user's onboarding
moment, then rank the pool by  cos(profile, movie) + λ · that time-aware prior.

Uses the Day 4 encoder + mean-centering and the Day 5 profile (graded, half-life) from their json
files and the cached item vectors in data/embeddings/ — no movie is re-encoded. Only short user
texts are encoded (the encoder is loaded from the local Hugging Face cache filled on Day 4).

Stages (all by default):
  tune      PSEUDO-COLD VALIDATION: warm validation users (fit on train) shown only their first n
            liked training movies (n = 1 / 3 / 5) and/or a simulated genre questionnaire (the 3 most
            frequent genres of their first 5 likes, as a sentence).
              1. time window W of the popularity prior (no onboarding content at all)
              2. per setting: pool size M x blend weight λ (and query prefix on/off for text)
  test      COLD TEST: the 5,000 users held out of ALL training data on Day 2, fit on train+val.
            NDCG@10 vs number of seeds (0/1/3/5): popularity (the CV baseline), time-aware popularity,
            TF-IDF (Day 3), embeddings only, semantic rerank (ours); text onboarding; text + seeds.
            Paired bootstrap vs popularity AND vs time-aware popularity (= what semantics adds).
            Chart: reports/figures/day06_cold_start_ndcg_vs_seeds.png
  examples  free-text demo queries + two real cold users with "because you liked X";
            single-request latency (encode + pool + rerank)

Outputs
  reports/results/day06_coldstart.json + .md, day06_coldstart_examples.md   (commit)
  reports/figures/day06_cold_start_ndcg_vs_seeds.png                        (commit)
  data/runs/day06_coldstart_cold_per_user.parquet                           (git-ignored)

Usage
  python scripts/run_coldstart.py --smoke          # ~3-5 min check
  python scripts/run_coldstart.py                  # full run
  python scripts/run_coldstart.py --stages examples
  python scripts/run_coldstart.py --plot-only      # redraw the chart from the json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # works without `pip install -e .`
from recsys.experiment import setup_runtime  # noqa: E402

setup_runtime()          # Windows OpenMP + SSL fixes; must run before torch / faiss are imported

import argparse  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

from recsys import config  # noqa: E402
from recsys import split as S  # noqa: E402
from recsys.baselines import PopularityRecommender, popularity_scores, user_item_matrix  # noqa: E402
from recsys.coldstart import (ColdStartRecommender, TimePopularity, combine_profiles,  # noqa: E402
                              first_liked, genre_match, genre_query, onboarding_genres,
                              onboarding_time, pool_rerank_topk, popularity_prior,
                              query_profiles)
from recsys.embeddings import EmbeddingRecommender, encode_texts, load_encoder  # noqa: E402
from recsys.experiment import (COLS, KMAX, TCOLS, chosen_settings, eval_users,  # noqa: E402
                               head_items, load_catalog, load_item_vectors, log, md_table, only,
                               open_results, read_results, row, save_results, split_fingerprint)
from recsys.metrics import beyond_accuracy, evaluate, paired_bootstrap  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SHOW = ["ndcg@5", "ndcg@10", "ndcg@20", "recall@10", "mrr@10", "hit_rate@10"]
OURS = "semantic rerank (ours)"


def cfg_name(m, w) -> str:
    return f"pool={'all' if m is None else m}, λ={w:g}"


def window_name(w) -> str:
    if w == "all-time":
        return "all-time"
    return "all before onboarding" if w is None else f"{w} days"


class Context:
    """Catalog, vectors, chosen settings from Days 3-5; encoder loaded only when needed."""

    def __init__(self, args, t0: float) -> None:
        self.args, self.t0 = args, t0
        cat = load_catalog()
        self.index, self.items, self.label = cat.index, cat.items, cat.label
        self.genres, self.vocab = cat.genres, cat.vocab
        chosen = chosen_settings()          # Day 3 popularity, Day 4 encoder, Day 5 profile
        self.day3 = read_results("day03_baselines.json")
        self.model_key, self.center = chosen["encoder"], chosen["center"]
        self.profile = chosen["profile"]
        self.pop_mode = chosen["popularity_mode"]
        self.raw = load_item_vectors(cat, self.model_key)
        self.emb = EmbeddingRecommender(self.raw, self.index, threshold=config.RELEVANCE_THRESHOLD,
                                        center=self.center, **self.profile)
        self.V = self.emb.item_vectors                        # centered (or normalised) item vectors
        self.prefix = config.EMB_MODELS[self.model_key]["query_prefix"]
        self._encoder = None
        self._warm = None
        self._cold = None
        log(f"catalog {len(self.index):,} movies; {self.model_key} "
            f"({'mean-centered' if self.center else 'raw'}); profile {self.profile}; "
            f"popularity mode={self.pop_mode}", t0)

    # ------------------------------------------------------------------ #
    def encoder(self):
        if self._encoder is None:
            self._encoder = load_encoder(config.EMB_MODELS[self.model_key]["hf_name"])
            log(f"loaded encoder {config.EMB_MODELS[self.model_key]['hf_name']}", self.t0)
        return self._encoder

    def encode_queries(self, texts: list[str], prefix: str) -> np.ndarray:
        """Unique texts are encoded once; '' (no genres) -> zero row -> prior only."""
        uniq = sorted({t for t in texts if t})
        out = np.zeros((len(texts), self.V.shape[1]), dtype=np.float32)
        if uniq:
            Q = encode_texts(self.encoder(), uniq, batch_size=config.EMB_BATCH_SIZE, prefix=prefix,
                             show_progress=False)
            Q = query_profiles(Q, self.raw, self.center)
            pos = {t: i for i, t in enumerate(uniq)}
            for r, t in enumerate(texts):
                if t:
                    out[r] = Q[pos[t]]
        return out

    def chosen_genres(self, seeds: pd.DataFrame, users: np.ndarray) -> list[list[str]]:
        """Simulated questionnaire per user from their (up to 5) seeds in time order."""
        cols = dict(tuple(seeds.groupby("userId", sort=False)["movieId"]))
        return [onboarding_genres(self.index.cols(cols[u].to_numpy()) if u in cols else [],
                                  self.genres, self.vocab, config.ONBOARDING_GENRES) for u in users]

    def setup(self, fit: pd.DataFrame, users: np.ndarray, truth: pd.DataFrame,
              seeds: pd.DataFrame) -> dict:
        """Shared state of one evaluation population (pseudo-cold val or cold test).
        fit: the training ratings (movieId, rating, timestamp) — other users only matter here."""
        tp = TimePopularity.from_ratings(fit, self.index, config.RELEVANCE_THRESHOLD,
                                         config.TIME_BUCKET_DAYS)
        fallback = popularity_scores(fit, self.index, self.pop_mode, config.RELEVANCE_THRESHOLD)
        head = head_items(fit["movieId"].to_numpy(), self.index)
        pop = PopularityRecommender(self.pop_mode, config.RELEVANCE_THRESHOLD)
        pop.index, pop.score = self.index, fallback
        chosen = self.chosen_genres(seeds, users)
        return {"users": users, "truth": truth, "seeds": seeds, "tp": tp,
                "prior_all": popularity_prior(fit, self.index, config.RELEVANCE_THRESHOLD),
                "t": onboarding_time(seeds, users), "fallback": fallback, "pop": pop, "head": head,
                "chosen": chosen, "queries": [genre_query(g) for g in chosen]}

    def warm(self) -> dict:
        """Pseudo-cold validation: warm val users (>= 5 liked train movies), fit on TRAIN."""
        if self._warm is None:
            val = S.load_view("val", config.SPLIT_DIR, columns=COLS)
            train = S.load_view("train", config.SPLIT_DIR, columns=TCOLS)
            cand = eval_users(val, None, config.SEED)
            liked = train.loc[train["rating"].to_numpy() >= config.RELEVANCE_THRESHOLD, "userId"]
            n_liked = liked.value_counts()
            del liked
            cand = cand[np.isin(cand, n_liked.index[n_liked >= config.N_COLD_SEEDS].to_numpy())]
            n = min(self.args.tune_users, len(cand))
            users = np.sort(np.random.default_rng(config.SEED + 6).choice(cand, size=n, replace=False))
            seeds = first_liked(only(train, users), config.N_COLD_SEEDS, config.RELEVANCE_THRESHOLD)
            self._warm = self.setup(train.drop(columns="userId"), users, only(val, users), seeds)
            del train
            gc.collect()
            log(f"pseudo-cold validation: {len(users):,} warm val users, first {config.N_COLD_SEEDS} "
                "liked train movies as seeds", self.t0)
        return self._warm

    def cold(self) -> dict:
        """The real cold-start users (never in training), fit on TRAIN + VAL."""
        if self._cold is None:
            seeds = S.load_cold_seeds(config.SPLIT_DIR).sort_values(["userId", "seed_rank"])
            truth = S.load_view("cold_test", config.SPLIT_DIR, columns=COLS)
            users = eval_users(truth, self.args.test_users or None, config.SEED)
            fit = S.load_view("train_full", config.SPLIT_DIR, columns=["movieId", "rating", "timestamp"])
            self._cold = self.setup(fit, users, only(truth, users), only(seeds, users))
            del fit
            gc.collect()
            log(f"cold test: {len(users):,} users never seen in training", self.t0)
        return self._cold

    # ------------------------------------------------------------------ #
    def seed_profiles(self, d: dict, n: int):
        s = d["seeds"][d["seeds"]["seed_rank"] <= n]
        seen = user_item_matrix(s, d["users"], self.index, dtype=np.int8)
        return self.emb.user_profiles(s, d["users"]), seen

    def prior(self, d: dict, window):
        if window == "all-time":
            return d["prior_all"]                 # the same for everyone (= popularity ranking)
        return lambda b: d["tp"].prior(d["t"][b], window)

    def run(self, d: dict, content, seen, window, pools, weights) -> dict:
        return pool_rerank_topk(content, len(d["users"]), self.prior(d, window), pools, weights,
                                seen, KMAX, self.index, d["fallback"], self.args.batch_users)

    def cos(self, P: np.ndarray):
        return lambda b: P[b] @ self.V.T

    def genres_of(self, d: dict):
        return lambda b: genre_match(d["chosen"][b], self.genres, self.vocab)

    def zero(self, d: dict):
        return lambda b: np.zeros((b.stop - b.start, len(self.index)), dtype=np.float32)


# --------------------------------------------------------------------------- #
def stage_tune(ctx: Context, results: dict) -> None:
    d = ctx.warm()
    runs: list[dict] = []
    none = user_item_matrix(d["seeds"].iloc[:0], d["users"], ctx.index, dtype=np.int8)

    def score(recs) -> dict:
        return evaluate(d["users"], recs, d["truth"], config.EVAL_KS)

    # 1. time window of the prior — no onboarding content at all
    m = score(d["pop"].recommend(none, KMAX))
    runs.append(row("popularity (all-time)", m, {"config": "train, incl. ratings after onboarding"}))
    log(f"  popularity (all-time)                ndcg@10={m['ndcg@10']:.4f}", ctx.t0)
    win_scores = []
    for w in config.TIME_WINDOWS_DAYS:
        recs = ctx.run(d, ctx.zero(d), none, w, [None], [1.0])[(None, 1.0)]
        m = score(recs)
        runs.append(row("time-aware popularity", m, {"config": window_name(w)}))
        win_scores.append((m["ndcg@10"], w))
        log(f"  time-aware popularity {window_name(w):22s} ndcg@10={m['ndcg@10']:.4f}", ctx.t0)
    best_nd, W = max(win_scores, key=lambda x: (x[0], x[1] == "all-time"))   # tie -> simpler
    log(f"  -> window {window_name(W)}", ctx.t0)

    # 2. pool size x blend weight per setting, with that window
    best: dict = {}
    pools, weights = list(config.POOL_SIZES), list(config.BLEND_WEIGHTS)

    def sweep(setting: str, grid: dict, extra: str = "") -> None:
        table = {}
        for (mm, w), recs in grid.items():
            nd = score(recs)
            runs.append(row(setting, nd, {"config": (extra + ", " if extra else "") + cfg_name(mm, w),
                                          "pool": mm, "lambda": w}))
            table[(mm, w)] = nd["ndcg@10"]
        (mm, w), nd = max(table.items(), key=lambda kv: (kv[1], kv[0][1]))   # tie -> larger λ
        if setting not in best or nd > best[setting]["ndcg@10"]:
            best[setting] = {"pool": mm, "lambda": w, "ndcg@10": nd, "config": extra}
        per_pool = "  ".join(
            f"{'all' if p is None else p}:{max(v for (q, _), v in table.items() if q == p):.4f}"
            for p in pools)
        log(f"  {setting:13s} {extra:9s} best per pool -> {per_pool}   pick {cfg_name(mm, w)}", ctx.t0)

    for n in config.COLD_SEED_COUNTS:
        if n:
            P, seen = ctx.seed_profiles(d, n)
            sweep(f"seeds_{n}", ctx.run(d, ctx.cos(P), seen, W, pools, weights))
    Qs = {}
    for use_prefix in ((True, False) if ctx.prefix else (False,)):
        Qs[use_prefix] = ctx.encode_queries(d["queries"], ctx.prefix if use_prefix else "")
        sweep("text", ctx.run(d, ctx.cos(Qs[use_prefix]), none, W, pools, weights),
              "prefix" if use_prefix else "no-prefix")
    use_prefix = best["text"]["config"] == "prefix"
    sweep("genre_pop", ctx.run(d, ctx.genres_of(d), none, W, pools, weights))
    for n in config.TEXT_SEED_COUNTS:
        P, seen = ctx.seed_profiles(d, n)
        sweep(f"text+seeds_{n}", ctx.run(d, ctx.cos(combine_profiles(Qs[use_prefix], P)), seen, W,
                                         pools, weights))

    results["tuning"] = {"population": "pseudo-cold validation (warm val users, first n liked train "
                                       "movies only; fit on train)",
                         "n_users": int(len(d["users"])), "runs": runs, "best": best,
                         "time_window": {"days": W, "ndcg@10": best_nd},
                         "example_queries": sorted(set(d["queries"]))[:5]}
    results["selected"] = {"model": ctx.model_key, "center": ctx.center, **ctx.profile,
                           "query_prefix": use_prefix, "time_window_days": W,
                           "blend": {k: {"pool": v["pool"], "lambda": v["lambda"]}
                                     for k, v in best.items() if k != "genre_pop"},
                           "genre_pop": {"pool": best["genre_pop"]["pool"],
                                         "lambda": best["genre_pop"]["lambda"]}}
    log(f"selected: {results['selected']}", ctx.t0)


def selected(results: dict) -> dict:
    if "selected" not in results or "time_window_days" not in results["selected"]:
        raise SystemExit("no tuned settings — run the 'tune' stage first")
    return results["selected"]


def stage_test(ctx: Context, results: dict) -> None:
    sel = selected(results)
    d = ctx.cold()
    W = sel["time_window_days"]
    rows: list[dict] = []
    per_user: dict[str, pd.DataFrame] = {}

    def add(model: str, setting: str, n: int, recs: np.ndarray, cfg: str = "—") -> None:
        m, pu = evaluate(d["users"], recs, d["truth"], config.EVAL_KS, return_per_user=True)
        ba = beyond_accuracy(recs, ctx.index.ids, d["head"], ctx.genres, k=10)
        rows.append(row(model, m, {**ba, "setting": setting, "n_seeds": n, "config": cfg}))
        per_user[f"{model}|{setting}"] = pu
        log(f"  {setting:13s} {model:26s} ndcg@10={m['ndcg@10']:.4f} recall@10={m['recall@10']:.4f} "
            f"coverage@10={ba['coverage@10']:.3f}  {cfg}", ctx.t0)

    def tuned(setting: str) -> tuple:
        b = sel["blend"][setting] if setting in sel["blend"] else sel[setting]
        return b["pool"], b["lambda"]

    for n in config.COLD_SEED_COUNTS:
        P, seen = ctx.seed_profiles(d, n)
        add("popularity", f"seeds_{n}", n, d["pop"].recommend(seen, KMAX), "all-time (Day 3 baseline)")
        if W != "all-time":               # all-time prior ranks exactly like popularity
            add("time-aware popularity", f"seeds_{n}", n,
                ctx.run(d, ctx.zero(d), seen, W, [None], [1.0])[(None, 1.0)], window_name(W))
        if n == 0:
            continue                       # no profile: every semantic variant IS the prior
        mm, lam = tuned(f"seeds_{n}")
        g = ctx.run(d, ctx.cos(P), seen, W, sorted({None, mm}, key=lambda x: (x is None, x or 0)),
                    sorted({0.0, lam}))
        add("embeddings only", f"seeds_{n}", n, g[(None, 0.0)], cfg_name(None, 0))
        add(OURS, f"seeds_{n}", n, g[(mm, lam)], cfg_name(mm, lam))
    none = user_item_matrix(d["seeds"].iloc[:0], d["users"], ctx.index, dtype=np.int8)
    Q = ctx.encode_queries(d["queries"], ctx.prefix if sel["query_prefix"] else "")
    mm, lam = tuned("text")
    g = ctx.run(d, ctx.cos(Q), none, W, sorted({None, mm}, key=lambda x: (x is None, x or 0)),
                sorted({0.0, lam}))
    add("text only", "text", 0, g[(None, 0.0)], cfg_name(None, 0))
    add(OURS, "text", 0, g[(mm, lam)], cfg_name(mm, lam))
    gm, gl = tuned("genre_pop")
    add("popular in your genres", "text", 0, ctx.run(d, ctx.genres_of(d), none, W, [gm], [gl])[(gm, gl)],
        cfg_name(gm, gl))
    for n in config.TEXT_SEED_COUNTS:
        P, seen = ctx.seed_profiles(d, n)
        mm, lam = tuned(f"text+seeds_{n}")
        add(OURS, f"text+seeds_{n}", n,
            ctx.run(d, ctx.cos(combine_profiles(Q, P)), seen, W, [mm], [lam])[(mm, lam)], cfg_name(mm, lam))

    # Day 3 TF-IDF on the same users (copied, not re-run: same split, same seeds)
    tfidf = {r["n_seeds"]: r for r in ctx.day3.get("cold_start", {}).get("rows", [])
             if r["model"] == "tfidf_knn"}

    # Paired bootstrap over the same users. vs popularity = the CV claim;
    # vs time-aware popularity = what the SEMANTIC part adds on top of the better prior.
    sig = []
    pairs = ([(OURS, f"seeds_{n}", f"seeds_{n}") for n in config.COLD_SEED_COUNTS if n] +
             [(OURS, "text", "seeds_0"), ("popular in your genres", "text", "seeds_0")] +
             [(OURS, f"text+seeds_{n}", f"seeds_{n}") for n in config.TEXT_SEED_COUNTS])
    for model, setting, base in pairs:
        a = per_user[f"{model}|{setting}"]["ndcg@10"].to_numpy()
        for base_model in ("popularity",) + (("time-aware popularity",) if W != "all-time" else ()):
            b = per_user[f"{base_model}|{base}"]["ndcg@10"].to_numpy()
            r = paired_bootstrap(a, b, n_boot=2000, seed=config.SEED)
            sig.append({"model": model, "setting": setting, "vs": f"{base_model} ({base})",
                        **{k: (round(v, 5) if isinstance(v, float) else [round(x, 5) for x in v])
                           for k, v in r.items()}})
            log(f"  {model} [{setting}] vs {base_model}: {100 * r['rel_change']:+.1f}%  "
                f"95% CI of diff [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]  p={r['p_le_0']:.3f}", ctx.t0)
    results["test"] = {"population": "cold_test — users never in training, fit on train+val",
                       "n_users": int(len(d["users"])), "time_window_days": W, "rows": rows,
                       "significance": sig,
                       "tfidf_day3": {str(k): {m: v.get(m) for m in ("ndcg@10", "recall@10", "mrr@10")}
                                      for k, v in tfidf.items()}}
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([df.assign(model=k) for k, df in per_user.items()]).to_parquet(
        config.RUNS_DIR / f"{ctx.tag}_cold_per_user.parquet", index=False)


def stage_examples(ctx: Context, results: dict) -> None:
    sel = selected(results)
    d = ctx.cold()
    W = sel["time_window_days"]
    now = d["tp"].now()

    def prior_at(t: int) -> np.ndarray:
        return d["prior_all"] if W == "all-time" else d["tp"].prior([t], W)[0]

    rec = ColdStartRecommender(ctx.raw, ctx.index, prior_at(now), d["fallback"],
                               blend=sel["blend"], center=ctx.center,
                               weighting=ctx.profile["weighting"], half_life=ctx.profile["half_life"],
                               threshold=config.RELEVANCE_THRESHOLD,
                               query_prefix=ctx.prefix if sel["query_prefix"] else "",
                               encoder=ctx.encoder())
    md = f"# Day 6 — cold-start examples ({ctx.model_key})\n\n"
    pool_desc = ("most-liked movies overall" if W == "all-time"
                 else f"most-liked movies in the {window_name(W)} before onboarding")
    md += (f"Pool = {pool_desc}; ranked by cos(profile, movie) + λ·prior. "
           f"Accuracy-tuned settings: `{json.dumps(sel['blend'])}`\n\n")
    explore = ColdStartRecommender(ctx.raw, ctx.index, prior_at(now), d["fallback"],
                                   blend={**sel["blend"], "text": dict(config.TEXT_EXPLORE)},
                                   center=ctx.center, threshold=config.RELEVANCE_THRESHOLD,
                                   query_prefix=ctx.prefix if sel["query_prefix"] else "",
                                   encoder=ctx.encoder())
    md += ("## Free text — explore mode (semantic search, not accuracy-tuned)\n\n"
           f"The accuracy-tuned text setting is λ = {sel['blend']['text']['lambda']:g} (text added no "
           "measurable NDCG), which returns the same popular list for every sentence. Explore mode = "
           f"pool of the {config.TEXT_EXPLORE['pool']:,} most-liked movies ranked by "
           f"cos(text, movie) + {config.TEXT_EXPLORE['lambda']:g}·prior.\n\n")
    for q in config.DEMO_QUERIES:
        out = explore.recommend(text=q, k=10)
        md += f"**“{q}”** → " + "; ".join(ctx.label[o["movieId"]] for o in out) + "\n\n"

    rng = np.random.default_rng(config.SEED + 66)
    picked = np.sort(rng.choice(d["users"], size=min(2, len(d["users"])), replace=False))
    md += ("## Real cold-start users (never in training)\n\nPrior at each user's own onboarding "
           "moment. ✓ = rated ≥ 4 among their last 10 ratings.\n\n")
    for u in picked:
        s = d["seeds"][d["seeds"]["userId"] == u].sort_values("seed_rank")
        seed_ids, seed_ratings = s["movieId"].tolist(), s["rating"].tolist()
        i = int(np.searchsorted(d["users"], u))
        pr = prior_at(int(d["t"][i]))
        liked = set(d["truth"].loc[(d["truth"]["userId"] == u) &
                                   (d["truth"]["rating"] >= config.RELEVANCE_THRESHOLD), "movieId"])
        md += (f"### User {u} — onboarded {pd.to_datetime(d['t'][i], unit='s').date()}\n\n"
               "Onboarding picks (first 5 likes): " + "; ".join(ctx.label[m] for m in seed_ids))
        md += f"\n\nSimulated questionnaire: *{d['queries'][i] or '—'}*\n\n"
        md += "| # | 5 seeds | because you liked | ✓ | questionnaire text only | ✓ |\n"
        md += "|---:|---|---|:-:|---|:-:|\n"
        a = rec.recommend(seed_ids=seed_ids, seed_ratings=seed_ratings, k=10, prior=pr)
        b = rec.recommend(text=d["queries"][i], k=10, exclude=seed_ids, prior=pr)
        for j in range(10):
            x = a[j] if j < len(a) else None
            y = b[j] if j < len(b) else None
            md += (f"| {j + 1} | {ctx.label[x['movieId']] if x else ''} | "
                   f"{ctx.label[x['because']] if x and x['because'] else '—'} | "
                   f"{'✓' if x and x['movieId'] in liked else ''} | "
                   f"{ctx.label[y['movieId']] if y else ''} | {'✓' if y and y['movieId'] in liked else ''} |\n")
        md += "\n"
    (config.RESULTS_DIR / f"{ctx.tag}_examples.md").write_text(md, encoding="utf-8")

    # latency of one onboarding request (what the Day 12 API will do), 1 request at a time
    texts = (config.DEMO_QUERIES * 25)[:ctx.args.latency_queries]
    rec.recommend(text=texts[0], k=10)                       # warm-up (first call is slow)
    lat_text, lat_seed = [], []
    for t in texts:
        t1 = time.perf_counter()
        rec.recommend(text=t, k=10)
        lat_text.append(1000 * (time.perf_counter() - t1))
    for u in d["users"][:200]:
        sl = d["seeds"][d["seeds"]["userId"] == u]["movieId"].tolist()
        t1 = time.perf_counter()
        rec.recommend(seed_ids=sl, k=10)
        lat_seed.append(1000 * (time.perf_counter() - t1))
    results["latency_ms"] = {
        "text (encode + pool + rerank)": {"p50": round(float(np.percentile(lat_text, 50)), 2),
                                          "p95": round(float(np.percentile(lat_text, 95)), 2),
                                          "n": len(lat_text)},
        "5 seeds (profile + pool + rerank)": {"p50": round(float(np.percentile(lat_seed, 50)), 2),
                                              "p95": round(float(np.percentile(lat_seed, 95)), 2),
                                              "n": len(lat_seed)}}
    log(f"  latency: {results['latency_ms']}", ctx.t0)


# --------------------------------------------------------------------------- #
def plot_seeds(res: dict, path: Path, model_key: str) -> None:
    """NDCG@10 vs number of onboarding seeds, one line per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    series = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]   # Day 5 order
    t = res["test"]
    rows = t["rows"]

    def pts(model: str, prefix: str = "seeds_") -> tuple[list, list]:
        r = sorted((x for x in rows if x["model"] == model and x["setting"].startswith(prefix)),
                   key=lambda x: x["n_seeds"])
        return [x["n_seeds"] for x in r], [x["ndcg@10"] for x in r]

    pop_x, pop_y = pts("popularity")
    tp_x, tp_y = pts("time-aware popularity")
    lines = [("popularity, all-time (baseline)", pop_x, pop_y, ink2, "--", 1.8)]
    if tp_x:
        lines.append((f"time-aware popularity ({window_name(t['time_window_days'])})", tp_x, tp_y,
                      series[3], "--", 1.8))
    else:
        tp_y = pop_y
    tf = t.get("tfidf_day3", {})
    if tf:
        xs = sorted(int(k) for k in tf)
        lines.append(("TF-IDF content (Day 3)", xs, [tf[str(k)]["ndcg@10"] for k in xs], series[4], "-", 1.6))
    x, y = pts("embeddings only")
    lines.append(("embeddings only (whole catalog)", [0] + x, [pop_y[0]] + y, series[1], "-", 1.6))
    x, y = pts(OURS)
    lines.append(("seeds → semantic rerank (ours)", [0] + x, [tp_y[0]] + y, series[0], "-", 2.4))
    tx = next(r for r in rows if r["model"] == OURS and r["setting"] == "text")
    x, y = pts(OURS, "text+seeds_")
    lines.append(("3-genre text + seeds → semantic rerank (x=0: text only)", [0] + x, [tx["ndcg@10"]] + y,
                  series[2], "-", 1.8))

    fig, ax = plt.subplots(figsize=(7.8, 5.8), facecolor=surface)
    ax.set_facecolor(surface)
    for name, xs, ys, color, ls, lw in lines:
        ax.plot(xs, ys, color=color, lw=lw, ls=ls, marker="o", ms=6, markeredgecolor=surface,
                markeredgewidth=1.5, label=name, zorder=3)
    gp = next(r for r in rows if r["model"] == "popular in your genres")
    ax.scatter([0], [gp["ndcg@10"]], marker="D", s=46, color=series[5], zorder=4,
               edgecolors=surface, linewidths=1.2, label="popular in your 3 genres (no encoder)")
    ax.set_xticks([0] + [n for n in config.COLD_SEED_COUNTS if n])
    ax.set_xlabel("onboarding seeds (liked movies the new user picked)", color=ink2)
    ax.set_ylabel("NDCG@10 on the user's last 10 ratings", color=ink2)
    ax.set_title(f"Cold start — {t['n_users']:,} users never seen in training ({model_key})",
                 color=ink, fontsize=11.5, loc="left")
    ax.grid(True, color=grid, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(0, None)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(grid)
    ax.tick_params(colors=ink2)
    ax.legend(frameon=False, fontsize=8.3, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2,
              labelcolor=ink)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)


def write_md(res: dict, path: Path) -> None:
    md = "# Day 6 — Cold start (onboarding seeds and/or text)\n\n"
    md += f"Split fingerprint `{res['split_fingerprint'][:12]}…`. Relevant = rating ≥ 4. "
    md += ("Model: pool = M most-liked movies in the W days before onboarding, ranked by "
           "cos(profile, movie) + λ·prior. W, M, λ tuned on pseudo-cold validation.\n\n")
    if "test" in res:
        t = res["test"]
        md += (f"## Cold test — {t['n_users']:,} users never seen in training (fit on train+val), "
               f"window {window_name(t['time_window_days'])}\n\n")
        md += md_table(t["rows"], ["setting", "model", "config"] + SHOW)
        md += "\n" + md_table(t["rows"], ["setting", "model", "coverage@10", "long_tail@10", "diversity@10"])
        if t.get("tfidf_day3"):
            md += "\nTF-IDF (Day 3, same users and seeds), NDCG@10: " + ", ".join(
                f"{k} seeds {v['ndcg@10']:.4f}" for k, v in sorted(t["tfidf_day3"].items())) + "\n"
        md += "\n### Paired bootstrap (NDCG@10, 2,000 resamples of users)\n\n"
        md += "| model | setting | vs | NDCG@10 | baseline | change | 95% CI of diff | p(≤0) |\n"
        md += "|---|---|---|---:|---:|---:|---|---:|\n"
        for s in t["significance"]:
            md += (f"| {s['model']} | {s['setting']} | {s['vs']} | {s['mean_a']:.4f} | {s['mean_b']:.4f} | "
                   f"{100 * s['rel_change']:+.1f}% | [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] | "
                   f"{s['p_le_0']:.3f} |\n")
        md += "\n"
    if "latency_ms" in res:
        md += "## One onboarding request (1 at a time, CPU)\n\n| path | p50 ms | p95 ms |\n|---|---:|---:|\n"
        for k, v in res["latency_ms"].items():
            md += f"| {k} | {v['p50']} | {v['p95']} |\n"
        md += "\n"
    if "tuning" in res:
        tu = res["tuning"]
        md += f"## Tuning — {tu['population']}, {tu['n_users']:,} users\n\n"
        md += f"Time window: **{window_name(tu['time_window']['days'])}**\n\n"
        md += md_table([r for r in tu["runs"] if "popularity" in r["model"]], ["model", "config", "ndcg@10", "recall@10"])
        md += "\n| setting | best | NDCG@10 |\n|---|---|---:|\n"
        for k, v in tu["best"].items():
            md += f"| {k} | {(v['config'] + ', ') if v['config'] else ''}{cfg_name(v['pool'], v['lambda'])} | {v['ndcg@10']:.4f} |\n"
        md += "\nExample questionnaire texts: " + "; ".join(f"*{q}*" for q in tu["example_queries"]) + "\n\n"
        md += f"Selected: `{json.dumps(res['selected'])}` (full grid in the json)\n"
    path.write_text(md, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stages", nargs="+", default=["tune", "test", "examples"],
                   choices=["tune", "test", "examples"])
    p.add_argument("--tune-users", type=int, default=config.COLD_TUNE_USERS)
    p.add_argument("--test-users", type=int, default=0, help="0 = all cold-start users")
    p.add_argument("--batch-users", type=int, default=500)
    p.add_argument("--latency-queries", type=int, default=200)
    p.add_argument("--smoke", action="store_true", help="small run to check everything works")
    p.add_argument("--plot-only", action="store_true", help="redraw the chart from the saved json")
    args = p.parse_args()
    if args.smoke:
        args.tune_users, args.test_users, args.latency_queries = 2000, 1000, 30
    tag = "day06_coldstart_smoke" if args.smoke else "day06_coldstart"
    out_json, results = open_results(tag)
    fig_path = config.FIG_DIR / f"{tag.replace('coldstart', 'cold_start')}_ndcg_vs_seeds.png"
    t0 = time.time()
    if args.plot_only:
        if "test" not in results:
            raise SystemExit(f"no 'test' results in {out_json.name}; run --stages test first")
        plot_seeds(results, fig_path, results.get("selected", {}).get("model", "bge-small"))
        log(f"redrew {fig_path.relative_to(config.ROOT)}", t0)
        return
    results["split_fingerprint"] = split_fingerprint()
    ctx = Context(args, t0)
    ctx.tag = tag
    for stage, fn in (("tune", stage_tune), ("test", stage_test), ("examples", stage_examples)):
        if stage in args.stages:
            log(f"--- stage {stage} ---", t0)
            fn(ctx, results)
            save_results(out_json, results)       # save after every stage
    results["seconds_last_run"] = round(time.time() - t0, 1)
    save_results(out_json, results)
    write_md(results, out_json.with_suffix(".md"))
    if "test" in results:
        plot_seeds(results, fig_path, ctx.model_key)
    log(f"wrote {out_json.relative_to(config.ROOT)}, .md and {fig_path.relative_to(config.ROOT)}", t0)


if __name__ == "__main__":
    main()
