"""Day 11: MMR diversity re-ranking + "Because you liked X" explanations.

Stages (all by default, in this order):
  tune     validation — the Day 9 ranker-valid users (5k; features already in the Day 8/9 table,
           history = train, labels = val). Ranker scores -> MMR re-ranking of each user's top-50 in
           two similarity spaces (embedding, genre) × λ grid. Rule (config, written before the run):
           per space the most diverse λ within 2% of the ranker's NDCG@10; then the space with the
           larger average diversity gain (genre + semantic). -> results["tune"]["selected"]
  test     ALL warm test users (history = train+val): popularity, the ranker (= λ 1) and every
           (space, λ) -> accuracy @5/10/20 + coverage, long tail, genre diversity, semantic
           diversity, #genres per list. Paired bootstrap of the selected MMR vs the ranker on
           accuracy AND diversity. Check: λ 1 reproduces the Day 10 ranker number.
  explain  500 test users: the served (MMR) list explained with SHAP reason groups + the cited
           movie; how often "Because you liked X" can be shown; distinct cited movies per list;
           faithfulness = remove the cited movie from the history and re-score, vs removing a
           random other recent like (control); 3 worked examples; explanation + MMR latency.

Outputs
  reports/results/day11_diversity.json + .md, day11_examples.md                (commit)
  reports/figures/day11_mmr_tradeoff.png                                       (commit)
  data/runs/day11_test_per_user.parquet

Usage
  python scripts/run_diversity.py --smoke             # ~2 min check (needs Day 8/9 --smoke runs)
  python scripts/run_diversity.py                     # full run (~50-60 min on the 8 GB laptop)
  python scripts/run_diversity.py --stages explain    # one stage (others kept in the json)
  python scripts/run_diversity.py --plot-only         # redraw chart + md from the json
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # works without `pip install -e .`
from recsys.experiment import setup_runtime  # noqa: E402

setup_runtime()          # Windows OpenMP fix; must run before faiss / lightgbm are imported

import argparse  # noqa: E402
import gc  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from recsys import config  # noqa: E402
from recsys import split as S  # noqa: E402
from recsys.diversity import distinct_genres, mmr_rerank, per_user_ild, pick_lambda  # noqa: E402
from recsys.embeddings import center_vectors, l2_normalize  # noqa: E402
from recsys.evaluation import project_file  # noqa: E402
from recsys.experiment import (COLS, SHOW, TCOLS, chosen_settings, eval_users, head_items,  # noqa: E402
                               load_catalog, load_item_vectors, log, md_table, only, open_results,
                               read_results, rows_of, save_results, split_fingerprint)
from recsys.explain import Explainer, counterfactual, liked_history  # noqa: E402
from recsys.features import FEATURE_NAMES  # noqa: E402
from recsys.metrics import PAD, beyond_accuracy, evaluate, paired_bootstrap  # noqa: E402
from recsys.ranker import (build_two_stage, load_rank_data, pad_by_user, predict,  # noqa: E402
                           split_users, top_from_scores)

KS = config.EVAL_KS
KM = max(KS)
DIV_COLS = ["coverage@10", "long_tail@10", "genre_div@10", "sem_div@10", "genres@10"]
SPACE_NAME = {"embedding": "MMR · embedding similarity", "genre": "MMR · genre similarity"}


def sfx(name: str, smoke: bool) -> str:
    if not smoke:
        return name
    stem, _, ext = name.rpartition(".")
    return f"{stem}_smoke.{ext}"


class Context:
    def __init__(self, args, t0: float) -> None:
        self.args, self.t0 = args, t0
        self.cat = load_catalog()
        self.chosen = chosen_settings()
        self.raw = load_item_vectors(self.cat, self.chosen["encoder"])
        self.d9 = read_results(sfx("day09_ranker.json", args.smoke))
        self.d10 = read_results(sfx("day10_final.json", args.smoke))
        if "selected" not in self.d9:
            raise SystemExit("no Day 9 results — run scripts/run_ranker.py first")
        self.sel = self.d9["selected"]
        self.n_pop, self.m_faiss, self.f_m = self.sel["pop"], self.sel["faiss"], self.sel["faiss_feature"]
        # the SAME vectors the ranker's features use (Day 4: mean-centered) -> semantic diversity
        self.V = center_vectors(self.raw) if self.chosen["center"] else l2_normalize(self.raw)
        self.spaces = {"embedding": self.V, "genre": np.asarray(self.cat.genres, dtype=np.float32)}
        self.ts = None
        self.hist = None             # test users' train+val history, kept for the explain stage
        log(f"catalog {len(self.cat.index):,}; ranker {self.sel['label']}, {self.sel['best_iter']} trees; "
            f"MMR pool {config.MMR_POOL}, λ {list(config.MMR_LAMBDAS)}", t0)

    def warm(self):
        import lightgbm as lgb

        path = project_file(self.sel["model_file"])
        if not path.exists():
            raise SystemExit(f"{path.name} missing — run scripts/run_ranker.py first")
        return lgb.Booster(model_file=str(path))

    def two_stage(self, train_full: pd.DataFrame):
        if self.ts is None:
            self.ts = build_two_stage(self.cat, self.raw, self.chosen, train_full, self.n_pop,
                                      self.m_faiss, self.f_m, booster=self.warm())
        return self.ts

    def measure(self, users, R, truth, head) -> tuple[dict, pd.DataFrame]:
        """Accuracy @K + diversity @10 for one recommendation matrix; per-user frame for bootstraps."""
        m, pu = evaluate(users, R, truth, KS, return_per_user=True)
        ba = beyond_accuracy(R, self.cat.index.ids, head, self.cat.genres, k=10)
        g_ild, s_ild, n_g = [], [], []
        for s in range(0, len(R), 5000):              # chunks: (5000, 10, 384) floats at most
            g_ild.append(per_user_ild(R[s:s + 5000], self.cat.index, self.spaces["genre"]))
            s_ild.append(per_user_ild(R[s:s + 5000], self.cat.index, self.V))
            n_g.append(distinct_genres(R[s:s + 5000], self.cat.index, self.cat.genres))
        div = pd.DataFrame({"userId": users, "genre_ild": np.concatenate(g_ild),
                            "sem_ild": np.concatenate(s_ild), "n_genres": np.concatenate(n_g)})
        row = {**{k: round(v, 5) for k, v in m.items() if isinstance(v, float)}, "n_users": m["n_users"],
               "coverage@10": round(ba["coverage@10"], 5), "long_tail@10": round(ba["long_tail@10"], 5),
               "genre_div@10": round(float(np.nanmean(div["genre_ild"])), 5),
               "sem_div@10": round(float(np.nanmean(div["sem_ild"])), 5),
               "genres@10": round(float(div["n_genres"].mean()), 3)}
        return row, pu[["userId"] + SHOW].merge(div, on="userId", how="left")


def div_gain(rows: list[dict]) -> None:
    """Adds 'div_gain' = average relative gain in genre and semantic diversity vs λ 1 (in place)."""
    base = next(r for r in rows if r["lambda"] == 1.0)
    for r in rows:
        r["div_gain"] = round(0.5 * (r["genre_div@10"] / base["genre_div@10"] - 1)
                              + 0.5 * (r["sem_div@10"] / base["sem_div@10"] - 1), 5)


def mmr_all(ctx: Context, C, S_, k: int) -> dict[tuple[str, float], np.ndarray]:
    out = {}
    for space, V in ctx.spaces.items():
        for lam, R in mmr_rerank(C, S_, V, ctx.cat.index, config.MMR_LAMBDAS, k, config.MMR_POOL).items():
            out[(space, lam)] = R.astype(np.int32)
    return out


def select(rows: list[dict]) -> dict:
    """The pre-registered rule: per space the most diverse λ within budget, then the better space."""
    picks = {}
    for space in config.MMR_SPACES:
        sr = [r for r in rows if r.get("space") == space]
        div_gain(sr)
        picks[space] = pick_lambda(sr, config.MMR_MAX_NDCG_LOSS, diversity="div_gain")
    best = max(config.MMR_SPACES, key=lambda s: (picks[s]["div_gain"], -config.MMR_SPACES.index(s)))
    return {"space": best, "lambda": picks[best]["lambda"], "pool": config.MMR_POOL,
            "per_space": {s: {"lambda": p["lambda"], "ndcg@10": p["ndcg@10"], "div_gain": p["div_gain"],
                              "genre_div@10": p["genre_div@10"], "sem_div@10": p["sem_div@10"]}
                          for s, p in picks.items()}}


def mmr_rows(ctx, users, tops, truth, head, per_user=None, keep=()) -> list[dict]:
    rows = []
    for (space, lam), R in tops.items():
        r, pu = ctx.measure(users, R, truth, head)
        rows.append({"model": "ranker (λ 1)" if lam == 1.0 else SPACE_NAME[space], "space": space,
                     "lambda": lam, **r})
        if per_user is not None and (space, lam) in keep:
            per_user[(space, lam)] = pu
    return rows


# --------------------------------------------------------------------------- #
# Stage 1: tune on validation
# --------------------------------------------------------------------------- #
def stage_tune(ctx: Context, results: dict) -> None:
    import pyarrow.parquet as pq

    name = "day08_features_val.parquet" if ctx.sel["candidates"] == "union" else "day09_features_val_pop500.parquet"
    path = config.RUNS_DIR / sfx(name, ctx.args.smoke)
    users = np.unique(pq.read_table(path, columns=["userId"]).column("userId").to_numpy().astype(np.int64))
    _, va_u = split_users(users, config.RANKER_VALID_SHARE, config.SEED + 9)       # = Day 9 split
    d = load_rank_data(path).subset(va_u)
    gc.collect()
    train = S.load_view("train", config.SPLIT_DIR, columns=["userId", "movieId"])
    head = head_items(train["movieId"].to_numpy(), ctx.cat.index)
    del train
    val = S.load_view("val", config.SPLIT_DIR, columns=COLS)
    truth = only(val, va_u)
    del val
    log(f"tune: {len(va_u):,} ranker-valid users (Day 9 split; history = train, labels = val)", ctx.t0)
    score = predict(ctx.warm(), d.X)
    u, Sm = pad_by_user(d.users, score, -np.inf)
    _, Cm = pad_by_user(d.users, d.movies, PAD)
    _, Pm = pad_by_user(d.users, -d.X[:, FEATURE_NAMES.index("pop_rank")].astype(np.float64), -np.inf)
    Pm = np.where(np.isnan(Pm), -np.inf, Pm)
    del d, score
    gc.collect()
    rows = []
    r, _ = ctx.measure(u, top_from_scores(Cm, Pm, KM), truth, head)
    rows.append({"model": "popularity", **r})
    tops: dict = {}
    for s in range(0, len(u), ctx.args.batch_users):
        for key, R in mmr_all(ctx, Cm[s:s + ctx.args.batch_users], Sm[s:s + ctx.args.batch_users], KM).items():
            tops.setdefault(key, []).append(R)
    tops = {k: np.vstack(v) for k, v in tops.items()}
    rows += mmr_rows(ctx, u, tops, truth, head)
    sel = select([r for r in rows if "space" in r])
    for r in rows:
        if "space" in r:
            log(f"  {r['space']:9s} λ {r['lambda']:.2f}  ndcg@10={r['ndcg@10']:.4f}  genre div={r['genre_div@10']:.3f}"
                f"  sem div={r['sem_div@10']:.3f}  genres={r['genres@10']:.2f}  gain={r.get('div_gain', 0):+.3f}", ctx.t0)
    log(f"  selected: {sel['space']} similarity, λ {sel['lambda']} "
        f"(per space: {', '.join(s + ' λ ' + str(p['lambda']) for s, p in sel['per_space'].items())})", ctx.t0)
    results["tune"] = {"split": "val (history = train, labels = val), Day 9 ranker-valid users",
                       "n_users": int(len(u)), "rule": {"max_ndcg_loss": config.MMR_MAX_NDCG_LOSS,
                                                         "pool": config.MMR_POOL,
                                                         "diversity": "mean relative gain of genre + semantic ILD@10"},
                       "rows": rows, "selected": sel}


# --------------------------------------------------------------------------- #
# Stage 2: test
# --------------------------------------------------------------------------- #
def stage_test(ctx: Context, results: dict) -> None:
    a = ctx.args
    if "tune" not in results:
        raise SystemExit("run the 'tune' stage first")
    sel = results["tune"]["selected"]
    train_full = S.load_view("train_full", config.SPLIT_DIR, columns=TCOLS)
    test = S.load_view("test", config.SPLIT_DIR, columns=COLS)
    users = eval_users(test, a.test_users, config.SEED)
    truth = only(test, users)
    del test
    ts = ctx.two_stage(train_full)
    head = head_items(train_full["movieId"].to_numpy(), ctx.cat.index)
    ctx.hist = only(train_full, users)
    del train_full
    gc.collect()
    log(f"test: {len(users):,} warm users; selected {sel['space']} λ {sel['lambda']}", ctx.t0)
    tops: dict = {}
    pop = []
    t_feat = t_mmr = 0.0
    t_start = time.time()
    for s in range(0, len(users), a.batch_users):
        bu = users[s:s + a.batch_users]
        t1 = time.time()
        fb = ts.feature_batch(rows_of(ctx.hist, bu), bu)
        S_ = ts.scores(fb)
        t2 = time.time()
        for key, R in mmr_all(ctx, fb.candidates, S_, KM).items():
            tops.setdefault(key, []).append(R)
        t_mmr += time.time() - t2
        t_feat += t2 - t1
        pop.append(top_from_scores(fb.candidates, ts.score_with(fb, None, []), KM).astype(np.int32))
        if (s // a.batch_users) % 40 == 0:
            done = min(s + a.batch_users, len(users))
            log(f"  {done:,}/{len(users):,} users ({(time.time() - t_start) / done * 1000:.1f} ms/user)", ctx.t0)
        del fb, S_
    tops = {k: np.vstack(v) for k, v in tops.items()}
    rows, per_user = [], {}
    r, per_user["popularity"] = ctx.measure(users, np.vstack(pop), truth, head)
    rows.append({"model": "popularity", **r})
    chosen = (sel["space"], float(sel["lambda"]))
    keep = {("embedding", 1.0), chosen} | {(s, float(p["lambda"])) for s, p in sel["per_space"].items()}
    rows += mmr_rows(ctx, users, tops, truth, head, per_user, keep)
    del tops
    gc.collect()
    for r in rows:
        log(f"  {r['model']:28s} λ {r.get('lambda', ''):>4}  ndcg@10={r['ndcg@10']:.4f}  genre div={r['genre_div@10']:.3f}"
            f"  sem div={r['sem_div@10']:.3f}  genres={r['genres@10']:.2f}  coverage={r['coverage@10']:.3f}", ctx.t0)
    ranker = per_user[("embedding", 1.0)]
    for df in per_user.values():                      # paired = the same users in the same order
        assert np.array_equal(df["userId"].to_numpy(), ranker["userId"].to_numpy())
    sigs = []
    for key in [chosen] + [k for k in keep if k not in (chosen, ("embedding", 1.0))]:
        for metric in ["ndcg@10", "recall@10", "mrr@10", "genre_ild", "sem_ild", "n_genres"]:
            x, y = per_user[key][metric].to_numpy(), ranker[metric].to_numpy()
            ok = ~(np.isnan(x) | np.isnan(y))
            b = paired_bootstrap(x[ok], y[ok], n_boot=2000, seed=config.SEED)
            sigs.append({"model": f"{SPACE_NAME[key[0]]}, λ {key[1]}", "vs": "ranker (λ 1)", "metric": metric,
                         "selected": key == chosen, **{k: v for k, v in b.items()}})
            log(f"    {sigs[-1]['model']:38s} {metric:10s} {100 * b['rel_change']:+6.1f}%  "
                f"CI [{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]", ctx.t0)
    # λ 1 must be exactly the Day 10 ranker (same users, same model, same history)
    d10 = next((x for x in ctx.d10.get("test", {}).get("rows", []) if x.get("key") == "ranker"), None)
    lam1 = next(r for r in rows if r.get("lambda") == 1.0 and r["space"] == "embedding")
    checks = {"ranker_equals_day10": (None if d10 is None or ctx.d10["test"]["n_users"] != len(users)
                                      else bool(abs(lam1["ndcg@10"] - d10["ndcg@10"]) < 5e-5)),
              "ranker_ndcg@10": lam1["ndcg@10"], "day10_ranker_ndcg@10": d10 and d10["ndcg@10"],
              "lambda1_same_in_both_spaces": bool(next(r for r in rows if r.get("lambda") == 1.0
                                                       and r["space"] == "genre")["ndcg@10"] == lam1["ndcg@10"])}
    if checks["ranker_equals_day10"] is not None:
        log(f"  check: λ 1 = Day 10 ranker ({lam1['ndcg@10']:.4f} vs {d10['ndcg@10']:.4f}): "
            f"{'✓' if checks['ranker_equals_day10'] else 'DIFFERENT'}", ctx.t0)
    n = len(users)
    results["test"] = {"split": "test (history = train+val, labels = test), warm users", "n_users": int(n),
                       "selected": sel, "rows": rows, "significance": sigs, "checks": checks,
                       "latency": {"stage1_features_ranker_ms_per_user": round(1000 * t_feat / n, 2),
                                   "mmr_all_20_settings_ms_per_user": round(1000 * t_mmr / n, 3),
                                   "note": f"batches of {a.batch_users} users, CPU; varies run to run"}}
    pd.concat([df.assign(model="popularity" if k == "popularity" else f"{k[0]}_{k[1]}")
               for k, df in per_user.items()]).to_parquet(
        config.RUNS_DIR / sfx("day11_test_per_user.parquet", a.smoke), index=False)


# --------------------------------------------------------------------------- #
# Stage 3: explanations
# --------------------------------------------------------------------------- #
def stage_explain(ctx: Context, results: dict) -> None:
    a = ctx.args
    sel = (results.get("tune") or {}).get("selected")
    if not sel:
        raise SystemExit("run the 'tune' stage first")
    mmr = {"lambda": sel["lambda"], "pool": sel["pool"], "space": sel["space"]}
    test = S.load_view("test", config.SPLIT_DIR, columns=COLS)
    pool_users = eval_users(test, None, config.SEED)
    users = np.sort(np.random.default_rng(config.SEED + 11).choice(
        pool_users, size=min(a.explain_users, len(pool_users)), replace=False))
    truth = only(test, users)
    del test
    if ctx.ts is None or ctx.hist is None:
        train_full = S.load_view("train_full", config.SPLIT_DIR, columns=TCOLS)
        ctx.two_stage(train_full)
        hist = only(train_full, users)
        del train_full
    else:
        hist = only(ctx.hist, users)
    gc.collect()
    ts = ctx.ts
    ex = Explainer.from_two_stage(ts, ctx.cat, config.EXPLAIN_MIN_COS)
    liked_test = set(map(tuple, truth.loc[truth["rating"] >= config.RELEVANCE_THRESHOLD,
                                          ["userId", "movieId"]].to_numpy().tolist()))
    rng = np.random.default_rng(config.SEED + 12)
    recs_all, distinct, cf_parts, examples = [], {"ranker": [], "mmr": []}, [], []
    t_rank = t_mmr = t_exp = 0.0
    log(f"explain: {len(users):,} test users, served list = MMR {sel['space']} λ {sel['lambda']}; "
        f"'Because you liked X' from cos >= {config.EXPLAIN_MIN_COS}", ctx.t0)
    for s in range(0, len(users), 100):
        bu = users[s:s + 100]
        h = rows_of(hist, bu)
        C, ranks = ts.stage1(h, bu)
        fb = ts.builder.transform(bu, h, C, ranks)
        S_ = ts.scores(fb)
        t1 = time.time()
        top_r = top_from_scores(C, S_, 10)
        t2 = time.time()
        top_m = ts.top(fb, S_, 10, mmr)
        t3 = time.time()
        exp_m = ex.explain(fb, h, top_m)
        t4 = time.time()
        exp_r = ex.explain(fb, h, top_r)
        t_rank, t_mmr, t_exp = t_rank + t2 - t1, t_mmr + t3 - t2, t_exp + t4 - t3
        liked = liked_history(h, bu, ctx.cat.index, config.RELEVANCE_THRESHOLD, ts.builder.maxcos_history)
        pairs, ctrl = [], []
        for i, u in enumerate(bu):
            for key, lst in (("mmr", exp_m[i]), ("ranker", exp_r[i])):
                cited = {r["because"] for r in lst if r["because"] is not None}
                distinct[key].append(len(cited))
            for r in exp_m[i]:
                recs_all.append({"userId": int(u), **{k: r[k] for k in ("movieId", "reason", "because", "because_cos")},
                                 "hit": (int(u), r["movieId"]) in liked_test})
                if r["because"] is None:
                    continue
                cited_col = int(ctx.cat.index.cols([r["because"]])[0])
                others = [c for c in liked[i] if c != cited_col]
                if not others:
                    continue
                pairs.append((i, r["movieId"], r["because"]))
                ctrl.append((i, r["movieId"], int(ctx.cat.index.ids[rng.choice(others)])))
        if pairs:
            cf = counterfactual(ts, h, bu, C, ranks, pairs + ctrl)
            n_p = len(pairs)
            both = cf.iloc[:n_p].reset_index(drop=True).add_suffix("_cited").join(
                cf.iloc[n_p:].reset_index(drop=True).add_suffix("_control"))
            cf_parts.append(both)
        if len(examples) < a.example_users:
            for i, u in enumerate(bu):
                if len(examples) >= a.example_users:
                    break
                if not any((int(u), r["movieId"]) in liked_test for r in exp_m[i]):
                    continue
                lk = h[(h["userId"].to_numpy() == u) & (h["rating"].to_numpy() >= config.RELEVANCE_THRESHOLD)]
                g = ctx.cat.genres
                gen = lambda m: "/".join(ctx.cat.vocab[j] for j in np.flatnonzero(g[ctx.cat.index.cols([m])[0]]))  # noqa: E731
                lab = lambda m: ctx.cat.label.get(int(m), str(m))  # noqa: E731
                examples.append({
                    "userId": int(u), "n_ratings": int((h["userId"].to_numpy() == u).sum()),
                    "recent_liked": [lab(m) for m in lk["movieId"].to_numpy()[-5:][::-1]],
                    "ranker": [{"movie": lab(r["movieId"]), "genres": gen(r["movieId"]),
                                "hit": (int(u), r["movieId"]) in liked_test} for r in exp_r[i]],
                    "mmr": [{"movie": lab(r["movieId"]), "genres": gen(r["movieId"]), "text": r["text"],
                             "reason": r["reason"], "hit": (int(u), r["movieId"]) in liked_test}
                            for r in exp_m[i]],
                    "genres_ranker": int(distinct_genres(top_r[i:i + 1], ctx.cat.index, g)[0]),
                    "genres_mmr": int(distinct_genres(top_m[i:i + 1], ctx.cat.index, g)[0])})
        del fb, S_
    recs = pd.DataFrame(recs_all)
    n = len(recs)
    reason_share = {k: round(float(v), 4) for k, v in recs["reason"].value_counts(normalize=True).items()}
    hit_reason = {k: round(float(v), 4) for k, v in recs[recs["hit"]]["reason"].value_counts(normalize=True).items()}
    cos = recs["because_cos"].dropna().to_numpy(np.float64)
    out = {"n_users": int(len(users)), "n_recs": int(n), "min_cos": config.EXPLAIN_MIN_COS,
           "served": f"MMR {sel['space']} λ {sel['lambda']}",
           "reason_share": reason_share, "reason_share_of_hits": hit_reason,
           "because_share": round(float(recs["because"].notna().mean()), 4),
           "best_cos_quantiles": {f"p{q}": round(float(np.percentile(cos, q)), 3) for q in (10, 25, 50, 75, 90)}
           if cos.size else {},
           "distinct_cited_per_list": {k: round(float(np.mean(v)), 2) for k, v in distinct.items()},
           "latency_ms_per_user": {"ranker_topk": round(1000 * t_rank / len(users), 3),
                                   "mmr": round(1000 * t_mmr / len(users), 3),
                                   "explain_shap_10": round(1000 * t_exp / len(users), 2),
                                   "note": "on top of stage 1 + features + ranker; CPU, varies"},
           "examples": examples}
    if cf_parts:
        cf = pd.concat(cf_parts, ignore_index=True)
        drop_c = (cf["score_before_cited"] - cf["score_after_cited"]).to_numpy()
        drop_r = (cf["score_before_control"] - cf["score_after_control"]).to_numpy()
        b = paired_bootstrap(drop_c, drop_r, n_boot=2000, seed=config.SEED)
        out["counterfactual"] = {
            "n_pairs": int(len(cf)),
            "cited": {"mean_score_drop": round(float(drop_c.mean()), 4),
                      "share_score_down": round(float((drop_c > 0).mean()), 4),
                      "mean_rank_change": round(float((cf["rank_after_cited"] - cf["rank_before_cited"]).mean()), 2),
                      "share_out_of_top10": round(float(((cf["rank_before_cited"] <= 10)
                                                         & (cf["rank_after_cited"] > 10)).mean()), 4)},
            "control": {"mean_score_drop": round(float(drop_r.mean()), 4),
                        "share_score_down": round(float((drop_r > 0).mean()), 4),
                        "mean_rank_change": round(float((cf["rank_after_control"] - cf["rank_before_control"]).mean()), 2),
                        "share_out_of_top10": round(float(((cf["rank_before_control"] <= 10)
                                                           & (cf["rank_after_control"] > 10)).mean()), 4)},
            "cited_minus_control": {"diff": round(b["diff"], 4), "ci95": [round(x, 4) for x in b["ci95"]],
                                    "p_le_0": b["p_le_0"]},
            "note": "stage 1 fixed; all stage-2 features recomputed from the history without that movie"}
        c = out["counterfactual"]
        log(f"  counterfactual ({c['n_pairs']:,} recs): remove cited -> score drop {c['cited']['mean_score_drop']:.3f}, "
            f"rank {c['cited']['mean_rank_change']:+.1f}; random other like -> {c['control']['mean_score_drop']:.3f}, "
            f"rank {c['control']['mean_rank_change']:+.1f}; diff CI {c['cited_minus_control']['ci95']}", ctx.t0)
    log(f"  reasons: {reason_share}; 'because' shown for {100 * out['because_share']:.0f}% of recs; "
        f"distinct cited movies per list: ranker {out['distinct_cited_per_list']['ranker']}, "
        f"MMR {out['distinct_cited_per_list']['mmr']}", ctx.t0)
    results["explain"] = out


# --------------------------------------------------------------------------- #
# Chart + markdown
# --------------------------------------------------------------------------- #
BLUE, ORANGE, GRAY, INK, INK2, GRID = "#2a78d6", "#eb6834", "#9a9993", "#0b0b0b", "#52514e", "#ecebe8"


def plot(res: dict, smoke: bool) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "test" not in res:
        return []
    t = res["test"]
    sel = t["selected"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
    for ax, xkey, xlabel in ((axes[0], "genre_div@10", "genre diversity@10 (1 − mean genre cosine)"),
                             (axes[1], "sem_div@10", "semantic diversity@10 (1 − mean embedding cosine)")):
        for space, col, mk in (("embedding", BLUE, "o"), ("genre", ORANGE, "s")):
            pts = sorted((r["lambda"], r[xkey], r["ndcg@10"]) for r in t["rows"] if r.get("space") == space)
            ax.plot([p[1] for p in pts], [p[2] for p in pts], color=col, linewidth=2, marker=mk, markersize=6,
                    markeredgecolor="white", markeredgewidth=1.2, label=SPACE_NAME[space], zorder=2)
            lo = pts[0]
            ax.annotate(f"λ {lo[0]:g}", (lo[1], lo[2]), textcoords="offset points", xytext=(4, -10),
                        fontsize=7, color=INK2)
        r1 = next(r for r in t["rows"] if r.get("lambda") == 1.0)
        ax.annotate("ranker (λ 1)", (r1[xkey], r1["ndcg@10"]), textcoords="offset points", xytext=(-2, 9),
                    fontsize=7, color=INK2, ha="left")
        s = next(r for r in t["rows"] if r.get("space") == sel["space"] and r["lambda"] == sel["lambda"])
        ax.scatter([s[xkey]], [s["ndcg@10"]], s=170, facecolors="none", edgecolors=INK, linewidths=1.4, zorder=3)
        ax.annotate(f"selected on val\n{sel['space']} λ {sel['lambda']:g}", (s[xkey], s["ndcg@10"]),
                    textcoords="offset points", xytext=(8, -26), fontsize=7, color=INK)
        p = next(r for r in t["rows"] if r["model"] == "popularity")
        ax.scatter([p[xkey]], [p["ndcg@10"]], s=42, color=GRAY, edgecolors="white", linewidths=1.2, zorder=3)
        ax.annotate("popularity", (p[xkey], p["ndcg@10"]), textcoords="offset points", xytext=(5, 4),
                    fontsize=7, color=INK2)
        ax.set_xlabel(xlabel, color=INK2, fontsize=8)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        for sp_ in ("left", "bottom"):
            ax.spines[sp_].set_color("#c3c2b7")
        ax.tick_params(colors=INK2, labelsize=8)
    axes[0].set_ylabel("NDCG@10 (test)", color=INK2, fontsize=9)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(f"Accuracy vs diversity — MMR on the ranker's top-{config.MMR_POOL}, "
                 f"{t['n_users']:,} warm test users", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    out = config.FIG_DIR / ("day11_mmr_tradeoff%s.png" % ("_smoke" if smoke else ""))
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return [out]


def fmt(s: dict) -> dict:
    return {**s, "change": f"{100 * s['rel_change']:+.1f}%", "diff": round(s["diff"], 5),
            "95% CI of diff": f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]",
            "p_le_0": round(s["p_le_0"], 4)}


def write_md(res: dict, path: Path, figs: list[Path]) -> None:
    acc = ["ndcg@5", "ndcg@10", "ndcg@20", "recall@10", "mrr@10", "hit_rate@10"]
    md = "# Day 11 — Diversity re-ranking (MMR) + explanations\n\n"
    md += (f"Split fingerprint `{res.get('split_fingerprint', '')[:12]}…`. MMR re-ranks the ranker's top-"
           f"{config.MMR_POOL}: next = argmax λ·rel − (1−λ)·max sim to the movies already picked "
           "(rel = ranker score min-max scaled per user). λ 1 = the ranker. Genre diversity = 1 − mean pairwise "
           "cosine of genre vectors in the top-10 (the Day 3 metric); semantic diversity = the same with the "
           "mean-centered bge vectors.\n\n")
    if "tune" in res:
        tu = res["tune"]
        sel = tu["selected"]
        md += (f"## Choice on validation ({tu['n_users']:,} Day 9 ranker-valid users)\n\n"
               f"Rule (written before the run): per space, the λ with the largest average diversity gain "
               f"(genre + semantic) whose NDCG@10 is within {100 * tu['rule']['max_ndcg_loss']:.0f}% of the ranker; "
               f"then the space with the larger gain. **Selected: {sel['space']} similarity, λ {sel['lambda']}.**\n\n")
        md += md_table([r for r in tu["rows"]], ["model", "lambda", "ndcg@10", "recall@10"] + DIV_COLS + ["div_gain"],
                       left=("model",)) + "\n"
    if "test" in res:
        t = res["test"]
        md += f"## Test ({t['n_users']:,} warm users, history = train+val)\n\n"
        main = [r for r in t["rows"] if r["model"] == "popularity" or r.get("lambda") in
                (1.0, t["selected"]["lambda"]) or any(r.get("space") == s and r["lambda"] == p["lambda"]
                                                       for s, p in t["selected"]["per_space"].items())]
        main = [r for i, r in enumerate(main) if not (r.get("lambda") == 1.0 and r.get("space") == "genre")]
        md += "Key rows:\n\n" + md_table(main, ["model", "lambda"] + acc, left=("model",)) + "\n"
        md += md_table(main, ["model", "lambda"] + DIV_COLS, left=("model",)) + "\n"
        md += ("Paired bootstrap vs the ranker (2,000 resamples of users; `ild` = per-user diversity, "
               "`n_genres` = distinct genres in the top-10):\n\n")
        md += md_table([fmt(s) for s in t["significance"]],
                       ["model", "metric", "change", "diff", "95% CI of diff", "p_le_0"], left=("model", "metric")) + "\n"
        c = t["checks"]
        if c.get("ranker_equals_day10") is not None:
            md += f"Check: λ 1 = the Day 10 ranker ({c['ranker_ndcg@10']:.4f}): {'✓' if c['ranker_equals_day10'] else '✗'}\n\n"
        lat = t["latency"]
        md += (f"Cost: MMR for all 20 settings {lat['mmr_all_20_settings_ms_per_user']} ms/user on top of "
               f"{lat['stage1_features_ranker_ms_per_user']} ms/user for stage 1 + features + ranker ({lat['note']}).\n\n")
        md += "<details><summary>Full λ grid (test)</summary>\n\n" + md_table(
            t["rows"], ["model", "lambda", "ndcg@10", "recall@10"] + DIV_COLS, left=("model",)) + "\n</details>\n\n"
        for f in figs:
            md += f"![trade-off](../figures/{f.name})\n\n"
    if "explain" in res:
        e = res["explain"]
        md += (f"## Explanations ({e['n_users']:,} test users, {e['n_recs']:,} served recommendations, "
               f"{e['served']})\n\n")
        md += md_table([{"reason": k, "share of recs": v, "share of hits": e["reason_share_of_hits"].get(k, 0.0)}
                        for k, v in e["reason_share"].items()], ["reason", "share of recs", "share of hits"],
                       left=("reason",)) + "\n"
        md += (f"A cited movie (\"Because you liked X\", cos ≥ {e['min_cos']}) is available for "
               f"{100 * e['because_share']:.0f}% of recommendations. Best-cosine quantiles: "
               f"{', '.join(f'{k} {v}' for k, v in e['best_cos_quantiles'].items())}. Distinct cited movies per "
               f"top-10: ranker {e['distinct_cited_per_list']['ranker']}, MMR {e['distinct_cited_per_list']['mmr']}.\n\n")
        if "counterfactual" in e:
            c = e["counterfactual"]
            md += (f"**Faithfulness** ({c['n_pairs']:,} recommendations; {c['note']}):\n\n"
                   "| remove from history | mean score drop | score went down | mean rank change | left the top-10 |\n"
                   "|---|---:|---:|---:|---:|\n")
            for k, lab in (("cited", "the cited movie X"), ("control", "a random other recent like")):
                v = c[k]
                md += (f"| {lab} | {v['mean_score_drop']:.4f} | {100 * v['share_score_down']:.0f}% | "
                       f"{v['mean_rank_change']:+.1f} | {100 * v['share_out_of_top10']:.0f}% |\n")
            d = c["cited_minus_control"]
            md += f"\nCited − control score drop: {d['diff']:+.4f}, 95% CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}].\n\n"
        lat = e["latency_ms_per_user"]
        md += (f"Serving cost per user: MMR {lat['mmr']} ms, explanations (TreeSHAP on 10 rows + cited movie) "
               f"{lat['explain_shap_10']} ms ({lat['note']}). Worked examples: `day11_examples.md`.\n")
    path.write_text(md, encoding="utf-8")


def write_examples(res: dict, path: Path) -> None:
    e = res.get("explain", {})
    md = "# Day 11 — worked examples (test users)\n\n"
    md += ("Ranker top-10 vs the served MMR top-10 with its explanation. ✓ = the user liked it in the test "
           "window. Users chosen as the first sampled users with at least one hit in the served list.\n\n")
    for ex in e.get("examples", []):
        md += (f"## User {ex['userId']} ({ex['n_ratings']} ratings)\n\nRecently liked: "
               f"{'; '.join(ex['recent_liked'])}\n\nGenres covered: ranker {ex['genres_ranker']} → MMR "
               f"{ex['genres_mmr']}\n\n| # | ranker | | MMR (served) | | why |\n|---:|---|---|---|---|---|\n")
        for i, (r, m) in enumerate(zip(ex["ranker"], ex["mmr"]), 1):
            md += (f"| {i} | {r['movie']} · {r['genres']} | {'✓' if r['hit'] else ''} | {m['movie']} · {m['genres']} "
                   f"| {'✓' if m['hit'] else ''} | {m['text']} |\n")
        md += "\n"
    path.write_text(md, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stages", nargs="+", default=["tune", "test", "explain"], choices=["tune", "test", "explain"])
    p.add_argument("--batch-users", type=int, default=500)
    p.add_argument("--test-users", type=int, default=None, help="cap warm test users (default: all)")
    p.add_argument("--explain-users", type=int, default=config.EXPLAIN_USERS)
    p.add_argument("--example-users", type=int, default=config.EXAMPLE_USERS)
    p.add_argument("--smoke", action="store_true", help="small run on the Day 8/9 smoke files")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.test_users, args.explain_users = args.test_users or 3000, min(args.explain_users, 200)
    tag = "day11_diversity_smoke" if args.smoke else "day11_diversity"
    out_json, results = open_results(tag)
    t0 = time.time()
    if not args.plot_only:
        results["split_fingerprint"] = split_fingerprint()
        ctx = Context(args, t0)
        for stage, fn in (("tune", stage_tune), ("test", stage_test), ("explain", stage_explain)):
            if stage in args.stages:
                log(f"--- stage {stage} ---", t0)
                fn(ctx, results)
                save_results(out_json, results)
        results["seconds_last_run"] = round(time.time() - t0, 1)
        save_results(out_json, results)
    figs = plot(results, args.smoke)
    write_md(results, out_json.with_suffix(".md"), figs)
    write_examples(results, out_json.parent / ("day11_examples%s.md" % ("_smoke" if args.smoke else "")))
    log(f"wrote {out_json.relative_to(config.ROOT)}, .md, examples and {len(figs)} figure(s)", t0)


if __name__ == "__main__":
    main()
