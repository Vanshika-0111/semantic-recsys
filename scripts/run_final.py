"""Day 10: the final numbers — warm test users, ablations, real cold-start users, SHAP.

Every setting was chosen on validation (Days 3-9). This script trains only the ABLATION rankers
(on the Day 9 validation table, same user split, same config) and then measures on test once.

Stages (all by default, in this order):
  ablation  retrain the Day 9 ranker on the same 15k ranker-train users, same config, with a
            feature set removed:
              no_semantic  union candidates, without the 5 semantic features
              no_semantics popular-500 candidates (no FAISS candidates) + no semantic features
                           = the ranker with NO embeddings anywhere
              no_year      union candidates, without item_year / year_gap
            -> data/runs/day10_ranker_<name>.txt. Validation table vs the Day 9 ranker (bootstrap).
  test      ALL warm test users (history = train+val, labels = test), K = 5/10/20:
              popularity (Day 3 baseline), release-date-aware popularity, the ranker (ours),
              ranker + release filter, the three ablations; TF-IDF and embedding rows from
              Days 3/5 (same users) -> paired bootstrap of everything vs the ranker / popularity.
              Diagnostics: how often popularity recommends movies released after the user's last
              rating, and how many relevant test movies are such movies.
  cold      the 5,000 real cold-start users (never in training), 0/1/3/5 onboarding picks:
              popularity, Day 6 semantic rerank, cold ranker, warm ranker, and the policy
              (0 -> popularity, 1-2 -> cold ranker, >= 3 -> warm ranker)
  shap      exact TreeSHAP of the ranker on 1,000 test users' candidates: mean |SHAP| per
            feature and group (all candidates, and the top-10 actually shown), two worked examples

Outputs
  reports/results/day10_final.json + .md                                          (commit)
  reports/figures/day10_test_ndcg.png, day10_cold_test.png, day10_shap.png         (commit)
  data/runs/day10_ranker_*.txt, day10_test_per_user.parquet, day10_cold_per_user.parquet

Usage
  python scripts/run_final.py --smoke             # ~3-5 min check (needs Day 8/9 --smoke runs)
  python scripts/run_final.py                     # full run (~45-60 min on the 8 GB laptop)
  python scripts/run_final.py --stages cold shap  # some stages (the others are kept in the json)
  python scripts/run_final.py --plot-only         # redraw charts + md from the json
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
from recsys.baselines import popularity_scores, user_item_matrix  # noqa: E402
from recsys.coldstart import TimePopularity, onboarding_time, pool_rerank_topk, popularity_prior  # noqa: E402
from recsys.evaluation import (FEATURE_GROUPS, SEMANTIC_FEATURES, YEAR_FEATURES, compare,  # noqa: E402
                               group_shares, policy_model, project_file, release_aware_topk,
                               released_after, shap_table, significant, top_reasons, tree_shap,
                               user_cutoff_year, without)
from recsys.experiment import (COLS, SHOW, TCOLS, chosen_settings, eval_users, head_items,  # noqa: E402
                               load_catalog, load_item_vectors, log, md_table, only, open_results,
                               read_results, row, rows_of, save_results, split_fingerprint)
from recsys.features import FEATURE_NAMES  # noqa: E402
from recsys.metrics import PAD, beyond_accuracy, evaluate  # noqa: E402
from recsys.ranker import (RankData, build_two_stage, predict, split_users,  # noqa: E402
                           top_from_scores, topk_by_score, train_ranker)

KS = config.EVAL_KS
KM = max(KS)
PER_USER_COLS = ["userId"] + SHOW
OURS = "two-stage ranker (ours)"
NAMES = {   # model key -> display name
    "popularity": "popularity (Day 3 baseline)",
    "release_pop": "release-aware popularity",
    "ranker": OURS,
    "ranker_release": "ranker + release filter",
    "no_semantic": "ranker w/o semantic features",
    "no_semantics": "ranker w/o any semantics (pop-500, no semantic features)",
    "no_year": "ranker w/o release-year features",
    "tfidf": "TF-IDF item-kNN (Day 3)",
    "embeddings": "bge-small embeddings (Day 5)",
}
ABLATIONS = {   # name -> (candidate variant, features)
    "no_semantic": ("union", without(SEMANTIC_FEATURES)),
    "no_semantics": ("pop500", without(SEMANTIC_FEATURES)),
    "no_year": ("union", without(YEAR_FEATURES)),
}
COLD_NAMES = {"popularity": "popularity", "day6": "Day 6 semantic rerank", "cold": "cold ranker",
              "warm": "warm ranker", "policy": "policy (ours)"}


def sfx(name: str, smoke: bool) -> str:
    if not smoke:
        return name
    stem, dot, ext = name.rpartition(".")
    return f"{stem}_smoke.{ext}"


class Context:
    def __init__(self, args, t0: float) -> None:
        self.args, self.t0 = args, t0
        self.cat = load_catalog()
        self.chosen = chosen_settings()
        self.raw = load_item_vectors(self.cat, self.chosen["encoder"])
        sm = args.smoke
        self.d3 = read_results(sfx("day03_baselines.json", sm))
        self.d5 = read_results(sfx("day05_retrieval.json", sm))
        self.d6 = read_results("day06_coldstart.json") or read_results("day06_coldstart_smoke.json")
        self.d8 = read_results(sfx("day08_candidates.json", sm))
        self.d9 = read_results(sfx("day09_ranker.json", sm))
        if "selected" not in self.d9 or "cold" not in self.d9:
            raise SystemExit("no Day 9 results — run scripts/run_ranker.py first")
        self.sel = self.d9["selected"]
        self.n_pop, self.m_faiss, self.f_m = self.sel["pop"], self.sel["faiss"], self.sel["faiss_feature"]
        self.variants = {"union": (self.n_pop, self.m_faiss, self.f_m),
                         "pop500": (config.RANKER_POP_ONLY, 0, self.f_m)}
        self.years = pd.to_numeric(self.cat.items["year"], errors="coerce").to_numpy(np.float64)
        log(f"catalog {len(self.cat.index):,}; ranker {self.sel['label']}, "
            f"{self.sel['config']['labels']} labels / {self.sel['config']['num_leaves']} leaves, "
            f"{self.sel['best_iter']} trees", t0)

    def run_file(self, name: str) -> Path:
        return config.RUNS_DIR / sfx(name, self.args.smoke)

    def booster(self, path: Path):
        import lightgbm as lgb

        if not path.exists():
            raise SystemExit(f"{path.name} missing — run the stage that writes it first")
        return lgb.Booster(model_file=str(path))

    def warm(self):
        return self.booster(project_file(self.sel["model_file"]))

    def cold_booster(self):
        return self.booster(project_file(self.d9["cold"]["model_file"]))

    def ablation(self, name: str):
        return self.booster(self.run_file(f"day10_ranker_{name}.txt"))


def load_rank_data(path: Path) -> RankData:
    import pyarrow.parquet as pq

    if not path.exists():
        raise SystemExit(f"{path.name} missing — run scripts/run_candidates.py / run_ranker.py first")
    tbl = pq.read_table(path)
    X = np.empty((tbl.num_rows, len(FEATURE_NAMES)), dtype=np.float32)
    for j, f in enumerate(FEATURE_NAMES):
        X[:, j] = tbl.column(f).to_numpy()
    d = RankData(X, tbl.column("rating").to_numpy().astype(np.float32),
                 tbl.column("userId").to_numpy().astype(np.int64),
                 tbl.column("movieId").to_numpy().astype(np.int64), list(FEATURE_NAMES))
    del tbl
    gc.collect()
    return d


def sig(name: str, vs: str, a: pd.DataFrame, b: pd.DataFrame, metric: str = "ndcg@10") -> dict:
    r = compare(a, b, metric, n_boot=2000, seed=config.SEED)
    return {"model": name, "vs": vs, **r}


def fmt_sig(s: dict) -> dict:
    return {**s, "change": f"{100 * s['rel_change']:+.1f}%",
            "95% CI of diff": f"[{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]",
            "significant": "yes" if significant(s) else "no"}


# --------------------------------------------------------------------------- #
# Stage 1: ablation rankers (validation only)
# --------------------------------------------------------------------------- #
def stage_ablation(ctx: Context, results: dict) -> None:
    import pyarrow.parquet as pq

    a = ctx.args
    files = {"union": ctx.run_file("day08_features_val.parquet"),
             "pop500": ctx.run_file("day09_features_val_pop500.parquet")}
    users = np.unique(pq.read_table(files["union"], columns=["userId"]).column("userId").to_numpy().astype(np.int64))
    tr_u, va_u = split_users(users, config.RANKER_VALID_SHARE, config.SEED + 9)   # = Day 9 split
    val = S.load_view("val", config.SPLIT_DIR, columns=COLS)
    truth = only(val, va_u)
    del val
    cfg = ctx.sel["config"]
    log(f"ablation: {len(tr_u):,} ranker-train / {len(va_u):,} ranker-valid users (Day 9 split), "
        f"config {cfg['labels']} labels / {cfg['num_leaves']} leaves", ctx.t0)
    rows, per_user, models = [], {}, {}
    full = ctx.warm()
    for variant in ("union", "pop500"):
        d = load_rank_data(files[variant])
        va = d.subset(va_u)
        with_pos = np.unique(d.users[d.rating >= config.RELEVANCE_THRESHOLD])
        tr = d.subset(np.intersect1d(tr_u, with_pos))
        del d
        gc.collect()
        if variant == "union":
            pr = va.X[:, FEATURE_NAMES.index("pop_rank")].astype(np.float64)
            for key, s in (("popularity", -pr), ("ranker", predict(full, va.X))):
                u, top = topk_by_score(va.users, va.movies, s, KM)
                m, pu = evaluate(u, top, truth, KS, return_per_user=True)
                rows.append(row(NAMES[key], m, {"candidates": ctx.sel["label"], "features": len(FEATURE_NAMES)}))
                per_user[key] = pu
                log(f"  {NAMES[key]:58s} ndcg@10={m['ndcg@10']:.4f}", ctx.t0)
        for name, (cand, feats) in ABLATIONS.items():
            if cand != variant:
                continue
            trc, vac = tr.columns(feats), va.columns(feats)
            booster, info = train_ranker(trc, vac, cfg, config.SEED, a.max_rounds, a.early_stop)
            path = ctx.run_file(f"day10_ranker_{name}.txt")
            booster.save_model(str(path), num_iteration=info["best_iter"])
            u, top = topk_by_score(vac.users, vac.movies, predict(booster, vac.X), KM)
            m, pu = evaluate(u, top, truth, KS, return_per_user=True)
            label = (f"popular {config.RANKER_POP_ONLY}" if cand == "pop500" else ctx.sel["label"])
            rows.append(row(NAMES[name], m, {"candidates": label, "features": len(feats),
                                             "best_iter": info["best_iter"]}))
            per_user[name] = pu
            models[name] = {"candidates": cand, "variant": list(ctx.variants[cand]), "features": feats,
                            "dropped": [f for f in FEATURE_NAMES if f not in feats],
                            "best_iter": info["best_iter"],
                            "model_file": path.relative_to(config.ROOT).as_posix()}
            log(f"  {NAMES[name]:58s} ndcg@10={m['ndcg@10']:.4f}  ({info['best_iter']} trees, "
                f"{info['seconds']:.0f}s)", ctx.t0)
            del trc, vac, booster
            gc.collect()
        del tr, va
        gc.collect()
    sigs = [sig(NAMES["ranker"], NAMES[k], per_user["ranker"], per_user[k]) for k in ABLATIONS]
    sigs += [sig(NAMES[k], NAMES["popularity"], per_user[k], per_user["popularity"]) for k in ABLATIONS]
    for s in sigs:
        log(f"  {s['model']} vs {s['vs']}: {100 * s['rel_change']:+.1f}% p={s['p_le_0']:.3f}", ctx.t0)
    results["ablation"] = {"split": "val (history = train, labels = val), Day 9 ranker-valid users",
                           "n_train_users": int(len(tr_u)), "n_valid_users": int(len(va_u)),
                           "config": cfg, "rows": rows, "significance": sigs, "models": models}


# --------------------------------------------------------------------------- #
# Stage 2: warm test users
# --------------------------------------------------------------------------- #
def stage_test(ctx: Context, results: dict) -> None:
    a = ctx.args
    if "ablation" not in results:
        raise SystemExit("run the 'ablation' stage first")
    train_full = S.load_view("train_full", config.SPLIT_DIR, columns=TCOLS)
    test = S.load_view("test", config.SPLIT_DIR, columns=COLS)
    users = eval_users(test, a.test_users, config.SEED)
    truth = only(test, users)
    del test
    ts = build_two_stage(ctx.cat, ctx.raw, ctx.chosen, train_full, ctx.n_pop, ctx.m_faiss, ctx.f_m,
                         booster=ctx.warm())
    head = head_items(train_full["movieId"].to_numpy(), ctx.cat.index)
    hist = only(train_full, users)
    del train_full
    gc.collect()
    cutoff = user_cutoff_year(hist, users)
    ablation = {n: (ctx.ablation(n), m["features"], m["candidates"])
                for n, m in results["ablation"]["models"].items()}
    log(f"test: {len(users):,} warm users, history = train+val; models: popularity, release-aware "
        f"popularity, ranker (+ release filter), {len(ablation)} ablations", ctx.t0)

    keys = ["popularity", "release_pop", "ranker", "ranker_release"] + list(ablation)
    tops: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    t_feat = t_pred = 0.0
    late_pop_slots = late_slots = 0
    t_start = time.time()
    for s in range(0, len(users), a.batch_users):
        bu = users[s:s + a.batch_users]
        h = rows_of(hist, bu)
        t1 = time.time()
        fbs = ts.feature_batches(h, bu, ctx.variants)
        t2 = time.time()
        fb = fbs["union"]
        S_r = ts.scores(fb)
        t3 = time.time()
        t_feat += t2 - t1
        t_pred += t3 - t2
        cut = cutoff[s:s + len(bu)]
        late = released_after(fb.candidates, ctx.cat.index, ctx.years, cut)
        tops["popularity"].append(top_from_scores(fb.candidates, ts.score_with(fb, None, []), KM))
        tops["ranker"].append(top_from_scores(fb.candidates, S_r, KM))
        tops["ranker_release"].append(top_from_scores(fb.candidates, np.where(late, -np.inf, S_r), KM))
        seen = user_item_matrix(h, bu, ctx.cat.index, dtype=np.int8)
        tops["release_pop"].append(release_aware_topk(ts.pop.score, ctx.cat.index, seen, ctx.years, cut,
                                                      KM, config.RELEASE_POOL))
        for name, (booster, feats, cand) in ablation.items():
            f = fbs[cand]
            tops[name].append(top_from_scores(f.candidates, ts.score_with(f, booster, feats), KM))
        p10 = tops["popularity"][-1][:, :10]
        late_pop_slots += int(released_after(p10, ctx.cat.index, ctx.years, cut).sum())
        late_slots += int((p10 != PAD).sum())
        if (s // a.batch_users) % 40 == 0:
            done = min(s + a.batch_users, len(users))
            log(f"  {done:,}/{len(users):,} users ({(time.time() - t_start) / done * 1000:.1f} ms/user)", ctx.t0)
        del fbs, fb, S_r, late, seen
    n = len(users)
    rows, per_user = [], {}
    for k in keys:
        R = np.vstack(tops.pop(k))
        m, pu = evaluate(users, R, truth, KS, return_per_user=True)
        ba = beyond_accuracy(R, ctx.cat.index.ids, head, ctx.cat.genres, k=10)
        rows.append(row(NAMES[k], m, {**ba, "key": k}))
        per_user[k] = pu[PER_USER_COLS]
        log(f"  {NAMES[k]:58s} ndcg@10={m['ndcg@10']:.4f} recall@10={m['recall@10']:.4f} "
            f"coverage@10={ba['coverage@10']:.3f}", ctx.t0)
        del R
    rows[keys.index("ranker")]["ms_per_user"] = round(1000 * (t_feat + t_pred) / n, 2)

    # relevant test movies released after the user's last training rating (the most a release
    # filter can cost) and how often plain popularity shows such movies
    rel = truth[truth["rating"].to_numpy() >= config.RELEVANCE_THRESHOLD]
    pos = np.searchsorted(users, rel["userId"].to_numpy())
    ry = ctx.years[ctx.cat.index.cols(rel["movieId"].to_numpy())]
    with np.errstate(invalid="ignore"):
        rel_late = float(np.mean(ry > cutoff[pos]))
    diag = {"popularity_top10_slots_released_after_cutoff": round(late_pop_slots / max(late_slots, 1), 4),
            "relevant_test_items_released_after_cutoff": round(rel_late, 4),
            "cutoff": "year of the user's last train+val rating"}
    log(f"  diagnostics: popularity top-10 slots released after the user's last rating "
        f"{100 * diag['popularity_top10_slots_released_after_cutoff']:.1f}%; relevant test movies "
        f"released after it {100 * rel_late:.1f}%", ctx.t0)

    # Day 3 / Day 5 content baselines: same split, same users -> rows + per-user files
    extra = {"tfidf": (ctx.d3, "tfidf_knn", "day03_tfidf_knn%s_test_per_user.parquet" % ("_smoke" if ctx.args.smoke else "")),
             "embeddings": (ctx.d5, "(day5 profile)", "day05_day5_" + ctx.chosen["encoder"] + "_test_per_user.parquet")}
    for k, (res, match, fname) in extra.items():
        r = next((dict(x) for x in res.get("test", {}).get("rows", []) if match in x["model"]), None)
        p = config.RUNS_DIR / fname
        if r is None or not p.exists():
            log(f"  {NAMES[k]}: results or {fname} missing — skipped", ctx.t0)
            continue
        pu = pd.read_parquet(p, columns=PER_USER_COLS)
        pu = pu[np.isin(pu["userId"].to_numpy(), users)]
        if len(pu) != n:
            log(f"  {NAMES[k]}: per-user file covers {len(pu):,} of {n:,} users — compared on those", ctx.t0)
        if res.get("test", {}).get("n_users") == n:
            rows.append({**r, "model": NAMES[k], "key": k, "from": f"Day {3 if k == 'tfidf' else 5} json"})
        else:   # a different user sample (e.g. smoke) -> recompute the row from the per-user file
            rows.append({"model": NAMES[k], "key": k, "n_users": len(pu),
                         **{c: round(float(pu[c].mean()), 5) for c in SHOW}})
        per_user[k] = pu

    pop10 = rows[0]["ndcg@10"]
    d3pop = next((x["ndcg@10"] for x in ctx.d3.get("test", {}).get("rows", []) if x["model"] == "popularity"), None)
    checks = {"popularity_equals_day3": (None if d3pop is None or ctx.d3["test"]["n_users"] != n
                                         else bool(abs(pop10 - d3pop) < 5e-5)),
              "popularity_ndcg@10": pop10, "day3_popularity_ndcg@10": d3pop}
    if checks["popularity_equals_day3"] is not None:
        log(f"  check: popularity through the two-stage path = Day 3 baseline "
            f"({pop10:.4f} vs {d3pop:.4f}): {'✓' if checks['popularity_equals_day3'] else 'DIFFERENT'}", ctx.t0)

    R_ = per_user["ranker"]
    sigs = [sig(OURS, NAMES[k], R_, per_user[k]) for k in
            ["popularity", "release_pop", "tfidf", "embeddings", *ablation, "ranker_release"] if k in per_user]
    sigs += [sig(NAMES[k], NAMES["popularity"], per_user[k], per_user["popularity"])
             for k in ["release_pop", *ablation] if k in per_user]
    sigs.append(sig(NAMES["no_year"], NAMES["release_pop"], per_user["no_year"], per_user["release_pop"])
                if "no_year" in per_user else None)
    sigs = [s for s in sigs if s]
    by_metric = [sig(OURS, NAMES["popularity"], R_, per_user["popularity"], m) for m in SHOW]
    for s in sigs:
        log(f"  {s['model'][:45]:45s} vs {s['vs'][:40]:40s} {100 * s['rel_change']:+7.1f}%  "
            f"CI [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}] p={s['p_le_0']:.3f}", ctx.t0)
    results["test"] = {"split": "test (history = train+val, labels = test), all warm users",
                       "n_users": int(n), "rows": rows, "significance": sigs,
                       "ranker_vs_popularity_by_metric": by_metric, "diagnostics": diag, "checks": checks,
                       "latency": {"stage1_features_ms_per_user": round(1000 * t_feat / n, 2),
                                   "ranker_predict_ms_per_user": round(1000 * t_pred / n, 2),
                                   "note": "batches of %d users, CPU; varies run to run" % a.batch_users}}
    pd.concat([df.assign(model=k) for k, df in per_user.items()]).to_parquet(
        ctx.run_file("day10_test_per_user.parquet"), index=False)


# --------------------------------------------------------------------------- #
# Stage 3: real cold-start users
# --------------------------------------------------------------------------- #
def stage_cold(ctx: Context, results: dict) -> None:
    a = ctx.args
    fit = S.load_view("train_full", config.SPLIT_DIR, columns=TCOLS)
    seeds = S.load_cold_seeds(config.SPLIT_DIR).sort_values(["userId", "seed_rank"], kind="stable")
    cold_test = S.load_view("cold_test", config.SPLIT_DIR, columns=COLS)
    users = eval_users(cold_test, a.cold_users, config.SEED)
    truth = only(cold_test, users)
    seeds = only(seeds, users).reset_index(drop=True)
    ts = build_two_stage(ctx.cat, ctx.raw, ctx.chosen, fit, ctx.n_pop, ctx.m_faiss, ctx.f_m, booster=ctx.warm())
    # Day 6 prior exactly as tuned there: all-time liked counts, or a time window before onboarding
    window = ctx.d6.get("selected", {}).get("time_window_days", "all-time")
    prior_all = popularity_prior(fit, ctx.cat.index, config.RELEVANCE_THRESHOLD)
    tp = (None if window == "all-time" else
          TimePopularity.from_ratings(fit, ctx.cat.index, config.RELEVANCE_THRESHOLD, config.TIME_BUCKET_DAYS))
    t_onb = onboarding_time(seeds, users)          # last of the 5 seeds = 'now' for every seed count
    fallback = popularity_scores(fit, ctx.cat.index, ctx.chosen["popularity_mode"], config.RELEVANCE_THRESHOLD)
    del fit, cold_test
    gc.collect()
    warm, cold = ts.booster, ctx.cold_booster()
    blend = ctx.d6.get("selected", {}).get("blend", {})
    V = ts.emb.item_vectors
    log(f"cold: {len(users):,} users never in training (history for statistics = train+val); "
        f"policy: cold ranker below {config.COLD_POLICY_WARM_FROM} picks, warm ranker from "
        f"{config.COLD_POLICY_WARM_FROM}; Day 6 prior: {window}", ctx.t0)
    rows, sigs, per_user = [], [], []
    for n in config.COLD_SEED_COUNTS:
        h_all = seeds[seeds["seed_rank"].to_numpy() <= n]
        if n == 0:   # no profile: every model is the popularity list
            seen = user_item_matrix(h_all, users, ctx.cat.index, dtype=np.int8)
            R = ts.pop.recommend(seen, KM)
            m, pu = evaluate(users, R, truth, KS, return_per_user=True)
            for key in ("popularity", "policy"):
                rows.append(row(COLD_NAMES[key], m, {"seeds": 0, "key": key,
                                                     "serves": "popularity" if key == "policy" else ""}))
                per_user.append(pu[PER_USER_COLS].assign(model=key, seeds=0))
            log(f"  seeds 0: popularity (= policy) ndcg@10={m['ndcg@10']:.4f}", ctx.t0)
            continue
        b6 = blend.get(f"seeds_{n}", {"pool": 20, "lambda": 5.0})
        tops = {k: [] for k in ("popularity", "day6", "cold", "warm")}
        for s in range(0, len(users), a.batch_users):
            bu = users[s:s + a.batch_users]
            h = rows_of(h_all, bu)
            fb = ts.feature_batch(h, bu)
            tops["popularity"].append(top_from_scores(fb.candidates, ts.score_with(fb, None, []), KM))
            tops["warm"].append(top_from_scores(fb.candidates, ts.score_with(fb, warm, ts.features), KM))
            tops["cold"].append(top_from_scores(fb.candidates, ts.score_with(fb, cold, list(FEATURE_NAMES)), KM))
            P = ts.emb.user_profiles(h, bu)
            seen = user_item_matrix(h, bu, ctx.cat.index, dtype=np.int8)
            tb = t_onb[s:s + len(bu)]
            prior = prior_all if tp is None else (lambda b, tb=tb: tp.prior(tb[b], window))
            r6 = pool_rerank_topk(lambda b: P[b] @ V.T, len(bu), prior, [b6["pool"]], [b6["lambda"]],
                                  seen, KM, ctx.cat.index, fallback)
            tops["day6"].append(r6[(b6["pool"], b6["lambda"])])
        pus = {}
        for key, parts in tops.items():
            m, pu = evaluate(users, np.vstack(parts), truth, KS, return_per_user=True)
            cfg = f"pool {b6['pool']}, λ {b6['lambda']}" if key == "day6" else ""
            rows.append(row(COLD_NAMES[key], m, {"seeds": n, "key": key, "config": cfg}))
            pus[key] = pu[PER_USER_COLS]
        pick = policy_model(n, config.COLD_POLICY_WARM_FROM)
        pus["policy"] = pus[pick]
        rows.append({**next(r for r in rows if r["seeds"] == n and r["key"] == pick),
                     "model": COLD_NAMES["policy"], "key": "policy", "serves": COLD_NAMES[pick], "config": ""})
        for key, pu in pus.items():
            per_user.append(pu.assign(model=key, seeds=n))
        line = "  ".join(f"{k} {r['ndcg@10']:.4f}" for k, r in
                         ((r["key"], r) for r in rows if r["seeds"] == n))
        log(f"  seeds {n}: {line}", ctx.t0)
        pairs = [("cold", "popularity"), ("cold", "day6"), ("warm", "popularity"), ("warm", "day6"),
                 ("warm", "cold"), ("policy", "popularity"), ("policy", "day6"), ("day6", "popularity")]
        for x, y in pairs:
            sigs.append({**sig(COLD_NAMES[x], COLD_NAMES[y], pus[x], pus[y]), "seeds": n})
            s_ = sigs[-1]
            log(f"    {COLD_NAMES[x]:22s} vs {COLD_NAMES[y]:22s} {100 * s_['rel_change']:+6.1f}%  "
                f"CI [{s_['ci95'][0]:+.4f}, {s_['ci95'][1]:+.4f}] p={s_['p_le_0']:.3f}", ctx.t0)

    # reproduce Day 6 (same users, same fit, same settings) -> its numbers must come back
    same_users = ctx.d6.get("test", {}).get("n_users") == len(users)
    d6 = ({(r["setting"], r["model"]): r["ndcg@10"] for r in ctx.d6.get("test", {}).get("rows", [])}
          if same_users else {})
    checks = {}
    for r in rows:
        ref = d6.get((f"seeds_{r['seeds']}", "popularity" if r["key"] == "popularity"
                      else "semantic rerank (ours)" if r["key"] == "day6" else None))
        if ref is not None and r["key"] in ("popularity", "day6"):
            checks[f"{r['key']}_seeds_{r['seeds']}"] = {"day10": r["ndcg@10"], "day6": ref,
                                                         "same": bool(abs(r["ndcg@10"] - ref) < 5e-5)}
    if checks:
        ok = all(c["same"] for c in checks.values())
        log(f"  check: popularity and Day 6 rerank reproduce the Day 6 test numbers: "
            f"{'✓' if ok else 'DIFFERENT (expected only if the user sample differs)'}", ctx.t0)
    results["cold"] = {"population": "cold_test — users never in training; history for item statistics "
                                     "= train+val; their onboarding picks = first n liked movies",
                       "n_users": int(len(users)), "policy": {"warm_from": config.COLD_POLICY_WARM_FROM,
                                                              "chosen_on": "pseudo-cold validation (Day 9)"},
                       "rows": rows, "significance": sigs, "checks": checks}
    pd.concat(per_user).to_parquet(ctx.run_file("day10_cold_per_user.parquet"), index=False)


# --------------------------------------------------------------------------- #
# Stage 4: SHAP
# --------------------------------------------------------------------------- #
def stage_shap(ctx: Context, results: dict) -> None:
    a = ctx.args
    train_full = S.load_view("train_full", config.SPLIT_DIR, columns=TCOLS)
    test = S.load_view("test", config.SPLIT_DIR, columns=COLS)
    pool = eval_users(test, None, config.SEED)
    n_u = min(a.shap_users, len(pool))
    users = np.sort(np.random.default_rng(config.SEED + 10).choice(pool, size=n_u, replace=False))
    truth = only(test, users)
    del test
    ts = build_two_stage(ctx.cat, ctx.raw, ctx.chosen, train_full, ctx.n_pop, ctx.m_faiss, ctx.f_m,
                         booster=ctx.warm())
    hist = only(train_full, users)
    del train_full
    gc.collect()
    names = ts.features
    F = len(names)
    acc = {"all": [np.zeros(F), np.zeros(F), 0], "top10": [np.zeros(F), np.zeros(F), 0]}
    max_add_err = 0.0
    liked = set(map(tuple, truth.loc[truth["rating"] >= config.RELEVANCE_THRESHOLD, ["userId", "movieId"]]
                    .to_numpy().tolist()))
    examples = []
    log(f"shap: {n_u:,} warm test users, TreeSHAP over every candidate", ctx.t0)
    for s in range(0, len(users), a.batch_users):
        bu = users[s:s + a.batch_users]
        h = rows_of(hist, bu)
        fb = ts.feature_batch(h, bu)
        U, K = fb.candidates.shape
        X = np.stack([fb.X[f].ravel() for f in names], axis=1)
        valid = (fb.candidates != PAD).ravel()
        contrib = np.zeros((U * K, F + 1))
        contrib[valid] = tree_shap(ts.booster, X[valid])
        raw = predict(ts.booster, X[valid])
        max_add_err = max(max_add_err, float(np.max(np.abs(contrib[valid].sum(axis=1) - raw))))
        S_ = np.where(fb.candidates == PAD, -np.inf, contrib.sum(axis=1).reshape(U, K))
        order = np.argsort(-S_, axis=1, kind="stable")[:, :10]
        top_mask = np.zeros((U, K), dtype=bool)
        np.put_along_axis(top_mask, order, True, axis=1)
        top_mask &= fb.candidates != PAD
        C3 = contrib[:, :F].reshape(U, K, F)
        for key, mask in (("all", fb.candidates != PAD), ("top10", top_mask)):
            acc[key][0] += np.abs(C3[mask]).sum(axis=0)
            acc[key][1] += C3[mask].sum(axis=0)
            acc[key][2] += int(mask.sum())
        if len(examples) < 2:   # first users whose top-5 contains a movie they liked in test
            for i, u in enumerate(bu):
                top5 = order[i, :5]
                hits = [(u, int(fb.candidates[i, j])) in liked for j in top5]
                if any(hits) and len(examples) < 2:
                    lk = h[(h["userId"].to_numpy() == u) & (h["rating"].to_numpy() >= config.RELEVANCE_THRESHOLD)]
                    examples.append({
                        "userId": int(u), "n_ratings": int((h["userId"].to_numpy() == u).sum()),
                        "recent_liked": [ctx.cat.label.get(int(m), str(m)) for m in lk["movieId"].to_numpy()[-5:][::-1]],
                        "top5": [{"movie": ctx.cat.label.get(int(fb.candidates[i, j]), str(fb.candidates[i, j])),
                                  "score": round(float(S_[i, j]), 3), "liked_in_test": bool(hit),
                                  "reasons": top_reasons(C3[i, j], X.reshape(U, K, F)[i, j], names)}
                                 for j, hit in zip(top5, hits)]})
        del fb, X, contrib, C3
    tables = {k: shap_table(v[0], v[1], v[2], names) for k, v in acc.items()}
    gain = {r["feature"]: r["gain_share"] for r in ctx.d9.get("train", {}).get("importance", {})
            .get(ctx.sel["candidates"], [])}
    for r in tables["all"]:
        r["gain_share_day9"] = gain.get(r["feature"])
    results["shap"] = {"n_users": int(n_u), "rows_all": int(acc["all"][2]), "rows_top10": int(acc["top10"][2]),
                       "method": "LightGBM pred_contrib (exact TreeSHAP, raw lambdarank score)",
                       "additivity_max_error": max_add_err, "all": tables["all"], "top10": tables["top10"],
                       "groups_all": group_shares(tables["all"]), "groups_top10": group_shares(tables["top10"]),
                       "examples": examples}
    top = ", ".join(f"{r['feature']} {100 * r['share']:.0f}%" for r in tables["all"][:5])
    log(f"  mean |SHAP| top 5: {top}; additivity error {max_add_err:.1e}", ctx.t0)
    log("  groups: " + ", ".join(f"{g['group']} {100 * g['share']:.0f}%" for g in results["shap"]["groups_all"]), ctx.t0)


# --------------------------------------------------------------------------- #
# Charts + markdown
# --------------------------------------------------------------------------- #
GRAY, BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#9a9993", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2 = "#0b0b0b", "#52514e"


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    for sp_ in ("left", "bottom"):
        ax.spines[sp_].set_color("#c3c2b7")
    ax.tick_params(colors=INK2, labelsize=8)


def plot(res: dict, smoke: bool) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    tag = "_smoke" if smoke else ""
    if "test" in res:
        rows = sorted(res["test"]["rows"], key=lambda r: r["ndcg@10"])
        color = {"ranker": BLUE, "ranker_release": BLUE, "no_semantic": ORANGE, "no_semantics": ORANGE,
                 "no_year": ORANGE}
        fig, ax = plt.subplots(figsize=(8, 0.42 * len(rows) + 1.2))
        y = np.arange(len(rows))
        ax.barh(y, [r["ndcg@10"] for r in rows], height=0.62,
                color=[color.get(r["key"], GRAY) for r in rows], edgecolor="white", linewidth=2)
        ax.set_yticks(y, [r["model"] for r in rows], fontsize=8, color=INK)
        for yi, r in zip(y, rows):
            ax.text(r["ndcg@10"], yi, f"  {r['ndcg@10']:.4f}", va="center", fontsize=8, color=INK2)
        ax.set_xlabel("NDCG@10 (test)", color=INK2, fontsize=9)
        ax.set_title(f"Warm users — test, {res['test']['n_users']:,} users\n"
                     "blue = ours · orange = ablations · gray = baselines", fontsize=9, color=INK, loc="left")
        ax.xaxis.grid(True, color="#ecebe8", linewidth=0.8)
        ax.set_axisbelow(True)
        _style(ax)
        fig.tight_layout()
        p = config.FIG_DIR / f"day10_test_ndcg{tag}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        out.append(p)
    if "cold" in res:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        series = [("popularity", GRAY, "o"), ("day6", AQUA, "s"), ("cold", ORANGE, "^"),
                  ("warm", MAGENTA, "v"), ("policy", BLUE, "D")]
        for key, col, mk in series:
            pts = sorted((r["seeds"], r["ndcg@10"]) for r in res["cold"]["rows"] if r["key"] == key)
            if not pts:
                continue
            lw = 2.6 if key == "policy" else 1.6
            ax.plot(*zip(*pts), marker=mk, markersize=7, color=col, linewidth=lw, label=COLD_NAMES[key],
                    markeredgecolor="white", markeredgewidth=1.5, zorder=3 if key == "policy" else 2)
        ax.set_xticks(config.COLD_SEED_COUNTS)
        ax.set_xlabel("onboarding picks", color=INK2, fontsize=9)
        ax.set_ylabel("NDCG@10 (test)", color=INK2, fontsize=9)
        ax.set_title(f"Real cold-start users — {res['cold']['n_users']:,} users never in training",
                     fontsize=10, color=INK, loc="left")
        ax.yaxis.grid(True, color="#ecebe8", linewidth=0.8)
        ax.legend(fontsize=8, frameon=False)
        _style(ax)
        fig.tight_layout()
        p = config.FIG_DIR / f"day10_cold_test{tag}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        out.append(p)
    if "shap" in res:
        gcol = dict(zip(FEATURE_GROUPS, [BLUE, GRAY, YELLOW, MAGENTA, AQUA]))
        rows = res["shap"]["all"][::-1]
        fig, ax = plt.subplots(figsize=(7.5, 0.3 * len(rows) + 1.3))
        y = np.arange(len(rows))
        ax.barh(y, [r["share"] for r in rows], height=0.66, color=[gcol.get(r["group"], GRAY) for r in rows],
                edgecolor="white", linewidth=2)
        ax.set_yticks(y, [r["feature"] for r in rows], fontsize=8, color=INK)
        ax.set_xlabel("share of mean |SHAP| (test candidates)", color=INK2, fontsize=9)
        ax.set_title(f"What drives the ranker — {res['shap']['n_users']:,} test users", fontsize=10,
                     color=INK, loc="left")
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=f"{g} ({100 * next((x['share'] for x in res['shap']['groups_all'] if x['group'] == g), 0):.0f}%)")
                           for g, c in gcol.items()], fontsize=7, frameon=False, loc="lower right")
        ax.xaxis.grid(True, color="#ecebe8", linewidth=0.8)
        ax.set_axisbelow(True)
        _style(ax)
        fig.tight_layout()
        p = config.FIG_DIR / f"day10_shap{tag}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        out.append(p)
    return out


def sig_table(sigs: list[dict], extra_cols: tuple = ()) -> str:
    return md_table([fmt_sig(s) for s in sigs], list(extra_cols) + ["model", "vs", "change", "95% CI of diff",
                                                                    "p_le_0", "significant"],
                    left=("model", "vs"))


def write_md(res: dict, path: Path, figs: list[Path]) -> None:
    md = "# Day 10 — Final test results\n\n"
    md += (f"Split fingerprint `{res['split_fingerprint'][:12]}…`. Every setting was chosen on validation "
           "(Days 3–9); test is used once, here. Relevant = rating ≥ 4. Paired bootstrap = 2,000 resamples "
           "of users; `p_le_0` = share of resamples where the first model is not better.\n\n")
    fig = {p.stem.replace("_smoke", ""): p.name for p in figs}
    if "test" in res:
        t = res["test"]
        md += f"## Warm users — test ({t['n_users']:,} users, history = train+val)\n\n"
        md += md_table(t["rows"], ["model"] + SHOW) + "\n"
        md += md_table(t["rows"], ["model", "coverage@10", "long_tail@10", "diversity@10"]) + "\n"
        md += "Paired bootstrap, NDCG@10:\n\n" + sig_table(t["significance"]) + "\n"
        md += "Ranker vs popularity on every metric:\n\n" + md_table(
            [{**fmt_sig(s), "metric": s["metric"]} for s in t["ranker_vs_popularity_by_metric"]],
            ["metric", "change", "95% CI of diff", "p_le_0"], left=("metric",)) + "\n"
        d = t["diagnostics"]
        md += (f"Release dates: {100 * d['popularity_top10_slots_released_after_cutoff']:.1f}% of popularity's "
               f"top-10 slots are movies released after the user's last training rating; "
               f"{100 * d['relevant_test_items_released_after_cutoff']:.1f}% of relevant test movies are such "
               f"movies (the most a release filter can cost).\n\n")
        lat = t["latency"]
        md += (f"Serving cost (batch path, CPU): stage 1 + features {lat['stage1_features_ms_per_user']} ms/user, "
               f"LightGBM {lat['ranker_predict_ms_per_user']} ms/user — {lat['note']}.\n\n")
        if "day10_test_ndcg" in fig:
            md += f"![warm test](../figures/{fig['day10_test_ndcg']})\n\n"
    if "ablation" in res:
        ab = res["ablation"]
        md += (f"## Ablations — validation ({ab['n_valid_users']:,} ranker-valid users; retrained on the same "
               f"{ab['n_train_users']:,} users, same config)\n\n")
        md += md_table(ab["rows"], ["model", "candidates", "features", "ndcg@10", "recall@10", "mrr@10"],
                       left=("model", "candidates")) + "\n" + sig_table(ab["significance"]) + "\n"
    if "cold" in res:
        c = res["cold"]
        md += (f"## Real cold-start users — test ({c['n_users']:,} users never in training)\n\n"
               f"Policy (chosen on pseudo-cold validation, Day 9): 0 picks → popularity, fewer than "
               f"{c['policy']['warm_from']} → cold ranker, otherwise the warm ranker.\n\n")
        table = {}
        for r in c["rows"]:
            table.setdefault(r["seeds"], {"seeds": r["seeds"]})[COLD_NAMES[r["key"]]] = r["ndcg@10"]
        md += "NDCG@10:\n\n" + md_table(list(table.values()), ["seeds"] + list(COLD_NAMES.values())) + "\n"
        md += md_table(c["rows"], ["seeds", "model", "serves"] + SHOW, left=("model", "serves")) + "\n"
        md += sig_table(c["significance"], ("seeds",)) + "\n"
        if c.get("checks"):
            ok = all(v["same"] for v in c["checks"].values())
            md += f"Reproduces the Day 6 test numbers (popularity, semantic rerank): {'✓' if ok else '✗'}\n\n"
        if "day10_cold_test" in fig:
            md += f"![cold test](../figures/{fig['day10_cold_test']})\n\n"
    if "shap" in res:
        s = res["shap"]
        md += (f"## SHAP — {s['n_users']:,} test users, {s['rows_all']:,} candidate rows ({s['method']}; "
               f"additivity error {s['additivity_max_error']:.1e})\n\n")
        md += "By group (all candidates / the top-10 shown):\n\n"
        top10 = {g["group"]: g["share"] for g in s["groups_top10"]}
        md += md_table([{**g, "share (top-10)": top10.get(g["group"])} for g in s["groups_all"]],
                       ["group", "share", "share (top-10)"], left=("group",)) + "\n"
        md += md_table(s["all"], ["feature", "group", "share", "mean_abs_shap", "mean_shap", "gain_share_day9"],
                       left=("feature", "group")) + "\n"
        for ex in s["examples"]:
            md += (f"**Example — user {ex['userId']}** ({ex['n_ratings']} ratings; recently liked: "
                   f"{'; '.join(ex['recent_liked'])})\n\n| # | recommended | liked in test | top reasons (SHAP) |\n"
                   "|---:|---|---|---|\n")
            for i, r in enumerate(ex["top5"], 1):
                why = ", ".join(f"{x['feature']}={x['value']} ({x['shap']:+.2f})" for x in r["reasons"])
                md += f"| {i} | {r['movie']} | {'✓' if r['liked_in_test'] else ''} | {why} |\n"
            md += "\n"
        if "day10_shap" in fig:
            md += f"![shap](../figures/{fig['day10_shap']})\n"
    path.write_text(md, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stages", nargs="+", default=["ablation", "test", "cold", "shap"],
                   choices=["ablation", "test", "cold", "shap"])
    p.add_argument("--batch-users", type=int, default=500)
    p.add_argument("--test-users", type=int, default=None, help="cap warm test users (default: all)")
    p.add_argument("--cold-users", type=int, default=None, help="cap cold users (default: all)")
    p.add_argument("--shap-users", type=int, default=config.SHAP_USERS)
    p.add_argument("--max-rounds", type=int, default=config.RANKER_MAX_ROUNDS)
    p.add_argument("--early-stop", type=int, default=config.RANKER_EARLY_STOP)
    p.add_argument("--smoke", action="store_true", help="small run on the Day 8/9 smoke files")
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    if args.smoke:
        args.test_users, args.cold_users = args.test_users or 3000, args.cold_users or 1000
        args.shap_users, args.max_rounds = min(args.shap_users, 200), 200
    tag = "day10_final_smoke" if args.smoke else "day10_final"
    out_json, results = open_results(tag)
    t0 = time.time()
    if not args.plot_only:
        results["split_fingerprint"] = split_fingerprint()
        ctx = Context(args, t0)
        for stage, fn in (("ablation", stage_ablation), ("test", stage_test), ("cold", stage_cold),
                          ("shap", stage_shap)):
            if stage in args.stages:
                log(f"--- stage {stage} ---", t0)
                fn(ctx, results)
                save_results(out_json, results)
        results["seconds_last_run"] = round(time.time() - t0, 1)
        save_results(out_json, results)
    figs = plot(results, args.smoke)
    write_md(results, out_json.with_suffix(".md"), figs)
    log(f"wrote {out_json.relative_to(config.ROOT)}, .md and {len(figs)} figures", t0)


if __name__ == "__main__":
    main()
