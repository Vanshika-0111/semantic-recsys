"""Day 9: stage 2 — LightGBM LambdaMART ranker on the Day 8 candidates (validation only).

Input: the Day 8 feature table (20,000 validation users; features from TRAIN, labels = their val
ratings). The users are split once into ranker-train (75%) and ranker-valid (25%); every choice
today is made on ranker-valid. The test set is not touched — Day 10 locks the final numbers.

Stages (all by default):
  features  the SAME 20k users with a different stage 1: popular top-500 only, FAISS kept as a
            FEATURE (in_faiss / faiss_rank = in the FAISS top-100?) instead of adding candidates
            -> data/runs/day09_features_val_pop500.parquet. Also re-computes one batch of the Day 8
            union table through recsys.ranker.TwoStage and checks it matches (the Day 10 / API
            path builds features exactly like the training table)
  train     for both candidate sets (union 400+100 vs popularity-500, the Day 8 open question):
            LambdaMART over the grid (labels binary|graded x num_leaves), early stopping on
            ranker-valid; full-catalog metrics @5/10/20 vs popularity on the same users; paired
            bootstrap; gain importance. Picks the best (candidates, config) by val NDCG@10
            -> data/runs/day09_ranker.txt (the Day 10 model)
  cold      "cold mode" (Day 6 note): the same features for PSEUDO-cold users (warm val users who
            show only their first 1/3/5 liked train movies). Trains a cold ranker with the chosen
            config, then compares at 1/3/5 seeds: popularity, the Day 6 semantic rerank (pool 20),
            the warm ranker applied as is, and the cold ranker -> data/runs/day09_cold_ranker.txt

Outputs
  reports/results/day09_ranker.json + .md, reports/figures/day09_ranker_val.png    (commit)
  data/runs/day09_*.parquet / .txt                                                (git-ignored)

Usage
  python scripts/run_ranker.py --smoke              # ~2-3 min check (needs Day 8 --smoke run)
  python scripts/run_ranker.py                      # full run
  python scripts/run_ranker.py --stages train       # one stage (others are kept in the json)
  python scripts/run_ranker.py --stages train --quick   # 2 grid points instead of 4
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # works without `pip install -e .`
from recsys.experiment import setup_runtime  # noqa: E402

setup_runtime()          # Windows OpenMP fix; must run before faiss / lightgbm are imported

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import gc  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from recsys import config  # noqa: E402
from recsys import split as S  # noqa: E402
from recsys.baselines import popularity_scores, user_item_matrix  # noqa: E402
from recsys.coldstart import first_liked, pool_rerank_topk, popularity_prior  # noqa: E402
from recsys.experiment import (COLS, SHOW, TCOLS, chosen_settings, eval_users, load_catalog,  # noqa: E402
                               load_item_vectors, log, md_table, only, open_results, read_results,
                               row, rows_of, save_results, split_fingerprint)
from recsys.evaluation import project_file  # noqa: E402
from recsys.features import FEATURE_NAMES, attach_labels  # noqa: E402
from recsys.metrics import PAD, evaluate, paired_bootstrap  # noqa: E402
from recsys.ranker import (RankData, build_two_stage, importance, predict, split_users,  # noqa: E402
                           topk_by_score, train_ranker)

UNION_FILE = "day08_features_val.parquet"
POP_FILE = "day09_features_val_pop500.parquet"
MODEL_FILE = "day09_ranker.txt"
COLD_MODEL_FILE = "day09_cold_ranker.txt"
KS = config.EVAL_KS
KM = max(KS)


def smoke_name(name: str, smoke: bool) -> str:
    return name.replace(".parquet", "_smoke.parquet").replace(".txt", "_smoke.txt") if smoke else name


class Context:
    def __init__(self, args, t0: float) -> None:
        self.args, self.t0 = args, t0
        self.cat = load_catalog()
        self.chosen = chosen_settings()
        self.raw = load_item_vectors(self.cat, self.chosen["encoder"])
        d8 = read_results("day08_candidates_smoke.json" if args.smoke else "day08_candidates.json")
        if "selected" not in d8:
            raise SystemExit("no Day 8 results — run scripts/run_candidates.py first")
        self.d8 = d8
        s = d8["selected"]
        self.n_pop, self.m_faiss = int(s["pop"]), int(s["faiss"])
        self.variants = {   # name -> (n_pop, m_faiss, faiss feature length, feature file)
            "union": (self.n_pop, self.m_faiss, self.m_faiss, UNION_FILE),
            f"pop{config.RANKER_POP_ONLY}": (config.RANKER_POP_ONLY, 0, self.m_faiss, POP_FILE),
        }
        log(f"catalog {len(self.cat.index):,}; encoder {self.chosen['encoder']}; Day 8 union "
            f"pop {self.n_pop} + faiss {self.m_faiss}; variants {list(self.variants)}", t0)

    def path(self, name: str) -> Path:
        return config.RUNS_DIR / smoke_name(name, self.args.smoke)

    def label(self, variant: str) -> str:
        n, m, f, _ = self.variants[variant]
        return (f"popular {n} ∪ FAISS {m}" if m else f"popular {n} (FAISS top-{f} as feature)")


def load_rank_data(path: Path) -> RankData:
    import pyarrow.parquet as pq

    if not path.exists():
        raise SystemExit(f"{path.name} missing — run the stage that writes it first")
    tbl = pq.read_table(path)
    n = tbl.num_rows
    X = np.empty((n, len(FEATURE_NAMES)), dtype=np.float32)
    for j, f in enumerate(FEATURE_NAMES):
        X[:, j] = tbl.column(f).to_numpy()
    d = RankData(X, tbl.column("rating").to_numpy().astype(np.float32),
                 tbl.column("userId").to_numpy().astype(np.int64),
                 tbl.column("movieId").to_numpy().astype(np.int64), list(FEATURE_NAMES))
    del tbl
    gc.collect()
    return d


def feature_users(ctx: Context) -> np.ndarray:
    import pyarrow.parquet as pq

    p = ctx.path(UNION_FILE)
    if not p.exists():
        raise SystemExit(f"{p.name} missing — run scripts/run_candidates.py (stage features)")
    return np.unique(pq.read_table(p, columns=["userId"]).column("userId").to_numpy().astype(np.int64))


# --------------------------------------------------------------------------- #
def stage_features(ctx: Context, results: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    a = ctx.args
    users = feature_users(ctx)
    train = S.load_view("train", config.SPLIT_DIR, columns=TCOLS)
    val = S.load_view("val", config.SPLIT_DIR, columns=COLS)
    truth = only(val, users)
    del val
    name = f"pop{config.RANKER_POP_ONLY}"
    n, m, f, fname = ctx.variants[name]
    ts = build_two_stage(ctx.cat, ctx.raw, ctx.chosen, train, n, m, f)
    hist = only(train, users)
    del train
    gc.collect()
    log(f"features: {len(users):,} users (= the Day 8 table), candidates {ctx.label(name)}", ctx.t0)

    # 1. the union table through the serving path must equal the Day 8 table
    bu = users[:min(a.batch_users, len(users))]
    union = dataclasses.replace(ts, n_pop=ctx.n_pop, m_faiss=ctx.m_faiss, faiss_feature_m=ctx.m_faiss)
    mine = union.feature_batch(rows_of(hist, bu), bu).to_frame()
    tbl = pq.read_table(ctx.path(UNION_FILE), filters=[("userId", "<=", int(bu[-1]))])
    day8 = tbl.to_pandas()
    same_rows = len(mine) == len(day8) and np.array_equal(mine["movieId"].to_numpy(), day8["movieId"].to_numpy())
    max_diff = (float(np.nanmax(np.abs(mine[FEATURE_NAMES].to_numpy(np.float64) -
                                       day8[FEATURE_NAMES].to_numpy(np.float64))))
                if same_rows else float("inf"))
    nan_same = same_rows and bool(np.array_equal(mine[FEATURE_NAMES].isna().to_numpy(),
                                                 day8[FEATURE_NAMES].isna().to_numpy()))
    ok = same_rows and nan_same and max_diff < 1e-4
    log(f"  check: TwoStage rebuilds the Day 8 union features for {len(bu)} users: "
        f"{'identical' if ok else 'DIFFERENT — investigate'} (max |diff| {max_diff:.2e})", ctx.t0)
    del tbl, day8, mine

    # 2. popularity-500 candidates, FAISS as a feature
    out_path = ctx.path(fname)
    writer = None
    n_rows = n_pos = 0
    t1 = time.time()
    for s in range(0, len(users), a.batch_users):
        bu = users[s:s + a.batch_users]
        fb = ts.feature_batch(rows_of(hist, bu), bu)
        df = attach_labels(fb.to_frame(), rows_of(truth, bu))
        t = pa.Table.from_pandas(df, preserve_index=False)
        writer = writer or pq.ParquetWriter(out_path, t.schema, compression="zstd")
        writer.write_table(t)
        n_rows += len(df)
        n_pos += int((df["rating"].to_numpy() >= config.RELEVANCE_THRESHOLD).sum())
        if (s // a.batch_users) % 10 == 0:
            log(f"  {min(s + a.batch_users, len(users)):,}/{len(users):,} users, {n_rows:,} rows", ctx.t0)
        del fb, df, t
    writer.close()
    secs = time.time() - t1
    results["features"] = {
        "n_users": int(len(users)), "union_rebuild_identical": ok, "union_rebuild_max_diff": max_diff,
        name: {"candidates": ctx.label(name), "rows": int(n_rows), "relevant_rows": int(n_pos),
               "file": str(out_path.relative_to(config.ROOT)),
               "file_mb": round(out_path.stat().st_size / 2**20, 1),
               "ms_per_user": round(1000 * secs / len(users), 2)},
        "union": {"candidates": ctx.label("union"), "rows": ctx.d8.get("features", {}).get("rows"),
                  "relevant_rows": ctx.d8.get("features", {}).get("relevant_rows"),
                  "file": str(ctx.path(UNION_FILE).relative_to(config.ROOT))},
    }
    log(f"wrote {out_path.name}: {n_rows:,} rows ({n_pos:,} relevant), "
        f"{results['features'][name]['file_mb']} MB", ctx.t0)


# --------------------------------------------------------------------------- #
def grid(args) -> list[dict]:
    g = [dict(config.RANKER_BASE, **c) for c in config.RANKER_GRID]
    if args.quick or args.smoke:
        g = [c for c in g if c["num_leaves"] == min(x["num_leaves"] for x in g)]
    return g


def cfg_name(c: dict) -> str:
    return f"{c['labels']} labels, {c['num_leaves']} leaves"


def metrics_row(name: str, users, recs, truth, extra: dict) -> tuple[dict, pd.DataFrame]:
    m, pu = evaluate(users, recs, truth, KS, return_per_user=True)
    return row(name, m, extra), pu


def sig_row(name: str, a: pd.DataFrame, b: pd.DataFrame, vs: str) -> dict:
    r = paired_bootstrap(a["ndcg@10"].to_numpy(), b["ndcg@10"].to_numpy(), n_boot=2000, seed=config.SEED)
    return {"model": name, "vs": vs, **{k: (round(v, 5) if isinstance(v, float) else [round(x, 5) for x in v])
                                        for k, v in r.items()}}


def stage_train(ctx: Context, results: dict) -> None:
    a = ctx.args
    users = feature_users(ctx)
    tr_u, va_u = split_users(users, config.RANKER_VALID_SHARE, config.SEED + 9)
    val = S.load_view("val", config.SPLIT_DIR, columns=COLS)
    truth = only(val, va_u)                     # ALL their val ratings, incl. movies stage 1 missed
    del val
    gc.collect()
    log(f"train: {len(tr_u):,} ranker-train / {len(va_u):,} ranker-valid users (split by user)", ctx.t0)
    G = grid(a)
    rows, per_user, runs, best, imp = [], {}, [], {}, {}
    for variant in ctx.variants:
        d = load_rank_data(ctx.path(ctx.variants[variant][3]))
        va = d.subset(va_u)
        with_pos = np.unique(d.users[d.rating >= config.RELEVANCE_THRESHOLD])
        tr = d.subset(np.intersect1d(tr_u, with_pos))   # = drop_users_without_positives, one copy
        del d
        gc.collect()
        log(f"--- {variant}: {ctx.label(variant)} — train {len(tr):,} rows "
            f"({len(np.unique(tr.users)):,} users with a relevant candidate), valid {len(va):,} rows", ctx.t0)
        if variant == "union":           # popularity = candidates in pop_rank order (same for both)
            pr = va.X[:, FEATURE_NAMES.index("pop_rank")]
            u, top = topk_by_score(va.users, va.movies, -pr.astype(np.float64), KM)
            r, pu = metrics_row("popularity", u, top, truth, {"config": "Day 3 baseline, same users"})
            rows.append(r)
            per_user["popularity"] = pu
            lab = make_oracle(va)
            u, top = topk_by_score(va.users, va.movies, lab, KM)
            orc = evaluate(u, top, truth, (10,))["ndcg@10"]
            log(f"  popularity ndcg@10={r['ndcg@10']:.4f}; oracle on these candidates {orc:.4f}", ctx.t0)
        else:
            lab = make_oracle(va)
            u, top = topk_by_score(va.users, va.movies, lab, KM)
            orc = evaluate(u, top, truth, (10,))["ndcg@10"]
        for cfg in G:
            booster, info = train_ranker(tr, va, cfg, config.SEED, a.max_rounds, a.early_stop,
                                         log=(lambda msg: log(msg, ctx.t0)) if a.verbose else None)
            u, top = topk_by_score(va.users, va.movies, predict(booster, va.X), KM)
            m, pu = evaluate(u, top, truth, KS, return_per_user=True)
            run = {"candidates": variant, "config": cfg_name(cfg), **info,
                   **{k: round(m[k], 5) for k in SHOW}, "params": cfg}
            runs.append(run)
            log(f"  {cfg_name(cfg):26s} ndcg@10={m['ndcg@10']:.4f}  best_iter={info['best_iter']:4d}  "
                f"{info['seconds']:.0f}s", ctx.t0)
            if variant not in best or m["ndcg@10"] > best[variant]["ndcg@10"]:
                best[variant] = {"cfg": cfg, "info": info, "ndcg@10": m["ndcg@10"], "pu": pu, "m": m}
                booster.save_model(str(ctx.path(f"day09_ranker_{variant}.txt")),
                                   num_iteration=info["best_iter"])
                imp[variant] = importance(booster, FEATURE_NAMES)
            del booster
        b = best[variant]
        rows.append(row(f"LightGBM ranker — {ctx.label(variant)}", b["m"],
                        {"config": cfg_name(b["cfg"]), "oracle@10": round(orc, 5), "candidates": variant}))
        per_user[variant] = b["pu"]
        del tr, va
        gc.collect()

    v_union, v_pop = list(ctx.variants)
    winner = max(ctx.variants, key=lambda v: (best[v]["ndcg@10"], v == v_union))
    shutil.copyfile(ctx.path(f"day09_ranker_{winner}.txt"), ctx.path(MODEL_FILE))
    sig = [sig_row(f"ranker ({v})", per_user[v], per_user["popularity"], "popularity") for v in ctx.variants]
    sig.append(sig_row(f"ranker ({v_union})", per_user[v_union], per_user[v_pop], f"ranker ({v_pop})"))
    for s in sig:
        log(f"  {s['model']:20s} vs {s['vs']:20s} {100 * s['rel_change']:+.1f}%  "
            f"95% CI [{s['ci95'][0]:+.4f}, {s['ci95'][1]:+.4f}]  p={s['p_le_0']:.3f}", ctx.t0)
    n, m, f, _ = ctx.variants[winner]
    w = best[winner]
    results["train"] = {"split": "val (history = train, labels = val), ranker-valid users",
                        "n_train_users": int(len(tr_u)), "n_valid_users": int(len(va_u)),
                        "rows": rows, "runs": runs, "significance": sig, "importance": imp}
    results["selected"] = {
        "candidates": winner, "pop": n, "faiss": m, "faiss_feature": f, "label": ctx.label(winner),
        "config": w["cfg"], "best_iter": w["info"]["best_iter"], "val_ndcg@10": round(w["ndcg@10"], 5),
        "features": list(FEATURE_NAMES), "model_file": ctx.path(MODEL_FILE).relative_to(config.ROOT).as_posix(),
        "rule": "highest ranker-valid NDCG@10 over (candidate set x grid); tie -> union"}
    pop10 = rows[0]["ndcg@10"]
    log(f"selected: {ctx.label(winner)}, {cfg_name(w['cfg'])}, {w['info']['best_iter']} trees: "
        f"val ndcg@10 {w['ndcg@10']:.4f} vs popularity {pop10:.4f} "
        f"({100 * (w['ndcg@10'] / pop10 - 1):+.1f}%)", ctx.t0)


def make_oracle(d: RankData) -> np.ndarray:
    """Score = held-out rating: relevant candidates first (the ceiling of any ranker)."""
    return np.where(d.rating >= config.RELEVANCE_THRESHOLD, d.rating, 0).astype(np.float64)


# --------------------------------------------------------------------------- #
def stage_cold(ctx: Context, results: dict) -> None:
    import lightgbm as lgb

    a = ctx.args
    sel = results.get("selected")
    if not sel:
        raise SystemExit("no selected ranker — run the 'train' stage first")
    warm = lgb.Booster(model_file=str(project_file(sel["model_file"])))
    train = S.load_view("train", config.SPLIT_DIR, columns=TCOLS)
    val = S.load_view("val", config.SPLIT_DIR, columns=COLS)
    cand = eval_users(val, None, config.SEED)
    liked = train.loc[train["rating"].to_numpy() >= config.RELEVANCE_THRESHOLD, "userId"].value_counts()
    cand = cand[np.isin(cand, liked.index[liked >= max(config.COLD_RANKER_SEEDS)].to_numpy())]
    n_users = min(a.cold_users, len(cand))
    users = np.sort(np.random.default_rng(config.SEED + 11).choice(cand, size=n_users, replace=False))
    tr_u, va_u = split_users(users, config.RANKER_VALID_SHARE, config.SEED + 12)
    truth = only(val, users)
    del val
    seeds = first_liked(only(train, users), max(config.COLD_RANKER_SEEDS), config.RELEVANCE_THRESHOLD)
    ts = build_two_stage(ctx.cat, ctx.raw, ctx.chosen, train, sel["pop"], sel["faiss"], sel["faiss_feature"],
                         booster=warm)
    prior = popularity_prior(train, ctx.cat.index, config.RELEVANCE_THRESHOLD)
    fallback = popularity_scores(train, ctx.cat.index, ctx.chosen["popularity_mode"], config.RELEVANCE_THRESHOLD)
    del train
    gc.collect()
    rng = np.random.default_rng(config.SEED + 13)
    n_of = dict(zip(users, rng.choice(config.COLD_RANKER_SEEDS, size=len(users))))
    log(f"cold: {len(users):,} pseudo-cold val users ({len(tr_u):,} train / {len(va_u):,} valid); "
        f"seeds = first n liked train movies, n in {config.COLD_RANKER_SEEDS}; "
        f"candidates {sel['label']}", ctx.t0)

    # rows_of needs frames that hold ONLY the users of one sorted list -> one frame per side
    seeds_by = {"train": only(seeds, tr_u), "valid": only(seeds, va_u)}
    truth_by = {"train": only(truth, tr_u), "valid": only(truth, va_u)}

    def seeds_for(us: np.ndarray, n: int | None, side: str) -> pd.DataFrame:
        s = rows_of(seeds_by[side], us)
        lim = np.array([n_of[u] for u in s["userId"].to_numpy()]) if n is None else n
        return s[s["seed_rank"].to_numpy() <= lim]

    def table(us: np.ndarray, side: str) -> RankData:   # mixed seed counts, one random n per user
        parts = []
        for s in range(0, len(us), a.batch_users):
            bu = us[s:s + a.batch_users]
            df = attach_labels(ts.feature_batch(seeds_for(bu, None, side), bu).to_frame(),
                               rows_of(truth_by[side], bu))
            parts.append(RankData.from_frame(df))
        return RankData(np.concatenate([p.X for p in parts]), np.concatenate([p.rating for p in parts]),
                        np.concatenate([p.users for p in parts]), np.concatenate([p.movies for p in parts]),
                        list(FEATURE_NAMES))

    t1 = time.time()
    trd, vad = table(tr_u, "train").drop_users_without_positives(), table(va_u, "valid")
    log(f"  features: train {len(trd):,} rows, valid {len(vad):,} rows ({time.time() - t1:.0f}s)", ctx.t0)
    cold_b, info = train_ranker(trd, vad, sel["config"], config.SEED, a.max_rounds, a.early_stop)
    cold_b.save_model(str(ctx.path(COLD_MODEL_FILE)), num_iteration=info["best_iter"])
    cold_b = lgb.Booster(model_file=str(ctx.path(COLD_MODEL_FILE)))
    imp = importance(cold_b, FEATURE_NAMES)
    del trd, vad
    gc.collect()
    log(f"  cold ranker: {info['best_iter']} trees, {info['seconds']:.0f}s", ctx.t0)

    blend = read_results("day06_coldstart.json").get("selected", {}).get("blend", {})
    truth_va = truth_by["valid"]
    rows, sig = [], []
    for n in config.COLD_RANKER_SEEDS:
        tops = {k: [] for k in ("popularity", "day6", "warm", "cold")}
        b6 = blend.get(f"seeds_{n}", {"pool": 20, "lambda": 5.0})
        for s in range(0, len(va_u), a.batch_users):
            bu = va_u[s:s + a.batch_users]
            h = seeds_for(bu, n, "valid")
            fb = ts.feature_batch(h, bu)
            X = np.stack([fb.X[f].ravel() for f in FEATURE_NAMES], axis=1)
            U, K = fb.candidates.shape
            for key, S_ in (("popularity", -np.tile(np.arange(K, dtype=np.float64), (U, 1))),
                            ("warm", predict(warm, X).reshape(U, K)),
                            ("cold", predict(cold_b, X).reshape(U, K))):
                S_ = np.where(fb.candidates == PAD, -np.inf, S_)
                o = np.argsort(-S_, axis=1, kind="stable")[:, :KM]
                tops[key].append(np.take_along_axis(fb.candidates, o, axis=1))
            P = ts.emb.user_profiles(h, bu)
            seen = user_item_matrix(h, bu, ctx.cat.index, dtype=np.int8)
            V = ts.emb.item_vectors
            r6 = pool_rerank_topk(lambda b: P[b] @ V.T, len(bu), prior, [b6["pool"]], [b6["lambda"]],
                                  seen, KM, ctx.cat.index, fallback)
            tops["day6"].append(r6[(b6["pool"], b6["lambda"])])
        names = {"popularity": "popularity", "day6": "Day 6 semantic rerank",
                 "warm": "warm ranker (as is)", "cold": "cold ranker (ours)"}
        pus = {}
        for key, parts in tops.items():
            m, pu = evaluate(va_u, np.vstack(parts), truth_va, KS, return_per_user=True)
            rows.append(row(names[key], m, {"seeds": n, "config": f"pool {b6['pool']}, λ {b6['lambda']}"
                                            if key == "day6" else ""}))
            pus[key] = pu
            log(f"  seeds {n}: {names[key]:42s} ndcg@10={m['ndcg@10']:.4f}", ctx.t0)
        for key in ("cold", "warm"):
            for base in ("popularity", "day6"):
                if key == "warm" and base == "day6":
                    continue
                sig.append({**sig_row(names[key], pus[key], pus[base], names[base]), "seeds": n})
                s_ = sig[-1]
                log(f"    {names[key]} vs {names[base]}: {100 * s_['rel_change']:+.1f}% "
                    f"CI [{s_['ci95'][0]:+.4f}, {s_['ci95'][1]:+.4f}] p={s_['p_le_0']:.3f}", ctx.t0)
    results["cold"] = {"population": "pseudo-cold validation: warm val users shown only their first n liked "
                                     "train movies; history for item statistics = train",
                       "n_train_users": int(len(tr_u)), "n_valid_users": int(len(va_u)),
                       "config": sel["config"], "best_iter": info["best_iter"], "rows": rows,
                       "significance": sig, "importance": imp,
                       "model_file": ctx.path(COLD_MODEL_FILE).relative_to(config.ROOT).as_posix()}


# --------------------------------------------------------------------------- #
def plot(res: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "train" not in res:
        return
    fig, axes = plt.subplots(1, 2 if "cold" in res else 1, figsize=(11 if "cold" in res else 6, 4))
    axes = np.atleast_1d(axes)
    rows = res["train"]["rows"]
    names = [f"ranker\n{r['candidates']}" if "candidates" in r else r["model"] for r in rows]
    vals = [r["ndcg@10"] for r in rows]
    ax = axes[0]
    bars = ax.bar(range(len(rows)), vals, color=["#9aa0a6"] + ["#3b6fb6"] * (len(rows) - 1))
    ax.set_xticks(range(len(rows)), names, fontsize=8)
    for b_, v in zip(bars, vals):
        ax.text(b_.get_x() + b_.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("NDCG@10 (ranker-valid users)")
    ax.set_title(f"Warm users — validation ({res['train']['n_valid_users']:,} users)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    if "cold" in res:
        ax = axes[1]
        cr = res["cold"]["rows"]
        for model in dict.fromkeys(r["model"] for r in cr):
            pts = [(r["seeds"], r["ndcg@10"]) for r in cr if r["model"] == model]
            ax.plot(*zip(*pts), marker="o", label=model)
        ax.set_xticks(config.COLD_RANKER_SEEDS)
        ax.set_xlabel("onboarding seeds")
        ax.set_ylabel("NDCG@10")
        ax.set_title(f"Pseudo-cold users — validation ({res['cold']['n_valid_users']:,} users)", fontsize=10)
        ax.legend(fontsize=7, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_md(res: dict, path: Path, fig: Path) -> None:
    md = "# Day 9 — LightGBM LambdaMART ranker (validation)\n\n"
    md += (f"Split fingerprint `{res['split_fingerprint'][:12]}…`. Features = Day 8 (history = train), "
           "labels = val ratings. The 20k feature users are split once by user into ranker-train / "
           "ranker-valid; every number here is on ranker-valid users and is computed over their FULL "
           "val ratings (a relevant movie stage 1 missed counts as a miss). Test is untouched (Day 10).\n\n")
    if "selected" in res:
        s = res["selected"]
        md += (f"**Selected for Day 10:** {s['label']}, {cfg_name(s['config'])}, {s['best_iter']} trees "
               f"(`{s['model_file']}`) — {s['rule']}.\n\n")
    if "train" in res:
        t = res["train"]
        md += (f"## Warm users — {t['n_valid_users']:,} ranker-valid users "
               f"(trees fit on {t['n_train_users']:,} others)\n\n")
        md += md_table(t["rows"], ["model", "config"] + SHOW + ["oracle@10"])
        md += "\nPaired bootstrap (resampling users, NDCG@10):\n\n"
        md += md_table([{**s_, "change": f"{100 * s_['rel_change']:+.1f}%",
                         "95% CI of diff": f"[{s_['ci95'][0]:+.4f}, {s_['ci95'][1]:+.4f}]"} for s_ in t["significance"]],
                       ["model", "vs", "change", "95% CI of diff", "p_le_0"], left=("model", "vs"))
        md += "\n### Grid (early stopping on ranker-valid)\n\n"
        md += md_table(t["runs"], ["candidates", "config", "best_iter", "ndcg@10", "recall@10", "mrr@10",
                                   "lgb_valid_ndcg@10", "seconds"], left=("candidates", "config"))
        md += ("\n`lgb_valid_ndcg@10` = LightGBM's own metric: candidates only, and a user with no "
               "relevant candidate counts as 1 — use it for early stopping only.\n\n")
        for v, rows in t["importance"].items():
            md += f"### Gain importance — {v}\n\n" + md_table(rows, ["feature", "gain_share", "splits"],
                                                              left=("feature",)) + "\n"
    if "features" in res:
        f = res["features"]
        md += (f"## Feature tables\n\nTwoStage (the Day 10 / API path) rebuilds the Day 8 union features "
               f"{'identically ✓' if f['union_rebuild_identical'] else 'with DIFFERENCES ⚠️'} "
               f"(max |diff| {f['union_rebuild_max_diff']:.1e}).\n\n")
        md += md_table([{"variant": k, **v} for k, v in f.items() if isinstance(v, dict)],
                       ["variant", "candidates", "rows", "relevant_rows", "file"],
                       left=("variant", "candidates", "file")) + "\n"
    if "cold" in res:
        c = res["cold"]
        md += (f"## Cold mode — {c['n_valid_users']:,} pseudo-cold validation users\n\n"
               f"Cold ranker: {cfg_name(c['config'])}, {c['best_iter']} trees, trained on "
               f"{c['n_train_users']:,} other pseudo-cold users. {c['population']}.\n\n")
        md += md_table(c["rows"], ["seeds", "model", "config"] + SHOW, left=("model", "config"))
        md += "\n" + md_table([{**s_, "change": f"{100 * s_['rel_change']:+.1f}%",
                                "95% CI of diff": f"[{s_['ci95'][0]:+.4f}, {s_['ci95'][1]:+.4f}]"}
                               for s_ in c["significance"]],
                              ["seeds", "model", "vs", "change", "95% CI of diff", "p_le_0"], left=("model", "vs"))
        md += "\n### Gain importance — cold ranker\n\n" + md_table(c["importance"][:10], ["feature", "gain_share", "splits"],
                                                                   left=("feature",))
    if fig.exists():
        md += f"\n![Day 9 validation](../figures/{fig.name})\n"
    path.write_text(md, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stages", nargs="+", default=["features", "train", "cold"],
                   choices=["features", "train", "cold"])
    p.add_argument("--batch-users", type=int, default=500)
    p.add_argument("--max-rounds", type=int, default=config.RANKER_MAX_ROUNDS)
    p.add_argument("--early-stop", type=int, default=config.RANKER_EARLY_STOP)
    p.add_argument("--cold-users", type=int, default=config.COLD_RANKER_USERS)
    p.add_argument("--quick", action="store_true", help="only the smaller-tree half of the grid")
    p.add_argument("--verbose", action="store_true", help="print LightGBM progress every 100 rounds")
    p.add_argument("--smoke", action="store_true", help="small run on the Day 8 smoke table")
    args = p.parse_args()
    if args.smoke:
        args.max_rounds, args.cold_users = 200, 3000
    tag = "day09_ranker_smoke" if args.smoke else "day09_ranker"
    out_json, results = open_results(tag)
    t0 = time.time()
    results["split_fingerprint"] = split_fingerprint()
    ctx = Context(args, t0)
    for stage, fn in (("features", stage_features), ("train", stage_train), ("cold", stage_cold)):
        if stage in args.stages:
            log(f"--- stage {stage} ---", t0)
            fn(ctx, results)
            save_results(out_json, results)
    results["seconds_last_run"] = round(time.time() - t0, 1)
    save_results(out_json, results)
    fig = config.FIG_DIR / f"{tag}_val.png"
    plot(results, fig)
    write_md(results, out_json.with_suffix(".md"), fig)
    log(f"wrote {out_json.relative_to(config.ROOT)}, .md and {fig.relative_to(config.ROOT)}", t0)


if __name__ == "__main__":
    main()
