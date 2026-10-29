"""Day 12: serving — release-filter check on validation, the API bundle, a golden check, latency.

Stages (all by default, in this order):
  release  validation — the Day 9 ranker-valid users (history = train, labels = val). Day 10 found
           "ranker + release filter" +2.8% on TEST but it was never checked on validation, so it
           could not be used. Here: ranker vs ranker + filter (hide movies released after the
           user's last training-rating year), also with the served genre MMR. Rule (config, written
           before the run): on iff the filter's NDCG@10 gain has a bootstrap CI above 0.
  bundle   history = train+val (= Day 10 test setting) -> data/serving/: item vectors, item
           statistics, popularity, cold-start prior, both rankers, settings, 1,000 demo users
           (their train+val history + test ratings). No ratings needed to start the API.
  check    golden check — the Engine, loaded ONLY from the bundle, must return the same lists as
           the batch code that produced the CV numbers: warm users (ranker, ranker + MMR) and real
           cold users at 1/3/5 picks (policy model). Plus NDCG@10 of the served config on the demo
           users (a sanity check, not a new claim).
  latency  one request at a time, after warm-up: warm user (ranker / + MMR + explanations),
           cold start 0/1/3/5 picks, free-text explore, title search. p50/p95/p99 through the HTTP
           stack (FastAPI TestClient in-process, or --url for a running server / Docker container),
           a per-step breakdown inside the engine, startup time, bundle size, memory.

Outputs
  reports/results/day12_serving.json + .md                          (commit)
  data/serving/  (bundle; git-ignored — rebuilt by the `bundle` stage)

Usage
  python scripts/run_serving.py --smoke                    # ~1 min check (needs the Day 9 smoke run)
  python scripts/run_serving.py                            # full (~10-15 min on the 8 GB laptop)
  python scripts/run_serving.py --stages latency --url http://localhost:8000   # a running server
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # works without `pip install -e .`
from recsys.experiment import setup_runtime  # noqa: E402

setup_runtime()          # Windows OpenMP fix; must run before faiss / lightgbm are imported

import argparse  # noqa: E402
import gc  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from recsys import config  # noqa: E402
from recsys import split as S  # noqa: E402
from recsys.baselines import user_item_matrix  # noqa: E402
from recsys.coldstart import popularity_prior  # noqa: E402
from recsys.diversity import mmr_rerank  # noqa: E402
from recsys.evaluation import (policy_model, project_file, released_after, significant,  # noqa: E402
                               user_cutoff_year)
from recsys.experiment import (COLS, TCOLS, chosen_settings, eval_users, load_catalog,  # noqa: E402
                               load_item_vectors, log, md_table, only, open_results, read_results,
                               rows_of, save_results, split_fingerprint)
from recsys.metrics import PAD, evaluate, paired_bootstrap  # noqa: E402
from recsys.ranker import (build_two_stage, load_rank_data, pad_by_user, predict,  # noqa: E402
                           split_users, top_from_scores)
from recsys.serving import Engine, export_bundle, latency_summary, release_filter  # noqa: E402

KS = config.EVAL_KS
KM = max(KS)
STAGES = ["release", "bundle", "check", "latency"]


def sfx(name: str, smoke: bool) -> str:
    if not smoke:
        return name
    stem, _, ext = name.rpartition(".")
    return f"{stem}_smoke.{ext}"


class Context:
    def __init__(self, args, t0: float) -> None:
        self.args, self.t0 = args, t0
        sm = args.smoke
        self.d6 = read_results("day06_coldstart.json") or read_results("day06_coldstart_smoke.json")
        self.d9 = read_results(sfx("day09_ranker.json", sm))
        self.d10 = read_results(sfx("day10_final.json", sm))
        self.d11 = read_results(sfx("day11_diversity.json", sm))
        if "selected" not in self.d9:
            raise SystemExit("no Day 9 results — run scripts/run_ranker.py first")
        self.sel = self.d9["selected"]
        self.n_pop, self.m_faiss, self.f_m = self.sel["pop"], self.sel["faiss"], self.sel["faiss_feature"]
        self.mmr = dict(self.d11.get("tune", {}).get("selected", {"space": "genre", "lambda": 0.7, "pool": 50}))
        self.mmr = {k: self.mmr[k] for k in ("space", "lambda", "pool")}
        self.bundle_dir = config.SERVING_DIR.with_name(config.SERVING_DIR.name + ("_smoke" if sm else ""))
        self._cat = self._raw = self._fit = self._ts = None

    # lazily loaded, shared by stages
    @property
    def cat(self):
        if self._cat is None:
            self._cat = load_catalog()
        return self._cat

    @property
    def chosen(self) -> dict:
        return chosen_settings()

    @property
    def raw(self):
        if self._raw is None:
            self._raw = load_item_vectors(self.cat, self.chosen["encoder"])
        return self._raw

    def booster(self, which: str):
        import lightgbm as lgb

        rel = self.sel["model_file"] if which == "warm" else self.d9.get("cold", {}).get("model_file")
        if not rel:
            return None
        path = project_file(rel)
        if not path.exists():
            raise SystemExit(f"{path.name} missing — run scripts/run_ranker.py first")
        return lgb.Booster(model_file=str(path))

    def train_full(self) -> pd.DataFrame:
        if self._fit is None:
            self._fit = S.load_view("train_full", config.SPLIT_DIR, columns=TCOLS)
        return self._fit

    def two_stage(self):
        if self._ts is None:
            self._ts = build_two_stage(self.cat, self.raw, self.chosen, self.train_full(), self.n_pop,
                                       self.m_faiss, self.f_m, booster=self.booster("warm"))
        return self._ts

    def years(self) -> np.ndarray:
        return pd.to_numeric(self.cat.items["year"], errors="coerce").to_numpy(np.float64)


def sig(a: pd.DataFrame, b: pd.DataFrame, metric: str = "ndcg@10") -> dict:
    m = a[["userId", metric]].merge(b[["userId", metric]], on="userId", suffixes=("_a", "_b"))
    r = paired_bootstrap(m[f"{metric}_a"].to_numpy(), m[f"{metric}_b"].to_numpy(), 2000, config.SEED)
    return {k: (round(v, 5) if isinstance(v, float) else [round(x, 5) for x in v]) for k, v in r.items()}


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


# --------------------------------------------------------------------------- #
# Stage 1: release filter on validation
# --------------------------------------------------------------------------- #
def stage_release(ctx: Context, results: dict) -> None:
    import pyarrow.parquet as pq

    name = "day08_features_val.parquet" if ctx.sel["candidates"] == "union" else "day09_features_val_pop500.parquet"
    path = config.RUNS_DIR / sfx(name, ctx.args.smoke)
    users = np.unique(pq.read_table(path, columns=["userId"]).column("userId").to_numpy().astype(np.int64))
    _, va_u = split_users(users, config.RANKER_VALID_SHARE, config.SEED + 9)       # = Day 9 split
    d = load_rank_data(path).subset(va_u)
    warm = ctx.booster("warm")
    score = predict(warm, d.columns(list(warm.feature_name())).X)
    u, Sm = pad_by_user(d.users, score, -np.inf)
    _, Cm = pad_by_user(d.users, d.movies, PAD)
    del d, score
    gc.collect()
    train = S.load_view("train", config.SPLIT_DIR, columns=["userId", "timestamp"])
    cutoff = user_cutoff_year(only(train, u), u)
    del train
    val = S.load_view("val", config.SPLIT_DIR, columns=COLS)
    truth = only(val, u)
    del val
    years = ctx.years()
    log(f"release: {len(u):,} ranker-valid users (Day 9 split; history = train, labels = val)", ctx.t0)
    Sf = release_filter(Sm, Cm, ctx.cat.index, years, cutoff)
    G = np.asarray(ctx.cat.genres, dtype=np.float32)
    lam, pool = float(ctx.mmr["lambda"]), int(ctx.mmr["pool"])
    space = G if ctx.mmr["space"] == "genre" else None
    if space is None:
        from recsys.embeddings import center_vectors, l2_normalize
        space = center_vectors(ctx.raw) if ctx.chosen["center"] else l2_normalize(ctx.raw)
    lists = {"ranker": top_from_scores(Cm, Sm, KM), "ranker + release filter": top_from_scores(Cm, Sf, KM)}
    for nm, SS in (("ranker + MMR (served)", Sm), ("ranker + MMR + release filter", Sf)):
        parts = [mmr_rerank(Cm[s:s + 2000], SS[s:s + 2000], space, ctx.cat.index, [lam], KM, pool)[lam]
                 for s in range(0, len(u), 2000)]
        lists[nm] = np.vstack(parts)
    rows, pus = [], {}
    for nm, R in lists.items():
        m, pu = evaluate(u, R, truth, KS, return_per_user=True)
        late = released_after(R[:, :10], ctx.cat.index, years, cutoff).sum() / max((R[:, :10] != PAD).sum(), 1)
        rows.append({"model": nm, **{k: round(v, 5) for k, v in m.items() if isinstance(v, float)},
                     "n_users": m["n_users"], "late_share@10": round(float(late), 4)})
        pus[nm] = pu
        log(f"  {nm:32s} ndcg@10={m['ndcg@10']:.4f}  recall@10={m['recall@10']:.4f}  "
            f"top-10 slots released after cutoff {100 * late:.1f}%", ctx.t0)
    s_rank = sig(pus["ranker + release filter"], pus["ranker"])
    s_mmr = sig(pus["ranker + MMR + release filter"], pus["ranker + MMR (served)"])
    on = significant(s_rank)
    ref9 = ctx.sel.get("val_ndcg@10")
    ref11 = next((r["ndcg@10"] for r in ctx.d11.get("tune", {}).get("rows", [])
                  if r.get("space") == ctx.mmr["space"] and r.get("lambda") == lam), None)
    checks = {"ranker_equals_day9": None if ref9 is None else bool(abs(rows[0]["ndcg@10"] - ref9) < 5e-5),
              "day9_val_ndcg@10": ref9,
              "mmr_equals_day11": None if ref11 is None else bool(abs(rows[2]["ndcg@10"] - ref11) < 5e-5),
              "day11_val_ndcg@10": ref11}
    log(f"  filter vs ranker:      {100 * s_rank['rel_change']:+.1f}%  CI [{s_rank['ci95'][0]:+.4f}, "
        f"{s_rank['ci95'][1]:+.4f}] p={s_rank['p_le_0']:.3f}", ctx.t0)
    log(f"  filter vs served MMR:  {100 * s_mmr['rel_change']:+.1f}%  CI [{s_mmr['ci95'][0]:+.4f}, "
        f"{s_mmr['ci95'][1]:+.4f}] p={s_mmr['p_le_0']:.3f}", ctx.t0)
    log(f"  checks: ranker = Day 9 val {checks['ranker_equals_day9']}, MMR = Day 11 val "
        f"{checks['mmr_equals_day11']}", ctx.t0)
    log(f"  decision ({config.SERVE_RELEASE_RULE}): release filter {'ON' if on else 'OFF'} for demo users", ctx.t0)
    results["release"] = {"split": "val (history = train, labels = val), Day 9 ranker-valid users",
                          "n_users": int(len(u)), "cutoff": "year of the user's last train rating",
                          "rule": config.SERVE_RELEASE_RULE, "rows": rows,
                          "significance": {"filter_vs_ranker": s_rank, "filter_vs_served_mmr": s_mmr},
                          "checks": checks, "decision": {"enabled": bool(on)}}


# --------------------------------------------------------------------------- #
# Stage 2: the bundle
# --------------------------------------------------------------------------- #
def stage_bundle(ctx: Context, results: dict) -> None:
    a = ctx.args
    release_on = bool(results.get("release", {}).get("decision", {}).get("enabled", False))
    if "release" not in results:
        log("  (no release stage in the results -> release filter OFF)", ctx.t0)
    fit = ctx.train_full()
    ts = ctx.two_stage()
    test = S.load_view("test", config.SPLIT_DIR, columns=COLS)
    demo = eval_users(test, a.demo_users, config.SEED + 12)
    truth = only(test, demo)
    del test
    hist = only(fit, demo)
    prior = popularity_prior(fit, ctx.cat.index, config.RELEVANCE_THRESHOLD)
    window = ctx.d6.get("selected", {}).get("time_window_days", "all-time")
    enc_key = ctx.chosen["encoder"]
    prefix = (config.EMB_MODELS[enc_key]["query_prefix"]
              if ctx.d6.get("selected", {}).get("query_prefix", True) else "")
    cold_rel = ctx.d9.get("cold", {}).get("model_file")
    settings = {
        "policy": {"warm_from": config.COLD_POLICY_WARM_FROM, "chosen_on": "pseudo-cold validation (Day 9)"},
        "mmr": ctx.mmr,
        "release_filter": {"enabled": release_on, "rule": config.SERVE_RELEASE_RULE,
                           "applies_to": "demo users only (they replay the past); live users: no cutoff"},
        "explain": {"min_cos": config.EXPLAIN_MIN_COS},
        "explore": dict(config.TEXT_EXPLORE),
        "encoder": {"key": enc_key, "hf_name": config.EMB_MODELS[enc_key]["hf_name"], "query_prefix": prefix},
        "pick_rating": config.SERVE_PICK_RATING,
        "poster_base": config.TMDB_POSTER_BASE,
    }
    meta = {"created": datetime.now().isoformat(timespec="seconds"), "git_commit": git_commit(),
            "split_fingerprint": split_fingerprint(),
            "history": f"train+val ({len(fit):,} ratings) — the Day 10 test setting",
            "warm_ranker": {"label": ctx.sel["label"], "trees": ctx.sel["best_iter"],
                            "val_ndcg@10": ctx.sel.get("val_ndcg@10")},
            "cold_prior": f"all-time liked popularity (Day 6 chose '{window}')",
            "demo_users": f"{len(demo):,} seeded warm test users; demo_truth = their test ratings"}
    t1 = time.time()
    man = export_bundle(ctx.bundle_dir, ctx.cat, ctx.raw, ts, warm_model=project_file(ctx.sel["model_file"]),
                        cold_model=project_file(cold_rel) if cold_rel else None, prior=prior,
                        demo_history=hist, demo_truth=truth, settings=settings, meta=meta)
    files = {p.name: round(p.stat().st_size / 2**20, 2) for p in sorted(ctx.bundle_dir.iterdir())}
    log(f"bundle: {ctx.bundle_dir.relative_to(config.ROOT)} — {man['size_mb']} MB, {len(files)} files, "
        f"{len(demo):,} demo users, release filter {'ON' if release_on else 'OFF'} "
        f"({time.time() - t1:.0f}s)", ctx.t0)
    for f, mb in files.items():
        log(f"  {f:22s} {mb:8.2f} MB", ctx.t0)
    results["bundle"] = {"dir": str(ctx.bundle_dir.relative_to(config.ROOT)).replace("\\", "/"),
                         "size_mb": man["size_mb"], "files_mb": files, "n_items": man["n_items"],
                         "n_demo_users": man["n_demo_users"], "settings": settings,
                         "history": meta["history"], "git_commit": meta["git_commit"]}


# --------------------------------------------------------------------------- #
# Stage 3: golden check
# --------------------------------------------------------------------------- #
def same(engine_items: list[dict], ref: np.ndarray) -> bool:
    return [r["movieId"] for r in engine_items] == [int(m) for m in ref if m != PAD]


def stage_check(ctx: Context, results: dict) -> None:
    a = ctx.args
    eng = Engine.load(ctx.bundle_dir)
    ts = ctx.two_stage()
    n = a.check_users
    users = eng.demo_users[:n]
    hist = eng.demo_history
    t1 = time.time()
    R = ts.recommend(rows_of(hist, users), users, 10)
    Rm = ts.recommend(rows_of(hist, users), users, 10, mmr=ctx.mmr)
    ok = ok_m = 0
    for i, uid in enumerate(users):
        ok += same(eng.recommend_user(int(uid), 10, mmr=None, explain=False, release=False)["items"], R[i])
        ok_m += same(eng.recommend_user(int(uid), 10, mmr=ctx.mmr, explain=False, release=False)["items"], Rm[i])
    log(f"check warm: Engine == batch TwoStage for {ok}/{len(users)} users (ranker), "
        f"{ok_m}/{len(users)} (ranker + MMR)", ctx.t0)

    # real cold users (never in training) — picks = their first n liked, real ratings
    seeds = S.load_cold_seeds(config.SPLIT_DIR).sort_values(["userId", "seed_rank"], kind="stable")
    cold_test = S.load_view("cold_test", config.SPLIT_DIR, columns=COLS)
    cu = eval_users(cold_test, n, config.SEED)
    seeds = only(seeds, cu).reset_index(drop=True)
    boosters = {"warm": ts.booster, "cold": ctx.booster("cold")}
    cold_ok = {}
    for k_ in (1, 3, 5):
        h = seeds[seeds["seed_rank"].to_numpy() <= k_]
        fb = ts.feature_batch(h, cu)
        pick = policy_model(k_, config.COLD_POLICY_WARM_FROM)
        b = boosters[pick] if boosters[pick] is not None else boosters["warm"]
        Rc = top_from_scores(fb.candidates, ts.score_with(fb, b, list(b.feature_name())), 10)
        good = 0
        for i, uid in enumerate(cu):
            hu = h[h["userId"] == uid]
            out = eng.cold_start(hu["movieId"].tolist(), hu["rating"].tolist(), k=10, mmr=None, explain=False)
            good += same(out["items"], Rc[i]) and out["model"] == f"{pick} ranker"
        cold_ok[f"picks_{k_}"] = {"identical": int(good), "n": int(len(cu)), "model": f"{pick} ranker"}
        log(f"check cold, {k_} picks ({pick} ranker): Engine == batch for {good}/{len(cu)} users", ctx.t0)

    # served config on ALL demo users vs their test ratings (sanity; 1,000 users -> noisy)
    served, plain = [], []
    for uid in eng.demo_users:
        served.append([r["movieId"] for r in eng.recommend_user(int(uid), KM, explain=False)["items"]])
        plain.append([r["movieId"] for r in eng.recommend_user(int(uid), KM, mmr=None, explain=False,
                                                               release=False)["items"]])
    seen = user_item_matrix(hist, eng.demo_users, eng.index, dtype=np.int8)
    pop = ts.pop.recommend(seen, KM)
    pad = lambda L: np.array([x + [PAD] * (KM - len(x)) for x in L], dtype=np.int64)  # noqa: E731
    sanity = []
    for nm, M in (("popularity", pop), ("ranker", pad(plain)), ("served (MMR"
                  + (" + release filter" if eng.release_filter_default else "") + ")", pad(served))):
        m = evaluate(eng.demo_users, M, eng.demo_truth, KS)
        sanity.append({"model": nm, "ndcg@10": round(m["ndcg@10"], 5), "recall@10": round(m["recall@10"], 5),
                       "n_users": m["n_users"]})
        log(f"  demo users, {nm:28s} ndcg@10={m['ndcg@10']:.4f}", ctx.t0)
    results["check"] = {"warm": {"identical_ranker": int(ok), "identical_mmr": int(ok_m), "n": int(len(users))},
                        "cold": cold_ok, "demo_sanity": sanity, "seconds": round(time.time() - t1, 1),
                        "all_identical": bool(ok == ok_m == len(users) and
                                              all(v["identical"] == v["n"] for v in cold_ok.values()))}


# --------------------------------------------------------------------------- #
# Stage 4: latency
# --------------------------------------------------------------------------- #
def rss_mb() -> float | None:
    """Current memory of this process (psutil). None without psutil — the peak from `resource`
    would include the 31M ratings loaded by earlier stages, which the API never loads."""
    try:
        import psutil
    except ImportError:
        return None
    return round(psutil.Process().memory_info().rss / 2**20, 1)


def stage_latency(ctx: Context, results: dict) -> None:
    a = ctx.args
    rng = np.random.default_rng(config.SEED)
    n, warm_n = a.requests, config.SERVE_LATENCY_WARMUP
    out: dict = {"requests_per_endpoint": n, "warmup": warm_n}
    if a.url:
        import httpx
        client = httpx.Client(base_url=a.url, timeout=60)
        h = client.get("/health").json()
        users = np.array(client.get("/users/demo", params={"limit": 1000}).json()["user_ids"])
        out.update({"target": a.url, "health": h})
        eng = None
        log(f"latency: {a.url} ({h.get('items')} movies, {len(users)} demo users)", ctx.t0)
    else:
        from fastapi.testclient import TestClient

        from recsys.api import create_app
        ctx._fit = ctx._ts = None                      # drop the ratings of earlier stages first
        gc.collect()
        m0 = rss_mb()
        t1 = time.time()
        eng = Engine.load(ctx.bundle_dir)
        out["engine_load_s"] = round(time.time() - t1, 2)
        out["rss_after_load_mb"] = rss_mb()
        out["rss_before_load_mb"] = m0
        if m0 is not None:
            out["engine_memory_mb"] = round(out["rss_after_load_mb"] - m0, 1)
        client = TestClient(create_app(engine=eng))
        users = eng.demo_users
        out["target"] = "in-process (FastAPI TestClient: routing + validation + JSON, no network)"
        log(f"latency: engine loaded in {out['engine_load_s']}s, +{out.get('engine_memory_mb')} MB", ctx.t0)
    popular = (eng.index.ids[np.argsort(-eng.ts.pop.score)[:300]] if eng is not None else
               np.array([i["movieId"] for q in ("the", "a", "of", "man", "love") for i in
                         client.get("/movies/search", params={"q": q, "limit": 50}).json()["items"]]))
    queries = config.DEMO_QUERIES
    titles = ["star", "love", "the dark", "matrix", "toy", "king", "night", "war", "life", "man"]

    def picks(k_):
        return [int(x) for x in rng.choice(popular, size=k_, replace=False)]

    cases = {
        "warm user — served (ranker + MMR + explanations)":
            lambda i: client.get(f"/recommend/{int(users[i % len(users)])}"),
        "warm user — ranker only": lambda i: client.get(f"/recommend/{int(users[i % len(users)])}",
                                                        params={"diversify": False, "explain": False}),
        "cold start — 0 picks (popularity)": lambda i: client.post("/recommend/cold-start", json={}),
        "cold start — 1 pick (cold ranker)": lambda i: client.post("/recommend/cold-start", json={"picks": picks(1)}),
        "cold start — 3 picks (warm ranker)": lambda i: client.post("/recommend/cold-start", json={"picks": picks(3)}),
        "cold start — 5 picks (warm ranker)": lambda i: client.post("/recommend/cold-start", json={"picks": picks(5)}),
        "free text — explore mode": lambda i: client.post("/recommend/cold-start",
                                                          json={"text": queries[i % len(queries)]}),
        "title search": lambda i: client.get("/movies/search", params={"q": titles[i % len(titles)]}),
    }
    rows = []
    for name, call in cases.items():
        if name.startswith("free text"):
            t1 = time.time()
            r = call(0)
            if r.status_code != 200:
                log(f"  {name}: skipped ({r.status_code}: {r.json().get('detail')})", ctx.t0)
                rows.append({"endpoint": name, "n": 0, "note": f"unavailable ({r.status_code})"})
                continue
            out["encoder_first_request_s"] = round(time.time() - t1, 2)
        for i in range(warm_n):
            call(i)
        wall, inner = [], []
        for i in range(n):
            t1 = time.perf_counter()
            r = call(i)
            wall.append(1000 * (time.perf_counter() - t1))
            if r.status_code != 200:
                raise SystemExit(f"{name}: HTTP {r.status_code} {r.text[:200]}")
            inner.append(r.json().get("latency_ms", np.nan))
        s = latency_summary(wall)
        rows.append({"endpoint": name, **s, "engine_p50": round(float(np.nanmedian(inner)), 2)})
        log(f"  {name:52s} p50 {s['p50']:7.2f} ms  p95 {s['p95']:7.2f} ms  p99 {s['p99']:7.2f} ms", ctx.t0)
    out["rows"] = rows
    if eng is not None:
        steps: dict = {}
        mmr = eng.mmr_default
        for i in range(n):
            uid = int(users[i % len(users)])
            eng.rank(eng.user_history(uid), 10, "warm", mmr, True,
                     eng.cutoff_year(eng.user_history(uid)) if eng.release_filter_default else None, steps)
        out["warm_breakdown_ms"] = {k: round(v / n, 2) for k, v in steps.items()}
        log("  warm served, mean ms per step: " +
            ", ".join(f"{k[:-3]} {v}" for k, v in out["warm_breakdown_ms"].items()), ctx.t0)
        out["rss_end_mb"] = rss_mb()
    out["note"] = "one request at a time on this machine; latencies vary run to run — quote as approximate"
    results["latency_server" if a.url else "latency"] = out     # a --url run never replaces the default


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def write_md(res: dict, path: Path) -> None:
    md = "# Day 12 — Serving: FastAPI + bundle + latency\n\n"
    if "release" in res:
        r = res["release"]
        s = r["significance"]
        md += (f"## Release filter — validation ({r['n_users']:,} Day 9 ranker-valid users)\n\n"
               f"Rule (written before the run): {r['rule']}.\n\n"
               + md_table(r["rows"], ["model", "ndcg@10", "recall@10", "mrr@10", "late_share@10"], left=("model",))
               + f"\nFilter vs ranker: {100 * s['filter_vs_ranker']['rel_change']:+.1f}% "
                 f"(CI [{s['filter_vs_ranker']['ci95'][0]:+.4f}, {s['filter_vs_ranker']['ci95'][1]:+.4f}], "
                 f"p = {s['filter_vs_ranker']['p_le_0']:.3f}); with the served MMR "
                 f"{100 * s['filter_vs_served_mmr']['rel_change']:+.1f}% (p = {s['filter_vs_served_mmr']['p_le_0']:.3f}).\n\n"
               f"**Decision: release filter {'ON' if r['decision']['enabled'] else 'OFF'}** for demo users. "
               f"Checks: ranker = Day 9 val {r['checks']['ranker_equals_day9']}, "
               f"MMR = Day 11 val {r['checks']['mmr_equals_day11']}.\n\n"
               "`late_share@10` = share of top-10 slots released after the user's cutoff year.\n\n")
    if "bundle" in res:
        b = res["bundle"]
        md += (f"## Bundle — `{b['dir']}` ({b['size_mb']} MB, {b['n_items']:,} movies, "
               f"{b['n_demo_users']:,} demo users)\n\nHistory: {b['history']}. Settings: MMR {b['settings']['mmr']}, "
               f"policy warm ranker from {b['settings']['policy']['warm_from']} picks, release filter "
               f"{'ON' if b['settings']['release_filter']['enabled'] else 'OFF'}, explore {b['settings']['explore']}.\n\n"
               "| file | MB |\n|---|---:|\n" + "".join(f"| {f} | {mb} |\n" for f, mb in b["files_mb"].items()) + "\n")
    if "check" in res:
        c = res["check"]
        w = c["warm"]
        md += ("## Golden check — API engine (bundle only) vs the evaluated batch code\n\n"
               f"- Warm users: identical top-10 for {w['identical_ranker']}/{w['n']} (ranker) and "
               f"{w['identical_mmr']}/{w['n']} (ranker + MMR)\n"
               + "".join(f"- Real cold users, {k.split('_')[1]} pick(s) ({v['model']}): identical for "
                         f"{v['identical']}/{v['n']}\n" for k, v in c["cold"].items())
               + f"- **All identical: {'✓' if c['all_identical'] else '✗ — investigate'}**\n\n"
               f"Demo users (sanity only — {c['demo_sanity'][0]['n_users']:,} users, noisy; the CV numbers are "
               "Day 10/11):\n\n"
               + md_table(c["demo_sanity"], ["model", "ndcg@10", "recall@10", "n_users"], left=("model",)) + "\n")
    for key in ("latency", "latency_server"):
        if key not in res:
            continue
        L = res[key]
        md += (f"## Latency — {L['requests_per_endpoint']} sequential requests per endpoint, after "
               f"{L['warmup']} warm-up\n\nTarget: {L['target']}.\n\n"
               + md_table([{k: (f"{v:.1f}" if isinstance(v, float) else v) for k, v in r.items()}
                           for r in L["rows"] if r.get("n")],
                          ["endpoint", "p50", "p95", "p99", "mean", "engine_p50"], left=("endpoint",)))
        md += "\nAll times in ms. `engine_p50` = time inside the engine (the rest is HTTP, validation, JSON).\n"
        if L.get("warm_breakdown_ms"):
            md += ("\nWarm served request, mean ms per step: "
                   + ", ".join(f"{k[:-3]} {v}" for k, v in L["warm_breakdown_ms"].items()) + ".\n")
        extra = [f"engine load {L['engine_load_s']} s" if "engine_load_s" in L else None,
                 f"engine adds ≈ {L['engine_memory_mb']} MB of memory (process {L['rss_after_load_mb']} MB)"
                 if L.get("engine_memory_mb") is not None else None,
                 f"first text request (loads the encoder) {L['encoder_first_request_s']} s"
                 if "encoder_first_request_s" in L else None]
        md += "\n" + "; ".join(x for x in extra if x) + ("." if any(extra) else "") + "\n\n" + L["note"] + ".\n"
    path.write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES)
    p.add_argument("--smoke", action="store_true", help="small run against the *_smoke results")
    p.add_argument("--url", help="latency stage: benchmark a running server instead of in-process")
    p.add_argument("--requests", type=int, default=None)
    p.add_argument("--demo-users", type=int, default=None)
    p.add_argument("--check-users", type=int, default=None)
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    sm = args.smoke
    args.requests = args.requests or (30 if sm else config.SERVE_LATENCY_REQUESTS)
    args.demo_users = args.demo_users or (100 if sm else config.SERVE_DEMO_USERS)
    args.check_users = args.check_users or (30 if sm else config.SERVE_CHECK_USERS)
    t0 = time.time()
    ctx = Context(args, t0)
    path, results = open_results("day12_serving_smoke" if sm else "day12_serving")
    results["split_fingerprint"] = split_fingerprint()
    fn = {"release": stage_release, "bundle": stage_bundle, "check": stage_check, "latency": stage_latency}
    for st in STAGES:
        if st in args.stages:
            t1 = time.time()
            fn[st](ctx, results)
            results.setdefault("seconds_last_run", {})[st] = round(time.time() - t1, 1)
            save_results(path, results)
    write_md(results, path.with_suffix(".md"))
    log(f"wrote {path.relative_to(config.ROOT)} + .md", t0)


if __name__ == "__main__":
    main()
