"""Day 3: popularity + TF-IDF content-based baselines on the frozen split.

Stages (all by default):
  tune  fit on TRAIN, score on VAL for a sample of users; pick each model's settings
  test  fit on TRAIN+VAL, score on TEST for ALL users  -> the numbers that go in the README
  cold  held-out cold-start users with 0 / 1 / 3 / 5 onboarding seeds

Outputs
  reports/results/day03_baselines.json  + .md    (small, commit these)
  data/runs/day03_<model>_test_per_user.parquet   (per-user metrics for later significance tests)
  data/runs/tfidf_knn_<N>.npz                     (cached item-item neighbours)

Usage
  python scripts/run_baselines.py --smoke          # ~1-2 min check on a few thousand users
  python scripts/run_baselines.py                  # full run
  python scripts/run_baselines.py --stages test cold   # reuse settings chosen earlier
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # works without `pip install -e .`
from recsys import config  # noqa: E402
from recsys import split as S  # noqa: E402
from recsys.baselines import (ItemIndex, PopularityRecommender,  # noqa: E402
                              TfidfKNNRecommender, popularity_scores, user_item_matrix)
from recsys.experiment import (COLS, KMAX, SHOW, eval_users, head_items, load_catalog,  # noqa: E402
                               log, md_table, only, open_results, row, save_results,
                               split_fingerprint)
from recsys.metrics import beyond_accuracy, evaluate  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402


def similarity(docs: pd.Series, index: ItemIndex, n: int, rebuild: bool, t0: float) -> sp.csr_matrix:
    path = config.RUNS_DIR / f"tfidf_knn_{n}.npz"
    ids_path = path.with_suffix(".ids.npy")
    if path.exists() and ids_path.exists() and not rebuild:
        if np.array_equal(np.load(ids_path), index.ids):
            log(f"loaded cached TF-IDF neighbours {path.name}", t0)
            return sp.load_npz(path).tocsr()
    model = TfidfKNNRecommender(n_neighbors=n).fit_items(docs, index)
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, model.sim)
    np.save(ids_path, index.ids)
    log(f"built TF-IDF ({model.vocab_size:,} terms) + {n}-NN graph, nnz={model.sim.nnz:,}", t0)
    return model.sim


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stages", nargs="+", default=["tune", "test", "cold"],
                   choices=["tune", "test", "cold"])
    p.add_argument("--tune-users", type=int, default=config.TUNE_USERS)
    p.add_argument("--test-users", type=int, default=0, help="0 = all test users")
    p.add_argument("--neighbors", type=int, default=config.KNN_NEIGHBORS)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--rebuild-knn", action="store_true")
    p.add_argument("--smoke", action="store_true", help="tiny run to check everything works")
    args = p.parse_args()
    if args.smoke:
        args.tune_users, args.test_users = 3000, 5000
    tag = "day03_baselines_smoke" if args.smoke else "day03_baselines"
    out_json, results = open_results(tag)
    results["split_fingerprint"] = split_fingerprint()
    t0 = time.time()

    # ---------------- catalog + item text ----------------
    cat = load_catalog()
    index, docs, genres = cat.index, cat.docs, cat.genres
    log(f"catalog: {len(index):,} movies; example document:\n    {docs.iloc[0][:200]}...", t0)
    sim = similarity(docs, index, args.neighbors, args.rebuild_knn, t0)

    def tfidf(cfg: dict) -> TfidfKNNRecommender:
        m = TfidfKNNRecommender(n_neighbors=args.neighbors, threshold=config.RELEVANCE_THRESHOLD, **cfg)
        return m.set_similarity(sim, index)

    selected = results.get("selected", {"popularity": {"mode": "count"},
                                        "tfidf_knn": dict(config.TFIDF_DEFAULT)})

    # ---------------- tune on validation ----------------
    if "tune" in args.stages:
        train = S.load_view("train", config.SPLIT_DIR, columns=COLS)
        val = S.load_view("val", config.SPLIT_DIR, columns=COLS)
        users = eval_users(val, args.tune_users, config.SEED)
        val, train_u = only(val, users), only(train, users)
        seen = user_item_matrix(train_u, users, index, dtype=np.int8)
        log(f"tune: {len(users):,} val users, train {len(train):,} ratings", t0)
        runs, pop_score = [], {}
        for mode in config.POP_MODES:
            pop = PopularityRecommender(mode, config.RELEVANCE_THRESHOLD).fit(train, index)
            pop_score[mode] = pop.score
            m = evaluate(users, pop.recommend(seen, KMAX), val, config.EVAL_KS)
            runs.append(row("popularity", m, {"config": f"mode={mode}"}))
            log(f"  popularity mode={mode:6s} ndcg@10={m['ndcg@10']:.4f}", t0)
        best_pop = max((r for r in runs), key=lambda r: r["ndcg@10"])["config"].split("=")[1]
        for cfg in config.TFIDF_GRID:
            model = tfidf(cfg)
            W = model.user_weights(train_u, users)
            recs = model.recommend(W, seen, KMAX, pop_score[best_pop], args.batch_size)
            m = evaluate(users, recs, val, config.EVAL_KS)
            name = f"weighting={cfg['weighting']}, history={cfg['max_history'] or 'all'}"
            runs.append(row("tfidf_knn", m, {"config": name}))
            log(f"  tfidf_knn {name:35s} ndcg@10={m['ndcg@10']:.4f}", t0)
        tf_runs = [r for r in runs if r["model"] == "tfidf_knn"]
        best_tf = config.TFIDF_GRID[int(np.argmax([r["ndcg@10"] for r in tf_runs]))]
        selected = {"popularity": {"mode": best_pop}, "tfidf_knn": dict(best_tf)}
        results["tuning"] = {"split": "val (fit on train)", "n_users": int(len(users)), "runs": runs}
        results["selected"] = selected
        log(f"selected: {selected}", t0)
        del train, val, train_u, seen
        gc.collect()

    pop_mode = selected["popularity"]["mode"]
    tf_cfg = selected["tfidf_knn"]
    train_full = None

    # ---------------- final test ----------------
    if "test" in args.stages:
        train_full = S.load_view("train_full", config.SPLIT_DIR, columns=COLS)
        test = S.load_view("test", config.SPLIT_DIR, columns=COLS)
        users = eval_users(test, args.test_users, config.SEED)
        test = only(test, users)
        seen = user_item_matrix(only(train_full, users), users, index, dtype=np.int8)
        head = head_items(train_full["movieId"].to_numpy(), index)
        n_head = len(head)
        log(f"test: {len(users):,} users, train_full {len(train_full):,} ratings", t0)

        pop = PopularityRecommender(pop_mode, config.RELEVANCE_THRESHOLD).fit(train_full, index)

        def run_tfidf() -> np.ndarray:
            model = tfidf(tf_cfg)
            W = model.user_weights(only(train_full, users), users)
            return model.recommend(W, seen, KMAX, pop.score, args.batch_size)

        test_rows = []
        for name, make in (("popularity", lambda: pop.recommend(seen, KMAX)),
                           ("tfidf_knn", run_tfidf)):
            t1 = time.time()
            recs = make()
            secs = time.time() - t1
            m, per_user = evaluate(users, recs, test, config.EVAL_KS, return_per_user=True)
            ba = beyond_accuracy(recs, index.ids, head, genres, k=10)
            per_user.to_parquet(config.RUNS_DIR / f"{tag.replace('baselines', name)}_test_per_user.parquet",
                                index=False)
            cfg = f"mode={pop_mode}" if name == "popularity" else \
                f"weighting={tf_cfg['weighting']}, history={tf_cfg['max_history'] or 'all'}"
            test_rows.append(row(name, m, {**ba, "config": cfg,
                                           "ms_per_user": 1000 * secs / len(users)}))
            log(f"  {name:10s} ndcg@10={m['ndcg@10']:.4f} recall@10={m['recall@10']:.4f} "
                f"coverage@10={ba['coverage@10']:.3f} ({secs:.0f}s)", t0)
            del recs, per_user
            gc.collect()
        results["test"] = {"split": "test (fit on train+val)", "n_users": int(len(users)),
                           "head_items": n_head, "rows": test_rows}
        del test, seen
        gc.collect()

    # ---------------- cold-start users ----------------
    if "cold" in args.stages:
        if train_full is None:
            train_full = S.load_view("train_full", config.SPLIT_DIR, columns=["movieId", "rating"])
        fallback = popularity_scores(train_full, index, pop_mode, config.RELEVANCE_THRESHOLD)
        pop = PopularityRecommender(pop_mode, config.RELEVANCE_THRESHOLD)
        pop.index, pop.score = index, fallback
        del train_full
        gc.collect()
        seeds = S.load_cold_seeds(config.SPLIT_DIR).sort_values(["userId", "seed_rank"])
        cold_test = S.load_view("cold_test", config.SPLIT_DIR, columns=COLS)
        users = eval_users(cold_test, None, config.SEED)
        # seeds are all liked movies -> binary weights; history cap irrelevant for <= 5 seeds
        cold_model = tfidf({"weighting": "liked", "max_history": None})
        cold_rows = []
        for n in config.COLD_SEED_COUNTS:
            s = seeds[seeds["seed_rank"] <= n]
            seen = user_item_matrix(s, users, index, dtype=np.int8)
            for name, recs in (
                ("popularity", pop.recommend(seen, KMAX)),
                ("tfidf_knn", cold_model.recommend(cold_model.user_weights(s, users), seen,
                                                   KMAX, fallback, args.batch_size)),
            ):
                m = evaluate(users, recs, cold_test, config.EVAL_KS)
                cold_rows.append(row(name, m, {"n_seeds": n}))
            log(f"  cold seeds={n}: popularity ndcg@10={cold_rows[-2]['ndcg@10']:.4f}  "
                f"tfidf_knn ndcg@10={cold_rows[-1]['ndcg@10']:.4f}", t0)
        results["cold_start"] = {"split": "cold_test (users never in training)",
                                 "n_users": int(len(users)), "rows": cold_rows,
                                 "note": "0 seeds: TF-IDF has no profile and falls back to popularity"}

    results["seconds_last_run"] = round(time.time() - t0, 1)
    save_results(out_json, results)
    write_markdown(results, out_json.with_suffix(".md"))
    log(f"wrote {out_json.relative_to(config.ROOT)} and .md", t0)


def write_markdown(res: dict, path: Path) -> None:
    md = "# Day 3 — Baselines\n\n"
    md += f"Split fingerprint `{res['split_fingerprint'][:12]}…`. Relevant = rating ≥ 4. "
    md += "Already-rated movies are never recommended.\n\n"
    if "test" in res:
        t = res["test"]
        md += f"## Test — fit on train+val, {t['n_users']:,} users\n\n"
        md += md_table(t["rows"], ["model", "config"] + SHOW)
        md += "\n" + md_table(t["rows"], ["model", "coverage@10", "long_tail@10", "diversity@10",
                                          "ms_per_user"])
        md += (f"\nlong_tail@10 = share of recommended slots outside the {t['head_items']:,} "
               "most-rated movies; diversity@10 = mean pairwise (1 − cosine) of genre vectors.\n\n")
    if "cold_start" in res:
        c = res["cold_start"]
        md += f"## Cold-start users — {c['n_users']:,} users never seen in training\n\n"
        md += md_table(c["rows"], ["model", "n_seeds", "ndcg@10", "recall@10", "mrr@10", "hit_rate@10"])
        md += f"\n{c['note']}.\n\n"
    if "tuning" in res:
        tu = res["tuning"]
        md += f"## Tuning — fit on train, scored on val ({tu['n_users']:,} sampled users)\n\n"
        md += md_table(tu["runs"], ["model", "config", "ndcg@10", "recall@10", "mrr@10"])
        md += f"\nSelected: `{json.dumps(res['selected'])}`\n"
    path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
