"""Day 8: stage-1 candidates (popular top-N ∪ FAISS top-M) + leak-free ranker features.

Stage 1 must CONTAIN the movies a user will like; the Day 9 ranker can only reorder them. Day 5
showed popularity retrieves far more of them than embeddings, but the two lists barely overlap
(3.5%), so a union adds movies popularity never finds. This script measures how much, picks the
union on validation, and writes the (user, candidate) feature table the ranker trains on.

Uses the earlier choices from their json files (Day 3 popularity mode, Day 4 encoder + centering,
Day 5 profile) and the cached item vectors — nothing is re-encoded.

Stages (all by default):
  recall    VALIDATION (fit on train, same 30k users as Days 3-5): recall of every union
            popular top-N ∪ FAISS top-M (N x M grid), next to popularity alone at the same size
            -> picks the union for Day 9 (best recall with FAISS in it, at most CANDIDATE_MAX)
  features  VALIDATION, RANKER_USERS users: features for every (user, candidate) pair from TRAIN
            only, labels = their val ratings -> data/runs/day08_features_val.parquet (Day 9 input).
            Checks: popularity order reproduces the popularity baseline exactly; each feature
            alone as a ranker (a single feature far above the rest would point to leakage);
            the oracle = the best NDCG@10 any ranker could reach with these candidates
  test      TEST (fit on train+val, all warm test users): recall of the selected union vs
            popularity and FAISS alone — the stage-1 ceiling for the final Day 10 numbers

Outputs
  reports/results/day08_candidates.json + .md                         (commit)
  data/runs/day08_features_val.parquet                                (git-ignored, Day 9)

Usage
  python scripts/run_candidates.py --smoke              # ~2-3 min check
  python scripts/run_candidates.py                      # full run
  python scripts/run_candidates.py --stages features    # one stage (others are kept in the json)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # works without `pip install -e .`
from recsys.experiment import setup_runtime  # noqa: E402

setup_runtime()          # Windows OpenMP fix; must run before faiss is imported

import argparse  # noqa: E402
import gc  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from recsys import config  # noqa: E402
from recsys import split as S  # noqa: E402
from recsys.baselines import PopularityRecommender, user_item_matrix  # noqa: E402
from recsys.candidates import faiss_candidates, union_candidates  # noqa: E402
from recsys.embeddings import EmbeddingRecommender, FlatIndex  # noqa: E402
from recsys.experiment import (COLS, TCOLS, chosen_settings, eval_users, load_catalog,  # noqa: E402
                               load_item_vectors, log, md_table, only, open_results, rows_of,
                               save_results, split_fingerprint)
from recsys.features import (FEATURE_NAMES, FEATURES, FeatureBuilder,  # noqa: E402
                             assert_no_label_in_history, attach_labels, oracle_topk,
                             rank_by_feature)
from recsys.metrics import PAD, evaluate  # noqa: E402

FEATURES_FILE = "day08_features_val.parquet"


class Context:
    """Catalog, vectors and earlier choices, loaded once."""

    def __init__(self, args, t0: float) -> None:
        self.args, self.t0 = args, t0
        cat = load_catalog()
        self.cat, self.index = cat, cat.index
        chosen = chosen_settings()
        self.model_key, self.center = chosen["encoder"], chosen["center"]
        self.pop_mode, self.profile = chosen["popularity_mode"], chosen["profile"]
        self.raw_vectors = load_item_vectors(cat, self.model_key)
        self.emb = EmbeddingRecommender(self.raw_vectors, self.index, weighting=self.profile["weighting"],
                                        threshold=config.RELEVANCE_THRESHOLD, center=self.center,
                                        half_life=self.profile["half_life"])
        self.flat = FlatIndex(self.emb.item_vectors, self.index.ids)   # same vectors as the profiles
        log(f"catalog {len(self.index):,} movies; encoder {self.model_key} "
            f"({'mean-centered' if self.center else 'raw'}); profile {self.profile}; "
            f"popularity={self.pop_mode}", t0)

    def retrieve(self, hist_u: pd.DataFrame, users: np.ndarray, pop: PopularityRecommender,
                 n_pop: int, m_faiss: int) -> tuple[np.ndarray, np.ndarray]:
        """Both stage-1 sources for these users: (popular top-n, FAISS top-m), seen removed."""
        seen = user_item_matrix(hist_u, users, self.index, dtype=np.int8)
        P = self.emb.user_profiles(hist_u, users)
        return (pop.recommend(seen, n_pop),
                faiss_candidates(self.flat, P, seen, m_faiss, self.args.faiss_batch))

    def builder(self) -> FeatureBuilder:
        items = self.cat.items
        years = pd.to_numeric(items["year"], errors="coerce").to_numpy(np.float64)
        runtime = (pd.to_numeric(items["runtime"], errors="coerce").replace(0, np.nan)
                   .to_numpy(np.float64) if "runtime" in items else None)
        return FeatureBuilder(self.raw_vectors, self.index, self.cat.genres, years, runtime,
                              center=self.center, weighting=self.profile["weighting"],
                              half_life=self.profile["half_life"],
                              threshold=config.RELEVANCE_THRESHOLD,
                              maxcos_history=config.MAXCOS_HISTORY,
                              recent_days=config.RECENT_DAYS, prior_votes=config.AVG_RATING_PRIOR)


def name(n: int, m: int) -> str:
    return f"pop {n} + faiss {m}" if m else f"pop {n}"


def selected_sizes(results: dict) -> dict:
    s = results.get("selected", config.CANDIDATE_DEFAULT)
    return {"pop": int(s["pop"]), "faiss": int(s["faiss"])}


# --------------------------------------------------------------------------- #
def stage_recall(ctx: Context, results: dict) -> None:
    a = ctx.args
    train = S.load_view("train", config.SPLIT_DIR, columns=COLS)
    val = S.load_view("val", config.SPLIT_DIR, columns=COLS)
    users = eval_users(val, a.tune_users, config.SEED)               # = the Day 3-5 sample
    truth = only(val, users)
    pop = PopularityRecommender(ctx.pop_mode, config.RELEVANCE_THRESHOLD).fit(train, ctx.index)
    hist = only(train, users)
    del train
    gc.collect()
    Ns, Ms = config.CANDIDATE_POP_N, config.CANDIDATE_FAISS_M
    t1 = time.time()
    pop_l, faiss_l = ctx.retrieve(hist, users, pop, max(Ns) + max(Ms), max(Ms))
    log(f"recall: {len(users):,} val users, retrieval {1000 * (time.time() - t1) / len(users):.2f} ms/user", ctx.t0)

    def recall(C: np.ndarray, k: int) -> dict:
        m = evaluate(users, C, truth, (k,))
        return {"recall": m[f"recall@{k}"], "hit_rate": m[f"hit_rate@{k}"]}

    pop_at = {k: recall(pop_l, k) for k in sorted({n + m for n in Ns for m in Ms})}
    rows = []
    for n in Ns:
        for m in Ms:
            C, _ = union_candidates({"pop": pop_l, "faiss": faiss_l}, {"pop": n, "faiss": m})
            r = recall(C, n + m)
            ref = pop_at[n + m]
            rows.append({"candidates": name(n, m), "pop": n, "faiss": m, "size": n + m,
                         "avg_unique": round(float((C != PAD).sum(axis=1).mean()), 1),
                         "recall": round(r["recall"], 4), "hit_rate": round(r["hit_rate"], 4),
                         "recall_gain_from_faiss": round(r["recall"] - pop_at[n]["recall"], 4),
                         "pop_same_size_recall": round(ref["recall"], 4),
                         "vs_pop_same_size": round(r["recall"] - ref["recall"], 4)})
            log(f"  {name(n, m):20s} recall@{n + m:<4d}={r['recall']:.4f} "
                f"(popularity alone at {n + m}: {ref['recall']:.4f}; FAISS adds "
                f"{rows[-1]['recall_gain_from_faiss']:+.4f} over pop {n})", ctx.t0)
    faiss_rows = [{"candidates": f"faiss {m}", "size": m, **{k: round(v, 4) for k, v in
                                                            recall(faiss_l, m).items()}}
                  for m in Ms if m]
    ok = [r for r in rows if r["faiss"] > 0 and r["size"] <= config.CANDIDATE_MAX]
    best = max(ok, key=lambda r: (r["recall"], -r["size"]))
    pop_best = max((r for r in rows if r["faiss"] == 0 and r["size"] <= config.CANDIDATE_MAX),
                   key=lambda r: r["recall"])
    # Share of the selected union's recall that only FAISS contributed: union = pop top-N plus
    # FAISS-only movies, so that share = (recall(union) - recall(pop top-N)) / recall(union)
    faiss_share = best["recall_gain_from_faiss"] / best["recall"] if best["recall"] else 0.0
    results["recall"] = {"split": "val (fit on train)", "n_users": int(len(users)),
                         "max_size": config.CANDIDATE_MAX, "rows": rows, "faiss_rows": faiss_rows,
                         "pop_only_best": pop_best,
                         "selected_recall_share_only_faiss": round(float(faiss_share), 4)}
    results["selected"] = {"pop": best["pop"], "faiss": best["faiss"], "val_recall": best["recall"],
                           "rule": f"highest val recall with FAISS in the union, size <= {config.CANDIDATE_MAX}"}
    log(f"selected: {name(best['pop'], best['faiss'])} recall={best['recall']:.4f} "
        f"(popularity alone at {best['size']}: {best['pop_same_size_recall']:.4f}; best popularity-only "
        f"<= {config.CANDIDATE_MAX}: {pop_best['candidates']} {pop_best['recall']:.4f}); "
        f"{100 * results['recall']['selected_recall_share_only_faiss']:.1f}% of the relevant movies "
        f"in it come only from FAISS", ctx.t0)


def stage_features(ctx: Context, results: dict) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    a = ctx.args
    sizes = selected_sizes(results)
    train = S.load_view("train", config.SPLIT_DIR, columns=TCOLS)
    val = S.load_view("val", config.SPLIT_DIR, columns=COLS)
    users = eval_users(val, a.ranker_users, config.SEED + 8)       # Day 9 splits these by user
    truth = only(val, users)
    pop = PopularityRecommender(ctx.pop_mode, config.RELEVANCE_THRESHOLD).fit(train, ctx.index)
    builder = ctx.builder().fit(train)                              # history = TRAIN only
    hist = only(train, users)
    del train
    gc.collect()
    assert_no_label_in_history(hist, truth)
    log(f"features: {len(users):,} val users, candidates {name(sizes['pop'], sizes['faiss'])}, "
        f"history = train (cutoff {pd.Timestamp(builder.stats['cutoff'], unit='s').date()}), "
        f"labels = val", ctx.t0)

    out_path = config.RUNS_DIR / (FEATURES_FILE if not a.smoke else FEATURES_FILE.replace(".parquet", "_smoke.parquet"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    k = 10
    tops = {f: [] for f in FEATURE_NAMES if ctx_direction(f) != 0}
    tops["(union order)"], tops["(oracle)"], pop_direct = [], [], []
    n_rows = n_pos = 0
    sums = {f: np.zeros(4) for f in FEATURE_NAMES}               # pos sum, pos n, neg sum, neg n
    t_feat = 0.0
    for s in range(0, len(users), a.batch_users):
        bu = users[s:s + a.batch_users]
        h = rows_of(hist, bu)
        t1 = time.time()
        pop_l, faiss_l = ctx.retrieve(h, bu, pop, sizes["pop"], sizes["faiss"])
        C, ranks = union_candidates({"pop": pop_l, "faiss": faiss_l}, sizes)
        fb = builder.transform(bu, h, C, ranks)
        t_feat += time.time() - t1
        tb = rows_of(truth, bu)
        df = attach_labels(fb.to_frame(), tb)
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        writer = writer or pq.ParquetWriter(out_path, tbl.schema, compression="zstd")
        writer.write_table(tbl)
        rel = df["rating"].to_numpy() >= config.RELEVANCE_THRESHOLD
        n_rows, n_pos = n_rows + len(df), n_pos + int(rel.sum())
        for f in FEATURE_NAMES:
            x = df[f].to_numpy(np.float64)
            for j, m in enumerate((rel, ~rel)):
                ok = m & ~np.isnan(x)
                sums[f][2 * j] += x[ok].sum()
                sums[f][2 * j + 1] += ok.sum()
        for f in tops:
            if f == "(union order)":
                tops[f].append(C[:, :k])
            elif f == "(oracle)":
                tops[f].append(oracle_topk(fb, tb, k))
            else:
                tops[f].append(rank_by_feature(fb, f, k))
        pop_direct.append(pop_l[:, :k])
        if (s // a.batch_users) % 10 == 0:
            log(f"  {min(s + a.batch_users, len(users)):,}/{len(users):,} users, {n_rows:,} rows", ctx.t0)
        del fb, df, tbl
    writer.close()

    # Check: sorting the candidates by pop_rank must give exactly the popularity baseline
    same = float(np.mean(np.all(np.vstack(tops["pop_rank"]) == np.vstack(pop_direct), axis=1)))
    log(f"  check: pop_rank order == popularity baseline for {100 * same:.2f}% of users "
        f"-> {'reproduced' if same == 1.0 else 'DIFFERENT — investigate'}", ctx.t0)
    one = []
    for f, parts in tops.items():
        m = evaluate(users, np.vstack(parts), truth, (k,))
        one.append({"feature": f, "ndcg@10": round(m["ndcg@10"], 4), "recall@10": round(m["recall@10"], 4),
                    "mrr@10": round(m["mrr@10"], 4)})
    one.sort(key=lambda r: -r["ndcg@10"])
    desc = {f: d for f, _, d in FEATURES}
    stats = []
    for f in FEATURE_NAMES:
        ps, pn, ns, nn = sums[f]
        stats.append({"feature": f, "mean_relevant": round(ps / pn, 4) if pn else None,
                      "mean_other": round(ns / nn, 4) if nn else None,
                      "missing_share": round(1 - (pn + nn) / n_rows, 4), "meaning": desc[f]})
    results["features"] = {
        "split": "history = train, labels = val", "n_users": int(len(users)), "candidates": sizes,
        "rows": int(n_rows), "relevant_rows": int(n_pos), "avg_candidates": round(n_rows / len(users), 1),
        "ms_per_user": round(1000 * t_feat / len(users), 2),
        "file": str(out_path.relative_to(config.ROOT)), "file_mb": round(out_path.stat().st_size / 2**20, 1),
        "pop_rank_reproduces_popularity": same, "one_feature": one, "stats": stats}
    for r in one:
        log(f"  alone: {r['feature']:20s} ndcg@10={r['ndcg@10']:.4f}", ctx.t0)
    log(f"wrote {out_path.name}: {n_rows:,} rows ({n_pos:,} relevant), "
        f"{results['features']['file_mb']} MB, {results['features']['ms_per_user']} ms/user", ctx.t0)


def ctx_direction(f: str) -> int:
    return {n: d for n, d, _ in FEATURES}[f]


def stage_test(ctx: Context, results: dict) -> None:
    a = ctx.args
    sizes = selected_sizes(results)
    K = sizes["pop"] + sizes["faiss"]
    train_full = S.load_view("train_full", config.SPLIT_DIR, columns=COLS)
    test = S.load_view("test", config.SPLIT_DIR, columns=COLS)
    users = eval_users(test, a.test_users, config.SEED)
    test = only(test, users)
    pop = PopularityRecommender(ctx.pop_mode, config.RELEVANCE_THRESHOLD).fit(train_full, ctx.index)
    hist = only(train_full, users)
    del train_full
    gc.collect()
    log(f"test: {len(users):,} users (fit on train+val), union {name(sizes['pop'], sizes['faiss'])}", ctx.t0)
    lists = {"union": [], "pop_same": [], "pop_n": [], "faiss": []}
    per = {key: [] for key in lists}
    t1 = time.time()
    for s in range(0, len(users), a.test_batch):
        bu = users[s:s + a.test_batch]
        h, tr = rows_of(hist, bu), rows_of(test, bu)
        pop_l, faiss_l = ctx.retrieve(h, bu, pop, K, sizes["faiss"])
        C, _ = union_candidates({"pop": pop_l, "faiss": faiss_l}, sizes)
        for key, R, k in (("union", C, K), ("pop_same", pop_l, K),
                          ("pop_n", pop_l[:, :sizes["pop"]], sizes["pop"]), ("faiss", faiss_l, sizes["faiss"])):
            if k == 0:
                continue
            _, pu = evaluate(bu, R, tr, (k,), return_per_user=True)
            per[key].append(pu[[f"recall@{k}", f"hit_rate@{k}"]].to_numpy())
        if (s // a.test_batch) % 10 == 0:
            log(f"  {min(s + a.test_batch, len(users)):,}/{len(users):,} users", ctx.t0)
    secs = time.time() - t1
    labels = {"union": f"union: {name(sizes['pop'], sizes['faiss'])}",
              "pop_same": f"popularity top-{K} (same size)",
              "pop_n": f"popularity top-{sizes['pop']} (union without FAISS)",
              "faiss": f"FAISS top-{sizes['faiss']} alone"}
    rows = []
    for key, parts in per.items():
        if parts:
            M = np.vstack(parts)
            rows.append({"candidates": labels[key], "recall": round(float(M[:, 0].mean()), 4),
                         "hit_rate": round(float(M[:, 1].mean()), 4)})
    results["test"] = {"split": "test (fit on train+val)", "n_users": int(len(users)),
                       "candidates": sizes, "rows": rows,
                       "ms_per_user_retrieval": round(1000 * secs / len(users), 2)}
    for r in rows:
        log(f"  {r['candidates']:42s} recall={r['recall']:.4f} hit={r['hit_rate']:.4f}", ctx.t0)


# --------------------------------------------------------------------------- #
def write_md(res: dict, path: Path, ctx: Context) -> None:
    md = "# Day 8 — Stage-1 candidates (popular ∪ FAISS) + ranker features\n\n"
    md += (f"Split fingerprint `{res['split_fingerprint'][:12]}…`. FAISS = exact Flat search over "
           f"`{ctx.model_key}` {'mean-centered' if ctx.center else 'raw'} vectors with the Day 5 profile "
           f"({ctx.profile}); popularity mode `{ctx.pop_mode}`. Seen movies are removed from both "
           "sources. Relevant = rating ≥ 4. Recall@size = share of a user's relevant held-out movies "
           "that are in the candidate set.\n\n")
    if "selected" in res:
        s = res["selected"]
        md += f"**Selected for Day 9:** popular top-{s['pop']} ∪ FAISS top-{s['faiss']} ({s['rule']}).\n\n"
    if "test" in res:
        t = res["test"]
        md += f"## Test — fit on train+val, {t['n_users']:,} users\n\n"
        md += md_table(t["rows"], ["candidates", "recall", "hit_rate"], left=("candidates",))
        md += f"\nRetrieval (both sources + union): {t['ms_per_user_retrieval']} ms per user, batched.\n\n"
    if "recall" in res:
        r = res["recall"]
        md += f"## Candidate recall — validation, {r['n_users']:,} users (fit on train)\n\n"
        md += md_table(r["rows"], ["candidates", "size", "avg_unique", "recall", "hit_rate",
                                   "recall_gain_from_faiss", "pop_same_size_recall", "vs_pop_same_size"],
                       left=("candidates",))
        md += "\nFAISS alone: " + ", ".join(f"top-{x['size']} {x['recall']:.4f}" for x in r["faiss_rows"])
        md += (f".\n\nOf the relevant movies inside the selected union, "
               f"{100 * r['selected_recall_share_only_faiss']:.1f}% were found only by FAISS. "
               f"Best popularity-only set ≤ {r['max_size']}: {r['pop_only_best']['candidates']} "
               f"(recall {r['pop_only_best']['recall']:.4f}).\n\n")
    if "features" in res:
        f = res["features"]
        md += (f"## Ranker features — {f['n_users']:,} validation users, history = train, labels = val\n\n"
               f"{f['rows']:,} (user, candidate) rows, {f['relevant_rows']:,} relevant "
               f"({100 * f['relevant_rows'] / f['rows']:.2f}%), {f['avg_candidates']} candidates per user; "
               f"`{f['file']}` ({f['file_mb']} MB); retrieval + features {f['ms_per_user']} ms per user.\n\n"
               f"Check: ranking the candidates by `pop_rank` reproduces the popularity baseline for "
               f"{100 * f['pop_rank_reproduces_popularity']:.2f}% of users.\n\n")
        md += "### Each feature alone as the ranker (NDCG@10 on these users)\n\n"
        md += md_table(f["one_feature"], ["feature", "ndcg@10", "recall@10", "mrr@10"], left=("feature",))
        md += ("\n`(oracle)` = relevant movies first: the ceiling for any ranker on these candidates. "
               "`(union order)` = popular first, then FAISS-only.\n\n### Feature means: relevant vs other rows\n\n")
        md += md_table(f["stats"], ["feature", "mean_relevant", "mean_other", "missing_share", "meaning"],
                       left=("feature", "meaning"))
    path.write_text(md, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stages", nargs="+", default=["recall", "features", "test"],
                   choices=["recall", "features", "test"])
    p.add_argument("--tune-users", type=int, default=config.TUNE_USERS)
    p.add_argument("--ranker-users", type=int, default=config.RANKER_USERS)
    p.add_argument("--test-users", type=int, default=0, help="0 = all test users")
    p.add_argument("--batch-users", type=int, default=500, help="users per feature batch")
    p.add_argument("--test-batch", type=int, default=2000)
    p.add_argument("--faiss-batch", type=int, default=500)
    p.add_argument("--smoke", action="store_true", help="small run to check everything works")
    args = p.parse_args()
    if args.smoke:
        args.tune_users, args.ranker_users, args.test_users = 3000, 2000, 5000
    tag = "day08_candidates_smoke" if args.smoke else "day08_candidates"
    out_json, results = open_results(tag)
    t0 = time.time()
    results["split_fingerprint"] = split_fingerprint()
    ctx = Context(args, t0)
    for stage, fn in (("recall", stage_recall), ("features", stage_features), ("test", stage_test)):
        if stage in args.stages:
            log(f"--- stage {stage} ---", t0)
            fn(ctx, results)
            save_results(out_json, results)
    results["seconds_last_run"] = round(time.time() - t0, 1)
    save_results(out_json, results)
    write_md(results, out_json.with_suffix(".md"), ctx)
    log(f"wrote {out_json.relative_to(config.ROOT)} and .md", t0)


if __name__ == "__main__":
    main()
