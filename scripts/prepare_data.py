"""Join MovieLens 32M with TMDB metadata and apply the >=20 ratings k-core filter.

Outputs (data/processed/):
  ratings.parquet        userId, movieId, rating, timestamp  (sorted by user, time)
  items.parquet          metadata only for movies that survive filtering
  prepare_stats.json     funnel numbers for the README / EDA

Usage:
  python scripts/prepare_data.py                       # full data
  python scripts/prepare_data.py --sample-users 20000  # quick dev subset
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys import config  # noqa: E402
from recsys.data import (join_items, kcore_filter, load_links, load_movies,  # noqa: E402
                         load_ratings, load_tmdb)


def snapshot(r) -> dict:
    return {"ratings": int(len(r)), "users": int(r["userId"].nunique()),
            "items": int(r["movieId"].nunique())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ml-dir", type=Path, default=config.ML_DIR)
    p.add_argument("--tmdb-dir", type=Path, default=config.TMDB_DIR)
    p.add_argument("--out-dir", type=Path, default=config.PROCESSED_DIR)
    p.add_argument("--min-user", type=int, default=config.MIN_USER_RATINGS)
    p.add_argument("--min-item", type=int, default=config.MIN_ITEM_RATINGS)
    p.add_argument("--keep-no-overview", action="store_true",
                   help="keep movies without a TMDB overview (default: drop them)")
    p.add_argument("--sample-users", type=int, default=0,
                   help="randomly keep N users after filtering (dev runs)")
    args = p.parse_args()
    t0 = time.time()
    stats: dict = {"params": {k: str(v) for k, v in vars(args).items()}}

    print("Loading MovieLens ...")
    ratings = load_ratings(args.ml_dir)
    movies, links = load_movies(args.ml_dir), load_links(args.ml_dir)
    stats["raw"] = snapshot(ratings)
    stats["raw"]["catalog_movies"] = int(len(movies))
    n_dupes = int(ratings.duplicated(["userId", "movieId"]).sum())
    if n_dupes:
        ratings = ratings.sort_values("timestamp").drop_duplicates(["userId", "movieId"], keep="last")
    stats["duplicate_pairs_dropped"] = n_dupes
    print(f"  {stats['raw']}  ({time.time() - t0:.0f}s)")

    print("Loading TMDB ...")
    tmdb = load_tmdb(args.tmdb_dir)
    stats["tmdb_rows"] = int(len(tmdb))
    items = join_items(movies, links, tmdb)

    # Match rates: by catalog and weighted by ratings (what actually matters)
    rated = items[items["movieId"].isin(ratings["movieId"].unique())]
    weight = ratings["movieId"].value_counts()
    w = rated["movieId"].map(weight)
    stats["join"] = {
        "match_source_counts": rated["match_source"].value_counts().to_dict(),
        "items_with_overview_pct": round(float(100 * rated["has_overview"].mean()), 2),
        "ratings_on_items_with_overview_pct": round(float(100 * w[rated["has_overview"]].sum() / w.sum()), 2),
        "items_with_poster_pct": round(float(100 * rated["poster_path"].notna().mean()), 2)
        if "poster_path" in rated else None,
    }
    print(f"  join: {stats['join']}")

    if not args.keep_no_overview:
        ok = set(items.loc[items["has_overview"], "movieId"])
        ratings = ratings[ratings["movieId"].isin(ok)]
        stats["after_overview_filter"] = snapshot(ratings)

    print(f"k-core filter (users>={args.min_user}, items>={args.min_item}) ...")
    ratings, n_iter = kcore_filter(ratings, args.min_user, args.min_item)
    stats["kcore_iterations"] = n_iter

    if args.sample_users:
        rng = __import__("numpy").random.default_rng(config.SEED)
        users = ratings["userId"].unique()
        keep = rng.choice(users, size=min(args.sample_users, len(users)), replace=False)
        ratings, n_iter2 = kcore_filter(ratings[ratings["userId"].isin(keep)],
                                        args.min_user, args.min_item)
        stats["sampled_kcore_iterations"] = n_iter2

    stats["final"] = snapshot(ratings)
    f = stats["final"]
    stats["final"]["density_pct"] = round(100 * f["ratings"] / (f["users"] * f["items"]), 4)
    stats["final"]["kept_ratings_pct"] = round(100 * f["ratings"] / stats["raw"]["ratings"], 2)
    print(f"  final: {stats['final']}")

    ratings = ratings.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
    items = items[items["movieId"].isin(ratings["movieId"].unique())].reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(args.out_dir / "ratings.parquet", index=False)
    items.to_parquet(args.out_dir / "items.parquet", index=False)
    stats["seconds"] = round(time.time() - t0, 1)
    (args.out_dir / "prepare_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(f"Wrote {args.out_dir}  ({stats['seconds']}s). Next: python scripts/eda.py")


if __name__ == "__main__":
    main()
