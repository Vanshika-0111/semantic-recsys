"""Day 2: create and FREEZE the evaluation split.

Outputs (data/processed/splits/):
  ratings_split.parquet   userId, movieId, rating, timestamp, split (int8 code)
  cold_seeds.parquet      userId, movieId, rating, timestamp, seed_rank (1..5)
  split_manifest.json     protocol, counts per split, SHA-256 fingerprint

The script refuses to overwrite an existing split (the test set is frozen).

Usage:
  python scripts/make_split.py            # create the split (once)
  python scripts/make_split.py --verify   # check the saved split is unchanged
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys import config  # noqa: E402
from recsys import split as S  # noqa: E402

import pandas as pd  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in-file", type=Path, default=config.PROCESSED_DIR / "ratings.parquet")
    p.add_argument("--out-dir", type=Path, default=config.SPLIT_DIR)
    p.add_argument("--n-cold-users", type=int, default=config.N_COLD_USERS)
    p.add_argument("--verify", action="store_true", help="only verify the frozen split")
    p.add_argument("--force", action="store_true", help="overwrite an existing split (don't!)")
    args = p.parse_args()

    if args.verify:
        ok = S.verify_frozen(args.out_dir)
        print("Split fingerprint matches manifest: OK" if ok else "SPLIT CHANGED — fingerprint mismatch!")
        sys.exit(0 if ok else 1)

    if (args.out_dir / S.MANIFEST_FILE).exists() and not args.force:
        sys.exit(f"{args.out_dir} already has a frozen split. Use --verify to check it.")

    t0 = time.time()
    print("Loading ratings ...")
    ratings = S.ensure_sorted(pd.read_parquet(args.in_file))
    print(f"  {len(ratings):,} ratings, {ratings['userId'].nunique():,} users ({time.time() - t0:.0f}s)")

    print("Choosing cold-start users ...")
    from_end, _ = S.position_from_end(ratings["userId"].to_numpy())
    cold = S.pick_cold_users(ratings, from_end, args.n_cold_users, config.N_TEST,
                             config.N_COLD_SEEDS, config.RELEVANCE_THRESHOLD, config.SEED)
    del from_end

    print("Assigning splits ...")
    code = S.assign_split(ratings, cold, config.N_TEST, config.N_VAL, config.MIN_TRAIN_FOR_VAL)
    seeds = S.cold_seeds(ratings, code, config.N_COLD_SEEDS, config.RELEVANCE_THRESHOLD)
    ratings["split"] = code

    manifest = {
        "created_seconds": None,
        "protocol": {
            "n_test": config.N_TEST, "n_val": config.N_VAL,
            "min_train_for_val": config.MIN_TRAIN_FOR_VAL,
            "n_cold_users": args.n_cold_users, "n_cold_seeds": config.N_COLD_SEEDS,
            "relevance_threshold": config.RELEVANCE_THRESHOLD, "seed": config.SEED,
            "order": "per user by (timestamp, movieId)",
            "codes": S.SPLIT_NAMES,
        },
        **S.summarize(ratings, code, seeds, config.RELEVANCE_THRESHOLD),
        "fingerprint": S.fingerprint(ratings, code),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ratings.to_parquet(args.out_dir / S.RATINGS_FILE, index=False)
    seeds.to_parquet(args.out_dir / S.SEEDS_FILE, index=False)
    manifest["created_seconds"] = round(time.time() - t0, 1)
    (args.out_dir / S.MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))

    print("\nSplit summary")
    for name, s in manifest["splits"].items():
        print(f"  {name:13s} {s['ratings']:>12,} ratings  {s['users']:>8,} users  "
              f"{s['relevant_pct']}% relevant")
    print(f"  test items unseen in train: {manifest['test_items_unseen_in_train']}, "
          f"cold_test: {manifest['cold_test_items_unseen_in_train']}")
    print(f"\nFrozen: {args.out_dir}  fingerprint {manifest['fingerprint'][:12]}…  "
          f"({manifest['created_seconds']}s)")


if __name__ == "__main__":
    main()
