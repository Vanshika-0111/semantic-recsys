"""Day 13: smoke-test the PUBLIC demo setup before (or after) deploying it.

    python scripts/check_public_bundle.py

Downloads the bundle from Hugging Face into a fresh temp folder (ignoring data/serving), runs the
API in-process exactly as Streamlit Community Cloud does, and asks it for a few things: health,
title search, 3 onboarding picks, free-text search. If this passes on the laptop, the public demo
has everything it needs.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys import config  # noqa: E402
from recsys.client import ApiError, embedded_client  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repo", default="Vanshika-0111/semantic-recsys-bundle")
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        config.SERVING_DIR = Path(tmp) / "no-local-bundle"      # force the download path
        tempfile.tempdir = tmp
        t = time.time()
        c = embedded_client(bundle_repo=args.repo)
        print(f"downloaded + loaded {args.repo} in {time.time() - t:.0f} s")
        h = c.health()
        print(f"health: {h['items']:,} movies, demo users {h['demo_users']}, "
              f"ranker {h['warm_ranker_trees']} trees, text search {h['text_search']}")
        if h["demo_users"]:
            print("WARNING: the public bundle contains demo users' ratings — re-run upload_bundle.py")

        picks = []
        for q in ("The Matrix", "Inception", "Interstellar"):
            hit = c.search(q, 1)
            print(f"search {q!r} -> {hit[0]['title']} ({hit[0]['year']})" if hit else f"search {q!r} -> none")
            picks += [hit[0]["movieId"]] if hit else []
        out = c.cold_start(picks, k=5)
        print(f"{len(picks)} picks -> {out['model']}, {out['latency_ms']:.0f} ms:")
        for i in out["items"]:
            extra = f" · because you liked {i['because']['title']}" if i.get("because") else ""
            print(f"   {i['title']} ({i['year']}) — {i['explanation']}{extra}")
        try:
            ex = c.explore("heartwarming animated movie about friendship", k=5)
            print("free text ->", ", ".join(i["title"] for i in ex["items"]))
        except ApiError as e:
            print(f"free text -> not available ({e})")
    print("public demo setup OK")


if __name__ == "__main__":
    main()
