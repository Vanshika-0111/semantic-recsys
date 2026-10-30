"""Day 13: publish the serving bundle (data/serving) to a Hugging Face model repo.

    hf auth login                                   (once; token stays outside the repo)
    python scripts/upload_bundle.py                 -> <your-username>/semantic-recsys-bundle

The 1,000 demo users' ratings are NOT uploaded (MovieLens licence: no redistribution of ratings);
the public demo works without them. Before uploading, the bundle is loaded with the same Engine
the API uses and asked for one recommendation — a broken bundle never gets published.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys import config  # noqa: E402
from recsys.experiment import setup_runtime  # noqa: E402
from recsys.hub import upload_bundle  # noqa: E402

CODE_URL = "https://github.com/Vanshika-0111/semantic-recsys"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", type=Path, default=config.SERVING_DIR)
    ap.add_argument("--repo", help="default: <your-username>/semantic-recsys-bundle")
    ap.add_argument("--private", action="store_true", help="private repo (the Space then needs HF_TOKEN)")
    ap.add_argument("--allow-synthetic", action="store_true", help="for testing the upload only")
    args = ap.parse_args()

    man = json.loads((args.bundle / "bundle.json").read_text())
    if man.get("split_fingerprint") == "synthetic" and not args.allow_synthetic:
        sys.exit(f"{args.bundle} is the SYNTHETIC CI bundle, not the real one. Rebuild the real bundle: "
                 "python scripts/run_pipeline.py --steps serving --force")

    setup_runtime()
    from recsys.serving import Engine
    eng = Engine.load(args.bundle)
    picks = [int(m) for m in eng.index.ids[:3]]
    out = eng.cold_start(picks, k=5, explain=False)
    print(f"bundle OK: {len(eng.index):,} movies, {out['model']} answers {len(out['items'])} recs")

    from huggingface_hub import HfApi
    try:
        user = HfApi().whoami()["name"]
    except Exception as e:                                # noqa: BLE001
        sys.exit(f"not logged in to Hugging Face ({type(e).__name__}) — run:  hf auth login")
    repo = args.repo or f"{user}/semantic-recsys-bundle"
    print(f"uploading to {repo} (demo users left out) ...")
    url = upload_bundle(args.bundle, repo, private=args.private, code_url=CODE_URL)
    print(f"done: {url}")


if __name__ == "__main__":
    main()
