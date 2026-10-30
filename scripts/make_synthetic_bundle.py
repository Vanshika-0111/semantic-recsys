"""Day 13: write a SMALL synthetic serving bundle (80 movies, 2 real LightGBM rankers, ~0.1 MB).

CI uses it to build and start the Docker images without the real 47 MB bundle or any MovieLens
data. It has exactly the files and format of the real one, so `Engine.load` and the API run the
same code path. Not for demos — titles are "Movie 10", "Movie 20", ...

    python scripts/make_synthetic_bundle.py                       -> data/serving
    python scripts/make_synthetic_bundle.py --out /tmp/b --no-demo-users

Refuses to overwrite a REAL bundle (one whose split fingerprint isn't "synthetic").
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys import config  # noqa: E402
from recsys.synthetic import build_synthetic_bundle  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=config.SERVING_DIR)
    ap.add_argument("--no-demo-users", action="store_true", help="like the public bundle")
    args = ap.parse_args()

    man_path = args.out / "bundle.json"
    if man_path.exists():
        fp = json.loads(man_path.read_text()).get("split_fingerprint")
        if fp != "synthetic":
            sys.exit(f"{args.out} holds a REAL bundle (split {str(fp)[:12]}) — not overwriting it. "
                     "Use --out somewhere else.")
        shutil.rmtree(args.out)
    with tempfile.TemporaryDirectory() as work:
        man = build_synthetic_bundle(args.out, work_dir=work, demo_users=not args.no_demo_users)["man"]
    print(f"synthetic bundle -> {args.out}  ({man['n_items']} movies, {man['n_demo_users']} demo users, "
          f"{man['size_mb']} MB)")


if __name__ == "__main__":
    main()
