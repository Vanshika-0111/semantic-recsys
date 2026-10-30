"""Day 13: create / update the public demo — a Hugging Face Docker Space.

    python scripts/deploy_space.py              -> https://huggingface.co/spaces/<you>/semantic-recsys-demo

Uploads a snapshot of the code (src/recsys, app/, the Space Dockerfile + start script) to the
Space and sets its variable BUNDLE_REPO; Hugging Face then builds the image (~5–10 min) and starts
it. At startup the container downloads the bundle from BUNDLE_REPO, starts the API on
127.0.0.1:8000 and the Streamlit UI on port 7860 (the public one).

Commit + push to GitHub first: the snapshot's commit id is shown in the Space.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recsys.hub import deploy_space, git_info  # noqa: E402

CODE_URL = "https://github.com/Vanshika-0111/semantic-recsys"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--space", help="default: <your-username>/semantic-recsys-demo")
    ap.add_argument("--bundle-repo", help="default: <your-username>/semantic-recsys-bundle")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    try:
        user = HfApi().whoami()["name"]
    except Exception as e:                                # noqa: BLE001
        sys.exit(f"not logged in to Hugging Face ({type(e).__name__}) — run:  hf auth login")
    space = args.space or f"{user}/semantic-recsys-demo"
    bundle = args.bundle_repo or f"{user}/semantic-recsys-bundle"
    if not HfApi().repo_exists(bundle, repo_type="model"):
        sys.exit(f"bundle repo {bundle} not found — run scripts/upload_bundle.py first")

    commit, dirty = git_info(ROOT)
    if dirty:
        print("warning: uncommitted changes in src/ app/ deploy/ — the Space gets them, GitHub doesn't")
    url = deploy_space(ROOT, space, bundle, code_url=CODE_URL)
    print(f"deployed code {commit[:7]} -> {url}")
    print("Hugging Face is now building the image (5–10 min). Watch the 'Logs' button on that page.")


if __name__ == "__main__":
    main()
