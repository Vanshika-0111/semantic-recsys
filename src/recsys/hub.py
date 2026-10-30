"""Day 13: Hugging Face Hub helpers — publish the serving bundle, fetch it in the Space, deploy.

Layout of the public demo
  GitHub  Vanshika-0111/semantic-recsys   the code (home of everything)
  HF      <user>/semantic-recsys-bundle   model repo: the serving bundle WITHOUT demo users
  HF      <user>/semantic-recsys-demo     Docker Space: a snapshot of src/ + app/ + Dockerfile;
                                          downloads the bundle at startup, runs API + Streamlit

Why no demo users in public: the MovieLens licence does not allow redistributing the ratings.
The public bundle holds item vectors, metadata, aggregate item statistics and the two trained
rankers — no individual user's ratings.

CLI (used by the Space's start script):
    python -m recsys.hub download --repo <user>/semantic-recsys-bundle --out data/serving
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

DEMO_FILES = ("demo_history.parquet", "demo_truth.parquet")
BUNDLE_FILES = ("bundle.json", "arrays.npz", "items.parquet", "warm_ranker.txt", "cold_ranker.txt")
SPACE_SOURCES = ("src/recsys", "app")                     # code copied into the Space
SPACE_FILES = ("pyproject.toml", "requirements-serve.txt", "requirements-ui.txt",
               "deploy/start_space.sh")


# --------------------------------------------------------------------------- #
# Bundle -> Hub
# --------------------------------------------------------------------------- #
def stage_public_bundle(bundle_dir: Path, out_dir: Path, include_demo_users: bool = False) -> dict:
    """Copy the bundle into out_dir for publishing; demo-user files are left out unless asked.
    bundle.json is rewritten to say so (n_demo_users 0, public True). Returns that manifest."""
    bundle_dir, out_dir = Path(bundle_dir), Path(out_dir)
    missing = [f for f in BUNDLE_FILES if f != "cold_ranker.txt" and not (bundle_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"{bundle_dir} is not a complete bundle, missing {missing}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in BUNDLE_FILES + (DEMO_FILES if include_demo_users else ()):
        if (bundle_dir / f).exists():
            shutil.copy2(bundle_dir / f, out_dir / f)
    man = json.loads((bundle_dir / "bundle.json").read_text())
    if not include_demo_users:
        man["n_demo_users"] = 0
    man["public"] = not include_demo_users
    man["size_mb"] = round(sum(p.stat().st_size for p in out_dir.iterdir()) / 2**20, 1)
    (out_dir / "bundle.json").write_text(json.dumps(man, indent=2))
    return man


def bundle_card(repo_id: str, man: dict, code_url: str) -> str:
    """README.md of the bundle repo (the model card shown on huggingface.co)."""
    return f"""---
license: other
tags: [recommender-systems, learning-to-rank, lightgbm, faiss, sentence-transformers, movielens]
---
# Serving bundle — Semantic Recommendation & Ranking Engine

Everything the API needs at runtime, computed once from the training history: {man.get('n_items', '?'):,}
movie vectors (bge-small-en-v1.5, 384-d), movie metadata, aggregate item statistics (popularity,
Bayesian average rating, …), the warm and cold-start **LightGBM LambdaMART** rankers and the
validated serving settings. Loaded by `recsys.serving.Engine` — code: {code_url}

**No individual ratings are included.** The MovieLens licence does not allow redistributing the
ratings, so the demo users of the local bundle are left out ({man.get('n_demo_users', 0)} here).

- Ratings: MovieLens 32M — F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets:
  History and Context. ACM TiiS 5(4). GroupLens Research does not endorse this project.
  Non-commercial use only.
- Movie metadata / poster paths: TMDB (via a public Kaggle dump); not endorsed or certified by TMDB.

Provenance: split `{str(man.get('split_fingerprint', ''))[:12]}`, code `{str(man.get('git_commit') or '')[:7]}`,
bundle version {man.get('bundle_version')}.
"""


def upload_bundle(bundle_dir: Path, repo_id: str, *, private: bool = False,
                  include_demo_users: bool = False, code_url: str = "", token=None) -> str:
    """Create (if needed) the model repo and upload the staged bundle. Returns the repo URL."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    with tempfile.TemporaryDirectory() as tmp:
        man = stage_public_bundle(bundle_dir, Path(tmp), include_demo_users)
        (Path(tmp) / "README.md").write_text(bundle_card(repo_id, man, code_url), encoding="utf-8")
        url = api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=tmp,
                          commit_message=f"Serving bundle (code {str(man.get('git_commit') or '')[:7]})",
                          delete_patterns=["*.parquet"])          # drops demo files of an older upload
    return str(url)


# --------------------------------------------------------------------------- #
# Hub -> Space (runs inside the container at startup)
# --------------------------------------------------------------------------- #
def download_bundle(repo_id: str, out_dir: Path, revision: str | None = None, token=None) -> Path:
    from huggingface_hub import snapshot_download

    out_dir = Path(out_dir)
    snapshot_download(repo_id, repo_type="model", revision=revision, local_dir=out_dir, token=token,
                      allow_patterns=list(BUNDLE_FILES + DEMO_FILES))
    if not (out_dir / "bundle.json").exists():
        raise FileNotFoundError(f"{repo_id} has no bundle.json")
    return out_dir


# --------------------------------------------------------------------------- #
# Space
# --------------------------------------------------------------------------- #
def git_info(root: Path) -> tuple[str, bool]:
    """(commit sha or 'unknown', has uncommitted changes)."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", "src", "app", "deploy"],
                                    cwd=root, capture_output=True, text=True).stdout.strip())
        return sha, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", True


def stage_space(root: Path, out_dir: Path, *, title: str, code_url: str, commit: str) -> None:
    """The Space repo = Dockerfile + README (with the Space's YAML header) + a code snapshot."""
    root, out_dir = Path(root), Path(out_dir)
    for d in SPACE_SOURCES:
        shutil.copytree(root / d, out_dir / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in SPACE_FILES:
        (out_dir / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / f, out_dir / f)
    shutil.copy2(root / "deploy" / "hf_space" / "Dockerfile", out_dir / "Dockerfile")
    # a Windows checkout may have CRLF line endings; bash and Docker inside Linux need LF
    for f in (out_dir / "Dockerfile", out_dir / "deploy" / "start_space.sh"):
        f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n"))
    readme = (root / "deploy" / "hf_space" / "README.md").read_text(encoding="utf-8")
    readme = readme.replace("{{TITLE}}", title).replace("{{CODE_URL}}", code_url) \
                   .replace("{{COMMIT}}", commit[:7])
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def deploy_space(root: Path, space_id: str, bundle_repo: str, *, code_url: str,
                 title: str = "Semantic Movie Recommender", token=None) -> str:
    """Create/update the Docker Space from the current code and point it at the bundle repo.
    Every deploy uploads a new snapshot -> Hugging Face rebuilds the image."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    commit, dirty = git_info(root)
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.add_space_variable(space_id, "BUNDLE_REPO", bundle_repo)
    api.add_space_variable(space_id, "RECSYS_REPO_URL", code_url)
    with tempfile.TemporaryDirectory() as tmp:
        stage_space(root, Path(tmp), title=title, code_url=code_url, commit=commit)
        api.upload_folder(repo_id=space_id, repo_type="space", folder_path=tmp,
                          commit_message=f"Deploy {commit[:7]}{' (+ local changes)' if dirty else ''}",
                          delete_patterns=["src/**", "app/**", "deploy/**"])
    return f"https://huggingface.co/spaces/{space_id}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Hugging Face helpers (used inside the Space)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download the serving bundle")
    d.add_argument("--repo", required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--revision")
    args = ap.parse_args()
    if args.cmd == "download":
        out = download_bundle(args.repo, args.out, args.revision)
        print(f"bundle {args.repo} -> {out}")


if __name__ == "__main__":
    main()
