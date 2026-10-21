"""Download MovieLens 32M (GroupLens) and the Kaggle TMDB metadata dump.

Usage:  python scripts/download_data.py
Kaggle: needs ~/.kaggle/kaggle.json (Kaggle -> Settings -> API -> Create New Token)
        or KAGGLE_USERNAME / KAGGLE_KEY env vars.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys.config import ML_DIR, ML_URL, RAW_DIR, TMDB_DIR, TMDB_KAGGLE_SLUG  # noqa: E402


def _progress(block, block_size, total):
    done = block * block_size
    if total > 0:
        pct = min(100, done * 100 // total)
        print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB ({pct}%)", end="", flush=True)


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_movielens() -> None:
    if (ML_DIR / "ratings.csv").exists():
        print(f"MovieLens already at {ML_DIR}")
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / "ml-32m.zip"
    if not zip_path.exists():
        print(f"Downloading {ML_URL}")
        urllib.request.urlretrieve(ML_URL, zip_path, _progress)
        print()
    try:  # GroupLens publishes an md5 next to the zip
        expected = urllib.request.urlopen(ML_URL + ".md5").read().decode().split()[0]
        got = md5(zip_path)
        if got != expected:
            zip_path.unlink()
            raise SystemExit(f"MD5 mismatch ({got} != {expected}); deleted zip, re-run.")
        print("MD5 OK")
    except OSError as e:
        print(f"Could not verify MD5 ({e}); continuing.")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)
    print(f"Extracted to {ML_DIR}")


def download_tmdb() -> None:
    if TMDB_DIR.exists() and any(TMDB_DIR.glob("*.csv")):
        print(f"TMDB already at {TMDB_DIR}")
        return
    if shutil.which("kaggle") is None:
        raise SystemExit("kaggle CLI not found: pip install kaggle")
    TMDB_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", TMDB_KAGGLE_SLUG, "-p", str(TMDB_DIR), "--unzip"]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    download_movielens()
    download_tmdb()
    print("Done. Next: python scripts/prepare_data.py")
