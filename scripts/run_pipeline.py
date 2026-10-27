"""Day 7: the whole project in ONE command, in the right order, reproducibly.

    download -> prepare -> eda -> split -> baselines -> embeddings -> retrieval -> coldstart
      -> candidates -> ranker -> summary

Each step is the existing script (nothing is re-implemented here). A step whose outputs already
exist is SKIPPED, so a second run costs seconds — except the split, which is never rebuilt: if it
exists it is VERIFIED (fingerprint check), and the pipeline stops if it changed.

Every run writes
  reports/results/pipeline_manifest.json   python + package versions, git commit, OS, split
                                           fingerprint, and per step: ran / skipped, seconds (commit)
  reports/results/summary.md               headline numbers from all days (commit)
  data/logs/pipeline_<time>.log            everything the steps printed (git-ignored)

Usage
  python scripts/run_pipeline.py --dry-run             # show what would run, run nothing
  python scripts/run_pipeline.py                       # run whatever is missing
  python scripts/run_pipeline.py --smoke               # quick check of the experiment steps
  python scripts/run_pipeline.py --from retrieval --force   # redo retrieval + everything after
  python scripts/run_pipeline.py --steps baselines --force  # redo one step

Full run from nothing on an 8 GB laptop: ~2 h, mostly encoding 23k movies twice on the CPU
(Day 4, cached after that). With data + vectors already there, all experiments: ~30 min.
A step counts as done only if its results json matches the CURRENT split and its cached
files in data/ exist — so on a fresh clone (committed json, empty data/) everything re-runs.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # works without `pip install -e .`
from recsys import config  # noqa: E402
from recsys import split as S  # noqa: E402
from recsys.summary import load_all, provenance, write_summary  # noqa: E402

SCRIPTS = Path(__file__).resolve().parent
LOG_DIR = config.DATA_DIR / "logs"
MANIFEST = config.RESULTS_DIR / "pipeline_manifest.json"
PACKAGES = ["numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "faiss-cpu", "torch",
            "sentence-transformers", "lightgbm", "matplotlib"]


def current_fingerprint() -> str | None:
    p = config.SPLIT_DIR / S.MANIFEST_FILE
    return json.loads(p.read_text())["fingerprint"] if p.exists() else None


def has_results(name: str, keys: tuple[str, ...], artifacts: tuple[str, ...] = ()) -> Callable[[bool], bool]:
    """Done when reports/results/<name>[_smoke].json has every key in `keys`, was computed on the
    CURRENT split, and (full runs) its git-ignored artifacts exist (glob patterns under data/).
    On a fresh clone the committed json exists but data/ is empty -> the step re-runs."""
    def done(smoke: bool) -> bool:
        p = config.RESULTS_DIR / f"{name}{'_smoke' if smoke else ''}.json"
        if not p.exists():
            return False
        res = json.loads(p.read_text())
        if not all(k in res for k in keys) or res.get("split_fingerprint") != current_fingerprint():
            return False
        return smoke or all(any(config.DATA_DIR.glob(a)) for a in artifacts)
    return done


def files_exist(*paths: Path) -> Callable[[bool], bool]:
    return lambda smoke: all(p.exists() for p in paths)


@dataclass
class Step:
    name: str
    script: str
    done: Callable[[bool], bool]
    smoke_flag: bool = False            # pass --smoke to the script in a smoke run
    what: str = ""
    extra: list[str] = field(default_factory=list)


STEPS = [
    Step("download", "download_data.py",
         lambda smoke: (config.ML_DIR / "ratings.csv").exists() and any(config.TMDB_DIR.glob("*.csv")),
         what="MovieLens 32M + Kaggle TMDB dump (needs the Kaggle token)"),
    Step("prepare", "prepare_data.py",
         files_exist(config.PROCESSED_DIR / "ratings.parquet", config.PROCESSED_DIR / "items.parquet"),
         what="join + 20-core filter -> data/processed"),
    Step("eda", "eda.py", files_exist(config.REPORTS_DIR / "eda_summary.md"),
         what="figures + reports/eda_summary.md"),
    Step("split", "make_split.py", files_exist(config.SPLIT_DIR / S.MANIFEST_FILE),
         what="frozen time split + cold users (verified, never rebuilt)"),
    Step("baselines", "run_baselines.py",
         has_results("day03_baselines", ("test", "cold_start"),
                     ("runs/day03_popularity_test_per_user.parquet", "runs/tfidf_knn_*.npz")),
         smoke_flag=True, what="Day 3: popularity + TF-IDF"),
    Step("embeddings", "run_embeddings.py",
         has_results("day04_embeddings", ("selected",),
                     tuple(f"embeddings/{k}.npy" for k in config.EMB_MODELS)),
         smoke_flag=True, what="Day 4: MiniLM vs bge-small + FAISS Flat"),
    Step("retrieval", "run_retrieval.py",
         has_results("day05_retrieval", ("test", "ann"), ("runs/day05_*_test_per_user.parquet",)),
         smoke_flag=True, what="Day 5: profiles, candidate recall, Flat vs HNSW"),
    Step("coldstart", "run_coldstart.py",
         has_results("day06_coldstart", ("test",), ("runs/day06_coldstart_cold_per_user.parquet",)),
         smoke_flag=True, what="Day 6: cold-start onboarding"),
    Step("candidates", "run_candidates.py",
         has_results("day08_candidates", ("selected", "features", "test"),
                     ("runs/day08_features_val.parquet",)),
         smoke_flag=True, what="Day 8: popular ∪ FAISS candidates + ranker features"),
    Step("ranker", "run_ranker.py",
         has_results("day09_ranker", ("selected", "train", "cold"),
                     ("runs/day09_ranker.txt", "runs/day09_cold_ranker.txt")),
         smoke_flag=True, what="Day 9: LightGBM LambdaMART ranker (validation) + cold mode"),
]
NAMES = [s.name for s in STEPS]


# --------------------------------------------------------------------------- #
class Tee:
    """Print to the console and append to the log file."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = path.open("a", encoding="utf-8")

    def __call__(self, line: str = "") -> None:
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def run_script(script: str, args: list[str], out: Tee) -> int:
    """Run one script with this Python, streaming its output into the console and the log."""
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1",
           "MPLBACKEND": os.environ.get("MPLBACKEND", "Agg")}
    cmd = [sys.executable, str(SCRIPTS / script), *args]
    out(f"$ python scripts/{script} {' '.join(args)}".rstrip())
    proc = subprocess.Popen(cmd, cwd=config.ROOT, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        out("    " + line.rstrip("\n"))
    return proc.wait()


def git_state() -> dict:
    def git(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=config.ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"commit": git("rev-parse", "HEAD") or None,
            "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no"))}


def versions() -> dict:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def plan(args) -> list[tuple[Step, str]]:
    """(step, action) with action in run | skip | verify."""
    chosen = list(STEPS)
    if args.steps:
        chosen = [s for s in STEPS if s.name in args.steps]
    elif args.from_step:
        chosen = STEPS[NAMES.index(args.from_step):]
    out = []
    for s in chosen:
        done = s.done(args.smoke and s.smoke_flag)
        if s.name == "split":
            out.append((s, "verify" if done else "run"))      # never rebuilt, --force or not
        elif args.smoke and not s.smoke_flag and not done:
            out.append((s, "missing"))                         # smoke never downloads/prepares
        else:
            out.append((s, "run" if (args.force or not done) else "skip"))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", nargs="+", choices=NAMES, help="only these steps")
    p.add_argument("--from", dest="from_step", choices=NAMES, help="start at this step")
    p.add_argument("--force", action="store_true", help="re-run steps even if their outputs exist")
    p.add_argument("--smoke", action="store_true", help="small, fast runs of the experiment steps")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = p.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")              # λ, ≥, ✓ on a Windows console

    steps = plan(args)
    print("Plan:")
    for s, action in steps:
        print(f"  {s.name:11s} {action:7s} {s.what}")
    if args.dry_run:
        return
    missing = [s.name for s, a in steps if a == "missing"]
    if missing:
        sys.exit(f"--smoke needs the data steps done first: {missing}. Run without --smoke once.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Tee(LOG_DIR / f"pipeline_{stamp}{'_smoke' if args.smoke else ''}.log")
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    record = old.get("steps", {})
    t_all = time.time()
    for s, action in steps:
        out(f"\n=== {s.name}: {action} ===")
        if action == "skip":
            continue
        script_args = ["--verify"] if action == "verify" else list(s.extra)
        if args.smoke and s.smoke_flag:
            script_args.append("--smoke")
        t0 = time.time()
        code = run_script(s.script, script_args, out)
        secs = round(time.time() - t0, 1)
        if code != 0:
            out(f"!!! {s.name} failed (exit {code}) after {secs}s — fix it and re-run with --from {s.name}")
            sys.exit(code)
        if not args.smoke:
            record[s.name] = {"action": action, "seconds": secs,
                              "finished": datetime.now().isoformat(timespec="seconds")}
        out(f"=== {s.name}: ok ({secs}s) ===")

    summary = write_summary()
    ok, _ = provenance(load_all())
    out(f"\nwrote {summary.relative_to(config.ROOT)}"
        + ("" if ok else "  — WARNING: results come from different splits (see the file)"))
    if not args.smoke:
        manifest = {
            "last_run": datetime.now().isoformat(timespec="seconds"),
            "git": git_state(),
            "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "versions": versions(),
            "split_fingerprint": current_fingerprint(),
            "results_on_same_split": ok,
            "steps": {n: record[n] for n in NAMES if n in record},
        }
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(manifest, indent=2))
        out(f"wrote {MANIFEST.relative_to(config.ROOT)}")
    out(f"done in {time.time() - t_all:.0f}s; log: {out.f.name}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
