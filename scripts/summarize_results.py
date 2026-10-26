"""Day 7: one results page — reports/results/summary.md — built from every day's json.

Reads day03_baselines.json, day04_embeddings.json, day05_retrieval.json, day06_coldstart.json
(smoke files are ignored), checks they were all computed on the same frozen split, and writes the
headline tables + what each CV bullet is backed by. Takes ~1 s; no data is loaded.

Usage
  python scripts/summarize_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # works without `pip install -e .`
from recsys import config  # noqa: E402
from recsys.summary import load_all, provenance, write_summary  # noqa: E402


def main() -> None:
    path = write_summary()
    ok, _ = provenance(load_all())
    print(f"wrote {path.relative_to(config.ROOT)}"
          + ("" if ok else "  — WARNING: results come from different splits (see the file)"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
