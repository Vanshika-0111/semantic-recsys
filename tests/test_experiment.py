"""Day 7 tests: shared experiment helpers, the results summary, and the pipeline plan.

Toy catalog as in the other tests: movieIds 10, 20, 30, 40, 50 -> columns 0..4.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recsys import config  # noqa: E402
from recsys.baselines import ItemIndex  # noqa: E402
from recsys.experiment import (chosen_settings, eval_users, head_items, md_table, only,  # noqa: E402
                               read_results, row)
from recsys.summary import build_summary, cold_table, provenance, warm_rows  # noqa: E402

INDEX = ItemIndex([10, 20, 30, 40, 50])


# --------------------------------------------------------------------------- #
# users / items
# --------------------------------------------------------------------------- #
def test_eval_users_keeps_only_users_with_a_relevant_item():
    # u1 rated 5 and 2 -> has a relevant item; u2 only 3 -> skipped; u3 rated 4 -> kept (>= 4)
    truth = pd.DataFrame({"userId": [1, 1, 2, 3], "movieId": [10, 20, 10, 30],
                          "rating": [5.0, 2.0, 3.0, 4.0]})
    assert eval_users(truth, None, seed=0).tolist() == [1, 3]


def test_eval_users_sample_is_seeded_and_sorted():
    truth = pd.DataFrame({"userId": np.arange(100), "movieId": 10, "rating": 5.0})
    a, b = eval_users(truth, 10, seed=42), eval_users(truth, 10, seed=42)
    assert len(a) == 10 and np.array_equal(a, b) and np.all(np.diff(a) > 0)
    assert eval_users(truth, 1000, seed=42).tolist() == list(range(100))   # limit >= n -> all


def test_only_filters_rows_by_user():
    df = pd.DataFrame({"userId": [1, 2, 3, 2], "movieId": [10, 20, 30, 40]})
    assert only(df, np.array([2])).movieId.tolist() == [20, 40]


def test_head_items_by_hand():
    # training ratings per movie: 10 -> 2, 20 -> 1, 30 -> 3, 40 -> 0, 50 -> 1
    ids = np.array([30, 30, 10, 20, 30, 10, 50])
    # 40% of 5 movies = 2 -> the two most rated: 30 (3), 10 (2)
    assert head_items(ids, INDEX, share=0.4).tolist() == [30, 10]
    # 60% = 3 -> 20 and 50 tie at 1 -> catalog order (stable) picks 20
    assert head_items(ids, INDEX, share=0.6).tolist() == [30, 10, 20]


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def test_row_rounds_floats_and_keeps_extras():
    r = row("pop", {"ndcg@10": 0.123456789, "n_users": 7}, {"config": "x", "ms": 1.234567})
    assert r == {"model": "pop", "ndcg@10": 0.12346, "n_users": 7, "config": "x", "ms": 1.23457}


def test_md_table_alignment_and_formats():
    md = md_table([{"model": "a", "ndcg@10": 0.5, "n": 1234}], ["model", "ndcg@10", "n"])
    assert md.splitlines() == ["| model | ndcg@10 | n |", "|---|---:|---:|", "| a | 0.5000 | 1,234 |"]
    md = md_table([{"retriever": "b"}], ["retriever"], left=("retriever",))
    assert md.splitlines()[1] == "|---|"


# --------------------------------------------------------------------------- #
# settings chosen by earlier days
# --------------------------------------------------------------------------- #
def test_chosen_settings_defaults_when_nothing_was_run(tmp_path):
    s = chosen_settings(tmp_path)
    assert s == {"popularity_mode": "liked", "tfidf": dict(config.TFIDF_DEFAULT),
                 "encoder": "bge-small", "center": True,
                 "profile": {"weighting": "graded", "half_life": 5}}


def test_chosen_settings_reads_each_day(tmp_path):
    (tmp_path / "day03_baselines.json").write_text(json.dumps({"selected": {
        "popularity": {"mode": "count"}, "tfidf_knn": {"weighting": "liked", "max_history": 50}}}))
    (tmp_path / "day04_embeddings.json").write_text(json.dumps({"selected": {
        "model": "minilm", "center": False}}))
    (tmp_path / "day05_retrieval.json").write_text(json.dumps({"selected": {
        "weighting": "liked", "half_life": None}}))
    s = chosen_settings(tmp_path)
    assert s["popularity_mode"] == "count" and s["tfidf"]["max_history"] == 50
    assert s["encoder"] == "minilm" and s["center"] is False
    assert s["profile"] == {"weighting": "liked", "half_life": None}
    assert read_results("nope.json", tmp_path) == {}


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def _write_fake_results(d: Path, fp4: str = "abc") -> None:
    (d / "day03_baselines.json").write_text(json.dumps({
        "split_fingerprint": "abc",
        "test": {"n_users": 100, "rows": [{"model": "popularity", "config": "mode=liked", "ndcg@10": 0.05},
                                          {"model": "tfidf_knn", "config": "", "ndcg@10": 0.02}]}}))
    (d / "day04_embeddings.json").write_text(json.dumps({"split_fingerprint": fp4}))
    (d / "day06_coldstart.json").write_text(json.dumps({
        "split_fingerprint": "abc",
        "test": {"n_users": 50,
                 "rows": [{"model": "popularity", "setting": "seeds_5", "ndcg@10": 0.03},
                          {"model": "semantic rerank (ours)", "setting": "seeds_5", "ndcg@10": 0.033}],
                 "tfidf_day3": {"5": {"ndcg@10": 0.008}},
                 "significance": [{"model": "semantic rerank (ours)", "setting": "seeds_5",
                                   "vs": "popularity (seeds_5)", "rel_change": 0.1,
                                   "ci95": [0.001, 0.005], "p_le_0": 0.01}]}}))


def test_summary_tables_and_cv_status(tmp_path):
    _write_fake_results(tmp_path)
    from recsys.summary import load_all
    res = load_all(tmp_path)
    w = warm_rows(res)
    # tfidf 0.02 vs popularity 0.05 -> 0.02 / 0.05 - 1 = -60%
    assert [r["vs popularity"] for r in w] == ["+0.0%", "-60.0%"]
    five = next(c for c in cold_table(res) if c["seeds"] == 5)
    assert five == {"seeds": 5, "popularity": 0.03, "semantic rerank (ours)": 0.033,
                    "TF-IDF (Day 3)": 0.008}
    md = build_summary(tmp_path)
    assert "same frozen split ✓" in md
    assert "**+10.0% NDCG@10** over popularity with 5 onboarding picks" in md
    assert "≥ 0.0560" in md                                   # 1.12 × 0.05


def test_summary_flags_results_from_different_splits(tmp_path):
    _write_fake_results(tmp_path, fp4="zzz")
    from recsys.summary import load_all
    ok, text = provenance(load_all(tmp_path))
    assert not ok and "DIFFERENT SPLITS" in text


# --------------------------------------------------------------------------- #
# pipeline plan (which steps run / skip / verify) — no step is executed
# --------------------------------------------------------------------------- #
def _pipeline():
    spec = importlib.util.spec_from_file_location("run_pipeline", ROOT / "scripts" / "run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_pipeline"] = mod          # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


class Args:
    def __init__(self, **kw):
        self.steps, self.from_step, self.force, self.smoke = None, None, False, False
        self.__dict__.update(kw)


def test_pipeline_plan(monkeypatch):
    P = _pipeline()
    done = {"download", "prepare", "eda", "split", "baselines"}         # pretend these exist
    for s in P.STEPS:
        monkeypatch.setattr(s, "done", lambda smoke, n=s.name: n in done)
    plan = {s.name: a for s, a in P.plan(Args())}
    assert plan["download"] == "skip" and plan["split"] == "verify"
    assert plan["baselines"] == "skip" and plan["embeddings"] == "run"
    # --force re-runs finished steps but NEVER rebuilds the frozen split
    plan = {s.name: a for s, a in P.plan(Args(force=True))}
    assert plan["baselines"] == "run" and plan["split"] == "verify"
    # --from starts in the middle
    assert [s.name for s, _ in P.plan(Args(from_step="retrieval"))] == ["retrieval", "coldstart", "candidates", "ranker", "final", "diversity",
                                                                     "serving"]
    # --smoke never downloads data: a missing data step is reported, not run
    done.discard("prepare")
    plan = {s.name: a for s, a in P.plan(Args(smoke=True))}
    assert plan["prepare"] == "missing"


@pytest.mark.parametrize("name", ["run_baselines", "run_embeddings", "run_retrieval",
                                  "run_coldstart", "run_candidates", "run_ranker", "summarize_results",
                                  "run_pipeline"])
def test_scripts_do_not_import_other_scripts(name):
    """Day 7 rule: shared code lives in src/recsys, scripts never import each other."""
    text = (ROOT / "scripts" / f"{name}.py").read_text(encoding="utf-8")
    assert "from run_" not in text and "import run_" not in text
