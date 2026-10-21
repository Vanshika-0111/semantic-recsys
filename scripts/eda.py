"""EDA on the filtered data. Writes figures to reports/figures/ and a summary to
reports/eda_summary.md. Numbers here are descriptive only (full period) — they
must NOT be reused as model features (those come from the training split).

Usage: python scripts/eda.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from recsys import config  # noqa: E402

HOLDOUT = 10  # last-N per user held out on Day 2
INK, ACCENT, MUTED = "#1f2937", "#2563eb", "#9ca3af"
plt.rcParams.update({
    "figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
    "font.size": 9,
})


def save(fig, name: str) -> str:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    return f"figures/{name}"


def q(s: pd.Series) -> dict:
    return {"min": int(s.min()), "median": int(s.median()),
            "p90": int(s.quantile(0.9)), "max": int(s.max())}


def main() -> None:
    r = pd.read_parquet(config.PROCESSED_DIR / "ratings.parquet")
    items = pd.read_parquet(config.PROCESSED_DIR / "items.parquet")
    prep = json.loads((config.PROCESSED_DIR / "prepare_stats.json").read_text())
    S: dict = {}
    figs: dict = {}

    # ---- activity / long tail ---------------------------------------------
    upu = r["userId"].value_counts()
    ipi = r["movieId"].value_counts()  # sorted desc
    S["ratings_per_user"], S["ratings_per_item"] = q(upu), q(ipi)
    cum = ipi.cumsum() / ipi.sum()
    n = len(ipi)
    S["top1pct_items_share"] = round(100 * cum.iloc[max(0, n // 100 - 1)], 1)
    S["top10pct_items_share"] = round(100 * cum.iloc[max(0, n // 10 - 1)], 1)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(np.arange(1, n + 1), ipi.values, color=ACCENT, lw=1.5)
    ax.set(xscale="log", yscale="log", xlabel="Item popularity rank",
           ylabel="Ratings", title="Long tail: ratings per item")
    ax.text(0.03, 0.05, f"Top 1% of items = {S['top1pct_items_share']}% of ratings\n"
            f"Top 10% of items = {S['top10pct_items_share']}% of ratings",
            transform=ax.transAxes, ha="left", va="bottom", color=INK)
    figs["long_tail"] = save(fig, "long_tail.png")

    fig, ax = plt.subplots(figsize=(6, 3.2))
    bins = np.logspace(np.log10(upu.min()), np.log10(upu.max()), 50)
    ax.hist(upu.values, bins=bins, color=ACCENT)
    ax.axvline(upu.median(), color=INK, ls="--", lw=1)
    ax.text(upu.median(), ax.get_ylim()[1] * 0.92, f"  median {int(upu.median())}", color=INK)
    ax.set(xscale="log", xlabel="Ratings per user", ylabel="Users", title="User activity")
    figs["user_activity"] = save(fig, "user_activity.png")

    # ---- rating values -----------------------------------------------------
    dist = r["rating"].value_counts(normalize=True).sort_index()
    S["mean_rating"] = round(float(r["rating"].mean()), 3)
    S["pct_ratings_ge_4"] = round(100 * float((r["rating"] >= 4).mean()), 1)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(dist.index.astype(str), 100 * dist.values, color=ACCENT)
    ax.set(xlabel="Rating", ylabel="% of ratings", title="Rating distribution")
    figs["rating_dist"] = save(fig, "rating_dist.png")

    # ---- time --------------------------------------------------------------
    ts = pd.to_datetime(r["timestamp"], unit="s")
    S["time_range"] = f"{ts.min():%Y-%m-%d} to {ts.max():%Y-%m-%d}"
    per_year = ts.dt.year.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(per_year.index, per_year.values / 1e6, color=ACCENT)
    ax.set(xlabel="Year", ylabel="Ratings (millions)", title="Ratings per year")
    figs["ratings_per_year"] = save(fig, "ratings_per_year.png")

    g = r.groupby("userId")["timestamp"]
    span_days = (g.max() - g.min()) / 86400
    S["users_single_day_pct"] = round(100 * float((span_days < 1).mean()), 1)
    S["median_user_span_days"] = round(float(span_days.median()), 1)

    # ---- Day-2 split preview (last HOLDOUT per user) ------------------------
    rank_from_end = r.groupby("userId").cumcount(ascending=False)  # 0 = latest
    last_test = r.loc[rank_from_end == HOLDOUT - 1, ["userId", "timestamp"]].set_index("userId")
    last_train = r.loc[rank_from_end == HOLDOUT, ["userId", "timestamp"]].set_index("userId")
    both = last_test.join(last_train, lsuffix="_test", rsuffix="_train", how="inner")
    S["split_boundary_tie_pct"] = round(
        100 * float((both["timestamp_test"] == both["timestamp_train"]).mean()), 1)
    train = r[rank_from_end >= HOLDOUT]
    test = r[rank_from_end < HOLDOUT]
    train_cnt = test["movieId"].map(train["movieId"].value_counts()).fillna(0)
    S["test_on_items_lt5_train_ratings_pct"] = round(100 * float((train_cnt < 5).mean()), 2)
    S["test_items_unseen_in_train_pct"] = round(100 * float((train_cnt == 0).mean()), 2)

    # ---- text --------------------------------------------------------------
    words = items["overview"].fillna("").str.split().str.len()
    S["overview_words"] = {"median": int(words.median()), "p95": int(words.quantile(0.95)),
                           "pct_over_200": round(100 * float((words > 200).mean()), 2)}
    if "keywords" in items:
        S["items_with_keywords_pct"] = round(100 * float(items["keywords"].notna().mean()), 1)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(words.clip(upper=300), bins=60, color=ACCENT)
    ax.axvline(200, color=INK, ls="--", lw=1)
    ax.text(200, ax.get_ylim()[1] * 0.92, "  ~200 words ≈ MiniLM 256-token limit", color=INK)
    ax.set(xlabel="Overview length (words, clipped at 300)", ylabel="Movies",
           title="TMDB overview length")
    figs["overview_length"] = save(fig, "overview_length.png")

    genres = items["genres"].fillna("").str.split("|").explode()
    genres = genres[genres != ""].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(genres.index[::-1], genres.values[::-1], color=ACCENT)
    ax.set(xlabel="Movies", title="Genres (MovieLens labels)")
    figs["genres"] = save(fig, "genres.png")

    # ---- write summary -------------------------------------------------------
    f, raw = prep["final"], prep["raw"]
    md = [
        "# Day 1 EDA summary", "",
        "## Data funnel", "",
        "| Stage | Ratings | Users | Items |", "|---|---:|---:|---:|",
        f"| Raw MovieLens 32M | {raw['ratings']:,} | {raw['users']:,} | {raw['items']:,} |",
    ]
    if "after_overview_filter" in prep:
        a = prep["after_overview_filter"]
        md.append(f"| Items with TMDB overview | {a['ratings']:,} | {a['users']:,} | {a['items']:,} |")
    md += [
        f"| k-core (>= {prep['params']['min_user']} per user & item) | {f['ratings']:,} | {f['users']:,} | {f['items']:,} |",
        "",
        f"Kept {f['kept_ratings_pct']}% of ratings; density {f['density_pct']}%; "
        f"k-core converged in {prep['kcore_iterations']} iterations.",
        f"TMDB join: {prep['join']}", "",
        "## Key numbers", "", "```json", json.dumps(S, indent=2), "```", "",
        "## What this means for the next days", "",
        f"- **Popularity is strong** (top 1% of items = {S['top1pct_items_share']}% of ratings): "
        "the popularity baseline will be hard to beat on NDCG — report coverage/diversity too.",
        f"- **{S['users_single_day_pct']}% of users rated everything within one day**: "
        "recency decay in Day 5 profiles will matter little for them; check before tuning it.",
        f"- **{S['split_boundary_tie_pct']}% of users have a timestamp tie at the train/test boundary**: "
        "break ties deterministically (timestamp, then movieId) in the Day 2 split.",
        f"- **{S['test_items_unseen_in_train_pct']}% of held-out ratings are on items unseen in train** "
        f"({S['test_on_items_lt5_train_ratings_pct']}% on items with < 5 train ratings): "
        "this is where content embeddings can beat popularity.",
        f"- **Overviews: median {S['overview_words']['median']} words, "
        f"{S['overview_words']['pct_over_200']}% over 200**: "
        + ("MiniLM truncation is a non-issue." if S["overview_words"]["pct_over_200"] < 2
           else "put title + genres + keywords BEFORE the overview so truncation only cuts plot text."),
        "", "## Figures", "",
    ]
    md += [f"![{k}]({v})" for k, v in figs.items()]
    out = config.REPORTS_DIR / "eda_summary.md"
    out.write_text("\n".join(md) + "\n")
    print(json.dumps(S, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
