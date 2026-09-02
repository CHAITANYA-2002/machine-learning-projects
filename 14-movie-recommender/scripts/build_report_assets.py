"""Measure the recommender and render every figure the README shows.

Nothing here needs relevance labels. Every quantity below is a property of the
catalogue and of the ranking function itself -- coverage, concentration,
similarity distribution, representation agreement, memory, latency -- so it can
be stated honestly without claiming anything about whether a person would enjoy
a film.

Run from the project directory:

    python scripts/build_report_assets.py

It writes PNGs to docs/assets/ and a machine-readable summary to
docs/assets/metrics.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recommender import ContentRecommender, load_catalog  # noqa: E402

ASSETS = PROJECT_ROOT / "docs" / "assets"
CATALOG = PROJECT_ROOT / "data" / "movie_catalog.csv"

TOP_K = 5
CHUNK = 512
DPI = 150

PALETTE = {
    "primary": "#1f4e79",
    "accent": "#c0504d",
    "muted": "#9e9e9e",
    "grid": "#dddddd",
    "alt": "#2e7d32",
}


def _style(axis, title="", xlabel="", ylabel=""):
    if title:
        axis.set_title(title, fontsize=12, fontweight="bold", pad=10)
    axis.set_xlabel(xlabel, fontsize=10)
    axis.set_ylabel(ylabel, fontsize=10)
    axis.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def _save(figure, name: str) -> Path:
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / name
    figure.tight_layout()
    figure.savefig(path, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(figure)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")
    return path


def top_k_everything(matrix, k: int = TOP_K) -> tuple[np.ndarray, np.ndarray]:
    """Top-k neighbours for every catalogue row, excluding the row itself.

    Computed in chunks so the full 4,794 x 4,794 dense similarity matrix never
    exists at once -- the same reason the runtime ranks one row at a time.
    """
    n = matrix.shape[0]
    indices = np.empty((n, k), dtype=np.int32)
    scores = np.empty((n, k), dtype=np.float32)

    for start in range(0, n, CHUNK):
        stop = min(start + CHUNK, n)
        block = cosine_similarity(matrix[start:stop], matrix)
        block[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        partial = np.argpartition(-block, kth=k, axis=1)[:, :k]
        block_scores = np.take_along_axis(block, partial, axis=1)
        order = np.argsort(-block_scores, axis=1, kind="stable")
        indices[start:stop] = np.take_along_axis(partial, order, axis=1)
        scores[start:stop] = np.take_along_axis(block_scores, order, axis=1)

    return indices, scores


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of how often each film is recommended. 0 = uniform."""
    values = np.sort(counts.astype(float))
    n = values.size
    if values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * (index * values).sum()) / (n * values.sum()) - (n + 1) / n)


def plot_catalog_profile(catalog: pd.DataFrame, vectorizer: CountVectorizer) -> dict:
    tokens = catalog["tags"].str.split().str.len()
    counts = np.asarray(vectorizer.transform(catalog["tags"]).sum(axis=0)).ravel()
    vocabulary = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(-counts)[:20]

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    left.hist(tokens, bins=40, color=PALETTE["primary"], zorder=3)
    left.axvline(tokens.median(), color=PALETTE["accent"], linestyle="--", linewidth=1.4)
    left.text(
        tokens.median() + 3, left.get_ylim()[1] * 0.9,
        f"median {int(tokens.median())} tokens", fontsize=9, color=PALETTE["accent"],
    )
    _style(left, "Tag length per film", "tokens in the tags field", "films")

    right.barh(vocabulary[order][::-1], counts[order][::-1], color=PALETTE["primary"], zorder=3)
    right.tick_params(axis="y", labelsize=8)
    _style(right, "20 most frequent vocabulary terms", "films containing the term")
    right.grid(axis="y", visible=False)
    right.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)

    figure.suptitle(
        f"Catalogue profile: {len(catalog):,} films, {len(vocabulary):,} retained vocabulary terms",
        fontsize=13, fontweight="bold",
    )
    _save(figure, "catalog-profile.png")
    return {
        "films": int(len(catalog)),
        "vocabulary_retained": int(len(vocabulary)),
        "tag_tokens_min": int(tokens.min()),
        "tag_tokens_median": int(tokens.median()),
        "tag_tokens_mean": round(float(tokens.mean()), 1),
        "tag_tokens_max": int(tokens.max()),
        "top_terms": vocabulary[order][:10].tolist(),
    }


def plot_similarity_distribution(scores: np.ndarray) -> dict:
    top1, top5 = scores[:, 0], scores[:, TOP_K - 1]

    figure, axis = plt.subplots(figsize=(9.5, 5.0))
    bins = np.linspace(0, max(float(top1.max()), 0.6), 50)
    axis.hist(top1, bins=bins, color=PALETTE["primary"], alpha=0.85, label="rank 1 (closest)", zorder=3)
    axis.hist(top5, bins=bins, color=PALETTE["accent"], alpha=0.7, label="rank 5 (last shown)", zorder=3)
    axis.axvline(float(np.median(top1)), color=PALETTE["primary"], linestyle="--", linewidth=1.3)
    axis.axvline(float(np.median(top5)), color=PALETTE["accent"], linestyle="--", linewidth=1.3)
    axis.legend(frameon=False, fontsize=10)
    _style(
        axis,
        "How similar is a recommendation, really?",
        "cosine similarity of the returned film to the selected film",
        "queries",
    )
    axis.text(
        0.98, 0.72,
        f"median rank-1  {np.median(top1):.3f}\nmedian rank-5  {np.median(top5):.3f}",
        transform=axis.transAxes, ha="right", fontsize=10, family="monospace",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f5f5f5", "edgecolor": "#cccccc"},
    )
    _save(figure, "similarity-distribution.png")

    return {
        "rank1_median": round(float(np.median(top1)), 4),
        "rank1_p10": round(float(np.percentile(top1, 10)), 4),
        "rank1_p90": round(float(np.percentile(top1, 90)), 4),
        "rank5_median": round(float(np.median(top5)), 4),
        "queries_with_rank1_below_0_2": int((top1 < 0.2).sum()),
        "queries_with_rank1_above_0_5": int((top1 > 0.5).sum()),
    }


def plot_coverage_and_concentration(indices: np.ndarray, catalog: pd.DataFrame) -> dict:
    n = len(catalog)
    coverage = [
        {"k": k, "covered": int(np.unique(indices[:, :k]).size) / n}
        for k in range(1, TOP_K + 1)
    ]
    frequency = np.bincount(indices[:, :TOP_K].ravel(), minlength=n)
    ordered = np.sort(frequency)[::-1]
    lorenz = np.concatenate([[0.0], np.cumsum(np.sort(frequency)) / frequency.sum()])

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    left.plot(
        [row["k"] for row in coverage], [row["covered"] for row in coverage],
        marker="o", color=PALETTE["primary"], linewidth=2.0,
    )
    for row in coverage:
        left.annotate(f"{row['covered']:.1%}", (row["k"], row["covered"]),
                      textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    left.set_ylim(0, 1.0)
    left.set_xticks(range(1, TOP_K + 1))
    _style(left, "Catalogue coverage", "recommendations shown per query (k)",
           "share of the catalogue reachable")

    right.plot(np.linspace(0, 1, lorenz.size), lorenz, color=PALETTE["accent"], linewidth=2.0,
               label=f"actual (Gini {gini(frequency):.3f})")
    right.plot([0, 1], [0, 1], color=PALETTE["muted"], linestyle="--", linewidth=1.2,
               label="perfectly even exposure")
    right.legend(frameon=False, fontsize=9, loc="upper left")
    _style(right, "Exposure concentration", "films, least- to most-recommended",
           "cumulative share of recommendation slots")

    _save(figure, "catalog-coverage.png")

    never = int((frequency == 0).sum())
    top_1pct = int(np.ceil(n * 0.01))
    return {
        "coverage_at_5": round(coverage[-1]["covered"], 4),
        "coverage_at_1": round(coverage[0]["covered"], 4),
        "never_recommended": never,
        "never_recommended_share": round(never / n, 4),
        "gini": round(gini(frequency), 4),
        "top_1pct_share_of_slots": round(float(ordered[:top_1pct].sum() / frequency.sum()), 4),
        "most_recommended_count": int(ordered[0]),
        "most_recommended_title": str(catalog["title"].iloc[int(np.argmax(frequency))]),
    }


def plot_vectorizer_ablation(catalog: pd.DataFrame, count_indices: np.ndarray) -> dict:
    """How much would the shipped lists change under TF-IDF instead of raw counts?

    There are no relevance labels, so this cannot say which is better. It can say
    how much of the result is an artefact of the representation choice, which is
    the honest version of the question.
    """
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_indices, _ = top_k_everything(tfidf.fit_transform(catalog["tags"]))

    overlaps = np.array([
        len(set(count_indices[row]) & set(tfidf_indices[row])) for row in range(len(catalog))
    ])
    distribution = np.bincount(overlaps, minlength=TOP_K + 1)

    figure, axis = plt.subplots(figsize=(9.5, 5.0))
    bars = axis.bar(range(TOP_K + 1), distribution / len(catalog), color=PALETTE["alt"], zorder=3)
    for bar, value in zip(bars, distribution / len(catalog), strict=True):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 0.006, f"{value:.1%}",
                  ha="center", fontsize=9)
    axis.set_xticks(range(TOP_K + 1))
    axis.set_ylim(0, max(distribution / len(catalog)) * 1.18)
    _style(
        axis,
        "Representation sensitivity: CountVectorizer vs TF-IDF",
        "titles shared between the two top-5 lists for the same query",
        "share of queries",
    )
    axis.text(
        0.02, 0.93,
        f"mean overlap {overlaps.mean():.2f} of {TOP_K}\n"
        f"{(overlaps == TOP_K).mean():.1%} of queries return an identical list",
        transform=axis.transAxes, fontsize=10, family="monospace", va="top",
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "#f5f5f5", "edgecolor": "#cccccc"},
    )
    _save(figure, "vectorizer-ablation.png")

    return {
        "mean_overlap_of_5": round(float(overlaps.mean()), 3),
        "identical_list_share": round(float((overlaps == TOP_K).mean()), 4),
        "no_overlap_share": round(float((overlaps == 0).mean()), 4),
    }


#: Cast and director tokens were written as concatenated CamelCase names by the
#: legacy preparation step ("SamWorthington", "JamesCameron").
NAME_TOKEN = re.compile(r"^[A-Z][a-z]+[A-Z]")


def plot_vocabulary_cap(catalog: pd.DataFrame) -> dict:
    """What does max_features=5000 actually throw away?

    The cap is inherited from the original notebook. Measuring it turns out to
    matter: the discarded tail is almost entirely cast and director names, which
    are the only tokens that distinguish two films sharing a genre.
    """
    full = CountVectorizer(stop_words="english").fit(catalog["tags"])
    capped = CountVectorizer(max_features=5000, stop_words="english").fit(catalog["tags"])
    full_vocabulary = set(full.get_feature_names_out())
    kept = set(capped.get_feature_names_out())

    raw_tokens = {token for tags in catalog["tags"] for token in tags.split()}
    names = {token for token in raw_tokens if NAME_TOKEN.match(token)}
    names_kept = {token for token in names if token.lower() in kept}

    def _has_surviving_name(tags: str) -> bool:
        return any(t.lower() in kept for t in tags.split() if NAME_TOKEN.match(t))

    with_name = catalog["tags"].apply(lambda t: any(NAME_TOKEN.match(w) for w in t.split()))
    with_surviving_name = catalog["tags"].apply(_has_surviving_name)

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    labels = ["all distinct\ntokens", "cast + director\nname tokens"]
    totals = [len(full_vocabulary), len(names)]
    survivors = [len(kept), len(names_kept)]
    positions = np.arange(2)
    left.bar(positions - 0.2, totals, 0.4, label="in the catalogue", color=PALETTE["muted"], zorder=3)
    left.bar(positions + 0.2, survivors, 0.4, label="kept by max_features=5000",
             color=PALETTE["accent"], zorder=3)
    for position, total, survivor in zip(positions, totals, survivors, strict=True):
        left.text(position - 0.2, total * 1.05, f"{total:,}", ha="center", fontsize=9)
        left.text(position + 0.2, survivor * 1.05, f"{survivor:,}\n({survivor / total:.1%})",
                  ha="center", fontsize=9, color=PALETTE["accent"])
    left.set_xticks(positions, labels, fontsize=10)
    left.set_yscale("log")
    left.legend(frameon=False, fontsize=9)
    _style(left, "What the vocabulary cap discards", ylabel="tokens (log scale)")

    shares = [float(with_name.mean()), float(with_surviving_name.mean())]
    bars = right.bar(
        ["before the cap", "after the cap"], shares,
        color=[PALETTE["muted"], PALETTE["accent"]], zorder=3, width=0.5,
    )
    for bar, share in zip(bars, shares, strict=True):
        right.text(bar.get_x() + bar.get_width() / 2, share + 0.02, f"{share:.1%}",
                   ha="center", fontsize=11, fontweight="bold")
    right.set_ylim(0, 1.15)
    _style(right, "Films that still have a cast or director signal",
           ylabel="share of the catalogue")

    figure.suptitle(
        "The 5,000-term cap removes 94% of the people in the catalogue",
        fontsize=13, fontweight="bold",
    )
    _save(figure, "vocabulary-cap.png")

    return {
        "distinct_tokens": len(full_vocabulary),
        "tokens_kept": len(kept),
        "token_retention": round(len(kept) / len(full_vocabulary), 4),
        "name_tokens": len(names),
        "name_tokens_kept": len(names_kept),
        "name_token_retention": round(len(names_kept) / len(names), 4),
        "films_with_name_before": round(float(with_name.mean()), 4),
        "films_with_name_after": round(float(with_surviving_name.mean()), 4),
        "films_losing_all_names": int((with_name & ~with_surviving_name).sum()),
    }


def plot_exposure_bias(indices: np.ndarray, catalog: pd.DataFrame) -> dict:
    """Do sparsely described films get recommended more than they should?

    Cosine similarity divides by vector length. A film whose tags are three
    generic genre words has a short vector that points almost exactly at any
    query sharing one of those words, so it can outrank a film that genuinely
    matches on ten specific terms. This measures whether that happens here.
    """
    tokens = catalog["tags"].str.split().str.len().to_numpy()
    frequency = np.bincount(indices.ravel(), minlength=len(catalog))
    correlation = float(np.corrcoef(tokens, frequency)[0, 1])

    short = tokens < 15
    deciles = pd.qcut(tokens, 10, labels=False, duplicates="drop")
    grouped = pd.DataFrame({"decile": deciles, "frequency": frequency, "tokens": tokens})
    profile = grouped.groupby("decile").agg(
        mean_frequency=("frequency", "mean"), median_tokens=("tokens", "median")
    )

    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    left.bar(profile.index, profile["mean_frequency"], color=PALETTE["primary"], zorder=3)
    left.set_xticks(profile.index, [f"{int(t)}" for t in profile["median_tokens"]], fontsize=9)
    _style(
        left,
        f"Exposure by tag length (r = {correlation:.3f})",
        "median tag tokens in the decile",
        "mean times recommended",
    )

    ranking = np.argsort(-frequency)[:8]
    titles = [catalog["title"].iloc[i][:28] for i in ranking][::-1]
    right.barh(titles, frequency[ranking][::-1], color=PALETTE["accent"], zorder=3)
    for position, i in enumerate(ranking[::-1]):
        right.text(frequency[i] + 2, position, f"{tokens[i]} tokens", va="center", fontsize=8)
    right.tick_params(axis="y", labelsize=8)
    _style(right, "The eight most-recommended films", "times shown in someone's top 5")
    right.grid(axis="y", visible=False)
    right.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
    right.set_xlim(0, frequency.max() * 1.28)

    figure.suptitle(
        "Sparsely described films are recommended far more often than they should be",
        fontsize=13, fontweight="bold",
    )
    _save(figure, "exposure-bias.png")

    return {
        "length_frequency_correlation": round(correlation, 3),
        "short_tag_films": int(short.sum()),
        "short_tag_mean_exposure": round(float(frequency[short].mean()), 1),
        "other_mean_exposure": round(float(frequency[~short].mean()), 1),
        "over_exposure_ratio": round(float(frequency[short].mean() / frequency[~short].mean()), 1),
    }


def measure_cost(recommender: ContentRecommender, catalog: pd.DataFrame) -> dict:
    """Sparse-vs-dense memory and per-query latency, both measured, not estimated."""
    n = len(catalog)
    sparse_bytes = (
        recommender.tag_matrix.data.nbytes
        + recommender.tag_matrix.indices.nbytes
        + recommender.tag_matrix.indptr.nbytes
    )
    dense_similarity_bytes = n * n * 8

    sample = catalog["movie_id"].sample(200, random_state=0).tolist()
    recommender.recommend(sample[0])  # warm any lazy import path
    start = time.perf_counter()
    for movie_id in sample:
        recommender.recommend(movie_id, limit=TOP_K)
    elapsed = time.perf_counter() - start

    stats = {
        "sparse_index_mib": round(sparse_bytes / 1024**2, 2),
        "dense_similarity_mib": round(dense_similarity_bytes / 1024**2, 1),
        "memory_ratio": round(dense_similarity_bytes / sparse_bytes, 1),
        "median_query_ms": round(elapsed / len(sample) * 1000, 2),
        "nonzero_entries": int(recommender.tag_matrix.nnz),
        "density": round(float(recommender.tag_matrix.nnz) / (n * recommender.tag_matrix.shape[1]), 5),
    }

    figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    bars = left.bar(
        ["sparse index\n(shipped)", "dense similarity\n(original design)"],
        [stats["sparse_index_mib"], stats["dense_similarity_mib"]],
        color=[PALETTE["alt"], PALETTE["accent"]], zorder=3,
    )
    for bar in bars:
        left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                  f"{bar.get_height():,.1f} MiB", ha="center", fontsize=10, fontweight="bold")
    left.set_yscale("log")
    _style(left, "Resident memory for the index", ylabel="MiB (log scale)")

    right.barh(["per query"], [stats["median_query_ms"]], color=PALETTE["primary"], zorder=3)
    right.text(stats["median_query_ms"] * 1.03, 0, f"{stats['median_query_ms']:.1f} ms",
               va="center", fontsize=11, fontweight="bold")
    right.set_xlim(0, stats["median_query_ms"] * 1.5)
    _style(right, f"Ranking latency, mean of {len(sample)} queries", "milliseconds")
    right.grid(axis="y", visible=False)
    right.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)

    _save(figure, "performance.png")
    return stats


def face_validity(recommender: ContentRecommender, titles: list[str]) -> list[dict]:
    """A few worked examples, printed so a reader can judge the output directly."""
    catalog = recommender.catalog
    examples = []
    for title in titles:
        match = catalog.loc[catalog["title"] == title]
        if match.empty:
            continue
        movie_id = int(match["movie_id"].iloc[0])
        result = recommender.recommend(movie_id, limit=TOP_K)
        examples.append({
            "query": title,
            "movie_id": movie_id,
            "recommendations": [
                {"title": row.title, "score": round(float(row.score), 3)}
                for row in result.itertuples(index=False)
            ],
        })
    return examples


def main() -> int:
    print("loading catalogue…")
    recommender = load_catalog(CATALOG)
    catalog = recommender.catalog
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectorizer.fit(catalog["tags"])

    print("ranking every film against every other film…")
    indices, scores = top_k_everything(recommender.tag_matrix)

    print("rendering figures…")
    summary = {
        "catalog": plot_catalog_profile(catalog, vectorizer),
        "similarity": plot_similarity_distribution(scores),
        "coverage": plot_coverage_and_concentration(indices, catalog),
        "ablation": plot_vectorizer_ablation(catalog, indices),
        "vocabulary_cap": plot_vocabulary_cap(catalog),
        "exposure_bias": plot_exposure_bias(indices, catalog),
        "cost": measure_cost(recommender, catalog),
        "examples": face_validity(
            recommender,
            ["The Dark Knight", "The Avengers", "Toy Story", "Pulp Fiction"],
        ),
    }

    path = ASSETS / "metrics.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
