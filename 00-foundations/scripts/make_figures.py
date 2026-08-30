"""Regenerate every figure used by the README and the HTML walkthrough.

Run from the project directory:

    python scripts/make_figures.py

All figures are written to ``docs/assets/``. The script is deterministic: it
seeds NumPy and uses fixed splits, so re-running it produces byte-comparable
plots rather than a noisy diff.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import load_enjoy_sport, load_income, load_play_tennis  # noqa: E402
from src.find_s import ANY, FindS  # noqa: E402
from src.impurity import entropy, information_gain  # noqa: E402
from src.knn import KNearestNeighbours  # noqa: E402

ASSETS = PROJECT_ROOT / "docs" / "assets"
RANDOM_SEED = 0

# A single palette keeps the figures visually consistent across the walkthrough.
INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#2f6f4e"
ACCENT_ALT = "#b4532a"
GRID = "#dfe3e8"


def _style_axes(ax) -> None:
    """Apply the shared minimal axis styling used by every figure."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.title.set_color(INK)


def _save(fig, name: str) -> None:
    """Write a figure to the assets directory and report the path."""
    ASSETS.mkdir(parents=True, exist_ok=True)
    path = ASSETS / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


def figure_information_gain() -> None:
    """Bar chart of root information gain for each PlayTennis feature."""
    X, y = load_play_tennis()
    gains = {column: information_gain(y, X[column]) for column in X.columns}
    order = sorted(gains, key=gains.__getitem__, reverse=True)
    values = [gains[name] for name in order]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    colours = [ACCENT if name == order[0] else MUTED for name in order]
    bars = ax.bar(order, values, color=colours, width=0.6)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.006,
            f"{value:.4f}",
            ha="center",
            fontsize=9,
            color=INK,
        )

    ax.set_title(
        f"Information gain at the root (parent entropy = {entropy(y):.4f} bits)",
        fontsize=11,
    )
    ax.set_ylabel("Gain (bits)")
    ax.set_ylim(0, max(values) * 1.25)
    _style_axes(ax)
    _save(fig, "information_gain.png")


def figure_impurity_curves() -> None:
    """Entropy and Gini as a function of class balance."""
    proportions = np.linspace(0.001, 0.999, 400)
    entropies = [-(p * np.log2(p) + (1 - p) * np.log2(1 - p)) for p in proportions]
    ginis = [1 - (p**2 + (1 - p) ** 2) for p in proportions]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(proportions, entropies, color=ACCENT, lw=2, label="Entropy (bits)")
    ax.plot(proportions, ginis, color=ACCENT_ALT, lw=2, label="Gini impurity")
    ax.axvline(0.5, color=GRID, ls="--", lw=1)

    ax.set_title("Both measures peak at a 50/50 split and vanish at purity", fontsize=11)
    ax.set_xlabel("Proportion of the positive class")
    ax.set_ylabel("Impurity")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    _style_axes(ax)
    _save(fig, "impurity_curves.png")


def figure_find_s_generalisation() -> None:
    """Table showing the Find-S hypothesis after each positive example."""
    X, y = load_enjoy_sport()
    steps = FindS().history(X, y)

    fig, ax = plt.subplots(figsize=(9, 2.2))
    ax.axis("off")

    cell_text = [
        [value if value != ANY else "?" for value in step] for step in steps
    ]
    row_labels = [f"after positive #{i + 1}" for i in range(len(steps))]

    table = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=list(X.columns),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Highlight the constraints that have been relaxed to "?".
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        if row > 0 and cell.get_text().get_text() == "?":
            cell.set_facecolor("#fdf1e7")
            cell.get_text().set_color(ACCENT_ALT)

    ax.set_title(
        "Find-S generalises only when a positive example disagrees",
        fontsize=11,
        color=INK,
        pad=18,
    )
    _save(fig, "find_s_generalisation.png")


def figure_knn_k_sweep() -> None:
    """Train and test accuracy on Iris as k varies, from our implementation."""
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=1
    )

    k_values = range(1, 31)
    train_scores, test_scores = [], []
    for k in k_values:
        model = KNearestNeighbours(n_neighbors=k).fit(X_train, y_train)
        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(k_values, train_scores, color=MUTED, lw=1.8, label="Train accuracy")
    ax.plot(k_values, test_scores, color=ACCENT, lw=2, label="Test accuracy")

    best_k = list(k_values)[int(np.argmax(test_scores))]
    ax.axvline(best_k, color=ACCENT_ALT, ls="--", lw=1)
    ax.text(
        best_k + 0.4,
        min(test_scores) + 0.01,
        f"best k = {best_k}",
        color=ACCENT_ALT,
        fontsize=9,
    )

    ax.set_title(
        "k = 1 memorises the training set; larger k trades fit for stability",
        fontsize=11,
    )
    ax.set_xlabel("k (number of neighbours)")
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    _style_axes(ax)
    _save(fig, "knn_k_sweep.png")


def figure_kmeans_scaling() -> None:
    """Side-by-side k-means clusters before and after min-max scaling."""
    frame = load_income()
    features = ["Age", "Income($)"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    palette = [ACCENT, ACCENT_ALT, "#3a6ea5"]

    for ax, scaled in zip(axes, [False, True]):
        values = frame[features].to_numpy(dtype=float)
        if scaled:
            values = MinMaxScaler().fit_transform(values)

        model = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=10)
        labels = model.fit_predict(values)

        for cluster in range(3):
            mask = labels == cluster
            ax.scatter(
                values[mask, 0],
                values[mask, 1],
                s=45,
                color=palette[cluster],
                edgecolor="white",
                linewidth=0.6,
            )
        ax.scatter(
            model.cluster_centers_[:, 0],
            model.cluster_centers_[:, 1],
            marker="X",
            s=140,
            color=INK,
            label="centroid",
        )

        ax.set_title(
            "After min-max scaling" if scaled else "Raw features",
            fontsize=11,
        )
        ax.set_xlabel("Age (scaled)" if scaled else "Age")
        ax.set_ylabel("Income (scaled)" if scaled else "Income ($)")
        _style_axes(ax)

    fig.suptitle(
        "Unscaled income dominates the distance metric and the clusters are wrong",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    _save(fig, "kmeans_scaling.png")


def figure_kmeans_elbow() -> None:
    """Elbow plot of within-cluster sum of squares against k."""
    frame = load_income()
    values = MinMaxScaler().fit_transform(frame[["Age", "Income($)"]])

    k_values = range(1, 10)
    inertias = [
        KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10).fit(values).inertia_
        for k in k_values
    ]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(k_values, inertias, color=ACCENT, lw=2, marker="o", markersize=5)
    ax.axvline(3, color=ACCENT_ALT, ls="--", lw=1)
    ax.text(3.1, max(inertias) * 0.7, "elbow at k = 3", color=ACCENT_ALT, fontsize=9)

    ax.set_title("Within-cluster sum of squares falls sharply until k = 3", fontsize=11)
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Inertia")
    _style_axes(ax)
    _save(fig, "kmeans_elbow.png")


FIGURES = (
    figure_information_gain,
    figure_impurity_curves,
    figure_find_s_generalisation,
    figure_knn_k_sweep,
    figure_kmeans_scaling,
    figure_kmeans_elbow,
)


def main() -> None:
    """Regenerate every figure."""
    np.random.seed(RANDOM_SEED)
    for build in FIGURES:
        build()
    print(f"\n{len(FIGURES)} figures written to {ASSETS.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
