"""Create documentation figures from the verified California Housing baseline.

The diagrams use only dataset facts and metrics saved in the executed notebook;
they do not reproduce or redistribute the source dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ASSET_DIR = Path(__file__).resolve().parents[1] / "docs" / "assets"
PURPLE = "#5B3FD6"
BLUE = "#247BA0"
TEAL = "#138A72"
ORANGE = "#F4A261"
INK = "#152033"
MUTED = "#66758A"


def save(fig: plt.Figure, filename: str) -> None:
    """Save a high-resolution figure with consistent documentation styling."""
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ASSET_DIR / filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def pipeline_overview() -> None:
    """Show the order that protects the holdout from preprocessing leakage."""
    fig, ax = plt.subplots(figsize=(13, 3.1))
    labels = ["Public dataset", "80/20 split", "Fit on train", "Predict test", "Measure once"]
    details = ["20,640 districts", "16,512 / 4,128", "imputer + scaler + forest", "unseen rows", "MAE · RMSE · R²"]
    colors = ["#EEF1F8", "#E8F4F1", "#F1EDFF", "#EAF3FC", "#FFF3E2"]
    x_positions = np.linspace(0.08, 0.92, len(labels))
    for index, (x, label, detail, color) in enumerate(zip(x_positions, labels, details, colors)):
        ax.text(x, 0.62, label, ha="center", va="center", fontsize=13, fontweight="bold", color=INK,
                bbox={"boxstyle": "round,pad=0.8", "facecolor": color, "edgecolor": PURPLE if index == 2 else "#C9D2E1", "linewidth": 1.8})
        ax.text(x, 0.28, detail, ha="center", va="center", fontsize=10.5, color=MUTED)
        if index < len(labels) - 1:
            ax.annotate("", xy=(x_positions[index + 1] - 0.075, 0.62), xytext=(x + 0.075, 0.62),
                        arrowprops={"arrowstyle": "->", "color": "#8190A5", "lw": 2})
    ax.set(title="The evaluation boundary: transformations learn only from training districts")
    ax.axis("off")
    save(fig, "pipeline_overview.png")


def holdout_boundary() -> None:
    """Visualise exactly which rows may fit model state and which may not."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    train, test = 16512, 4128
    ax.barh([0], [train], color=TEAL, height=0.5, label="Training rows: fitting allowed")
    ax.barh([0], [test], left=[train], color=ORANGE, height=0.5, label="Test rows: held back")
    ax.text(train / 2, 0, f"{train:,} training rows\nfit transformations + model", ha="center", va="center", color="white", fontweight="bold")
    ax.text(train + test / 2, 0, f"{test:,} test rows\npredict only", ha="center", va="center", color=INK, fontweight="bold")
    ax.axvline(train, color="white", linewidth=3)
    ax.annotate("No test information crosses this boundary", xy=(train, 0.35), xytext=(train, 0.85), ha="center",
                arrowprops={"arrowstyle": "-|>", "color": PURPLE, "lw": 1.8}, color=PURPLE, fontweight="bold")
    ax.set_xlim(0, train + test)
    ax.set_yticks([])
    ax.set_xlabel("California Housing districts (20,640 total)")
    ax.set_title("Fixed 80/20 holdout split · random_state = 42", loc="left", fontweight="bold")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    save(fig, "holdout_boundary.png")


def model_structure() -> None:
    """Describe pipeline components and their purpose without implying causality."""
    fig, ax = plt.subplots(figsize=(13, 3.8))
    blocks = [
        ("8 census features", "district-level means\nand coordinates", "#EAF3FC"),
        ("Median imputer", "future-input\nrobustness", "#E8F4F1"),
        ("Standard scaler", "reusable interface\nfor comparisons", "#F1EDFF"),
        ("Random forest", "300 trees\nmin leaf = 2", "#FFF3E2"),
        ("Prediction", "median value\nin $100,000s", "#FDECEF"),
    ]
    positions = np.linspace(0.1, 0.9, len(blocks))
    for index, (x, (title, detail, color)) in enumerate(zip(positions, blocks)):
        ax.text(x, 0.63, title, ha="center", va="center", fontsize=12, color=INK, fontweight="bold",
                bbox={"boxstyle": "round,pad=0.75", "facecolor": color, "edgecolor": "#C9D2E1", "linewidth": 1.5})
        ax.text(x, 0.27, detail, ha="center", va="center", fontsize=10, color=MUTED)
        if index < len(blocks) - 1:
            ax.annotate("", xy=(positions[index + 1] - 0.08, 0.63), xytext=(x + 0.08, 0.63),
                        arrowprops={"arrowstyle": "->", "color": "#8190A5", "lw": 2})
    ax.set(title="Baseline pipeline: a nonlinear tabular model, not a property appraisal engine")
    ax.axis("off")
    save(fig, "model_structure.png")


def metric_cards() -> None:
    """Render the holdout metrics reported by the preserved executed notebook."""
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.5))
    cards = [
        ("MAE", "0.326", "≈ $32,600 typical absolute miss", TEAL),
        ("RMSE", "0.504", "≈ $50,400; penalises larger misses", ORANGE),
        ("R²", "0.806", "holdout variation captured", PURPLE),
    ]
    for ax, (title, value, description, color) in zip(axes, cards):
        ax.set_facecolor("#F8FAFD")
        ax.text(0.5, 0.72, title, ha="center", va="center", fontsize=14, color=MUTED, fontweight="bold")
        ax.text(0.5, 0.45, value, ha="center", va="center", fontsize=34, color=color, fontweight="bold")
        ax.text(0.5, 0.18, description, ha="center", va="center", fontsize=9.7, color=INK, wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#DCE3EF")
            spine.set_linewidth(1.5)
    fig.suptitle("Reported holdout result · target unit is $100,000s", fontsize=15, fontweight="bold", color=INK)
    save(fig, "holdout_metrics.png")


def feature_importance_chart() -> None:
    """Plot the recovered impurity importance ranking from the executed run."""
    names = ["MedInc", "AveOccup", "Latitude", "Longitude", "HouseAge", "AveRooms", "Population", "AveBedrms"]
    values = [0.535, 0.138, 0.088, 0.088, 0.053, 0.037, 0.033, 0.028]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    positions = np.arange(len(names))
    ax.barh(positions, values, color=[PURPLE] + [BLUE] * (len(names) - 1))
    ax.set_yticks(positions, names)
    ax.invert_yaxis()
    ax.set_xlabel("Impurity-based importance")
    ax.set_title("What the fitted forest relied on", loc="left", fontweight="bold")
    for position, value in zip(positions, values):
        ax.text(value + 0.008, position, f"{value:.3f}", va="center", color=INK, fontsize=10)
    ax.set_xlim(0, 0.62)
    ax.grid(axis="x", alpha=0.2)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.text(0, -1.2, "A reliance ranking is not a causal explanation; correlated features share and distort importance.", color=MUTED, fontsize=9.5)
    save(fig, "feature_importance.png")


def scope_boundary() -> None:
    """Make the valid project claim and its non-claims visually unambiguous."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    panels = [
        ("This project supports", ["Reproducing a 1990 census-district baseline", "Comparing leakage-safe tabular workflows", "Inspecting model error and feature reliance"], "#E8F4F1", TEAL),
        ("This project does not support", ["A current listing price or an individual appraisal", "A causal claim about income or location", "Deployment without current data and spatial validation"], "#FDECEF", "#C4455A"),
    ]
    for ax, (title, items, background, accent) in zip(axes, panels):
        ax.set_facecolor(background)
        ax.text(0.06, 0.86, title, transform=ax.transAxes, fontsize=16, fontweight="bold", color=accent)
        for index, item in enumerate(items):
            y = 0.62 - index * 0.23
            ax.text(0.08, y, "•", transform=ax.transAxes, fontsize=20, color=accent, va="center")
            ax.text(0.15, y, item, transform=ax.transAxes, fontsize=11.2, color=INK, va="center", wrap=True)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(accent)
            spine.set_linewidth(1.5)
    fig.suptitle("Interpretation boundary", fontsize=16, fontweight="bold", color=INK)
    save(fig, "interpretation_boundary.png")


if __name__ == "__main__":
    pipeline_overview()
    holdout_boundary()
    model_structure()
    metric_cards()
    feature_importance_chart()
    scope_boundary()
    print(f"Wrote 6 documentation figures to {ASSET_DIR}")
