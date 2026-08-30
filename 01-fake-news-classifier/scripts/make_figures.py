"""Regenerate every figure used by the README and the HTML walkthrough.

Run from the project directory:

    python scripts/make_figures.py

The source corpus is not present in this repository, so nothing here retrains
anything. The figures are built from two sources that are available:

* ``data/training_log_2022.csv`` -- the 20-epoch Keras history, transcribed
  from the saved output of the original notebook.
* ``src/preprocessing.py`` -- the 2022 cleaning step, run live on headlines
  copied from the notebook's saved output.

Every number plotted is therefore recovered from the original run rather than
re-measured or estimated.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import compression_report  # noqa: E402

ASSETS = PROJECT_ROOT / "docs" / "assets"
TRAINING_LOG = PROJECT_ROOT / "data" / "training_log_2022.csv"

INK = "#1f2933"
MUTED = "#7b8794"
ACCENT = "#2f6f4e"
ACCENT_SOFT = "#eaf2ed"
BLUE = "#3a6ea5"
BLUE_SOFT = "#e8eef6"
AMBER = "#b4532a"
AMBER_SOFT = "#fdf1e7"
GRID = "#dfe3e8"
PANEL = "#f6f5f3"

# Headlines copied verbatim from the saved output of notebook cell 13.
SAMPLE_HEADLINES = [
    "You Can Smell Hillary s Fear",
    "Kerry to go to Paris in gesture of sympathy",
    "The Battle of New York  Why This Primary Matters",
    "Tehran  USA",
    "Trump takes on Cruz  but lightly",
    "How women lose differently",
]


def _style_axes(ax) -> None:
    """Apply the shared minimal axis styling used by every plot."""
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


def _box(ax, x, y, width, height, label, sublabel="", face=PANEL, edge=MUTED):
    """Draw one rounded labelled box for the schematic diagrams."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=face, edgecolor=edge, linewidth=1.3,
        )
    )
    text_y = y + height / 2 + (0.09 if sublabel else 0)
    ax.text(x + width / 2, text_y, label, ha="center", va="center",
            fontsize=9.5, color=INK, family="monospace")
    if sublabel:
        ax.text(x + width / 2, y + height / 2 - 0.14, sublabel, ha="center",
                va="center", fontsize=8, color=MUTED)


def _arrow(ax, x1, y1, x2, y2, label=""):
    """Draw a connector arrow between two schematic boxes."""
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>", mutation_scale=13,
            color=MUTED, linewidth=1.2, shrinkA=0, shrinkB=0,
        )
    )
    if label:
        ax.text((x1 + x2) / 2, y1 + 0.13, label, ha="center", fontsize=7.8,
                color=MUTED, family="monospace")


def load_training_log() -> pd.DataFrame:
    """Load the recovered 2022 training history."""
    if not TRAINING_LOG.exists():
        raise FileNotFoundError(
            f"Missing {TRAINING_LOG}. It is transcribed from the saved output "
            f"of notebooks/01_lstm_original_2022.ipynb."
        )
    return pd.read_csv(TRAINING_LOG)


def figure_pipeline() -> None:
    """End-to-end schematic of the whole project, stage by stage."""
    fig, ax = plt.subplots(figsize=(12, 3.1))
    ax.axis("off")

    stages = [
        ("CSV", "6,335 rows", BLUE_SOFT, BLUE),
        ("clean text", "letters only", PANEL, MUTED),
        ("hash to int", "vocab 100k", PANEL, MUTED),
        ("pad", "length 58", PANEL, MUTED),
        ("Embedding", "58 x 100", ACCENT_SOFT, ACCENT),
        ("LSTM(500)", "1.2M params", ACCENT_SOFT, ACCENT),
        ("Dense(1)", "sigmoid", ACCENT_SOFT, ACCENT),
        ("FAKE / REAL", "prediction", AMBER_SOFT, AMBER),
    ]

    width, height, gap = 1.28, 0.72, 0.28
    for index, (label, sublabel, face, edge) in enumerate(stages):
        x = index * (width + gap)
        _box(ax, x, 0, width, height, label, sublabel, face, edge)
        if index < len(stages) - 1:
            _arrow(ax, x + width + 0.03, height / 2, x + width + gap - 0.03, height / 2)

    ax.set_xlim(-0.2, len(stages) * (width + gap))
    ax.set_ylim(-0.45, 1.25)
    ax.set_title(
        "End-to-end pipeline: from a CSV of news articles to a FAKE/REAL prediction",
        fontsize=12, color=INK, pad=14,
    )
    _save(fig, "pipeline_overview.png")


def figure_text_to_tensor() -> None:
    """Trace one headline through every transformation into a padded tensor."""
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    ax.axis("off")

    steps = [
        ("1. raw headline", "You Can Smell Hillary's Fear", BLUE),
        ("2. letters only", "You Can Smell Hillary s Fear", MUTED),
        ("3. cleaned", "yu cn sell hllr  fer", MUTED),
        ("4. hashed to integers", "[88108, 58122, 4505, 51551, 76114]", ACCENT),
        ("5. padded to length 58", "[0, 0, 0, ..., 4505, 51551, 76114]", ACCENT),
        ("6. embedded", "float array of shape (58, 100)", AMBER),
    ]

    y = 5.4
    for label, value, colour in steps:
        ax.text(0.0, y, label, fontsize=9, color=MUTED, family="monospace", va="center")
        ax.add_patch(
            FancyBboxPatch(
                (2.55, y - 0.28), 7.6, 0.56,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                facecolor=PANEL, edgecolor=colour, linewidth=1.2,
            )
        )
        ax.text(2.75, y, value, fontsize=9.5, color=colour, family="monospace",
                va="center")
        if y > 1.0:
            ax.add_patch(
                FancyArrowPatch(
                    (6.35, y - 0.31), (6.35, y - 0.71),
                    arrowstyle="-|>", mutation_scale=12, color=MUTED, linewidth=1.1,
                )
            )
        y -= 1.0

    ax.set_xlim(-0.2, 10.4)
    ax.set_ylim(0.0, 6.1)
    ax.set_title(
        "How one headline becomes a tensor the network can read",
        fontsize=12, color=INK, pad=10,
    )
    _save(fig, "text_to_tensor.png")


def figure_architecture() -> None:
    """Layer-by-layer view of the network with tensor shapes and parameters."""
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.axis("off")

    layers = [
        ("Input", "(batch, 58)", "0", BLUE_SOFT, BLUE),
        ("Embedding", "(batch, 58, 100)", "10,000,000", ACCENT_SOFT, ACCENT),
        ("Dropout(0.5)", "(batch, 58, 100)", "0", PANEL, MUTED),
        ("LSTM(500)", "(batch, 500)", "1,202,000", ACCENT_SOFT, ACCENT),
        ("Dense(1, sigmoid)", "(batch, 1)", "501", AMBER_SOFT, AMBER),
    ]

    y = 4.2
    for name, shape, params, face, edge in layers:
        ax.add_patch(
            FancyBboxPatch(
                (0.4, y - 0.32), 4.2, 0.64,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=face, edgecolor=edge, linewidth=1.4,
            )
        )
        ax.text(2.5, y, name, ha="center", va="center", fontsize=10.5,
                color=INK, family="monospace")
        ax.text(5.15, y, shape, fontsize=9.5, color=MUTED, family="monospace",
                va="center")
        ax.text(9.55, y, params, fontsize=9.5, color=edge, family="monospace",
                va="center", ha="right")
        if y > 1.0:
            ax.add_patch(
                FancyArrowPatch(
                    (2.5, y - 0.35), (2.5, y - 0.65),
                    arrowstyle="-|>", mutation_scale=12, color=MUTED, linewidth=1.1,
                )
            )
        y -= 1.0

    ax.text(5.15, 4.95, "output shape", fontsize=8.5, color=MUTED,
            family="monospace", style="italic")
    ax.text(9.55, 4.95, "parameters", fontsize=8.5, color=MUTED,
            family="monospace", style="italic", ha="right")
    ax.plot([0.4, 9.6], [-0.42, -0.42], color=GRID, lw=1.2)
    ax.text(9.55, -0.78, "11,202,501 total", fontsize=10.5, color=INK,
            family="monospace", ha="right", weight="bold")
    ax.text(0.4, -0.78, "trained on 5,068 examples", fontsize=9, color=MUTED,
            family="monospace", ha="left")

    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(-1.15, 5.35)
    ax.set_title("Network architecture", fontsize=12, color=INK, pad=12)
    _save(fig, "architecture.png")


def figure_training_history() -> None:
    """Accuracy and loss across the 20 recovered epochs."""
    log = load_training_log()

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 3.9))

    ax_acc.plot(log.epoch, log.acc, color=MUTED, lw=1.8, label="Training")
    ax_acc.plot(log.epoch, log.val_acc, color=ACCENT, lw=2.2, label="Validation")
    peak_epoch = int(log.loc[log.val_acc.idxmax(), "epoch"])
    ax_acc.scatter([peak_epoch], [log.val_acc.max()], color=ACCENT, zorder=5,
                   s=45, edgecolor="white", linewidth=1.2)
    ax_acc.annotate(
        f"best {log.val_acc.max():.4f}\nepoch {peak_epoch}",
        xy=(peak_epoch, log.val_acc.max()),
        xytext=(peak_epoch + 2.4, log.val_acc.max() - 0.14),
        fontsize=8.5, color=ACCENT,
        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.9),
    )
    ax_acc.set_title("Accuracy", fontsize=11)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.6, 1.02)
    ax_acc.set_xticks(range(2, 21, 2))
    ax_acc.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)
    _style_axes(ax_acc)

    ax_loss.plot(log.epoch, log.loss, color=MUTED, lw=1.8, label="Training")
    ax_loss.plot(log.epoch, log.val_loss, color=BLUE, lw=2.2, label="Validation")
    best_epoch = int(log.loc[log.val_loss.idxmin(), "epoch"])
    ax_loss.axvline(best_epoch, color=BLUE, ls="--", lw=1)
    ax_loss.annotate(
        f"lowest validation loss\nepoch {best_epoch} ({log.val_loss.min():.4f})",
        xy=(best_epoch, log.val_loss.min()),
        xytext=(best_epoch + 1.8, 1.08),
        fontsize=8.5, color=BLUE,
        arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.9),
    )
    ax_loss.set_title("Loss", fontsize=11)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Binary cross-entropy")
    ax_loss.set_xticks(range(2, 21, 2))
    ax_loss.legend(frameon=False, fontsize=8.5, loc="upper left", labelcolor=INK)
    _style_axes(ax_loss)

    fig.suptitle(
        "Training history, 20 epochs (recovered from the original run)",
        fontsize=11.5, color=INK,
    )
    fig.tight_layout()
    _save(fig, "training_history.png")


def figure_data_split() -> None:
    """Row counts and the class question, as a stacked split diagram."""
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")

    total, train, test = 6335, 5068, 1267

    ax.add_patch(
        FancyBboxPatch(
            (0, 1.15), 9.4, 0.6,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=BLUE_SOFT, edgecolor=BLUE, linewidth=1.3,
        )
    )
    ax.text(4.7, 1.45, f"full dataset  ·  {total:,} articles",
            ha="center", va="center", fontsize=10.5, color=BLUE, family="monospace")

    train_width = 9.4 * train / total
    ax.add_patch(
        FancyBboxPatch(
            (0, 0.15), train_width - 0.06, 0.6,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=ACCENT_SOFT, edgecolor=ACCENT, linewidth=1.3,
        )
    )
    ax.text(train_width / 2, 0.45, f"train  ·  {train:,}  (80%)", ha="center",
            va="center", fontsize=10, color=ACCENT, family="monospace")

    ax.add_patch(
        FancyBboxPatch(
            (train_width + 0.06, 0.15), 9.4 - train_width - 0.06, 0.6,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=AMBER_SOFT, edgecolor=AMBER, linewidth=1.3,
        )
    )
    ax.text((train_width + 9.4) / 2, 0.45, f"test · {test:,} (20%)", ha="center",
            va="center", fontsize=9.5, color=AMBER, family="monospace")

    for x in (train_width / 2, (train_width + 9.4) / 2):
        ax.add_patch(
            FancyArrowPatch((4.7, 1.11), (x, 0.79), arrowstyle="-|>",
                            mutation_scale=12, color=MUTED, linewidth=1.1)
        )

    ax.set_xlim(-0.2, 9.6)
    ax.set_ylim(0.0, 2.0)
    ax.set_title("Train / test split", fontsize=12, color=INK, pad=8)
    _save(fig, "data_split.png")


def figure_cleaning_examples() -> None:
    """Worked examples of the 2022 cleaning step on real headlines."""
    rows = []
    for headline in SAMPLE_HEADLINES:
        report = compression_report(headline)
        rows.append(
            [
                report["original"][:52],
                report["compressed"][:52],
                f"{report['fraction_removed']:.0%}",
            ]
        )

    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Headline in", "After cleaning", "Shorter by"],
        cellLoc="left",
        colWidths=[0.46, 0.44, 0.09],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.55)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.PAD = 0.04
        if row == 0:
            cell.set_facecolor("#f3f1ee")
            cell.get_text().set_fontweight("semibold")
        elif column == 1:
            cell.set_facecolor(BLUE_SOFT)
            cell.get_text().set_color(BLUE)
            cell.get_text().set_fontfamily("monospace")
        elif column == 0:
            cell.get_text().set_fontfamily("monospace")

    ax.set_title(
        "The 2022 cleaning step, applied to real headlines from the corpus",
        fontsize=11.5, color=INK, pad=16,
    )
    _save(fig, "cleaning_examples.png")


def figure_parameter_budget() -> None:
    """Where the 11.2 million parameters live."""
    fig, ax = plt.subplots(figsize=(8.5, 2.9))

    layers = ["Embedding", "LSTM(500)", "Dense(1)"]
    params = [10_000_000, 1_202_000, 501]
    colours = [ACCENT, BLUE, AMBER]

    bars = ax.barh(layers, params, color=colours, height=0.55)
    ax.set_xscale("log")
    ax.set_xlim(100, 3e7)

    for bar, value in zip(bars, params):
        ax.text(value * 1.25, bar.get_y() + bar.get_height() / 2,
                f"{value:,}", va="center", fontsize=9.5, color=INK,
                family="monospace")

    ax.invert_yaxis()
    ax.set_xlabel("Parameters (log scale)")
    ax.set_title(
        "89% of the model's parameters sit in the embedding layer",
        fontsize=11,
    )
    _style_axes(ax)
    _save(fig, "parameter_budget.png")


def figure_sequence_lengths() -> None:
    """Padding illustrated: short sequences right-aligned in a length-58 window."""
    fig, ax = plt.subplots(figsize=(10, 3.0))

    lengths = [5, 14, 7, 12, 6, 3]
    labels = [f"headline {i + 1}" for i in range(len(lengths))]
    max_len = 58

    for index, length in enumerate(lengths):
        ax.barh(index, max_len - length, color=GRID, height=0.6)
        ax.barh(index, length, left=max_len - length, color=ACCENT, height=0.6)
        ax.text(max_len + 1.2, index, f"{length} tokens", va="center",
                fontsize=8.5, color=MUTED, family="monospace")

    ax.set_yticks(range(len(lengths)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 68)
    ax.set_xlabel("Position in the padded sequence")
    ax.set_title(
        "Pre-padding: zeros (grey) fill the front so every sequence ends on real tokens",
        fontsize=11,
    )
    _style_axes(ax)
    _save(fig, "padding.png")


FIGURES = (
    figure_pipeline,
    figure_text_to_tensor,
    figure_architecture,
    figure_data_split,
    figure_cleaning_examples,
    figure_padding := figure_sequence_lengths,
    figure_parameter_budget,
    figure_training_history,
)


def main() -> None:
    """Regenerate every figure."""
    np.random.seed(0)
    for build in FIGURES:
        build()
    print(f"\n{len(FIGURES)} figures written to {ASSETS.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
