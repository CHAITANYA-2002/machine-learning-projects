"""Regenerate every figure used by the README and the HTML walkthrough.

Run from the project directory:

    python scripts/make_figures.py

The image dataset is not present in this repository, so nothing here retrains
anything and no figure shows a real photograph. The figures are built from:

* ``data/training_log_2022.csv`` -- the 25-epoch history, transcribed from the
  saved output of the original notebook.
* ``src/architecture.py`` -- the layer shapes and parameter counts, computed
  analytically and tested against the notebook's saved ``model.summary()``.
* ``src/augmentation.py`` -- the augmentation transforms, demonstrated on a
  synthetic test pattern rather than on any person's photograph.
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
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.architecture import build_architecture, total_parameters  # noqa: E402
from src.augmentation import (  # noqa: E402
    augment,
    describe,
    horizontal_flip,
    rotate,
    shear,
    shift,
    zoom,
)

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


def test_pattern(size: int = 150) -> Image.Image:
    """Build a synthetic, clearly asymmetric image for demonstrating transforms.

    A neutral geometric pattern is used deliberately: the augmentation effects
    are easier to see on straight edges than on a photograph, and no person's
    image needs to be redistributed to illustrate the pipeline.
    """
    image = Image.new("RGB", (size, size), "#ffffff")
    draw = ImageDraw.Draw(image)

    draw.rectangle([8, 8, size - 8, size - 8], outline=MUTED, width=2)
    draw.rectangle([22, 22, 68, 68], fill=ACCENT)
    draw.ellipse([84, 26, 126, 68], fill=BLUE)
    draw.polygon([(30, 128), (52, 88), (74, 128)], fill=AMBER)
    for offset in range(0, 40, 8):
        draw.line([(90, 92 + offset), (128, 92 + offset)], fill=INK, width=2)
    return image


def load_training_log() -> pd.DataFrame:
    """Load the recovered 2022 training history."""
    if not TRAINING_LOG.exists():
        raise FileNotFoundError(
            f"Missing {TRAINING_LOG}. It is transcribed from the saved output of "
            f"notebooks/01_gender_image_classifier.ipynb."
        )
    return pd.read_csv(TRAINING_LOG)


def figure_pipeline() -> None:
    """End-to-end schematic of the project."""
    fig, ax = plt.subplots(figsize=(12, 2.9))
    ax.axis("off")

    stages = [
        ("image folders", "2 classes", BLUE_SOFT, BLUE),
        ("resize", "150x150 RGB", PANEL, MUTED),
        ("rescale", "0-255 to 0-1", PANEL, MUTED),
        ("augment", "flip/shift/zoom", AMBER_SOFT, AMBER),
        ("4x conv+pool", "150 to 7 px", ACCENT_SOFT, ACCENT),
        ("flatten", "6,272 values", ACCENT_SOFT, ACCENT),
        ("dense 512", "+ dropout", ACCENT_SOFT, ACCENT),
        ("sigmoid", "one probability", AMBER_SOFT, AMBER),
    ]

    width, height, gap = 1.3, 0.7, 0.26
    for index, (label, sublabel, face, edge) in enumerate(stages):
        x = index * (width + gap)
        ax.add_patch(
            FancyBboxPatch(
                (x, 0), width, height,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=face, edgecolor=edge, linewidth=1.3,
            )
        )
        ax.text(x + width / 2, height / 2 + 0.09, label, ha="center", va="center",
                fontsize=9.5, color=INK, family="monospace")
        ax.text(x + width / 2, height / 2 - 0.14, sublabel, ha="center", va="center",
                fontsize=8, color=MUTED)
        if index < len(stages) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + width + 0.02, height / 2), (x + width + gap - 0.02, height / 2),
                    arrowstyle="-|>", mutation_scale=13, color=MUTED, linewidth=1.2,
                )
            )

    ax.set_xlim(-0.2, len(stages) * (width + gap))
    ax.set_ylim(-0.4, 1.15)
    ax.set_title(
        "End-to-end pipeline: from a folder of images to one probability",
        fontsize=12, color=INK, pad=12,
    )
    _save(fig, "pipeline_overview.png")


def figure_augmentation() -> None:
    """The augmentation transforms, each shown on the synthetic test pattern."""
    source = test_pattern()

    panels = [
        ("original", source),
        ("horizontal flip", horizontal_flip(source)),
        ("shift 20%", shift(source, 0.2, -0.15)),
        ("shear 10 degrees", shear(source, 10)),
        ("zoom 1.25x", zoom(source, 1.25)),
        ("zoom 0.8x", zoom(source, 0.8)),
        ("rotate 12 degrees", rotate(source, 12)),
        ("full pipeline", augment(source, seed=3)),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(11, 5.8))
    for ax, (label, image) in zip(axes.ravel(), panels):
        ax.imshow(np.asarray(image))
        ax.set_title(label, fontsize=10, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)

    fig.suptitle(
        "Augmentation transforms, shown on a synthetic test pattern",
        fontsize=12, color=INK, y=0.99,
    )
    fig.text(
        0.5, 0.015,
        "The shear and rotation angles are exaggerated here for visibility; "
        "the notebook uses 0.2 degrees for both.",
        ha="center", fontsize=9, color=MUTED,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    _save(fig, "augmentation.png")


def figure_feature_maps() -> None:
    """How the tensor shrinks spatially and deepens through the network."""
    layers = [layer for layer in build_architecture() if len(layer.output_shape) == 3]

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    ax.axis("off")
    # Feature maps are square, so the boxes must be drawn square too.
    ax.set_aspect("equal")

    stages = [("input", 150, 3)] + [
        (layer.name.replace("max_pooling2d", "pool").replace("conv2d", "conv"),
         layer.output_shape[0], layer.output_shape[2])
        for layer in layers
    ]

    max_size = 150
    for index, (name, size, channels) in enumerate(stages):
        x = index * 1.16
        scaled = 0.9 * size / max_size
        colour = ACCENT if "conv" in name else (BLUE if "pool" in name else MUTED)
        face = ACCENT_SOFT if "conv" in name else (BLUE_SOFT if "pool" in name else PANEL)

        ax.add_patch(
            FancyBboxPatch(
                (x, 0.66 - scaled / 2), scaled, scaled,
                boxstyle="round,pad=0.006,rounding_size=0.02",
                facecolor=face, edgecolor=colour, linewidth=1.3,
            )
        )
        ax.text(x + scaled / 2, 1.19, name, ha="center", va="bottom", fontsize=7.6,
                color=MUTED, family="monospace", rotation=0)
        ax.text(x + scaled / 2, 0.14, f"{size}x{size}", ha="center", va="top",
                fontsize=8, color=INK, family="monospace")
        ax.text(x + scaled / 2, 0.02, f"{channels}ch", ha="center", va="top",
                fontsize=8, color=colour, family="monospace")

    ax.set_xlim(-0.1, len(stages) * 1.16)
    ax.set_ylim(-0.16, 1.34)
    ax.set_title(
        "Each block trades spatial size for feature depth: 150x150x3 becomes 7x7x128",
        fontsize=11.5, color=INK, pad=10,
    )
    _save(fig, "feature_maps.png")


def figure_architecture() -> None:
    """Layer-by-layer table of shapes and parameter counts."""
    layers = build_architecture()

    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axis("off")

    y = float(len(layers))
    for layer in layers:
        if layer.kind == "Conv2D":
            face, edge = ACCENT_SOFT, ACCENT
        elif layer.kind == "MaxPooling2D":
            face, edge = BLUE_SOFT, BLUE
        elif layer.kind == "Dense":
            face, edge = AMBER_SOFT, AMBER
        else:
            face, edge = PANEL, MUTED

        ax.add_patch(
            FancyBboxPatch(
                (0.3, y - 0.34), 3.5, 0.68,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=face, edgecolor=edge, linewidth=1.3,
            )
        )
        ax.text(2.05, y, layer.name, ha="center", va="center", fontsize=9.5,
                color=INK, family="monospace")
        shape = f"({', '.join(str(v) for v in layer.output_shape)})"
        ax.text(4.15, y, shape, fontsize=9, color=MUTED, family="monospace", va="center")
        ax.text(9.7, y, f"{layer.parameters:,}", fontsize=9, color=edge,
                family="monospace", va="center", ha="right")
        y -= 1.0

    ax.text(4.15, len(layers) + 0.75, "output shape", fontsize=8.5, color=MUTED,
            family="monospace", style="italic")
    ax.text(9.7, len(layers) + 0.75, "parameters", fontsize=8.5, color=MUTED,
            family="monospace", style="italic", ha="right")
    ax.plot([0.3, 9.75], [0.35, 0.35], color=GRID, lw=1.2)
    ax.text(9.7, 0.0, f"{total_parameters():,} total", fontsize=10.5, color=INK,
            family="monospace", ha="right", weight="bold")
    ax.text(0.3, 0.0, "3,309 training images", fontsize=9, color=MUTED,
            family="monospace", ha="left")

    ax.set_xlim(0.0, 10.1)
    ax.set_ylim(-0.45, len(layers) + 1.15)
    ax.set_title("Network architecture", fontsize=12, color=INK, pad=12)
    _save(fig, "architecture.png")


def figure_parameter_budget() -> None:
    """Where the 3.45 million parameters live."""
    layers = [layer for layer in build_architecture() if layer.parameters > 0]
    names = [layer.name for layer in layers]
    params = [layer.parameters for layer in layers]
    colours = [AMBER if layer.kind == "Dense" else ACCENT for layer in layers]

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    bars = ax.barh(names, params, color=colours, height=0.6)
    ax.set_xscale("log")
    ax.set_xlim(100, 1.2e7)

    for bar, value in zip(bars, params):
        ax.text(value * 1.2, bar.get_y() + bar.get_height() / 2, f"{value:,}",
                va="center", fontsize=9, color=INK, family="monospace")

    ax.invert_yaxis()
    ax.set_xlabel("Parameters (log scale)")
    share = max(params) / total_parameters()
    ax.set_title(
        f"The single dense layer holds {share:.0%} of all parameters",
        fontsize=11,
    )
    _style_axes(ax)
    _save(fig, "parameter_budget.png")


def figure_training_history() -> None:
    """Accuracy and loss across the 25 recovered epochs."""
    log = load_training_log()

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax_acc.plot(log.epoch, log.accuracy, color=ACCENT, lw=2.2)
    ax_acc.scatter([log.epoch.iloc[-1]], [log.accuracy.iloc[-1]], color=ACCENT,
                   s=45, zorder=5, edgecolor="white", linewidth=1.2)
    ax_acc.annotate(
        f"{log.accuracy.iloc[-1]:.4f}",
        xy=(log.epoch.iloc[-1], log.accuracy.iloc[-1]),
        xytext=(log.epoch.iloc[-1] - 5.5, log.accuracy.iloc[-1] - 0.045),
        fontsize=9, color=ACCENT,
        arrowprops=dict(arrowstyle="-", color=ACCENT, lw=0.9),
    )
    ax_acc.axhline(0.5, color=GRID, ls="--", lw=1)
    ax_acc.text(1, 0.512, "chance for two classes", fontsize=8.5, color=MUTED)
    ax_acc.set_title("Training accuracy", fontsize=11)
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(0.45, 0.85)
    ax_acc.set_xticks(range(0, 26, 5))
    _style_axes(ax_acc)

    ax_loss.plot(log.epoch, log.loss, color=BLUE, lw=2.2)
    ax_loss.set_title("Training loss", fontsize=11)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Binary cross-entropy")
    ax_loss.set_xticks(range(0, 26, 5))
    _style_axes(ax_loss)

    fig.suptitle(
        "Training history, 25 epochs (recovered from the original run)",
        fontsize=11.5, color=INK,
    )
    fig.tight_layout()
    _save(fig, "training_history.png")


def figure_augmentation_settings() -> None:
    """The augmentation configuration as a labelled table."""
    settings = describe()
    rows = [
        [setting.name, str(setting.value)[:8], setting.unit, setting.effect]
        for setting in settings
    ]

    fig, ax = plt.subplots(figsize=(10.5, 2.7))
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Setting", "Value", "Unit", "Effect"],
        cellLoc="left",
        colWidths=[0.22, 0.11, 0.11, 0.56],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.PAD = 0.03
        if row == 0:
            cell.set_facecolor("#f3f1ee")
            cell.get_text().set_fontweight("semibold")
        elif column in (0, 1):
            cell.get_text().set_fontfamily("monospace")
        if row > 0 and column == 2:
            unit = rows[row - 1][2]
            cell.set_facecolor(AMBER_SOFT if unit == "degrees" else BLUE_SOFT)
            cell.get_text().set_color(AMBER if unit == "degrees" else BLUE)

    ax.set_title(
        "Augmentation settings — note that the units are not all the same",
        fontsize=11.5, color=INK, pad=14,
    )
    _save(fig, "augmentation_settings.png")


FIGURES = (
    figure_pipeline,
    figure_feature_maps,
    figure_architecture,
    figure_parameter_budget,
    figure_augmentation,
    figure_augmentation_settings,
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
