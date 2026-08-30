"""Loaders for the three classical teaching datasets used in this project.

Each loader resolves its path relative to this file rather than the current
working directory, so notebooks, tests, and scripts all read the same data
regardless of where they are launched from.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _read(filename: str) -> pd.DataFrame:
    """Read a CSV from the project data directory with a clear error message."""
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Expected dataset at {path}. See data/README.md for provenance "
            f"and how to restore it."
        )
    return pd.read_csv(path)


def load_enjoy_sport() -> tuple[pd.DataFrame, pd.Series]:
    """Mitchell's EnjoySport dataset used to demonstrate concept learning.

    Returns:
        A ``(features, target)`` pair. The target is the ``enjoy sport``
        column with values ``"yes"``/``"no"``.
    """
    frame = _read("enjoy_sport.csv")
    return frame.drop(columns=["enjoy sport"]), frame["enjoy sport"]


def load_play_tennis() -> tuple[pd.DataFrame, pd.Series]:
    """Quinlan's PlayTennis dataset used to demonstrate decision-tree induction.

    The ``day`` column is an identifier, not a feature, so it is dropped here
    instead of leaving that trap for each caller to rediscover.

    Returns:
        A ``(features, target)`` pair. The target is the ``play`` column with
        values ``"Yes"``/``"No"``.
    """
    frame = _read("play_tennis.csv").drop(columns=["day"])
    return frame.drop(columns=["play"]), frame["play"]


def load_income() -> pd.DataFrame:
    """Age/income dataset used to demonstrate k-means clustering.

    Returns:
        The full frame including the ``Name`` column, which is retained for
        plot labelling but must not be used as a clustering feature.
    """
    return _read("income.csv")
