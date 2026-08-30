"""Impurity measures and information gain for decision-tree induction.

These three functions are the entire arithmetic core of ID3 and CART. Writing
them out makes explicit what ``criterion="entropy"`` and ``criterion="gini"``
actually compute inside scikit-learn, and the test-suite checks these values
against the impurity scikit-learn reports at the root of a fitted tree.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _class_proportions(y: Sequence) -> np.ndarray:
    """Return the proportion of each distinct label in ``y``."""
    labels = np.asarray(y)
    if labels.size == 0:
        return np.array([])
    _, counts = np.unique(labels, return_counts=True)
    return counts / counts.sum()


def entropy(y: Sequence) -> float:
    """Shannon entropy of a label distribution, in bits.

    Args:
        y: A sequence of class labels of any hashable type.

    Returns:
        Entropy in bits. An empty or pure node has entropy ``0.0``; a balanced
        two-class node has entropy ``1.0``.
    """
    proportions = _class_proportions(y)
    if proportions.size == 0:
        return 0.0
    # Zero-probability classes contribute nothing; they are excluded rather
    # than relying on a 0*log(0) convention that would emit a warning.
    nonzero = proportions[proportions > 0]
    return float(-np.sum(nonzero * np.log2(nonzero)))


def gini(y: Sequence) -> float:
    """Gini impurity of a label distribution.

    Args:
        y: A sequence of class labels of any hashable type.

    Returns:
        Impurity in ``[0, 1)``. A pure node scores ``0.0``; a balanced
        two-class node scores ``0.5``.
    """
    proportions = _class_proportions(y)
    if proportions.size == 0:
        return 0.0
    return float(1.0 - np.sum(proportions**2))


def information_gain(
    y: Sequence, feature_values: Sequence, criterion: str = "entropy"
) -> float:
    """Reduction in impurity obtained by splitting ``y`` on ``feature_values``.

    Args:
        y: Class labels for every sample.
        feature_values: The value of one nominal feature for every sample,
            aligned with ``y``.
        criterion: Either ``"entropy"`` (ID3 information gain) or ``"gini"``
            (CART impurity decrease).

    Returns:
        The parent impurity minus the sample-weighted mean child impurity.
        Always non-negative for these two criteria.

    Raises:
        ValueError: If the inputs differ in length or the criterion is unknown.
    """
    labels = np.asarray(y)
    values = np.asarray(feature_values)

    if labels.shape[0] != values.shape[0]:
        raise ValueError(
            f"y has {labels.shape[0]} samples but feature_values has "
            f"{values.shape[0]}."
        )

    measures = {"entropy": entropy, "gini": gini}
    if criterion not in measures:
        raise ValueError(
            f"Unknown criterion {criterion!r}; expected one of {sorted(measures)}."
        )
    measure = measures[criterion]

    if labels.size == 0:
        return 0.0

    # Weight each child by the fraction of samples that fall into it, which is
    # what makes this a *gain* rather than a raw impurity comparison.
    parent_impurity = measure(labels)
    weighted_child_impurity = 0.0
    for value in np.unique(values):
        mask = values == value
        weighted_child_impurity += mask.sum() / labels.size * measure(labels[mask])

    return float(parent_impurity - weighted_child_impurity)


def best_split(X, y: Sequence, criterion: str = "entropy") -> tuple[str, float]:
    """Return the feature with the highest information gain, and that gain.

    This is the single decision ID3 makes at every node.

    Args:
        X: A DataFrame of nominal features.
        y: Class labels aligned with the rows of ``X``.
        criterion: ``"entropy"`` or ``"gini"``.

    Returns:
        A ``(feature_name, gain)`` pair.

    Raises:
        ValueError: If ``X`` has no columns.
    """
    if X.shape[1] == 0:
        raise ValueError("X must have at least one feature column.")

    gains = {
        column: information_gain(y, X[column], criterion=criterion)
        for column in X.columns
    }
    best = max(gains, key=gains.__getitem__)
    return best, gains[best]
