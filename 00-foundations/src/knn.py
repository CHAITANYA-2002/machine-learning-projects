"""k-nearest-neighbours classification written from first principles.

KNN has no training step: ``fit`` only stores the data. All of the work happens
at prediction time, which is exactly why it is cheap to train and expensive to
serve. Keeping the distance computation visible here makes that cost obvious.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np


class KNearestNeighbours:
    """Majority-vote KNN classifier over a Minkowski distance.

    Note on ties: when the k-th and (k+1)-th nearest training points are
    exactly equidistant, the neighbour set is genuinely ambiguous and any
    choice among the tied points is equally valid. Two correct implementations
    can therefore return different predictions for such a query. This is common
    on coarsely-measured data — on Iris under Manhattan distance more than half
    of all queries have such a tie — and it is a property of the algorithm, not
    a defect. The test-suite compares against scikit-learn only on queries
    where the k-th neighbour is unambiguous.

    Args:
        n_neighbors: Number of neighbours polled for each prediction.
        p: Minkowski exponent. ``1`` is Manhattan distance, ``2`` is Euclidean.

    Attributes:
        X_: Training features retained for lookup, set by :meth:`fit`.
        y_: Training labels retained for voting, set by :meth:`fit`.
    """

    def __init__(self, n_neighbors: int = 5, p: int = 2) -> None:
        if n_neighbors < 1:
            raise ValueError(f"n_neighbors must be >= 1, got {n_neighbors}.")
        if p < 1:
            raise ValueError(f"p must be >= 1, got {p}.")
        self.n_neighbors = n_neighbors
        self.p = p
        self.X_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, X, y: Sequence) -> "KNearestNeighbours":
        """Memorise the training set.

        Args:
            X: Numeric feature matrix of shape ``(n_samples, n_features)``.
            y: Class labels of length ``n_samples``.

        Returns:
            self, so calls can be chained.

        Raises:
            ValueError: If the inputs disagree in length, or if fewer training
                samples are supplied than ``n_neighbors``.
        """
        features = np.asarray(X, dtype=float)
        labels = np.asarray(y)

        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"X has {features.shape[0]} rows but y has {labels.shape[0]}."
            )
        if features.shape[0] < self.n_neighbors:
            raise ValueError(
                f"n_neighbors={self.n_neighbors} exceeds the "
                f"{features.shape[0]} available training samples."
            )

        self.X_ = features
        self.y_ = labels
        return self

    def _distances(self, point: np.ndarray) -> np.ndarray:
        """Minkowski distance from one query point to every training sample."""
        return np.power(np.sum(np.abs(self.X_ - point) ** self.p, axis=1), 1.0 / self.p)

    def kneighbors(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Return distances to, and indices of, the nearest training samples.

        Args:
            X: Query points of shape ``(n_queries, n_features)``.

        Returns:
            A ``(distances, indices)`` pair, each of shape
            ``(n_queries, n_neighbors)``, sorted nearest-first.
        """
        self._check_fitted()
        queries = np.atleast_2d(np.asarray(X, dtype=float))

        all_distances = np.array([self._distances(point) for point in queries])
        # argpartition finds the k smallest in O(n); the subsequent argsort
        # then only orders those k, not the whole row.
        partitioned = np.argpartition(all_distances, self.n_neighbors - 1, axis=1)
        candidates = partitioned[:, : self.n_neighbors]
        rows = np.arange(len(queries))[:, None]
        order = np.argsort(all_distances[rows, candidates], axis=1)
        indices = candidates[rows, order]
        return all_distances[rows, indices], indices

    def predict(self, X) -> np.ndarray:
        """Predict a label for each query point by majority vote.

        Ties are broken in favour of the label seen first among the neighbours,
        which keeps predictions deterministic across runs.

        Args:
            X: Query points of shape ``(n_queries, n_features)``.

        Returns:
            Predicted labels of length ``n_queries``.
        """
        self._check_fitted()
        _, indices = self.kneighbors(X)
        return np.array([Counter(self.y_[row]).most_common(1)[0][0] for row in indices])

    def score(self, X, y: Sequence) -> float:
        """Return the mean accuracy of :meth:`predict` on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def _check_fitted(self) -> None:
        """Raise a clear error if the estimator has no stored training data."""
        if self.X_ is None or self.y_ is None:
            raise RuntimeError("Call fit() before using the estimator.")
