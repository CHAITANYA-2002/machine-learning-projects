"""Categorical Naive Bayes written from first principles.

The "naive" assumption is that features are conditionally independent given the
class. That is almost never true — on PlayTennis, humidity and outlook are
plainly related — yet the classifier still works well, which is the point worth
understanding early.

Two implementation details matter and are made explicit below:

* **Laplace smoothing.** Without it, a single unseen ``(feature, value, class)``
  combination drives the whole product to zero and the class can never be
  predicted, however strong the other evidence.
* **Log-space arithmetic.** Multiplying many small probabilities underflows to
  zero in floating point, so the probabilities are summed as logs instead.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


class CategoricalNaiveBayes:
    """Naive Bayes classifier for nominal (categorical) features.

    Args:
        alpha: Laplace/Lidstone smoothing strength. ``1.0`` is add-one
            smoothing; ``0.0`` disables smoothing and permits zero
            probabilities.

    Attributes:
        classes_: Sorted array of the distinct labels seen during fit.
        class_log_prior_: Log prior probability of each class.
        feature_log_prob_: For each feature index, a mapping from
            ``(class, value)`` to a smoothed log conditional probability.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")
        self.alpha = alpha
        self.classes_: np.ndarray | None = None
        self.class_log_prior_: dict | None = None
        self.feature_log_prob_: list[dict] | None = None
        self._categories: list[np.ndarray] | None = None
        self._class_counts: dict | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence) -> "CategoricalNaiveBayes":
        """Estimate class priors and smoothed conditional probabilities.

        Args:
            X: Nominal feature matrix of shape ``(n_samples, n_features)``.
            y: Class labels of length ``n_samples``.

        Returns:
            self, so calls can be chained.

        Raises:
            ValueError: If ``X`` and ``y`` disagree in length or ``X`` is empty.
        """
        features = np.asarray(X, dtype=object)
        labels = np.asarray(y, dtype=object)

        if features.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {features.shape}.")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"X has {features.shape[0]} rows but y has {labels.shape[0]}."
            )
        if features.shape[0] == 0:
            raise ValueError("Cannot fit on an empty dataset.")

        self.classes_ = np.unique(labels)
        n_samples = labels.shape[0]

        # Priors: P(class), estimated as the observed class frequency.
        self._class_counts = {
            label: int((labels == label).sum()) for label in self.classes_
        }
        self.class_log_prior_ = {
            label: float(np.log(count / n_samples))
            for label, count in self._class_counts.items()
        }

        # The category vocabulary is fixed at fit time so that the smoothing
        # denominator is consistent and unseen values at predict time are
        # handled explicitly rather than silently.
        self._categories = [
            np.unique(features[:, column]) for column in range(features.shape[1])
        ]

        self.feature_log_prob_ = [
            self._conditional_log_probs(features[:, column], labels, column)
            for column in range(features.shape[1])
        ]
        return self

    def _conditional_log_probs(
        self, column_values: np.ndarray, labels: np.ndarray, column: int
    ) -> dict:
        """Return ``{(class, value): log P(value | class)}`` for one feature."""
        categories = self._categories[column]
        n_categories = len(categories)

        table = {}
        for label in self.classes_:
            in_class = column_values[labels == label]
            denominator = in_class.size + self.alpha * n_categories
            for value in categories:
                numerator = (in_class == value).sum() + self.alpha
                # A zero numerator is only reachable with alpha == 0, in which
                # case -inf is the mathematically correct log probability.
                table[(label, value)] = (
                    float(np.log(numerator / denominator))
                    if numerator > 0
                    else -np.inf
                )
        return table

    def _joint_log_likelihood(self, instance: Sequence) -> dict:
        """Return the unnormalised log posterior for each class."""
        scores = {}
        for label in self.classes_:
            total = self.class_log_prior_[label]
            for column, value in enumerate(instance):
                table = self.feature_log_prob_[column]
                if (label, value) in table:
                    total += table[(label, value)]
                elif self.alpha > 0:
                    # Value never seen in training. The smoothed estimate gives
                    # it the mass of one unobserved category, widening the
                    # denominator to admit that extra category.
                    n_categories = len(self._categories[column]) + 1
                    n_in_class = self._class_counts[label]
                    total += float(
                        np.log(self.alpha / (n_in_class + self.alpha * n_categories))
                    )
                else:
                    # Without smoothing an unseen value makes the class
                    # strictly impossible, which is the correct consequence.
                    total = -np.inf
                    break
            scores[label] = total
        return scores

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict the most probable class for each instance.

        Args:
            X: Nominal feature matrix with the column order used during fit.

        Returns:
            Predicted labels of length ``n_samples``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        self._check_fitted()
        instances = np.atleast_2d(np.asarray(X, dtype=object))

        predictions = []
        for row in instances:
            scores = self._joint_log_likelihood(row)
            predictions.append(max(scores, key=scores.__getitem__))
        return np.array(predictions, dtype=object)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Return normalised posterior probabilities for each class.

        The log scores are shifted by their maximum before exponentiating —
        the standard log-sum-exp trick — so that large negative log
        probabilities do not underflow to zero.

        Args:
            X: Nominal feature matrix with the column order used during fit.

        Returns:
            A DataFrame with one row per instance and one column per class.
        """
        self._check_fitted()
        instances = np.atleast_2d(np.asarray(X, dtype=object))

        rows = []
        for instance in instances:
            scores = self._joint_log_likelihood(instance)
            log_values = np.array([scores[label] for label in self.classes_])
            shifted = np.exp(log_values - log_values.max())
            rows.append(shifted / shifted.sum())

        return pd.DataFrame(rows, columns=self.classes_)

    def score(self, X, y: Sequence) -> float:
        """Return the mean accuracy of :meth:`predict` on the given data."""
        return float(np.mean(self.predict(X) == np.asarray(y, dtype=object)))

    def _check_fitted(self) -> None:
        """Raise a clear error if the estimator has not been fitted."""
        if self.classes_ is None:
            raise RuntimeError("Call fit() before using the estimator.")
