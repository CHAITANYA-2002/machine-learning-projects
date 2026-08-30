"""Find-S: the maximally-specific concept-learning algorithm.

Find-S is the first algorithm in Tom Mitchell's *Machine Learning* (1997) and
is a useful starting point precisely because of what it cannot do. It searches
the hypothesis space for the most specific hypothesis consistent with the
positive examples, and it ignores negative examples entirely.

The implementation below is deliberately literal so the algorithm's behaviour —
and its well-known limitations — stay visible.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Sentinel attribute values used by the hypothesis representation.
NULL = "0"  # matches nothing; the maximally specific constraint
ANY = "?"  # matches anything; the maximally general constraint


class FindS:
    """Maximally-specific concept learner over nominal attributes.

    Attributes:
        hypothesis_: The learned hypothesis as a tuple of per-attribute
            constraints, available only after :meth:`fit`.
        positive_label: The target value treated as a positive example.
    """

    def __init__(self, positive_label: str = "yes") -> None:
        self.positive_label = positive_label
        self.hypothesis_: tuple[str, ...] | None = None

    def fit(self, X: pd.DataFrame | np.ndarray, y: Sequence[str]) -> "FindS":
        """Learn the most specific hypothesis consistent with positive examples.

        Args:
            X: Instances with one column per nominal attribute.
            y: Target labels; entries equal to ``positive_label`` are positive.

        Returns:
            self, so calls can be chained.

        Raises:
            ValueError: If ``X`` and ``y`` differ in length, or if the data
                contains no positive example, in which case Find-S has no
                defined output.
        """
        instances = np.asarray(X, dtype=object)
        labels = np.asarray(y, dtype=object)

        if len(instances) != len(labels):
            raise ValueError(
                f"X has {len(instances)} rows but y has {len(labels)} labels."
            )

        positives = instances[labels == self.positive_label]
        if len(positives) == 0:
            raise ValueError(
                f"No example has the positive label {self.positive_label!r}. "
                f"Find-S generalises only from positive examples, so the "
                f"hypothesis is undefined."
            )

        # Start maximally specific, then minimally generalise over each
        # positive example. Negative examples are never consulted -- that is
        # the algorithm's defining weakness, not an oversight here.
        hypothesis = tuple(positives[0])
        for example in positives[1:]:
            hypothesis = self._generalise(hypothesis, example)

        self.hypothesis_ = hypothesis
        return self

    @staticmethod
    def _generalise(
        hypothesis: tuple[str, ...], example: Sequence[str]
    ) -> tuple[str, ...]:
        """Return a new hypothesis relaxing each attribute that disagrees.

        A new tuple is returned rather than mutating in place so that the
        sequence of hypotheses can be inspected step by step.
        """
        return tuple(
            constraint if constraint == value else ANY
            for constraint, value in zip(hypothesis, example)
        )

    def history(self, X: pd.DataFrame | np.ndarray, y: Sequence[str]) -> list[tuple]:
        """Return the hypothesis after each positive example.

        Useful for teaching and for the accompanying notebook, where the point
        is to watch the hypothesis generalise one example at a time.
        """
        instances = np.asarray(X, dtype=object)
        labels = np.asarray(y, dtype=object)
        positives = instances[labels == self.positive_label]
        if len(positives) == 0:
            return []

        steps = [tuple(positives[0])]
        for example in positives[1:]:
            steps.append(self._generalise(steps[-1], example))
        return steps

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Classify instances by matching them against the learned hypothesis.

        Args:
            X: Instances with the same attribute order used during :meth:`fit`.

        Returns:
            A boolean array that is ``True`` where the instance satisfies every
            constraint in the hypothesis.

        Raises:
            RuntimeError: If called before :meth:`fit`.
        """
        if self.hypothesis_ is None:
            raise RuntimeError("Call fit() before predict().")

        instances = np.asarray(X, dtype=object)
        return np.array(
            [self._matches(self.hypothesis_, row) for row in instances], dtype=bool
        )

    @staticmethod
    def _matches(hypothesis: tuple[str, ...], instance: Sequence[str]) -> bool:
        """Return True when an instance satisfies every attribute constraint."""
        return all(
            constraint == ANY or constraint == value
            for constraint, value in zip(hypothesis, instance)
        )
