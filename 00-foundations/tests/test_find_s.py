"""Tests for the Find-S concept learner.

Find-S has no scikit-learn counterpart, so these tests pin it against the
worked example in Mitchell's *Machine Learning* (1997), chapter 2, and against
the algorithm's documented limitations.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.datasets import load_enjoy_sport
from src.find_s import ANY, FindS


class TestFindSOnEnjoySport:
    """The canonical EnjoySport walkthrough."""

    def test_learns_the_textbook_hypothesis(self):
        X, y = load_enjoy_sport()
        model = FindS().fit(X, y)
        assert model.hypothesis_ == (
            "sunny",
            "warm",
            ANY,
            "strong",
            ANY,
            ANY,
        )

    def test_hypothesis_generalises_one_example_at_a_time(self):
        """Each step may only relax constraints, never tighten them."""
        X, y = load_enjoy_sport()
        steps = FindS().history(X, y)

        assert steps[0] == ("sunny", "warm", "normal", "strong", "warm", "same")
        assert steps[-1] == ("sunny", "warm", ANY, "strong", ANY, ANY)

        for earlier, later in zip(steps, steps[1:]):
            for before, after in zip(earlier, later):
                assert after == before or after == ANY

    def test_accepts_every_positive_training_example(self):
        """A correct Find-S hypothesis is consistent with all positives."""
        X, y = load_enjoy_sport()
        model = FindS().fit(X, y)
        predictions = model.predict(X)
        assert predictions[np.asarray(y) == "yes"].all()

    def test_rejects_the_negative_training_example(self):
        """Not guaranteed in general, but true for this dataset."""
        X, y = load_enjoy_sport()
        model = FindS().fit(X, y)
        predictions = model.predict(X)
        assert not predictions[np.asarray(y) == "no"].any()


class TestFindSBehaviour:
    def test_single_positive_example_stays_maximally_specific(self):
        X = [["sunny", "warm"]]
        model = FindS().fit(X, ["yes"])
        assert model.hypothesis_ == ("sunny", "warm")

    def test_conflicting_positives_generalise_to_all_any(self):
        X = [["sunny", "warm"], ["rainy", "cold"]]
        model = FindS().fit(X, ["yes", "yes"])
        assert model.hypothesis_ == (ANY, ANY)

    def test_negative_examples_are_ignored(self):
        """Find-S's defining weakness: negatives cannot change the result."""
        X_positives_only = [["sunny", "warm"], ["sunny", "cold"]]
        without = FindS().fit(X_positives_only, ["yes", "yes"]).hypothesis_

        X_with_negative = X_positives_only + [["rainy", "warm"]]
        with_negative = FindS().fit(X_with_negative, ["yes", "yes", "no"]).hypothesis_

        assert without == with_negative

    def test_custom_positive_label(self):
        X = [["a", "b"], ["a", "c"]]
        model = FindS(positive_label="Yes").fit(X, ["Yes", "Yes"])
        assert model.hypothesis_ == ("a", ANY)


class TestFindSErrors:
    def test_no_positive_examples_raises(self):
        with pytest.raises(ValueError, match="positive label"):
            FindS().fit([["sunny", "warm"]], ["no"])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="rows"):
            FindS().fit([["sunny"], ["rainy"]], ["yes"])

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            FindS().predict([["sunny", "warm"]])

    def test_history_with_no_positives_is_empty(self):
        assert FindS().history([["sunny"]], ["no"]) == []
