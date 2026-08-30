"""Tests for the impurity measures.

The hand-worked values below come from Quinlan's PlayTennis example, which is
the standard reference for ID3 information gain. Where possible each value is
also cross-checked against scikit-learn so the tests fail if the arithmetic
drifts away from the library's definition.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.datasets import load_play_tennis
from src.impurity import best_split, entropy, gini, information_gain


class TestEntropy:
    def test_pure_node_has_zero_entropy(self):
        assert entropy(["yes"] * 10) == 0.0

    def test_balanced_binary_node_has_one_bit(self):
        assert entropy(["yes"] * 5 + ["no"] * 5) == pytest.approx(1.0)

    def test_balanced_four_class_node_has_two_bits(self):
        assert entropy(["a", "b", "c", "d"]) == pytest.approx(2.0)

    def test_empty_input_is_zero(self):
        assert entropy([]) == 0.0

    def test_play_tennis_root_entropy(self):
        """PlayTennis is 9 Yes / 5 No, giving the textbook value 0.940 bits."""
        _, y = load_play_tennis()
        assert entropy(y) == pytest.approx(0.9402859586706311)

    def test_entropy_is_invariant_to_label_names(self):
        assert entropy([0, 0, 1]) == pytest.approx(entropy(["x", "x", "y"]))


class TestGini:
    def test_pure_node_has_zero_impurity(self):
        assert gini(["yes"] * 10) == 0.0

    def test_balanced_binary_node_is_one_half(self):
        assert gini(["yes"] * 5 + ["no"] * 5) == pytest.approx(0.5)

    def test_empty_input_is_zero(self):
        assert gini([]) == 0.0

    def test_play_tennis_root_gini(self):
        """9/14 Yes and 5/14 No gives 1 - (9/14)^2 - (5/14)^2."""
        _, y = load_play_tennis()
        expected = 1 - (9 / 14) ** 2 - (5 / 14) ** 2
        assert gini(y) == pytest.approx(expected)

    def test_gini_never_exceeds_entropy_for_binary_labels(self):
        for n_yes in range(1, 10):
            labels = ["yes"] * n_yes + ["no"] * (10 - n_yes)
            assert gini(labels) <= entropy(labels)


class TestAgreementWithSklearn:
    """Our impurity must equal the impurity scikit-learn reports at the root."""

    @pytest.mark.parametrize("criterion", ["entropy", "gini"])
    def test_root_impurity_matches_sklearn(self, criterion):
        X, y = load_play_tennis()
        X_encoded = X.apply(lambda column: column.astype("category").cat.codes)

        tree = DecisionTreeClassifier(criterion=criterion, random_state=0)
        tree.fit(X_encoded, y)
        sklearn_root_impurity = tree.tree_.impurity[0]

        ours = entropy(y) if criterion == "entropy" else gini(y)
        assert ours == pytest.approx(sklearn_root_impurity)


class TestInformationGain:
    def test_perfect_split_recovers_full_parent_entropy(self):
        y = ["yes", "yes", "no", "no"]
        feature = ["a", "a", "b", "b"]
        assert information_gain(y, feature) == pytest.approx(entropy(y))

    def test_useless_split_yields_zero_gain(self):
        y = ["yes", "no", "yes", "no"]
        feature = ["a", "a", "a", "a"]
        assert information_gain(y, feature) == pytest.approx(0.0)

    def test_play_tennis_outlook_gain(self):
        """Outlook is the textbook winner at the root with gain 0.247 bits."""
        X, y = load_play_tennis()
        assert information_gain(y, X["outlook"]) == pytest.approx(0.2467, abs=1e-4)

    def test_play_tennis_all_textbook_gains(self):
        """Every root gain from Mitchell's worked PlayTennis example."""
        X, y = load_play_tennis()
        expected = {
            "outlook": 0.2467,
            "temp": 0.0292,
            "humidity": 0.1518,
            "wind": 0.0481,
        }
        for feature, gain in expected.items():
            assert information_gain(y, X[feature]) == pytest.approx(gain, abs=1e-4)

    def test_gain_is_never_negative(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            y = rng.choice(["yes", "no"], size=30)
            feature = rng.choice(list("abc"), size=30)
            assert information_gain(y, feature) >= -1e-12

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="samples"):
            information_gain(["yes", "no"], ["a"])

    def test_unknown_criterion_raises(self):
        with pytest.raises(ValueError, match="Unknown criterion"):
            information_gain(["yes", "no"], ["a", "b"], criterion="chi2")


class TestBestSplit:
    def test_outlook_wins_at_the_play_tennis_root(self):
        """ID3 chooses Outlook first; this is the algorithm's headline result."""
        X, y = load_play_tennis()
        feature, gain = best_split(X, y)
        assert feature == "outlook"
        assert gain == pytest.approx(0.2467, abs=1e-4)

    def test_best_split_matches_sklearn_root_feature(self):
        """scikit-learn's tree must split on the same feature we choose."""
        X, y = load_play_tennis()
        X_encoded = X.apply(lambda column: column.astype("category").cat.codes)

        tree = DecisionTreeClassifier(criterion="entropy", random_state=0)
        tree.fit(X_encoded, y)
        sklearn_feature = X.columns[tree.tree_.feature[0]]

        ours, _ = best_split(X, y)
        assert ours == sklearn_feature

    def test_empty_feature_set_raises(self):
        X, y = load_play_tennis()
        with pytest.raises(ValueError, match="at least one feature"):
            best_split(X[[]], y)
