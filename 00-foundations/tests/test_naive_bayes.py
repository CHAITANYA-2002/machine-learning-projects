"""Tests for the from-scratch Categorical Naive Bayes classifier.

Agreement with ``sklearn.naive_bayes.CategoricalNB`` is checked on both
predictions and posterior probabilities, using matching smoothing settings.
The remaining tests cover the two details that are easy to get wrong: Laplace
smoothing and log-space arithmetic.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder

from src.datasets import load_play_tennis
from src.naive_bayes import CategoricalNaiveBayes


@pytest.fixture(scope="module")
def play_tennis():
    return load_play_tennis()


@pytest.fixture(scope="module")
def play_tennis_encoded(play_tennis):
    """Integer-encoded PlayTennis, as scikit-learn's CategoricalNB requires."""
    X, y = play_tennis
    encoder = OrdinalEncoder()
    return encoder.fit_transform(X), np.asarray(y)


class TestAgreementWithSklearn:
    @pytest.mark.parametrize("alpha", [1.0, 0.5, 2.0])
    def test_predictions_match_sklearn(self, play_tennis, play_tennis_encoded, alpha):
        X, y = play_tennis
        X_encoded, y_encoded = play_tennis_encoded

        ours = CategoricalNaiveBayes(alpha=alpha).fit(X, y)
        theirs = CategoricalNB(alpha=alpha).fit(X_encoded, y_encoded)

        np.testing.assert_array_equal(ours.predict(X), theirs.predict(X_encoded))

    @pytest.mark.parametrize("alpha", [1.0, 0.5])
    def test_posterior_probabilities_match_sklearn(
        self, play_tennis, play_tennis_encoded, alpha
    ):
        X, y = play_tennis
        X_encoded, y_encoded = play_tennis_encoded

        ours = CategoricalNaiveBayes(alpha=alpha).fit(X, y)
        theirs = CategoricalNB(alpha=alpha).fit(X_encoded, y_encoded)

        # Both estimators order their classes by sorted label, so the columns
        # line up without reordering.
        np.testing.assert_allclose(
            ours.predict_proba(X).to_numpy(),
            theirs.predict_proba(X_encoded),
            rtol=1e-9,
        )

    def test_class_log_priors_match_sklearn(self, play_tennis, play_tennis_encoded):
        X, y = play_tennis
        X_encoded, y_encoded = play_tennis_encoded

        ours = CategoricalNaiveBayes().fit(X, y)
        theirs = CategoricalNB().fit(X_encoded, y_encoded)

        our_priors = [ours.class_log_prior_[label] for label in ours.classes_]
        np.testing.assert_allclose(our_priors, theirs.class_log_prior_, rtol=1e-9)


class TestNaiveBayesBehaviour:
    def test_play_tennis_priors_are_nine_and_five_of_fourteen(self, play_tennis):
        X, y = play_tennis
        model = CategoricalNaiveBayes().fit(X, y)
        assert np.exp(model.class_log_prior_["Yes"]) == pytest.approx(9 / 14)
        assert np.exp(model.class_log_prior_["No"]) == pytest.approx(5 / 14)

    def test_probabilities_sum_to_one(self, play_tennis):
        X, y = play_tennis
        model = CategoricalNaiveBayes().fit(X, y)
        row_sums = model.predict_proba(X).to_numpy().sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-9)

    def test_smoothing_prevents_zero_probability(self):
        """The whole point of Laplace smoothing, in one test.

        Class 'b' never co-occurs with feature value 'x'. Without smoothing
        that combination is impossible; with smoothing it stays merely
        unlikely.
        """
        X = [["x"], ["x"], ["y"], ["y"]]
        y = ["a", "a", "b", "b"]

        smoothed = CategoricalNaiveBayes(alpha=1.0).fit(X, y)
        assert smoothed.predict_proba([["x"]]).loc[0, "b"] > 0

        unsmoothed = CategoricalNaiveBayes(alpha=0.0).fit(X, y)
        assert unsmoothed.predict_proba([["x"]]).loc[0, "b"] == pytest.approx(0.0)

    def test_unseen_feature_value_does_not_crash(self, play_tennis):
        """A value absent from training must still yield a prediction."""
        X, y = play_tennis
        model = CategoricalNaiveBayes(alpha=1.0).fit(X, y)
        prediction = model.predict([["Snowy", "Hot", "High", "Weak"]])
        assert prediction[0] in set(model.classes_)

    def test_log_space_avoids_underflow_on_wide_input(self):
        """400 features would underflow to zero if probabilities were multiplied."""
        n_features = 400
        X = [["a"] * n_features, ["b"] * n_features] * 5
        y = ["yes", "no"] * 5

        model = CategoricalNaiveBayes().fit(X, y)
        probabilities = model.predict_proba([["a"] * n_features])

        assert np.isfinite(probabilities.to_numpy()).all()
        assert probabilities.to_numpy().sum() == pytest.approx(1.0)

    def test_separable_data_is_classified_perfectly(self):
        X = [["a", "p"], ["a", "p"], ["b", "q"], ["b", "q"]]
        y = ["yes", "yes", "no", "no"]
        model = CategoricalNaiveBayes().fit(X, y)
        assert model.score(X, y) == 1.0


class TestNaiveBayesErrors:
    def test_negative_alpha_rejected(self):
        with pytest.raises(ValueError, match="alpha"):
            CategoricalNaiveBayes(alpha=-1.0)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="rows"):
            CategoricalNaiveBayes().fit([["a"], ["b"]], ["yes"])

    def test_empty_dataset_raises(self):
        with pytest.raises(ValueError, match="empty"):
            CategoricalNaiveBayes().fit(np.empty((0, 2), dtype=object), [])

    def test_one_dimensional_input_raises(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            CategoricalNaiveBayes().fit(np.array(["a", "b"], dtype=object), ["y", "n"])

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            CategoricalNaiveBayes().predict([["a"]])
