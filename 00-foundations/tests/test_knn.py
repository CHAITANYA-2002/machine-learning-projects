"""Tests for the from-scratch KNN classifier.

The substantive tests assert agreement with ``sklearn.neighbors``: identical
neighbour indices, identical distances, and identical predictions on the Iris
dataset. If our implementation drifts, these fail.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.knn import KNearestNeighbours


@pytest.fixture(scope="module")
def iris_split():
    """A fixed Iris train/test split shared by the comparison tests."""
    data = load_iris()
    return train_test_split(data.data, data.target, test_size=0.2, random_state=1)


def untied_queries(model, X_query) -> np.ndarray:
    """Return a mask of queries whose k-th neighbour is unambiguous.

    Iris measurements are recorded to one decimal place, so exact distance
    ties are common — under Manhattan distance more than half of the test
    queries have one. When the k-th and (k+1)-th neighbours are equidistant,
    *which* of them enters the neighbour set is arbitrary, and two correct
    implementations may legitimately disagree. Comparisons against
    scikit-learn are therefore restricted to the unambiguous queries.
    """
    queries = np.atleast_2d(np.asarray(X_query, dtype=float))
    k = model.n_neighbors

    mask = []
    for query in queries:
        distances = np.sort(model._distances(query))
        # No (k+1)-th neighbour exists, so no tie is possible.
        if distances.size <= k:
            mask.append(True)
        else:
            mask.append(not np.isclose(distances[k - 1], distances[k]))
    return np.array(mask, dtype=bool)


class TestAgreementWithSklearn:
    @pytest.mark.parametrize("k", [1, 3, 5, 10])
    def test_predictions_match_sklearn(self, iris_split, k):
        X_train, X_test, y_train, _ = iris_split

        ours = KNearestNeighbours(n_neighbors=k).fit(X_train, y_train)
        theirs = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

        unambiguous = untied_queries(ours, X_test)
        assert unambiguous.any(), "No unambiguous queries left to compare."
        np.testing.assert_array_equal(
            ours.predict(X_test)[unambiguous],
            theirs.predict(X_test)[unambiguous],
        )

    @pytest.mark.parametrize("k", [1, 3, 5])
    def test_neighbour_distances_match_sklearn(self, iris_split, k):
        X_train, X_test, y_train, _ = iris_split

        ours = KNearestNeighbours(n_neighbors=k).fit(X_train, y_train)
        theirs = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

        our_distances, _ = ours.kneighbors(X_test)
        their_distances, _ = theirs.kneighbors(X_test)
        np.testing.assert_allclose(our_distances, their_distances, rtol=1e-10)

    def test_manhattan_distance_matches_sklearn(self, iris_split):
        X_train, X_test, y_train, _ = iris_split

        ours = KNearestNeighbours(n_neighbors=5, p=1).fit(X_train, y_train)
        theirs = KNeighborsClassifier(n_neighbors=5, p=1).fit(X_train, y_train)

        unambiguous = untied_queries(ours, X_test)
        assert unambiguous.any(), "No unambiguous queries left to compare."
        np.testing.assert_array_equal(
            ours.predict(X_test)[unambiguous],
            theirs.predict(X_test)[unambiguous],
        )

    def test_accuracy_matches_sklearn(self, iris_split):
        X_train, X_test, y_train, y_test = iris_split

        ours = KNearestNeighbours(n_neighbors=10).fit(X_train, y_train)
        theirs = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)

        unambiguous = untied_queries(ours, X_test)
        assert ours.score(X_test[unambiguous], y_test[unambiguous]) == pytest.approx(
            theirs.score(X_test[unambiguous], y_test[unambiguous])
        )


class TestDistanceTies:
    """Ties are a real property of KNN, not an implementation defect."""

    def test_iris_manhattan_has_many_kth_neighbour_ties(self, iris_split):
        """Documents why the comparisons above filter tied queries."""
        X_train, X_test, y_train, _ = iris_split
        model = KNearestNeighbours(n_neighbors=5, p=1).fit(X_train, y_train)

        n_tied = (~untied_queries(model, X_test)).sum()
        assert n_tied > 0, "Expected Iris/Manhattan to produce distance ties."

    def test_tied_neighbours_share_the_kth_distance(self, iris_split):
        """Where we disagree with sklearn, the distances still agree exactly."""
        X_train, X_test, y_train, _ = iris_split

        ours = KNearestNeighbours(n_neighbors=5, p=1).fit(X_train, y_train)
        theirs = KNeighborsClassifier(n_neighbors=5, p=1).fit(X_train, y_train)

        our_distances, _ = ours.kneighbors(X_test)
        their_distances, _ = theirs.kneighbors(X_test)

        # The neighbour *sets* may differ under a tie; the sorted distance
        # profile must not.
        np.testing.assert_allclose(our_distances, their_distances, rtol=1e-10)


class TestKnnBehaviour:
    def test_k_equals_one_memorises_training_data(self):
        """With k=1 every training point is its own nearest neighbour."""
        X = [[0.0], [1.0], [2.0], [10.0]]
        y = ["a", "a", "b", "b"]
        model = KNearestNeighbours(n_neighbors=1).fit(X, y)
        assert model.score(X, y) == 1.0

    def test_neighbours_are_returned_nearest_first(self):
        X = [[0.0], [1.0], [2.0], [3.0]]
        model = KNearestNeighbours(n_neighbors=3).fit(X, ["a", "b", "c", "d"])
        distances, indices = model.kneighbors([[0.0]])
        np.testing.assert_array_equal(indices[0], [0, 1, 2])
        assert list(distances[0]) == sorted(distances[0])

    def test_majority_vote_beats_a_single_close_outlier(self):
        """Two far 'b' points outvote one near 'a' point when k=3."""
        X = [[0.0], [5.0], [6.0]]
        model = KNearestNeighbours(n_neighbors=3).fit(X, ["a", "b", "b"])
        assert model.predict([[1.0]])[0] == "b"

    def test_euclidean_distance_is_computed_correctly(self):
        """The 3-4-5 triangle, checked by hand."""
        model = KNearestNeighbours(n_neighbors=1).fit([[3.0, 4.0]], ["a"])
        distances, _ = model.kneighbors([[0.0, 0.0]])
        assert distances[0][0] == pytest.approx(5.0)


class TestKnnErrors:
    def test_zero_neighbours_rejected(self):
        with pytest.raises(ValueError, match="n_neighbors"):
            KNearestNeighbours(n_neighbors=0)

    def test_invalid_minkowski_exponent_rejected(self):
        with pytest.raises(ValueError, match="p must be"):
            KNearestNeighbours(p=0)

    def test_k_larger_than_training_set_rejected(self):
        with pytest.raises(ValueError, match="exceeds"):
            KNearestNeighbours(n_neighbors=5).fit([[0.0], [1.0]], ["a", "b"])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="rows"):
            KNearestNeighbours(n_neighbors=1).fit([[0.0], [1.0]], ["a"])

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            KNearestNeighbours().predict([[0.0]])
