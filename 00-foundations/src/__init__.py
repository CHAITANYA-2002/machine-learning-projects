"""From-scratch implementations of the classical machine-learning algorithms.

Every estimator in this package is written from first principles using only
NumPy. The accompanying test-suite asserts that each one agrees with the
equivalent scikit-learn estimator, so the implementations are verified rather
than merely asserted to be correct.
"""

from src.datasets import load_enjoy_sport, load_income, load_play_tennis
from src.find_s import FindS
from src.impurity import entropy, gini, information_gain
from src.knn import KNearestNeighbours
from src.naive_bayes import CategoricalNaiveBayes

__all__ = [
    "CategoricalNaiveBayes",
    "FindS",
    "KNearestNeighbours",
    "entropy",
    "gini",
    "information_gain",
    "load_enjoy_sport",
    "load_income",
    "load_play_tennis",
]
