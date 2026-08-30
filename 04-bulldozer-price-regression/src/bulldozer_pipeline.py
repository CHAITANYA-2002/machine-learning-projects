"""Leakage-aware utilities for bulldozer auction-price regression.

The original notebook manually encoded categories in train and test data. That
approach can create different columns and category codes. This module keeps all
learned transformations in one scikit-learn pipeline instead.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder


def add_sale_date_features(frame: pd.DataFrame, *, date_column: str = "saledate") -> pd.DataFrame:
    """Return a copy with calendar features derived from a valid auction date.

    The raw date is dropped after derivation: tree models use the decomposed
    values directly, and retaining a datetime column would complicate numeric
    preprocessing. Copying avoids silently changing the caller's raw data.
    """
    if date_column not in frame:
        raise KeyError(f"Missing required date column: {date_column}")

    result = frame.copy()
    dates = pd.to_datetime(result[date_column], errors="raise")
    result["saleYear"] = dates.dt.year
    result["saleMonth"] = dates.dt.month
    result["saleDay"] = dates.dt.day
    result["saleDayOfWeek"] = dates.dt.dayofweek
    result["saleDayOfYear"] = dates.dt.dayofyear
    return result.drop(columns=[date_column])


def temporal_split(frame: pd.DataFrame, *, cutoff: str | pd.Timestamp, date_column: str = "saledate") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split rows into strictly earlier training data and later validation data."""
    if date_column not in frame:
        raise KeyError(f"Missing required date column: {date_column}")

    dates = pd.to_datetime(frame[date_column], errors="raise")
    boundary = pd.Timestamp(cutoff)
    train = frame.loc[dates < boundary].copy()
    validation = frame.loc[dates >= boundary].copy()
    if train.empty or validation.empty:
        raise ValueError("Temporal split produced an empty partition; check the cutoff and dates.")
    return train, validation


def rmsle(actual: Iterable[float], predicted: Iterable[float]) -> float:
    """Calculate RMSLE and reject values outside the metric's valid domain."""
    actual_array = np.asarray(list(actual), dtype=float)
    predicted_array = np.asarray(list(predicted), dtype=float)
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted must have the same shape")
    if actual_array.size == 0:
        raise ValueError("actual and predicted must not be empty")
    if np.any(actual_array < 0) or np.any(predicted_array < 0):
        raise ValueError("RMSLE requires non-negative actual and predicted values")
    return float(np.sqrt(np.mean((np.log1p(actual_array) - np.log1p(predicted_array)) ** 2)))


def _normalise_categorical_missing(values: pd.DataFrame) -> pd.DataFrame:
    """Convert Python ``None`` values to ``np.nan`` for scikit-learn's imputer."""
    return values.replace({None: np.nan})


def build_regression_pipeline(*, random_state: int = 42, n_estimators: int = 120) -> Pipeline:
    """Build a train-fitted pipeline safe for missing/unseen categorical values.

    Column names are selected by dtype when ``fit`` runs. Unknown-aware ordinal
    encoding maps a newly seen test category to a fixed sentinel instead of
    forcing fragile manual alignment or a large one-hot feature matrix.
    """
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True))])
    categorical_pipeline = Pipeline(
        [
            ("normalise_missing", FunctionTransformer(_normalise_categorical_missing, validate=False)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # Keep the high-cardinality competition data compact. New categories
            # become -1 rather than changing the feature matrix at prediction time.
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, lambda data: data.select_dtypes(include=np.number).columns),
            ("categorical", categorical_pipeline, lambda data: data.select_dtypes(exclude=np.number).columns),
        ],
        remainder="drop",
    )
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])
