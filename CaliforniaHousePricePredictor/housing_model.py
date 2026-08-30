"""Reproducible California housing regression workflow."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def split_data(
    features: Any, targets: Any, *, test_size: float = 0.2, random_state: int = 42
):
    """Create a repeatable holdout split before any preprocessing is fitted."""
    return train_test_split(features, targets, test_size=test_size, random_state=random_state)


def build_pipeline(*, random_state: int = 42) -> Pipeline:
    """Build a baseline that imputes, scales, and trains without holdout leakage."""
    regressor = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", regressor),
        ]
    )


def evaluate_predictions(actual: Any, predicted: Any) -> dict[str, float]:
    """Return complementary regression metrics in the target's native units."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted, squared=False)),
        "r2": float(r2_score(actual, predicted)),
    }


def feature_importance(pipeline: Pipeline, feature_names: list[str]) -> dict[str, float]:
    """Expose the fitted forest's impurity-based feature importances."""
    model = pipeline.named_steps["model"]
    values = np.asarray(model.feature_importances_)
    return dict(sorted(zip(feature_names, values), key=lambda item: item[1], reverse=True))
