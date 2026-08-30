"""Leakage-aware feature engineering and evaluation for NIFTY price analysis."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error


FEATURE_COLUMNS = ["prev_close", "sma_5", "sma_20"]


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create next-day targets using only information known before that trading day."""
    frame = data.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.sort_values("Date").drop_duplicates("Date")
    frame["prev_close"] = frame["Close"].shift(1)
    frame["sma_5"] = frame["Close"].shift(1).rolling(5).mean()
    frame["sma_20"] = frame["Close"].shift(1).rolling(20).mean()
    frame["target_close"] = frame["Close"]
    return frame.dropna(subset=[*FEATURE_COLUMNS, "target_close"]).reset_index(drop=True)


def chronological_split(data: pd.DataFrame, *, test_fraction: float = 0.2):
    """Split ordered observations so all test dates occur after all training dates."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    cutoff = int(len(data) * (1 - test_fraction))
    return data.iloc[:cutoff].copy(), data.iloc[cutoff:].copy()


def evaluate_forecast(actual: Any, predicted: Any) -> dict[str, float]:
    """Report error in index points and percentage terms."""
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(mean_squared_error(actual, predicted, squared=False)),
        "mape": float(mean_absolute_percentage_error(actual, predicted) * 100),
    }
