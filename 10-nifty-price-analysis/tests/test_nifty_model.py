import numpy as np
import pandas as pd

from src.nifty_model import create_features, chronological_split, evaluate_forecast


def _prices(rows=40):
    return pd.DataFrame({"Date": pd.date_range("2023-01-01", periods=rows, freq="D"), "Close": np.arange(rows, dtype=float) + 100})


def test_create_features_uses_only_prior_prices():
    featured = create_features(_prices())
    assert {"prev_close", "sma_5", "sma_20", "target_close"}.issubset(featured.columns)
    first = featured.iloc[0]
    assert first["prev_close"] < first["target_close"]


def test_chronological_split_never_uses_future_rows_for_training():
    featured = create_features(_prices(50))
    train, test = chronological_split(featured, test_fraction=0.2)
    assert train["Date"].max() < test["Date"].min()
    assert len(train) + len(test) == len(featured)


def test_evaluate_forecast_reports_price_metrics():
    metrics = evaluate_forecast(np.array([100.0, 102.0]), np.array([101.0, 101.0]))
    assert set(metrics) == {"mae", "rmse", "mape"}
    assert metrics["mae"] == 1.0
