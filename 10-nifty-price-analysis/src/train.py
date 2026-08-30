"""Run the verified chronological NIFTY closing-price baseline."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.nifty_model import FEATURE_COLUMNS, chronological_split, create_features, evaluate_forecast


def data_path() -> Path:
    """Return the project-local CSV path regardless of the current shell directory."""
    return Path(__file__).resolve().parents[1] / "data" / "NSEI.csv"


def main() -> None:
    featured = create_features(pd.read_csv(data_path()))
    train, test = chronological_split(featured)
    model = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    model.fit(train[FEATURE_COLUMNS], train["target_close"])
    predictions = model.predict(test[FEATURE_COLUMNS])
    metrics = evaluate_forecast(test["target_close"], predictions)

    print("NIFTY closing-price baseline")
    print(f"Training dates: {train.Date.min():%Y-%m-%d} to {train.Date.max():%Y-%m-%d}")
    print(f"Test dates: {test.Date.min():%Y-%m-%d} to {test.Date.max():%Y-%m-%d}")
    print(f"MAE: {metrics['mae']:.2f} index points")
    print(f"RMSE: {metrics['rmse']:.2f} index points")
    print(f"MAPE: {metrics['mape']:.2f}%")


if __name__ == "__main__":
    main()
