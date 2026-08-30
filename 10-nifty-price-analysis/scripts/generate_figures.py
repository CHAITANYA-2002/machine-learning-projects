"""Generate the data-backed figures embedded in the project documentation."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.nifty_model import FEATURE_COLUMNS, chronological_split, create_features
from src.train import data_path


ASSET_DIRECTORY = Path(__file__).resolve().parents[1] / "docs" / "assets"


def fitted_holdout() -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """Fit the documented chronological baseline and retain its holdout data."""
    featured = create_features(pd.read_csv(data_path()))
    train, test = chronological_split(featured)
    model = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
    model.fit(train[FEATURE_COLUMNS], train["target_close"])
    return train, test, model


def style_axis(axis: plt.Axes) -> None:
    """Remove non-informative chart borders while preserving a readable grid."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.25)


def main() -> None:
    """Write one scope chart and one honest chronological-holdout chart."""
    ASSET_DIRECTORY.mkdir(parents=True, exist_ok=True)
    train, test, model = fitted_holdout()
    all_rows = pd.concat([train, test], ignore_index=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.plot(all_rows["Date"], all_rows["target_close"], color="#28679e", linewidth=1.25, label="NIFTY close")
    axis.axvline(test["Date"].min(), color="#c76242", linestyle="--", linewidth=1.6, label="Chronological holdout begins")
    axis.set(title="NIFTY closing prices across the included historical period", xlabel="Date", ylabel="Index close")
    axis.legend(frameon=False, loc="upper left")
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(ASSET_DIRECTORY / "price_history_split.png", dpi=170, bbox_inches="tight")
    plt.close(figure)

    predictions = model.predict(test[FEATURE_COLUMNS])
    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.plot(test["Date"], test["target_close"], color="#1d5f91", linewidth=1.5, label="Actual close")
    axis.plot(test["Date"], predictions, color="#d06b45", linewidth=1.25, label="Linear baseline prediction")
    axis.set(title="Chronological holdout: actual and predicted closing prices", xlabel="Date", ylabel="Index close")
    axis.legend(frameon=False, loc="upper left")
    style_axis(axis)
    figure.tight_layout()
    figure.savefig(ASSET_DIRECTORY / "holdout_actual_vs_predicted.png", dpi=170, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
