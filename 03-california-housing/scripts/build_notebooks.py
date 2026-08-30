"""Generate the two explanatory notebooks without overwriting the legacy notebook.

The original combined notebook remains in ``notebooks/01_house_price_modeling.ipynb``.
This script writes clear EDA and modelling companions with deliberately empty outputs,
so readers execute each stage against their own locally fetched dataset.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"


def markdown(text: str) -> dict:
    """Create a Markdown cell; dedenting happens in the source literals below."""
    return {"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip().splitlines()]}


def code(text: str) -> dict:
    """Create an unexecuted code cell with comments explaining each decision."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def notebook(cells: Iterable[dict]) -> dict:
    """Return a standards-compliant Python 3 notebook document."""
    return {
        "cells": list(cells),
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


EDA_CELLS = [
    markdown(
        """
        # California Housing — Exploratory Data Analysis

        ## Purpose

        This notebook characterises the **scikit-learn California Housing** dataset before any model is trained. The target is the 1990 census-district median house value, stored in units of $100,000. It is a historical, area-level learning exercise—not a live market predictor or individual property appraisal.

        **Reader contract:** every table and plot is descriptive. A correlation, cluster, or feature pattern is not proof that one variable causes house value.
        """
    ),
    markdown(
        """
        ## Analysis map

        ```text
        fetch maintained source → verify shape and missingness → describe distributions
        → inspect correlations and geography → record modelling implications
        ```

        The dataset is fetched at runtime. This keeps the repository light and makes the data provenance explicit; it also means the first run needs network access or an existing scikit-learn cache.
        """
    ),
    code(
        """
        # Import only the tools needed for data inspection and static visuals.
        # A consistent theme makes plots legible without changing the evidence.
        from sklearn.datasets import fetch_california_housing
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="notebook")
        RANDOM_STATE = 42  # Kept for continuity with the modelling notebook.
        """
    ),
    code(
        """
        # Fetch the maintained dataset as labelled pandas objects. `as_frame=True`
        # preserves feature names, which prevents column-order mistakes in EDA.
        housing = fetch_california_housing(as_frame=True)
        X = housing.data.copy()
        y = housing.target.rename("MedHouseVal")
        housing_frame = pd.concat([X, y], axis=1)

        print(f"Rows: {len(housing_frame):,}")
        print(f"Predictors: {X.shape[1]}")
        print("Target unit: $100,000s")
        housing_frame.head()
        """
    ),
    markdown(
        """
        ## 1. Data-quality boundary

        Before reading relationships, confirm the basic contract: each row is a district, all eight predictors are numeric, and no feature values are missing in the fetched version. The target cap at **5.00001** is a semantic issue rather than a null-value issue—it removes variation at the expensive end and must remain visible in interpretation.
        """
    ),
    code(
        """
        # Count nulls column by column rather than assuming the public dataset is complete.
        # This result informs whether a model needs an imputation strategy.
        quality = pd.DataFrame({
            "dtype": housing_frame.dtypes.astype(str),
            "missing": housing_frame.isna().sum(),
            "missing_pct": housing_frame.isna().mean().mul(100).round(3),
            "unique_values": housing_frame.nunique(),
        })
        display(quality)
        print(f"Target maximum: {y.max():.5f} ($100,000s) — the documented cap.")
        """
    ),
    code(
        """
        # Summaries reveal scale and skew. We include 1st/99th percentiles because
        # population and occupancy contain extreme districts that a mean can hide.
        summary = housing_frame.describe(percentiles=[0.01, 0.5, 0.99]).T
        display(summary)
        """
    ),
    markdown(
        """
        ## 2. Target distribution and censoring

        A regression score only has meaning in the context of the target distribution. The histogram and the count at the cap make the dataset's high-end censoring visible. A model cannot recover distinctions the label itself does not contain.
        """
    ),
    code(
        """
        # The first panel shows the overall target distribution. The second counts
        # districts at the cap, a direct diagnostic for label censoring.
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
        sns.histplot(y, bins=45, kde=True, color="#5B3FD6", ax=axes[0])
        axes[0].set(title="District median house value", xlabel="Value ($100,000s)", ylabel="District count")

        cap = y.max()
        cap_count = (y == cap).sum()
        axes[1].bar(["below cap", "at cap"], [len(y) - cap_count, cap_count], color=["#247BA0", "#F4A261"])
        axes[1].set(title="Target censoring diagnostic", ylabel="District count")
        axes[1].text(1, cap_count, f"{cap_count:,}", ha="center", va="bottom", fontweight="bold")
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 3. Feature distributions, associations, and location

        The correlation heatmap is a compact screen for linear association, not a model and not a causal diagram. The scatter uses a sample to avoid painting more than 20,000 points on top of one another. Latitude/longitude is visualised separately because location can encode regional structure that a random row split may leak across the holdout boundary.
        """
    ),
    code(
        """
        # Spearman correlation is shown alongside Pearson-style visual intuition:
        # it is less sensitive to the very large occupancy and population values.
        correlation = housing_frame.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation, cmap="vlag", center=0, square=True, ax=ax)
        ax.set_title("Spearman rank correlations (descriptive only)")
        plt.show()
        """
    ),
    code(
        """
        # A fixed sample makes the plot reproducible while preserving readable density.
        # Income and location are selected because the baseline later relies on them;
        # seeing their structure prevents an importance chart from being over-read.
        plot_frame = housing_frame.sample(n=4_000, random_state=RANDOM_STATE)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        sns.scatterplot(data=plot_frame, x="MedInc", y="MedHouseVal", alpha=0.22, s=16, color="#5B3FD6", ax=axes[0])
        axes[0].set(title="Income and target", xlabel="Median income (tens of thousands)", ylabel="Value ($100,000s)")
        scatter = axes[1].scatter(plot_frame["Longitude"], plot_frame["Latitude"], c=plot_frame["MedHouseVal"], s=7, alpha=0.45, cmap="viridis")
        axes[1].set(title="Geographic distribution of the target", xlabel="Longitude", ylabel="Latitude")
        fig.colorbar(scatter, ax=axes[1], label="Value ($100,000s)")
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 4. EDA conclusions that constrain modelling

        1. The source has no missing features today, but a training-only imputer remains a safe future-input guard.
        2. Extreme population and occupancy values make robust summaries and residual checks more useful than a single average error.
        3. The target cap limits high-value interpretation.
        4. Coordinates contain strong regional structure, so a random split is only a first baseline. Spatially blocked validation is needed before claiming transfer to unseen areas.

Continue with [`02_model_development.ipynb`](02_model_development.ipynb) for the leakage-safe baseline.
        """
    ),
]


MODEL_CELLS = [
    markdown(
        """
        # California Housing — Model Development and Evaluation

        ## Objective

        Train one reproducible random-forest baseline for the historical California Housing target. The goal is a transparent benchmark with an untouched test set—not a production valuation system.

        **Evaluation promise:** split first; fit every learned transformation on training rows only; generate one prediction for each holdout row; interpret several complementary metrics together.
        """
    ),
    markdown(
        """
        ## Why this pipeline

        ```text
        source rows → fixed split → [fit imputer → fit scaler → fit forest] on train
                    → transform/predict test with training state → metrics and diagnostics
        ```

        A `Pipeline` makes this order executable. It prevents a later edit from accidentally fitting preprocessing statistics on the holdout set.
        """
    ),
    code(
        """
        # Use the maintained loader and the small, test-covered project helpers.
        # The repository root must be the current directory when running this notebook;
        # use `jupyter nbconvert --execute notebooks/02_model_development.ipynb`
        # from the project root for a reproducible command-line run.
        from sklearn.datasets import fetch_california_housing
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns

        from src.housing_model import build_pipeline, evaluate_predictions, feature_importance, split_data

        sns.set_theme(style="whitegrid", context="notebook")
        RANDOM_STATE = 42
        """
    ),
    code(
        """
        # Fetch named columns and preserve the target's unit ($100,000s).
        # We do not commit an opaque data copy; scikit-learn caches the public source.
        housing = fetch_california_housing(as_frame=True)
        X = housing.data
        y = housing.target
        print(f"Source rows: {len(X):,} | Feature columns: {list(X.columns)}")
        """
    ),
    markdown(
        """
        ## 1. Reserve the holdout before fitting anything

        The test set is a simulation of future unseen districts. It cannot influence imputation medians, scaling parameters, tree splits, hyperparameter choice, or final reported metrics. `random_state=42` makes this baseline repeatable, but does **not** prove geographic generalisation.
        """
    ),
    code(
        """
        # Split raw rows first. The helper is unit tested for reproducibility and
        # conservation of all rows, which protects this critical evaluation boundary.
        X_train, X_test, y_train, y_test = split_data(
            X, y, test_size=0.20, random_state=RANDOM_STATE
        )
        print(f"Training rows: {len(X_train):,}")
        print(f"Untouched test rows: {len(X_test):,}")
        """
    ),
    markdown(
        """
        ## 2. Fit the baseline

        The pipeline uses median imputation, standard scaling, and a 300-tree random forest with at least two rows per leaf. Trees do not require scaling; the scaler is retained inside the pipeline so the preprocessing contract is reusable for a future linear or distance-based comparison. It is not the reason the forest works.
        """
    ),
    code(
        """
        # `.fit` is called only on training data. The pipeline forwards each stage
        # in order and stores the learned state needed later to transform test rows.
        pipeline = build_pipeline(random_state=RANDOM_STATE)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        # A shape check is a compact guard against accidental row loss or broadcasting.
        assert predictions.shape == (len(X_test),)
        """
    ),
    markdown(
        """
        ## 3. Measure complementary error views

        - **MAE**: average absolute miss in target units; easiest to translate to dollars.
        - **RMSE**: penalises larger misses more heavily, so it is sensitive to tails.
        - **R²**: fraction of holdout target variance captured relative to a constant baseline.

        None of these says that every geography or price range has the same error.
        """
    ),
    code(
        """
        # Keep evaluation in the target's native unit, then add a dollar translation
        # only as an aid to interpretation. The conversion does not change the metric.
        metrics = evaluate_predictions(y_test, predictions)
        metric_table = pd.DataFrame([metrics], index=["Random forest baseline"])
        display(metric_table.round(3))
        print(f"Typical MAE in dollars: approximately ${metrics['mae'] * 100_000:,.0f}")
        """
    ),
    code(
        """
        # Diagnostics reveal patterns a scalar score hides. The diagonal is perfect
        # prediction; residuals above zero mean the model predicted too low.
        residuals = y_test - predictions
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.7))
        axes[0].scatter(y_test, predictions, alpha=0.28, s=12, color="#247BA0")
        limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
        axes[0].plot(limits, limits, "--", color="#E45756", label="perfect prediction")
        axes[0].set(title="Holdout predictions versus actual", xlabel="Actual ($100,000s)", ylabel="Predicted ($100,000s)")
        axes[0].legend()
        sns.histplot(residuals, bins=38, kde=True, color="#5B3FD6", ax=axes[1])
        axes[1].axvline(0, color="#152033", linestyle="--")
        axes[1].set(title="Holdout residual distribution", xlabel="Actual − predicted ($100,000s)")
        fig.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 4. Inspect feature reliance without claiming cause

        Random-forest impurity importance describes how useful a feature was for splits in this fitted model. It is affected by correlated features and split opportunities. It does **not** mean changing median income, latitude, or any other column would mechanically change house value.
        """
    ),
    code(
        """
        # The helper sorts the fitted forest's importances descending and is tested
        # independently. Plot all eight values to avoid hiding lower-ranked features.
        importance = pd.Series(feature_importance(pipeline, list(X.columns)))
        fig, ax = plt.subplots(figsize=(9, 4.8))
        importance.sort_values().plot.barh(ax=ax, color="#5B3FD6")
        ax.set(title="Impurity-based feature reliance", xlabel="Importance")
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 5. Decision record and next experiment

        This baseline supports a reproducible 1990 district-level benchmark. It does not support real-estate appraisal, current pricing, causal interpretation, or geographic-transfer claims.

        The most valuable next experiment is **spatially blocked cross-validation**: form geographic groups first, train on some regions, and test on withheld regions. Pair that with permutation importance and error slices by location and target range before trusting any deployment-oriented conclusion.
        """
    ),
]


def write_notebook(filename: str, cells: list[dict]) -> None:
    """Write one generated companion notebook in a stable, readable JSON format."""
    path = NOTEBOOK_DIR / filename
    path.write_text(json.dumps(notebook(cells), indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {path.name}")


if __name__ == "__main__":
    write_notebook("01_exploratory_data_analysis.ipynb", EDA_CELLS)
    write_notebook("02_model_development.ipynb", MODEL_CELLS)
