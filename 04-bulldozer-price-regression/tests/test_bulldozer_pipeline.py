import numpy as np
import pandas as pd
import pytest

from src.bulldozer_pipeline import (
    add_sale_date_features,
    build_regression_pipeline,
    rmsle,
    temporal_split,
)


def test_date_features_are_added_without_mutating_the_source_frame():
    source = pd.DataFrame({"saledate": ["2011-01-02", "2012-03-04"], "state": ["CA", "TX"]})

    transformed = add_sale_date_features(source)

    assert "saledate" in source.columns
    assert "saledate" not in transformed.columns
    assert transformed["saleYear"].tolist() == [2011, 2012]
    assert transformed["saleDayOfWeek"].tolist() == [6, 6]


def test_temporal_split_never_places_future_rows_in_training():
    frame = pd.DataFrame({"saledate": pd.to_datetime(["2011-12-31", "2012-01-01", "2012-05-01"]), "SalePrice": [1, 2, 3]})

    train, validation = temporal_split(frame, cutoff="2012-01-01")

    assert train["saledate"].max() < pd.Timestamp("2012-01-01")
    assert validation["saledate"].min() >= pd.Timestamp("2012-01-01")
    assert len(train) + len(validation) == len(frame)


def test_rmsle_matches_manual_calculation_and_rejects_negative_values():
    assert rmsle([10, 100], [20, 50]) == pytest.approx(np.sqrt(np.mean((np.log1p([10, 100]) - np.log1p([20, 50])) ** 2)))

    with pytest.raises(ValueError, match="non-negative"):
        rmsle([10], [-1])


def test_pipeline_handles_missing_and_unseen_categories_without_column_alignment():
    X_train = pd.DataFrame({"YearMade": [2000, 2005, np.nan, 2010], "state": ["CA", "TX", "CA", None], "UsageBand": ["Low", "High", None, "Medium"]})
    y_train = pd.Series([15000, 23000, 17000, 30000])
    X_future = pd.DataFrame({"YearMade": [2008], "state": ["NV"], "UsageBand": ["Unknown"]})

    pipeline = build_regression_pipeline(random_state=7, n_estimators=10)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_future)

    assert predictions.shape == (1,)
    assert np.isfinite(predictions).all()
