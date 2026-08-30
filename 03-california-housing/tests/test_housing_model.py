import numpy as np
import subprocess
import sys
from pathlib import Path
from sklearn.datasets import make_regression

from src.housing_model import build_pipeline, evaluate_predictions, split_data


def test_split_is_reproducible_and_preserves_all_rows():
    X, y = make_regression(n_samples=60, n_features=8, random_state=7)
    first = split_data(X, y, test_size=0.2, random_state=42)
    second = split_data(X, y, test_size=0.2, random_state=42)

    assert len(first[0]) + len(first[1]) == 60
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


def test_pipeline_fits_and_produces_one_prediction_per_test_row():
    X, y = make_regression(n_samples=80, n_features=8, noise=5, random_state=4)
    X_train, X_test, y_train, _ = split_data(X, y, test_size=0.25, random_state=4)
    pipeline = build_pipeline(random_state=4)
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    assert predictions.shape == (len(X_test),)
    assert np.isfinite(predictions).all()


def test_evaluate_predictions_reports_standard_regression_metrics():
    metrics = evaluate_predictions(np.array([2.0, 4.0]), np.array([1.0, 5.0]))

    assert set(metrics) == {"mae", "rmse", "r2"}
    assert metrics["mae"] == 1.0
    assert metrics["rmse"] == 1.0


def test_training_script_imports_from_the_project_root():
    """The documented direct command must resolve project-local imports."""
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import runpy; runpy.run_path('src/train.py', run_name='import_check')",
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
