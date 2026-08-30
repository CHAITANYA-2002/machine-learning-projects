"""Train and evaluate the California House Price Predictor from the command line."""

from sklearn.datasets import fetch_california_housing

from housing_model import build_pipeline, evaluate_predictions, feature_importance, split_data


RANDOM_STATE = 42


def main() -> None:
    # The dataset is fetched by scikit-learn on the first run and cached locally.
    housing = fetch_california_housing(as_frame=True)
    X_train, X_test, y_train, y_test = split_data(
        housing.data, housing.target, test_size=0.2, random_state=RANDOM_STATE
    )

    pipeline = build_pipeline(random_state=RANDOM_STATE)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = evaluate_predictions(y_test, predictions)

    print("California House Price Predictor")
    print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print(f"MAE:  {metrics['mae']:.3f} ($100,000s)")
    print(f"RMSE: {metrics['rmse']:.3f} ($100,000s)")
    print(f"R2:   {metrics['r2']:.3f}")
    print("\nTop feature importances:")
    for feature, importance in list(feature_importance(pipeline, list(X_train.columns)).items())[:5]:
        print(f"  {feature}: {importance:.3f}")


if __name__ == "__main__":
    main()
