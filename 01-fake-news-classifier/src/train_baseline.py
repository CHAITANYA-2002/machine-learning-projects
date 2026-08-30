"""Train a reproducible fake-news text baseline from a local CSV dataset."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.fake_news_model import build_pipeline, combine_text, validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/fake_or_real_news.csv"))
    args = parser.parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset not found: {args.data}. See README for the expected local layout.")

    data = pd.read_csv(args.data)
    validate_dataset(data)
    X = combine_text(data)
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipeline = build_pipeline().fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print(classification_report(y_test, predictions, zero_division=0))
    print(f"Weighted precision: {precision_score(y_test, predictions, average='weighted', zero_division=0):.3f}")
    print(f"Weighted recall: {recall_score(y_test, predictions, average='weighted', zero_division=0):.3f}")
    print(f"Weighted F1: {f1_score(y_test, predictions, average='weighted', zero_division=0):.3f}")


if __name__ == "__main__":
    main()
