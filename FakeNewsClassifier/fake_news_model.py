"""Data preparation and baseline model construction for news-label classification."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


REQUIRED_COLUMNS = {"title", "text", "label"}


def validate_dataset(data: pd.DataFrame) -> None:
    """Fail early when the expected source schema is unavailable."""
    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    if data["label"].nunique(dropna=True) < 2:
        raise ValueError("Dataset must contain at least two labels.")


def combine_text(data: pd.DataFrame) -> pd.Series:
    """Combine headline and body without losing words or propagating missing values."""
    title = data["title"].fillna("").astype(str).str.strip()
    body = data["text"].fillna("").astype(str).str.strip()
    return (title + " " + body).str.strip()


def build_pipeline() -> Pipeline:
    """Create a transparent, sparse-text baseline suitable for small news datasets."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.95)),
            ("model", LogisticRegression(max_iter=2_000, class_weight="balanced")),
        ]
    )
