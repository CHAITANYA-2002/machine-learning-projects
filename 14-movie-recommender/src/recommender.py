"""Content-based movie recommendations built from a local catalogue."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


REQUIRED_COLUMNS = ("movie_id", "title", "tags")


class CatalogError(ValueError):
    """Raised when the local catalogue cannot support recommendations."""


def validate_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Return a normalized catalogue after enforcing its minimal contract."""

    if not isinstance(catalog, pd.DataFrame):
        raise CatalogError("catalog must be a pandas DataFrame")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in catalog]
    if missing_columns:
        raise CatalogError(
            "catalog is missing required columns: " + ", ".join(missing_columns)
        )

    cleaned = catalog.loc[:, REQUIRED_COLUMNS].copy().drop_duplicates().reset_index(drop=True)
    if cleaned.empty:
        raise CatalogError("catalog must contain at least one movie")
    numeric_ids = pd.to_numeric(cleaned["movie_id"], errors="coerce")
    valid_ids = (
        numeric_ids.notna()
        & np.isfinite(numeric_ids)
        & numeric_ids.gt(0)
        & numeric_ids.mod(1).eq(0)
    )
    if not valid_ids.all():
        raise CatalogError("catalog movie_id values must be positive integers")
    cleaned["movie_id"] = numeric_ids.astype("int64")
    if not cleaned["movie_id"].is_unique:
        raise CatalogError("catalog movie_id values must be unique")

    cleaned["title"] = cleaned["title"].fillna("").astype(str).str.strip()
    if (cleaned["title"] == "").any():
        raise CatalogError("catalog titles must be non-empty")

    cleaned["tags"] = cleaned["tags"].fillna("").astype(str).str.strip()
    if not cleaned["tags"].str.len().gt(0).any():
        raise CatalogError("catalog has no usable tags")

    return cleaned


@dataclass(frozen=True)
class ContentRecommender:
    """A sparse-vector content recommender keyed by stable movie IDs."""

    catalog: pd.DataFrame
    tag_matrix: object

    @classmethod
    def from_catalog(cls, catalog: pd.DataFrame) -> "ContentRecommender":
        """Validate a catalogue and fit the same count-vector representation as the notebook."""

        cleaned = validate_catalog(catalog)
        vectorizer = CountVectorizer(max_features=5000, stop_words="english")
        try:
            tag_matrix = vectorizer.fit_transform(cleaned["tags"])
        except ValueError as error:
            raise CatalogError("catalog has no usable tags") from error
        return cls(catalog=cleaned, tag_matrix=tag_matrix)

    def recommend(self, movie_id: object, limit: int = 5) -> pd.DataFrame:
        """Return the highest cosine-similarity recommendations for one catalogue item."""

        if not isinstance(limit, Integral) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        limit = int(limit)

        selected_rows = self.catalog.index[self.catalog["movie_id"] == movie_id]
        if selected_rows.empty:
            raise KeyError(f"movie_id {movie_id!r} was not found in the catalog")

        selected_index = int(selected_rows[0])
        scores = cosine_similarity(self.tag_matrix[selected_index], self.tag_matrix).ravel()
        ranked_indices = np.argsort(-scores, kind="stable")
        recommendation_indices = [
            int(index) for index in ranked_indices if int(index) != selected_index
        ][:limit]

        recommendations = self.catalog.iloc[recommendation_indices][["movie_id", "title"]].copy()
        recommendations["score"] = scores[recommendation_indices]
        return recommendations.reset_index(drop=True)


def load_catalog(path: str | Path) -> ContentRecommender:
    """Load the project CSV catalogue and return a ready-to-query recommender."""

    catalog_path = Path(path)
    if not catalog_path.is_file():
        raise CatalogError(f"catalog file was not found: {catalog_path}")
    if catalog_path.suffix.lower() != ".csv":
        raise CatalogError("catalog must be a CSV file")
    return ContentRecommender.from_catalog(pd.read_csv(catalog_path))
