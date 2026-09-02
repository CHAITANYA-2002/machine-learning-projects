from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recommender import (  # noqa: E402
    CatalogError,
    ContentRecommender,
    load_catalog,
    validate_catalog,
)


@pytest.fixture
def catalog() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "movie_id": [10, 20, 30, 40, 50],
            "title": ["Orbit", "Orbit", "Garden", "Galaxy", "Kitchen"],
            "tags": [
                "space adventure starship",
                "space adventure astronaut",
                "plants garden nature",
                "space adventure starship galaxy",
                "cooking food kitchen",
            ],
        }
    )


def test_validate_catalog_requires_expected_columns() -> None:
    with pytest.raises(CatalogError, match="missing required columns"):
        validate_catalog(pd.DataFrame({"title": ["Orbit"]}))


def test_validate_catalog_deduplicates_exact_duplicate_records(catalog: pd.DataFrame) -> None:
    duplicate_rows = pd.concat([catalog, catalog.iloc[[0]]], ignore_index=True)

    validated = validate_catalog(duplicate_rows)

    assert len(validated) == len(catalog)
    assert validated["movie_id"].is_unique


def test_validate_catalog_rejects_conflicting_duplicate_movie_ids(catalog: pd.DataFrame) -> None:
    duplicate_ids = pd.concat([catalog, catalog.iloc[[0]]], ignore_index=True)
    duplicate_ids.loc[len(duplicate_ids) - 1, "title"] = "Orbit: Alternate Cut"

    with pytest.raises(CatalogError, match="unique"):
        validate_catalog(duplicate_ids)


@pytest.mark.parametrize("invalid_id", ["not-a-number", 10.5, 0, -1])
def test_validate_catalog_rejects_invalid_movie_ids(
    catalog: pd.DataFrame, invalid_id: object
) -> None:
    catalog["movie_id"] = catalog["movie_id"].astype(object)
    catalog.loc[0, "movie_id"] = invalid_id

    with pytest.raises(CatalogError, match="positive integers"):
        validate_catalog(catalog)


def test_recommendations_are_ranked_and_never_include_selected_movie(
    catalog: pd.DataFrame,
) -> None:
    recommender = ContentRecommender.from_catalog(catalog)

    recommendations = recommender.recommend(10, limit=3)

    assert recommendations["movie_id"].tolist() == [40, 20, 30]
    assert 10 not in recommendations["movie_id"].tolist()
    assert recommendations["score"].is_monotonic_decreasing


def test_recommendations_use_movie_id_to_disambiguate_duplicate_titles(
    catalog: pd.DataFrame,
) -> None:
    recommender = ContentRecommender.from_catalog(catalog)

    first_orbit = recommender.recommend(10, limit=1)
    second_orbit = recommender.recommend(20, limit=1)

    assert first_orbit.iloc[0]["movie_id"] == 40
    assert second_orbit.iloc[0]["movie_id"] == 10


def test_recommend_rejects_unknown_ids_and_invalid_limits(catalog: pd.DataFrame) -> None:
    recommender = ContentRecommender.from_catalog(catalog)

    with pytest.raises(KeyError, match="not found"):
        recommender.recommend(999, limit=1)
    with pytest.raises(ValueError, match="positive"):
        recommender.recommend(10, limit=0)
    with pytest.raises(ValueError, match="positive integer"):
        recommender.recommend(10, limit=1.5)


def test_empty_tag_corpus_is_rejected(catalog: pd.DataFrame) -> None:
    catalog["tags"] = ""

    with pytest.raises(CatalogError, match="no usable tags"):
        ContentRecommender.from_catalog(catalog)


def test_load_catalog_reads_csv_and_builds_recommender(
    catalog: pd.DataFrame, tmp_path: Path
) -> None:
    catalog_path = tmp_path / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    recommender = load_catalog(catalog_path)

    assert recommender.catalog["movie_id"].tolist() == [10, 20, 30, 40, 50]


def test_committed_catalog_supports_five_recommendations() -> None:
    recommender = load_catalog(PROJECT_ROOT / "data" / "movie_catalog.csv")

    recommendations = recommender.recommend(recommender.catalog.iloc[0]["movie_id"], limit=5)

    assert len(recommender.catalog) == 4794
    assert recommender.catalog["movie_id"].is_unique
    assert recommender.catalog["title"].is_unique
    assert len(recommendations) == 5
    assert recommendations["score"].is_monotonic_decreasing
