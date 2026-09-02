"""Streamlit interface for the local content-based movie recommender."""

from __future__ import annotations

import os
from pathlib import Path
import sys

import requests
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from recommender import CatalogError, ContentRecommender, load_catalog  # noqa: E402


CATALOG_PATH = PROJECT_ROOT / "data" / "movie_catalog.csv"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{movie_id}"


@st.cache_resource
def get_recommender() -> ContentRecommender:
    """Load and vectorize the local catalogue once per Streamlit process."""

    return load_catalog(CATALOG_PATH)


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id: int, api_key: str | None) -> str | None:
    """Return a TMDB poster URL when the caller supplied a usable API key."""

    if not api_key:
        return None
    try:
        response = requests.get(
            TMDB_MOVIE_URL.format(movie_id=movie_id),
            params={"api_key": api_key, "language": "en-US"},
            timeout=10,
        )
        response.raise_for_status()
    except requests.RequestException:
        return None

    poster_path = response.json().get("poster_path")
    if not isinstance(poster_path, str) or not poster_path:
        return None
    return f"{TMDB_IMAGE_BASE_URL}{poster_path}"


def main() -> None:
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
    st.title("Content-Based Movie Recommender")
    st.caption(
        "Recommendations are based on local overview, genre, keyword, cast, and director tags."
    )

    try:
        recommender = get_recommender()
    except CatalogError as error:
        st.error(str(error))
        st.stop()

    labels = {
        movie_id: f"{title} (TMDB ID: {movie_id})"
        for movie_id, title in recommender.catalog[["movie_id", "title"]].itertuples(index=False)
    }
    selected_movie_id = st.selectbox(
        "Choose a film",
        options=list(labels),
        format_func=labels.__getitem__,
    )

    if not st.button("Show recommendations", type="primary"):
        return

    recommendations = recommender.recommend(selected_movie_id, limit=5)
    tmdb_api_key = os.getenv("TMDB_API_KEY")
    if not tmdb_api_key:
        st.info("Set TMDB_API_KEY to display optional poster images; recommendations work without it.")

    columns = st.columns(len(recommendations))
    for column, movie in zip(columns, recommendations.itertuples(index=False)):
        with column:
            st.subheader(movie.title)
            poster_url = fetch_poster(int(movie.movie_id), tmdb_api_key)
            if poster_url:
                st.image(poster_url, use_container_width=True)
            st.caption(f"Cosine similarity: {movie.score:.3f}")


if __name__ == "__main__":
    main()
