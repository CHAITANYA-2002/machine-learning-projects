# Movie catalogue data

`movie_catalog.csv` is the runnable local catalogue for this project. It contains three fields exported from the preserved upstream `movie_list.pkl` artefact, after removing 12 rows with ambiguous repeated titles:

| Field | Meaning |
| --- | --- |
| `movie_id` | TMDB numeric identifier used to distinguish movies with the same title and to request an optional poster. |
| `title` | Movie title displayed in the interface. |
| `tags` | Preprocessed text assembled from overview, genres, keywords, up to three cast members, and director names. |

The original notebook merges the separate TMDB movie and credit tables on `title`. Three repeated titles (`Batman`, `The Host`, and `Out of the Blue`) create 12 rows whose overview/metadata and cast/director features can be paired with the wrong film. The runnable CSV excludes all rows for those repeated titles, leaving **4,794 unambiguous records**. This is safer than presenting a plausible-looking but incorrect recommendation.

The CSV is committed because it is small enough to version and is needed to run the recommender. It contains no user data. The original notebook expects the separate TMDB 5000 Movies and Credits CSV files to reproduce this feature-construction step; those source files are not included in this repository. A future rebuild should merge the source tables on their numeric TMDB movie ID, then retain the previously excluded titles when their features can be joined correctly.

The application treats this catalogue as a static historical snapshot. It does not claim current TMDB metadata, availability, ratings, or personalisation.
