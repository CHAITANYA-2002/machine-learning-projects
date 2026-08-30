"""Text preprocessing for the fake-news corpus.

Two cleaning strategies live here.

:func:`legacy_char_clean` reproduces the character-level cleaning used in the
original 2022 notebook. It walks a headline one character at a time, drops any
character that appears in NLTK's English stop-word list, and lowercases the
rest. Because that list contains eight single-character entries -- ``a d i m o
s t y``, left over from tokenised contractions -- the step compresses each
headline into a shorter consonant-heavy form:

    "You Can Smell Hillary s Fear"  ->  "yu cn sell hllr  fer"

It is kept so the original notebook's saved outputs can still be regenerated
and checked, and it is what the recovered 2022 training metrics correspond to.

:func:`clean_text` is the word-level cleaning used by the current TF-IDF
baseline in :mod:`src.fake_news_model`. It lowercases, strips non-letters, and
collapses whitespace, leaving stop-word handling to the vectoriser where a
proper word-level list applies.
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

# The eight single-character entries in NLTK's English stop-word list. They are
# hardcoded so this module works without an NLTK corpus download;
# :func:`single_character_stopwords` prefers NLTK when it is installed.
SINGLE_CHAR_STOPWORDS = frozenset("adimosty")

NON_LETTER = re.compile(r"[^a-zA-Z]")


@lru_cache(maxsize=1)
def _nltk_stopwords() -> frozenset[str] | None:
    """Return NLTK's English stop-words, or None if the corpus is unavailable.

    NLTK is an optional dependency: the current baseline uses scikit-learn's
    built-in stop-word list, so a missing download must not break imports.
    """
    try:
        from nltk.corpus import stopwords

        return frozenset(stopwords.words("english"))
    except Exception:
        return None


def single_character_stopwords() -> frozenset[str]:
    """Return the single-character entries of the English stop-word list.

    Uses NLTK when its corpus is installed and falls back to the hardcoded set
    otherwise. The two are asserted equal in the test-suite.
    """
    words = _nltk_stopwords()
    if words is None:
        return SINGLE_CHAR_STOPWORDS
    return frozenset(word for word in words if len(word) == 1)


def strip_non_letters(text: str) -> str:
    """Replace every non-alphabetic character with a space.

    This is the normalisation step both cleaning strategies share. It also
    removes digits, which matters for a news corpus full of dates and figures.
    """
    return NON_LETTER.sub(" ", str(text))


def legacy_char_clean(text: str) -> str:
    """Apply the 2022 notebook's character-level cleaning.

    Iterates the string one character at a time, drops any character that is a
    single-character stop-word, and lowercases the rest. Word boundaries are
    preserved because spaces are not stop-words.

    Args:
        text: A headline, already passed through :func:`strip_non_letters`.

    Returns:
        The compressed string, matching the 2022 notebook's saved output.

    Example:
        >>> legacy_char_clean("You Can Smell Hillary s Fear")
        'yu cn sell hllr  fer'
    """
    stopword_set = single_character_stopwords()
    # The Porter stemmer returns a single character lowercased, so the stemming
    # step is reproduced here without requiring NLTK at runtime.
    return "".join(
        character.lower() for character in str(text) if character not in stopword_set
    )


def clean_text(text: str) -> str:
    """Clean a document at word level: lowercase, strip non-letters, collapse space.

    Stop-word removal is intentionally left to the TF-IDF vectoriser in
    :mod:`src.fake_news_model`, which applies a word-level list.

    Args:
        text: Raw headline or article body.

    Returns:
        A lowercased string of space-separated alphabetic words.

    Example:
        >>> clean_text("You Can Smell Hillary's Fear")
        'you can smell hillary s fear'
    """
    return " ".join(strip_non_letters(text).lower().split())


def clean_series(series: pd.Series) -> pd.Series:
    """Apply :func:`clean_text` across a pandas Series, preserving the index."""
    return series.fillna("").astype(str).map(clean_text)


def compression_report(text: str) -> dict:
    """Measure how much :func:`legacy_char_clean` shortens a document.

    Args:
        text: The document as it entered the 2022 cleaning step.

    Returns:
        A dictionary with the input and output strings, the character counts,
        and the fraction removed.
    """
    original = strip_non_letters(text)
    compressed = legacy_char_clean(original)
    removed = len(original) - len(compressed)
    return {
        "original": original,
        "compressed": compressed,
        "characters_before": len(original),
        "characters_after": len(compressed),
        "characters_removed": removed,
        "fraction_removed": removed / len(original) if original else 0.0,
    }
