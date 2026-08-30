"""Tests for the text preprocessing.

`TestLegacyCleaning` pins :func:`legacy_char_clean` against strings copied from
the original notebook's saved output, so the 2022 pipeline stays reproducible
even though the source corpus is no longer available.
"""

from __future__ import annotations

import pytest

from src.preprocessing import (
    SINGLE_CHAR_STOPWORDS,
    clean_series,
    clean_text,
    compression_report,
    legacy_char_clean,
    single_character_stopwords,
    strip_non_letters,
)

# Copied verbatim from the saved output of cells 13 and 16 of
# notebooks/01_lstm_original_2022.ipynb.
NOTEBOOK_OUTPUT_PAIRS = [
    ("You Can Smell Hillary s Fear", "yu cn sell hllr  fer"),
    ("Kerry to go to Paris in gesture of sympathy", "kerr  g  pr n geure f ph"),
    (
        "The Battle of New York  Why This Primary Matters",
        "the ble f new yrk  wh th prr mer",
    ),
    ("Tehran  USA", "tehrn  usa"),
    ("Trump takes on Cruz  but lightly", "trup ke n cruz  bu lghl"),
    ("How women lose differently", "hw wen le fferenl"),
]


class TestLegacyCleaning:
    """The 2022 character-level cleaning, reproduced exactly."""

    @pytest.mark.parametrize("original,expected", NOTEBOOK_OUTPUT_PAIRS)
    def test_matches_saved_notebook_output(self, original, expected):
        assert legacy_char_clean(original) == expected

    def test_single_character_stopword_list(self):
        assert single_character_stopwords() == SINGLE_CHAR_STOPWORDS
        assert sorted(SINGLE_CHAR_STOPWORDS) == list("adimosty")

    @pytest.mark.parametrize("letter", sorted(SINGLE_CHAR_STOPWORDS))
    def test_each_stopword_character_is_removed(self, letter):
        assert legacy_char_clean(f"x{letter}x") == "xx"

    def test_other_characters_survive(self):
        assert legacy_char_clean("bzb") == "bzb"

    def test_operates_at_character_level(self):
        # 'cat' contains 'a' and 't', so only 'c' remains.
        assert legacy_char_clean("cat") == "c"

    def test_word_boundaries_are_preserved(self):
        result = legacy_char_clean("Tehran  USA")
        assert " " in result
        assert result == "tehrn  usa"


class TestCompressionReport:
    def test_reports_removed_character_count(self):
        report = compression_report("You Can Smell Hillary s Fear")
        assert report["compressed"] == "yu cn sell hllr  fer"
        assert report["characters_before"] == 28
        assert report["characters_after"] == 20
        assert report["characters_removed"] == 8

    def test_typical_compression_ratio(self):
        total_before = total_after = 0
        for original, _ in NOTEBOOK_OUTPUT_PAIRS:
            report = compression_report(original)
            total_before += report["characters_before"]
            total_after += report["characters_after"]

        fraction = 1 - total_after / total_before
        assert 0.15 < fraction < 0.40

    def test_empty_input_does_not_divide_by_zero(self):
        assert compression_report("")["fraction_removed"] == 0.0


class TestWordLevelCleaning:
    def test_lowercases_and_strips_punctuation(self):
        assert clean_text("You Can Smell Hillary's Fear") == "you can smell hillary s fear"

    def test_collapses_repeated_whitespace(self):
        assert clean_text("Tehran,   USA!!") == "tehran usa"

    def test_words_are_kept_intact(self):
        assert clean_text("cat") == "cat"
        assert clean_text("Hillary") == "hillary"

    def test_digits_become_separators(self):
        assert clean_text("Top 10 stories") == "top stories"

    def test_handles_none_and_missing_values(self):
        import pandas as pd

        series = pd.Series(["Hello World", None, "Bye"])
        assert clean_series(series).tolist() == ["hello world", "", "bye"]

    def test_strip_non_letters_leaves_spaces_behind(self):
        assert strip_non_letters("a-b") == "a b"


class TestStrategiesDiffer:
    def test_the_two_strategies_produce_different_output(self):
        text = "You Can Smell Hillary s Fear"
        assert legacy_char_clean(text) != clean_text(text)

    def test_word_level_cleaning_retains_more_letters(self):
        """Compare letters, not raw length: whitespace handling differs too."""
        for original, _ in NOTEBOOK_OUTPUT_PAIRS:
            word_level = sum(character.isalpha() for character in clean_text(original))
            char_level = sum(
                character.isalpha() for character in legacy_char_clean(original)
            )
            assert word_level > char_level
