import pandas as pd

from src.fake_news_model import combine_text, validate_dataset


def test_combine_text_uses_title_and_body_and_handles_missing_values():
    frame = pd.DataFrame({"title": ["Headline", None], "text": ["Article body", "Body"]})
    combined = combine_text(frame)
    assert combined.tolist() == ["Headline Article body", "Body"]


def test_validate_dataset_requires_expected_columns_and_two_labels():
    valid = pd.DataFrame({"title": ["A", "B"], "text": ["one", "two"], "label": ["FAKE", "REAL"]})
    validate_dataset(valid)

    invalid = valid.drop(columns="text")
    try:
        validate_dataset(invalid)
    except ValueError as error:
        assert "text" in str(error)
    else:
        raise AssertionError("Expected validation failure")
