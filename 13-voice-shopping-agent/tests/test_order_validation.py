import pytest

from src.order_validation import validate_item_intent, validate_order_item


def test_accepts_a_minimal_item_and_trims_optional_details():
    assert validate_item_intent({"item": "  milk ", "details": " full cream "}) == {
        "item": "milk",
        "details": "full cream",
    }


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"item": ""},
        {"item": 12},
        {"item": "milk", "details": ["not", "text"]},
        {"item": "x" * 121},
    ],
)
def test_rejects_malformed_or_unsafe_item_intents(payload):
    with pytest.raises(ValueError):
        validate_item_intent(payload)


def test_accepts_a_bounded_order_quantity():
    assert validate_order_item({"item": "milk", "details": None, "quantity": 2}) == {
        "item": "milk",
        "details": None,
        "quantity": 2,
    }


@pytest.mark.parametrize("quantity", [0, -1, 1.5, True, 101, "2"])
def test_rejects_unsafe_order_quantities(quantity):
    with pytest.raises(ValueError):
        validate_order_item({"item": "milk", "details": None, "quantity": quantity})
