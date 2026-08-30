"""Small, dependency-free validation boundary for model-produced cart intent."""

from __future__ import annotations

from typing import Any

MAX_TEXT_LENGTH = 120
MAX_QUANTITY = 100


def _clean_text(value: Any, field: str, *, allow_none: bool = False) -> str | None:
    """Return bounded text or reject values that are unsafe to automate."""
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError(f"{field} cannot be empty")
    if len(cleaned) > MAX_TEXT_LENGTH:
        raise ValueError(f"{field} exceeds {MAX_TEXT_LENGTH} characters")
    return cleaned


def validate_item_intent(payload: Any) -> dict[str, str | None]:
    """Validate the narrow item schema expected by the store automation.

    Large-language-model output is untrusted input. This function accepts only
    an item name and optional descriptive text; price, retailer, quantity, and
    checkout decisions remain explicit application/user controls.
    """
    if not isinstance(payload, dict):
        raise ValueError("item intent must be an object")
    return {
        "item": _clean_text(payload.get("item"), "item"),
        "details": _clean_text(payload.get("details"), "details", allow_none=True),
    }


def validate_order_item(payload: Any) -> dict[str, str | int | None]:
    """Validate a model-proposed cart line before it reaches browser actions."""
    item = validate_item_intent(payload)
    quantity = payload.get("quantity")
    if isinstance(quantity, bool) or not isinstance(quantity, int):
        raise ValueError("quantity must be an integer")
    if not 1 <= quantity <= MAX_QUANTITY:
        raise ValueError(f"quantity must be between 1 and {MAX_QUANTITY}")
    return {**item, "quantity": quantity}
