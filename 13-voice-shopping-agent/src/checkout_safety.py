"""Hard boundary between cart assistance and sensitive financial actions."""

from __future__ import annotations

from typing import Protocol


MANUAL_CHECKOUT_MESSAGE = (
    "Your cart is ready for review. Please complete checkout and payment manually "
    "in the visible retailer browser page."
)


class VoiceOutput(Protocol):
    """The minimal voice interface needed to announce a safety boundary."""

    def speak(self, message: str) -> None:
        """Speak or otherwise present a user-facing message."""


def require_manual_checkout(voice: VoiceOutput) -> bool:
    """Stop before payment, OTP, address, or irreversible order submission."""
    voice.speak(MANUAL_CHECKOUT_MESSAGE)
    return False
