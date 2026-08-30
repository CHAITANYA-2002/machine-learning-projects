from src.checkout_safety import MANUAL_CHECKOUT_MESSAGE, require_manual_checkout


class _VoiceSpy:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def speak(self, message: str) -> None:
        self.messages.append(message)


def test_sensitive_checkout_is_explicitly_left_to_the_user() -> None:
    voice = _VoiceSpy()

    assert require_manual_checkout(voice) is False
    assert voice.messages == [MANUAL_CHECKOUT_MESSAGE]
