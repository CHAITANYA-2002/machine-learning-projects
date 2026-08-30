"""Behavioural checks for the portable colour-normalisation implementation."""

from __future__ import annotations

import numpy as np

from src.color_normalization import channel_statistics_rgb, normalize_rgb_to_reference


def _two_colour_patch(left: tuple[int, int, int], right: tuple[int, int, int]) -> np.ndarray:
    """Create a tiny RGB image with non-zero variation in every channel."""
    patch = np.empty((8, 10, 3), dtype=np.uint8)
    patch[:, :5] = left
    patch[:, 5:] = right
    return patch


def test_normalization_preserves_image_contract() -> None:
    source = _two_colour_patch((30, 80, 150), (90, 160, 210))
    reference = _two_colour_patch((170, 55, 35), (230, 135, 90))

    result = normalize_rgb_to_reference(source, reference)

    assert result.shape == source.shape
    assert result.dtype == np.uint8
    assert np.all((result >= 0) & (result <= 255))
    assert not np.shares_memory(source, result)


def test_reference_changes_output_colour_distribution() -> None:
    source = _two_colour_patch((30, 80, 150), (90, 160, 210))
    warm_reference = _two_colour_patch((170, 55, 35), (230, 135, 90))
    cool_reference = _two_colour_patch((15, 110, 170), (80, 200, 245))

    warm_result = normalize_rgb_to_reference(source, warm_reference)
    cool_result = normalize_rgb_to_reference(source, cool_reference)

    assert not np.array_equal(warm_result, cool_result)
    assert np.linalg.norm(warm_result.mean(axis=(0, 1)) - cool_result.mean(axis=(0, 1))) > 10


def test_constant_channels_do_not_divide_by_zero() -> None:
    source = np.full((6, 7, 3), (100, 120, 140), dtype=np.uint8)
    reference = np.full((6, 7, 3), (180, 60, 40), dtype=np.uint8)

    result = normalize_rgb_to_reference(source, reference)

    assert np.isfinite(result).all()
    assert result.dtype == np.uint8


def test_reported_statistics_are_lab_channel_pairs() -> None:
    image = _two_colour_patch((30, 80, 150), (90, 160, 210))

    statistics = channel_statistics_rgb(image)

    assert tuple(statistics) == ("L", "a", "b")
    assert all(value[1] > 0 for value in statistics.values())
