"""Tests for the augmentation pipeline reproduction."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.augmentation import (
    AUGMENTATION,
    TARGET_SIZE,
    augment,
    describe,
    horizontal_flip,
    rescale,
    rotate,
    shear,
    shift,
    zoom,
)


@pytest.fixture
def asymmetric_image() -> Image.Image:
    """A 100x100 image with a red square in the top-left quadrant only.

    Asymmetry matters: a symmetric test image cannot detect a flip.
    """
    image = Image.new("RGB", (100, 100), (255, 255, 255))
    for x in range(10, 40):
        for y in range(10, 40):
            image.putpixel((x, y), (200, 40, 40))
    return image


def _red_pixels(image: Image.Image) -> int:
    """Count the red pixels in an image, used to measure zoom."""
    array = np.asarray(image)
    return int(((array[:, :, 0] > 150) & (array[:, :, 1] < 100)).sum())


class TestConfiguration:
    def test_matches_the_notebook_settings(self):
        assert AUGMENTATION == {
            "rescale": 1.0 / 255,
            "horizontal_flip": True,
            "width_shift_range": 0.2,
            "height_shift_range": 0.2,
            "shear_range": 0.2,
            "rotation_range": 0.2,
            "zoom_range": 0.2,
        }

    def test_target_size_matches_the_network_input(self):
        assert TARGET_SIZE == (150, 150)

    def test_every_setting_is_described(self):
        described = {setting.name for setting in describe()}
        assert described == set(AUGMENTATION)

    def test_units_are_recorded_because_they_differ(self):
        """Shifts and zoom are fractions; shear and rotation are degrees."""
        units = {setting.name: setting.unit for setting in describe()}
        assert units["width_shift_range"] == "fraction"
        assert units["zoom_range"] == "fraction"
        assert units["rotation_range"] == "degrees"
        assert units["shear_range"] == "degrees"


class TestRescale:
    def test_maps_to_unit_interval(self):
        array = np.array([[0, 128, 255]], dtype=np.uint8)
        result = rescale(array)
        assert result.min() == 0.0
        assert result.max() == pytest.approx(1.0)

    def test_preserves_shape(self):
        array = np.zeros((150, 150, 3), dtype=np.uint8)
        assert rescale(array).shape == (150, 150, 3)


class TestGeometricTransforms:
    def test_horizontal_flip_moves_content_to_the_other_side(self, asymmetric_image):
        flipped = horizontal_flip(asymmetric_image)
        # The red square starts on the left and must end up on the right.
        assert asymmetric_image.getpixel((25, 25)) != (255, 255, 255)
        assert flipped.getpixel((74, 25)) != (255, 255, 255)
        assert flipped.getpixel((25, 25)) == (255, 255, 255)

    def test_flipping_twice_is_the_identity(self, asymmetric_image):
        twice = horizontal_flip(horizontal_flip(asymmetric_image))
        np.testing.assert_array_equal(np.asarray(twice), np.asarray(asymmetric_image))

    def test_shift_moves_content(self, asymmetric_image):
        shifted = shift(asymmetric_image, 0.2, 0.0)
        assert not np.array_equal(np.asarray(shifted), np.asarray(asymmetric_image))

    def test_zero_shift_changes_nothing(self, asymmetric_image):
        np.testing.assert_array_equal(
            np.asarray(shift(asymmetric_image, 0.0, 0.0)), np.asarray(asymmetric_image)
        )

    def test_all_transforms_preserve_image_size(self, asymmetric_image):
        for transformed in (
            horizontal_flip(asymmetric_image),
            shift(asymmetric_image, 0.2, 0.2),
            rotate(asymmetric_image, 15),
            shear(asymmetric_image, 10),
            zoom(asymmetric_image, 1.2),
            zoom(asymmetric_image, 0.8),
        ):
            assert transformed.size == asymmetric_image.size

    def test_zoom_in_enlarges_the_content(self, asymmetric_image):
        assert _red_pixels(zoom(asymmetric_image, 1.5)) > _red_pixels(asymmetric_image)

    def test_zoom_out_shrinks_the_content(self, asymmetric_image):
        assert _red_pixels(zoom(asymmetric_image, 0.5)) < _red_pixels(asymmetric_image)

    def test_non_positive_zoom_raises(self, asymmetric_image):
        with pytest.raises(ValueError, match="must be positive"):
            zoom(asymmetric_image, 0)


class TestAugmentPipeline:
    def test_output_keeps_the_input_size(self, asymmetric_image):
        assert augment(asymmetric_image, seed=0).size == asymmetric_image.size

    def test_is_deterministic_for_a_given_seed(self, asymmetric_image):
        first = augment(asymmetric_image, seed=7)
        second = augment(asymmetric_image, seed=7)
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))

    def test_different_seeds_give_different_images(self, asymmetric_image):
        variants = {
            np.asarray(augment(asymmetric_image, seed=seed)).tobytes()
            for seed in range(6)
        }
        assert len(variants) > 1

    def test_does_not_modify_the_source(self, asymmetric_image):
        before = np.asarray(asymmetric_image).copy()
        augment(asymmetric_image, seed=3)
        np.testing.assert_array_equal(np.asarray(asymmetric_image), before)
