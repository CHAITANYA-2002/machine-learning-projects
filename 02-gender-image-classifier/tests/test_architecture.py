"""Tests for the analytically reconstructed CNN architecture.

Every expected value below is copied from the ``model.summary()`` output saved
in cell 23 of ``notebooks/01_gender_image_classifier.ipynb``. If the
reconstruction in ``src/architecture.py`` ever drifts from what Keras actually
built, these fail.
"""

from __future__ import annotations

import pytest

from src.architecture import (
    INPUT_SHAPE,
    build_architecture,
    conv_output_size,
    conv_parameters,
    dense_parameters,
    pool_output_size,
    summary,
    total_parameters,
)

# Transcribed verbatim from the notebook's saved model.summary() output:
# (layer name, output shape excluding batch, parameter count).
KERAS_SUMMARY = [
    ("conv2d", (148, 148, 32), 896),
    ("max_pooling2d", (74, 74, 32), 0),
    ("conv2d_1", (72, 72, 64), 18_496),
    ("max_pooling2d_1", (36, 36, 64), 0),
    ("conv2d_2", (34, 34, 128), 73_856),
    ("max_pooling2d_2", (17, 17, 128), 0),
    ("conv2d_3", (15, 15, 128), 147_584),
    ("max_pooling2d_3", (7, 7, 128), 0),
    ("flatten", (6272,), 0),
    ("dropout", (6272,), 0),
    ("dense", (512,), 3_211_776),
    ("dense_1", (1,), 513),
]

KERAS_TOTAL_PARAMS = 3_453_121


class TestMatchesKerasSummary:
    """The reconstruction must agree with the real model, layer by layer."""

    def test_layer_count_matches(self):
        assert len(build_architecture()) == len(KERAS_SUMMARY)

    @pytest.mark.parametrize("index,expected", enumerate(KERAS_SUMMARY))
    def test_each_layer_matches(self, index, expected):
        name, shape, parameters = expected
        layer = build_architecture()[index]
        assert layer.name == name
        assert layer.output_shape == shape
        assert layer.parameters == parameters

    def test_total_parameters_match(self):
        assert total_parameters() == KERAS_TOTAL_PARAMS

    def test_summary_reports_the_total(self):
        assert f"{KERAS_TOTAL_PARAMS:,}" in summary()


class TestShapeArithmetic:
    def test_valid_convolution_loses_two_pixels_per_side(self):
        assert conv_output_size(150) == 148
        assert conv_output_size(74) == 72

    def test_pooling_halves_and_discards_the_remainder(self):
        assert pool_output_size(148) == 74
        assert pool_output_size(15) == 7  # not 7.5

    def test_the_full_spatial_chain(self):
        """150 -> 148 -> 74 -> 72 -> 36 -> 34 -> 17 -> 15 -> 7."""
        size = INPUT_SHAPE[0]
        chain = [size]
        for _ in range(4):
            size = conv_output_size(size)
            chain.append(size)
            size = pool_output_size(size)
            chain.append(size)
        assert chain == [150, 148, 74, 72, 36, 34, 17, 15, 7]

    def test_oversized_kernel_raises(self):
        with pytest.raises(ValueError, match="exceeds input size"):
            conv_output_size(2, kernel=3)


class TestParameterArithmetic:
    def test_first_convolution_counts_rgb_channels(self):
        # (3 * 3 * 3 + 1) * 32
        assert conv_parameters(in_channels=3, filters=32) == 896

    def test_convolution_cost_scales_with_input_channels(self):
        assert conv_parameters(in_channels=32, filters=64) == 18_496
        assert conv_parameters(in_channels=64, filters=128) == 73_856
        assert conv_parameters(in_channels=128, filters=128) == 147_584

    def test_dense_layer_includes_one_bias_per_unit(self):
        assert dense_parameters(6272, 512) == 3_211_776
        assert dense_parameters(512, 1) == 513

    def test_the_dense_layer_dominates_the_budget(self):
        """93% of the parameters sit in the single 6272 -> 512 layer."""
        layers = build_architecture()
        dense = next(layer for layer in layers if layer.name == "dense")
        assert dense.parameters / total_parameters(layers) > 0.9

    def test_pooling_and_flatten_are_free(self):
        for layer in build_architecture():
            if layer.kind in {"MaxPooling2D", "Flatten", "Dropout"}:
                assert layer.parameters == 0


class TestArchitectureIsConfigurable:
    def test_narrower_network_has_fewer_parameters(self):
        narrow = build_architecture(conv_filters=(16, 32, 64, 64), dense_units=128)
        assert total_parameters(narrow) < total_parameters()

    def test_smaller_input_shrinks_the_flatten_layer(self):
        small = build_architecture(input_shape=(64, 64, 3))
        flatten = next(layer for layer in small if layer.name == "flatten")
        assert flatten.output_shape[0] < 6272
