"""A small, portable implementation of reference-based LAB normalisation.

The function in this module implements the global mean/standard-deviation
transfer commonly associated with Reinhard-style colour normalisation.  It is
intended for reproducible research experiments, not for diagnostic image use.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


_D65_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)
_RGB_TO_XYZ = np.array(
    [[0.4124564, 0.3575761, 0.1804375],
     [0.2126729, 0.7151522, 0.0721750],
     [0.0193339, 0.1191920, 0.9503041]],
    dtype=np.float64,
)
_XYZ_TO_RGB = np.linalg.inv(_RGB_TO_XYZ)
_LAB_DELTA = 6 / 29


def _validate_rgb(image: np.ndarray, name: str) -> np.ndarray:
    """Reject ambiguous inputs before a colour-space conversion.

    Requiring an H×W×3 uint8 RGB array keeps the public contract explicit and
    avoids silently treating grayscale, alpha, BGR, or floating-point data as
    something it is not.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{name} must have shape (height, width, 3) in RGB order.")
    if array.dtype != np.uint8:
        raise TypeError(f"{name} must use uint8 pixels; received {array.dtype}.")
    if not array.size:
        raise ValueError(f"{name} must not be empty.")
    return array


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an sRGB uint8 array to CIE LAB using a D65 reference white."""
    rgb_unit = rgb.astype(np.float64) / 255.0
    # Undo sRGB gamma before applying the linear RGB-to-XYZ transform.
    linear_rgb = np.where(
        rgb_unit <= 0.04045,
        rgb_unit / 12.92,
        ((rgb_unit + 0.055) / 1.055) ** 2.4,
    )
    xyz = linear_rgb @ _RGB_TO_XYZ.T / _D65_WHITE
    f_xyz = np.where(
        xyz > _LAB_DELTA**3,
        np.cbrt(xyz),
        xyz / (3 * _LAB_DELTA**2) + 4 / 29,
    )
    l_channel = 116 * f_xyz[..., 1] - 16
    a_channel = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
    b_channel = 200 * (f_xyz[..., 1] - f_xyz[..., 2])
    return np.stack((l_channel, a_channel, b_channel), axis=-1)


def _lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert CIE LAB values back to clipped, uint8 sRGB pixels."""
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    f_xyz = np.stack((fx, fy, fz), axis=-1)
    xyz_ratio = np.where(
        f_xyz > _LAB_DELTA,
        f_xyz**3,
        3 * _LAB_DELTA**2 * (f_xyz - 4 / 29),
    )
    # Out-of-gamut LAB values can create small negative linear RGB values.
    # They cannot be displayed in sRGB, so clamp before applying a fractional
    # exponent and then record clipping in a full research pipeline.
    linear_rgb = np.clip((xyz_ratio * _D65_WHITE) @ _XYZ_TO_RGB.T, 0.0, None)
    # Reapply the sRGB transfer function after the linear-space transform.
    srgb = np.where(
        linear_rgb <= 0.0031308,
        12.92 * linear_rgb,
        1.055 * linear_rgb ** (1 / 2.4) - 0.055,
    )
    return np.rint(np.clip(srgb, 0.0, 1.0) * 255).astype(np.uint8)


def channel_statistics_rgb(image: np.ndarray) -> Mapping[str, tuple[float, float]]:
    """Return LAB channel mean and standard deviation for an RGB image."""
    lab = _rgb_to_lab(_validate_rgb(image, "image"))
    return {
        channel: (float(lab[..., index].mean()), float(lab[..., index].std()))
        for index, channel in enumerate(("L", "a", "b"))
    }


def normalize_rgb_to_reference(
    source: np.ndarray,
    reference: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Match source LAB channel moments to a chosen reference image.

    For each LAB channel, ``(source - source_mean)`` is rescaled by the ratio
    of reference and source standard deviations, then shifted to the reference
    mean.  ``epsilon`` prevents a uniform source channel from causing a divide
    by zero.  Clipping is unavoidable when converting back to displayable
    8-bit RGB and should therefore be measured in a real study.
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    source_lab = _rgb_to_lab(_validate_rgb(source, "source"))
    reference_lab = _rgb_to_lab(_validate_rgb(reference, "reference"))

    source_mean = source_lab.mean(axis=(0, 1), keepdims=True)
    source_std = source_lab.std(axis=(0, 1), keepdims=True)
    reference_mean = reference_lab.mean(axis=(0, 1), keepdims=True)
    reference_std = reference_lab.std(axis=(0, 1), keepdims=True)

    adjusted_lab = (
        (source_lab - source_mean)
        * (reference_std / np.maximum(source_std, epsilon))
        + reference_mean
    )
    return _lab_to_rgb(adjusted_lab)
