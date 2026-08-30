"""The training-time image augmentation, reproduced without TensorFlow.

The original notebook configured a ``keras.preprocessing.image.ImageDataGenerator``
with the settings in :data:`AUGMENTATION`. This module documents what each of
those settings means and implements the same geometric transforms with Pillow,
so the pipeline can be demonstrated and tested without TensorFlow and without
the image dataset.

**Units matter here, and they are not consistent across the settings.** In
``ImageDataGenerator``:

* ``width_shift_range`` and ``height_shift_range`` are *fractions* of the image
  dimension, so ``0.2`` means a shift of up to 20%.
* ``zoom_range`` is a *fraction*, so ``0.2`` means a zoom between 0.8x and 1.2x.
* ``shear_range`` is measured in *degrees*.
* ``rotation_range`` is also measured in *degrees*, so ``0.2`` is a fifth of a
  degree, not 20%.

Augmentation is applied only while training. Validation and test images are
rescaled but never transformed, because the goal is to measure performance on
images as they actually arrive.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

# The exact configuration used in the original notebook.
AUGMENTATION: dict[str, float | bool] = {
    "rescale": 1.0 / 255,
    "horizontal_flip": True,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "shear_range": 0.2,
    "rotation_range": 0.2,
    "zoom_range": 0.2,
}

# Every image is resized to this before entering the network.
TARGET_SIZE: tuple[int, int] = (150, 150)


@dataclass(frozen=True)
class Setting:
    """One augmentation setting, its value, unit, and effect."""

    name: str
    value: float | bool
    unit: str
    effect: str


def describe() -> list[Setting]:
    """Return each augmentation setting with its unit and practical effect.

    Used by the documentation and the figure script so the explanation and the
    configuration can never drift apart.
    """
    return [
        Setting("rescale", 1 / 255, "factor", "maps pixel values from 0-255 to 0-1"),
        Setting("horizontal_flip", True, "on/off", "mirrors the image left to right"),
        Setting("width_shift_range", 0.2, "fraction", "slides up to 20% horizontally"),
        Setting("height_shift_range", 0.2, "fraction", "slides up to 20% vertically"),
        Setting("shear_range", 0.2, "degrees", "slants the image by up to 0.2 degrees"),
        Setting("rotation_range", 0.2, "degrees", "rotates by up to 0.2 degrees"),
        Setting("zoom_range", 0.2, "fraction", "scales between 0.8x and 1.2x"),
    ]


def rescale(image: np.ndarray) -> np.ndarray:
    """Map an 8-bit image array to floats in ``[0, 1]``.

    Neural networks train poorly on raw 0-255 values: large inputs produce
    large activations and unstable gradients.
    """
    return np.asarray(image, dtype=np.float32) * AUGMENTATION["rescale"]


def horizontal_flip(image: Image.Image) -> Image.Image:
    """Mirror an image left to right."""
    return image.transpose(Image.FLIP_LEFT_RIGHT)


def shift(image: Image.Image, width_fraction: float, height_fraction: float) -> Image.Image:
    """Translate an image by a fraction of its own dimensions.

    Args:
        image: The image to shift.
        width_fraction: Horizontal shift as a fraction of width.
        height_fraction: Vertical shift as a fraction of height.

    Returns:
        The shifted image, with vacated pixels filled by the nearest edge.
    """
    dx = int(round(image.width * width_fraction))
    dy = int(round(image.height * height_fraction))
    return image.transform(
        image.size,
        Image.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.BILINEAR,
        fillcolor=(255, 255, 255),
    )


def rotate(image: Image.Image, degrees: float) -> Image.Image:
    """Rotate an image about its centre by ``degrees``."""
    return image.rotate(degrees, resample=Image.BILINEAR, fillcolor=(255, 255, 255))


def shear(image: Image.Image, degrees: float) -> Image.Image:
    """Slant an image horizontally by ``degrees``."""
    factor = math.tan(math.radians(degrees))
    return image.transform(
        image.size,
        Image.AFFINE,
        (1, factor, -factor * image.height / 2, 0, 1, 0),
        resample=Image.BILINEAR,
        fillcolor=(255, 255, 255),
    )


def zoom(image: Image.Image, factor: float) -> Image.Image:
    """Scale an image about its centre, cropping or padding back to size.

    Args:
        image: The image to zoom.
        factor: Values above 1 magnify; below 1 shrink.

    Raises:
        ValueError: If ``factor`` is not positive.
    """
    if factor <= 0:
        raise ValueError(f"zoom factor must be positive, got {factor}.")

    width, height = image.size
    scaled = image.resize(
        (max(1, int(width * factor)), max(1, int(height * factor))),
        resample=Image.BILINEAR,
    )

    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    offset = ((width - scaled.width) // 2, (height - scaled.height) // 2)
    if factor >= 1:
        canvas.paste(scaled.crop((-offset[0], -offset[1], -offset[0] + width, -offset[1] + height)))
    else:
        canvas.paste(scaled, offset)
    return canvas


def augment(image: Image.Image, seed: int | None = None) -> Image.Image:
    """Apply one random draw of the full augmentation pipeline.

    Each call samples independently from every configured range, so the network
    sees a different variation of the same source image on every epoch.

    Args:
        image: The source image.
        seed: Optional seed, for reproducible output.

    Returns:
        A transformed copy. The original is not modified.
    """
    rng = random.Random(seed)
    result = image

    if AUGMENTATION["horizontal_flip"] and rng.random() < 0.5:
        result = horizontal_flip(result)

    result = shift(
        result,
        rng.uniform(-AUGMENTATION["width_shift_range"], AUGMENTATION["width_shift_range"]),
        rng.uniform(-AUGMENTATION["height_shift_range"], AUGMENTATION["height_shift_range"]),
    )
    result = shear(result, rng.uniform(-AUGMENTATION["shear_range"], AUGMENTATION["shear_range"]))
    result = rotate(result, rng.uniform(-AUGMENTATION["rotation_range"], AUGMENTATION["rotation_range"]))
    result = zoom(result, rng.uniform(1 - AUGMENTATION["zoom_range"], 1 + AUGMENTATION["zoom_range"]))
    return result
