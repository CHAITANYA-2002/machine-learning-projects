"""The CNN architecture, described analytically.

The original notebook built its network with ``keras.Sequential`` and printed a
``model.summary()``. This module reconstructs that same architecture as plain
data — layer specifications, the shape of the tensor flowing between them, and
the parameter count of each — using only the arithmetic Keras itself applies.

Doing it this way means the architecture can be inspected, tested, and
documented without installing TensorFlow or obtaining the image dataset. The
test-suite asserts every shape and parameter count here matches the
``model.summary()`` output preserved in the notebook.

The two formulas that do all the work:

* **Convolution output size** with ``padding='valid'`` and stride 1::

      out = in - kernel + 1

* **Convolution parameters**, including one bias per filter::

      params = (kernel_h * kernel_w * in_channels + 1) * filters

Max pooling has no parameters and halves each spatial dimension, discarding any
odd remainder (``floor`` division).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# The input the generator produces: 150x150 RGB, rescaled to [0, 1].
INPUT_SHAPE: tuple[int, int, int] = (150, 150, 3)


@dataclass(frozen=True)
class Layer:
    """One layer's name, output shape, and trainable parameter count.

    Attributes:
        name: The layer name as Keras assigns it.
        kind: The layer class, e.g. ``"Conv2D"``.
        output_shape: Shape of the tensor this layer emits, excluding batch.
        parameters: Number of trainable parameters.
        note: A short description of what the layer does.
    """

    name: str
    kind: str
    output_shape: tuple[int, ...]
    parameters: int
    note: str = field(default="")


def conv_output_size(size: int, kernel: int = 3) -> int:
    """Return the spatial size after a valid-padded, stride-1 convolution.

    Args:
        size: Input height or width.
        kernel: Square kernel size.

    Returns:
        ``size - kernel + 1``.

    Raises:
        ValueError: If the kernel is larger than the input.
    """
    if kernel > size:
        raise ValueError(f"kernel {kernel} exceeds input size {size}.")
    return size - kernel + 1


def conv_parameters(in_channels: int, filters: int, kernel: int = 3) -> int:
    """Return the trainable parameter count of a Conv2D layer.

    Each filter holds ``kernel * kernel * in_channels`` weights plus one bias.

    Args:
        in_channels: Channels in the incoming tensor.
        filters: Number of filters this layer learns.
        kernel: Square kernel size.

    Returns:
        The total parameter count.
    """
    return (kernel * kernel * in_channels + 1) * filters


def pool_output_size(size: int, pool: int = 2) -> int:
    """Return the spatial size after max pooling, discarding any remainder.

    A 15x15 feature map pooled by 2 becomes 7x7, not 7.5 — the final row and
    column are dropped.
    """
    return size // pool


def dense_parameters(inputs: int, units: int) -> int:
    """Return the trainable parameter count of a Dense layer, including biases."""
    return (inputs + 1) * units


def build_architecture(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    conv_filters: tuple[int, ...] = (32, 64, 128, 128),
    dense_units: int = 512,
    dropout_rate: float = 0.5,
) -> list[Layer]:
    """Reconstruct the full layer stack with shapes and parameter counts.

    The network is four convolution/pooling blocks, then a flatten, dropout,
    and two dense layers. Each convolution block doubles or holds the channel
    count while pooling halves the spatial dimensions — the standard trade of
    spatial resolution for feature depth.

    Args:
        input_shape: ``(height, width, channels)`` of the input image.
        conv_filters: Filter count for each convolution block, in order.
        dense_units: Units in the penultimate dense layer.
        dropout_rate: Fraction of activations dropped during training.

    Returns:
        The layers in order, from first convolution to the output.
    """
    height, width, channels = input_shape
    layers: list[Layer] = []

    for index, filters in enumerate(conv_filters):
        height = conv_output_size(height)
        width = conv_output_size(width)
        layers.append(
            Layer(
                name="conv2d" if index == 0 else f"conv2d_{index}",
                kind="Conv2D",
                output_shape=(height, width, filters),
                parameters=conv_parameters(channels, filters),
                note=f"learns {filters} filters over the previous {channels} channels",
            )
        )
        channels = filters

        height = pool_output_size(height)
        width = pool_output_size(width)
        layers.append(
            Layer(
                name="max_pooling2d" if index == 0 else f"max_pooling2d_{index}",
                kind="MaxPooling2D",
                output_shape=(height, width, channels),
                parameters=0,
                note="halves both spatial dimensions, keeping the strongest response",
            )
        )

    flattened = height * width * channels
    layers.append(
        Layer(
            name="flatten",
            kind="Flatten",
            output_shape=(flattened,),
            parameters=0,
            note=f"unrolls the {height}x{width}x{channels} map into one long vector",
        )
    )
    layers.append(
        Layer(
            name="dropout",
            kind="Dropout",
            output_shape=(flattened,),
            parameters=0,
            note=f"zeroes {dropout_rate:.0%} of activations during training only",
        )
    )
    layers.append(
        Layer(
            name="dense",
            kind="Dense",
            output_shape=(dense_units,),
            parameters=dense_parameters(flattened, dense_units),
            note="the fully-connected layer that combines all spatial features",
        )
    )
    layers.append(
        Layer(
            name="dense_1",
            kind="Dense",
            output_shape=(1,),
            parameters=dense_parameters(dense_units, 1),
            note="a single sigmoid output: probability of class 1",
        )
    )
    return layers


def total_parameters(layers: list[Layer] | None = None) -> int:
    """Return the total trainable parameter count of the architecture."""
    return sum(layer.parameters for layer in (layers or build_architecture()))


def summary() -> str:
    """Render the architecture as an aligned text table.

    Mirrors the shape of Keras's own ``model.summary()`` so the two can be
    compared side by side.
    """
    layers = build_architecture()
    lines = [
        f"{'Layer (type)':<28}{'Output Shape':<22}{'Param #':>12}",
        "=" * 62,
    ]
    for layer in layers:
        shape = f"(None, {', '.join(str(value) for value in layer.output_shape)})"
        lines.append(f"{layer.name + ' (' + layer.kind + ')':<28}{shape:<22}{layer.parameters:>12,}")
    lines.append("=" * 62)
    lines.append(f"{'Total params:':<50}{total_parameters(layers):>12,}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
