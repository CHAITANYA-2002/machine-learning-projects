"""Add explanatory markdown and inline comments to the 2022 notebook.

The notebook was written as 45 consecutive code cells with essentially no
prose. This script annotates it so a reader can follow what each stage does
without reading the code line by line.

Two rules govern what this script may do:

1. **No code is changed.** Every original code line survives verbatim.
2. **No output is cleared.** The saved outputs are what
   ``data/training_log_2022.csv`` and the architecture tests are derived from.

The script is idempotent -- a marker on each inserted cell means a second run
changes nothing.

Run from the project directory:

    python scripts/annotate_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = PROJECT_ROOT / "notebooks" / "01_gender_image_classifier.ipynb"

MARKER = "<!-- annotated -->"

SECTIONS: dict[int, str] = {
    0: """# Image Classifier — a convolutional neural network in Keras

Trains a CNN to sort images into the two classes given by the folders they were
loaded from.

This notebook is from 2022 and is preserved as it was written. Markdown and
comments were added later so the workflow can be followed end to end. The full
write-up, with diagrams, is in the [README](../README.md).

## What happens here

| Stage | Cells | What it produces |
|---|---|---|
| Set up the data generator | 3–6 | 3,309 images, resized and augmented on the fly |
| Build the network | 7–23 | A 3.45M-parameter CNN |
| Compile | 24 | Adam optimiser, binary cross-entropy loss |
| Train | 28 | 25 epochs over the image set |
| Predict and save | 29–43 | Predictions on the training data, model pickled |

## The task

Binary image classification. The model reads a 150×150 RGB image and outputs a
single number between 0 and 1 — the probability that it belongs to class 1.
The two classes are simply the two subfolders the images were loaded from.

## Running it

The image dataset is not included in this repository — see
[`data/README.md`](../data/README.md). The saved outputs below are from the
original run, so the workflow can be read through without it.

---

## 1 · Imports""",

    3: """## 2 · Point at the image folders

`ImageDataGenerator.flow_from_directory` infers the class labels from the
subfolder names, so a directory laid out like this needs no label file:

```
data/
├── class_a/    ← every image in here gets label 0
└── class_b/    ← every image in here gets label 1
```

The path below is the local machine it was written on; substitute your own.""",

    4: """## 3 · Configure the data generator

`ImageDataGenerator` streams images from disk in batches rather than loading
all 3,309 into memory at once, and applies random transformations as it goes.

Each epoch therefore sees slightly different versions of the same photographs —
flipped, shifted, zoomed — which multiplies the effective size of the dataset
and makes the network less able to memorise individual images.

| Setting | Value | Unit | Effect |
|---|---|---|---|
| `rescale` | 1/255 | factor | maps pixel values from 0–255 to 0–1 |
| `horizontal_flip` | True | on/off | mirrors the image left to right |
| `width_shift_range` | 0.2 | fraction | slides up to 20% horizontally |
| `height_shift_range` | 0.2 | fraction | slides up to 20% vertically |
| `shear_range` | 0.2 | degrees | slants the image |
| `rotation_range` | 0.2 | degrees | rotates the image |
| `zoom_range` | 0.2 | fraction | scales between 0.8× and 1.2× |

Note the units are not uniform: the shift and zoom ranges are *fractions* of
the image, while shear and rotation are measured in *degrees*.

These transforms are reproduced in
[`src/augmentation.py`](../src/augmentation.py) and illustrated in the
[README](../README.md).""",

    6: """### Load the images

`target_size=(150, 150)` resizes every image to a fixed square, because a
network's dense layers need a fixed input size. `class_mode='binary'` produces
a single 0/1 label per image rather than a one-hot vector.

The output records what was found: **3,309 images across 2 classes**.""",

    7: """## 4 · Build the network

The architecture is four convolution/pooling blocks followed by a classifier
head. Each block detects patterns at a coarser scale than the last:

| Block | Filters | Output size |
|---|---|---|
| 1 | 32 | 74×74 |
| 2 | 64 | 36×36 |
| 3 | 128 | 17×17 |
| 4 | 128 | 7×7 |

**What a convolution does.** It slides a small 3×3 window across the image,
computing a weighted sum at each position. The weights are learned, so early
layers converge on simple detectors — edges, corners, colour transitions —
while later layers combine those into larger structures.

**What pooling does.** `MaxPooling2D((2,2))` takes the strongest value in each
2×2 square, halving the width and height. This discards precise position while
keeping the presence of a feature, which is what makes the network tolerant of
small shifts.

The pattern of shrinking spatially while growing in depth is the standard
convolutional trade: less *where*, more *what*.

This architecture is reconstructed analytically in
[`src/architecture.py`](../src/architecture.py), whose tests assert it matches
the `model.summary()` output below exactly.""",

    18: """### Flatten and classify

`Flatten` unrolls the final 7×7×128 feature map into a single vector of
**6,272** values. `Dropout(0.5)` then randomly zeroes half of them on each
training pass, forcing the network to spread its reasoning across many features
rather than depending on a few.

The `Dense(512)` layer that follows connects all 6,272 inputs to 512 units,
which costs 3,211,776 parameters — 93% of the whole model. The final
`Dense(1, activation='sigmoid')` squeezes those 512 values into one probability.""",

    23: """### The parameter count

`model.summary()` reports **3,453,121 trainable parameters** across the whole
network, and shows how the tensor shrinks from 150×150 down to 7×7.""",

    24: """## 5 · Compile

`binary_crossentropy` is the standard loss for a two-class problem with a
sigmoid output: it penalises confident wrong answers much more heavily than
uncertain ones. `adam` is an adaptive optimiser that adjusts the learning rate
per parameter as training proceeds.""",

    28: """## 6 · Train

25 epochs, 100 batches of 32 images each — so roughly 3,200 images per epoch,
approximately one full pass over the 3,309 available.

The output below is the original 2022 run. It is transcribed to
[`data/training_log_2022.csv`](../data/training_log_2022.csv) and charted in
the [README](../README.md).

At about 71 seconds per epoch, the run took roughly **30 minutes** on CPU.""",

    29: """## 7 · Save the model

`pickle` writes the trained model to disk so it can be reloaded without
retraining. Keras also offers its own `model.save()`, which stores the
architecture, weights, and optimiser state in a portable format — generally the
better choice for Keras models.""",

    31: """## 8 · Predict

Runs the trained model over the image set and converts its output probabilities
into class labels.

The model emits one float per image, such as `0.574` or `0.838`. Rounding at a
threshold of 0.5 turns each into a 0 or a 1.""",
}

COMMENTS: dict[int, str] = {
    3: "# Local path to the image folders; one subfolder per class.",
    5: "# Augmentation: random flips, shifts, shears and zooms on every epoch.",
    6: "# Resize everything to 150x150; labels come from the folder names.",
    10: "# 32 filters, 3x3 window -> output 148x148 (valid padding loses 2px).",
    11: "# Halve both dimensions: 148x148 -> 74x74.",
    12: "# Deeper: 64 filters over the previous 32 channels.",
    14: "# Deeper still: 128 filters.",
    18: "# Unroll the final 7x7x128 feature map into 6,272 values.",
    19: "# Zero half the activations during training to reduce overfitting.",
    20: "# The largest layer: 6,272 x 512 = 3.2M parameters.",
    21: "# One sigmoid output = probability of class 1.",
    23: "# 3,453,121 trainable parameters in total.",
    24: "# Binary cross-entropy is the standard loss for two-class problems.",
    28: "# 25 epochs x 100 batches x 32 images ~= one pass over the data per epoch.",
    30: "# Persist the trained model so it can be reloaded without retraining.",
    31: "# Model outputs one probability per image.",
    41: "# Round at 0.5 to turn probabilities into class labels.",
}

CLOSING = """---

## 9 · Results

| Metric | Epoch 1 | Epoch 25 |
|---|---|---|
| Training accuracy | 0.5753 | **0.7581** |
| Training loss | 0.6746 | **0.4923** |

Accuracy rose from roughly chance (0.5753, where guessing scores 0.50) to
**0.7581** over 25 epochs, and the loss fell steadily from 0.6746 to 0.4923.
The curves were still improving when training stopped, so the model had not yet
converged.

**These are training figures.** `flow_from_directory` was pointed at a single
folder with no `validation_split`, so every image the model was scored on was
also an image it trained on. Measuring generalisation would need a held-out set
of images the model never saw — ideally split so that no individual appears in
both halves.

## What is in the rest of this project

| Path | What it holds |
|---|---|
| [`README.md`](../README.md) | The full walkthrough, with diagrams |
| [`src/architecture.py`](../src/architecture.py) | The layer shapes and parameter counts, tested against the summary above |
| [`src/augmentation.py`](../src/augmentation.py) | The augmentation transforms, reproduced and tested |
| [`data/training_log_2022.csv`](../data/training_log_2022.csv) | The recovered 25-epoch history |

## Scope

This model sorts images into the two folders it was trained on. Those folder
labels describe how the dataset was organised — they are not a judgement about
any person, and appearance does not determine identity.

A model trained on a limited image collection can just as easily learn lighting,
camera angle, hairstyle, background, or image source as anything else. Nothing
here should be used for identification, screening, access control, moderation,
or any decision about a person.
"""


def markdown_cell(text: str) -> dict:
    """Build a markdown cell carrying the idempotency marker."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": f"{text}\n\n{MARKER}".split("\n"),
    }


def already_annotated(notebook: dict) -> bool:
    """Return True when this script has already run on the notebook."""
    return any(
        cell["cell_type"] == "markdown" and MARKER in "".join(cell["source"])
        for cell in notebook["cells"]
    )


def prepend_comment(cell: dict, comment: str) -> None:
    """Insert a comment block at the top of a code cell, once."""
    source = cell.get("source", [])
    if comment.split("\n")[0] in "".join(source):
        return
    cell["source"] = [f"{line}\n" for line in comment.split("\n")] + source


def drop_trailing_empty_cells(cells: list) -> int:
    """Remove empty cells from the end of the notebook."""
    removed = 0
    while cells and not "".join(cells[-1].get("source", [])).strip():
        cells.pop()
        removed += 1
    return removed


def annotate(path: Path) -> None:
    """Apply every annotation to the notebook at ``path``."""
    notebook = json.loads(path.read_text(encoding="utf-8"))

    if already_annotated(notebook):
        print(f"{path.name}: already annotated")
        return

    original = notebook["cells"]
    annotated: list = []

    for index, cell in enumerate(original):
        if index in SECTIONS:
            annotated.append(markdown_cell(SECTIONS[index]))
        if index in COMMENTS and cell["cell_type"] == "code":
            prepend_comment(cell, COMMENTS[index])
        # Drop the stray empty markdown and raw cells left in the original.
        if cell["cell_type"] in {"markdown", "raw"} and not "".join(
            cell.get("source", [])
        ).strip():
            continue
        annotated.append(cell)

    removed = drop_trailing_empty_cells(annotated)
    annotated.append(markdown_cell(CLOSING))

    notebook["cells"] = annotated
    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(
        f"{path.name}: +{len(SECTIONS)} markdown sections, "
        f"+{len(COMMENTS)} comment blocks, "
        f"-{removed} empty cells, "
        f"{len(original)} -> {len(annotated)} cells"
    )


def main() -> None:
    """Annotate the original notebook."""
    if not NOTEBOOK.exists():
        raise SystemExit(f"Notebook not found: {NOTEBOOK}")
    annotate(NOTEBOOK)


if __name__ == "__main__":
    main()
