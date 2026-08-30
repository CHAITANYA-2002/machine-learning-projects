"""Add explanatory markdown to the notebooks that still have none.

Most projects in this portfolio carry their own ``scripts/annotate_notebook.py``.
This tool covers the remaining notebooks across several projects in one pass, so
every notebook in the repository opens with a description of what it does and
carries section headings through its stages.

Two rules govern what this tool may do:

1. **No code is changed.** Every original code line survives verbatim.
2. **No output is cleared.** Saved outputs are the only surviving evidence from
   the original runs.

The tool is idempotent -- a marker on each inserted cell means a second run
changes nothing.

Run from the repository root:

    python tools/annotate_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MARKER = "<!-- annotated -->"

# Per notebook: an intro cell, then markdown keyed by the code-cell index it
# should precede.
SPECS: dict[str, dict] = {
    "00-foundations/notebooks/01_find_s_concept_learning.ipynb": {
        "intro": """# Find-S — learning a concept from positive examples

Implements the Find-S algorithm from Tom Mitchell's *Machine Learning* (1997),
chapter 2, on the EnjoySport dataset.

## The idea

Find-S searches for the **most specific hypothesis** consistent with the
positive training examples. It starts with the first positive example as its
hypothesis, then walks the rest: wherever a new positive example disagrees with
the current hypothesis on some attribute, that attribute is relaxed to `?`,
meaning "any value will do".

| Symbol | Meaning |
|---|---|
| a specific value, e.g. `sunny` | the attribute must equal this |
| `?` | any value matches |

A clean, tested implementation of this lives in
[`src/find_s.py`](../src/find_s.py), and the
[README](../README.md) explains the algorithm's behaviour with diagrams.""",
        "sections": {
            1: """## Load the data

Four days described by six nominal attributes, with a yes/no target. The
dataset is deliberately tiny — the point is to follow the hypothesis by hand.""",
            3: """## Split features from the target

`concepts` holds the six attribute columns; `target` holds the yes/no label.""",
            7: """## The algorithm

Two passes. The first finds the first positive example and adopts it wholesale
as the starting hypothesis. The second walks every example, and for each
positive one compares it attribute by attribute against the current hypothesis,
relaxing any attribute that disagrees to `?`.

Negative examples are skipped entirely — Find-S generalises only from positives.""",
            8: """## The learned hypothesis

The result reads as: *sunny, warm, any humidity, strong wind, any water, any
forecast*.""",
        },
    },
    "00-foundations/notebooks/03_decision_tree_gini.ipynb": {
        "intro": """# Decision tree with the Gini criterion

Trains a scikit-learn decision tree on Quinlan's PlayTennis dataset using
**Gini impurity** as the split criterion, and prints the resulting rules.

## Gini impurity

At any node, Gini measures how mixed the labels are:

```
Gini = 1 - Σ p(class)²
```

A node where every sample shares one label scores `0`. A balanced two-class node
scores `0.5`. A tree picks, at each step, the split that reduces this the most.

The sister notebook `02_decision_tree_entropy.ipynb` does the same with entropy.
Both criteria usually build the same tree — they differ in scale, not shape.
Implementations of both live in [`src/impurity.py`](../src/impurity.py).""",
        "sections": {
            1: """## Load the data

14 days of weather observations with a yes/no decision about playing tennis.""",
            3: """## Drop the identifier

`day` is a row label (`D1`–`D14`), not a weather feature. Left in, the tree
would split on it and score perfectly by memorising row identifiers.""",
            4: """## One-hot encode the features

Every feature is categorical, so `get_dummies` turns each value into its own
0/1 column. `drop_first=True` omits one column per feature, which avoids a
redundant column that is fully determined by the others.""",
            5: """## Fit the tree

`criterion="gini"` selects Gini impurity. `random_state=0` makes tie-breaking
between equally good splits deterministic.""",
            6: """## Read out the rules

`export_text` prints the tree as nested if/else rules — the reason decision
trees are considered interpretable, unlike most other models.""",
        },
    },
    "00-foundations/notebooks/06_knn_synthetic_fruits.ipynb": {
        "intro": """# k-nearest neighbours on a synthetic fruit dataset

Generates a small labelled dataset of fruit measurements and classifies it with
KNN, sweeping `k` to see how the choice of neighbourhood size affects accuracy.

## How KNN works

There is no training step. To classify a new point, the algorithm measures the
distance to every training point, keeps the `k` closest, and takes a majority
vote among their labels.

| `k` | Behaviour |
|---|---|
| 1 | every training point is its own nearest neighbour — memorisation |
| large | smoother boundaries, less sensitive to individual points |

**Feature scaling is essential.** Distance is dominated by whichever feature has
the largest numeric range, so weight in grams would swamp size in centimetres
unless both are standardised first — which is why `StandardScaler` appears below.

A from-scratch KNN, verified against scikit-learn, is in
[`src/knn.py`](../src/knn.py).""",
        "sections": {
            2: """## Generate the data

Synthetic measurements for four fruit types: colour code, weight, and size.
Because the data is generated rather than measured, the accuracy figures here
describe the generator, not any real classification problem.""",
            3: """## Explore the features

A pairplot shows how separable the classes are on each pair of features —
useful before modelling, because well-separated clusters mean an easy problem.""",
            4: """## Split and scale

The split holds back 20% for testing. The scaler is fitted on the **training
data only** and then applied to both halves, so no information about the test
set leaks into the transformation.""",
            5: """## Choose k

Trains a model for each `k` from 1 to 29 and records the error rate, so the
best neighbourhood size can be read off rather than guessed.""",
            6: """## Evaluate

Fits the final model and reports precision, recall, and F1 per class.""",
            7: """## Predict a new fruit

Wraps the model in a function that takes raw measurements, applies the same
scaler, and returns a label.""",
        },
    },
    "07-covid-xray-classifier/notebooks/02_scratch_experiments.ipynb": {
        "intro": """# Scratch experiments — dataset setup

An exploratory notebook kept alongside the main
[`01_covid_xray_classification.ipynb`](01_covid_xray_classification.ipynb).

It covers the first steps only: importing Keras layers, cloning the chest X-ray
dataset, and locating the train and test directories. It stops before any model
is built.

Preserved because it records where the dataset came from. For the full
workflow, read the main notebook and the [README](../README.md).""",
        "sections": {
            1: """## Fetch the dataset

Clones the chest X-ray image set from its public GitHub repository. The images
are not committed to this repository — see [`data/README.md`](../data/README.md).""",
            2: """## Locate the image directories

The loader infers class labels from folder names, so the directory layout
(`train/` and `test/`, each split by class) is what defines the labels.""",
            3: """## List the files

Counts what is in each folder, confirming the download arrived intact.""",
        },
    },
    "08-gait-trajectory/notebooks/01_gait_trajectory_final.ipynb": {
        "intro": """# Gait trajectory prediction — dense network vs LSTM

Predicts three lower-limb joint angles — **hip, knee, and ankle** — from three
input measurements, and compares two network architectures on the task.

## The problem

Human walking is cyclical and smooth. Given a set of gait measurements, the goal
is to reconstruct the joint angles across the gait cycle. This is a **regression**
problem with three continuous outputs rather than a classification problem.

| | |
|---|---|
| Inputs | 3 columns (`data[:, 0:3]`) |
| Targets | 3 joint angles (`data[:, 3:6]`) |
| Normalisers | 3 columns (`data[:, 6:9]`) used to scale the error |
| Training | 500 epochs, batch size 1 |

## The custom loss

Rather than plain mean-squared error, the loss divides the absolute error by a
per-sample normaliser `Z`:

```python
sd = tf.math.abs(y_true - y_pred) / Z
```

This makes the error **proportional** rather than absolute, so a 2-degree miss on
a large joint angle is not penalised more heavily than a 2-degree miss on a small
one.

## Structure of this notebook

| Cells | What happens |
|---|---|
| 0–8 | Dense network: build, train, predict, plot |
| 9–18 | LSTM: same task, sequential architecture |
| 19–29 | 3D wireframe plots of the dense-network results |
| 30–35 | The same plots for the LSTM results |

The two architectures are compared in the [README](../README.md).""",
        "sections": {
            1: """## Load the training data

Splits the CSV into inputs `X`, target joint angles `Y`, and the normalisers `Z`
used by the custom loss.""",
            3: """## Build the dense network

A fully-connected stack: 3 → 128 → 512 → … Each layer is connected to every
unit in the previous one, so the model treats the three inputs as an unordered
set of numbers rather than a sequence.""",
            4: """## Define the proportional loss

Divides the absolute error by `Z`, making the penalty relative to the size of
the quantity being predicted.""",
            5: """## Train the dense network

500 epochs at batch size 1 — the weights update after every single sample.""",
            6: """## Load the test data

A separate CSV of held-out measurements, so the model is scored on data it did
not train on.""",
            7: """## Predict and export

Runs the trained model over the test inputs and writes the predictions to CSV
for the plotting cells further down.""",
            8: """## Plot predicted against measured

Overlays the model's output on the measured trajectory. For gait work this
visual check matters as much as the loss value: a curve can have a low average
error while still having the wrong shape.""",
            9: """---

# Part 2 — the LSTM

The same task, using a recurrent architecture instead of a dense one.

**Why an LSTM might help.** Gait is a sequence: each point in the cycle follows
from the last. An LSTM reads its input in order and carries a memory across
steps, so it can in principle model that continuity, whereas the dense network
above sees each sample independently.""",
            10: """## Reload the data for the recurrent model

The same CSV, reshaped — an LSTM expects a 3D tensor of
`(samples, timesteps, features)` rather than the 2D matrix a dense layer takes.""",
            12: """## Build the LSTM

Stacked LSTM layers with `return_sequences=True`, so each layer passes a full
sequence to the next rather than just its final state.""",
            14: """## Train the LSTM

Same budget as the dense network — 500 epochs, batch size 1 — so the comparison
between the two is like for like.""",
            15: """## Predict with the LSTM

Exports the LSTM's predictions to CSV for the comparison plots below.""",
            19: """---

# Part 3 — comparing the two models

Plots the measured trajectory against both the dense-network and LSTM
predictions for each of the three joints.""",
            22: """## Joint-by-joint comparison

Three cells, one per joint: hip, knee, and ankle. Each overlays the measured
curve with both models' predictions over the same gait cycle.""",
            25: """## 3D surface plots

Builds a coordinate mesh so the trajectory can be drawn as a wireframe surface
across time, giving a view of the whole gait cycle at once rather than a single
slice.""",
            27: """### Dense-network surfaces

One wireframe per joint, from the dense network's predictions.""",
            30: """---

# Part 4 — LSTM surfaces

The same three wireframe plots, built from the LSTM's predictions so the two
architectures can be compared visually.""",
        },
    },
    "08-gait-trajectory/notebooks/02_dense_network.ipynb": {
        "intro": """# Dense network for gait trajectory prediction

The dense-network half of the gait study, kept as a standalone notebook.

Predicts three joint angles from three input measurements using a
fully-connected network, trained with the proportional loss described in
[`01_gait_trajectory_final.ipynb`](01_gait_trajectory_final.ipynb).

A dense network treats its inputs as an unordered set of numbers. It has no
notion that gait is a sequence — that is what the LSTM in
[`03_lstm.ipynb`](03_lstm.ipynb) adds.

See the [README](../README.md) for the comparison between the two.""",
        "sections": {
            1: """## Load the training data

Inputs `X`, target joint angles `Y`, and the per-sample normalisers `Z`.""",
            3: """## Build the network

3 → 128 → 512 → … fully connected, ReLU activations throughout.""",
            4: """## The proportional loss

Absolute error divided by `Z`, so the penalty scales with the magnitude of the
quantity being predicted.""",
            5: """## Train

500 epochs, batch size 1.""",
            6: """## Saving and reloading

Commented-out cells showing how to persist the model. Because the loss is a
custom function it must be passed back through `custom_objects` on load,
otherwise Keras cannot reconstruct it.""",
            9: """## Predict on held-out data

Runs the trained model over a separate test CSV.""",
            10: """## Export training predictions

Writes predictions to `train.csv` for the plotting cells.""",
            11: """## Evaluate on a single subject

Loads one subject's trial and compares predicted against measured angles.""",
            13: """## Plot predicted against measured

The visual check: does the predicted curve follow the shape of the real one?""",
            14: """## Per-sample error

Computes the squared error normalised by `Z` for each sample, so individual
poor predictions can be located rather than hidden inside an average.""",
        },
    },
    "08-gait-trajectory/notebooks/03_lstm.ipynb": {
        "intro": """# LSTM for gait trajectory prediction

The recurrent half of the gait study.

Same task as [`02_dense_network.ipynb`](02_dense_network.ipynb) — predict three
joint angles from three inputs — but with an LSTM rather than a dense stack.

## Why a recurrent model

Gait is periodic: each point in the cycle follows from the one before it. An
LSTM reads its input in order and maintains a memory across steps, so it can
represent that continuity. A dense network sees each sample in isolation.

The cost is shape handling: an LSTM expects a 3D tensor of
`(samples, timesteps, features)`, so the input is reshaped before training.

See the [README](../README.md) for how the two architectures compare.""",
        "sections": {
            2: """## Load and reshape

Reshapes the 2D input matrix into the 3D tensor the recurrent layers require.""",
            4: """## Build the LSTM

Stacked LSTM layers with `return_sequences=True` so each passes a full sequence
to the next, ending in a dense layer that maps to the three joint angles.""",
            5: """## Inspect the architecture

`model.summary()` shows the tensor shape flowing between layers and the
parameter count at each.""",
            6: """## Train

500 epochs at batch size 1, matching the dense network's budget.""",
            7: """## Predict and export

Writes the LSTM's predictions to `trainLSTM.csv`, kept separate from the dense
network's output so the two can be plotted together.""",
            8: """## Predict on held-out data

Runs the model over the separate test CSV.""",
            9: """## Evaluate on a single subject

The same single-trial check used for the dense network.""",
            10: """## Plot predicted against measured

The visual comparison of predicted and measured trajectories.""",
        },
    },
    "09-histology-color-normalization/notebooks/01_batch_colour_normalisation.ipynb": {
        "intro": """# Batch colour normalisation for histology images

Standardises the colour appearance of a batch of stained tissue images by
transferring the colour statistics of one reference image onto all the others.

## Why stain normalisation matters

Histology slides are stained with dyes such as haematoxylin and eosin. The
resulting colour varies considerably between laboratories, staining batches,
scanners, and even the age of the reagents — while the underlying tissue is
unchanged.

That variation is a problem for any downstream analysis: a model can end up
keying on which lab produced a slide rather than on the tissue itself.
Normalisation removes the appearance difference while preserving structure.

## The method — Reinhard colour transfer

1. Convert both images from BGR to the **LAB** colour space
2. Compute the mean and standard deviation of each LAB channel
3. Rescale the source image's channels to match the reference's statistics
4. Convert back to BGR

LAB is used rather than RGB because it separates lightness (L) from colour (A
and B), so the transfer can adjust colour without disturbing brightness
structure.

```
source image ──▶ LAB ──▶ match mean & std to reference ──▶ BGR ──▶ normalised
```

A tested implementation is in
[`src/color_normalization.py`](../src/color_normalization.py), and the
[README](../README.md) explains the workflow with diagrams.

> **Note.** This notebook and
> [`02_reinhard_experiments.ipynb`](02_reinhard_experiments.ipynb) are near
> identical; both are preserved as they were saved.""",
        "sections": {
            0: """## Install the colour-transfer package""",
            2: """## Mount storage and list the images

Written for Google Colab, so it mounts Drive and collects the image paths. The
image set is not committed to this repository — see
[`data/README.md`](../data/README.md).""",
            3: """## Resize helper

Scales images to a fixed width while preserving aspect ratio, so the batch is
consistent and manageable in memory.""",
            4: """## Channel statistics

Splits an image into its L, A and B channels and returns the mean and standard
deviation of each. These six numbers are the entire description of an image's
colour appearance that the transfer uses.""",
            5: """## Scaling helpers

After the transfer, values can fall outside the valid 0–255 range. These helpers
either clip them or rescale the whole array back into range.""",
            7: """## The colour transfer

Subtract the source's channel means, rescale by the ratio of standard
deviations, add the reference's means, then convert back to BGR.""",
            10: """## Normalise the batch

Applies the transfer to every image against the chosen reference and collects
the before/after pairs for comparison.""",
            11: """## HSV inspection

Converts images to HSV as an alternative view of colour distribution — a
cross-check that the normalisation behaved as intended.""",
            12: """## Images as arrays

Loads images into fixed-size NumPy arrays so they can be compared numerically
rather than by eye.""",
            13: """## Correlation between images

Computes and plots a Pearson correlation matrix across the image set. If
normalisation worked, images should agree more closely after it than before.""",
            15: """## Mutual information

Pearson correlation only detects **linear** relationships. Mutual information
captures any statistical dependence, linear or not, so it is a stricter check
that structure has been preserved through the transfer.""",
            17: """## Visualise the result

Plots the mutual-information matrix as a heatmap.""",
        },
    },
    "09-histology-color-normalization/notebooks/02_reinhard_experiments.ipynb": {
        "intro": """# Reinhard colour transfer — experiments

A near-duplicate of
[`01_batch_colour_normalisation.ipynb`](01_batch_colour_normalisation.ipynb),
preserved as it was saved.

Both apply Reinhard colour transfer to a batch of stained histology images:
convert to LAB, match each image's channel means and standard deviations to a
reference, convert back.

Read
[`01_batch_colour_normalisation.ipynb`](01_batch_colour_normalisation.ipynb)
for the fully annotated walkthrough, or the [README](../README.md) for the
method explained with diagrams.""",
        "sections": {
            2: """## Mount storage and list the images

Colab-specific setup. The image set is not committed here — see
[`data/README.md`](../data/README.md).""",
            4: """## Channel statistics

Mean and standard deviation of each LAB channel: the six numbers that describe
an image's colour appearance.""",
            7: """## The colour transfer

Rescales the source image's LAB channels to match the reference's statistics.""",
            10: """## Normalise the batch""",
            13: """## Correlation between images

Checks numerically whether the images agree more closely after normalisation.""",
        },
    },
    "09-histology-color-normalization/notebooks/03_scratch_experiments.ipynb": {
        "intro": """# Scratch experiments — image similarity metrics

Exploratory cells for comparing images numerically, kept alongside the main
normalisation notebooks.

Defines helpers that load images as arrays and measure agreement between them
with the **Pearson correlation coefficient**, used to check whether colour
normalisation brought a set of images into closer alignment.

The cells repeat with small variations — different resizing and different
directory handling — as the approach was worked out. For the full workflow see
[`01_batch_colour_normalisation.ipynb`](01_batch_colour_normalisation.ipynb)
and the [README](../README.md).""",
        "sections": {
            0: """## Load an image as an array and correlate two images""",
            1: """## Extend across a directory

The same comparison applied over a folder of images rather than a single pair.""",
            3: """## Resize before comparing

Correlation needs both arrays to be the same shape, so images are resized to a
common size first.""",
        },
    },
    "10-nifty-price-analysis/notebooks/02_scratch_experiments.ipynb": {
        "intro": """# Scratch experiments — NIFTY closing price

Exploratory notebook kept alongside the main
[`01_nifty_market_analysis.ipynb`](01_nifty_market_analysis.ipynb).

Builds a linear-regression baseline that predicts the NIFTY closing price from
the previous day's close and a 50-day moving average.

> **On the split used here.** These cells use `train_test_split`, which shuffles
> rows at random. For time series that lets the model train on future days and
> predict past ones, so the resulting score is optimistic. The main notebook
> uses a **chronological** split instead — training on the earliest data and
> testing on the most recent — which is the sound approach for forecasting. The
> difference between the two is discussed in the [README](../README.md).

Preserved as a record of the first attempt.""",
        "sections": {
            1: """## Load the data

Daily NIFTY index records: date, open, high, low, close, and volume.""",
            2: """## Parse dates and handle gaps

Converts the date column to datetime and deals with missing values so the
series is continuous.""",
            3: """## Plot the price history

The first thing to do with any time series: look at it. Trend, volatility, and
any obvious breaks in the data show up immediately.""",
            4: """## Build features

Two predictors:

| Feature | Meaning |
|---|---|
| `Prev_Close` | yesterday's closing price |
| `SMA_50` | the 50-day simple moving average |

Both are **lagged** — computed only from data available before the day being
predicted, which is what keeps them legitimate as forecasting inputs.""",
            5: """## Split into train and test""",
            6: """## Scale and fit

Standardises the features, then fits a linear regression.""",
            10: """## Evaluate

Reports mean squared error and R². Note that on a trending price series a high
R² is easy to achieve by predicting close to yesterday's value, so it should be
compared against that naive baseline rather than read on its own.""",
            11: """## Plot predicted against actual

Overlays the predicted series on the real one.""",
        },
    },
    "11-neural-net-experiments/notebooks/01_neural_network_experiments.ipynb": {
        "intro": """# Neural network experiments on synthetic data

A Keras sandbox comparing network architectures. The data is **generated at
random**, so this notebook is about the mechanics of building and training
networks rather than about solving a task.

## The data

```python
X = np.random.rand(1000, 1000)      # 1,000 samples, 1,000 random features
y = np.random.randint(2, size=1000) # independent random binary labels
```

The labels are independent of the features by construction, so there is no
relationship to learn. That makes the expected validation accuracy **50%**, and
gives a clean way to see what regularisation actually does: a model that scores
near chance is reporting honestly, while a model that scores much higher on the
training set is fitting noise.

## The two architectures

| | Simple | Regularised |
|---|---|---|
| Hidden layers | 1 (128 units) | 2 (256, 128 units) |
| Batch normalisation | — | after each hidden layer |
| Dropout | — | 0.5 after each hidden layer |
| Epochs | 10 | 20 |

## What is practised here

Building `Sequential` models, batch normalisation, dropout, train/validation
splitting, saving and reloading models, and mixed-precision training via
`LossScaleOptimizer`.

Full write-up in the [README](../README.md).""",
        "sections": {
            0: """## The simple network

One hidden layer of 128 units, then a sigmoid output.

`LossScaleOptimizer` enables **mixed precision**: most operations run in 16-bit
floating point instead of 32-bit, which halves memory use and speeds up training
on supported GPUs. It scales the loss upward before the backward pass so small
gradients do not underflow to zero in 16-bit.""",
            1: """## Evaluate the simple network

Reports loss and accuracy on the held-out split.""",
            2: """## Split the data

Holds back a portion of the random data for validation.""",
            3: """## The regularised network

Wider and deeper, with batch normalisation and dropout after each hidden layer.

**Batch normalisation** rescales each layer's outputs to a stable distribution,
keeping gradients well-behaved. **Dropout(0.5)** randomly zeroes half the
activations on each training pass, so the network cannot depend on any single
unit.""",
            4: """## Save the model

Writes the trained model in TensorFlow's SavedModel directory format.""",
            5: """## Repeat runs

The same architecture trained again — with random labels and no fixed seed,
repeated runs show how much of any apparent result is just variance.""",
            7: """## Write the generated data to CSV

Persists a generated dataset so the same random data can be reused across cells
rather than regenerated each time.""",
            10: """## A smaller input

Reduces the feature count from 1,000 to 100, which trains much faster while
demonstrating the same behaviour.""",
        },
    },
    "12-image-super-resolution/notebooks/01_esrgan_super_resolution.ipynb": {
        "intro": """# Image upscaling experiments

Explores increasing image resolution, working up from simple resampling toward
the ESRGAN super-resolution model.

## The two approaches

**Classical resampling** — `PIL.Image.resize` with a filter such as `LANCZOS`
computes each new pixel as a weighted average of nearby originals. It is fast
and predictable, but it cannot add detail that is not already present: enlarging
a blurry image with it produces a larger blurry image.

**Learned super-resolution (ESRGAN)** — a network trained on many
low/high-resolution pairs that has learned what fine texture typically looks
like, so it can synthesise plausible detail rather than interpolating between
existing pixels.

```
low-resolution input
   ├── resize()  ──▶ larger, same detail
   └── ESRGAN    ──▶ larger, synthesised detail
```

> **What "synthesised" means.** ESRGAN generates detail that is *plausible*, not
> detail that was recorded. The output should not be treated as recovering
> information the original image did not contain — which matters for any
> forensic, medical, or evidential use.

This notebook covers the resampling experiments. The ESRGAN architecture and
inference code are in [`src/`](../src/); pretrained weights are not included —
see the [README](../README.md).""",
        "sections": {
            0: """## Imports

PyTorch and torchvision for the ESRGAN work; Pillow for the resampling
experiments.""",
            2: """## Load an image

Sample images are in [`data/samples/`](../data/samples/).""",
            3: """## Upscale by resampling

Doubles the dimensions with `resize`. The result has four times the pixels but
no additional detail — each new pixel is interpolated from its neighbours.""",
            4: """## Sharpen after upscaling

`ImageEnhance.Sharpness` increases local contrast at edges, which makes a
resampled image *look* crisper. It does not recover detail; it emphasises what
the interpolation already produced.""",
            5: """## Inspect as an array

Converts to NumPy to check dimensions and pixel value ranges directly.""",
            6: """## Repeat on another image""",
            7: """## Upscale to 4K

Resamples to 3840×2160, the largest target here and the case where the limits of
interpolation are most visible.""",
        },
    },
}


def markdown_cell(text: str) -> dict:
    """Build a markdown cell carrying the idempotency marker."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": f"{text}\n\n{MARKER}".split("\n"),
    }


def already_annotated(notebook: dict) -> bool:
    """Return True when this tool has already run on the notebook."""
    return any(
        cell["cell_type"] == "markdown" and MARKER in "".join(cell["source"])
        for cell in notebook["cells"]
    )


def annotate(path: Path, spec: dict) -> str:
    """Insert the intro and section markdown into one notebook."""
    notebook = json.loads(path.read_text(encoding="utf-8"))

    if already_annotated(notebook):
        return "already annotated"

    original = notebook["cells"]
    annotated: list = [markdown_cell(spec["intro"])]
    sections = spec.get("sections", {})

    # Section keys index code cells in their original order, so empty and
    # non-code cells are not counted when matching.
    code_index = 0
    for cell in original:
        if cell["cell_type"] == "code":
            if code_index in sections:
                annotated.append(markdown_cell(sections[code_index]))
            code_index += 1
        # Drop empty cells; they carry no information and clutter the reader.
        if not "".join(cell.get("source", [])).strip():
            continue
        annotated.append(cell)

    notebook["cells"] = annotated
    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return (
        f"+1 intro, +{len(sections)} sections, "
        f"{len(original)} -> {len(annotated)} cells"
    )


def main() -> None:
    """Annotate every notebook named in SPECS."""
    missing = []
    for relative, spec in SPECS.items():
        path = REPO_ROOT / relative
        if not path.exists():
            missing.append(relative)
            continue
        print(f"{relative}: {annotate(path, spec)}")

    if missing:
        print("\nNot found:")
        for relative in missing:
            print(f"  {relative}")


if __name__ == "__main__":
    main()
