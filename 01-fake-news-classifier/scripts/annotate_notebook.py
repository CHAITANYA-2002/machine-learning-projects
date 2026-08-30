"""Add explanatory markdown and inline comments to the 2022 notebook.

The notebook was written as 45 consecutive code cells with no markdown and no
comments. This script annotates it so a reader can follow what each stage does
without reading the code line by line:

* section headings and plain-English explanation before each stage
* short comments inside code cells naming what the line produces
* a closing summary of the results

Two rules govern what this script may do:

1. **No code is changed.** Every original code line survives verbatim.
2. **No output is cleared.** The saved outputs are what
   ``data/training_log_2022.csv`` is transcribed from.

The script is idempotent -- a marker on each inserted cell means a second run
changes nothing.

Run from the project directory:

    python scripts/annotate_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = PROJECT_ROOT / "notebooks" / "01_lstm_original_2022.ipynb"

# Identifies cells this script inserted, so re-running is a no-op.
MARKER = "<!-- annotated -->"

# Markdown inserted *before* the original code cell at each index.
SECTIONS: dict[int, str] = {
    0: """# Fake News Classifier — LSTM on news headlines

Classifies news articles as `FAKE` or `REAL` using a recurrent neural network
built in Keras.

This notebook is from October 2022 and is preserved as it was written. Markdown
and comments were added later so the workflow can be followed end to end. The
full write-up, with diagrams, is in the [README](../README.md).

## What happens here

| Stage | Cells | What it produces |
|---|---|---|
| Load the data | 1–6 | A DataFrame of 6,335 news articles |
| Encode the label | 7–9 | A binary target array |
| Clean the text | 10–16 | Normalised headline strings |
| Convert to numbers | 17–23 | A padded integer matrix, 6,335 × 58 |
| Build the network | 24–31 | An 11.2M-parameter Embedding → LSTM → Dense model |
| Split and train | 32–39 | A model trained for 20 epochs |
| Review the results | 40–41 | Accuracy and loss curves |

## The task

Given the headline of a news article, predict which of the dataset's two label
values it carries. This is text classification against a dataset's own label
column — it is not fact-checking, and the label reflects how the dataset was
assembled rather than any verified truth value.

## Running it

The source corpus is not included in this repository — see
[`data/README.md`](../data/README.md). The saved outputs below are from the
original run, so the workflow can be read through without it.

---

## 1 · Imports

NumPy for arrays, pandas for the table.""",

    1: """## 2 · Load the dataset

`fake_or_real_news.csv` holds 6,335 articles with four columns:

| Column | Contents |
|---|---|
| `Unnamed: 0` | The original CSV row index — not a feature |
| `title` | The article headline |
| `text` | The article body |
| `label` | `FAKE` or `REAL` |

The next few cells drop the index column and inspect what is left.""",

    5: """## 3 · Combine the headline and body

Builds a `comb` column joining each headline to its article body, so a single
text field carries both.""",

    7: """## 4 · Encode the label

The network needs a number, not the strings `FAKE` and `REAL`.

`pd.get_dummies` turns the label column into two indicator columns, ordered
alphabetically as `['FAKE', 'REAL']`. Keeping column 1 gives a single binary
target where **`1` means REAL and `0` means FAKE**.""",

    10: """## 5 · Select and normalise the text

Takes the headline column as the text to model, then replaces every
non-alphabetic character with a space. That removes punctuation, symbols, and
digits, leaving words separated by whitespace.""",

    14: """## 6 · Clean the text

Runs each headline through NLTK's Porter stemmer and stop-word list, character
by character, lowercasing as it goes and dropping anything that appears in the
English stop-word list.

Because that list includes eight single-character entries (`a d i m o s t y`),
the step also compresses each headline into a shorter form while keeping word
boundaries intact:

```
"You Can Smell Hillary s Fear"   →   "yu cn sell hllr  fer"
"Kerry to go to Paris..."        →   "kerr  g  pr n geure f ph"
```

This is reproduced as `legacy_char_clean()` in
[`src/preprocessing.py`](../src/preprocessing.py), so the transformation can
still be applied and inspected today.""",

    17: """## 7 · Turn words into numbers

A network cannot read strings, so each word becomes an integer.

Keras `one_hot` hashes every word to a number in the range `[1, vocab_size)`.
Despite the name it does not build a one-hot vector — it returns one integer per
word. With `vocab_size=100000` the embedding layer will hold a 100,000-row
lookup table.""",

    21: """### Find the sequence length

Sequences fed to an LSTM must all be the same length. This measures the longest
headline in the corpus — **58 tokens** — which becomes the padded width for
every row.""",

    24: """## 8 · Build the network

Four layers:

| Layer | Role |
|---|---|
| `Embedding(100000, 100)` | Maps each word integer to a 100-dimensional vector the model learns |
| `Dropout(0.5)` | Randomly zeroes half the values during training, to reduce overfitting |
| `LSTM(500)` | Reads the sequence in order and produces one 500-value summary |
| `Dense(1, sigmoid)` | Squashes that summary to a single probability between 0 and 1 |

The embedding table dominates the parameter count: 100,000 × 100 = **10 million**
of the model's 11.2 million weights.""",

    31: """### Compile

Sets the loss to binary cross-entropy, the standard choice for a two-class
problem with a sigmoid output, and tracks accuracy during training.""",

    33: """## 9 · Split into training and test sets

80% of the rows train the model; the remaining 20% are held back to measure it.

That gives **5,068 training** and **1,267 test** articles.""",

    39: """## 10 · Train

20 passes over the training set, 64 articles at a time, measuring accuracy and
loss on the held-out set after each pass.

The output below is the original 2022 run. It is transcribed to
[`data/training_log_2022.csv`](../data/training_log_2022.csv) and charted in the
[README](../README.md).""",

    40: """## 11 · Plot the training history

Charts accuracy and loss, for both the training and held-out sets, across all
20 epochs.""",
}

# Comments prepended inside the original code cell at each index.
COMMENTS: dict[int, str] = {
    1: "# 6,335 news articles. Not in this repository -- see data/README.md.",
    3: "# 'Unnamed: 0' is the original CSV row index, not a feature.",
    5: "# Join headline and body into one text field.",
    7: "# get_dummies orders columns alphabetically: ['FAKE', 'REAL'].",
    8: "# Keep column 1, so y == 1 means REAL and y == 0 means FAKE.",
    10: "# Use the headline column as the text to model.",
    12: "# Replace every non-letter with a space (removes punctuation and digits).",
    15: "# Lowercase, stem, and drop stop-word characters from each headline.",
    17: "# Keras one_hot hashes each word to an integer; it is not a one-hot vector.",
    18: "# Each headline becomes a list of integer word IDs.",
    21: "# Longest headline in the corpus, used as the padded sequence width.",
    22: "# 'pre' padding puts zeros at the front, so sequences end on real tokens.",
    26: "# 100,000 words x 100 dimensions = 10,000,000 parameters.",
    28: "# 500 LSTM units reading the 58-token sequence.",
    29: "# Single sigmoid output: probability the article is REAL.",
    31: "# Binary cross-entropy is the standard loss for two-class problems.",
    36: "# 80/20 split -> 5,068 training rows, 1,267 test rows.",
    39: "# 20 epochs, batch size 64. Output below is the original 2022 run.",
}

CLOSING = """---

## 12 · Results

| Metric | Epoch 1 | Best | Epoch 20 |
|---|---|---|---|
| Training accuracy | 0.6638 | 0.9830 | 0.9830 |
| Validation accuracy | 0.7727 | **0.8287** (epoch 5) | 0.8043 |
| Training loss | 0.7353 | 0.0483 | 0.0483 |
| Validation loss | 0.5010 | **0.4159** (epoch 4) | 0.8080 |

The model reached **82.9% accuracy** on held-out articles at its best epoch,
against a roughly balanced two-class problem where guessing would give about
50%.

Training accuracy continues climbing to 0.9830 while validation accuracy settles
around 0.80 — the network fits the training set more closely than it
generalises, which is expected for 11.2 million parameters learning from 5,068
examples.

## What is in the rest of this project

| Path | What it holds |
|---|---|
| [`README.md`](../README.md) | The full walkthrough, with diagrams |
| [`src/preprocessing.py`](../src/preprocessing.py) | The cleaning steps, importable and tested |
| [`src/fake_news_model.py`](../src/fake_news_model.py) | A TF-IDF + logistic-regression pipeline |
| [`src/train_baseline.py`](../src/train_baseline.py) | Command-line training entry point |
| [`data/training_log_2022.csv`](../data/training_log_2022.csv) | The recovered 20-epoch history |

## Scope

Predicting a dataset's `label` column is text classification, not fact-checking.
The label is whatever the dataset's author recorded. Nothing here should be used
to judge the truth of an article, the credibility of a publisher, or the honesty
of a person.
"""


def markdown_cell(text: str) -> dict:
    """Build a markdown cell carrying the idempotency marker."""
    body = f"{text}\n\n{MARKER}"
    return {"cell_type": "markdown", "metadata": {}, "source": body.split("\n")}


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
    """Remove empty code cells from the end of the notebook."""
    removed = 0
    while (
        cells
        and cells[-1]["cell_type"] == "code"
        and not "".join(cells[-1].get("source", [])).strip()
    ):
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
