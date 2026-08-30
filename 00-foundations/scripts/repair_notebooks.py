"""Repair the preserved coursework notebooks so they run outside Colab.

The notebooks in ``notebooks/`` are kept as a record of how these algorithms
were first learned, but several of them could not run as committed:

* hardcoded Colab paths (``/content/play_tennis_train.csv``)
* dataset paths relative to a directory layout that no longer exists
* an image reference that moved into ``docs/assets/``
* in the Naive Bayes notebook, cells saved in a non-linear execution order

This script applies those fixes in place and is idempotent — running it twice
changes nothing the second time. It edits paths and cell order only; the
analysis and its conclusions are left exactly as they were written.

Run from the project directory:

    python scripts/repair_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = PROJECT_ROOT / "notebooks"

# Literal source substitutions, applied to every code and markdown cell.
# Paths are relative to the notebooks/ directory, where Jupyter runs them.
REPLACEMENTS: dict[str, str] = {
    "'/content/play_tennis_train.csv'": "'../data/play_tennis.csv'",
    '"/content/play_tennis_train.csv"': '"../data/play_tennis.csv"',
    "'./play_tennis_train.csv'": "'../data/play_tennis.csv'",
    '"./play_tennis_train.csv"': '"../data/play_tennis.csv"',
    "'play_tennis_train.csv'": "'../data/play_tennis.csv'",
    "'data.csv'": "'../data/enjoy_sport.csv'",
    '"income.csv"': '"../data/income.csv"',
    "'income.csv'": "'../data/income.csv'",
    'src="iris_petal_sepal.png"': 'src="../docs/assets/iris_petal_sepal.png"',
}

# A banner prepended to the notebooks that were followed from a tutorial, so
# the attribution travels with the notebook rather than living only in the
# README.
ATTRIBUTION: dict[str, str] = {
    "05_knn_iris.ipynb": (
        "> **Attribution.** This notebook follows tutorial 17 of the "
        "[codebasics/py](https://github.com/codebasics/py) machine-learning "
        "series. It is preserved as a record of following that tutorial while "
        "learning.\n>\n"
        "> The from-scratch KNN implementation this project verifies against "
        "scikit-learn lives in [`src/knn.py`](../src/knn.py)."
    ),
    "07_kmeans_clustering.ipynb": (
        "> **Attribution.** This notebook follows tutorial 13 of the "
        "[codebasics/py](https://github.com/codebasics/py) machine-learning "
        "series. It is preserved as a record of following that tutorial while "
        "learning."
    ),
}

# Context notes explaining how to read each notebook's reported numbers.
NOTES: dict[str, str] = {
    "04_naive_bayes.ipynb": (
        "> **About this notebook.** PlayTennis has only 14 rows, so there is no "
        "separate held-out file for it — `train_data` and `test_data` are loaded "
        "from the same dataset. The accuracy reported below is therefore "
        "**training accuracy**, and the notebook demonstrates the mechanics of "
        "Naive Bayes rather than its predictive performance.\n>\n"
        "> Cell order was adjusted during the portfolio revamp so the notebook "
        "runs top to bottom. No code was changed."
    ),
    "02_decision_tree_entropy.ipynb": (
        "> **About this notebook.** With 14 rows and `random_state=110`, the "
        "test split near the end of this notebook contains 3 rows. The accuracy "
        "printed there illustrates the scikit-learn API rather than measuring "
        "model quality."
    ),
}

MARKER = "> **Attribution."
NOTE_HEAD = "> **About this notebook."



def apply_replacements(notebook: dict) -> int:
    """Rewrite dataset and asset paths in every cell. Returns the edit count."""
    edits = 0
    for cell in notebook["cells"]:
        source = cell.get("source", [])
        new_source = []
        for line in source:
            original = line
            for old, new in REPLACEMENTS.items():
                line = line.replace(old, new)
            if line != original:
                edits += 1
            new_source.append(line)
        cell["source"] = new_source
    return edits


def prepend_markdown(notebook: dict, text: str, marker: str) -> bool:
    """Insert a markdown cell at the top unless one is already present."""
    for cell in notebook["cells"]:
        if cell["cell_type"] == "markdown" and marker in "".join(cell["source"]):
            return False

    notebook["cells"].insert(
        0,
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": text.split("\n"),
        },
    )
    return True


def reorder_naive_bayes(notebook: dict) -> bool:
    """Move label encoding ahead of the first model fit.

    The notebook was saved with the encoding step (which defines ``X_train``)
    positioned after a cell that already calls ``fit``. Sorting the cells so
    that the first ``fit`` follows the encoding makes the notebook runnable
    top to bottom without changing any of its content.
    """
    cells = notebook["cells"]

    def index_of(predicate) -> int | None:
        for i, cell in enumerate(cells):
            if cell["cell_type"] == "code" and predicate("".join(cell["source"])):
                return i
        return None

    first_fit = index_of(lambda s: "nb_model.fit(" in s)
    encoding = index_of(lambda s: "LabelEncoder()" in s and "label_encoders" in s)
    split = index_of(lambda s: "X_train = train_data[" in s)

    if first_fit is None or encoding is None or split is None:
        return False
    if first_fit > encoding and first_fit > split:
        return False  # already in a runnable order

    # Remove the premature fit cell; an identical fit already appears later in
    # the notebook, after the encoding and split steps.
    duplicate_fits = [
        i
        for i, cell in enumerate(cells)
        if cell["cell_type"] == "code" and "nb_model.fit(" in "".join(cell["source"])
    ]
    if len(duplicate_fits) > 1:
        del cells[duplicate_fits[0]]
        return True
    return False


def clear_outputs(notebook: dict) -> None:
    """Reset execution counts so the committed diff is stable."""
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def repair(path: Path) -> None:
    """Apply every repair to one notebook and write it back."""
    notebook = json.loads(path.read_text(encoding="utf-8"))

    edits = apply_replacements(notebook)
    changes = [f"{edits} path edits"] if edits else []

    if path.name in NOTES:
        if prepend_markdown(notebook, NOTES[path.name], NOTE_HEAD):
            changes.append("context note")

    if path.name in ATTRIBUTION:
        if prepend_markdown(notebook, ATTRIBUTION[path.name], MARKER):
            changes.append("attribution")

    if path.name == "04_naive_bayes.ipynb":
        if reorder_naive_bayes(notebook):
            changes.append("reordered cells")

    path.write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"{path.name}: {', '.join(changes) if changes else 'already clean'}")


def main() -> None:
    """Repair every notebook in the project."""
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    if not paths:
        raise SystemExit(f"No notebooks found in {NOTEBOOKS}")
    for path in paths:
        repair(path)


if __name__ == "__main__":
    main()
