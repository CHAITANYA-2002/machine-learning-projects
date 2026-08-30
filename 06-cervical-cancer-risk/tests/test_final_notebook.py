"""Regression checks for the portfolio notebook's structure and safety rules."""

from pathlib import Path
import json


PROJECT_DIR = Path(__file__).resolve().parents[1]
FINAL_NOTEBOOK = PROJECT_DIR / "notebooks" / "03_final_modeling.ipynb"


def _sources(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_final_notebook_exists_and_has_explanatory_markdown():
    assert FINAL_NOTEBOOK.exists()
    notebook = json.loads(FINAL_NOTEBOOK.read_text(encoding="utf-8"))
    assert sum(cell["cell_type"] == "markdown" for cell in notebook["cells"]) >= 20


def test_final_notebook_preserves_legacy_analysis_and_safe_modeling():
    notebook = json.loads(FINAL_NOTEBOOK.read_text(encoding="utf-8"))
    source = _sources(notebook)

    # Original notebook strengths retained.
    for required in ("raw.tail(20)", "describe()", "sns.heatmap", "hist(", "XGBClassifier"):
        assert required in source

    # Safety improvements required in the replacement workflow.
    for leakage_column in ("Hinselmann", "Schiller", "Citology", "Dx:Cancer"):
        assert leakage_column in source
    assert "stratify=y" in source
    assert "Pipeline(" in source
    assert "average_precision_score" in source


def test_final_notebook_has_no_saved_execution_errors():
    notebook = json.loads(FINAL_NOTEBOOK.read_text(encoding="utf-8"))
    errors = [
        output
        for cell in notebook["cells"]
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors
