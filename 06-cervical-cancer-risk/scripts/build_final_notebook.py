"""Build the consolidated final notebook from the focused EDA and model notebooks.

The notebook keeps the original project's useful exploratory views while replacing
its leakage-prone preprocessing and accuracy-only evaluation with the reviewed
workflow. Run this script only when the source notebooks change.
"""

from pathlib import Path

import nbformat as nbf


PROJECT_DIR = Path(__file__).resolve().parents[1]
EDA_PATH = PROJECT_DIR / "01_exploratory_data_analysis.ipynb"
MODEL_PATH = PROJECT_DIR / "02_model_development.ipynb"
FINAL_PATH = PROJECT_DIR / "final_cervical_cancer_risk_modeling.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(source)


def code(source: str):
    return nbf.v4.new_code_cell(source)


def clean_copy(cell):
    """Copy a source cell without stale outputs from a previous execution."""
    if cell.cell_type == "markdown":
        return markdown(cell.source)
    return code(cell.source)


def main() -> None:
    eda = nbf.read(EDA_PATH, as_version=4)
    model = nbf.read(MODEL_PATH, as_version=4)
    cells = [
        markdown(
            "# Final Cervical Cancer Risk-Factor Modeling Project\n\n"
            "This consolidated notebook preserves the original project's useful "
            "exploration while applying the reviewed, leakage-aware modeling "
            "workflow. Run top-to-bottom for a reproducible result.\n\n"
            "> **Educational use only:** this work is not clinically validated and "
            "must not be used for diagnosis or medical decisions."
        ),
        markdown(
            "## What was preserved and what was improved\n\n"
            "The original notebook's raw-data preview, dataset summary, missingness "
            "inspection, correlation study, distributions, and XGBoost experiment "
            "are retained in this final version. Duplicate installation cells and "
            "unsafe steps were removed. In particular, preprocessing is now fitted "
            "inside pipelines, the split is stratified, diagnostic leakage features "
            "are excluded, and evaluation reports class-sensitive metrics rather "
            "than accuracy alone."
        ),
    ]

    # The focused EDA notebook is the source of truth for cleaning and reasoning.
    for cell in eda.cells:
        if cell.cell_type == "markdown" and cell.source.startswith("# Cervical Cancer Risk"):
            continue
        cells.append(clean_copy(cell))

    cells.extend(
        [
            markdown(
                "## Original exploration views, retained in a clearer form\n\n"
                "These cells preserve the original notebook's most useful inspection "
                "steps. They use the cleaned `data` table so they are consistent with "
                "the modern workflow, and the figures are sized for readability."
            ),
            code(
                "# Original raw-data inspection, retained for quick orientation.\n"
                "raw.tail(20)"
            ),
            code(
                "# Original schema and summary checks, retained before modeling.\n"
                "data.info()\n"
                "display(data.describe().T)"
            ),
            code(
                "# Original missingness heatmap, now applied after explicit cleaning.\n"
                "plt.figure(figsize=(14, 7))\n"
                "sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='magma')\n"
                "plt.title('Missingness pattern across patients and features')\n"
                "plt.xlabel('Features')\n"
                "plt.show()"
            ),
            code(
                "# Original correlation view, excluding diagnostic leakage columns.\n"
                "plt.figure(figsize=(16, 13))\n"
                "correlation_view = data[risk_features + [TARGET]].corr()\n"
                "sns.heatmap(correlation_view, cmap='vlag', center=0, square=False)\n"
                "plt.title('Correlation heatmap of retained risk factors and biopsy target')\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ),
            code(
                "# Original distribution overview, limited to retained features for readability.\n"
                "data[risk_features].hist(bins=12, figsize=(18, 18), color='#4C78A8')\n"
                "plt.suptitle('Distributions of retained risk-factor features', y=1.02)\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ),
            markdown(
                "# Model development and honest evaluation\n\n"
                "The next section reuses the cleaned EDA data. Its preprocessing, "
                "cross-validation, and final evaluation remain isolated from the holdout "
                "test data."
            ),
        ]
    )

    # Keep the specialized model imports, then replace its duplicate load/clean cell
    # with a lightweight bridge from the EDA data already in memory.
    for index, cell in enumerate(model.cells):
        if index == 0:  # duplicate notebook title
            continue
        if index == 3:  # duplicate data-loading explanation
            continue
        if index == 4:
            cells.extend(
                [
                    markdown(
                        "## Create the leakage-safe feature matrix\n\n"
                        "The EDA section has already loaded and cleaned `data`. We now "
                        "form the predictors and binary target without re-reading the CSV."
                    ),
                    code(
                        "X = data.drop(columns=[TARGET, *LEAKAGE_COLUMNS])\n"
                        "y = data[TARGET].astype(int)\n\n"
                        "assert not y.isna().any(), 'Target labels cannot be missing.'\n"
                        "assert set(y.unique()) == {0, 1}, 'This workflow expects a binary target.'\n\n"
                        "print(f'Features retained: {X.shape[1]}')\n"
                        "print(f'Positive class: {y.sum()} of {len(y)} ({y.mean():.2%})')"
                    ),
                ]
            )
            continue
        cells.append(clean_copy(cell))

    cells.extend(
        [
            markdown(
                "## Retained XGBoost comparison, rebuilt safely\n\n"
                "The original project used XGBoost. This version retains that model "
                "with the same leakage boundary, training-only imputation, class "
                "imbalance handling, and holdout metrics as the other models. Its "
                "parameters are intentionally moderate to reduce overfitting on this "
                "small dataset."
            ),
            code(
                "from xgboost import XGBClassifier\n\n"
                "# Weight positive cases by the training-set imbalance ratio.\n"
                "positive_weight = (y_train == 0).sum() / (y_train == 1).sum()\n\n"
                "xgboost_pipeline = Pipeline(\n"
                "    steps=[\n"
                "        ('preprocess', tree_preprocessor),\n"
                "        ('model', XGBClassifier(\n"
                "            n_estimators=200, max_depth=3, learning_rate=0.05,\n"
                "            subsample=0.8, colsample_bytree=0.8,\n"
                "            scale_pos_weight=positive_weight, eval_metric='logloss',\n"
                "            random_state=RANDOM_STATE, n_jobs=-1,\n"
                "        )),\n"
                "    ]\n"
                ")\n\n"
                "xgboost_pipeline.fit(X_train, y_train)\n"
                "xgb_probability = xgboost_pipeline.predict_proba(X_test)[:, 1]\n"
                "xgb_prediction = (xgb_probability >= 0.5).astype(int)\n\n"
                "xgb_results = pd.DataFrame([{\n"
                "    'accuracy': accuracy_score(y_test, xgb_prediction),\n"
                "    'precision': precision_score(y_test, xgb_prediction, zero_division=0),\n"
                "    'recall': recall_score(y_test, xgb_prediction, zero_division=0),\n"
                "    'f1': f1_score(y_test, xgb_prediction, zero_division=0),\n"
                "    'roc_auc': roc_auc_score(y_test, xgb_probability),\n"
                "    'pr_auc': average_precision_score(y_test, xgb_probability),\n"
                "}], index=['XGBoost'])\n"
                "display(xgb_results.round(3))"
            ),
            markdown(
                "### Final interpretation\n\n"
                "Compare models using the full metric table rather than selecting the "
                "highest accuracy. With only 11 positive cases in the holdout data, "
                "results have considerable uncertainty. The notebook demonstrates a "
                "careful experiment; it does not establish clinical usefulness."
            ),
        ]
    )

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {"name": "python", "version": "3.11"}
    nbf.write(notebook, FINAL_PATH)
    print(f"Wrote {FINAL_PATH.name} with {len(cells)} cells.")


if __name__ == "__main__":
    main()
