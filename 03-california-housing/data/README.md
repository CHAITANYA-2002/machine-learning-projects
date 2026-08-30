# Data boundary

This project intentionally does not commit a copied California Housing dataset. The runnable notebooks and `src/train.py` call scikit-learn's `fetch_california_housing(as_frame=True)`, which downloads the maintained source on first use and caches it locally.

## What is fetched

The dataset contains 20,640 California census-district rows with eight numeric, district-level predictors and a target named `MedHouseVal`. The target is historical (1990 census-derived) and expressed in units of $100,000. It is capped at 5.00001, so values at the top end are censored.

## Reproducibility

Run the project from its root after installing `requirements.txt`:

```powershell
python src/train.py
```

The first run needs network access unless the dataset is already in scikit-learn's cache. Do not substitute a scraped, current, or property-level file and present the resulting score as equivalent evidence.

## Repository rule

Do not add the downloaded cache, personal property records, addresses, or any raw extract to Git. If the input changes, document its source, date, licence, column semantics, target construction, missing-data policy, and split strategy before comparing results with this baseline.
