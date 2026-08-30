# Data

## `fake_or_real_news.csv` — not included

The source corpus is **not committed to this repository** and is not
redistributed here. It is the widely-circulated `fake_or_real_news.csv`
assembled by George McIntire, used in his 2017 Data Science Bootcamp fake-news
tutorial and mirrored in many teaching repositories since.

| Property | Value |
|---|---|
| Rows | 6,335 |
| Columns | `Unnamed: 0`, `title`, `text`, `label` |
| Labels | `FAKE`, `REAL` |
| Split used in 2022 | 5,068 train / 1,267 test (80/20, unseeded) |

Row and column counts are taken from the preserved notebook outputs, not from a
copy of the file.

### Why it is not committed

Two reasons. The licence and redistribution terms of the assembled corpus are
not clearly stated by any of its mirrors, and the articles within it are
third-party copyrighted news text. Committing it would propagate an unclear
rights situation into this repository.

### To run the code

Place a CSV with the columns `title`, `text`, and `label` at
`data/fake_or_real_news.csv`. `src/train_baseline.py` validates the schema
before training and fails with a clear message rather than downloading a
substitute or reporting a fabricated number:

```bash
python -m src.train_baseline --data data/fake_or_real_news.csv
```

Any dataset with those three columns works. The pipeline makes no assumption
that the labels are `FAKE`/`REAL` specifically.

### On the label itself

`label` is the value the dataset's author recorded. It is not a verified truth
value, and this project makes no fact-checking claim. See the "What this is
not" section of the [README](../README.md).

---

## `training_log_2022.csv` — included

The 20-epoch Keras history from the original 2022 training run, parsed directly
from the saved stdout of cell 39 of
[`notebooks/01_lstm_original_2022.ipynb`](../notebooks/01_lstm_original_2022.ipynb).

| Column | Meaning |
|---|---|
| `epoch` | 1–20 |
| `loss` | Training binary cross-entropy |
| `acc` | Training accuracy |
| `val_loss` | Validation binary cross-entropy |
| `val_acc` | Validation accuracy |

This is the only quantitative evidence that survives from the original run, and
every figure in the README is built from it. It is committed because it is 21
lines of numbers this project produced itself, with no third-party rights
attached.

**It is a transcript, not a re-run.** The model, its weights, and the split
that produced these numbers are all gone. Reproducing them exactly is
impossible — the 2022 split had no `random_state`.
