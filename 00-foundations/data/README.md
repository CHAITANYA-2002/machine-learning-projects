# Datasets

Three small, classical teaching datasets. All are committed to this repository
because each is a few hundred bytes and has no licensing or privacy constraint —
unlike the image datasets used elsewhere in this portfolio.

| File | Rows | Source | Used by |
|---|---|---|---|
| `enjoy_sport.csv` | 4 | Mitchell, *Machine Learning* (McGraw-Hill, 1997), ch. 2 | Find-S |
| `play_tennis.csv` | 14 | Quinlan, "Induction of Decision Trees", *Machine Learning* 1(1), 1986 | Decision trees, Naive Bayes |
| `income.csv` | 22 | [codebasics/py](https://github.com/codebasics/py) ML tutorial series | k-means |

Iris is not stored here. It ships with scikit-learn and is loaded with
`sklearn.datasets.load_iris()`, so committing a copy would only risk drift.

---

## `enjoy_sport.csv`

Mitchell's EnjoySport example, the standard vehicle for demonstrating concept
learning. Six nominal attributes describing a day, and a boolean target for
whether a water-sport enthusiast enjoys it.

```
sky, air temp, humidity, wind, water, forecast -> enjoy sport
```

Three positive examples and one negative. Deliberately tiny: the point is to
follow the hypothesis by hand, not to measure accuracy.

**Note on the negative example.** Find-S never reads it. Row 3
(`rainy, cold, high, strong, warm, change -> no`) exists so you can verify that
removing it changes nothing about the learned hypothesis — which is the
algorithm's central weakness.

## `play_tennis.csv`

Quinlan's PlayTennis dataset, the worked example behind ID3. Fourteen days
described by four nominal weather attributes, with a boolean target.

```
outlook, temp, humidity, wind -> play
```

Class balance is 9 `Yes` / 5 `No`, which gives the textbook root entropy of
**0.9403 bits**. Splitting on `outlook` yields the highest information gain,
**0.2467 bits** — the number every ID3 walkthrough reproduces, and the value
asserted in `tests/test_impurity.py`.

**The `day` column is an identifier, not a feature.** `load_play_tennis()`
drops it. Leaving it in gives a decision tree fourteen unique values to split
on, which produces a perfect-looking tree that has memorised the row labels and
generalises to nothing. This is the smallest possible demonstration of leakage.

**There is no test split.** The original notebook referenced a
`play_tennis_test.csv` that was never present in this repository; that reference
was a bug and has been removed. Fourteen rows cannot support a meaningful
held-out split, so the accuracy figures reported for this dataset are training
accuracy and are labelled as such throughout.

## `income.csv`

Twenty-two name/age/income records used to demonstrate k-means and, more
importantly, why feature scaling is not optional for distance-based methods.

```
Name, Age, Income($)
```

Income spans roughly 45,000–162,000 while age spans 26–43. Euclidean distance is
therefore almost entirely determined by income, and unscaled k-means recovers
horizontal income bands rather than the age/income groups visible to the eye.
`docs/assets/kmeans_scaling.png` shows both fits side by side.

**`Name` is not a feature.** It is retained only for labelling plots.
