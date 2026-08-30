# %% [markdown]
# # Final NIFTY Closing-Price Analysis
#
# This final notebook preserves the original project's trend and moving-average ideas
# while correcting its validation: future dates are never mixed into training data.
# This is educational market analysis, not investment advice.

# %% [markdown]
# ## 1. Load and inspect the included data
#
# Date ordering is the key integrity requirement for time-series work, so we parse
# dates and inspect the timeline before creating any forecast features.

# %%
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nifty_model import FEATURE_COLUMNS, chronological_split, create_features, evaluate_forecast

raw = pd.read_csv(Path("NSEI.csv"))
raw["Date"] = pd.to_datetime(raw["Date"])
print(f"Rows: {len(raw):,}; period: {raw.Date.min():%Y-%m-%d} to {raw.Date.max():%Y-%m-%d}")
raw.head()

# %% [markdown]
# ## 2. Original trend-analysis idea, retained
#
# The original notebook visualized closing-price movement. This version retains that
# useful view and makes the date axis and data scope explicit.

# %%
plt.figure(figsize=(13, 5))
plt.plot(raw["Date"], raw["Close"], color="#174ea6", linewidth=1)
plt.title("NIFTY closing price over the included historical period")
plt.xlabel("Date")
plt.ylabel("Close (index points)")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Create features without peeking ahead
#
# Every feature is shifted by one day. The close on day *t* is the target; neither
# it nor a moving average containing it is allowed into that row's inputs.

# %%
featured = create_features(raw)
featured[["Date", *FEATURE_COLUMNS, "target_close"]].head()

# %% [markdown]
# ## 4. Chronological holdout evaluation
#
# The earliest 80% of observations train the model. The latest 20% test it. This
# mimics how a forecast is made in practice: the future is unavailable at training.

# %%
train, test = chronological_split(featured, test_fraction=0.2)
model = Pipeline([("scale", StandardScaler()), ("model", LinearRegression())])
model.fit(train[FEATURE_COLUMNS], train["target_close"])
predictions = model.predict(test[FEATURE_COLUMNS])
metrics = evaluate_forecast(test["target_close"], predictions)
pd.DataFrame([metrics], index=["Chronological holdout"]).round(3)

# %% [markdown]
# ## 5. Actual versus predicted prices
#
# This plot shows exactly where the model follows or misses holdout movements. A low
# percentage error does not establish a tradable strategy; it is only a historical
# point-forecast measurement.

# %%
plt.figure(figsize=(13, 5))
plt.plot(test["Date"], test["target_close"], label="Actual close", color="#174ea6")
plt.plot(test["Date"], predictions, label="Predicted close", color="#e45756", alpha=.85)
plt.title("Chronological holdout: actual versus predicted NIFTY close")
plt.xlabel("Date")
plt.ylabel("Close (index points)")
plt.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Limits and next steps
#
# The included data ends in September 2023. The model omits news, macroeconomic
# events, transaction costs, risk management, and regime changes. Future work should
# use walk-forward validation, compare naive persistence, and evaluate an entire
# strategy—not just one-day price error.
