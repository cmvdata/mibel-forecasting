"""Seasonal naive benchmark.

For day-ahead electricity prices, Lago et al. (2021) recommend a weekly
seasonal naive baseline: the forecast for hour *h* of day *d* is the
realised price at hour *h* of day *d − 7*. This captures both the daily
profile and the day-of-week effect (weekday/weekend) without any
parameters, and it is the first line of defence against more elaborate
models that do not actually beat trivial seasonality.
"""

from __future__ import annotations

import pandas as pd


class SeasonalNaive:
    """Weekly seasonal naive: y_hat[t] = y[t - lag_days · 24h].

    Parameters
    ----------
    target_col
        Column in the training/test DataFrame that holds the target.
    lag_days
        Lag in days. Default 7 (Lago 2021 canonical benchmark).
    """

    def __init__(self, *, target_col: str, lag_days: int = 7) -> None:
        if lag_days < 1:
            raise ValueError("lag_days must be >= 1")
        self.target_col = target_col
        self.lag_days = lag_days
        self._history: pd.Series | None = None

    def fit(self, train_df: pd.DataFrame) -> "SeasonalNaive":
        if self.target_col not in train_df.columns:
            raise KeyError(f"target_col {self.target_col!r} missing from train_df")
        self._history = train_df[self.target_col].copy()
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        if self._history is None:
            raise RuntimeError("call fit() before predict()")
        lookup = test_df.index - pd.Timedelta(days=self.lag_days)
        values = self._history.reindex(lookup).to_numpy()
        return pd.Series(values, index=test_df.index, name="y_pred")
