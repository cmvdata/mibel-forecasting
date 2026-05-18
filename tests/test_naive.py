from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mibel_forecasting.models.naive import SeasonalNaive


def _weekly_panel() -> pd.DataFrame:
    # 4 weeks of synthetic weekly-periodic data: y[t] = sin(2π · hour_of_week / 168)
    idx = pd.date_range("2024-01-01", "2024-01-28 23:00", freq="h", tz="Europe/Madrid")
    hour_of_week = (idx.dayofweek * 24 + idx.hour).to_numpy()
    y = np.sin(2 * np.pi * hour_of_week / 168.0)
    return pd.DataFrame({"price_es": y}, index=idx)


def test_naive_perfect_on_weekly_periodic_signal():
    df = _weekly_panel()
    train, test = df.iloc[:-24], df.iloc[-24:]
    model = SeasonalNaive(target_col="price_es").fit(train)
    pred = model.predict(test)
    np.testing.assert_allclose(pred.to_numpy(), test["price_es"].to_numpy(), atol=1e-12)


def test_naive_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        SeasonalNaive(target_col="price_es").predict(_weekly_panel().iloc[-24:])


def test_naive_unknown_target_col_raises():
    with pytest.raises(KeyError):
        SeasonalNaive(target_col="nope").fit(_weekly_panel())


def test_naive_lag_days_is_respected():
    df = _weekly_panel()
    train, test = df.iloc[:-24], df.iloc[-24:]
    model = SeasonalNaive(target_col="price_es", lag_days=1).fit(train)
    pred = model.predict(test)
    expected = train["price_es"].iloc[-24:]
    np.testing.assert_allclose(pred.to_numpy(), expected.to_numpy(), atol=1e-12)


def test_naive_missing_lookup_returns_nan():
    df = _weekly_panel()
    # Train window too short for a 7-day lag on the test
    train, test = df.iloc[:24], df.iloc[24:48]
    model = SeasonalNaive(target_col="price_es").fit(train)
    pred = model.predict(test)
    assert pred.isna().all()
