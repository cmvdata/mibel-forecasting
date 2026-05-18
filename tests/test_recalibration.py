from __future__ import annotations

import numpy as np
import pandas as pd

from mibel_forecasting.evaluation.recalibration import rolling_forecast
from mibel_forecasting.models.naive import SeasonalNaive


def _periodic_panel(days: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h", tz="Europe/Madrid")
    hour_of_week = (idx.dayofweek * 24 + idx.hour).to_numpy()
    y = np.sin(2 * np.pi * hour_of_week / 168.0)
    return pd.DataFrame({"price_es": y}, index=idx)


def test_rolling_forecast_naive_on_periodic_is_exact():
    df = _periodic_panel(days=30)
    out = rolling_forecast(
        df,
        target_col="price_es",
        model_factory=lambda: SeasonalNaive(target_col="price_es"),
        train_size="14D",
        test_size="1D",
        step="1D",
        test_start="2024-01-15",
        test_end="2024-01-20",
    )
    assert len(out) == 6 * 24
    np.testing.assert_allclose(out["y_pred"].to_numpy(), out["y_true"].to_numpy(), atol=1e-12)


def test_rolling_forecast_empty_when_window_does_not_fit():
    df = _periodic_panel(days=5)
    out = rolling_forecast(
        df,
        target_col="price_es",
        model_factory=lambda: SeasonalNaive(target_col="price_es"),
        train_size="365D",
        test_size="1D",
        test_start="2024-01-02",
        test_end="2024-01-03",
    )
    assert len(out) == 0
    assert list(out.columns) == ["y_true", "y_pred"]


def test_rolling_forecast_columns_and_index():
    df = _periodic_panel(days=21)
    out = rolling_forecast(
        df,
        target_col="price_es",
        model_factory=lambda: SeasonalNaive(target_col="price_es"),
        train_size="10D",
        test_start="2024-01-15",
        test_end="2024-01-15",
    )
    assert list(out.columns) == ["y_true", "y_pred"]
    assert isinstance(out.index, pd.DatetimeIndex)
    assert out.index.tz is not None
