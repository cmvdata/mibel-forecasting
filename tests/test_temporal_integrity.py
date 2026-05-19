"""Temporal-integrity guarantees for UTC DAM panels and rolling evaluation.

Hardens the rolling-forecast pipeline against three classes of bug
that would silently corrupt downstream metrics:

- non-24-hour test days (calendar / DST / partial-window edge cases),
- duplicated or missing hourly timestamps in the concatenated output,
- non-determinism across runs with bit-identical inputs.

The loader publishes UTC by default since commit ``b38b456``, so DST
transitions in Europe/Madrid translate to plain 24-hour UTC days and
should be no-ops here. The DST tests below guard against any future
timezone refactor that would re-introduce 23-/25-hour panes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mibel_forecasting.evaluation.recalibration import rolling_forecast
from mibel_forecasting.models.lear import LEAR
from mibel_forecasting.models.naive import SeasonalNaive


def _utc_panel(*, start: str, end: str, with_exog: bool = False) -> pd.DataFrame:
    """Synthetic hourly UTC panel with weekly-periodic price (and optional exog)."""
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    hour_of_week = (idx.dayofweek * 24 + idx.hour).to_numpy()
    price = 30.0 + 20.0 * np.sin(2 * np.pi * hour_of_week / 168.0)
    out = pd.DataFrame({"price_es": price}, index=idx)
    if with_exog:
        out["es_demand_fc"] = 28000 + 4000 * np.sin(2 * np.pi * hour_of_week / 24.0)
        out["es_wind_fc"] = 3000 + 2000 * np.cos(2 * np.pi * hour_of_week / 24.0)
    return out


def test_single_day_forecast_has_exactly_24_hours():
    df = _utc_panel(start="2024-01-01", end="2024-02-15 23:00")
    out = rolling_forecast(
        df,
        target_col="price_es",
        model_factory=lambda: SeasonalNaive(target_col="price_es"),
        train_size="14D",
        test_size="1D",
        test_start="2024-01-20",
        test_end="2024-01-20",
    )
    assert len(out) == 24
    counts = out.index.hour.value_counts().sort_index()
    assert list(counts.index) == list(range(24))
    assert (counts == 1).all()


def test_multi_day_forecast_has_no_duplicated_timestamps():
    df = _utc_panel(start="2024-01-01", end="2024-02-15 23:00")
    out = rolling_forecast(
        df,
        target_col="price_es",
        model_factory=lambda: SeasonalNaive(target_col="price_es"),
        train_size="14D",
        test_size="1D",
        test_start="2024-01-20",
        test_end="2024-01-25",
    )
    assert not out.index.duplicated().any()
    assert len(out) == 6 * 24


def test_multi_day_forecast_has_no_hourly_gaps():
    df = _utc_panel(start="2024-01-01", end="2024-02-15 23:00")
    out = rolling_forecast(
        df,
        target_col="price_es",
        model_factory=lambda: SeasonalNaive(target_col="price_es"),
        train_size="14D",
        test_size="1D",
        test_start="2024-01-20",
        test_end="2024-01-25",
    )
    expected = pd.date_range("2024-01-20", "2024-01-25 23:00", freq="h", tz="UTC")
    pd.testing.assert_index_equal(out.index, expected)


def test_dst_spring_forward_day_remains_24_hours_in_utc():
    # Europe/Madrid spring-forward: 2024-03-31. Under UTC the day has 24h.
    df = _utc_panel(start="2024-03-01", end="2024-04-15 23:00")
    out = rolling_forecast(
        df,
        target_col="price_es",
        model_factory=lambda: SeasonalNaive(target_col="price_es"),
        train_size="14D",
        test_size="1D",
        test_start="2024-03-31",
        test_end="2024-03-31",
    )
    assert len(out) == 24
    assert out["y_pred"].notna().all()


def test_dst_fall_back_day_remains_24_hours_in_utc():
    # Europe/Madrid fall-back: 2024-10-27. Under UTC the day has 24h.
    df = _utc_panel(start="2024-10-01", end="2024-11-15 23:00")
    out = rolling_forecast(
        df,
        target_col="price_es",
        model_factory=lambda: SeasonalNaive(target_col="price_es"),
        train_size="14D",
        test_size="1D",
        test_start="2024-10-27",
        test_end="2024-10-27",
    )
    assert len(out) == 24
    assert out["y_pred"].notna().all()


def test_window_spanning_both_dst_transitions_has_clean_grid():
    # Same year, spring + fall DST inside the window.
    df = _utc_panel(start="2024-03-01", end="2024-11-15 23:00")
    out = rolling_forecast(
        df,
        target_col="price_es",
        model_factory=lambda: SeasonalNaive(target_col="price_es"),
        train_size="14D",
        test_size="1D",
        test_start="2024-03-31",
        test_end="2024-10-27",
    )
    expected = pd.date_range("2024-03-31", "2024-10-27 23:00", freq="h", tz="UTC")
    pd.testing.assert_index_equal(out.index, expected)
    assert not out.index.duplicated().any()


def test_rolling_forecast_is_deterministic_with_naive():
    df = _utc_panel(start="2024-01-01", end="2024-03-01 23:00")
    kwargs = dict(
        target_col="price_es",
        model_factory=lambda: SeasonalNaive(target_col="price_es"),
        train_size="14D",
        test_size="1D",
        test_start="2024-02-01",
        test_end="2024-02-14",
    )
    out1 = rolling_forecast(df, **kwargs)
    out2 = rolling_forecast(df, **kwargs)
    pd.testing.assert_frame_equal(out1, out2)


def test_rolling_forecast_is_deterministic_with_lear():
    """LEAR + Lasso is deterministic; two independent runs must agree exactly,
    including the derived rMAE figure used by the robustness notebook."""
    df = _utc_panel(start="2024-01-01", end="2024-04-01 23:00", with_exog=True)
    kwargs = dict(
        target_col="price_es",
        model_factory=lambda: LEAR(
            target_col="price_es",
            exogenous_cols=("es_demand_fc", "es_wind_fc"),
        ),
        train_size="45D",
        test_size="1D",
        test_start="2024-02-20",
        test_end="2024-02-22",
    )
    out1 = rolling_forecast(df, **kwargs)
    out2 = rolling_forecast(df, **kwargs)
    pd.testing.assert_frame_equal(out1, out2)

    # Same predictions ⇒ identical MAE ⇒ identical rMAE. Guard the metric
    # too in case the assert above is ever relaxed to a tolerance.
    mae1 = float(np.mean(np.abs(out1["y_true"] - out1["y_pred"])))
    mae2 = float(np.mean(np.abs(out2["y_true"] - out2["y_pred"])))
    assert mae1 == mae2
