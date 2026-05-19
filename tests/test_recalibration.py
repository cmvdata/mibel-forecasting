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


def test_rolling_forecast_masks_target_before_predict():
    """The realised target must never reach the model's predict() input."""

    seen_targets: list[bool] = []

    class _SnoopingModel:
        """Records whether any non-NA price_es reached predict()."""

        target_col = "price_es"

        def fit(self, train_df):
            return self

        def predict(self, test_df):
            seen_targets.append(test_df[self.target_col].notna().any())
            return pd.Series(0.0, index=test_df.index, name="y_pred")

    df = _periodic_panel(days=21)
    out = rolling_forecast(
        df,
        target_col="price_es",
        model_factory=_SnoopingModel,
        train_size="10D",
        test_start="2024-01-15",
        test_end="2024-01-17",
    )
    assert len(seen_targets) >= 1, "rolling_forecast did not invoke predict()"
    assert not any(seen_targets), (
        "rolling_forecast leaked the realised target into predict(): "
        f"saw non-NA target in {sum(seen_targets)}/{len(seen_targets)} windows"
    )
    # y_true should still carry the realised values for the metrics layer.
    assert out["y_true"].notna().all()


def test_rolling_forecast_target_masking_raises_strict_spy_never_fires():
    """Strict variant: a spy that *raises* on any non-NA target must never
    fire when fed through rolling_forecast. If masking ever regresses,
    this test crashes inside the loop instead of silently passing."""

    class _StrictSpy:
        target_col = "price_es"

        def fit(self, train_df):
            return self

        def predict(self, test_df):
            if test_df[self.target_col].notna().any():
                raise AssertionError(
                    "rolling_forecast handed the realised target to predict()"
                )
            return pd.Series(0.0, index=test_df.index, name="y_pred")

    df = _periodic_panel(days=21)
    rolling_forecast(
        df,
        target_col="price_es",
        model_factory=_StrictSpy,
        train_size="10D",
        test_start="2024-01-15",
        test_end="2024-01-17",
    )


def test_rolling_forecast_masks_target_across_multi_day_test_size():
    """All hours of a 3-day test window — not just the first day — must
    have the target column masked before predict() runs."""

    n_seen_hours: list[int] = []

    class _MultiDaySpy:
        target_col = "price_es"

        def fit(self, train_df):
            return self

        def predict(self, test_df):
            if test_df[self.target_col].notna().any():
                raise AssertionError(
                    f"realised target leaked across {len(test_df)} test rows"
                )
            n_seen_hours.append(len(test_df))
            return pd.Series(0.0, index=test_df.index, name="y_pred")

    df = _periodic_panel(days=30)
    rolling_forecast(
        df,
        target_col="price_es",
        model_factory=_MultiDaySpy,
        train_size="14D",
        test_size="3D",
        step="3D",
        test_start="2024-01-16",
        test_end="2024-01-22",
    )
    assert n_seen_hours, "spy was never called"
    assert all(n == 3 * 24 for n in n_seen_hours), (
        f"expected 72 hours per window, got {n_seen_hours}"
    )


def test_rolling_forecast_masks_target_but_preserves_other_columns():
    """Masking touches only the target column; every other column the
    model relies on (exogenous features, calendar dummies, …) must reach
    predict() with its original values."""

    seen_exog_intact: list[bool] = []

    class _ExogChecker:
        target_col = "price_es"

        def fit(self, train_df):
            return self

        def predict(self, test_df):
            # If the loader / evaluator ever NA-ed an exogenous column by
            # accident, this would catch it.
            seen_exog_intact.append(test_df["es_demand_fc"].notna().all())
            return pd.Series(0.0, index=test_df.index, name="y_pred")

    df = _periodic_panel(days=21).assign(es_demand_fc=lambda x: 30_000.0)
    rolling_forecast(
        df,
        target_col="price_es",
        model_factory=_ExogChecker,
        train_size="10D",
        test_start="2024-01-15",
        test_end="2024-01-17",
    )
    assert seen_exog_intact and all(seen_exog_intact), (
        "rolling_forecast masked more than just the target column"
    )
