from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mibel_forecasting.models.lear import (
    LEAR,
    _arcsinh_median,
    _arcsinh_median_apply,
    _arcsinh_median_invert,
    _feature_row_for_day,
    _pivot_hourly,
)


def _synthetic_panel(days: int = 400, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h", tz="Europe/Madrid")
    rng = np.random.default_rng(seed)
    hour = idx.hour
    dow = idx.dayofweek
    price = (
        50
        + 10 * np.sin(2 * np.pi * hour / 24)
        + 5 * (dow < 5).astype(float)
        + rng.normal(0, 1.0, len(idx))
    )
    demand = (
        25000 + 5000 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 500, len(idx))
    )
    wind = 5000 + 3000 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 800, len(idx))
    return pd.DataFrame(
        {"price_es": price, "es_demand_fc": demand, "es_wind_fc": wind}, index=idx
    )


def test_lear_default_feature_count_matches_lago_2021():
    m = LEAR(target_col="price_es")
    # 96 price lags + 7 DoW + 2 exog x 72 = 247 (Lago 2021 Sec 3.2)
    assert m.n_features == 247


def test_lear_feature_count_scales_with_exogenous():
    m = LEAR(target_col="price_es", exogenous_cols=("a", "b", "c"))
    assert m.n_features == 96 + 3 * 72 + 7


def test_lear_allows_empty_exogenous_and_yields_103_features():
    m = LEAR(target_col="price_es", exogenous_cols=())
    # 96 price lags + 7 DoW = 103 features (pure autoregressive control)
    assert m.n_features == 103


def test_arcsinh_median_round_trip_recovers_input():
    rng = np.random.default_rng(42)
    x = rng.normal(50, 20, size=(200, 5))
    scaled, median, mad = _arcsinh_median(x)
    back = _arcsinh_median_invert(scaled, median, mad)
    np.testing.assert_allclose(back, x, atol=1e-9)


def test_arcsinh_median_apply_matches_fit_on_same_data():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(50, 3))
    scaled, median, mad = _arcsinh_median(x)
    scaled_again = _arcsinh_median_apply(x, median, mad)
    np.testing.assert_allclose(scaled, scaled_again, atol=1e-12)


def test_arcsinh_median_handles_zero_mad_column():
    x = np.column_stack(
        [np.linspace(-1, 1, 20), np.full(20, 7.0)]  # second column is constant
    )
    scaled, _median, mad = _arcsinh_median(x)
    assert not np.isnan(scaled).any()
    assert mad[1] == 1.0  # fallback


def test_pivot_hourly_drops_incomplete_days():
    df = _synthetic_panel(days=5)
    # Knock out one hour to simulate DST spring-forward
    df = df.drop(df.index[26])
    price_w, _exog_w = _pivot_hourly(df, "price_es", ("es_demand_fc", "es_wind_fc"))
    assert price_w.shape[1] == 24
    # The day with a missing hour should have been dropped
    assert len(price_w) == 4


def test_feature_row_returns_none_when_lag_missing():
    df = _synthetic_panel(days=10)
    price_w, exog_w = _pivot_hourly(df, "price_es", ("es_demand_fc", "es_wind_fc"))
    # Day D=4 needs D-7 which only exists if we have day -3 (we don't).
    day_4 = price_w.index[4]
    feat = _feature_row_for_day(day_4, price_w, exog_w)
    assert feat is None


def test_feature_row_has_expected_length_when_lags_available():
    df = _synthetic_panel(days=20)
    price_w, exog_w = _pivot_hourly(df, "price_es", ("es_demand_fc", "es_wind_fc"))
    feat = _feature_row_for_day(price_w.index[10], price_w, exog_w)
    assert feat is not None
    assert feat.shape == (247,)


def test_lear_fit_trains_24_models():
    m = LEAR(target_col="price_es").fit(_synthetic_panel(days=400))
    assert len(m._lassos) == 24


def test_lear_predict_recovers_low_noise_synthetic_signal():
    df = _synthetic_panel(days=400)
    train = df.iloc[:-24]
    test = df.iloc[-24:].copy()
    test["price_es"] = np.nan

    m = LEAR(target_col="price_es").fit(train)
    pred = m.predict(test)
    assert pred.notna().all()
    realised = df.loc[test.index, "price_es"]
    mae = (pred - realised).abs().mean()
    # Signal amplitude ~10, noise std ~1 — LEAR with 247 features should be << 5
    assert mae < 3.0, f"unexpected MAE on synthetic data: {mae}"


def test_lear_predict_before_fit_raises():
    df = _synthetic_panel(days=400)
    test = df.iloc[-24:].copy()
    test["price_es"] = np.nan
    with pytest.raises(RuntimeError):
        LEAR(target_col="price_es").predict(test)


def test_lear_missing_target_col_raises():
    df = _synthetic_panel(days=400)
    with pytest.raises(KeyError):
        LEAR(target_col="does_not_exist").fit(df)


def test_lear_missing_exogenous_col_raises():
    df = _synthetic_panel(days=400)
    with pytest.raises(KeyError):
        LEAR(target_col="price_es", exogenous_cols=("no_such_col",)).fit(df)


def test_lear_predict_does_not_use_test_target():
    """Changing the realised target in test_df must not change predictions."""
    df = _synthetic_panel(days=400)
    train = df.iloc[:-24]
    test_nan = df.iloc[-24:].copy()
    test_nan["price_es"] = np.nan
    test_poisoned = df.iloc[-24:].copy()
    test_poisoned["price_es"] = 9999.0

    m = LEAR(target_col="price_es").fit(train)
    pred_nan = m.predict(test_nan)
    pred_poisoned = m.predict(test_poisoned)
    pd.testing.assert_series_equal(pred_nan, pred_poisoned, check_names=False)


def test_lear_predict_handles_partial_hour_test_day():
    """Real DAM panels have hourly-grid gaps (e.g. some unsafe v8 columns
    are NaN at scattered hours and ``.dropna()`` drops the row). A test
    day with fewer than 24 hours in the panel must NOT make ``predict``
    crash inside Lasso with 'Input X contains NaN'.

    Reproduces the exact failure mode that broke notebook 03 on the
    first real partial day in 2022 H2 (2022-09-23 has 22/24 hours):
    ``_pivot_one(test_df, col)`` returned a 22-column DataFrame, which
    after concat with the 24-column training pivot left two
    silently-NaN hour columns that fed straight into Lasso.

    Expected behaviour: predict returns cleanly. The partial day's
    predictions are NaN (insufficient information to forecast it);
    no exception is raised.
    """
    df = _synthetic_panel(days=400)
    train = df.iloc[:-24]
    test = df.iloc[-24:].copy()  # single day = the canonical rolling-forecast slice
    test["price_es"] = np.nan
    # Drop two hours so the day is partial (22 of 24).
    test = test.drop([test.index[3], test.index[5]])
    assert len(test) == 22

    m = LEAR(target_col="price_es").fit(train)
    pred = m.predict(test)  # MUST NOT raise
    assert len(pred) == 22
    assert pred.isna().all(), "partial day cannot be predicted; should stay NaN cleanly"


def test_lear_fit_robust_when_n_samples_close_to_n_features():
    """LassoLarsIC's OLS-based noise-variance estimator requires comfortably
    more samples than features. With ``n_samples <= n_features`` sklearn
    raises ``"the number of samples is smaller than the number of features"``;
    even with ``n_samples == n_features + 1`` the estimate is unreliable. LEAR
    must pass a unit-variance prior whenever ``n_samples`` is not above
    ``n_features + N_HOURS`` rows of safety margin.

    Reproduces the failure of the full-year robustness backtest where the
    rolling training window for LEAR demand+solar+wind (319 features)
    shrank close to the n_features boundary on at least one rolling step
    after partial-day dropping.
    """
    base = _synthetic_panel(days=335)
    rng = np.random.default_rng(1)
    solar = (
        3000
        + 2500 * np.maximum(0, np.sin(2 * np.pi * (base.index.hour - 6) / 24))
        + rng.normal(0, 400, len(base))
    )
    df = base.assign(es_solar_fc=solar)
    train = df.iloc[:-24]  # ~334 days minus 7 lag rows ~= 327 training samples

    # demand+solar+wind has 96 + 3*72 + 7 = 319 features — the boundary case.
    m = LEAR(
        target_col="price_es",
        exogenous_cols=("es_demand_fc", "es_solar_fc", "es_wind_fc"),
    ).fit(train)
    assert m.n_features == 319


def test_lear_partial_day_skipped_uniformly_across_variants():
    """The partial-day filter must be uniform across LEAR configurations.

    Without the uniform filter, AR-only (no exogenous columns) silently
    predicts the available hours of a partial day while LEAR with
    exogenous columns skips it entirely — producing a biased rMAE when
    those configurations are compared on the same regime window. Both
    variants must skip identically.
    """
    df = _synthetic_panel(days=400)
    train = df.iloc[:-24]
    test = df.iloc[-24:].copy()
    test["price_es"] = np.nan
    test = test.drop([test.index[3], test.index[5]])
    assert len(test) == 22

    for exog in [(), ("es_demand_fc", "es_wind_fc")]:
        m = LEAR(target_col="price_es", exogenous_cols=exog).fit(train)
        pred = m.predict(test)
        assert pred.isna().all(), (
            f"variant exog={exog} predicted a partial day; should have skipped "
            f"({pred.notna().sum()}/{len(pred)} hours non-NaN)"
        )


def test_lear_multiday_window_no_leakage_from_within_test():
    """Day N+1's prediction must be insensitive to day N's realised target."""
    df = _synthetic_panel(days=400)
    train = df.iloc[:-3 * 24]
    test_baseline = df.iloc[-3 * 24:].copy()
    test_baseline["price_es"] = np.nan

    test_poisoned = df.iloc[-3 * 24:].copy()
    test_poisoned["price_es"] = np.nan
    # Inject a wildly wrong realised target into the FIRST day of the
    # test window. If LEAR were leaking it via the lag block, the
    # predictions for day 2 / day 3 would shift.
    first_day_mask = test_poisoned.index.date == test_poisoned.index[0].date()
    test_poisoned.loc[first_day_mask, "price_es"] = 9999.0

    m = LEAR(target_col="price_es").fit(train)
    pred_baseline = m.predict(test_baseline)
    pred_poisoned = m.predict(test_poisoned)
    pd.testing.assert_series_equal(pred_baseline, pred_poisoned, check_names=False)
