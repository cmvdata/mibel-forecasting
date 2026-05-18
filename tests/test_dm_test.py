from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mibel_forecasting.evaluation.dm_test import (
    DMResult,
    _newey_west_lag,
    diebold_mariano,
)


def _series(values: np.ndarray) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(values), freq="h", tz="Europe/Madrid")
    return pd.Series(values, index=idx, dtype=float)


def test_newey_west_lag_matches_andrews_1991():
    # n=100  → floor(4 * 1^(2/9)) = 4
    # n=1000 → floor(4 * 10^(2/9)) ≈ floor(4 * 1.6681) = 6
    assert _newey_west_lag(100, horizon=1) == 4
    assert _newey_west_lag(1000, horizon=1) == 6


def test_newey_west_lag_floored_by_horizon():
    # Even with very small n, lag should be at least horizon - 1
    assert _newey_west_lag(10, horizon=24) == 23


def test_diebold_mariano_against_identical_predictions():
    rng = np.random.default_rng(0)
    y = _series(rng.normal(50, 5, 500))
    pred = _series(rng.normal(50, 5, 500))
    result = diebold_mariano(y, pred, pred)
    assert isinstance(result, DMResult)
    # Identical predictions → loss differential is identically zero, the
    # implementation reports stat=0, p=1 rather than NaN from a 0/0 HAC t-stat.
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.mean_loss_diff == pytest.approx(0.0, abs=1e-12)
    assert not result.reject_h0()


def test_diebold_mariano_detects_strong_winner():
    # pred_perfect predicts the realised series exactly; pred_random is noise.
    rng = np.random.default_rng(1)
    y_arr = rng.normal(0, 1, 500)
    y = _series(y_arr)
    pred_random = _series(rng.normal(0, 1, 500))
    pred_perfect = _series(y_arr)
    result = diebold_mariano(y, pred_random, pred_perfect)
    # pred1 (random) loses → statistic should be strongly positive
    assert result.statistic > 5
    assert result.reject_h0()


def test_diebold_mariano_invalid_power_raises():
    y = _series(np.zeros(50))
    p = _series(np.zeros(50))
    with pytest.raises(ValueError):
        diebold_mariano(y, p, p, power=3)


def test_diebold_mariano_invalid_horizon_raises():
    y = _series(np.zeros(50))
    p = _series(np.zeros(50))
    with pytest.raises(ValueError):
        diebold_mariano(y, p, p, horizon=0)


def test_diebold_mariano_too_few_observations_raises():
    y = _series(np.zeros(5))
    p = _series(np.zeros(5))
    with pytest.raises(ValueError, match="Too few observations"):
        diebold_mariano(y, p, p)


def test_diebold_mariano_drops_nan_pairwise():
    rng = np.random.default_rng(2)
    arr = rng.normal(0, 1, 200)
    y = _series(arr)
    p1 = y.copy()
    p1.iloc[::10] = np.nan
    p2 = _series(rng.normal(0, 1, 200))
    result = diebold_mariano(y, p1, p2)
    # 20 rows dropped → 180 remain
    assert result.n_obs == 180
