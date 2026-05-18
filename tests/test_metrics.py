from __future__ import annotations

import numpy as np
import pandas as pd

from mibel_forecasting.evaluation.metrics import by_hour, mae, rmae, smape


def _series(values, start="2024-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="h", tz="Europe/Madrid")
    return pd.Series(values, index=idx)


def test_mae_basic():
    y = _series([1.0, 2.0, 3.0, 4.0])
    yhat = _series([1.0, 2.0, 4.0, 6.0])
    assert mae(y, yhat) == pytest_approx_value(0.75)


def test_smape_lago_formula_known_value():
    # |y-ŷ|/(|y|+|ŷ|) = 0.5 for (1, 3) → sMAPE = 100%
    y = _series([1.0, 1.0])
    yhat = _series([3.0, 3.0])
    assert smape(y, yhat) == pytest_approx_value(100.0)


def test_smape_zero_denominator_skipped():
    y = _series([0.0, 1.0])
    yhat = _series([0.0, 1.0])
    assert smape(y, yhat) == pytest_approx_value(0.0)


def test_rmae_against_self_is_one():
    y = _series([1.0, 2.0, 3.0])
    yhat = _series([1.5, 2.5, 3.5])
    assert rmae(y, yhat, yhat) == pytest_approx_value(1.0)


def test_rmae_perfect_model_is_zero():
    y = _series([1.0, 2.0, 3.0])
    naive = _series([1.5, 2.5, 3.5])
    assert rmae(y, y, naive) == 0.0


def test_metrics_drop_nan_pairwise():
    y = _series([1.0, np.nan, 3.0, 4.0])
    yhat = _series([1.0, 2.0, np.nan, 5.0])
    # Only positions 0 and 3 are common
    assert mae(y, yhat) == pytest_approx_value(0.5)


def test_by_hour_returns_one_value_per_hour():
    # Two full days of constant per-hour error: hour 0 → err 1, hour 1 → err 2, etc.
    idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="Europe/Madrid")
    err = np.tile(np.arange(24, dtype=float), 2)
    y = pd.Series(np.zeros(48), index=idx)
    yhat = pd.Series(err, index=idx)
    out = by_hour(y, yhat, mae)
    assert len(out) == 24
    np.testing.assert_allclose(out.to_numpy(), np.arange(24, dtype=float))


def pytest_approx_value(v: float, tol: float = 1e-9):
    import pytest

    return pytest.approx(v, abs=tol)
