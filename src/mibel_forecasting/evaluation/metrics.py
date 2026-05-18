"""Forecast accuracy metrics.

All metrics follow the definitions of Lago et al. (2021) so results are
directly comparable to the published EPF benchmarks. NaN values in
``y_true`` or ``y_pred`` are dropped pairwise before computing.

Functions
---------
mae
    Mean absolute error.
smape
    Symmetric MAPE, Lago 2021 form (zero-safe denominator).
rmae
    Relative MAE versus a naive benchmark. ``< 1`` means the model beats
    the naive; ``> 1`` means it is worse.
by_hour
    Apply any of the above hour-by-hour (0–23) and return a Series indexed
    by hour of day. Useful for diagnosing where a model wins or loses.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def _align(*series: pd.Series) -> list[np.ndarray]:
    df = pd.concat({f"s{i}": s for i, s in enumerate(series)}, axis=1).dropna()
    return [df[f"s{i}"].to_numpy() for i in range(len(series))]


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt, yp = _align(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def smape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Symmetric MAPE in percent. Lago 2021 form: 2·|y-ŷ| / (|y|+|ŷ|).

    Returns a percent value in [0, 200]. Zeros in the denominator are
    skipped (rather than producing inf), which is the convention used in
    epftoolbox.
    """
    yt, yp = _align(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    denom = np.abs(yt) + np.abs(yp)
    mask = denom > 0
    if not mask.any():
        return 0.0
    return float(np.mean(2.0 * np.abs(yt[mask] - yp[mask]) / denom[mask]) * 100.0)


def rmae(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_naive: pd.Series,
) -> float:
    """Relative MAE: MAE(model) / MAE(naive)."""
    naive_mae = mae(y_true, y_naive)
    if naive_mae == 0 or np.isnan(naive_mae):
        return float("nan")
    return mae(y_true, y_pred) / naive_mae


def by_hour(
    y_true: pd.Series,
    y_pred: pd.Series,
    metric_fn: Callable[..., float],
    *extra: pd.Series,
) -> pd.Series:
    """Group by ``hour`` and apply ``metric_fn`` (sMAPE, MAE, ...).

    ``extra`` is forwarded as additional positional series — useful for
    :func:`rmae`, which needs ``y_naive`` as a third argument.
    """
    parts = pd.concat({"yt": y_true, "yp": y_pred}, axis=1)
    for i, s in enumerate(extra):
        parts[f"e{i}"] = s
    parts = parts.dropna()
    parts["hour"] = parts.index.hour

    out: dict[int, float] = {}
    for hour, group in parts.groupby("hour"):
        args = [group["yt"], group["yp"]] + [group[f"e{i}"] for i in range(len(extra))]
        out[int(hour)] = metric_fn(*args)
    return pd.Series(out, name=metric_fn.__name__).sort_index()
