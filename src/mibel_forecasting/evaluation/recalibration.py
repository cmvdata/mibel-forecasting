"""Rolling-window recalibration loop.

Implements the daily-recalibration evaluation common in EPF (Lago 2021):
for every day in the test horizon, the model is re-fit on the preceding
``train_size`` window and asked to forecast the 24 hours of the test day.
Returns a long DataFrame of realisations versus predictions that the
metrics module then consumes.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from mibel_forecasting.data.splits import rolling_window_split
from mibel_forecasting.models.base import Model


def rolling_forecast(
    df: pd.DataFrame,
    *,
    target_col: str,
    model_factory: Callable[[], Model],
    train_size: pd.Timedelta | str = "1095D",
    test_size: pd.Timedelta | str = "1D",
    step: pd.Timedelta | str = "1D",
    test_start: str | pd.Timestamp | None = None,
    test_end: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Run a rolling-window evaluation and collect ``(y_true, y_pred)``.

    Parameters
    ----------
    df
        Hourly panel (DatetimeIndex) containing at least ``target_col``.
    target_col
        Realised target column to compare against the prediction.
    model_factory
        Zero-arg callable returning a fresh :class:`Model` instance. A new
        instance is built for each rolling window, so the model is
        recalibrated from scratch every step.
    train_size, test_size, step
        Window sizes. See :func:`rolling_window_split`.
    test_start, test_end
        Inclusive bounds on the test-window start times.

    Returns
    -------
    DataFrame
        Indexed by test timestamp, columns ``y_true`` and ``y_pred``.
    """
    chunks = []
    for train_idx, test_idx in rolling_window_split(
        df.index,
        train_size=train_size,
        test_size=test_size,
        step=step,
        test_start=test_start,
        test_end=test_end,
    ):
        train = df.loc[train_idx]
        test = df.loc[test_idx]
        model = model_factory()
        model.fit(train)
        y_pred = model.predict(test)
        chunks.append(
            pd.DataFrame(
                {"y_true": test[target_col].to_numpy(), "y_pred": y_pred.to_numpy()},
                index=test_idx,
            )
        )
    if not chunks:
        return pd.DataFrame(columns=["y_true", "y_pred"])
    return pd.concat(chunks)
