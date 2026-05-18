"""Train/test splits for rolling-window evaluation.

Implements the daily rolling window common in EPF literature (Lago 2021):
for every day in the test horizon, the model is trained on the preceding
``train_size`` days and asked to forecast the 24 hours of the test day.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd


def rolling_window_split(
    index: pd.DatetimeIndex,
    *,
    train_size: pd.Timedelta | str = "1095D",
    test_size: pd.Timedelta | str = "1D",
    step: pd.Timedelta | str = "1D",
    test_start: str | pd.Timestamp | None = None,
    test_end: str | pd.Timestamp | None = None,
) -> Iterator[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """Yield ``(train_idx, test_idx)`` pairs over a rolling window.

    Parameters
    ----------
    index
        The full panel index. Must be a sorted ``DatetimeIndex``.
    train_size, test_size, step
        Window sizes. Accept anything ``pd.Timedelta`` accepts.
    test_start, test_end
        Inclusive bounds for test-window start times. If omitted, the
        iterator starts as soon as a full ``train_size`` window fits
        before the first available timestamp and ends at the last
        timestamp.

    Yields
    ------
    train_idx, test_idx : DatetimeIndex
        Sub-indices of ``index``. ``test_idx`` is always immediately
        adjacent to ``train_idx`` (no gap).
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    if not index.is_monotonic_increasing:
        raise ValueError("index must be sorted")

    train_td = pd.Timedelta(train_size)
    test_td = pd.Timedelta(test_size)
    step_td = pd.Timedelta(step)

    first_possible = index[0] + train_td
    last_possible = index[-1] - test_td + pd.Timedelta("1ns")

    start = pd.Timestamp(test_start) if test_start is not None else first_possible
    end = pd.Timestamp(test_end) if test_end is not None else last_possible

    tz = index.tz
    if tz is not None:
        if start.tzinfo is None:
            start = start.tz_localize(tz)
        if end.tzinfo is None:
            end = end.tz_localize(tz)

    if start < first_possible:
        start = first_possible
    if end > last_possible:
        end = last_possible

    test_start_ts = start
    while test_start_ts <= end:
        train_lo = test_start_ts - train_td
        train_hi = test_start_ts - pd.Timedelta("1ns")
        test_hi = test_start_ts + test_td - pd.Timedelta("1ns")

        train_idx = index[(index >= train_lo) & (index <= train_hi)]
        test_idx = index[(index >= test_start_ts) & (index <= test_hi)]
        if len(train_idx) and len(test_idx):
            yield train_idx, test_idx

        test_start_ts = test_start_ts + step_td
