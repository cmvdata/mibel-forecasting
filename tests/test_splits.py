from __future__ import annotations

import pandas as pd
import pytest

from mibel_forecasting.data.splits import rolling_window_split


def _hourly_index(start: str, end: str, tz: str | None = "Europe/Madrid") -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="h", tz=tz)


def test_rolling_split_no_overlap_and_adjacency():
    idx = _hourly_index("2022-01-01", "2024-12-31 23:00")
    splits = list(
        rolling_window_split(
            idx,
            train_size="730D",
            test_size="1D",
            step="1D",
            test_start="2024-01-01",
            test_end="2024-01-10",
        )
    )
    assert len(splits) == 10
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()
        assert (test_idx.min() - train_idx.max()) <= pd.Timedelta("1h")
        assert len(test_idx) in (23, 24, 25)  # DST tolerant


def test_rolling_split_train_size_correct():
    idx = _hourly_index("2022-01-01", "2024-12-31 23:00")
    train_idx, _ = next(
        rolling_window_split(
            idx,
            train_size="365D",
            test_size="1D",
            test_start="2024-01-15",
            test_end="2024-01-15",
        )
    )
    # one calendar year of hourly data is ~8760 hours, give or take DST
    assert 8758 <= len(train_idx) <= 8762


def test_rolling_split_unsorted_index_raises():
    idx = _hourly_index("2024-01-01", "2024-01-05")[::-1]
    with pytest.raises(ValueError):
        list(rolling_window_split(idx))


def test_rolling_split_works_with_naive_index():
    idx = _hourly_index("2022-01-01", "2024-06-30", tz=None)
    splits = list(
        rolling_window_split(
            idx,
            train_size="365D",
            test_size="1D",
            test_start="2024-06-25",
            test_end="2024-06-26",
        )
    )
    assert len(splits) == 2
