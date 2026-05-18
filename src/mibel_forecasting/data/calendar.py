"""Calendar features for the EPF panel.

Adds hour, day-of-week, weekend flag, and Spanish holiday flag. Following
Lago et al. (2021), these enter the LEAR/DNN models as plain integer
columns rather than one-hot dummies, since the linear and neural models
handle the encoding internally.
"""

from __future__ import annotations

import pandas as pd


def add_calendar_features(
    df: pd.DataFrame,
    *,
    country: str = "ES",
    add_holiday: bool = True,
    add_weekend: bool = True,
    add_hour: bool = True,
    add_dow: bool = True,
) -> pd.DataFrame:
    """Add calendar columns to a DataFrame indexed by ``DatetimeIndex``.

    The added columns (when enabled) are ``hour``, ``dow``,
    ``is_weekend``, ``is_holiday``. Existing columns with the same names
    are overwritten.

    The holiday calendar uses ``python-holidays`` for the requested ISO
    country code (default ``ES`` — national Spanish holidays only; regional
    holidays are not included).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must have a DatetimeIndex")

    out = df.copy()
    idx = out.index

    if add_hour:
        out["hour"] = idx.hour.astype("int8")
    if add_dow:
        out["dow"] = idx.dayofweek.astype("int8")
    if add_weekend:
        out["is_weekend"] = (idx.dayofweek >= 5).astype("int8")
    if add_holiday:
        import holidays  # imported lazily to keep import cost low

        years = sorted({d.year for d in idx})
        hols = holidays.country_holidays(country, years=years)
        # Use the local wall-clock date: `tz_localize(None)` strips tz info
        # without converting to UTC (unlike `tz_convert(None)`).
        local_idx = idx.tz_localize(None) if idx.tz is not None else idx
        dates = local_idx.date
        out["is_holiday"] = pd.Series(
            [d in hols for d in dates], index=idx, dtype="int8"
        )

    return out
