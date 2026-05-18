from __future__ import annotations

import pandas as pd

from mibel_forecasting.data.calendar import add_calendar_features


def _hourly_panel(start: str, end: str, tz: str | None = "Europe/Madrid") -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="h", tz=tz)
    return pd.DataFrame({"x": range(len(idx))}, index=idx)


def test_add_calendar_features_columns():
    df = add_calendar_features(_hourly_panel("2024-01-01", "2024-01-07"))
    for col in ("hour", "dow", "is_weekend", "is_holiday"):
        assert col in df.columns


def test_holiday_flag_marks_known_spanish_holidays():
    # 2024-01-01 (Año Nuevo) and 2024-12-25 (Navidad) are national holidays.
    df = add_calendar_features(_hourly_panel("2024-01-01", "2024-12-31"))
    assert df.loc["2024-01-01", "is_holiday"].eq(1).all()
    assert df.loc["2024-12-25", "is_holiday"].eq(1).all()
    # 2024-01-02 is a regular working Tuesday.
    assert df.loc["2024-01-02", "is_holiday"].eq(0).all()


def test_weekend_flag_for_saturday_and_sunday():
    df = add_calendar_features(_hourly_panel("2024-01-06", "2024-01-08"))
    # 2024-01-06 Sat, 2024-01-07 Sun, 2024-01-08 Mon
    assert df.loc["2024-01-06", "is_weekend"].eq(1).all()
    assert df.loc["2024-01-07", "is_weekend"].eq(1).all()
    assert df.loc["2024-01-08", "is_weekend"].eq(0).all()


def test_calendar_features_disable_flags():
    df = add_calendar_features(
        _hourly_panel("2024-01-01", "2024-01-02"),
        add_holiday=False,
        add_weekend=False,
        add_dow=False,
    )
    assert "is_holiday" not in df.columns
    assert "is_weekend" not in df.columns
    assert "dow" not in df.columns
    assert "hour" in df.columns


def test_calendar_features_naive_index_works():
    df = _hourly_panel("2024-01-01", "2024-01-03", tz=None)
    out = add_calendar_features(df)
    assert "is_holiday" in out.columns
    assert out.loc["2024-01-01", "is_holiday"].eq(1).all()
