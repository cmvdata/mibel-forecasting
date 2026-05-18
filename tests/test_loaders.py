from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from mibel_forecasting.data import loaders
from mibel_forecasting.data.loaders import (
    CID_DEFAULT_COLS,
    DAM_V8_EXOGENOUS,
    load_cid_panel,
    load_dam_panel,
)

_DAM_PATH = os.environ.get(
    "MIBEL_DAM_PARQUET",
    r"C:/Users/Carlo/Desktop/Projects/mibel-congestion-monitor/data/processed/mibel_dataset_20190101_20241231_v8.parquet",
)
_CID_PATH = os.environ.get(
    "MIBEL_CID_PARQUET",
    r"C:/Users/Carlo/Desktop/Projects/mibel-congestion-monitor/neuro_detector/data/processed/features_2022_2024.parquet",
)

_dam_available = Path(_DAM_PATH).exists()
_cid_available = Path(_CID_PATH).exists()
cid_only = pytest.mark.skipif(not _cid_available, reason=f"CID parquet missing: {_CID_PATH}")
dam_only = pytest.mark.skipif(not _dam_available, reason=f"DAM parquet missing: {_DAM_PATH}")


def _synthetic_esios(start: str, end: str, base: float = 60.0) -> pd.Series:
    """A deterministic hourly UTC series for monkeypatching ESIOS pulls."""
    idx = pd.date_range(start, end, freq="h", tz="UTC")
    return pd.Series(base + (idx.hour - 12) * 0.5, index=idx, name="value", dtype=float)


def _patch_esios(monkeypatch, *, es=70.0, pt=65.0, fr=55.0):
    def _factory(base):
        def _impl(*, start, end, cache_dir=None, refresh=False):
            return _synthetic_esios(str(start), str(end), base=base)
        return _impl

    monkeypatch.setattr(loaders, "pull_dam_es", _factory(es))
    monkeypatch.setattr(loaders, "pull_dam_pt", _factory(pt))
    monkeypatch.setattr(loaders, "pull_dam_fr", _factory(fr))


def test_load_dam_basic_without_v8(monkeypatch):
    _patch_esios(monkeypatch)
    df = load_dam_panel(
        start="2024-06-01", end="2024-06-03",
        v8_exogenous=None,
    )
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "Europe/Madrid"
    assert list(df.columns) == ["price_es", "price_pt", "price_fr"]
    assert df["price_es"].notna().all()


def test_load_dam_target_col_relabels(monkeypatch):
    _patch_esios(monkeypatch)
    df = load_dam_panel(
        start="2024-06-01", end="2024-06-01",
        target_col="y",
        v8_exogenous=None,
    )
    assert df.columns[0] == "y"


def test_load_dam_can_skip_extra_prices(monkeypatch):
    _patch_esios(monkeypatch)
    df = load_dam_panel(
        start="2024-06-01", end="2024-06-01",
        include_price_pt=False,
        include_price_fr=False,
        v8_exogenous=None,
    )
    assert list(df.columns) == ["price_es"]


def test_load_dam_naive_timezone_param(monkeypatch):
    _patch_esios(monkeypatch)
    df = load_dam_panel(
        start="2024-06-01", end="2024-06-01",
        timezone=None,
        v8_exogenous=None,
    )
    assert df.index.tz is None


@dam_only
def test_load_dam_with_v8_exogenous(monkeypatch):
    _patch_esios(monkeypatch)
    df = load_dam_panel(
        start="2024-06-01", end="2024-06-03",
        v8_parquet_path=_DAM_PATH,
    )
    expected_head = ["price_es", "price_pt", "price_fr"]
    assert list(df.columns)[:3] == expected_head
    for col in DAM_V8_EXOGENOUS:
        assert col in df.columns


@dam_only
def test_load_dam_v8_exogenous_unknown_column_raises(monkeypatch):
    _patch_esios(monkeypatch)
    with pytest.raises(KeyError):
        load_dam_panel(
            start="2024-06-01", end="2024-06-01",
            v8_exogenous=("does_not_exist",),
            v8_parquet_path=_DAM_PATH,
        )


@cid_only
def test_load_cid_basic_shape_and_index():
    df = load_cid_panel(_CID_PATH)
    assert len(df) > 0
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.columns[0] == "mic_price"
    for col in CID_DEFAULT_COLS:
        assert col in df.columns


@cid_only
def test_load_cid_includes_missing_flag_aligned_with_target():
    df = load_cid_panel(_CID_PATH, reindex_hourly=False)
    assert "mic_price_missing" in df.columns
    expected = df["mic_price"].isna().astype("int8")
    pd.testing.assert_series_equal(df["mic_price_missing"], expected, check_names=False)


@cid_only
def test_load_cid_reindex_marks_gaps_as_missing():
    df = load_cid_panel(_CID_PATH, reindex_hourly=True)
    assert df.loc[df["mic_price"].isna(), "mic_price_missing"].eq(1).all()


@cid_only
def test_load_cid_fallback_day_no_duplicate_index():
    df = load_cid_panel(_CID_PATH, start="2024-10-27", end="2024-10-27")
    assert df.index.is_unique
    assert 22 <= len(df) <= 25


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("ESIOS_API_TOKEN"), reason="ESIOS_API_TOKEN not set")
def test_load_dam_real_pull_matches_esios_fresh():
    """End-to-end sanity: the panel target equals an independent ESIOS pull."""
    import requests

    token = os.environ["ESIOS_API_TOKEN"]
    # Fresh independent pull of indicator 600 geo=3 for one week
    r = requests.get(
        "https://api.esios.ree.es/indicators/600",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": token,
        },
        params=[
            ("start_date", "2024-06-01T00:00:00Z"),
            ("end_date", "2024-06-07T23:59:59Z"),
            ("geo_ids[]", "3"),
        ],
        timeout=60,
    )
    r.raise_for_status()
    rows = r.json()["indicator"]["values"]
    fresh = pd.DataFrame(rows)
    fresh["dt"] = pd.to_datetime(fresh["datetime_utc"], utc=True)
    fresh_es = (
        fresh.set_index("dt")["value"]
        .astype(float)
        .groupby(lambda t: t.floor("h"))
        .mean()
    )

    panel = load_dam_panel(
        start="2024-06-01", end="2024-06-07",
        v8_exogenous=None,
    )
    panel_es_utc = panel["price_es"].copy()
    panel_es_utc.index = panel_es_utc.index.tz_convert("UTC")
    merged = pd.concat([panel_es_utc.rename("panel"), fresh_es.rename("fresh")], axis=1).dropna()
    # MAD-vs-UTC clipping leaves the intersection ~2h short of a full week.
    assert len(merged) >= 160
    diff = (merged["panel"] - merged["fresh"]).abs()
    assert diff.max() < 1e-6, f"loader vs ESIOS fresh diverges: max diff {diff.max()}"
