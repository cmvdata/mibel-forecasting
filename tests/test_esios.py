from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from mibel_forecasting.data import esios
from mibel_forecasting.data.esios import (
    ESIOSConfigError,
    _cache_path,
    pull_indicator,
)


def _seed_cache(cache: Path, indicator: int, geo: int, year: int, month: int) -> pd.Series:
    """Write a synthetic monthly Parquet so pull_indicator hits the cache."""
    idx = pd.date_range(f"{year}-{month:02d}-01", periods=24 * 3, freq="h", tz="UTC")
    series = pd.Series(range(len(idx)), index=idx, dtype=float, name="value")
    path = _cache_path(cache, indicator, geo, year, month)
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_frame("value").to_parquet(path)
    return series


def test_pull_indicator_reads_cache_without_network(tmp_path, monkeypatch):
    seeded = _seed_cache(tmp_path, indicator=600, geo=3, year=2024, month=6)

    def _no_network(*a, **kw):
        raise AssertionError("pull_indicator should not have hit the network on cache hit")

    monkeypatch.setattr(esios, "_fetch_month", _no_network)

    out = pull_indicator(
        600, 3,
        start="2024-06-01T00:00:00Z",
        end="2024-06-02T23:59:59Z",
        cache_dir=tmp_path,
    )
    assert out.index.tz is not None
    assert str(out.index.tz) == "UTC"
    # We seeded 72 hourly rows; the slice should keep the first 48 (2 days)
    assert 47 <= len(out) <= 48
    pd.testing.assert_series_equal(
        out.head(5), seeded.head(5).loc[out.head(5).index], check_names=False
    )


def test_pull_indicator_concatenates_months(tmp_path, monkeypatch):
    _seed_cache(tmp_path, 600, 3, 2024, 6)
    _seed_cache(tmp_path, 600, 3, 2024, 7)
    monkeypatch.setattr(esios, "_fetch_month", lambda *a, **kw: pytest.fail("unexpected fetch"))

    out = pull_indicator(
        600, 3,
        start="2024-06-01T00:00:00Z",
        end="2024-07-31T23:59:59Z",
        cache_dir=tmp_path,
    )
    assert out.index.is_monotonic_increasing
    assert out.index.is_unique
    assert (out.index.month == 6).any()
    assert (out.index.month == 7).any()


def test_pull_indicator_fetches_and_caches_on_miss(tmp_path, monkeypatch):
    calls: list[tuple[int, int, int, int]] = []

    def _fake_fetch(indicator_id, geo_id, year, month, **_kw):
        calls.append((indicator_id, geo_id, year, month))
        idx = pd.date_range(f"{year}-{month:02d}-01", periods=24, freq="h", tz="UTC")
        return pd.Series(range(24), index=idx, dtype=float, name="value")

    monkeypatch.setattr(esios, "_fetch_month", _fake_fetch)
    out = pull_indicator(
        600, 3,
        start="2024-06-01T00:00:00Z",
        end="2024-06-01T23:59:59Z",
        cache_dir=tmp_path,
    )
    assert len(calls) == 1
    assert calls[0] == (600, 3, 2024, 6)
    assert _cache_path(tmp_path, 600, 3, 2024, 6).exists()

    # Second call must not re-fetch.
    pull_indicator(
        600, 3,
        start="2024-06-01T00:00:00Z",
        end="2024-06-01T23:59:59Z",
        cache_dir=tmp_path,
    )
    assert len(calls) == 1


def test_pull_indicator_refresh_re_fetches_even_with_cache(tmp_path, monkeypatch):
    _seed_cache(tmp_path, 600, 3, 2024, 6)
    fetched = []

    def _fake_fetch(indicator_id, geo_id, year, month, **_kw):
        fetched.append((year, month))
        idx = pd.date_range(f"{year}-{month:02d}-01", periods=24, freq="h", tz="UTC")
        return pd.Series([42.0] * 24, index=idx, name="value")

    monkeypatch.setattr(esios, "_fetch_month", _fake_fetch)
    pull_indicator(
        600, 3,
        start="2024-06-01T00:00:00Z",
        end="2024-06-01T23:59:59Z",
        cache_dir=tmp_path,
        refresh=True,
    )
    assert fetched == [(2024, 6)]


def test_pull_indicator_missing_token_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("ESIOS_API_TOKEN", raising=False)

    def _real_fetch_no_token(*a, **kw):
        # Force the real path to ensure _resolve_token is hit.
        from mibel_forecasting.data.esios import _resolve_token

        _resolve_token()
        return pd.Series(dtype=float, name="value")

    monkeypatch.setattr(esios, "_fetch_month", _real_fetch_no_token)
    with pytest.raises(ESIOSConfigError, match="ESIOS_API_TOKEN"):
        pull_indicator(
            600, 3,
            start="2024-06-01T00:00:00Z",
            end="2024-06-01T23:59:59Z",
            cache_dir=tmp_path,
        )


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("ESIOS_API_TOKEN"), reason="ESIOS_API_TOKEN not set")
def test_pull_indicator_real_pull_2024_06_01(tmp_path):
    out = pull_indicator(
        600, 3,
        start="2024-06-01T00:00:00Z",
        end="2024-06-01T23:59:59Z",
        cache_dir=tmp_path,
    )
    assert len(out) == 24
    assert str(out.index.tz) == "UTC"
    # Cache should now contain that month's parquet
    assert _cache_path(tmp_path, 600, 3, 2024, 6).exists()
