"""Tests for ``mibel_forecasting.features.technical_indicators``.

Coverage:

1. **No-leakage regression per indicator column** — poisoning every
   timestamp strictly after ``T`` with a huge value must not change
   the indicator at any timestamp ``≤ T`` by a single bit.
2. **Index preservation** — output index equals input index exactly.
3. **Missing timestamps** — gaps in the input panel are preserved in
   the output; no synthetic rows are added.
4. **Warm-up NaN pattern** — on a clean synthetic panel, after enough
   history every column is fully populated (only the initial warm-up
   rows are NaN).
5. **Smoke test on a real DAM panel slice** — gated by
   ``@pytest.mark.network`` because it pulls ESIOS price data; runs
   locally with a populated cache, skipped in CI.

Parameter values come from the audit at
``reports/diagnostics/demir_2019_ti_parameter_audit_2026_05.md``;
this test module does not duplicate or re-derive them.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mibel_forecasting.features.technical_indicators import (
    TI_COLUMNS,
    compute_technical_indicators,
)


def _synthetic_panel(days: int = 200, seed: int = 0) -> pd.DataFrame:
    """UTC-indexed hourly panel with a periodic-plus-noise ``price_es``
    series. Used by every test that does not need real DAM data."""
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h", tz="UTC")
    rng = np.random.default_rng(seed)
    hour = idx.hour
    dow = idx.dayofweek
    price = (
        50
        + 10 * np.sin(2 * np.pi * hour / 24)
        + 5 * (dow < 5).astype(float)
        + rng.normal(0, 1.0, len(idx))
    )
    return pd.DataFrame({"price_es": price}, index=idx)


@pytest.mark.parametrize("column", TI_COLUMNS)
def test_no_leakage_per_indicator(column: str):
    """Per-column leakage regression. Choose a target ``T`` well past
    every indicator's warm-up window. Build two panels: one clean,
    one with ``price_es`` poisoned to 9999 at every timestamp
    strictly after ``T``. Compute the technical indicators on both
    and assert that the column under test is byte-identical for every
    timestamp ``≤ T``."""
    panel = _synthetic_panel(days=200)
    # Day 150, hour 0 — past every audit-pinned warm-up window
    # (Bollinger ``n = 58`` is the longest).
    target_t = panel.index[150 * 24]

    panel_clean = panel.copy()
    panel_poisoned = panel.copy()
    panel_poisoned.loc[panel_poisoned.index > target_t, "price_es"] = 9999.0

    ti_clean = compute_technical_indicators(panel_clean)
    ti_poisoned = compute_technical_indicators(panel_poisoned)

    mask = ti_clean.index <= target_t
    pd.testing.assert_series_equal(
        ti_clean.loc[mask, column],
        ti_poisoned.loc[mask, column],
        check_names=False,
        check_exact=True,
    )


def test_no_leakage_targeted_t_plus_1_and_t_plus_2():
    """Tighter variant of the leakage regression: poison only the two
    timestamps immediately after ``T`` (at the same hour-of-day, so
    they actually enter the lagged hour-group series) and assert all
    indicators at ``≤ T`` are byte-identical."""
    panel = _synthetic_panel(days=200)
    target_t = panel.index[150 * 24]

    panel_poisoned = panel.copy()
    # T + 1 day and T + 2 days at the same hour — the rows that would
    # enter the next-day shifted view of T's hour-group.
    poison_ts = [target_t + pd.Timedelta(days=1), target_t + pd.Timedelta(days=2)]
    for ts in poison_ts:
        panel_poisoned.loc[ts, "price_es"] = 9999.0

    ti_clean = compute_technical_indicators(panel)
    ti_poisoned = compute_technical_indicators(panel_poisoned)
    mask = ti_clean.index <= target_t
    pd.testing.assert_frame_equal(
        ti_clean.loc[mask], ti_poisoned.loc[mask], check_exact=True
    )


def test_index_preservation():
    """Output index must equal input index exactly (same length, same
    timestamps, same tz)."""
    panel = _synthetic_panel(days=100)
    ti = compute_technical_indicators(panel)
    pd.testing.assert_index_equal(ti.index, panel.index)
    assert len(ti) == len(panel)


def test_missing_timestamps_preserved():
    """Panel with gaps must produce output with the same gaps; no
    synthetic rows are added at the missing positions."""
    panel = _synthetic_panel(days=100)
    drop_timestamps = [
        pd.Timestamp("2024-02-15 03:00", tz="UTC"),
        pd.Timestamp("2024-02-15 04:00", tz="UTC"),
        pd.Timestamp("2024-03-01 00:00", tz="UTC"),
    ]
    panel_with_gaps = panel.drop(drop_timestamps)
    ti = compute_technical_indicators(panel_with_gaps)
    pd.testing.assert_index_equal(ti.index, panel_with_gaps.index)
    for ts in drop_timestamps:
        assert ts not in ti.index


def test_warm_up_nan_pattern():
    """On a clean synthetic panel, after enough history every column
    is fully populated. The longest deterministic warm-up is Bollinger
    ``n = 58`` days plus the ``shift(1)``, i.e. ~59 days at every
    hour-of-day; conservatively, by day 70 onwards no NaN should
    remain."""
    panel = _synthetic_panel(days=300)
    ti = compute_technical_indicators(panel)
    # Day 70 onwards (the panel starts at 2024-01-01).
    cutoff = panel.index[70 * 24]
    tail = ti[ti.index >= cutoff]
    nan_counts = tail.isna().sum()
    assert (nan_counts == 0).all(), (
        f"unexpected NaN after warm-up: {nan_counts[nan_counts > 0].to_dict()}"
    )


def test_warm_up_nan_at_start_is_expected():
    """The very first row of each hour-of-day group must be NaN for
    every column (the ``shift(1)`` produces NaN there). This pins the
    warm-up behaviour: not just 'no late NaN' but also 'NaN where the
    audit's leakage rule guarantees one'."""
    panel = _synthetic_panel(days=100)
    ti = compute_technical_indicators(panel)
    # First 24 hours = one row per hour-of-day group → all NaN.
    first_day = ti.iloc[:24]
    assert first_day.isna().all().all()


def test_all_columns_present_with_expected_names():
    """The output column schema must equal :data:`TI_COLUMNS` exactly."""
    panel = _synthetic_panel(days=50)
    ti = compute_technical_indicators(panel)
    assert tuple(ti.columns) == TI_COLUMNS


def test_raises_when_price_es_missing():
    """Missing target column is a programming bug — fail loud, not
    silent NaN."""
    panel = _synthetic_panel(days=10).rename(columns={"price_es": "price_pt"})
    with pytest.raises(KeyError, match="price_es"):
        compute_technical_indicators(panel)


@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get("ESIOS_API_TOKEN"),
    reason="ESIOS_API_TOKEN not set",
)
def test_smoke_on_real_dam_panel_slice():
    """Smoke test on a real DAM panel slice. Pulls ESIOS price data
    (cached locally; CI skips this via the ``not network`` marker
    filter). Asserts the output is aligned to the input, columns are
    correct, and the post-warm-up rows are populated."""
    from mibel_forecasting.data.loaders import load_dam_panel

    panel = load_dam_panel(
        start="2024-01-01", end="2024-06-30", v8_exogenous=None
    )
    ti = compute_technical_indicators(panel)
    assert tuple(ti.columns) == TI_COLUMNS
    assert len(ti) == len(panel)
    pd.testing.assert_index_equal(ti.index, panel.index)

    # Post-warm-up (after ~60 days): every non-ROC indicator must be
    # fully populated. ti_roc is allowed to have NaN cells because the
    # audit §"Indicator 5. ROC" documents that MIBEL DAM 2024 has
    # ~6 % zero-price hours and ROC propagates NaN through its
    # ``p_{t-n} = 0`` division-by-zero edge case. The downstream LEAR
    # ``dropna(how='any')`` filter handles this cleanly.
    post_warmup = ti[ti.index >= pd.Timestamp("2024-04-01", tz="UTC")]
    non_roc_cols = [c for c in TI_COLUMNS if c != "ti_roc"]
    non_roc_nan = post_warmup[non_roc_cols].isna().sum()
    assert (non_roc_nan == 0).all(), (
        f"unexpected NaN in non-ROC columns post-warm-up: "
        f"{non_roc_nan[non_roc_nan > 0].to_dict()}"
    )
    # ti_roc NaN density should be in line with the ~6 % zero-price-hour
    # rate of MIBEL DAM 2024 — loose upper bound at 10 %.
    roc_nan_frac = post_warmup["ti_roc"].isna().mean()
    assert roc_nan_frac < 0.10, (
        f"ti_roc NaN fraction {roc_nan_frac:.2%} exceeds the ~6 % rate "
        "of zero-price hours expected on MIBEL DAM 2024"
    )
