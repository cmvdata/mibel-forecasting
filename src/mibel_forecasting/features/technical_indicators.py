"""Demir-style technical indicators for day-ahead price forecasting.

All formulas and parameter values in this module are pinned by the
audit at
``reports/diagnostics/demir_2019_ti_parameter_audit_2026_05.md``
(committed in ``0a81f1f`` and refined in ``c52c3ed``). Each parameter
constant below references the specific audit section that justifies
it; the audit's "Summary table of Phase-2 parameters" is the
single-line cross-reference.

**Leakage-safe semantics.** The audit's "Leakage-safe computation
rule" section requires that the indicator value at any timestamp
``T`` depend only on prices at timestamps strictly before ``T``,
respecting the hour-of-day grouping that Demir 2019 §2.3.3 specifies
("we treat the dataset as an assortment of prices from 24 separate
markets"). This module enforces that by applying ``.shift(1)`` once
on the per-hour daily series before any rolling / ewm / pct_change
operation, then passing the shifted series into every indicator.

**Gap handling.** Missing timestamps in the input panel are
preserved in the output (no synthetic rows). Within each hour-of-day
group the shift and the rolling / ewm windows operate by position,
so a gap shifts what was the "previous day" to whatever the previous
existing row in that hour-group is — never further into the future,
never forward-filled.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Parameters (cite the audit, never finance defaults)
# ---------------------------------------------------------------------------

EMA_SPAN: int = 2
"""EMA span ``s``. Audit §"Indicator 1. EMA": Demir Table 1 column "Best TI"
for HR and column "Second-Best TI" for LR both pin ``s = 2`` (linear-model
consensus, high confidence)."""

BOLLINGER_N: int = 58
"""Bollinger %B window ``n`` (in days at same hour). Audit §"Indicator 2.
Bollinger %B": Demir Table 1 "Best TI" for LR and "Second-Best TI" for HR
both pin ``n = 58``."""

MACD_S1: int = 2
"""MACD fast EMA span. Audit §"Indicator 3. MACD": Demir Table 1 "Third-Best
TI" for LR pins the Histogram with footnote ``** = (s_1=2, s_2=26, s=9)``."""

MACD_S2: int = 26
"""MACD slow EMA span. Same audit citation as ``MACD_S1`` — footnote ``**``."""

MACD_S_SIGNAL: int = 9
"""MACD signal EMA span. Same audit citation as ``MACD_S1`` — footnote ``**``."""

MOM_N: int = 58
"""Momentum lag ``n`` (in days at same hour). Audit §"Indicator 4. MOM":
no direct linear-model citation in Demir Table 1; ``n = 58`` is the
cross-model recurring optimum (AB / GB best, RF second-best). Medium
confidence — a MIBEL grid-search ablation is listed as future work."""

ROC_N: int = 49
"""Rate-of-change lag ``n``. Audit §"Indicator 5. ROC": no direct
linear-model citation; ``n = 49`` is the cross-model favoured deep-model
optimum (CNN / 2CNN best). Medium confidence."""

COPPOCK_N1: int = 18
"""Coppock first ROC lag. Audit §"Indicator 6. Coppock": Demir Table 1
"Second-Best TI" for ResNet, footnote ``* = (n_1=18, n_2=24, s=18)``.
Low confidence (no linear-model evidence at any parameter setting).
**NOT** the 2CNN_NN third-best ``(n_1=58, n_2=74, s=54)`` (footnote
``****``) — that pairing is explicitly rejected by the audit."""

COPPOCK_N2: int = 24
"""Coppock second ROC lag. Same audit citation as ``COPPOCK_N1`` —
footnote ``*``."""

COPPOCK_SPAN: int = 18
"""Coppock outer-EMA span. Same audit citation as ``COPPOCK_N1`` —
footnote ``*``."""

# ---------------------------------------------------------------------------
# Output column schema
# ---------------------------------------------------------------------------

TI_COLUMNS: tuple[str, ...] = (
    "ti_ema",
    "ti_bollinger_pct_b",
    "ti_macd",
    "ti_macd_signal",
    "ti_macd_histogram",
    "ti_mom",
    "ti_roc",
    "ti_coppock",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _shift_one_day_per_hour(series: pd.Series) -> pd.Series:
    """Shift each hour-of-day group by 1 position (= 1 calendar day at the
    same hour on a regular hourly grid). This implements the audit's
    "Leakage-safe computation rule": every indicator below operates on
    the shifted series rather than the raw price."""
    return series.groupby(series.index.hour).shift(1)


def _per_hour_transform(
    series: pd.Series, op: Callable[[pd.Series], pd.Series]
) -> pd.Series:
    """Apply ``op`` independently to each hour-of-day group and return
    the result aligned to the original index."""
    return series.groupby(series.index.hour).transform(op)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_technical_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute the six Demir 2019 technical indicators (eight columns)
    on the ``price_es`` series of a UTC-indexed hourly panel.

    Parameters
    ----------
    panel
        DataFrame indexed by tz-aware hourly timestamps. Must contain a
        ``price_es`` column. Other columns are ignored.

    Returns
    -------
    DataFrame
        Same index as ``panel``, columns equal to :data:`TI_COLUMNS`.
        Every cell is leakage-safe: its value at any timestamp ``T``
        depends only on prices at hours-of-day matching ``T``'s hour on
        days strictly before ``T``.

    Notes
    -----
    The full parameter and formula specification lives in
    ``reports/diagnostics/demir_2019_ti_parameter_audit_2026_05.md``.
    Do not change a parameter here without first updating the audit.
    """
    if "price_es" not in panel.columns:
        raise KeyError("panel must contain a 'price_es' column")

    price = panel["price_es"]

    # Single shift(1) at the top — the audit's "Leakage-safe computation
    # rule" applies this once per indicator, on the per-hour-of-day daily
    # series. Each indicator below consumes `shifted` rather than the raw
    # price, so no downstream operation can see ``p_t``.
    shifted = _shift_one_day_per_hour(price)

    # 1. EMA — Demir 2019 Eq. (2), pandas-equivalent
    #    ``ewm(span=s, adjust=True).mean()`` (audit §"Indicator 1. EMA":
    #    Demir's alpha = (s-1)/(s+1) is pandas' decay factor under
    #    ``adjust=True``).
    ti_ema = _per_hour_transform(
        shifted, lambda x: x.ewm(span=EMA_SPAN, adjust=True).mean()
    )

    # 2. Bollinger %B — Demir 2019 Eq. (9), with bands at ±2·MSD per
    #    Eqs. (7)-(8). MSD uses population standard deviation
    #    (``ddof=0``) per Eq. (6); the audit explicitly notes the
    #    pandas default ``ddof=1`` is wrong here.
    ti_pct_b = _per_hour_transform(shifted, _bollinger_pct_b)

    # 3. MACD components — Demir 2019 Eqs. (3), (4), (5).
    ema_fast = _per_hour_transform(
        shifted, lambda x: x.ewm(span=MACD_S1, adjust=True).mean()
    )
    ema_slow = _per_hour_transform(
        shifted, lambda x: x.ewm(span=MACD_S2, adjust=True).mean()
    )
    ti_macd_series = ema_fast - ema_slow
    ti_macd_signal = _per_hour_transform(
        ti_macd_series, lambda x: x.ewm(span=MACD_S_SIGNAL, adjust=True).mean()
    )
    ti_macd_histogram = ti_macd_series - ti_macd_signal

    # 4. MOM — Demir 2019 Eq. (11): MOM(p_t, n) = p_t - p_{t-n}.
    ti_mom = _per_hour_transform(shifted, lambda x: x - x.shift(MOM_N))

    # 5. ROC — Demir 2019 Eq. (12): (p_t - p_{t-n}) / p_{t-n}.
    #    pandas' ``pct_change`` propagates NaN to p_{t-n} = 0 cells via
    #    the standard division-by-zero handling — relevant because
    #    MIBEL DAM 2024 has ~6 % zero-price hours (audit §"Indicator 5.
    #    ROC"). The downstream LEAR fit's ``dropna(how='any')`` filter
    #    will exclude any day whose ROC features are NaN.
    ti_roc = _per_hour_transform(shifted, lambda x: x.pct_change(periods=ROC_N))

    # 6. Coppock — Demir 2019 Eq. (13):
    #    COPP = EMA(ROC(p_t, n1) + ROC(p_t, n2), s).
    roc_n1 = _per_hour_transform(
        shifted, lambda x: x.pct_change(periods=COPPOCK_N1)
    )
    roc_n2 = _per_hour_transform(
        shifted, lambda x: x.pct_change(periods=COPPOCK_N2)
    )
    coppock_input = roc_n1 + roc_n2
    ti_coppock = _per_hour_transform(
        coppock_input, lambda x: x.ewm(span=COPPOCK_SPAN, adjust=True).mean()
    )

    out = pd.DataFrame(
        {
            "ti_ema": ti_ema,
            "ti_bollinger_pct_b": ti_pct_b,
            "ti_macd": ti_macd_series,
            "ti_macd_signal": ti_macd_signal,
            "ti_macd_histogram": ti_macd_histogram,
            "ti_mom": ti_mom,
            "ti_roc": ti_roc,
            "ti_coppock": ti_coppock,
        },
        index=panel.index,
    )
    # ``pct_change(periods=n)`` returns ``+/- inf`` (not NaN) when the
    # n-day-lagged hour-price is exactly 0 and the current price is
    # non-zero. MIBEL DAM 2024 has ~6 % zero-price hours so this is a
    # real path. Coerce ``inf`` to ``NaN`` so the downstream
    # ``dropna(how='any')`` filter in ``LEAR._pivot_hourly`` excludes
    # affected days uniformly; without this, Lasso receives
    # ``Input X contains infinity`` and crashes the rolling-forecast
    # loop on the first 0-price hour inside a ROC lookback window.
    return out.replace([np.inf, -np.inf], np.nan)


def _bollinger_pct_b(x: pd.Series) -> pd.Series:
    """%B for one hour-of-day daily series. Internal helper kept at
    module scope so the ``transform`` call above doesn't pickle a
    closure (multiprocessing-friendly)."""
    sma = x.rolling(BOLLINGER_N).mean()
    msd = x.rolling(BOLLINGER_N).std(ddof=0)
    upper = sma + 2 * msd
    lower = sma - 2 * msd
    return (x - lower) / (upper - lower)
