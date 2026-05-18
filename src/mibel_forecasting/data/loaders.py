"""Panel loaders for the DAM (ESIOS-canonical) and CID (legacy parquet).

DAM
---
:func:`load_dam_panel` returns the hourly Spanish day-ahead panel:
- ``price_es`` is pulled directly from ESIOS indicator 600 geo=3
  (``Precio mercado SPOT Diario``, España). This is the canonical
  wholesale DAM price and is the target of the forecasting pipeline.
- ``price_pt`` (geo=1) and ``price_fr`` (geo=2) are pulled from the
  same indicator as optional exogenous features.
- Optional exogenous features (demand/wind/solar forecasts, NTC, gas, CO2)
  are merged from the ``mibel-congestion-monitor`` v8 parquet on request.
  The buggy ``price_es`` / ``price_fr`` columns of that parquet are
  **never** consumed here — see ``memory/project_v8_labels_bug.md``.

ESIOS responses are cached on disk by month, so subsequent calls do not
hit the API. See :mod:`mibel_forecasting.data.esios`.

CID
---
:func:`load_cid_panel` still reads the consolidated ``features_2022_2024.parquet``
panel produced by ``mibel-congestion-monitor`` because it carries the
Vilches (2026) microstructure features (``range_es``, ``mic_volume_mwh``,
``umm_active_mw``, etc.) that are not in ESIOS. The CID target
``mic_price`` is unaffected by the v8 label bug; the bug only affects
the ``price_es``/``price_fr`` columns of that panel, which are flagged
in the column docstring.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from mibel_forecasting.data.esios import pull_dam_es, pull_dam_fr, pull_dam_pt

load_dotenv()


DAM_V8_EXOGENOUS: tuple[str, ...] = (
    "es_demand_fc",
    "fr_demand_fc",
    "es_solar_fc",
    "es_wind_fc",
    "fr_solar_fc",
    "fr_wind_fc",
    "fr_nuclear_avail",
    "ntc_es_fr",
    "ntc_fr_es",
    "ttf_eur_mwh",
    "co2_eur_t",
)


CID_DEFAULT_COLS: tuple[str, ...] = (
    "price_es",  # NOTE: inherits the v8 label bug — actually PT/MIBEL-coupled.
    "price_fr",  # NOTE: inherits the v8 label bug — actually ES.
    "spread_da",
    "ntc_es_fr",
    "ntc_fr_es",
    "mic_volume_mwh",
    "range_es",
    "medio_es",
    "max_es",
    "min_es",
    "price_ratio_mi_md",
    "umm_active_mw",
)


def _resolve_path(arg: str | Path | None, env_var: str) -> Path:
    if arg is not None:
        path = Path(arg)
    else:
        raw = os.environ.get(env_var)
        if not raw:
            raise RuntimeError(
                f"No path provided and env var {env_var!r} is not set. "
                "Copy .env.example to .env and point it at the parquet."
            )
        path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"{path} (resolved from {env_var})")
    return path


def _attach_datetime_index(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    timezone: str | None,
) -> pd.DataFrame:
    """Attach a DatetimeIndex and resolve DST transitions.

    - Fall-back (e.g. last Sunday of October 03:00 → 02:00): when duplicate
      02:00 rows exist, the first occurrence is CEST and the second is CET.
    - Spring-forward (e.g. last Sunday of March 02:00 → 03:00): phantom
      02:00 rows are dropped (timestamp does not exist).
    """
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])
    df = df.drop(columns=[timestamp_col])

    if timezone is not None:
        is_dst = ts.duplicated(keep="last").to_numpy()
        ts_local = ts.dt.tz_localize(timezone, ambiguous=is_dst, nonexistent="NaT")
        valid = ts_local.notna().to_numpy()
        df = df.loc[valid]
        ts_local = ts_local[valid]
        df.index = pd.DatetimeIndex(ts_local, name="datetime")
    else:
        df.index = pd.DatetimeIndex(ts, name="datetime")

    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def _reindex_hourly(df: pd.DataFrame) -> pd.DataFrame:
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz=df.index.tz)
    out = df.reindex(full)
    out.index.name = "datetime"
    return out


def _slice_dates(
    df: pd.DataFrame, start: str | pd.Timestamp | None, end: str | pd.Timestamp | None
) -> pd.DataFrame:
    """Inclusive date slice with pandas partial-string-indexing semantics."""
    if start is None and end is None:
        return df

    def _bound(x: str | pd.Timestamp | None) -> str | pd.Timestamp | None:
        if x is None or isinstance(x, str):
            return x
        ts = pd.Timestamp(x)
        if df.index.tz is not None and ts.tzinfo is None:
            ts = ts.tz_localize(df.index.tz)
        return ts

    return df.loc[_bound(start):_bound(end)]


def _esios_to_panel_tz(s: pd.Series, timezone: str | None) -> pd.Series:
    """Convert a UTC-indexed ESIOS series to the requested panel timezone."""
    if timezone is None:
        return s.tz_convert(None) if s.index.tz is not None else s
    return s.tz_convert(timezone)


def _load_v8_exogenous(
    parquet_path: str | Path | None,
    *,
    columns: Sequence[str],
    timezone: str | None,
) -> pd.DataFrame:
    """Load the requested non-price columns from the v8 parquet."""
    path = _resolve_path(parquet_path, "MIBEL_DAM_PARQUET")
    raw = pd.read_parquet(path)
    missing = [c for c in columns if c not in raw.columns]
    if missing:
        raise KeyError(f"v8 exogenous columns missing: {missing}")
    keep = ["timestamp", *columns]
    return _attach_datetime_index(raw[keep], timestamp_col="timestamp", timezone=timezone)


def load_dam_panel(
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    target_col: str = "price_es",
    include_price_pt: bool = True,
    include_price_fr: bool = True,
    v8_exogenous: Sequence[str] | None = DAM_V8_EXOGENOUS,
    v8_parquet_path: str | Path | None = None,
    timezone: str | None = "Europe/Madrid",
    reindex_hourly: bool = True,
    drop_target_nan: bool = True,
    cache_dir: str | Path | None = None,
    refresh_esios: bool = False,
) -> pd.DataFrame:
    """Load the hourly DAM panel.

    Target ``price_es`` is pulled from ESIOS indicator 600 geo=3 directly
    (canonical Spanish wholesale price). Optional exogenous columns are
    merged in order: ESIOS Portugal / France prices first, then any
    requested v8 features.

    Parameters
    ----------
    start, end
        Inclusive bounds (date or timestamp).
    target_col
        Name of the target column in the returned DataFrame. The data is
        always ESIOS 600 geo=3; this parameter only relabels the column.
    include_price_pt, include_price_fr
        Whether to add ESIOS 600 geo=1 / geo=2 as ``price_pt`` / ``price_fr``.
    v8_exogenous
        Columns from the v8 parquet to merge as exogenous features. Pass
        ``None`` to skip the v8 join entirely (ESIOS-only panel).
    v8_parquet_path
        Override the v8 path (defaults to ``MIBEL_DAM_PARQUET`` env var).
    timezone
        Output index timezone. ``None`` returns naïve UTC.
    reindex_hourly
        If ``True``, fill any gaps in the hourly grid with NaN.
    drop_target_nan
        If ``True``, drop rows where the target is NaN.
    cache_dir
        Override the ESIOS cache directory.
    refresh_esios
        Force re-fetch of every required month even on cache hit.

    Returns
    -------
    DataFrame
        DatetimeIndex named ``datetime``. Column order is
        ``[target_col, price_pt?, price_fr?, *v8_exogenous]``.
    """
    es = pull_dam_es(start=start, end=end, cache_dir=cache_dir, refresh=refresh_esios)
    columns: dict[str, pd.Series] = {target_col: _esios_to_panel_tz(es, timezone)}

    if include_price_pt:
        pt = pull_dam_pt(start=start, end=end, cache_dir=cache_dir, refresh=refresh_esios)
        columns["price_pt"] = _esios_to_panel_tz(pt, timezone)

    if include_price_fr:
        fr = pull_dam_fr(start=start, end=end, cache_dir=cache_dir, refresh=refresh_esios)
        columns["price_fr"] = _esios_to_panel_tz(fr, timezone)

    panel = pd.DataFrame(columns)
    panel.index.name = "datetime"

    if v8_exogenous:
        exog = _load_v8_exogenous(v8_parquet_path, columns=v8_exogenous, timezone=timezone)
        panel = panel.join(exog, how="left")

    panel = _slice_dates(panel, start, end)

    if reindex_hourly:
        panel = _reindex_hourly(panel)

    if drop_target_nan:
        panel = panel[panel[target_col].notna()]

    return panel


def load_cid_panel(
    parquet_path: str | Path | None = None,
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    target_col: str = "mic_price",
    feature_cols: Sequence[str] | None = None,
    timezone: str | None = "Europe/Madrid",
    reindex_hourly: bool = True,
    add_missing_flag: bool = True,
) -> pd.DataFrame:
    """Load the CID panel from ``features_2022_2024.parquet``.

    The target ``mic_price`` (ESIOS 1727) is unaffected by the v8 label
    bug. ``price_es`` and ``price_fr`` in the returned panel **do**
    inherit the bug from the upstream parquet (price_es is actually
    Portugal/MIBEL-coupled, price_fr is actually Spain). This is
    documented and will be addressed when sub-objective 3.3 is built out.
    """
    path = _resolve_path(parquet_path, "MIBEL_CID_PARQUET")
    df = pd.read_parquet(path)
    df = _attach_datetime_index(df, timestamp_col="timestamp", timezone=timezone)

    feats = tuple(feature_cols) if feature_cols is not None else CID_DEFAULT_COLS
    missing = [c for c in (target_col, *feats) if c not in df.columns]
    if missing:
        raise KeyError(f"Columns missing from CID parquet: {missing}")

    cols = [target_col, *feats]
    df = df[cols]

    if add_missing_flag:
        df = df.assign(mic_price_missing=df[target_col].isna().astype("int8"))

    df = _slice_dates(df, start, end)
    if reindex_hourly:
        df = _reindex_hourly(df)
        if add_missing_flag:
            df["mic_price_missing"] = df["mic_price_missing"].fillna(1).astype("int8")
    return df
