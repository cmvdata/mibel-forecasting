"""Per-regime backtest runner for ``notebooks/03_lear_robustness.ipynb``.

Lives in the package (not in the notebook itself) because the notebook
calls it through ``concurrent.futures.ProcessPoolExecutor`` and Windows
multiprocessing requires worker entry points to be importable by name
— functions defined in a notebook cell are not.

The function is intentionally a thin wrapper over ``rolling_forecast``
so the parallelism boundary lives only at the regime level. Inside
``run_regime`` everything is sequential, which keeps the model code
unchanged and avoids per-day parallelism that would complicate
reproducibility.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd

from mibel_forecasting.data.loaders import load_dam_panel
from mibel_forecasting.evaluation.dm_test import diebold_mariano
from mibel_forecasting.evaluation.metrics import mae, smape
from mibel_forecasting.evaluation.recalibration import rolling_forecast
from mibel_forecasting.features.technical_indicators import (
    TI_COLUMNS,
    compute_technical_indicators,
)
from mibel_forecasting.models.lear import LEAR
from mibel_forecasting.models.naive import SeasonalNaive

MODEL_NAMES: tuple[str, ...] = (
    "naive",
    "LEAR ar-only",
    "LEAR demand+wind",
    "LEAR demand+solar+wind",
)

# TI-augmented variants for notebook 04. The strings here must match
# exactly the keys used by ``_factory_for`` below.
MODEL_NAMES_WITH_TI: tuple[str, ...] = (
    "naive",
    "LEAR ar-only",
    "LEAR ar-only + TI",
    "LEAR demand+wind",
    "LEAR demand+wind + TI",
    "LEAR demand+solar+wind + TI",
)


def _factory_for(name: str, target_col: str = "price_es") -> Callable:
    """Return a zero-arg factory for ``name``. Used as ``model_factory`` in
    ``rolling_forecast``. Kept here (not in the spec dict) so the dict is
    fully picklable across multiprocessing workers."""
    if name == "naive":
        return lambda: SeasonalNaive(target_col=target_col)
    if name == "LEAR ar-only":
        return lambda: LEAR(target_col=target_col, exogenous_cols=())
    if name == "LEAR ar-only + TI":
        return lambda: LEAR(
            target_col=target_col, exogenous_cols=(), ti_cols=TI_COLUMNS
        )
    if name == "LEAR demand+wind":
        return lambda: LEAR(
            target_col=target_col, exogenous_cols=("es_demand_fc", "es_wind_fc")
        )
    if name == "LEAR demand+wind + TI":
        return lambda: LEAR(
            target_col=target_col,
            exogenous_cols=("es_demand_fc", "es_wind_fc"),
            ti_cols=TI_COLUMNS,
        )
    if name == "LEAR demand+solar+wind":
        return lambda: LEAR(
            target_col=target_col,
            exogenous_cols=("es_demand_fc", "es_solar_fc", "es_wind_fc"),
        )
    if name == "LEAR demand+solar+wind + TI":
        return lambda: LEAR(
            target_col=target_col,
            exogenous_cols=("es_demand_fc", "es_solar_fc", "es_wind_fc"),
            ti_cols=TI_COLUMNS,
        )
    raise ValueError(f"unknown model name: {name!r}")


def _coverage_rows(
    regime: str, forecasts: dict[str, pd.DataFrame]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, f in forecasts.items():
        by_day = f["y_pred"].groupby(f.index.date).apply(
            lambda s: int(s.notna().sum())
        )
        rows.append(
            {
                "regime": regime,
                "model": model_name,
                "days in panel": len(by_day),
                "full days predicted": int((by_day == 24).sum()),
                "partial-hour days in panel": int(((by_day > 0) & (by_day < 24)).sum()),
                "skipped (lag missing or NaN)": int((by_day == 0).sum()),
            }
        )
    return rows


def _metric_rows(
    regime: str,
    forecasts: dict[str, pd.DataFrame],
    models: Sequence[str],
) -> list[dict[str, Any]]:
    naive_f = forecasts["naive"]
    naive_mae_val = mae(naive_f["y_true"], naive_f["y_pred"])
    rows: list[dict[str, Any]] = []
    for model_name in models:
        f = forecasts[model_name]
        m = mae(f["y_true"], f["y_pred"])
        s = smape(f["y_true"], f["y_pred"])
        r = m / naive_mae_val if naive_mae_val else float("nan")
        if model_name == "naive":
            dm_stat, dm_pval, nw_lag = float("nan"), float("nan"), float("nan")
        else:
            dm = diebold_mariano(
                f["y_true"], naive_f["y_pred"], f["y_pred"], horizon=24
            )
            dm_stat, dm_pval, nw_lag = dm.statistic, dm.p_value, dm.newey_west_lag
        rows.append(
            {
                "regime": regime,
                "model": model_name,
                "n_hours": len(f),
                "MAE (EUR/MWh)": m,
                "sMAPE (%)": s,
                "rMAE vs naive": r,
                "DM stat vs naive": dm_stat,
                "DM p-value vs naive": dm_pval,
                "NW lag": nw_lag,
            }
        )
    return rows


def cap_blas_threads() -> None:
    """``ProcessPoolExecutor`` initializer that caps BLAS thread pools at 1
    per worker process. Prevents the oversubscription / deadlock that
    stalled the first attempt at notebook 04 for >6 h on Windows.

    Defined at module scope (not in the notebook cell) so workers can
    import and call it during ``initializer=...``."""
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1)


def run_regime(spec: dict[str, Any]) -> dict[str, Any]:
    """Run the full naive + 3-LEAR-variant backtest on one regime.

    Parameters
    ----------
    spec
        Picklable dict with keys:

        - ``regime`` (str) — display label used in the output frames.
        - ``panel_start``, ``panel_end`` (date strings) — bounds passed
          to ``load_dam_panel``; the panel is loaded fresh per worker.
        - ``start``, ``end`` (date strings) — inclusive bounds on the
          test-window start times.
        - ``train_size`` (str such as ``"180D"`` or ``"365D"``).
        - ``models`` (sequence of str) — subset of ``MODEL_NAMES``.

    Returns
    -------
    dict
        Keys: ``regime``, ``forecasts`` (model_name → DataFrame),
        ``metrics`` (list of metric rows), ``coverage`` (list of
        coverage rows).
    """
    df = load_dam_panel(start=spec["panel_start"], end=spec["panel_end"]).dropna()
    if spec.get("with_ti", False):
        # Join the eight TI columns onto the panel BEFORE the per-model
        # backtest. Non-TI variants ignore them via ``ti_cols=()``; TI
        # variants reference them by name. The TIs are leakage-safe by
        # construction inside ``compute_technical_indicators`` (audit
        # ``demir_2019_ti_parameter_audit_2026_05.md``).
        ti_df = compute_technical_indicators(df)
        df = df.join(ti_df)

    models = list(spec["models"])
    forecasts: dict[str, pd.DataFrame] = {}
    for model_name in models:
        forecasts[model_name] = rolling_forecast(
            df,
            target_col="price_es",
            model_factory=_factory_for(model_name),
            train_size=spec["train_size"],
            test_size="1D",
            step="1D",
            test_start=spec["start"],
            test_end=spec["end"],
        )

    return {
        "regime": spec["regime"],
        "forecasts": forecasts,
        "metrics": _metric_rows(spec["regime"], forecasts, models),
        "coverage": _coverage_rows(spec["regime"], forecasts),
    }
