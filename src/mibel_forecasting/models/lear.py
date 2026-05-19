"""LEAR — Lasso Estimated AutoRegressive day-ahead price forecaster.

Port of the LEAR specification of Lago et al. (2021), Section 3.2,
implemented from scratch (no ``epftoolbox`` import). The reference
implementation is consulted only for behavioural parity:
https://github.com/jeslago/epftoolbox.

Model in one paragraph
----------------------
LEAR fits **24 independent Lasso regressions**, one per hour of the
target day D. The feature vector for day D is the same for all 24
hours (so feature construction is amortised across them):

- **Price lags** — the 24-hour profile of the realised price for days
  D-1, D-2, D-3 and D-7. → 96 features.
- **Exogenous variables** — for each exogenous column, the 24-hour
  profile on D-1, D-7 and **on D itself** (since exogenous DAM inputs
  are day-ahead forecasts known at gate-closure). → 72 per exogenous.
- **Day-of-week dummies** — 7 binary columns.

With the default ``exogenous_cols=("es_demand_fc", "es_wind_fc")`` this
gives 96 + 2·72 + 7 = **247 features**, matching Lago 2021 exactly.

Pre-processing follows the paper:

1. The target prices and the exogenous (non-dummy) features are
   transformed with the **arcsinh-median** (a.k.a. *Invariant*)
   transform — robust to negative prices and outliers and stable across
   recalibrations.
2. Per-hour ``LassoLarsIC(criterion="aic")`` picks ``alpha``; the final
   coefficients come from a fresh ``Lasso`` fit at that alpha.

Rolling recalibration is handled by the generic
:func:`mibel_forecasting.evaluation.recalibration.rolling_forecast`
loop: every call rebuilds the 24 per-hour models on the most recent
``calibration_window`` worth of history.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoLarsIC

PRICE_LAGS_DAYS: tuple[int, ...] = (1, 2, 3, 7)
EXOG_LAGGED_DAYS: tuple[int, ...] = (1, 7)
N_HOURS: int = 24
N_DOW: int = 7


def _arcsinh_median(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the Invariant (arcsinh-median) scaler column-wise.

    Returns ``(scaled, median, mad)`` so the same transform can be
    re-applied to test data and inverted at predict time.
    """
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median), axis=0)
    mad = np.where(mad == 0, 1.0, mad)  # avoid div-by-zero on flat columns
    scaled = np.arcsinh((x - median) / mad)
    return scaled, median, mad


def _arcsinh_median_apply(x: np.ndarray, median: np.ndarray, mad: np.ndarray) -> np.ndarray:
    return np.arcsinh((x - median) / mad)


def _arcsinh_median_invert(y: np.ndarray, median: np.ndarray, mad: np.ndarray) -> np.ndarray:
    return np.sinh(y) * mad + median


def _day_index(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Calendar-day index (tz-naive, midnight) for grouping."""
    naive = idx.tz_localize(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(pd.to_datetime(naive.date), name="day")


_HOUR_COLS: list[str] = [f"h{h:02d}" for h in range(N_HOURS)]


def _pivot_one(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Long → wide (one row per day, 24 columns ``h00..h23``).

    ``dropna=False`` is important at predict time, when the target
    column may be all-NaN — we still want the day's row to appear so
    the exogenous lookup lines up.

    The returned frame is reindexed to the canonical 24-column schema
    ``h00..h23`` so that days missing some hours (e.g. a test slice
    cut to 22 of 24 hours after ``.dropna()`` over unsafe v8 columns)
    surface those gaps as explicit NaN rather than as silently
    truncated column lists. The downstream ``dropna(how='any')`` filter
    then correctly excludes partial days.
    """
    work = pd.DataFrame(
        {
            "__day": _day_index(df.index),
            "__hour": df.index.hour,
            col: df[col].to_numpy(),
        }
    )
    w = work.pivot_table(
        index="__day", columns="__hour", values=col, aggfunc="first", dropna=False
    )
    w.columns = [f"h{int(h):02d}" for h in w.columns]
    return w.reindex(columns=_HOUR_COLS)


def _pivot_hourly(
    df: pd.DataFrame,
    target_col: str,
    exogenous_cols: Sequence[str],
    ti_cols: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Reshape a long hourly panel into ``(n_days x 24)`` wide tables.

    Used at fit time. Days with fewer than 24 hours of the target, any
    exogenous, or any technical-indicator column (DST spring-forward,
    gaps, NaN propagation through ROC's ``p_{t-n}=0`` cells) are
    dropped to keep the feature matrix rectangular.
    """
    price_w = _pivot_one(df, target_col)
    exog_w = {c: _pivot_one(df, c) for c in exogenous_cols}
    ti_w = {c: _pivot_one(df, c) for c in ti_cols}
    full = price_w.dropna(how="any").index
    for w in exog_w.values():
        full = full.intersection(w.dropna(how="any").index)
    for w in ti_w.values():
        full = full.intersection(w.dropna(how="any").index)
    return (
        price_w.loc[full],
        {c: w.loc[full] for c, w in exog_w.items()},
        {c: w.loc[full] for c, w in ti_w.items()},
    )


def _pivot_for_predict(
    df: pd.DataFrame, target_col: str, exogenous_cols: Sequence[str]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Same as :func:`_pivot_hourly` but tolerant of NaN targets.

    At prediction time the target column may be entirely NaN (we are
    about to forecast it); we still want the exogenous wide tables to
    line up day-by-day so the lookup of D-1/D-7 can succeed.
    """
    price_w = _pivot_one(df, target_col)
    exog_w = {c: _pivot_one(df, c) for c in exogenous_cols}
    full = price_w.index
    for w in exog_w.values():
        full = full.intersection(w.dropna(how="any").index)
    return price_w.loc[full], {c: w.loc[full] for c, w in exog_w.items()}


def _feature_row_for_day(
    day: pd.Timestamp,
    price_w: pd.DataFrame,
    exog_w: dict[str, pd.DataFrame],
    ti_w: dict[str, pd.DataFrame] | None = None,
) -> np.ndarray | None:
    """Build the feature vector for one target day D. ``None`` if any
    required day is missing from the wide tables.

    Layout (in order): 4 price lags x 24 hours = 96; for each exog,
    [D-1, D-7, D] x 24 = 72; for each technical indicator, [D] x 24 =
    24 (TIs already carry a 1-day lag inside ``compute_technical_indicators``,
    so day D is looked up directly with no second shift); 7 DoW
    dummies. Default (no TIs, demand+wind exog): 247 features, Lago.
    """
    if ti_w is None:
        ti_w = {}
    feats: list[np.ndarray] = []
    for lag in PRICE_LAGS_DAYS:
        d = day - pd.Timedelta(days=lag)
        if d not in price_w.index:
            return None
        feats.append(price_w.loc[d].to_numpy(dtype=float))
    for w in exog_w.values():
        for lag in EXOG_LAGGED_DAYS:
            d = day - pd.Timedelta(days=lag)
            if d not in w.index:
                return None
            feats.append(w.loc[d].to_numpy(dtype=float))
        if day not in w.index:
            return None
        feats.append(w.loc[day].to_numpy(dtype=float))
    for w in ti_w.values():
        if day not in w.index:
            return None
        feats.append(w.loc[day].to_numpy(dtype=float))
    dow = np.zeros(N_DOW, dtype=float)
    dow[day.dayofweek] = 1.0
    feats.append(dow)
    return np.concatenate(feats)


def _build_xy(
    price_w: pd.DataFrame,
    exog_w: dict[str, pd.DataFrame],
    *,
    target_days: pd.DatetimeIndex,
    ti_w: dict[str, pd.DataFrame] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[pd.Timestamp]]:
    """Assemble the (X, Y, days) training matrices over the given target days.

    Y has shape ``(n_days, 24)``: each row is the realised 24-hour price
    profile of that target day.
    """
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    used_days: list[pd.Timestamp] = []
    for day in target_days:
        if day not in price_w.index:
            continue
        feat = _feature_row_for_day(day, price_w, exog_w, ti_w)
        if feat is None:
            continue
        rows.append(feat)
        targets.append(price_w.loc[day].to_numpy(dtype=float))
        used_days.append(day)
    if not rows:
        return np.empty((0, 0)), np.empty((0, 24)), []
    X = np.vstack(rows)
    Y = np.vstack(targets)
    return X, Y, used_days


class LEAR:
    """Day-ahead price forecaster following Lago et al. (2021), §3.2.

    Parameters
    ----------
    target_col
        Column to forecast (e.g. ``"price_es"``).
    exogenous_cols
        Hourly columns used as exogenous inputs. Default is
        ``("es_demand_fc", "es_wind_fc")`` which yields exactly 247
        features and matches the canonical Lago benchmark.
    """

    def __init__(
        self,
        *,
        target_col: str,
        exogenous_cols: Sequence[str] = ("es_demand_fc", "es_wind_fc"),
        ti_cols: Sequence[str] = (),
    ) -> None:
        # Empty exogenous is allowed: LEAR collapses to a 96-price-lag +
        # 7-DoW autoregressive Lasso (103 features). Useful as a sanity
        # control to isolate the marginal contribution of the exogenous
        # features.
        #
        # ``ti_cols`` names pre-computed technical-indicator columns that
        # the caller must have joined into the panel before calling
        # ``fit`` / ``predict`` (see
        # ``mibel_forecasting.features.technical_indicators.compute_technical_indicators``).
        # TI columns enter the per-day feature row with a single 24-hour
        # block looked up at day D (the TIs already carry an internal
        # 1-day shift, so no further lag is applied here).
        self.target_col = target_col
        self.exogenous_cols = tuple(exogenous_cols)
        self.ti_cols = tuple(ti_cols)
        self._lassos: dict[int, Lasso] = {}
        self._x_median: np.ndarray | None = None
        self._x_mad: np.ndarray | None = None
        self._y_median: np.ndarray | None = None
        self._y_mad: np.ndarray | None = None
        self._n_features: int | None = None
        self._n_dummy_features: int = N_DOW
        self._price_w: pd.DataFrame | None = None
        self._exog_w: dict[str, pd.DataFrame] | None = None
        self._ti_w: dict[str, pd.DataFrame] | None = None

    @property
    def n_features(self) -> int:
        return (
            96
            + len(self.exogenous_cols) * 72
            + len(self.ti_cols) * N_HOURS
            + N_DOW
        )

    def _scale_x(self, X: np.ndarray, *, fit: bool) -> np.ndarray:
        """Apply arcsinh-median to the non-dummy block; leave dummies untouched."""
        n_non_dummy = self.n_features - self._n_dummy_features
        non_dummy = X[:, :n_non_dummy]
        dummy = X[:, n_non_dummy:]
        if fit:
            scaled, median, mad = _arcsinh_median(non_dummy)
            self._x_median, self._x_mad = median, mad
        else:
            assert self._x_median is not None and self._x_mad is not None
            scaled = _arcsinh_median_apply(non_dummy, self._x_median, self._x_mad)
        return np.hstack([scaled, dummy])

    def _scale_y(self, Y: np.ndarray, *, fit: bool) -> np.ndarray:
        if fit:
            scaled, median, mad = _arcsinh_median(Y)
            self._y_median, self._y_mad = median, mad
            return scaled
        assert self._y_median is not None and self._y_mad is not None
        return _arcsinh_median_apply(Y, self._y_median, self._y_mad)

    def fit(self, train_df: pd.DataFrame) -> LEAR:
        if self.target_col not in train_df.columns:
            raise KeyError(f"target_col {self.target_col!r} missing from train_df")
        missing = [c for c in self.exogenous_cols if c not in train_df.columns]
        if missing:
            raise KeyError(f"exogenous_cols missing from train_df: {missing}")
        missing_ti = [c for c in self.ti_cols if c not in train_df.columns]
        if missing_ti:
            raise KeyError(f"ti_cols missing from train_df: {missing_ti}")

        price_w, exog_w, ti_w = _pivot_hourly(
            train_df, self.target_col, self.exogenous_cols, self.ti_cols
        )
        self._price_w, self._exog_w, self._ti_w = price_w, exog_w, ti_w

        X, Y, _days = _build_xy(
            price_w, exog_w, target_days=price_w.index, ti_w=ti_w
        )
        if X.shape[0] == 0:
            raise ValueError("Not enough history to build any LEAR training row")
        self._n_features = X.shape[1]
        if self._n_features != self.n_features:
            raise AssertionError(
                f"Feature-vector size mismatch: built {self._n_features}, expected {self.n_features}"
            )

        Xs = self._scale_x(X, fit=True)
        Ys = self._scale_y(Y, fit=True)

        n_samples, n_feat = Xs.shape
        # LassoLarsIC's AIC criterion needs an estimate of the noise variance.
        # When the user does not pass one, sklearn fits an unregularised OLS
        # and uses its residual variance. That OLS path needs comfortably more
        # samples than features — sklearn itself raises "samples is smaller
        # than features" when ``n_samples <= n_features``, and the estimate is
        # noisy when n_samples sits just above n_features. We therefore fall
        # back to a unit-variance prior unless n_samples exceeds n_features by
        # at least ``N_HOURS`` rows of safety margin. The prior is defensible
        # because Y has been arcsinh-median scaled (O(1) by construction).
        noise_variance_kw = (
            {} if n_samples > n_feat + N_HOURS else {"noise_variance": 1.0}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for h in range(N_HOURS):
                y_h = Ys[:, h]
                lars = LassoLarsIC(
                    criterion="aic", max_iter=2500, **noise_variance_kw
                ).fit(Xs, y_h)
                alpha = max(lars.alpha_, 1e-8)
                lasso = Lasso(alpha=alpha, max_iter=2500).fit(Xs, y_h)
                self._lassos[h] = lasso
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        """Predict the 24-hour price profile for each day in ``test_df``.

        **Price-lag features come exclusively from the training history**
        captured at ``fit`` time — the target column of ``test_df`` is
        never read, so leakage from realised prices inside the test
        window is impossible regardless of the test horizon.

        Exogenous features for day D itself are read from ``test_df``
        (day-ahead forecasts known at gate-closure); their D-1 and D-7
        lags fall in the union of training and test history, with the
        latter taking precedence on overlap.

        Implications:

        - The realised target of the test window has zero effect on the
          predictions; passing it as ``NaN`` (the convention used by the
          rolling-forecast loop) is functionally identical to passing
          the true values.
        - Multi-day test horizons are leakage-free for the same reason —
          day N+1 cannot see day N's realised price because price lags
          only look at training history.
        - If, for some application, you need price lags that include the
          test window's earlier days (e.g. a multi-day forecast where
          the model is allowed to look at recently-predicted prices),
          do that explicitly outside this method.

        **Incomplete test days are skipped uniformly across LEAR
        configurations.** A day with fewer than 24 hours in ``test_df``
        (e.g. after a strict ``.dropna()`` over unsafe v8 columns) is
        not predicted — its rows stay NaN in the output Series. This
        rule applies whether or not the model has exogenous columns,
        so the rMAE of ``ar-only`` and ``demand+wind`` is averaged over
        the same set of days in any robustness comparison. The list of
        skipped days is exposed for diagnostics via the return value
        (NaN positions) rather than via a separate API.
        """
        if not self._lassos:
            raise RuntimeError("call fit() before predict()")
        assert self._price_w is not None and self._exog_w is not None

        # Partial-day filter (uniform across all configurations).
        # A test day is "full" iff test_df carries all 24 hours of it.
        day_idx = _day_index(test_df.index)
        hours_per_day = pd.Series(day_idx).value_counts()
        full_test_days = pd.DatetimeIndex(
            sorted(d for d, n in hours_per_day.items() if n == N_HOURS),
            name="day",
        )

        # Only the exogenous and technical-indicator columns are read
        # from test_df; the target column is intentionally ignored.
        test_exog_w = {c: _pivot_one(test_df, c) for c in self.exogenous_cols}
        test_ti_w = {c: _pivot_one(test_df, c) for c in self.ti_cols}
        if test_exog_w:
            # Defence-in-depth: even on a full-hour day, drop it if any
            # exogenous cell came in as NaN (should not happen post-dropna
            # but the cost of the check is negligible).
            clean_exog_days = full_test_days
            for w in test_exog_w.values():
                clean_exog_days = clean_exog_days.intersection(
                    w.dropna(how="any").index
                )
            test_exog_w = {c: w.loc[clean_exog_days] for c, w in test_exog_w.items()}
            full_test_days = clean_exog_days
        if test_ti_w:
            # Same defence-in-depth on TI columns. A test day with any
            # NaN TI cell (e.g. ROC ``p_{t-n}=0`` near a zero-price hour
            # in the panel) is excluded from prediction; honest reporting
            # of dropped days happens via the rolling-forecast output.
            clean_ti_days = full_test_days
            for w in test_ti_w.values():
                clean_ti_days = clean_ti_days.intersection(
                    w.dropna(how="any").index
                )
            test_ti_w = {c: w.loc[clean_ti_days] for c, w in test_ti_w.items()}
            full_test_days = clean_ti_days

        # Price lookup uses training history only — never test_df.
        full_price_w = self._price_w
        # Exogenous lookup unions train history and the (full-day-filtered)
        # test window. Test wins on overlap.
        full_exog_w: dict[str, pd.DataFrame] = {}
        for col in self.exogenous_cols:
            merged = pd.concat([self._exog_w[col], test_exog_w[col]])
            full_exog_w[col] = merged[~merged.index.duplicated(keep="last")].sort_index()
        # TI lookup also unions train + test; test wins on overlap.
        full_ti_w: dict[str, pd.DataFrame] = {}
        assert self._ti_w is not None
        for col in self.ti_cols:
            merged = pd.concat([self._ti_w[col], test_ti_w[col]])
            full_ti_w[col] = merged[~merged.index.duplicated(keep="last")].sort_index()

        out = pd.Series(index=test_df.index, dtype=float, name="y_pred")
        for day in full_test_days:
            feat = _feature_row_for_day(day, full_price_w, full_exog_w, full_ti_w)
            if feat is None:
                continue
            Xs = self._scale_x(feat.reshape(1, -1), fit=False)
            preds_scaled = np.array(
                [self._lassos[h].predict(Xs)[0] for h in range(N_HOURS)]
            )
            preds = _arcsinh_median_invert(preds_scaled, self._y_median, self._y_mad)
            day_mask = _day_index(test_df.index) == day
            for ts in test_df.index[day_mask]:
                out.loc[ts] = float(preds[ts.hour])
        return out
