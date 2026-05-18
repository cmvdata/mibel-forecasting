"""Diebold-Mariano test for comparing two forecasts on the same series.

Implements the standard Diebold-Mariano (1995) statistic for the
loss differential ``d_t = L(e1_t) - L(e2_t)``, with the long-run
variance estimated by Newey-West with the Andrews (1991) lag rule
``floor(4 · (n/100)^(2/9))``, lifted to at least ``h-1`` for an
``h``-step forecast horizon (Diebold-Mariano 1995, footnote 9).

The implementation delegates the HAC covariance to ``statsmodels``,
which is the same approach used by ``scripts/benchmarks_dm.py`` in the
sibling ``mibel-congestion-monitor`` repository — kept in sync so
results are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _newey_west_lag(n: int, horizon: int) -> int:
    """Andrews (1991) data-dependent lag, floored at ``horizon - 1``."""
    base = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    return max(base, max(horizon - 1, 0))


@dataclass(frozen=True)
class DMResult:
    """Output of :func:`diebold_mariano`."""

    statistic: float
    p_value: float
    mean_loss_diff: float
    n_obs: int
    horizon: int
    newey_west_lag: int
    power: int

    def reject_h0(self, alpha: float = 0.05) -> bool:
        """Whether the two-sided H0 of equal predictive accuracy is rejected."""
        return self.p_value < alpha


def diebold_mariano(
    y_true: pd.Series,
    pred1: pd.Series,
    pred2: pd.Series,
    *,
    horizon: int = 1,
    power: int = 1,
) -> DMResult:
    """Two-sided Diebold-Mariano test, ``pred1`` vs ``pred2``.

    The loss is ``|e|^power``. Common choices are ``power=1`` (MAE-style
    loss) and ``power=2`` (MSE-style). ``horizon`` is the forecast
    horizon in number of step-ahead periods and only enters via the
    minimum Newey-West lag.

    ``statistic > 0`` means pred1 has the higher loss — i.e. pred2 wins.
    Pair the series on a common index before passing; rows with any
    NaN are dropped pairwise.
    """
    if power not in (1, 2):
        raise ValueError(f"power must be 1 or 2, got {power}")
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    df = pd.concat({"y": y_true, "p1": pred1, "p2": pred2}, axis=1).dropna()
    if len(df) < 8:
        raise ValueError(f"Too few observations for DM ({len(df)}); need at least 8")

    e1 = (df["y"] - df["p1"]).to_numpy()
    e2 = (df["y"] - df["p2"]).to_numpy()
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    lag = _newey_west_lag(n, horizon)

    # Degenerate case: identical predictions → zero loss differential → the
    # HAC t-stat is 0/0. Report a non-rejection cleanly.
    if np.all(d == 0):
        return DMResult(
            statistic=0.0,
            p_value=1.0,
            mean_loss_diff=0.0,
            n_obs=n,
            horizon=horizon,
            newey_west_lag=lag,
            power=power,
        )

    model = sm.OLS(d, np.ones((n, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    stat = float(model.tvalues[0])
    pval = float(model.pvalues[0])

    return DMResult(
        statistic=stat,
        p_value=pval,
        mean_loss_diff=float(d.mean()),
        n_obs=n,
        horizon=horizon,
        newey_west_lag=lag,
        power=power,
    )
