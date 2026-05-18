"""Common interface for EPF models in this package.

Every concrete model implements :class:`Model`: ``fit(train_df)`` followed
by ``predict(test_df)``. The rolling-window evaluator
(:mod:`mibel_forecasting.evaluation.recalibration`) talks to models only
through this protocol, so swapping ``SeasonalNaive`` for ``LEAR`` or
``DNN`` requires no changes to the surrounding loop.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Model(Protocol):
    """Day-ahead price forecaster.

    ``fit`` receives the training panel (target column plus any exogenous
    columns the model wants). ``predict`` receives the test panel and
    returns a forecast series aligned to ``test_df.index``.
    """

    def fit(self, train_df: pd.DataFrame) -> Model:  # pragma: no cover - protocol
        ...

    def predict(self, test_df: pd.DataFrame) -> pd.Series:  # pragma: no cover - protocol
        ...
