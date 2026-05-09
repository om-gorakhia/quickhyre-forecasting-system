"""SARIMA forecaster.

Tuning choices:
- Uses pmdarima.auto_arima for automatic (p,d,q)(P,D,Q,m) selection instead
  of manual grid search. This is the standard practical approach — manual
  tuning of 43 states would be fragile and not worth the effort.
- Seasonal period m=52 (weekly data, yearly cycle). We cap P,Q at 1 and D at 1
  because m=52 makes higher seasonal orders computationally brutal and unstable.
- stepwise=True for speed (greedy search, not exhaustive).
- Suppress convergence warnings — auto_arima already handles non-convergence
  by trying the next candidate.
"""

import logging
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.base import ForecastModel

logger = logging.getLogger(__name__)


class SARIMAModel(ForecastModel):
    """Per-state SARIMA via pmdarima.auto_arima."""

    def __init__(self, seasonal_period: int = 52, max_order: int = 3):
        self._model = None
        self._seasonal_period = seasonal_period
        self._max_order = max_order
        self._last_date = None
        self._freq = None

    @property
    def name(self) -> str:
        return "sarima"

    def fit(self, train_df: pd.DataFrame) -> "SARIMAModel":
        import pmdarima as pm

        y = train_df["sales"].values
        self._last_date = train_df["date"].max()
        self._freq = pd.infer_freq(train_df["date"])

        # m=52 is expensive. If the series is short relative to the seasonal
        # period, fall back to non-seasonal ARIMA.
        use_seasonal = len(y) >= self._seasonal_period * 2

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = pm.auto_arima(
                y,
                seasonal=use_seasonal,
                m=self._seasonal_period if use_seasonal else 1,
                max_p=self._max_order,
                max_q=self._max_order,
                max_d=2,
                max_P=1,
                max_Q=1,
                max_D=1,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                n_fits=30,
            )

        order = self._model.order
        seasonal = self._model.seasonal_order
        logger.debug("SARIMA fitted: order=%s seasonal=%s AIC=%.1f", order, seasonal, self._model.aic())
        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        fc = self._model.predict(n_periods=horizon)
        dates = pd.date_range(
            start=self._last_date + pd.Timedelta(weeks=1),
            periods=horizon,
            freq=self._freq or "W-SAT",
        )
        return pd.DataFrame({"date": dates, "forecast": fc})

    def predict_insample(self, train_df: pd.DataFrame) -> pd.Series:
        return pd.Series(self._model.predict_in_sample(), index=train_df.index)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "SARIMAModel":
        with open(path, "rb") as f:
            return pickle.load(f)
