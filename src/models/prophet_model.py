"""Prophet forecaster.

Tuning choices:
- yearly_seasonality=True: the main seasonal pattern in beverage sales
- weekly_seasonality=False: data is weekly-aggregated, sub-weekly patterns
  are already averaged out. Enabling it would fit noise.
- seasonality_mode='multiplicative': beverage sales scale with magnitude
  (California's holiday bump is 10x Wyoming's). Multiplicative captures this
  better than additive.
- US holidays added via Prophet's built-in holiday support.
- changepoint_prior_scale=0.05: default is reasonable. Lower values = smoother
  trend. We keep the default and let Prophet's MAP estimation handle it.
- growth='linear': no saturation ceiling for dollar sales.

Prophet handles missing dates internally, but we feed it a complete weekly
series anyway for consistency with other models.
"""

import logging
import pickle
import warnings
from pathlib import Path

import pandas as pd

from src.models.base import ForecastModel

logger = logging.getLogger(__name__)


class ProphetModel(ForecastModel):
    """Per-state Prophet wrapper."""

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        seasonality_mode: str = "multiplicative",
    ):
        self._model = None
        self._yearly = yearly_seasonality
        self._weekly = weekly_seasonality
        self._mode = seasonality_mode
        self._freq = None

    @property
    def name(self) -> str:
        return "prophet"

    def fit(self, train_df: pd.DataFrame) -> "ProphetModel":
        from prophet import Prophet

        df = train_df.rename(columns={"date": "ds", "sales": "y"})
        self._freq = pd.infer_freq(df["ds"])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = Prophet(
                growth="linear",
                yearly_seasonality=self._yearly,
                weekly_seasonality=self._weekly,
                daily_seasonality=False,
                seasonality_mode=self._mode,
            )
            self._model.add_country_holidays(country_name="US")
            self._model.fit(df)

        return self

    def predict(self, horizon: int) -> pd.DataFrame:
        future = self._model.make_future_dataframe(
            periods=horizon, freq=self._freq or "W-SAT"
        )
        raw = self._model.predict(future)
        # Return only the forecast horizon (last `horizon` rows)
        out = raw.tail(horizon)[["ds", "yhat"]].copy()
        out.columns = ["date", "forecast"]
        out = out.reset_index(drop=True)
        return out

    def predict_insample(self, train_df: pd.DataFrame) -> pd.Series:
        df = train_df.rename(columns={"date": "ds", "sales": "y"})
        fitted = self._model.predict(df)
        return pd.Series(fitted["yhat"].values, index=train_df.index)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "ProphetModel":
        with open(path, "rb") as f:
            return pickle.load(f)
