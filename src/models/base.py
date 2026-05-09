"""Abstract base for all forecast models.

Every model wrapper must implement fit/predict/name so the training
loop and evaluation pipeline can treat them uniformly.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class ForecastModel(ABC):
    """Common interface for per-state forecasters."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> "ForecastModel":
        """Fit on a single state's training data.
        train_df has columns: date, sales (sorted ascending).
        """
        ...

    @abstractmethod
    def predict(self, horizon: int) -> pd.DataFrame:
        """Forecast `horizon` steps beyond training data.
        Returns df with columns: date, forecast.
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "ForecastModel":
        ...

    def predict_insample(self, train_df: pd.DataFrame) -> pd.Series | None:
        """Optional: return fitted values on training data for diagnostics."""
        return None
