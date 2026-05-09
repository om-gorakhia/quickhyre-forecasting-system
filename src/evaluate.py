"""Forecast evaluation metrics.

All metrics operate on aligned (actual, predicted) arrays.
sMAPE is the primary metric for cross-state, cross-model comparison
because it's scale-invariant and symmetric.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error. Undefined when actual contains zeros."""
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric MAPE — bounded [0, 200], handles near-zero values better."""
    denom = np.abs(actual) + np.abs(predicted)
    mask = denom > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(2.0 * np.abs(actual[mask] - predicted[mask]) / denom[mask]) * 100)


def mase(actual: np.ndarray, predicted: np.ndarray, train_series: np.ndarray, seasonal_period: int = 52) -> float:
    """Mean Absolute Scaled Error — scale-free, uses seasonal naive forecast as baseline.
    MASE < 1 means the model beats the seasonal naive forecast.
    seasonal_period=52 for weekly data (compare against same-week-last-year).
    Falls back to period=1 if series is shorter than one seasonal cycle.
    """
    if len(train_series) <= seasonal_period:
        seasonal_period = 1
    naive_errors = np.abs(train_series[seasonal_period:] - train_series[:-seasonal_period])
    scale = naive_errors.mean()
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(actual - predicted)) / scale)


def compute_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    train_series: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all metrics. Returns a dict suitable for logging/serialization."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    result = {
        "mae": mae(actual, predicted),
        "rmse": rmse(actual, predicted),
        "mape": mape(actual, predicted),
        "smape": smape(actual, predicted),
    }
    if train_series is not None:
        result["mase"] = mase(actual, predicted, np.asarray(train_series, dtype=np.float64))
    return result


def format_metrics(metrics: dict[str, float]) -> str:
    NUMERIC_KEYS = {"mae", "rmse", "mape", "smape", "mase", "fit_time_s"}
    parts = []
    for k, v in metrics.items():
        if k not in NUMERIC_KEYS:
            continue
        if k in ("mape", "smape"):
            parts.append(f"{k}={v:.2f}%")
        elif k == "mase":
            parts.append(f"{k}={v:.3f}")
        else:
            parts.append(f"{k}={v:,.0f}")
    return " | ".join(parts)
