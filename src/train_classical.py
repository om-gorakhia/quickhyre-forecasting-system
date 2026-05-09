"""Training loop for classical (per-state) models: SARIMA and Prophet.

Trains each model on each state independently, evaluates on the validation
split, and persists artifacts. One state failure doesn't crash the run.
"""

import logging
import time

import numpy as np
import pandas as pd

from config.settings import (
    DATE_COL, STATE_COL, TARGET_COL,
    FORECAST_WEEKS, ARTIFACTS_DIR,
)
from src.models.base import ForecastModel
from src.models.sarima import SARIMAModel
from src.models.prophet_model import ProphetModel
from src.dataset import get_state_series
from src.evaluate import compute_metrics, format_metrics

logger = logging.getLogger(__name__)


def _train_single_state(
    model: ForecastModel,
    state: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> dict | None:
    """Train one model on one state. Returns metrics dict or None on failure."""
    series_train = get_state_series(train_df, state)
    series_val = get_state_series(val_df, state)

    if len(series_train) < 20:
        logger.warning("[%s][%s] Only %d training rows — skipping", model.name, state, len(series_train))
        return None

    t0 = time.time()
    try:
        model.fit(series_train)
    except Exception as e:
        logger.error("[%s][%s] Fit failed: %s", model.name, state, e)
        return None
    fit_time = time.time() - t0

    # Predict over validation horizon
    try:
        preds = model.predict(horizon=len(series_val))
    except Exception as e:
        logger.error("[%s][%s] Predict failed: %s", model.name, state, e)
        return None

    # Align predictions with actual validation dates
    actual = series_val[TARGET_COL].values
    predicted = preds["forecast"].values

    # If prediction length doesn't match val length, trim to shorter
    n = min(len(actual), len(predicted))
    actual, predicted = actual[:n], predicted[:n]

    metrics = compute_metrics(actual, predicted, train_series=series_train[TARGET_COL].values)
    metrics["state"] = state
    metrics["model"] = model.name
    metrics["fit_time_s"] = round(fit_time, 2)
    metrics["n_train"] = len(series_train)
    metrics["n_val"] = n

    return metrics, model, preds


def train_model_all_states(
    model_cls: type,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Train a model class across all states. Returns metrics DataFrame."""
    states = sorted(train_df[STATE_COL].unique())
    all_metrics = []
    model_dir = ARTIFACTS_DIR / "models"
    preds_dir = ARTIFACTS_DIR / "predictions"
    model_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = model_kwargs or {}
    dummy = model_cls(**model_kwargs)
    logger.info("=== Training %s across %d states ===", dummy.name, len(states))
    t_total = time.time()

    for i, state in enumerate(states, 1):
        model = model_cls(**model_kwargs)
        result = _train_single_state(model, state, train_df, val_df)

        if result is None:
            continue

        metrics, fitted_model, preds = result
        all_metrics.append(metrics)

        # Save artifacts
        safe_state = state.lower().replace(" ", "_")
        fitted_model.save(model_dir / f"{dummy.name}_{safe_state}.pkl")
        preds.to_csv(preds_dir / f"{dummy.name}_{safe_state}_val.csv", index=False)

        if i % 10 == 0 or i == len(states):
            logger.info(
                "[%s] %d/%d states done | latest: %s → %s",
                dummy.name, i, len(states), state, format_metrics(metrics),
            )

    elapsed = time.time() - t_total
    metrics_df = pd.DataFrame(all_metrics)

    if len(metrics_df) > 0:
        logger.info(
            "[%s] Complete: %d/%d states in %.1fs | median sMAPE=%.2f%%",
            dummy.name, len(metrics_df), len(states), elapsed,
            metrics_df["smape"].median(),
        )
        # Save aggregated metrics
        metrics_df.to_csv(ARTIFACTS_DIR / "metrics" / f"{dummy.name}_metrics.csv", index=False)
    else:
        logger.error("[%s] All states failed!", dummy.name)

    return metrics_df
