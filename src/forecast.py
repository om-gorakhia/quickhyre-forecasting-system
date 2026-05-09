"""Generate 8-week production forecasts using the selected best model per state.

Loads trained model artifacts, runs recursive prediction per state using
each state's winner, and optionally blends top-2 via ensemble.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    DATE_COL, STATE_COL, TARGET_COL,
    FORECAST_WEEKS, ARTIFACTS_DIR,
)
from src.dataset import StateScaler
from src.selector import compute_ensemble_weights

logger = logging.getLogger(__name__)


def _load_model(model_name: str, state: str):
    """Load a fitted model for a given state.

    Classical models (sarima, prophet) are per-state pickles.
    Global models (xgboost, lstm) are single artifacts.
    """
    safe_state = state.lower().replace(" ", "_")

    if model_name == "sarima":
        from src.models.sarima import SARIMAModel
        return SARIMAModel.load(ARTIFACTS_DIR / "models" / f"sarima_{safe_state}.pkl")

    elif model_name == "prophet":
        from src.models.prophet_model import ProphetModel
        return ProphetModel.load(ARTIFACTS_DIR / "models" / f"prophet_{safe_state}.pkl")

    elif model_name == "xgboost":
        from src.models.xgboost_model import XGBoostForecaster
        return XGBoostForecaster.load(ARTIFACTS_DIR / "models" / "xgboost_global.pkl")

    elif model_name == "lstm":
        from src.models.lstm_model import LSTMForecaster
        scaler = StateScaler.load(ARTIFACTS_DIR / "scaler.pkl")
        return LSTMForecaster.load(ARTIFACTS_DIR / "models" / "lstm_global.pt", scaler)

    raise ValueError(f"Unknown model: {model_name}")


def _predict_state(model, model_name: str, state: str, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Run prediction for a single state, dispatching to the right API."""
    if model_name in ("sarima", "prophet"):
        return model.predict(horizon=horizon)
    elif model_name == "xgboost":
        return model.predict_recursive(history_df, state, horizon=horizon)
    elif model_name == "lstm":
        return model.predict_recursive(history_df, state, horizon=horizon)
    raise ValueError(f"Unknown model: {model_name}")


def generate_forecasts(
    selection: pd.DataFrame,
    history_df: pd.DataFrame,
    horizon: int = FORECAST_WEEKS,
    use_ensemble: bool = True,
) -> pd.DataFrame:
    """Generate forecasts for all states using selected models.

    Args:
        selection: DataFrame from selector.select_best_per_state()
        history_df: full feature-enriched training data
        horizon: weeks to forecast
        use_ensemble: if True, blend top-2 where eligible

    Returns:
        DataFrame with columns: state, date, forecast, model_used
    """
    all_forecasts = []
    # Cache loaded global models to avoid re-loading per state
    _model_cache = {}

    for _, row in selection.iterrows():
        state = row["state"]
        best_name = row["best_model"]

        try:
            # Load/cache best model
            if best_name not in _model_cache:
                _model_cache[best_name] = _load_model(best_name, state)
            elif best_name in ("sarima", "prophet"):
                # Per-state models must be loaded individually
                _model_cache[best_name] = _load_model(best_name, state)

            best_model = _model_cache[best_name]
            best_preds = _predict_state(best_model, best_name, state, history_df, horizon)

            # Ensemble with second model if eligible
            if use_ensemble and row.get("ensemble_eligible", False) and pd.notna(row.get("second_model")):
                second_name = row["second_model"]
                try:
                    if second_name not in _model_cache:
                        _model_cache[second_name] = _load_model(second_name, state)
                    elif second_name in ("sarima", "prophet"):
                        _model_cache[second_name] = _load_model(second_name, state)

                    second_model = _model_cache[second_name]
                    second_preds = _predict_state(second_model, second_name, state, history_df, horizon)

                    # Blend
                    w1, w2 = compute_ensemble_weights(row["best_smape"], row["second_smape"])
                    n = min(len(best_preds), len(second_preds))
                    blended = (
                        best_preds["forecast"].values[:n] * w1 +
                        second_preds["forecast"].values[:n] * w2
                    )
                    best_preds = best_preds.head(n).copy()
                    best_preds["forecast"] = blended
                    model_label = f"ensemble({best_name}:{w1:.2f}+{second_name}:{w2:.2f})"
                except Exception as e:
                    logger.warning("[%s] Ensemble fallback to %s: %s", state, best_name, e)
                    model_label = best_name
            else:
                model_label = best_name

            best_preds["state"] = state
            best_preds["model_used"] = model_label
            all_forecasts.append(best_preds)

        except Exception as e:
            logger.error("[%s] Forecast failed: %s", state, e)

    result = pd.concat(all_forecasts, ignore_index=True)
    result = result[["state", "date", "forecast", "model_used"]]
    logger.info("Forecasts generated: %d rows for %d states", len(result), result["state"].nunique())
    return result
