"""Train XGBoost global model, evaluate per state, save artifacts.

Usage:
    python scripts/train_xgboost.py
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    DATE_COL, STATE_COL, TARGET_COL,
    ARTIFACTS_DIR, FORECAST_WEEKS,
)
from src.split import load_splits
from src.models.xgboost_model import XGBoostForecaster
from src.evaluate import compute_metrics, format_metrics
from src.dataset import get_state_series

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_xgboost")


def evaluate_per_state(
    model: XGBoostForecaster,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate the global model per state using recursive multi-step prediction.

    This tests what actually matters: how well the model forecasts into the
    future, not how well it fits known feature rows (one-step).
    """
    # Also combine train+val for history context (the model was trained on train only)
    states = sorted(val_df[STATE_COL].unique())
    all_metrics = []
    preds_dir = ARTIFACTS_DIR / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    for state in states:
        series_val = get_state_series(val_df, state)
        train_series = get_state_series(train_df, state)
        n_val = len(series_val)

        try:
            preds = model.predict_recursive(train_df, state, horizon=n_val)
        except Exception as e:
            logger.error("[xgboost][%s] Recursive predict failed: %s", state, e)
            continue

        actual = series_val[TARGET_COL].values
        predicted = preds["forecast"].values
        n = min(len(actual), len(predicted))

        metrics = compute_metrics(
            actual[:n], predicted[:n],
            train_series=train_series[TARGET_COL].values,
        )
        metrics["state"] = state
        metrics["model"] = "xgboost"
        metrics["n_val"] = n
        all_metrics.append(metrics)

        safe_state = state.lower().replace(" ", "_")
        preds.to_csv(preds_dir / f"xgboost_{safe_state}_val.csv", index=False)

    return pd.DataFrame(all_metrics)


def main():
    splits = load_splits()
    train_df, val_df = splits["train"], splits["val"]

    (ARTIFACTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "models").mkdir(parents=True, exist_ok=True)

    # --- Train ---
    logger.info("=== Training XGBoost (global model) ===")
    t0 = time.time()
    model = XGBoostForecaster()
    model.fit(train_df, val_df)
    fit_time = time.time() - t0
    logger.info("Fit completed in %.1fs", fit_time)

    # --- Feature importance ---
    importance = model.get_feature_importance()
    logger.info("Top 10 features:\n%s", importance.head(10).to_string(index=False))
    importance.to_csv(ARTIFACTS_DIR / "metrics" / "xgboost_feature_importance.csv", index=False)

    # --- Save model ---
    model.save(ARTIFACTS_DIR / "models" / "xgboost_global.pkl")
    logger.info("Model saved")

    # --- One-step evaluation (sanity check) ---
    logger.info("=== One-step evaluation on val set ===")
    from src.models.xgboost_model import _encode_states
    val_enc, _ = _encode_states(val_df, model.state_mapping)
    val_for_eval = val_enc.dropna(subset=model.feature_cols)
    preds_1step = model.model.predict(val_for_eval[model.feature_cols].values)
    actual_1step = val_for_eval[TARGET_COL].values
    m = compute_metrics(actual_1step, preds_1step)
    logger.info("One-step (all states): %s", format_metrics(m))

    # --- Recursive multi-step evaluation per state ---
    logger.info("=== Recursive multi-step evaluation per state ===")
    t0 = time.time()
    metrics_df = evaluate_per_state(model, train_df, val_df)
    eval_time = time.time() - t0

    if len(metrics_df) > 0:
        logger.info(
            "Recursive eval: %d states in %.1fs | median sMAPE=%.2f%%",
            len(metrics_df), eval_time, metrics_df["smape"].median(),
        )
        logger.info("Per-state breakdown:\n%s",
            metrics_df[["state", "mae", "smape", "mase"]].to_string(index=False))
        metrics_df.to_csv(ARTIFACTS_DIR / "metrics" / "xgboost_metrics.csv", index=False)

    # --- Comparison with classical models ---
    try:
        sarima = pd.read_csv(ARTIFACTS_DIR / "metrics" / "sarima_metrics.csv")
        prophet = pd.read_csv(ARTIFACTS_DIR / "metrics" / "prophet_metrics.csv")
        combined = pd.concat([sarima, prophet, metrics_df], ignore_index=True)
        summary = combined.groupby("model")[["mae", "rmse", "smape", "mase"]].median()
        logger.info("\n=== Model Comparison (median across states) ===\n%s", summary.round(2).to_string())
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
