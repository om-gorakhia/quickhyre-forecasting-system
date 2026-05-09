"""Train LSTM global model, evaluate per state, save artifacts.

Usage:
    python scripts/train_lstm.py
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
from src.dataset import StateScaler, prepare_lstm_sequences, get_state_series, LSTM_FEATURES
from src.models.lstm_model import LSTMForecaster
from src.evaluate import compute_metrics, format_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_lstm")


def evaluate_per_state(
    model: LSTMForecaster,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate via recursive multi-step prediction per state."""
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
            logger.error("[lstm][%s] Recursive predict failed: %s", state, e)
            continue

        if len(preds) == 0:
            continue

        actual = series_val[TARGET_COL].values
        predicted = preds["forecast"].values
        n = min(len(actual), len(predicted))

        metrics = compute_metrics(
            actual[:n], predicted[:n],
            train_series=train_series[TARGET_COL].values,
        )
        metrics["state"] = state
        metrics["model"] = "lstm"
        metrics["n_val"] = n
        all_metrics.append(metrics)

        safe_state = state.lower().replace(" ", "_")
        preds.to_csv(preds_dir / f"lstm_{safe_state}_val.csv", index=False)

    return pd.DataFrame(all_metrics)


def main():
    splits = load_splits()
    train_df, val_df = splits["train"], splits["val"]

    (ARTIFACTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "models").mkdir(parents=True, exist_ok=True)

    # --- Load scaler (fitted during preprocessing) ---
    scaler = StateScaler.load(ARTIFACTS_DIR / "scaler.pkl")

    # --- Prepare sequences ---
    logger.info("=== Preparing LSTM sequences ===")
    X_train, y_train, meta_train = prepare_lstm_sequences(train_df, scaler)
    X_val, y_val, meta_val = prepare_lstm_sequences(val_df, scaler)

    # --- Train ---
    logger.info("=== Training LSTM (global model) ===")
    logger.info("Train: %s, Val: %s, Features: %d", X_train.shape, X_val.shape, X_train.shape[2])
    t0 = time.time()
    model = LSTMForecaster()
    model.fit(X_train, y_train, X_val, y_val, scaler)
    fit_time = time.time() - t0
    logger.info("Training completed in %.1fs", fit_time)

    # --- Save model ---
    model.save(ARTIFACTS_DIR / "models" / "lstm_global.pt")
    logger.info("Model saved")

    # --- One-step evaluation on val sequences ---
    logger.info("=== One-step evaluation on val sequences ===")
    preds_df = model.predict_sequences(X_val, meta_val)
    # Merge with actuals
    val_actuals = val_df[[STATE_COL, DATE_COL, TARGET_COL]].copy()
    merged = preds_df.merge(val_actuals, on=[STATE_COL, DATE_COL], how="inner")
    if len(merged) > 0:
        m = compute_metrics(merged[TARGET_COL].values, merged["forecast"].values)
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
        metrics_df.to_csv(ARTIFACTS_DIR / "metrics" / "lstm_metrics.csv", index=False)

    # --- Save training curves ---
    curves = pd.DataFrame({
        "epoch": range(1, len(model.train_losses) + 1),
        "train_loss": model.train_losses,
        "val_loss": model.val_losses,
    })
    curves.to_csv(ARTIFACTS_DIR / "metrics" / "lstm_training_curves.csv", index=False)

    # --- Cross-model comparison ---
    try:
        all_dfs = []
        for name in ["sarima", "prophet", "xgboost", "lstm"]:
            p = ARTIFACTS_DIR / "metrics" / f"{name}_metrics.csv"
            if p.exists():
                all_dfs.append(pd.read_csv(p))
        if len(all_dfs) > 1:
            combined = pd.concat(all_dfs, ignore_index=True)
            summary = combined.groupby("model")[["mae", "rmse", "smape", "mase"]].median()
            logger.info("\n=== All Models Comparison (median) ===\n%s", summary.round(2).to_string())
    except Exception:
        pass


if __name__ == "__main__":
    main()
