"""Unified training pipeline: train all models, select winners, generate forecasts.

Usage:
    python scripts/train_all.py                    # full pipeline
    python scripts/train_all.py --skip-classical   # skip SARIMA/Prophet (slow)
    python scripts/train_all.py --no-ensemble      # disable ensemble blending
    python scripts/train_all.py --select-only      # skip training, just select + forecast
    python scripts/train_all.py --force             # re-run everything, ignore existing artifacts

This script orchestrates the entire pipeline:
  1. Preprocessing (if needed)
  2. Train SARIMA + Prophet (per-state, parallel-safe)
  3. Train XGBoost (global)
  4. Train LSTM (global)
  5. Compute naive baselines for benchmarking
  6. Collect metrics, select best model per state
  7. Validate ensemble vs single-best
  8. Generate 8-week forecasts
  9. Generate summary plots, tables, HTML report
 10. Walk-forward backtest (XGBoost + LSTM)
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import (
    ARTIFACTS_DIR, FORECAST_WEEKS,
    STATE_COL, DATE_COL, TARGET_COL,
)

# Reproducibility: set hash seed before any hashing occurs
os.environ.setdefault("PYTHONHASHSEED", "42")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_all")

# Global flag — set by --force to skip artifact caching
FORCE_RERUN = False


def _should_skip(path: Path, label: str) -> bool:
    """Return True if artifact exists and we're not forcing a rerun."""
    if FORCE_RERUN:
        return False
    if path.exists():
        logger.info("[%s] Artifacts exist — skipping (use --force to re-run)", label)
        return True
    return False


def step_preprocess():
    """Run preprocessing if splits don't exist yet."""
    from config.settings import SPLITS_DIR
    if _should_skip(SPLITS_DIR / "train.parquet", "preprocess"):
        return
    logger.info("=== Preprocessing ===")
    from src.ingest import run_ingestion
    from src.features import build_features
    from src.split import temporal_split, save_splits
    df = run_ingestion()
    df = build_features(df)
    splits = temporal_split(df)
    from src.validate import validate_splits
    validate_splits(splits)
    save_splits(splits)


def step_train_classical():
    """Train SARIMA and Prophet."""
    from src.split import load_splits
    from src.models.sarima import SARIMAModel
    from src.models.prophet_model import ProphetModel
    from src.train_classical import train_model_all_states

    splits = load_splits()
    train_df, val_df = splits["train"], splits["val"]
    (ARTIFACTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)

    for cls, name in [(SARIMAModel, "sarima"), (ProphetModel, "prophet")]:
        csv = ARTIFACTS_DIR / "metrics" / f"{name}_metrics.csv"
        if _should_skip(csv, name):
            continue
        train_model_all_states(cls, train_df, val_df)


def step_train_xgboost():
    """Train XGBoost global model."""
    csv = ARTIFACTS_DIR / "metrics" / "xgboost_metrics.csv"
    if _should_skip(csv, "xgboost"):
        return

    from src.split import load_splits
    from src.models.xgboost_model import XGBoostForecaster, _encode_states
    from src.dataset import get_state_series
    from src.evaluate import compute_metrics

    splits = load_splits()
    train_df, val_df = splits["train"], splits["val"]
    (ARTIFACTS_DIR / "models").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    model = XGBoostForecaster()
    model.fit(train_df, val_df)
    model.save(ARTIFACTS_DIR / "models" / "xgboost_global.pkl")

    # Feature importance
    importance = model.get_feature_importance()
    importance.to_csv(ARTIFACTS_DIR / "metrics" / "xgboost_feature_importance.csv", index=False)
    logger.info("Top 5 features: %s", list(importance.head(5)["feature"]))

    # Evaluate per state
    states = sorted(val_df[STATE_COL].unique())
    all_metrics = []
    for state in states:
        series_val = get_state_series(val_df, state)
        train_series = get_state_series(train_df, state)
        try:
            preds = model.predict_recursive(train_df, state, horizon=len(series_val))
            actual = series_val[TARGET_COL].values
            predicted = preds["forecast"].values
            n = min(len(actual), len(predicted))
            metrics = compute_metrics(actual[:n], predicted[:n], train_series=train_series[TARGET_COL].values)
            metrics["state"] = state
            metrics["model"] = "xgboost"
            metrics["n_val"] = n
            all_metrics.append(metrics)
            safe = state.lower().replace(" ", "_")
            preds.to_csv(ARTIFACTS_DIR / "predictions" / f"xgboost_{safe}_val.csv", index=False)
        except Exception as e:
            logger.error("[xgboost][%s] Failed: %s", state, e)

    pd.DataFrame(all_metrics).to_csv(ARTIFACTS_DIR / "metrics" / "xgboost_metrics.csv", index=False)
    logger.info("[xgboost] Complete: %d states", len(all_metrics))


def step_train_lstm():
    """Train LSTM global model."""
    csv = ARTIFACTS_DIR / "metrics" / "lstm_metrics.csv"
    if _should_skip(csv, "lstm"):
        return

    from src.split import load_splits
    from src.dataset import StateScaler, prepare_lstm_sequences, get_state_series
    from src.models.lstm_model import LSTMForecaster
    from src.evaluate import compute_metrics

    splits = load_splits()
    train_df, val_df = splits["train"], splits["val"]

    scaler = StateScaler.load(ARTIFACTS_DIR / "scaler.pkl")
    X_train, y_train, _ = prepare_lstm_sequences(train_df, scaler)
    X_val, y_val, _ = prepare_lstm_sequences(val_df, scaler)

    model = LSTMForecaster()
    model.fit(X_train, y_train, X_val, y_val, scaler)
    model.save(ARTIFACTS_DIR / "models" / "lstm_global.pt")

    # Training curves
    pd.DataFrame({
        "epoch": range(1, len(model.train_losses) + 1),
        "train_loss": model.train_losses,
        "val_loss": model.val_losses,
    }).to_csv(ARTIFACTS_DIR / "metrics" / "lstm_training_curves.csv", index=False)

    # Evaluate per state
    states = sorted(val_df[STATE_COL].unique())
    all_metrics = []
    for state in states:
        series_val = get_state_series(val_df, state)
        train_series = get_state_series(train_df, state)
        try:
            preds = model.predict_recursive(train_df, state, horizon=len(series_val))
            if len(preds) == 0:
                continue
            actual = series_val[TARGET_COL].values
            predicted = preds["forecast"].values
            n = min(len(actual), len(predicted))
            metrics = compute_metrics(actual[:n], predicted[:n], train_series=train_series[TARGET_COL].values)
            metrics["state"] = state
            metrics["model"] = "lstm"
            metrics["n_val"] = n
            all_metrics.append(metrics)
            safe = state.lower().replace(" ", "_")
            preds.to_csv(ARTIFACTS_DIR / "predictions" / f"lstm_{safe}_val.csv", index=False)
        except Exception as e:
            logger.error("[lstm][%s] Failed: %s", state, e)

    pd.DataFrame(all_metrics).to_csv(ARTIFACTS_DIR / "metrics" / "lstm_metrics.csv", index=False)
    logger.info("[lstm] Complete: %d states", len(all_metrics))


def step_baselines():
    """Compute naive and seasonal-naive baselines for benchmarking."""
    out = ARTIFACTS_DIR / "metrics" / "baseline_metrics.csv"
    if _should_skip(out, "baselines"):
        return

    from src.split import load_splits
    from src.baseline import compute_naive_baselines

    splits = load_splits()
    compute_naive_baselines(splits["train"], splits["val"])


def step_backtest():
    """Run walk-forward backtest for XGBoost and LSTM."""
    out = ARTIFACTS_DIR / "metrics" / "backtest_results.csv"
    if _should_skip(out, "backtest"):
        return

    from src.split import load_splits
    from src.backtest import run_backtest

    splits = load_splits()
    full_df = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    run_backtest(full_df)


def step_validate_ensemble(selection: pd.DataFrame, metrics: pd.DataFrame, train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Measure whether ensemble actually improves over single-best on validation data."""
    from src.forecast import _load_model, _predict_state
    from src.selector import compute_ensemble_weights
    from src.evaluate import compute_metrics
    from src.dataset import get_state_series

    eligible = selection[selection["ensemble_eligible"] == True]
    if len(eligible) == 0:
        logger.info("No ensemble-eligible states — skipping validation")
        return

    results = []
    for _, row in eligible.iterrows():
        state = row["state"]
        best_name = row["best_model"]
        second_name = row["second_model"]

        try:
            # Get validation actuals
            val_state = get_state_series(val_df, state)
            actual = val_state[TARGET_COL].values
            train_state = get_state_series(train_df, state)
            horizon = len(actual)

            # Single best prediction
            best_model = _load_model(best_name, state)
            best_preds = _predict_state(best_model, best_name, state, train_df, horizon)
            best_forecast = best_preds["forecast"].values
            n = min(len(actual), len(best_forecast))
            best_metrics = compute_metrics(actual[:n], best_forecast[:n], train_series=train_state[TARGET_COL].values)

            # Second model prediction
            second_model = _load_model(second_name, state)
            second_preds = _predict_state(second_model, second_name, state, train_df, horizon)
            second_forecast = second_preds["forecast"].values
            n2 = min(n, len(second_forecast))

            # Ensemble
            w1, w2 = compute_ensemble_weights(row["best_smape"], row["second_smape"])
            ensemble_forecast = best_forecast[:n2] * w1 + second_forecast[:n2] * w2
            ensemble_metrics = compute_metrics(actual[:n2], ensemble_forecast, train_series=train_state[TARGET_COL].values)

            results.append({
                "state": state,
                "best_model": best_name,
                "second_model": second_name,
                "w_best": round(w1, 3),
                "w_second": round(w2, 3),
                "best_single_smape": round(best_metrics["smape"], 2),
                "ensemble_smape": round(ensemble_metrics["smape"], 2),
                "improved": ensemble_metrics["smape"] < best_metrics["smape"],
            })
        except Exception as e:
            logger.warning("[ensemble_val][%s] Skipped: %s", state, e)

    if results:
        df = pd.DataFrame(results)
        improved = df["improved"].sum()
        total = len(df)
        avg_delta = (df["best_single_smape"] - df["ensemble_smape"]).mean()
        logger.info(
            "Ensemble validation: improved %d/%d states (avg sMAPE delta: %.2f pp)",
            improved, total, avg_delta,
        )
        df.to_csv(ARTIFACTS_DIR / "metrics" / "ensemble_validation.csv", index=False)


def step_select_and_forecast(use_ensemble: bool = True):
    """Select best models, generate 8-week forecasts, build summary artifacts."""
    from src.split import load_splits
    from src.selector import (
        load_all_metrics, select_best_per_state,
        save_selection_artifacts, build_leaderboard,
    )
    from src.forecast import generate_forecasts

    splits = load_splits()
    train_df = splits["train"]
    val_df = splits["val"]

    # --- Collect and compare ---
    metrics = load_all_metrics()
    leaderboard = build_leaderboard(metrics)
    logger.info("\n=== Leaderboard (median metrics) ===\n%s", leaderboard.to_string())

    # --- Select ---
    selection = select_best_per_state(metrics)
    save_selection_artifacts(metrics, selection)

    logger.info("\n=== Best Model Per State ===")
    logger.info("Winner counts: %s", selection["best_model"].value_counts().to_dict())
    ensemble_count = selection["ensemble_eligible"].sum()
    logger.info("Ensemble-eligible states: %d / %d", ensemble_count, len(selection))

    # --- Validate ensemble ---
    logger.info("=== Validating Ensemble ===")
    step_validate_ensemble(selection, metrics, train_df, val_df)

    # --- Generate 8-week forecasts ---
    logger.info("=== Generating 8-week forecasts ===")
    forecasts = generate_forecasts(
        selection, train_df, horizon=FORECAST_WEEKS, use_ensemble=use_ensemble,
    )
    out_path = ARTIFACTS_DIR / "forecasts"
    out_path.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(out_path / "final_8week_forecasts.csv", index=False)
    logger.info("Forecasts saved -> %s", out_path / "final_8week_forecasts.csv")

    # Per-state forecast files
    for state, grp in forecasts.groupby("state"):
        safe = state.lower().replace(" ", "_")
        grp.to_csv(out_path / f"forecast_{safe}.csv", index=False)

    # --- Summary stats ---
    model_usage = forecasts.groupby("model_used").size()
    logger.info("Models used in final forecasts:\n%s", model_usage.to_string())

    # --- Plots ---
    logger.info("=== Generating plots ===")
    try:
        from src.plots import generate_all_plots
        generate_all_plots(metrics, selection, forecasts, train_df, val_df)
    except Exception as e:
        logger.warning("Plot generation failed (non-fatal): %s", e)

    return forecasts


def step_report():
    """Generate HTML summary report."""
    try:
        from src.report import generate_report
        path = generate_report()
        logger.info("=== HTML report: %s ===", path)
    except Exception as e:
        logger.warning("Report generation failed (non-fatal): %s", e)


def main():
    global FORCE_RERUN

    parser = argparse.ArgumentParser(description="Train all models and generate forecasts")
    parser.add_argument("--skip-classical", action="store_true", help="Skip SARIMA/Prophet training")
    parser.add_argument("--no-ensemble", action="store_true", help="Disable ensemble blending")
    parser.add_argument("--select-only", action="store_true", help="Skip training, run selection + forecast only")
    parser.add_argument("--force", action="store_true", help="Re-run all steps, ignore existing artifacts")
    args = parser.parse_args()

    FORCE_RERUN = args.force
    if FORCE_RERUN:
        logger.info("=== FORCE MODE: all steps will re-run ===")

    t0 = time.time()

    if not args.select_only:
        step_preprocess()
        if not args.skip_classical:
            logger.info("=== Training Classical Models (SARIMA + Prophet) ===")
            step_train_classical()
        logger.info("=== Training XGBoost ===")
        step_train_xgboost()
        logger.info("=== Training LSTM ===")
        step_train_lstm()

    # Baselines (fast, always worth running)
    logger.info("=== Computing Naive Baselines ===")
    step_baselines()

    step_select_and_forecast(use_ensemble=not args.no_ensemble)

    # Backtest
    if not args.select_only:
        logger.info("=== Running Walk-Forward Backtest ===")
        step_backtest()

    # Report (always regenerate)
    step_report()

    elapsed = time.time() - t0
    logger.info("=== Full pipeline completed in %.1fs ===", elapsed)


if __name__ == "__main__":
    main()
