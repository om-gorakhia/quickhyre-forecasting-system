"""Walk-forward backtesting for XGBoost and LSTM.

Validates model robustness beyond a single train/val split by running
expanding-window evaluation:

  Fold 1: train on weeks 1..T1,     predict T1+1..T1+H
  Fold 2: train on weeks 1..T1+H,   predict T1+H+1..T1+2H
  ...

Only XGBoost and LSTM are backtested (they're the top-2 models and retraining
is fast). SARIMA/Prophet are too slow for repeated retraining and already lost
the single-split comparison.

The output shows whether a model's advantage is stable across time or if it
got lucky on one particular val window.
"""

import logging
import time

import numpy as np
import pandas as pd

from config.settings import (
    DATE_COL, STATE_COL, TARGET_COL,
    ARTIFACTS_DIR, FORECAST_WEEKS,
)
from src.dataset import (
    StateScaler, prepare_lstm_sequences, get_state_series, LSTM_FEATURES,
)
from src.evaluate import compute_metrics

logger = logging.getLogger(__name__)

# Backtest config
N_FOLDS = 3
BACKTEST_HORIZON = 13  # weeks per fold (~3 months)


def _backtest_xgboost(full_df: pd.DataFrame, fold_dates: list[tuple]) -> pd.DataFrame:
    """Run XGBoost backtest across folds."""
    from src.models.xgboost_model import XGBoostForecaster

    all_metrics = []
    for fold_i, (train_end, val_start, val_end) in enumerate(fold_dates, 1):
        train = full_df[full_df[DATE_COL] < train_end].copy()
        val = full_df[(full_df[DATE_COL] >= val_start) & (full_df[DATE_COL] < val_end)].copy()

        if len(val) == 0:
            continue

        model = XGBoostForecaster()
        model.fit(train, val)

        states = sorted(val[STATE_COL].unique())
        for state in states:
            series_val = get_state_series(val, state)
            train_series = get_state_series(train, state)
            try:
                preds = model.predict_recursive(train, state, horizon=len(series_val))
                actual = series_val[TARGET_COL].values
                predicted = preds["forecast"].values
                n = min(len(actual), len(predicted))
                m = compute_metrics(actual[:n], predicted[:n], train_series=train_series[TARGET_COL].values)
                m["state"] = state
                m["model"] = "xgboost"
                m["fold"] = fold_i
                all_metrics.append(m)
            except Exception:
                pass

    return pd.DataFrame(all_metrics)


def _backtest_lstm(full_df: pd.DataFrame, fold_dates: list[tuple]) -> pd.DataFrame:
    """Run LSTM backtest across folds."""
    from src.models.lstm_model import LSTMForecaster

    all_metrics = []
    for fold_i, (train_end, val_start, val_end) in enumerate(fold_dates, 1):
        train = full_df[full_df[DATE_COL] < train_end].copy()
        val = full_df[(full_df[DATE_COL] >= val_start) & (full_df[DATE_COL] < val_end)].copy()

        if len(val) == 0:
            continue

        scaler = StateScaler().fit(train, LSTM_FEATURES)
        X_train, y_train, _ = prepare_lstm_sequences(train, scaler)
        X_val, y_val, _ = prepare_lstm_sequences(val, scaler)

        if len(X_train) == 0 or len(X_val) == 0:
            continue

        model = LSTMForecaster()
        model.fit(X_train, y_train, X_val, y_val, scaler)

        states = sorted(val[STATE_COL].unique())
        for state in states:
            series_val = get_state_series(val, state)
            train_series = get_state_series(train, state)
            try:
                preds = model.predict_recursive(train, state, horizon=len(series_val))
                if len(preds) == 0:
                    continue
                actual = series_val[TARGET_COL].values
                predicted = preds["forecast"].values
                n = min(len(actual), len(predicted))
                m = compute_metrics(actual[:n], predicted[:n], train_series=train_series[TARGET_COL].values)
                m["state"] = state
                m["model"] = "lstm"
                m["fold"] = fold_i
                all_metrics.append(m)
            except Exception:
                pass

    return pd.DataFrame(all_metrics)


def run_backtest(full_df: pd.DataFrame, n_folds: int = N_FOLDS, horizon: int = BACKTEST_HORIZON) -> pd.DataFrame:
    """Run expanding-window backtest for XGBoost and LSTM.

    Builds fold boundaries by walking backward from the end of the dataset,
    so each fold tests a different time period.
    """
    dates = sorted(full_df[DATE_COL].unique())
    max_date = dates[-1]

    # Build fold boundaries (walk backward from end)
    fold_dates = []
    for i in range(n_folds):
        val_end = max_date - pd.Timedelta(weeks=i * horizon)
        val_start = val_end - pd.Timedelta(weeks=horizon)
        train_end = val_start
        if train_end <= dates[52]:  # need at least 52 weeks of training
            break
        fold_dates.append((train_end, val_start, val_end))

    fold_dates.reverse()  # chronological order
    logger.info("Backtest: %d folds, horizon=%d weeks", len(fold_dates), horizon)
    for i, (te, vs, ve) in enumerate(fold_dates, 1):
        logger.info("  Fold %d: train<=%s, val=%s..%s", i, te.date(), vs.date(), ve.date())

    t0 = time.time()
    xgb_results = _backtest_xgboost(full_df, fold_dates)
    logger.info("XGBoost backtest: %.1fs", time.time() - t0)

    t0 = time.time()
    lstm_results = _backtest_lstm(full_df, fold_dates)
    logger.info("LSTM backtest: %.1fs", time.time() - t0)

    combined = pd.concat([xgb_results, lstm_results], ignore_index=True)

    # Summary
    if len(combined) > 0:
        summary = combined.groupby(["model", "fold"])["smape"].median()
        logger.info("Backtest median sMAPE by model×fold:\n%s", summary.round(2).to_string())

        stability = combined.groupby("model")["smape"].agg(["median", "std", "mean"])
        logger.info("Backtest stability:\n%s", stability.round(2).to_string())

    # Save
    out = ARTIFACTS_DIR / "metrics" / "backtest_results.csv"
    combined.to_csv(out, index=False)
    logger.info("Backtest results saved → %s", out)

    return combined
