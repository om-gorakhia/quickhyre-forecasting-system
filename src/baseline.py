"""Naive baseline forecasts for benchmarking.

Two baselines that any useful model must beat:
  - Naive (lag-1): repeat last observed value
  - Seasonal naive (lag-52): repeat value from same week last year

These are computed on the validation set and reported alongside model metrics
so evaluators can immediately see the improvement margin.
"""

import logging

import numpy as np
import pandas as pd

from config.settings import DATE_COL, STATE_COL, TARGET_COL, ARTIFACTS_DIR
from src.evaluate import compute_metrics

logger = logging.getLogger(__name__)


def compute_naive_baselines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute naive and seasonal-naive forecasts on the validation set.

    For each state:
      - naive: predict val[t] = last training value (repeat forward)
      - seasonal_naive: predict val[t] = train value from 52 weeks earlier

    Returns a metrics DataFrame comparable to model metrics CSVs.
    """
    all_metrics = []

    for state in sorted(val_df[STATE_COL].unique()):
        train_state = train_df[train_df[STATE_COL] == state].sort_values(DATE_COL)
        val_state = val_df[val_df[STATE_COL] == state].sort_values(DATE_COL)

        if len(train_state) == 0 or len(val_state) == 0:
            continue

        actual = val_state[TARGET_COL].values
        train_vals = train_state[TARGET_COL].values

        # --- Naive (lag-1): repeat last training value ---
        naive_pred = np.full_like(actual, train_vals[-1])
        m = compute_metrics(actual, naive_pred, train_series=train_vals)
        m["state"] = state
        m["model"] = "naive"
        m["n_val"] = len(actual)
        all_metrics.append(m)

        # --- Seasonal naive (lag-52): same week last year ---
        if len(train_vals) >= 52:
            # For each val week, use the value from 52 weeks back in the
            # combined train+val history
            full_series = np.concatenate([train_vals, actual])
            train_len = len(train_vals)
            seasonal_pred = np.array([
                full_series[train_len + i - 52] if (train_len + i - 52) >= 0
                else train_vals[-1]
                for i in range(len(actual))
            ])
            m = compute_metrics(actual, seasonal_pred, train_series=train_vals)
            m["state"] = state
            m["model"] = "seasonal_naive"
            m["n_val"] = len(actual)
            all_metrics.append(m)

    result = pd.DataFrame(all_metrics)

    if len(result) > 0:
        for model_name in ["naive", "seasonal_naive"]:
            subset = result[result["model"] == model_name]
            if len(subset) > 0:
                med_smape = subset["smape"].median()
                med_mae = subset["mae"].median()
                logger.info("[%s] Median sMAPE: %.2f%% | Median MAE: %.0f", model_name, med_smape, med_mae)

    # Save
    out = ARTIFACTS_DIR / "metrics" / "baseline_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    logger.info("Baseline metrics saved -> %s", out)

    return result
