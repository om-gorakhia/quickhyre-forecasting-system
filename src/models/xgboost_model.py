"""XGBoost forecaster with recursive multi-step prediction.

Architecture decision: GLOBAL model, not per-state.
- 142 rows per state × 23 features = underpowered per-state models
- 6106 pooled rows lets XGBoost learn shared seasonal/trend patterns
- State identity is encoded as an integer categorical feature so the model
  can still learn state-specific offsets
- This is the standard approach in production tabular forecasting (M5, Kaggle)

Recursive multi-step forecasting:
  XGBoost is trained on one-step-ahead (predict sales[t] from features[t]).
  For 8-week-ahead forecasting we roll forward one step at a time:
    1. Predict sales[t+1] using known features
    2. Append predicted value to history
    3. Recompute lag/rolling features for t+2 using the updated history
    4. Repeat 8 times
  This means errors compound — step 8 is noisier than step 1. That's inherent
  to recursive forecasting with lag-based models. The alternative (direct
  multi-output) would need 8 separate models and can't use lag_1 at all.
"""

import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from config.settings import (
    DATE_COL, STATE_COL, TARGET_COL, FREQ,
    LAG_PERIODS, SEASONAL_LAG, ROLLING_WINDOWS,
    FORECAST_WEEKS, ARTIFACTS_DIR,
)
from src.features import ALL_FEATURE_COLS
from src.evaluate import compute_metrics, format_metrics

logger = logging.getLogger(__name__)

# The features XGBoost trains on: all engineered features + state_encoded
XGBOOST_FEATURE_COLS = ALL_FEATURE_COLS + ["state_encoded"]


def _encode_states(df: pd.DataFrame, mapping: dict[str, int] | None = None):
    """Label-encode state column. Returns (df, mapping)."""
    if mapping is None:
        states = sorted(df[STATE_COL].unique())
        mapping = {s: i for i, s in enumerate(states)}
    df = df.copy()
    df["state_encoded"] = df[STATE_COL].map(mapping).astype(int)
    return df, mapping


class XGBoostForecaster:
    """Global XGBoost model across all states with recursive multi-step prediction."""

    def __init__(self, params: dict | None = None):
        self._default_params = {
            "objective": "reg:squarederror",
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 10,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 30,
        }
        if params:
            self._default_params.update(params)
        self.model: xgb.XGBRegressor | None = None
        self.state_mapping: dict[str, int] = {}
        self.feature_cols = XGBOOST_FEATURE_COLS

    @property
    def name(self) -> str:
        return "xgboost"

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame | None = None,
    ) -> "XGBoostForecaster":
        """Fit on the full training dataframe (all states, feature-enriched)."""
        train_enc, self.state_mapping = _encode_states(train_df)
        train_clean = train_enc.dropna(subset=self.feature_cols)

        X_train = train_clean[self.feature_cols].values
        y_train = train_clean[TARGET_COL].values

        es_rounds = self._default_params.pop("early_stopping_rounds", 30)
        params = dict(self._default_params)
        if val_df is not None and es_rounds:
            params["early_stopping_rounds"] = es_rounds
        self._default_params["early_stopping_rounds"] = es_rounds  # restore for save

        self.model = xgb.XGBRegressor(**params)

        fit_kwargs = {}
        if val_df is not None:
            val_enc, _ = _encode_states(val_df, self.state_mapping)
            val_clean = val_enc.dropna(subset=self.feature_cols)
            X_val = val_clean[self.feature_cols].values
            y_val = val_clean[TARGET_COL].values
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self.model.fit(X_train, y_train, verbose=False, **fit_kwargs)

        logger.info(
            "XGBoost trained: %d rows, best_iteration=%s",
            len(X_train),
            getattr(self.model, "best_iteration", "N/A"),
        )
        return self

    # ------------------------------------------------------------------
    # One-step prediction (for evaluation on existing splits)
    # ------------------------------------------------------------------

    def predict_onestep(self, df: pd.DataFrame) -> np.ndarray:
        """Predict sales for rows that already have features computed."""
        df_enc, _ = _encode_states(df, self.state_mapping)
        df_clean = df_enc.dropna(subset=self.feature_cols)
        X = df_clean[self.feature_cols].values
        return self.model.predict(X)

    # ------------------------------------------------------------------
    # Recursive multi-step prediction
    # ------------------------------------------------------------------

    def predict_recursive(
        self,
        history_df: pd.DataFrame,
        state: str,
        horizon: int = FORECAST_WEEKS,
    ) -> pd.DataFrame:
        """Forecast `horizon` weeks ahead for a single state by rolling forward.

        `history_df` must be the feature-enriched dataframe for this state,
        sorted by date. We need enough trailing history to recompute lags
        (at least max(LAG_PERIODS + [SEASONAL_LAG]) = 52 rows).
        """
        all_lags = sorted(set(LAG_PERIODS + [SEASONAL_LAG]))
        max_lag = max(all_lags)

        state_df = (
            history_df[history_df[STATE_COL] == state]
            .sort_values(DATE_COL)
            .copy()
        )

        # We maintain a rolling buffer of recent sales for lag/rolling computation
        sales_history = state_df[TARGET_COL].values.tolist()
        last_date = state_df[DATE_COL].max()
        state_enc = self.state_mapping[state]

        forecasts = []
        for step in range(horizon):
            next_date = last_date + pd.Timedelta(weeks=step + 1)
            row = self._build_forecast_row(
                sales_history, next_date, state_enc, all_lags,
            )
            pred = float(self.model.predict(row.reshape(1, -1))[0])
            forecasts.append({"date": next_date, "forecast": pred})
            sales_history.append(pred)

        return pd.DataFrame(forecasts)

    def _build_forecast_row(
        self,
        sales_history: list[float],
        date: pd.Timestamp,
        state_enc: int,
        all_lags: list[int],
    ) -> np.ndarray:
        """Construct a single feature row for a future date.

        Rebuilds each feature from scratch using the sales_history buffer.
        This is the core of recursive prediction — each feature must be
        computable from past data only.
        """
        row = {}

        # Calendar features
        row["month"] = date.month
        row["week_of_year"] = date.isocalendar().week
        row["year"] = date.year
        row["quarter"] = (date.month - 1) // 3 + 1
        row["day_of_week"] = date.dayofweek

        # Cyclical encodings
        row["month_sin"] = np.sin(2 * np.pi * date.month / 12)
        row["month_cos"] = np.cos(2 * np.pi * date.month / 12)
        woy = date.isocalendar().week
        row["week_of_year_sin"] = np.sin(2 * np.pi * woy / 52)
        row["week_of_year_cos"] = np.cos(2 * np.pi * woy / 52)

        # Holiday flag
        try:
            import holidays as _hol
            hset = set(_hol.US(years=date.year).keys())
            row["is_holiday_week"] = int(any(
                (date - pd.Timedelta(days=i)).date() in hset for i in range(7)
            ))
        except ImportError:
            row["is_holiday_week"] = 0

        # Trend
        # week_index is relative to training start — approximate
        row["week_index"] = len(sales_history)

        # Lag features
        for lag in all_lags:
            if lag <= len(sales_history):
                row[f"lag_{lag}"] = sales_history[-lag]
            else:
                row[f"lag_{lag}"] = sales_history[-1]  # fallback to most recent available

        # Rolling features (over the shifted series, i.e., not including current)
        for w in ROLLING_WINDOWS:
            recent = sales_history[-w:] if len(sales_history) >= w else sales_history[:]
            row[f"roll_mean_{w}"] = np.mean(recent)
            row[f"roll_std_{w}"] = np.std(recent, ddof=1) if len(recent) > 1 else 0.0

        # Derived
        if len(sales_history) >= 2:
            row["pct_change_1"] = (sales_history[-1] - sales_history[-2]) / sales_history[-2] if sales_history[-2] != 0 else 0.0
        else:
            row["pct_change_1"] = 0.0

        row["state_expanding_mean"] = np.mean(sales_history)

        # State encoding
        row["state_encoded"] = state_enc

        # Build array in correct feature order
        return np.array([row[f] for f in self.feature_cols], dtype=np.float32)

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> pd.DataFrame:
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_cols, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "state_mapping": self.state_mapping,
                "feature_cols": self.feature_cols,
                "params": self._default_params,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "XGBoostForecaster":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(params=data["params"])
        obj.model = data["model"]
        obj.state_mapping = data["state_mapping"]
        obj.feature_cols = data["feature_cols"]
        return obj
