"""Model-specific dataset preparation.

Converts the feature-enriched dataframe into formats each model family needs:
- XGBoost:  (X, y) numpy arrays with NaN-lag rows dropped
- LSTM:     (X_seq, y_seq) 3D tensors with sliding window
- ARIMA/Prophet: raw (date, sales) series — no transformation needed

Scaler fitting is always done on training data only, then applied to val/test.
This module owns that contract.
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    DATE_COL, STATE_COL, TARGET_COL,
    ARTIFACTS_DIR, LSTM_SEQ_LEN,
)
from src.features import ALL_FEATURE_COLS, LAG_FEATURES, ROLLING_FEATURES

logger = logging.getLogger(__name__)

# Features that XGBoost uses (all of them, NaN rows dropped)
XGBOOST_FEATURES = ALL_FEATURE_COLS

# Features fed into the LSTM (numeric only, no categoricals)
# We keep lags + rolling + cyclical + trend — skip raw month/week_of_year
# (the sin/cos versions carry the same info without ordinality issues)
LSTM_FEATURES = (
    [f for f in ALL_FEATURE_COLS if f not in ("month", "week_of_year", "day_of_week", "quarter", "year")]
)


# ---------------------------------------------------------------------------
# Scalers
# ---------------------------------------------------------------------------

class StateScaler:
    """Per-state MinMax scaling. Fit on train, transform any split.

    Stores per-state (min, max) for the target and per-column global
    (min, max) for features. This keeps state-specific magnitude info
    while normalizing the feature space.
    """

    def __init__(self):
        self.target_stats: dict[str, tuple[float, float]] = {}
        self.feature_min: pd.Series | None = None
        self.feature_max: pd.Series | None = None

    def fit(self, df: pd.DataFrame, feature_cols: list[str]) -> "StateScaler":
        for state, grp in df.groupby(STATE_COL):
            vals = grp[TARGET_COL].dropna()
            self.target_stats[state] = (vals.min(), vals.max())

        subset = df[feature_cols].dropna()
        self.feature_min = subset.min()
        self.feature_max = subset.max()
        return self

    def transform_target(self, series: pd.Series, state: str) -> pd.Series:
        lo, hi = self.target_stats[state]
        if hi == lo:
            return series * 0.0
        return (series - lo) / (hi - lo)

    def inverse_target(self, series: pd.Series, state: str) -> pd.Series:
        lo, hi = self.target_stats[state]
        return series * (hi - lo) + lo

    def transform_features(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        df = df.copy()
        denom = (self.feature_max - self.feature_min).replace(0, 1)
        df[feature_cols] = (df[feature_cols] - self.feature_min) / denom
        return df

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Scaler saved → %s", path)

    @classmethod
    def load(cls, path: Path) -> "StateScaler":
        with open(path, "rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# XGBoost dataset
# ---------------------------------------------------------------------------

def prepare_xgboost(
    df: pd.DataFrame,
    feature_cols: list[str] = XGBOOST_FEATURES,
    drop_na: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Prepare (X, y) arrays for XGBoost.

    Returns (X, y, metadata_df) where metadata_df keeps state/date for
    tracing predictions back. Rows with NaN features are dropped — these
    are the early rows where lags don't exist yet.
    """
    subset = df[[STATE_COL, DATE_COL, TARGET_COL] + feature_cols].copy()
    if drop_na:
        before = len(subset)
        subset = subset.dropna(subset=feature_cols)
        dropped = before - len(subset)
        if dropped:
            logger.info("XGBoost: dropped %d rows with NaN features (lag warm-up)", dropped)

    X = subset[feature_cols].values.astype(np.float32)
    y = subset[TARGET_COL].values.astype(np.float32)
    meta = subset[[STATE_COL, DATE_COL]].reset_index(drop=True)
    return X, y, meta


# ---------------------------------------------------------------------------
# LSTM dataset
# ---------------------------------------------------------------------------

def prepare_lstm_sequences(
    df: pd.DataFrame,
    scaler: StateScaler,
    feature_cols: list[str] = LSTM_FEATURES,
    seq_len: int = LSTM_SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Build sliding-window sequences for LSTM, per state.

    For each state:
      1. Drop rows with NaN features (early lag warm-up)
      2. Scale features and target using the pre-fit scaler
      3. Slide a window of `seq_len` rows → predict the next row's target

    Returns:
        X_seq: (N, seq_len, n_features)  float32
        y_seq: (N,)                       float32
        meta:  list of dicts with state, date (of the predicted row)
    """
    all_X, all_y, all_meta = [], [], []

    for state, grp in df.groupby(STATE_COL):
        grp = grp.sort_values(DATE_COL).reset_index(drop=True)

        # Drop NaN feature rows
        valid_mask = grp[feature_cols].notna().all(axis=1)
        grp = grp[valid_mask].reset_index(drop=True)
        if len(grp) < seq_len + 1:
            logger.warning("State %s: only %d valid rows, need %d — skipping", state, len(grp), seq_len + 1)
            continue

        # Scale
        feat_scaled = scaler.transform_features(grp, feature_cols)[feature_cols].values.astype(np.float32)
        target_scaled = scaler.transform_target(grp[TARGET_COL], state).values.astype(np.float32)

        # Sliding window
        for i in range(len(grp) - seq_len):
            all_X.append(feat_scaled[i : i + seq_len])
            all_y.append(target_scaled[i + seq_len])
            all_meta.append({"state": state, "date": grp.iloc[i + seq_len][DATE_COL]})

    X_seq = np.stack(all_X)
    y_seq = np.array(all_y, dtype=np.float32)
    logger.info("LSTM sequences: %s (X), %s (y), from %d states", X_seq.shape, y_seq.shape, df[STATE_COL].nunique())
    return X_seq, y_seq, all_meta


# ---------------------------------------------------------------------------
# ARIMA / Prophet helpers
# ---------------------------------------------------------------------------

def get_state_series(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """Extract a clean (date, sales) series for a single state.

    This is the input format for ARIMA and Prophet — no feature columns needed.
    """
    out = (
        df[df[STATE_COL] == state][[DATE_COL, TARGET_COL]]
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )
    return out


def get_prophet_df(df: pd.DataFrame, state: str) -> pd.DataFrame:
    """Prophet expects columns named 'ds' and 'y'."""
    s = get_state_series(df, state)
    return s.rename(columns={DATE_COL: "ds", TARGET_COL: "y"})


# ---------------------------------------------------------------------------
# Convenience: build everything from splits
# ---------------------------------------------------------------------------

def build_datasets(splits: dict[str, pd.DataFrame]) -> dict:
    """One-call dataset builder. Fits scaler on train, transforms all splits.

    Returns dict with:
        scaler: fitted StateScaler
        xgboost: {train: (X,y,meta), val: ..., test: ...}
        lstm:    {train: (X,y,meta), val: ..., test: ...}
    """
    scaler = StateScaler().fit(splits["train"], LSTM_FEATURES)
    scaler.save(ARTIFACTS_DIR / "scaler.pkl")

    result = {"scaler": scaler, "xgboost": {}, "lstm": {}}
    for split_name, df in splits.items():
        result["xgboost"][split_name] = prepare_xgboost(df)
        result["lstm"][split_name] = prepare_lstm_sequences(df, scaler)

    return result
