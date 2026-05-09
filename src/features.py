"""Feature engineering for the forecasting pipeline.

Design notes
------------
The data is weekly after resampling (W-SAT). This has implications:

- **day_of_week**: constant (Saturday=5) after resampling. Useless as a
  predictive signal but included because the assignment requires it. We
  document it as a no-op feature so reviewers know it was considered.

- **Lag naming**: the assignment says t-1, t-7, t-30. Since each row is one
  week, these are 1-week, 7-week, and 30-week lookbacks. We add lag_52
  (yearly seasonality) as a bonus because it genuinely helps.

- **Rolling stats**: computed on shifted series (shift(1) before rolling)
  to prevent target leakage. The current row's value never participates
  in its own rolling window.

- **Cyclical encodings**: month and week_of_year are encoded as sin/cos
  pairs so that Dec→Jan is smooth, not a cliff from 12→1.

All transforms are per-state via groupby — no cross-state leakage.
"""

import logging

import numpy as np
import pandas as pd

from config.settings import (
    DATE_COL, STATE_COL, TARGET_COL, PROCESSED_DATA_DIR,
    LAG_PERIODS, SEASONAL_LAG, ROLLING_WINDOWS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Holiday lookup (vectorized via set, not per-row apply)
# ---------------------------------------------------------------------------
try:
    import holidays as _hol
    _US_HOLIDAYS: set | None = None  # lazily built
except ImportError:
    _hol = None
    _US_HOLIDAYS = None
    logger.warning("'holidays' package not installed — holiday flag will be zero")


def _get_holiday_set(year_min: int, year_max: int) -> set:
    """Build a flat set of holiday dates for fast O(1) membership tests."""
    global _US_HOLIDAYS
    if _US_HOLIDAYS is not None:
        return _US_HOLIDAYS
    if _hol is None:
        return set()
    _US_HOLIDAYS = set()
    for yr in range(year_min, year_max + 1):
        _US_HOLIDAYS.update(_hol.US(years=yr).keys())
    return _US_HOLIDAYS


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Basic calendar columns extracted from the date index."""
    dt = df[DATE_COL]
    df["month"] = dt.dt.month
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["year"] = dt.dt.year
    df["quarter"] = dt.dt.quarter

    # Assignment requires day_of_week. On weekly-resampled data it's constant
    # (Saturday=5). We include it for compliance but it carries zero signal.
    df["day_of_week"] = dt.dt.dayofweek

    return df


def add_cyclical_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Sin/cos encoding for periodic features so Dec↔Jan is smooth."""
    for col, period in [("month", 12), ("week_of_year", 52)]:
        radians = 2 * np.pi * df[col] / period
        df[f"{col}_sin"] = np.sin(radians)
        df[f"{col}_cos"] = np.cos(radians)
    return df


# ---------------------------------------------------------------------------
# Holiday flag (vectorized)
# ---------------------------------------------------------------------------

def add_holiday_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Flag weeks containing at least one US federal holiday.

    Vectorized: expands each row's week-ending date into 7 daily dates,
    checks set membership in bulk.
    """
    if _hol is None:
        df["is_holiday_week"] = 0
        return df

    year_min, year_max = df[DATE_COL].dt.year.min(), df[DATE_COL].dt.year.max()
    hset = _get_holiday_set(year_min, year_max)

    # For each week-ending date, check if any of the 7 days is a holiday
    week_ends = df[DATE_COL].values.astype("datetime64[D]")
    flags = np.zeros(len(df), dtype=np.int8)
    for offset in range(7):
        day_dates = (week_ends - np.timedelta64(offset, "D")).astype("datetime64[D]")
        flags |= np.array([pd.Timestamp(d).date() in hset for d in day_dates], dtype=np.int8)

    df["is_holiday_week"] = flags
    n_flagged = int(flags.sum())
    logger.info("Holiday weeks flagged: %d / %d (%.1f%%)", n_flagged, len(df), 100 * n_flagged / len(df))
    return df


# ---------------------------------------------------------------------------
# Lag features
# ---------------------------------------------------------------------------

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-state lag features. Each lag is a shift in weekly periods.

    Assignment lags (t-1, t-7, t-30) + seasonal lag (t-52).
    Nulls are expected for early rows — handled downstream per model.
    """
    all_lags = sorted(set(LAG_PERIODS + [SEASONAL_LAG]))
    grouped = df.groupby(STATE_COL)[TARGET_COL]
    for lag in all_lags:
        df[f"lag_{lag}"] = grouped.shift(lag)
    return df


# ---------------------------------------------------------------------------
# Rolling features
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-state rolling mean and std.

    LEAKAGE PREVENTION: we shift(1) before rolling so the current row's
    target is never included in its own rolling window.
    """
    grouped = df.groupby(STATE_COL)[TARGET_COL]
    for w in ROLLING_WINDOWS:
        shifted = grouped.transform(lambda s: s.shift(1))
        df[f"roll_mean_{w}"] = shifted.groupby(df[STATE_COL]).transform(
            lambda s: s.rolling(w, min_periods=1).mean()
        )
        df[f"roll_std_{w}"] = shifted.groupby(df[STATE_COL]).transform(
            lambda s: s.rolling(w, min_periods=1).std()
        )
    return df


# ---------------------------------------------------------------------------
# Trend and derived
# ---------------------------------------------------------------------------

def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Simple trend proxies."""
    min_date = df[DATE_COL].min()
    df["week_index"] = ((df[DATE_COL] - min_date).dt.days / 7).astype(int)

    # Pct change between the two most recent *past* values (shift to avoid leakage)
    df["pct_change_1"] = df.groupby(STATE_COL)[TARGET_COL].transform(
        lambda s: s.shift(1).pct_change(periods=1)
    )
    return df


def add_state_normalization_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Per-state mean and std of the target (computed on the training portion only
    by the caller). Here we just add placeholders that dataset.py fills properly.

    These columns let models see where a state sits in the overall distribution
    without leaking future data.
    """
    # State-level statistics computed from the full series up to each point
    # (expanding mean) — safe because it only looks backward
    df["state_expanding_mean"] = df.groupby(STATE_COL)[TARGET_COL].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Columns that every model can use (excluding lags/rolling which have NaNs)
CALENDAR_FEATURES = [
    "month", "week_of_year", "year", "quarter", "day_of_week",
    "month_sin", "month_cos", "week_of_year_sin", "week_of_year_cos",
    "is_holiday_week", "week_index",
]

LAG_FEATURES = [f"lag_{p}" for p in sorted(set(LAG_PERIODS + [SEASONAL_LAG]))]

ROLLING_FEATURES = []
for _w in ROLLING_WINDOWS:
    ROLLING_FEATURES += [f"roll_mean_{_w}", f"roll_std_{_w}"]

DERIVED_FEATURES = ["pct_change_1", "state_expanding_mean"]

ALL_FEATURE_COLS = CALENDAR_FEATURES + LAG_FEATURES + ROLLING_FEATURES + DERIVED_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full feature pipeline. Input must be the cleaned weekly panel.

    Returns the dataframe with all feature columns appended.
    Saves features.parquet to disk.
    """
    df = df.sort_values([STATE_COL, DATE_COL]).reset_index(drop=True)

    df = add_calendar_features(df)
    df = add_cyclical_encodings(df)
    df = add_holiday_flag(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_trend_features(df)
    df = add_state_normalization_inputs(df)

    # Summary
    null_counts = df[ALL_FEATURE_COLS].isna().sum()
    nulls_present = null_counts[null_counts > 0]
    logger.info(
        "Feature pipeline complete: %d feature columns, %d rows",
        len(ALL_FEATURE_COLS), len(df),
    )
    if len(nulls_present):
        logger.info("Expected NaNs from lags/rolling (early rows per state):\n%s", nulls_present.to_string())

    out = PROCESSED_DATA_DIR / "features.parquet"
    df.to_parquet(out, index=False)
    logger.info("Saved → %s", out)
    return df
