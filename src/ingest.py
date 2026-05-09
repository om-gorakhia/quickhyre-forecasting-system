"""Data ingestion: load raw Excel, parse dates, validate, and produce a clean weekly panel."""

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from config.settings import (
    RAW_EXCEL, CLEANED_PARQUET, PROCESSED_DATA_DIR,
    DATE_COL, STATE_COL, TARGET_COL, FREQ,
)

logger = logging.getLogger(__name__)


def load_raw(path: Path = RAW_EXCEL) -> pd.DataFrame:
    """Read Excel and standardize column names."""
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    rename = {"total": TARGET_COL, "date": DATE_COL, "state": STATE_COL}
    df = df.rename(columns=rename)
    logger.info("Loaded %d rows, columns: %s", len(df), list(df.columns))
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Handle the mixed date formats (YYYY-MM-DD and DD-MM-YYYY) in the source."""
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="mixed", dayfirst=True)
    n_nat = df[DATE_COL].isna().sum()
    if n_nat > 0:
        logger.warning("%d dates could not be parsed — dropping those rows", n_nat)
        df = df.dropna(subset=[DATE_COL])
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Run sanity checks and drop the useless category column."""
    # Category is always 'Beverages'
    if "category" in df.columns:
        unique_cats = df["category"].unique()
        if len(unique_cats) == 1:
            logger.info("Dropping constant column 'category' (value: %s)", unique_cats[0])
            df = df.drop(columns=["category"])
        else:
            logger.warning("Unexpected categories found: %s — keeping column", unique_cats)

    # Duplicates
    n_dup = df.duplicated(subset=[STATE_COL, DATE_COL]).sum()
    if n_dup > 0:
        logger.warning("Dropping %d duplicate (state, date) rows", n_dup)
        df = df.drop_duplicates(subset=[STATE_COL, DATE_COL], keep="last")

    # Target validation
    n_neg = (df[TARGET_COL] < 0).sum()
    n_null = df[TARGET_COL].isna().sum()
    if n_neg > 0:
        logger.warning("%d negative sales values found", n_neg)
    if n_null > 0:
        logger.warning("%d null sales values found", n_null)

    logger.info(
        "Validated: %d rows, %d states, date range %s to %s",
        len(df), df[STATE_COL].nunique(),
        df[DATE_COL].min().date(), df[DATE_COL].max().date(),
    )
    return df


def resample_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample each state to a regular weekly grid.

    Strategy: snap each observation to the nearest weekly period end (W-SAT),
    then reindex to fill gaps. For duplicate weeks after snapping, keep the mean.
    Impute missing weeks via linear interpolation (bounded by forward/back fill
    for edges).
    """
    df = df.copy()
    # Snap to weekly period
    df[DATE_COL] = df[DATE_COL].dt.to_period(FREQ).dt.to_timestamp(FREQ)

    # If snapping created duplicates within a state-week, average them
    df = df.groupby([STATE_COL, DATE_COL], as_index=False)[TARGET_COL].mean()

    # Build complete weekly index across the observed range
    full_dates = pd.date_range(
        df[DATE_COL].min(), df[DATE_COL].max(), freq=FREQ
    )
    states = sorted(df[STATE_COL].unique())
    full_index = pd.MultiIndex.from_product(
        [states, full_dates], names=[STATE_COL, DATE_COL]
    )

    df = df.set_index([STATE_COL, DATE_COL]).reindex(full_index)

    # Track missing before imputation
    missing_per_state = df[TARGET_COL].isna().groupby(level=STATE_COL).sum()
    total_missing = int(missing_per_state.sum())
    if total_missing > 0:
        logger.info(
            "Missing weeks to impute: %d total (%.1f%% of grid)",
            total_missing, 100 * total_missing / len(df),
        )

    # Interpolate per state: linear for interior, ffill/bfill for edges
    df[TARGET_COL] = df.groupby(level=STATE_COL)[TARGET_COL].transform(
        lambda s: s.interpolate(method="linear").ffill().bfill()
    )

    remaining_nulls = df[TARGET_COL].isna().sum()
    if remaining_nulls > 0:
        logger.error("%d nulls remain after imputation — investigate", remaining_nulls)

    df = df.reset_index()
    return df, missing_per_state


def run_ingestion() -> pd.DataFrame:
    """Full pipeline: load → parse → validate → resample → save."""
    from src.validate import validate_raw_input, validate_weekly_panel

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    df = parse_dates(df)
    df = validate(df)
    validate_raw_input(df)
    df, missing_report = resample_to_weekly(df)
    validate_weekly_panel(df)

    # Final sanity report
    states = sorted(df[STATE_COL].unique())
    weeks_per_state = df.groupby(STATE_COL)[DATE_COL].nunique()
    logger.info("--- Ingestion Summary ---")
    logger.info("States: %d", len(states))
    logger.info("Date range: %s to %s", df[DATE_COL].min().date(), df[DATE_COL].max().date())
    logger.info("Weeks per state: min=%d, max=%d", weeks_per_state.min(), weeks_per_state.max())
    logger.info("Total rows: %d", len(df))
    logger.info("Nulls remaining: %d", df[TARGET_COL].isna().sum())

    # Save
    df.to_parquet(CLEANED_PARQUET, index=False)
    logger.info("Saved cleaned data to %s", CLEANED_PARQUET)

    # Save missing-week report
    missing_report.to_csv(PROCESSED_DATA_DIR / "missing_weeks_report.csv")

    return df
