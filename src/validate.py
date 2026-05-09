"""Lightweight data validation — fail fast with clear messages.

Called at pipeline entry points to catch broken inputs before
burning 30 minutes on SARIMA training.
"""

import logging

import pandas as pd
import numpy as np

from config.settings import DATE_COL, STATE_COL, TARGET_COL

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when data fails a critical assumption."""
    pass


def validate_raw_input(df: pd.DataFrame) -> None:
    """Check the raw dataframe right after loading from Excel."""
    required = {DATE_COL, STATE_COL, TARGET_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValidationError(
            f"Missing required columns: {missing}. "
            f"Found: {list(df.columns)}"
        )

    if len(df) == 0:
        raise ValidationError("Input dataframe is empty")

    if df[TARGET_COL].isna().all():
        raise ValidationError("Target column is entirely null")

    n_states = df[STATE_COL].nunique()
    if n_states < 2:
        raise ValidationError(f"Expected multiple states, found {n_states}")

    logger.info("Raw input validated: %d rows, %d states", len(df), n_states)


def validate_weekly_panel(df: pd.DataFrame) -> None:
    """Check the cleaned weekly panel before feature engineering."""
    if df[TARGET_COL].isna().any():
        n = df[TARGET_COL].isna().sum()
        raise ValidationError(f"{n} null values in target after imputation")

    if (df[TARGET_COL] < 0).any():
        n = (df[TARGET_COL] < 0).sum()
        logger.warning("%d negative sales values — may indicate data quality issues", n)

    # Check weekly regularity per state
    for state in df[STATE_COL].unique()[:3]:  # spot-check first 3
        dates = df[df[STATE_COL] == state][DATE_COL].sort_values()
        gaps = dates.diff().dropna()
        non_weekly = (gaps != pd.Timedelta(weeks=1)).sum()
        if non_weekly > 0:
            raise ValidationError(
                f"State '{state}' has {non_weekly} non-weekly gaps after resampling"
            )

    logger.info("Weekly panel validated: %d rows, no nulls", len(df))


def validate_splits(splits: dict[str, pd.DataFrame]) -> None:
    """Check that train/val/test splits are temporally ordered and non-empty."""
    for name in ["train", "val"]:
        if name not in splits or len(splits[name]) == 0:
            raise ValidationError(f"Split '{name}' is empty — check date boundaries")

    train_max = splits["train"][DATE_COL].max()
    val_min = splits["val"][DATE_COL].min()
    if train_max >= val_min:
        raise ValidationError(
            f"Train/val overlap: train_max={train_max}, val_min={val_min}"
        )

    train_states = set(splits["train"][STATE_COL].unique())
    val_states = set(splits["val"][STATE_COL].unique())
    missing = train_states - val_states
    if missing:
        logger.warning("States in train but not val: %s", missing)

    logger.info(
        "Splits validated: train=%d, val=%d, test=%d",
        len(splits["train"]), len(splits["val"]), len(splits.get("test", [])),
    )


def validate_artifacts_exist() -> None:
    """Check that required artifacts exist before starting the API."""
    from config.settings import ARTIFACTS_DIR

    required = [
        ARTIFACTS_DIR / "forecasts" / "final_8week_forecasts.csv",
        ARTIFACTS_DIR / "metrics" / "best_model_per_state.csv",
        ARTIFACTS_DIR / "metrics" / "all_models_metrics.csv",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        raise ValidationError(
            f"Missing artifacts: {[str(p) for p in missing]}. "
            "Run 'make train' or 'python scripts/train_all.py' first."
        )
    logger.info("All required artifacts present")
