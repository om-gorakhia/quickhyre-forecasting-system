"""Time-series-safe train/validation/test splitting.

Splits are purely temporal — no shuffling, no leakage.
Each state gets the same date boundaries so models are comparable.
"""

import logging

import pandas as pd

from config.settings import (
    DATE_COL, STATE_COL, TARGET_COL,
    VAL_START, TEST_START, SPLITS_DIR,
)

logger = logging.getLogger(__name__)


def temporal_split(
    df: pd.DataFrame,
    val_start: str = VAL_START,
    test_start: str = TEST_START,
) -> dict[str, pd.DataFrame]:
    """Split dataframe into train/val/test by date boundaries.

    Returns dict with keys 'train', 'val', 'test'.
    """
    val_dt = pd.Timestamp(val_start)
    test_dt = pd.Timestamp(test_start)

    train = df[df[DATE_COL] < val_dt].copy()
    val = df[(df[DATE_COL] >= val_dt) & (df[DATE_COL] < test_dt)].copy()
    test = df[df[DATE_COL] >= test_dt].copy()

    for name, split in [("train", train), ("val", val), ("test", test)]:
        n_states = split[STATE_COL].nunique()
        n_weeks = split[DATE_COL].nunique()
        date_range = f"{split[DATE_COL].min().date()} → {split[DATE_COL].max().date()}" if len(split) else "empty"
        logger.info("%-5s: %5d rows | %2d states | %3d weeks | %s", name, len(split), n_states, n_weeks, date_range)

    return {"train": train, "val": val, "test": test}


def save_splits(splits: dict[str, pd.DataFrame]) -> None:
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, split_df in splits.items():
        path = SPLITS_DIR / f"{name}.parquet"
        split_df.to_parquet(path, index=False)
    logger.info("Splits saved to %s", SPLITS_DIR)


def load_splits() -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(SPLITS_DIR / f"{name}.parquet")
        for name in ["train", "val", "test"]
    }


def get_state_split(splits: dict[str, pd.DataFrame], state: str) -> dict[str, pd.DataFrame]:
    """Filter splits for a single state."""
    return {k: v[v[STATE_COL] == state].copy() for k, v in splits.items()}
