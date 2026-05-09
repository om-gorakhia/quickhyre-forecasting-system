"""Shared fixtures for the test suite."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATE_COL, STATE_COL, TARGET_COL, FREQ


@pytest.fixture
def sample_weekly_df():
    """Minimal weekly panel: 2 states, 60 weeks each."""
    dates = pd.date_range("2021-01-02", periods=60, freq=FREQ)
    rows = []
    rng = np.random.RandomState(42)
    for state in ["California", "Texas"]:
        base = 1000 if state == "California" else 500
        sales = base + rng.normal(0, 50, size=len(dates)).cumsum()
        for d, s in zip(dates, sales):
            rows.append({DATE_COL: d, STATE_COL: state, TARGET_COL: max(s, 10)})
    return pd.DataFrame(rows)


@pytest.fixture
def sample_featured_df(sample_weekly_df):
    """Weekly panel with all features added."""
    from src.features import build_features
    import tempfile, os
    from config import settings

    # Redirect output to temp so tests don't pollute data/
    orig = settings.PROCESSED_DATA_DIR
    with tempfile.TemporaryDirectory() as tmp:
        settings.PROCESSED_DATA_DIR = Path(tmp)
        df = build_features(sample_weekly_df.copy())
        settings.PROCESSED_DATA_DIR = orig
    return df
