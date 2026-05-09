"""Integration tests for the data pipeline (ingest -> features -> split -> dataset)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATE_COL, STATE_COL, TARGET_COL, FREQ
from src.features import ALL_FEATURE_COLS, build_features
from src.split import temporal_split
from src.dataset import (
    StateScaler, prepare_xgboost, prepare_lstm_sequences,
    XGBOOST_FEATURES, LSTM_FEATURES,
)
from src.evaluate import compute_metrics


@pytest.fixture
def pipeline_df():
    """120-week panel for 2 states — enough for splits and sequences."""
    dates = pd.date_range("2020-01-04", periods=120, freq=FREQ)
    rows = []
    rng = np.random.RandomState(99)
    for state in ["California", "Texas"]:
        base = 1000 if state == "California" else 500
        sales = base + np.sin(np.arange(120) * 2 * np.pi / 52) * 100 + rng.normal(0, 20, 120)
        for d, s in zip(dates, sales):
            rows.append({DATE_COL: d, STATE_COL: state, TARGET_COL: float(s)})
    return pd.DataFrame(rows)


class TestEndToEnd:
    def test_features_then_split_preserves_data(self, pipeline_df, tmp_path):
        """Features + split should not lose or duplicate rows."""
        import config.settings as settings
        orig = settings.PROCESSED_DATA_DIR
        settings.PROCESSED_DATA_DIR = tmp_path
        df = build_features(pipeline_df.copy())
        settings.PROCESSED_DATA_DIR = orig

        splits = temporal_split(df, val_start="2021-06-01", test_start="2022-01-01")
        total = sum(len(s) for s in splits.values())
        assert total == len(df), "Split lost or duplicated rows"

    def test_xgboost_dataset_no_nans(self, pipeline_df, tmp_path):
        """XGBoost prep should drop NaN rows, leaving no NaN features."""
        import config.settings as settings
        orig = settings.PROCESSED_DATA_DIR
        settings.PROCESSED_DATA_DIR = tmp_path
        df = build_features(pipeline_df.copy())
        settings.PROCESSED_DATA_DIR = orig

        X, y, meta = prepare_xgboost(df)
        assert not np.isnan(X).any(), "NaN survived in XGBoost features"
        assert len(X) > 0

    def test_lstm_sequences_shape(self, pipeline_df, tmp_path):
        """LSTM sequences should be 3D with correct dimensions."""
        import config.settings as settings
        orig = settings.PROCESSED_DATA_DIR
        settings.PROCESSED_DATA_DIR = tmp_path
        df = build_features(pipeline_df.copy())
        settings.PROCESSED_DATA_DIR = orig

        scaler = StateScaler().fit(df, LSTM_FEATURES)
        X, y, meta = prepare_lstm_sequences(df, scaler)
        assert X.ndim == 3
        assert X.shape[1] == 12  # LSTM_SEQ_LEN
        assert X.shape[2] == len(LSTM_FEATURES)
        assert len(y) == len(X)

    def test_scaler_roundtrip(self, pipeline_df, tmp_path):
        """Scale + inverse should recover original values."""
        import config.settings as settings
        orig = settings.PROCESSED_DATA_DIR
        settings.PROCESSED_DATA_DIR = tmp_path
        df = build_features(pipeline_df.copy())
        settings.PROCESSED_DATA_DIR = orig

        scaler = StateScaler().fit(df, LSTM_FEATURES)
        ca = df[df[STATE_COL] == "California"][TARGET_COL]
        scaled = scaler.transform_target(ca, "California")
        restored = scaler.inverse_target(scaled, "California")
        np.testing.assert_allclose(ca.values, restored.values, rtol=1e-6)

    def test_metrics_on_real_looking_data(self):
        """Smoke test: metrics should be finite on reasonable data."""
        actual = np.array([100, 200, 150, 180, 220])
        pred = np.array([110, 190, 160, 170, 230])
        train = np.array([80, 90, 100, 110, 120, 130, 140])
        m = compute_metrics(actual, pred, train_series=train)
        for k, v in m.items():
            assert np.isfinite(v), f"Metric {k} is not finite"
