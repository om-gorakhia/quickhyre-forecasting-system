"""Smoke tests for model training and prediction.

These use tiny synthetic data to verify the full fit->predict cycle works
without errors — not to check accuracy. Runs in <10 seconds total.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import DATE_COL, STATE_COL, TARGET_COL, FREQ
from src.features import build_features, ALL_FEATURE_COLS


@pytest.fixture
def tiny_panel(tmp_path):
    """120-week panel for 2 states — enough for lag warm-up (52 weeks) + training + val."""
    import config.settings as settings
    orig = settings.PROCESSED_DATA_DIR

    dates = pd.date_range("2019-01-05", periods=120, freq=FREQ)
    rng = np.random.RandomState(42)
    rows = []
    for state in ["StateA", "StateB"]:
        base = 500 if state == "StateA" else 300
        sales = base + np.sin(np.arange(120) * 2 * np.pi / 52) * 50 + rng.normal(0, 10, 120)
        for d, s in zip(dates, sales):
            rows.append({DATE_COL: d, STATE_COL: state, TARGET_COL: max(float(s), 1.0)})

    df = pd.DataFrame(rows)
    settings.PROCESSED_DATA_DIR = tmp_path
    df = build_features(df)
    settings.PROCESSED_DATA_DIR = orig

    # Split: first 90 weeks train, last 30 val
    split_date = dates[89]
    train = df[df[DATE_COL] <= split_date].copy()
    val = df[df[DATE_COL] > split_date].copy()
    return train, val


class TestXGBoost:
    def test_fit_predict_cycle(self, tiny_panel):
        from src.models.xgboost_model import XGBoostForecaster

        train, val = tiny_panel
        model = XGBoostForecaster(params={"n_estimators": 10, "max_depth": 3})
        model.fit(train, val)

        preds = model.predict_recursive(train, "StateA", horizon=4)
        assert len(preds) == 4
        assert "forecast" in preds.columns
        assert "date" in preds.columns
        assert all(np.isfinite(preds["forecast"]))

    def test_feature_importance(self, tiny_panel):
        from src.models.xgboost_model import XGBoostForecaster

        train, val = tiny_panel
        model = XGBoostForecaster(params={"n_estimators": 50, "max_depth": 4})
        model.fit(train, val)

        imp = model.get_feature_importance()
        assert len(imp) > 0
        assert imp["importance"].sum() > 0  # features are used

    def test_save_load_roundtrip(self, tiny_panel, tmp_path):
        from src.models.xgboost_model import XGBoostForecaster

        train, val = tiny_panel
        model = XGBoostForecaster(params={"n_estimators": 10})
        model.fit(train, val)

        path = tmp_path / "xgb_test.pkl"
        model.save(path)
        loaded = XGBoostForecaster.load(path)

        p1 = model.predict_recursive(train, "StateA", horizon=2)
        p2 = loaded.predict_recursive(train, "StateA", horizon=2)
        np.testing.assert_allclose(p1["forecast"].values, p2["forecast"].values)


class TestLSTM:
    def test_fit_predict_cycle(self, tiny_panel):
        from src.models.lstm_model import LSTMForecaster
        from src.dataset import StateScaler, prepare_lstm_sequences, LSTM_FEATURES

        train, val = tiny_panel
        scaler = StateScaler().fit(train, LSTM_FEATURES)
        X_train, y_train, _ = prepare_lstm_sequences(train, scaler)
        X_val, y_val, _ = prepare_lstm_sequences(val, scaler)

        model = LSTMForecaster(hparams={"max_epochs": 3, "patience": 2})
        model.fit(X_train, y_train, X_val, y_val, scaler)

        preds = model.predict_recursive(train, "StateA", horizon=4)
        assert len(preds) == 4
        assert all(np.isfinite(preds["forecast"]))

    def test_training_curves_recorded(self, tiny_panel):
        from src.models.lstm_model import LSTMForecaster
        from src.dataset import StateScaler, prepare_lstm_sequences, LSTM_FEATURES

        train, val = tiny_panel
        scaler = StateScaler().fit(train, LSTM_FEATURES)
        X_train, y_train, _ = prepare_lstm_sequences(train, scaler)
        X_val, y_val, _ = prepare_lstm_sequences(val, scaler)

        model = LSTMForecaster(hparams={"max_epochs": 3, "patience": 2})
        model.fit(X_train, y_train, X_val, y_val, scaler)

        assert len(model.train_losses) > 0
        assert len(model.val_losses) > 0


class TestSelector:
    def test_ensemble_weights_sum_to_one(self):
        from src.selector import compute_ensemble_weights

        w1, w2 = compute_ensemble_weights(5.0, 8.0)
        assert w1 + w2 == pytest.approx(1.0)
        assert w1 > w2  # better model gets more weight

    def test_selection_picks_lowest_smape(self):
        from src.selector import select_best_per_state

        metrics = pd.DataFrame([
            {"state": "A", "model": "xgboost", "smape": 5.0, "mase": 1.0, "mae": 100, "rmse": 120, "mape": 5.0},
            {"state": "A", "model": "lstm", "smape": 3.0, "mase": 0.8, "mae": 80, "rmse": 100, "mape": 3.0},
        ])
        result = select_best_per_state(metrics)
        assert result.iloc[0]["best_model"] == "lstm"


class TestBaseline:
    def test_naive_baselines(self, tiny_panel):
        from src.baseline import compute_naive_baselines

        train, val = tiny_panel
        result = compute_naive_baselines(train, val)
        assert len(result) > 0
        assert "naive" in result["model"].values
        assert "seasonal_naive" in result["model"].values
        assert result["smape"].notna().all()
