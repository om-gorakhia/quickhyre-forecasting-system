"""Tests for evaluation metrics."""

import numpy as np
import pytest

from src.evaluate import mae, rmse, smape, mase, compute_metrics


class TestMAE:
    def test_perfect_prediction(self):
        a = np.array([1.0, 2.0, 3.0])
        assert mae(a, a) == 0.0

    def test_known_value(self):
        a = np.array([10.0, 20.0, 30.0])
        p = np.array([12.0, 18.0, 33.0])
        # |10-12| + |20-18| + |30-33| = 2 + 2 + 3 = 7, mean = 7/3
        assert mae(a, p) == pytest.approx(7.0 / 3)


class TestSMAPE:
    def test_perfect_prediction(self):
        a = np.array([1.0, 2.0, 3.0])
        assert smape(a, a) == pytest.approx(0.0)

    def test_bounded_0_200(self):
        a = np.array([100.0])
        p = np.array([0.0])
        assert 0 <= smape(a, p) <= 200

    def test_symmetric(self):
        a = np.array([10.0, 20.0])
        p = np.array([15.0, 25.0])
        assert smape(a, p) == pytest.approx(smape(p, a))


class TestMASE:
    def test_naive_baseline_short_series(self):
        """Short series falls back to period=1. MASE=1 when model matches naive."""
        train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([6.0, 7.0])
        predicted = np.array([5.0, 8.0])
        result = mase(actual, predicted, train)
        assert result == pytest.approx(1.0)

    def test_seasonal_period_52(self):
        """With 60+ weeks of training, uses seasonal_period=52."""
        rng = np.random.RandomState(0)
        train = 100 + rng.normal(0, 5, 60)  # 60 weeks > 52, uses seasonal period
        actual = np.array([100.0, 105.0])
        predicted = np.array([101.0, 104.0])
        result = mase(actual, predicted, train, seasonal_period=52)
        assert np.isfinite(result)
        assert result > 0


class TestComputeMetrics:
    def test_returns_all_keys(self):
        a = np.array([1.0, 2.0, 3.0])
        p = np.array([1.1, 2.2, 2.8])
        train = np.array([0.5, 1.0, 1.5, 2.0])
        m = compute_metrics(a, p, train_series=train)
        assert set(m.keys()) == {"mae", "rmse", "mape", "smape", "mase"}

    def test_without_train_series(self):
        a = np.array([1.0, 2.0])
        m = compute_metrics(a, a)
        assert "mase" not in m
