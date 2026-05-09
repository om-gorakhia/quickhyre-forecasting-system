"""Tests for FastAPI endpoints."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def mock_store():
    """Patch ForecastStore with mock data so tests don't need artifacts."""
    from api.data import ForecastStore

    store = ForecastStore()
    store._loaded = True
    store.states = ["California", "Texas"]
    store.forecasts = pd.DataFrame([
        {"state": "California", "date": "2023-04-01", "forecast": 1000.0, "model_used": "lstm"},
        {"state": "California", "date": "2023-04-08", "forecast": 1050.0, "model_used": "lstm"},
        {"state": "Texas", "date": "2023-04-01", "forecast": 500.0, "model_used": "xgboost"},
        {"state": "Texas", "date": "2023-04-08", "forecast": 520.0, "model_used": "xgboost"},
    ])
    store.selection = pd.DataFrame([
        {"state": "California", "best_model": "lstm", "best_smape": 5.0, "best_mae": 100,
         "second_model": "xgboost", "ensemble_eligible": True},
        {"state": "Texas", "best_model": "xgboost", "best_smape": 6.0, "best_mae": 80,
         "second_model": "lstm", "ensemble_eligible": False},
    ])
    store.metrics = pd.DataFrame([
        {"state": "California", "model": "lstm", "mae": 100, "rmse": 120, "smape": 5.0, "mase": 0.8},
        {"state": "California", "model": "xgboost", "mae": 110, "rmse": 130, "smape": 6.0, "mase": 0.9},
        {"state": "Texas", "model": "xgboost", "mae": 80, "rmse": 95, "smape": 6.0, "mase": 0.7},
        {"state": "Texas", "model": "lstm", "mae": 90, "rmse": 100, "smape": 7.0, "mase": 0.85},
    ])
    store.leaderboard = pd.DataFrame([
        {"model": "lstm", "median_smape": 5.0, "mean_smape": 5.5, "median_mae": 95, "median_mase": 0.82},
        {"model": "xgboost", "median_smape": 6.0, "mean_smape": 6.0, "median_mae": 95, "median_mase": 0.8},
    ])
    return store


@pytest.fixture
def client(mock_store):
    with patch("api.routes.store", mock_store), patch("api.data.store", mock_store):
        from api.app import app
        with TestClient(app) as c:
            yield c


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "version" in data


class TestStates:
    def test_list_states(self, client):
        r = client.get("/api/v1/states")
        assert r.status_code == 200
        assert r.json() == ["California", "Texas"]


class TestForecast:
    def test_get_state_forecast(self, client):
        r = client.get("/api/v1/forecast/California")
        assert r.status_code == 200
        data = r.json()
        assert data["state"] == "California"
        assert len(data["forecasts"]) == 2
        assert data["forecasts"][0]["model_used"] == "lstm"

    def test_case_insensitive(self, client):
        r = client.get("/api/v1/forecast/california")
        assert r.status_code == 200
        assert r.json()["state"] == "California"

    def test_not_found(self, client):
        r = client.get("/api/v1/forecast/Narnia")
        assert r.status_code == 404

    def test_all_forecasts(self, client):
        r = client.get("/api/v1/forecast")
        assert r.status_code == 200
        data = r.json()
        assert data["total_states"] == 2

    def test_forecast_query_param(self, client):
        r = client.get("/api/v1/forecast?state=Texas")
        assert r.status_code == 200
        assert r.json()["state"] == "Texas"


class TestModelInfo:
    def test_model_info(self, client):
        r = client.get("/api/v1/model-info/California")
        assert r.status_code == 200
        data = r.json()
        assert data["best_model"] == "lstm"
        assert len(data["all_models"]) == 2

    def test_model_info_not_found(self, client):
        r = client.get("/api/v1/model-info/Narnia")
        assert r.status_code == 404


class TestMetrics:
    def test_leaderboard(self, client):
        r = client.get("/api/v1/metrics/summary")
        assert r.status_code == 200
        data = r.json()
        assert len(data["leaderboard"]) == 2
        assert data["total_states"] == 2
