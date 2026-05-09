"""Pydantic models for API request validation and response serialization."""

from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    timestamp: datetime
    version: str


class ForecastPoint(BaseModel):
    date: str
    forecast: float
    model_used: str


class StateForecastResponse(BaseModel):
    state: str
    horizon_weeks: int
    forecasts: list[ForecastPoint]
    generated_at: datetime


class AllForecastsResponse(BaseModel):
    total_states: int
    horizon_weeks: int
    forecasts: dict[str, list[ForecastPoint]]
    generated_at: datetime


class ModelInfo(BaseModel):
    state: str
    best_model: str
    best_smape: float
    best_mae: float
    second_model: str | None
    ensemble_eligible: bool
    all_models: list[dict]


class LeaderboardEntry(BaseModel):
    model: str
    median_smape: float
    mean_smape: float
    median_mae: float
    median_mase: float
    states_won: int


class MetricsSummaryResponse(BaseModel):
    leaderboard: list[LeaderboardEntry]
    total_states: int
    generated_at: datetime


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
