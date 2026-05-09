"""API route definitions."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query

from api.schemas import (
    HealthResponse,
    StateForecastResponse,
    AllForecastsResponse,
    ModelInfo,
    MetricsSummaryResponse,
    ErrorResponse,
    ForecastPoint,
    LeaderboardEntry,
)
from api.data import store

API_VERSION = "1.0.0"

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if store.is_loaded else "degraded",
        timestamp=datetime.now(timezone.utc),
        version=API_VERSION,
    )


@router.get("/states", response_model=list[str])
def list_states():
    """List all available states."""
    return store.states


@router.get(
    "/forecast/{state}",
    response_model=StateForecastResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_state_forecast(state: str):
    """Get 8-week forecast for a specific state."""
    # Normalize input: title-case
    state = state.strip().title()
    if state not in store.states:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found. Use /states for valid options.")
    points = store.get_state_forecast(state)
    return StateForecastResponse(
        state=state,
        horizon_weeks=len(points),
        forecasts=[ForecastPoint(**p) for p in points],
        generated_at=datetime.now(timezone.utc),
    )


@router.get(
    "/forecast",
    response_model=StateForecastResponse | AllForecastsResponse,
    responses={404: {"model": ErrorResponse}},
)
def get_forecast(state: str | None = Query(default=None, description="State name (optional, returns all if omitted)")):
    """Get forecasts. Pass ?state=Texas for one state, or omit for all."""
    if state:
        state = state.strip().title()
        if state not in store.states:
            raise HTTPException(status_code=404, detail=f"State '{state}' not found.")
        points = store.get_state_forecast(state)
        return StateForecastResponse(
            state=state,
            horizon_weeks=len(points),
            forecasts=[ForecastPoint(**p) for p in points],
            generated_at=datetime.now(timezone.utc),
        )
    else:
        all_fc = store.get_all_forecasts()
        return AllForecastsResponse(
            total_states=len(all_fc),
            horizon_weeks=8,
            forecasts={s: [ForecastPoint(**p) for p in pts] for s, pts in all_fc.items()},
            generated_at=datetime.now(timezone.utc),
        )


@router.get(
    "/model-info/{state}",
    response_model=ModelInfo,
    responses={404: {"model": ErrorResponse}},
)
def get_model_info(state: str):
    """Get model selection details for a state."""
    state = state.strip().title()
    info = store.get_model_info(state)
    if info is None:
        raise HTTPException(status_code=404, detail=f"State '{state}' not found.")
    return ModelInfo(
        state=state,
        best_model=info["best_model"],
        best_smape=info["best_smape"],
        best_mae=info["best_mae"],
        second_model=info.get("second_model"),
        ensemble_eligible=info.get("ensemble_eligible", False),
        all_models=info["all_models"],
    )


@router.get("/metrics/summary", response_model=MetricsSummaryResponse)
def get_metrics_summary():
    """Get the global model leaderboard."""
    entries = store.get_leaderboard()
    return MetricsSummaryResponse(
        leaderboard=[LeaderboardEntry(**e) for e in entries],
        total_states=len(store.states),
        generated_at=datetime.now(timezone.utc),
    )
