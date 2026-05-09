"""Data loading layer for the API.

Loads forecast artifacts once at startup and serves them from memory.
No model inference at request time — predictions are pre-computed.
"""

import logging
from pathlib import Path

import pandas as pd

from config.settings import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


class ForecastStore:
    """In-memory store for pre-computed forecasts and metrics."""

    def __init__(self):
        self.forecasts: pd.DataFrame | None = None
        self.selection: pd.DataFrame | None = None
        self.metrics: pd.DataFrame | None = None
        self.leaderboard: pd.DataFrame | None = None
        self.states: list[str] = []
        self._loaded = False

    def load(self) -> None:
        base = ARTIFACTS_DIR

        fc_path = base / "forecasts" / "final_8week_forecasts.csv"
        sel_path = base / "metrics" / "best_model_per_state.csv"
        metrics_path = base / "metrics" / "all_models_metrics.csv"
        lb_path = base / "metrics" / "leaderboard.csv"

        missing = [p for p in [fc_path, sel_path, metrics_path] if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"Required artifacts not found: {[str(p) for p in missing]}. "
                "Run 'python scripts/train_all.py' first."
            )

        self.forecasts = pd.read_csv(fc_path)
        self.selection = pd.read_csv(sel_path)
        self.metrics = pd.read_csv(metrics_path)
        if lb_path.exists():
            self.leaderboard = pd.read_csv(lb_path)

        self.states = sorted(self.forecasts["state"].unique().tolist())
        self._loaded = True
        logger.info("ForecastStore loaded: %d states, %d forecast rows", len(self.states), len(self.forecasts))

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def get_state_forecast(self, state: str) -> list[dict]:
        df = self.forecasts[self.forecasts["state"] == state].sort_values("date")
        return df[["date", "forecast", "model_used"]].to_dict(orient="records")

    def get_all_forecasts(self) -> dict[str, list[dict]]:
        result = {}
        for state in self.states:
            result[state] = self.get_state_forecast(state)
        return result

    def get_model_info(self, state: str) -> dict | None:
        row = self.selection[self.selection["state"] == state]
        if row.empty:
            return None
        r = row.iloc[0].to_dict()
        # Attach all model metrics for this state
        state_metrics = self.metrics[self.metrics["state"] == state]
        r["all_models"] = state_metrics[["model", "mae", "rmse", "smape", "mase"]].to_dict(orient="records")
        return r

    def get_leaderboard(self) -> list[dict]:
        if self.leaderboard is None:
            return []
        # Count wins per model
        wins = self.selection["best_model"].value_counts().to_dict()
        rows = []
        for _, r in self.leaderboard.iterrows():
            rows.append({
                "model": r.get("model", r.name) if "model" in r else r.name,
                "median_smape": r["median_smape"],
                "mean_smape": r["mean_smape"],
                "median_mae": r["median_mae"],
                "median_mase": r["median_mase"],
                "states_won": wins.get(r.get("model", r.name) if "model" in r else r.name, 0),
            })
        return rows


# Singleton
store = ForecastStore()
