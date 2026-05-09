"""Model selection and optional ensemble.

Selection rule (simple and defensible):
  For each state, pick the model with the lowest sMAPE on the validation set.
  sMAPE is chosen because it's scale-invariant (fair across California and
  Wyoming) and symmetric (penalizes over- and under-prediction equally).

  Tie-breaking: lower MASE wins (rewards beating the naive baseline by more).

Ensemble (optional, opt-in):
  Weighted average of the top-2 models per state, weighted by inverse sMAPE.
  Only used if the 2nd-best model's sMAPE is within 50% of the best.
  Otherwise the best model's prediction is used alone.

  Rationale: simple ensembles reduce variance and smooth out per-step
  prediction noise from recursive forecasting. No stacking, no meta-learner —
  just a weighted average that's easy to explain in an interview.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import ARTIFACTS_DIR, STATE_COL

logger = logging.getLogger(__name__)

MODEL_NAMES = ["sarima", "prophet", "xgboost", "lstm"]
PRIMARY_METRIC = "smape"
TIEBREAK_METRIC = "mase"

# Ensemble: include 2nd model only if its sMAPE is within this ratio of the best
ENSEMBLE_THRESHOLD = 1.5


def load_all_metrics() -> pd.DataFrame:
    """Load per-state metrics from all trained models into one dataframe."""
    dfs = []
    for name in MODEL_NAMES:
        path = ARTIFACTS_DIR / "metrics" / f"{name}_metrics.csv"
        if path.exists():
            dfs.append(pd.read_csv(path))
        else:
            logger.warning("Metrics not found for %s — skipping", name)
    if not dfs:
        raise FileNotFoundError("No model metrics found in artifacts/metrics/")
    return pd.concat(dfs, ignore_index=True)


def select_best_per_state(metrics: pd.DataFrame) -> pd.DataFrame:
    """Select the best model for each state.

    Returns a dataframe with columns: state, best_model, smape, mase,
    second_model, second_smape, ensemble_eligible.
    """
    rows = []
    for state, grp in metrics.groupby("state"):
        ranked = grp.sort_values(
            [PRIMARY_METRIC, TIEBREAK_METRIC], ascending=True
        ).reset_index(drop=True)

        best = ranked.iloc[0]
        result = {
            "state": state,
            "best_model": best["model"],
            "best_smape": best[PRIMARY_METRIC],
            "best_mase": best.get(TIEBREAK_METRIC, np.nan),
            "best_mae": best["mae"],
        }

        if len(ranked) > 1:
            second = ranked.iloc[1]
            result["second_model"] = second["model"]
            result["second_smape"] = second[PRIMARY_METRIC]
            ratio = second[PRIMARY_METRIC] / best[PRIMARY_METRIC] if best[PRIMARY_METRIC] > 0 else 99
            result["ensemble_eligible"] = ratio <= ENSEMBLE_THRESHOLD
        else:
            result["second_model"] = None
            result["second_smape"] = np.nan
            result["ensemble_eligible"] = False

        rows.append(result)

    selection = pd.DataFrame(rows)
    logger.info(
        "Selection complete: %s",
        selection["best_model"].value_counts().to_dict(),
    )
    return selection


def compute_ensemble_weights(best_smape: float, second_smape: float) -> tuple[float, float]:
    """Inverse-sMAPE weighting for the top-2 ensemble."""
    inv_best = 1.0 / max(best_smape, 0.01)
    inv_second = 1.0 / max(second_smape, 0.01)
    total = inv_best + inv_second
    return inv_best / total, inv_second / total


def build_leaderboard(metrics: pd.DataFrame) -> pd.DataFrame:
    """Global leaderboard: aggregate metrics per model."""
    agg = metrics.groupby("model").agg(
        median_smape=(PRIMARY_METRIC, "median"),
        mean_smape=(PRIMARY_METRIC, "mean"),
        median_mae=("mae", "median"),
        median_rmse=("rmse", "median"),
        median_mase=(TIEBREAK_METRIC, "median"),
        states_count=("state", "count"),
    ).sort_values("median_smape")
    return agg.round(2)


def build_state_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    """Pivot: rows=states, columns=models, values=sMAPE. Highlights winner."""
    pivot = metrics.pivot_table(
        index="state", columns="model", values=PRIMARY_METRIC
    )
    pivot["best_model"] = pivot.idxmin(axis=1)
    pivot["best_smape"] = pivot.drop(columns="best_model").min(axis=1)
    return pivot.sort_values("best_smape").round(2)


def find_worst_states(metrics: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Find states where even the best model struggles."""
    best_per_state = metrics.loc[metrics.groupby("state")[PRIMARY_METRIC].idxmin()]
    return (
        best_per_state
        .sort_values(PRIMARY_METRIC, ascending=False)
        .head(n)[["state", "model", PRIMARY_METRIC, "mae", TIEBREAK_METRIC]]
        .round(2)
    )


def save_selection_artifacts(
    metrics: pd.DataFrame,
    selection: pd.DataFrame,
    output_dir: Path | None = None,
) -> None:
    """Save all summary tables to CSV."""
    out = output_dir or ARTIFACTS_DIR / "metrics"
    out.mkdir(parents=True, exist_ok=True)

    # Combined metrics
    metrics.to_csv(out / "all_models_metrics.csv", index=False)

    # Selection table
    selection.to_csv(out / "best_model_per_state.csv", index=False)

    # Leaderboard
    leaderboard = build_leaderboard(metrics)
    leaderboard.to_csv(out / "leaderboard.csv")

    # State comparison pivot
    comparison = build_state_comparison(metrics)
    comparison.to_csv(out / "state_comparison.csv")

    # Worst states
    worst = find_worst_states(metrics)
    worst.to_csv(out / "worst_states.csv", index=False)

    logger.info("Selection artifacts saved to %s", out)
