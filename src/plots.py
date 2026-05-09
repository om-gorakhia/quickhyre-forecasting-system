"""Summary plots for the forecasting pipeline.

Generates publication-ready matplotlib figures for the assignment demo.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.settings import ARTIFACTS_DIR, STATE_COL, DATE_COL, TARGET_COL

logger = logging.getLogger(__name__)

PLOT_DIR = ARTIFACTS_DIR / "plots"
COLORS = {"sarima": "#2196F3", "prophet": "#FF9800", "xgboost": "#4CAF50", "lstm": "#E91E63", "ensemble": "#9C27B0"}


def plot_leaderboard(metrics: pd.DataFrame) -> Path:
    """Bar chart of median sMAPE per model."""
    agg = metrics.groupby("model")["smape"].median().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(agg.index, agg.values, color=[COLORS.get(m, "#999") for m in agg.index])
    ax.bar_label(bars, fmt="%.2f%%", padding=4)
    ax.set_xlabel("Median sMAPE (%)")
    ax.set_title("Model Leaderboard — Median sMAPE Across States")
    ax.invert_yaxis()
    plt.tight_layout()
    path = PLOT_DIR / "leaderboard.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_state_comparison(metrics: pd.DataFrame, top_n: int = 15) -> Path:
    """Grouped bar chart showing sMAPE per model for the top-N most interesting states."""
    # Pick states with highest variance in model performance (most interesting)
    spread = metrics.groupby("state")["smape"].agg(["min", "max"])
    spread["range"] = spread["max"] - spread["min"]
    interesting = spread.nlargest(top_n, "range").index

    subset = metrics[metrics["state"].isin(interesting)]
    pivot = subset.pivot_table(index="state", columns="model", values="smape")
    pivot = pivot.loc[pivot.min(axis=1).sort_values().index]

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot.plot(kind="barh", ax=ax, color=[COLORS.get(c, "#999") for c in pivot.columns])
    ax.set_xlabel("sMAPE (%)")
    ax.set_title(f"Per-State sMAPE — Top {top_n} States by Model Spread")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = PLOT_DIR / "state_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_winner_distribution(selection: pd.DataFrame) -> Path:
    """Pie chart of which model wins how many states."""
    counts = selection["best_model"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = [COLORS.get(m, "#999") for m in counts.index]
    ax.pie(counts.values, labels=[f"{m}\n({c} states)" for m, c in counts.items()],
           colors=colors, autopct="%1.0f%%", startangle=90)
    ax.set_title("Best Model Distribution Across States")
    plt.tight_layout()
    path = PLOT_DIR / "winner_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_forecast_sample(
    state: str,
    history_df: pd.DataFrame,
    forecasts: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    n_history_weeks: int = 52,
) -> Path:
    """Plot historical data + forecast for a single state."""
    hist = history_df[history_df[STATE_COL] == state].sort_values(DATE_COL).tail(n_history_weeks)
    fc = forecasts[forecasts["state"] == state].sort_values("date")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hist[DATE_COL], hist[TARGET_COL], "k-", linewidth=1.5, label="Historical")

    if val_df is not None:
        val_state = val_df[val_df[STATE_COL] == state].sort_values(DATE_COL)
        if len(val_state):
            ax.plot(val_state[DATE_COL], val_state[TARGET_COL], "k--", linewidth=1, alpha=0.5, label="Actual (val)")

    model_used = fc["model_used"].iloc[0] if len(fc) else "?"
    ax.plot(fc["date"], fc["forecast"], "r-o", markersize=4, linewidth=2, label=f"Forecast ({model_used})")

    ax.set_title(f"{state} — 8-Week Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales ($)")
    ax.legend()
    ax.ticklabel_format(style="plain", axis="y")
    plt.xticks(rotation=30)
    plt.tight_layout()

    safe = state.lower().replace(" ", "_")
    path = PLOT_DIR / f"forecast_{safe}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_training_curves() -> Path | None:
    """Plot LSTM training/validation loss curves."""
    csv_path = ARTIFACTS_DIR / "metrics" / "lstm_training_curves.csv"
    if not csv_path.exists():
        return None

    curves = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(curves["epoch"], curves["train_loss"], label="Train Loss")
    ax.plot(curves["epoch"], curves["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (scaled)")
    ax.set_title("LSTM Training Curves")
    ax.legend()
    plt.tight_layout()
    path = PLOT_DIR / "lstm_training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all_plots(
    metrics: pd.DataFrame,
    selection: pd.DataFrame,
    forecasts: pd.DataFrame,
    history_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    sample_states: list[str] | None = None,
) -> list[Path]:
    """Generate all summary plots. Returns list of saved paths."""
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    paths.append(plot_leaderboard(metrics))
    paths.append(plot_state_comparison(metrics))
    paths.append(plot_winner_distribution(selection))

    tc = plot_training_curves()
    if tc:
        paths.append(tc)

    # Sample forecasts: pick 4 diverse states if not specified
    if sample_states is None:
        sample_states = ["California", "Wyoming", "Texas", "Iowa"]
    for state in sample_states:
        if state in forecasts["state"].values:
            paths.append(plot_forecast_sample(state, history_df, forecasts, val_df))

    logger.info("Generated %d plots in %s", len(paths), PLOT_DIR)
    return paths
