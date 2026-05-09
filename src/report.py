"""Generate an HTML summary report of the full pipeline results.

Produces a single self-contained HTML file that an evaluator can open
in a browser — no server needed. Embeds plots as base64 images.
"""

import base64
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from config.settings import ARTIFACTS_DIR

logger = logging.getLogger(__name__)


def _img_tag(path: Path) -> str:
    """Embed a PNG as a base64 <img> tag."""
    if not path.exists():
        return f"<p><em>(plot not found: {path.name})</em></p>"
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%;margin:8px 0;">'


def _df_to_html(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Render a DataFrame as a styled HTML table."""
    return df.head(max_rows).to_html(
        index=False, classes="data-table", float_format=lambda x: f"{x:,.2f}",
    )


def generate_report() -> Path:
    """Build the HTML summary report from saved artifacts."""
    plots_dir = ARTIFACTS_DIR / "plots"
    metrics_dir = ARTIFACTS_DIR / "metrics"

    # Load data
    leaderboard = pd.read_csv(metrics_dir / "leaderboard.csv") if (metrics_dir / "leaderboard.csv").exists() else None
    selection = pd.read_csv(metrics_dir / "best_model_per_state.csv") if (metrics_dir / "best_model_per_state.csv").exists() else None
    worst = pd.read_csv(metrics_dir / "worst_states.csv") if (metrics_dir / "worst_states.csv").exists() else None
    backtest = pd.read_csv(metrics_dir / "backtest_results.csv") if (metrics_dir / "backtest_results.csv").exists() else None
    importance = pd.read_csv(metrics_dir / "xgboost_feature_importance.csv") if (metrics_dir / "xgboost_feature_importance.csv").exists() else None
    baseline = pd.read_csv(metrics_dir / "baseline_metrics.csv") if (metrics_dir / "baseline_metrics.csv").exists() else None
    ensemble_report = pd.read_csv(metrics_dir / "ensemble_validation.csv") if (metrics_dir / "ensemble_validation.csv").exists() else None

    # Winner counts
    winner_counts = ""
    if selection is not None:
        counts = selection["best_model"].value_counts()
        winner_counts = ", ".join(f"{m}: {c}" for m, c in counts.items())
        ensemble_n = int(selection["ensemble_eligible"].sum())
        winner_counts += f" | Ensemble-eligible: {ensemble_n}/{len(selection)}"

    # Baseline comparison
    baseline_section = ""
    if baseline is not None:
        baseline_agg = baseline.groupby("model")["smape"].median().round(2)
        baseline_section = f"""
        <h2>Baseline Comparison</h2>
        <p>Any useful model must beat these naive benchmarks:</p>
        <table class="data-table">
            <tr><th>Baseline</th><th>Median sMAPE</th></tr>
            {"".join(f'<tr><td>{m}</td><td>{v:.2f}%</td></tr>' for m, v in baseline_agg.items())}
        </table>
        <p>Model improvement over seasonal naive:</p>
        <table class="data-table">
            <tr><th>Model</th><th>Median sMAPE</th><th>vs Seasonal Naive</th></tr>
        """
        if leaderboard is not None:
            sn_smape = baseline_agg.get("seasonal_naive", None)
            for _, row in leaderboard.iterrows():
                model = row.get("model", row.name) if "model" in row else row.name
                ms = row["median_smape"]
                if sn_smape and sn_smape > 0:
                    improvement = f"{(1 - ms/sn_smape)*100:.0f}% better"
                else:
                    improvement = "N/A"
                baseline_section += f"<tr><td>{model}</td><td>{ms:.2f}%</td><td>{improvement}</td></tr>"
        baseline_section += "</table>"

    # Ensemble validation
    ensemble_section = ""
    if ensemble_report is not None and len(ensemble_report) > 0:
        improved = (ensemble_report["ensemble_smape"] < ensemble_report["best_single_smape"]).sum()
        total = len(ensemble_report)
        avg_delta = (ensemble_report["best_single_smape"] - ensemble_report["ensemble_smape"]).mean()
        ensemble_section = f"""
        <h2>Ensemble Validation</h2>
        <p>Measured on validation set: ensemble improved {improved}/{total} eligible states
        (avg sMAPE reduction: {avg_delta:.2f} pp).</p>
        {_df_to_html(ensemble_report.head(15))}
        """

    # Backtest
    backtest_section = ""
    if backtest is not None:
        bt_summary = backtest.groupby("model")["smape"].agg(["median", "mean", "std"]).round(2)
        backtest_section = f"""
        <h2>Walk-Forward Backtest</h2>
        <p>3-fold expanding window, 13-week horizon per fold. Validates model stability across time.</p>
        {bt_summary.to_html(classes="data-table")}
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Sales Forecasting - Pipeline Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         max-width: 960px; margin: 40px auto; padding: 0 20px; color: #333;
         line-height: 1.6; }}
  h1 {{ border-bottom: 2px solid #2196F3; padding-bottom: 8px; }}
  h2 {{ color: #1976D2; margin-top: 32px; }}
  .data-table {{ border-collapse: collapse; width: 100%; margin: 12px 0; }}
  .data-table th, .data-table td {{ border: 1px solid #ddd; padding: 8px 12px;
                                     text-align: left; }}
  .data-table th {{ background: #f5f5f5; font-weight: 600; }}
  .data-table tr:nth-child(even) {{ background: #fafafa; }}
  .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                   gap: 12px; margin: 16px 0; }}
  .metric-card {{ background: #f5f5f5; padding: 16px; border-radius: 8px;
                   border-left: 4px solid #2196F3; }}
  .metric-card .value {{ font-size: 24px; font-weight: 700; color: #1976D2; }}
  .metric-card .label {{ font-size: 13px; color: #666; }}
  .plot-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .plot-grid img {{ border: 1px solid #eee; border-radius: 4px; }}
  .timestamp {{ color: #999; font-size: 12px; }}
  @media (max-width: 700px) {{ .plot-grid {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<h1>Beverage Sales Forecasting &mdash; Pipeline Report</h1>
<p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

<div class="metric-grid">
  <div class="metric-card">
    <div class="value">43</div>
    <div class="label">US States</div>
  </div>
  <div class="metric-card">
    <div class="value">4</div>
    <div class="label">Model Families</div>
  </div>
  <div class="metric-card">
    <div class="value">8</div>
    <div class="label">Forecast Weeks</div>
  </div>
  <div class="metric-card">
    <div class="value">{winner_counts.split('|')[0].strip() if winner_counts else 'N/A'}</div>
    <div class="label">Model Wins</div>
  </div>
</div>

<h2>Model Leaderboard</h2>
{_df_to_html(leaderboard) if leaderboard is not None else '<p>Not available</p>'}

<div class="plot-grid">
{_img_tag(plots_dir / "leaderboard.png")}
{_img_tag(plots_dir / "winner_distribution.png")}
</div>

{baseline_section}

<h2>Model Selection</h2>
<p>Per-state winner by sMAPE: {winner_counts}</p>
{_df_to_html(selection) if selection is not None else ''}

<h2>Hardest States</h2>
{_df_to_html(worst) if worst is not None else '<p>Not available</p>'}

{ensemble_section}

{backtest_section}

<h2>Feature Importance (XGBoost)</h2>
{_df_to_html(importance.head(10)) if importance is not None else '<p>Not available</p>'}

<h2>Visualizations</h2>
<div class="plot-grid">
{_img_tag(plots_dir / "state_comparison.png")}
{_img_tag(plots_dir / "lstm_training_curves.png")}
{_img_tag(plots_dir / "forecast_california.png")}
{_img_tag(plots_dir / "forecast_texas.png")}
</div>

<h2>Reproducibility</h2>
<table class="data-table">
  <tr><th>Control</th><th>Value</th></tr>
  <tr><td>Python seed</td><td>42</td></tr>
  <tr><td>NumPy seed</td><td>42</td></tr>
  <tr><td>PyTorch seed</td><td>42</td></tr>
  <tr><td>XGBoost random_state</td><td>42</td></tr>
  <tr><td>PYTHONHASHSEED</td><td>Set via Makefile</td></tr>
  <tr><td>Train/val split</td><td>Temporal (2022-10-01)</td></tr>
  <tr><td>Data leakage checks</td><td>shift(1) rolling, per-state groupby, train-only scaler</td></tr>
</table>

</body>
</html>"""

    out = ARTIFACTS_DIR / "report.html"
    out.write_text(html, encoding="utf-8")
    logger.info("Report saved -> %s", out)
    return out
