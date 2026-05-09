"""Generate a polished demo video for the QuickHyre forecasting assignment.

Uses matplotlib for slide generation, edge-tts for voiceover, and moviepy
for compositing. Produces a 3-5 minute video with slides, narration, and
subtitles.

Usage:
    python demo/generate_video.py

Output:
    demo/Sales_Forecasting_Demo.mp4
    demo/subtitles.srt
    demo/thumbnail.png
"""

import asyncio
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEMO_DIR = Path(__file__).parent
PROJECT_ROOT = DEMO_DIR.parent
SLIDES_DIR = DEMO_DIR / "slides"
AUDIO_DIR = DEMO_DIR / "audio"
WIDTH, HEIGHT = 1920, 1080
FPS = 30
BG_COLOR = "#0F172A"        # dark navy
ACCENT = "#3B82F6"          # blue
ACCENT2 = "#10B981"         # green
TEXT_COLOR = "#F8FAFC"       # near-white
MUTED = "#94A3B8"           # gray
CARD_BG = "#1E293B"         # darker card

VOICE = "en-US-GuyNeural"   # professional male voice

# ---------------------------------------------------------------------------
# Scene definitions
# ---------------------------------------------------------------------------

SCENES = [
    {
        "id": "title",
        "duration": 12,
        "narration": (
            "This is a demo of my end-to-end sales forecasting system, "
            "built as a data science assignment for QuickHyre. "
            "The system takes weekly beverage sales data across 43 US states, "
            "trains four different model families, selects the best per state, "
            "and serves 8-week forecasts through a REST API. "
            "Let me walk you through how it works."
        ),
    },
    {
        "id": "dataset",
        "duration": 15,
        "narration": (
            "The dataset contains 8,084 rows of weekly beverage sales across 43 US states, "
            "spanning from January 2019 to March 2023. "
            "The raw data has two challenges: mixed date formats in the same column, "
            "and irregular spacing that leaves 37% of the weekly grid missing after resampling. "
            "The pipeline handles both — parsing mixed dates with pandas, "
            "then filling gaps with linear interpolation. "
            "After cleaning, we have 11,051 rows on a regular weekly grid."
        ),
    },
    {
        "id": "features",
        "duration": 16,
        "narration": (
            "The feature engineering pipeline builds 23 features per row. "
            "These include 4 lag features at 1, 7, 30, and 52 weeks back; "
            "6 rolling statistics — means and standard deviations over 4, 8, and 13-week windows; "
            "cyclical sin-cosine encodings for month and week of year; "
            "a holiday flag, trend proxy, and per-state expanding mean. "
            "The key design point is leakage prevention. "
            "Every rolling feature uses shift-1 before computing the window, "
            "so the current row's target never appears in its own features. "
            "All transforms are per-state via groupby — no cross-state information leakage."
        ),
    },
    {
        "id": "models",
        "duration": 18,
        "narration": (
            "I train four fundamentally different model families. "
            "SARIMA, a classical statistical model fitted per state using auto-ARIMA with yearly seasonality. "
            "Prophet, Facebook's additive model with US holidays and multiplicative seasonality. "
            "XGBoost, a single global gradient boosting model pooling all 43 states — "
            "because 142 rows per state isn't enough for 23 features. "
            "And LSTM, a 2-layer PyTorch recurrent network, also trained globally. "
            "Both XGBoost and LSTM use recursive multi-step forecasting: "
            "predict one week, append the prediction to history, rebuild all features, "
            "and repeat 8 times. No future data is ever used."
        ),
    },
    {
        "id": "results",
        "duration": 18,
        "narration": (
            "Here are the results. LSTM leads with 5.08% median symmetric MAPE, "
            "followed by XGBoost at 6.27%. "
            "SARIMA and Prophet trail at 12 and 18 percent. "
            "For context, the naive baseline — just repeating last week's value — scores 5.58%. "
            "So LSTM does beat naive, but the improvement is modest. "
            "The real story is in the feature importance: "
            "rolling means account for 90% of XGBoost's signal, "
            "meaning this data is dominated by simple momentum. "
            "LSTM wins 27 of 43 states, XGBoost wins 15. "
            "26 states use an ensemble blend of the top two, "
            "which improved 19 of those 26 states by an average 1.4 percentage points."
        ),
    },
    {
        "id": "backtest",
        "duration": 12,
        "narration": (
            "A single train-validation split can be misleading. "
            "The walk-forward backtest runs 3 folds with a 13-week horizon each, "
            "using an expanding training window. "
            "XGBoost proves more stable with a standard deviation of 2.88 "
            "versus 4.57 for LSTM. "
            "This confirms the ensemble strategy: "
            "LSTM adds value on specific states, but XGBoost provides the reliable foundation."
        ),
    },
    {
        "id": "api",
        "duration": 14,
        "narration": (
            "The forecasts are served through a FastAPI application with 6 endpoints. "
            "Health check, list states, forecast by state, model info, and a global leaderboard. "
            "The API loads pre-computed predictions at startup — "
            "no model inference at request time, so responses are sub-millisecond. "
            "State names are case-insensitive. "
            "Interactive Swagger documentation is auto-generated at the docs endpoint. "
            "The entire system is tested with 57 automated tests covering features, "
            "metrics, ingestion, all four models, API endpoints, and validation guards."
        ),
    },
    {
        "id": "closing",
        "duration": 12,
        "narration": (
            "To summarize: this system handles the full lifecycle — "
            "from messy raw data to clean predictions served through an API. "
            "23 leakage-free features, 4 model families, "
            "validated ensemble selection, walk-forward backtesting, "
            "57 tests, and an HTML report that summarizes everything. "
            "The honest limitation is the 37% imputation rate and no exogenous variables. "
            "The natural next step would be probabilistic forecasting — "
            "prediction intervals instead of point estimates. "
            "Thanks for watching."
        ),
    },
]


# ---------------------------------------------------------------------------
# Slide rendering helpers
# ---------------------------------------------------------------------------

def _fig(dpi=100):
    fig = plt.figure(figsize=(WIDTH / dpi, HEIGHT / dpi), dpi=dpi, facecolor=BG_COLOR)
    return fig


def _text(ax, x, y, text, size=28, color=TEXT_COLOR, ha="left", va="top",
          weight="normal", family="sans-serif", wrap_width=None):
    if wrap_width:
        text = "\n".join(textwrap.wrap(text, width=wrap_width))
    ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va,
            weight=weight, family=family, transform=ax.transAxes)


def _card(ax, x, y, w, h, color=CARD_BG):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015",
        facecolor=color, edgecolor="#334155", linewidth=1.5,
        transform=ax.transAxes,
    )
    ax.add_patch(rect)


def _badge(ax, x, y, text, bg=ACCENT, size=16):
    ax.text(x, y, f"  {text}  ", fontsize=size, color="white", weight="bold",
            ha="left", va="center", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor=bg, edgecolor="none"))


def _save(fig, name):
    path = SLIDES_DIR / f"{name}.png"
    fig.savefig(path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return path


def _header(ax, title, subtitle=None):
    _badge(ax, 0.05, 0.93, "QuickHyre Assignment", bg="#6366F1", size=14)
    _text(ax, 0.05, 0.88, title, size=38, weight="bold")
    if subtitle:
        _text(ax, 0.05, 0.82, subtitle, size=20, color=MUTED)


# ---------------------------------------------------------------------------
# Individual slide generators
# ---------------------------------------------------------------------------

def slide_title():
    fig = _fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # Big title
    _text(ax, 0.5, 0.65, "Beverage Sales\nForecasting System",
          size=56, weight="bold", ha="center", va="center")
    _text(ax, 0.5, 0.48, "QuickHyre Data Science Assignment",
          size=26, color=ACCENT, ha="center", va="center", weight="bold")
    _text(ax, 0.5, 0.40,
          "4 Models  •  43 States  •  REST API  •  57 Tests",
          size=20, color=MUTED, ha="center", va="center")

    # Stat cards
    stats = [("43", "US States"), ("23", "Features"), ("4", "Model Families"), ("8", "Forecast Weeks")]
    for i, (val, label) in enumerate(stats):
        cx = 0.15 + i * 0.2
        _card(ax, cx - 0.07, 0.18, 0.14, 0.12)
        _text(ax, cx, 0.27, val, size=36, weight="bold", ha="center", va="center", color=ACCENT)
        _text(ax, cx, 0.21, label, size=14, color=MUTED, ha="center", va="center")

    return _save(fig, "01_title")


def slide_dataset():
    fig = _fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _header(ax, "Dataset & Preprocessing", "From messy Excel to clean weekly panel")

    # Data stats table
    rows = [
        ("Raw rows", "8,084"),
        ("States", "43"),
        ("Date range", "Jan 2019 – Mar 2023"),
        ("After resampling", "11,051 rows (43 × 257 weeks)"),
        ("Imputation rate", "37.4% of weekly grid"),
        ("Date formats", "Mixed YYYY-MM-DD & DD-MM-YYYY"),
    ]
    for i, (k, v) in enumerate(rows):
        y = 0.72 - i * 0.065
        _text(ax, 0.08, y, k, size=20, color=MUTED)
        _text(ax, 0.40, y, v, size=20, weight="bold")

    # Pipeline steps
    _text(ax, 0.08, 0.30, "Pipeline Steps", size=24, weight="bold", color=ACCENT)
    steps = ["1. Load Excel, normalize headers",
             "2. Parse mixed date formats",
             "3. Validate & deduplicate",
             "4. Resample to W-SAT weekly grid",
             "5. Linear interpolation for gaps",
             "6. Post-validate: zero nulls"]
    for i, step in enumerate(steps):
        _text(ax, 0.08, 0.24 - i * 0.038, step, size=17, color="#CBD5E1")

    return _save(fig, "02_dataset")


def slide_features():
    fig = _fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _header(ax, "Feature Engineering", "23 features with documented leakage prevention")

    groups = [
        ("Calendar (11)", ["month, week_of_year, year, quarter",
                           "month_sin, month_cos (cyclical)",
                           "week_of_year_sin, week_of_year_cos",
                           "is_holiday_week, week_index, day_of_week"]),
        ("Lags (4)", ["lag_1  (1 week)",
                      "lag_7  (7 weeks)",
                      "lag_30 (30 weeks)",
                      "lag_52 (52 weeks / yearly)"]),
        ("Rolling (6)", ["roll_mean/std over 4, 8, 13 weeks",
                         "shift(1) before rolling = no leakage"]),
        ("Derived (2)", ["pct_change_1 (shifted)",
                         "state_expanding_mean (shifted)"]),
    ]

    y0 = 0.74
    for gi, (title, items) in enumerate(groups):
        cx = 0.06 + gi * 0.235
        _card(ax, cx, y0 - 0.03, 0.22, 0.06 + len(items) * 0.035)
        _text(ax, cx + 0.01, y0 + 0.025, title, size=18, weight="bold", color=ACCENT)
        for j, item in enumerate(items):
            _text(ax, cx + 0.01, y0 - 0.015 - j * 0.035, item, size=14, color="#CBD5E1")

    # Leakage prevention callout
    _card(ax, 0.06, 0.12, 0.88, 0.14, color="#1A2332")
    _badge(ax, 0.08, 0.23, "LEAKAGE PREVENTION", bg="#DC2626", size=14)
    _text(ax, 0.08, 0.19, "• shift(1) before every rolling window — current target excluded",
          size=16, color="#FCA5A5")
    _text(ax, 0.08, 0.165, "• groupby(state) for all transforms — no cross-state leakage",
          size=16, color="#FCA5A5")
    _text(ax, 0.08, 0.14, "• StateScaler fit on train only — no future distribution info",
          size=16, color="#FCA5A5")

    return _save(fig, "03_features")


def slide_models():
    fig = _fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _header(ax, "Four Model Families")

    models = [
        ("SARIMA", "Statistical", ["auto_arima, m=52", "Per-state fitting", "Stepwise search"], "#8B5CF6"),
        ("Prophet", "Facebook", ["Multiplicative mode", "US holidays", "Per-state fitting"], "#EC4899"),
        ("XGBoost", "Gradient Boosting", ["Global model (6,106 rows)", "500 trees, depth=6", "State label-encoded"], ACCENT),
        ("LSTM", "Deep Learning", ["PyTorch, 2-layer, h=64", "Global model, seq=12", "AdamW + cosine LR"], ACCENT2),
    ]

    for i, (name, mtype, details, color) in enumerate(models):
        cx = 0.06 + i * 0.235
        _card(ax, cx, 0.38, 0.22, 0.36)
        _badge(ax, cx + 0.01, 0.71, mtype, bg=color, size=12)
        _text(ax, cx + 0.01, 0.66, name, size=26, weight="bold")
        for j, d in enumerate(details):
            _text(ax, cx + 0.01, 0.58 - j * 0.05, f"• {d}", size=15, color="#CBD5E1")

    # Recursive prediction callout
    _card(ax, 0.06, 0.12, 0.88, 0.18, color="#1A2332")
    _text(ax, 0.08, 0.27, "Recursive Multi-Step Forecasting", size=20, weight="bold", color=ACCENT)
    _text(ax, 0.08, 0.22,
          "Predict week t+1 → append to history → rebuild features → predict t+2 → ... → repeat 8 times",
          size=16, color="#CBD5E1")
    _text(ax, 0.08, 0.17,
          "Every feature is recomputed from the sales history buffer. No future data used at any step.",
          size=15, color=MUTED)

    return _save(fig, "04_models")


def slide_results():
    fig = _fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _header(ax, "Results & Model Comparison")

    # Leaderboard bars
    models_data = [
        ("LSTM", 5.08, ACCENT2, "27/43 wins"),
        ("XGBoost", 6.27, ACCENT, "15/43 wins"),
        ("SARIMA", 12.46, "#8B5CF6", "1/43 wins"),
        ("Prophet", 18.52, "#EC4899", "0/43 wins"),
    ]
    bar_top = 0.72
    max_smape = 20
    for i, (name, smape, color, wins) in enumerate(models_data):
        y = bar_top - i * 0.075
        bar_w = (smape / max_smape) * 0.45
        _text(ax, 0.06, y + 0.01, name, size=18, weight="bold", va="center")
        rect = mpatches.FancyBboxPatch(
            (0.20, y - 0.015), bar_w, 0.04, boxstyle="round,pad=0.005",
            facecolor=color, transform=ax.transAxes)
        ax.add_patch(rect)
        _text(ax, 0.21 + bar_w, y + 0.01, f"{smape}%", size=16, va="center", weight="bold")
        _text(ax, 0.76, y + 0.01, wins, size=15, va="center", color=MUTED)

    # Baseline comparison
    _card(ax, 0.06, 0.32, 0.42, 0.13)
    _text(ax, 0.08, 0.43, "Baselines", size=18, weight="bold", color=ACCENT)
    _text(ax, 0.08, 0.39, "Naive (repeat last):     5.58%", size=15, color="#CBD5E1")
    _text(ax, 0.08, 0.355, "Seasonal naive (52w):  12.38%", size=15, color="#CBD5E1")

    # Ensemble
    _card(ax, 0.52, 0.32, 0.42, 0.13)
    _text(ax, 0.54, 0.43, "Ensemble Validation", size=18, weight="bold", color=ACCENT2)
    _text(ax, 0.54, 0.39, "26/43 states use ensemble blend", size=15, color="#CBD5E1")
    _text(ax, 0.54, 0.355, "Improved 19/26 (avg −1.39pp sMAPE)", size=15, color="#CBD5E1")

    # Feature importance
    _card(ax, 0.06, 0.05, 0.88, 0.22)
    _text(ax, 0.08, 0.24, "XGBoost Feature Importance — rolling means explain ~90% of signal",
          size=17, weight="bold", color=ACCENT)
    feats = [("roll_mean_13", 46.2), ("roll_mean_8", 30.8), ("roll_mean_4", 12.4),
             ("lag_1", 8.8), ("others", 1.8)]
    for i, (feat, imp) in enumerate(feats):
        y = 0.18 - i * 0.028
        bar_w = (imp / 50) * 0.55
        rect = mpatches.FancyBboxPatch(
            (0.28, y - 0.008), bar_w, 0.02, boxstyle="round,pad=0.003",
            facecolor=ACCENT if i < 4 else MUTED, alpha=0.7, transform=ax.transAxes)
        ax.add_patch(rect)
        _text(ax, 0.08, y + 0.003, feat, size=13, color="#CBD5E1", va="center")
        _text(ax, 0.29 + bar_w, y + 0.003, f"{imp}%", size=12, va="center", color=MUTED)

    return _save(fig, "05_results")


def slide_backtest():
    fig = _fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _header(ax, "Walk-Forward Backtest", "3-fold expanding window validates model stability")

    # Fold visualization
    fold_data = [
        ("Fold 1", "Train → Jun 2022", "Val: Jun–Sep 2022"),
        ("Fold 2", "Train → Sep 2022", "Val: Sep–Dec 2022"),
        ("Fold 3", "Train → Dec 2022", "Val: Dec–Mar 2023"),
    ]
    for i, (fold, train, val) in enumerate(fold_data):
        y = 0.68 - i * 0.1
        _card(ax, 0.06, y - 0.02, 0.88, 0.08)
        _text(ax, 0.08, y + 0.025, fold, size=18, weight="bold", color=ACCENT)
        train_w = 0.35 + i * 0.08
        rect = mpatches.FancyBboxPatch(
            (0.22, y + 0.005), train_w, 0.025, boxstyle="round,pad=0.003",
            facecolor="#334155", transform=ax.transAxes)
        ax.add_patch(rect)
        val_rect = mpatches.FancyBboxPatch(
            (0.22 + train_w + 0.01, y + 0.005), 0.12, 0.025, boxstyle="round,pad=0.003",
            facecolor=ACCENT, alpha=0.5, transform=ax.transAxes)
        ax.add_patch(val_rect)
        _text(ax, 0.25, y + 0.018, "Training", size=11, color=MUTED, va="center")
        _text(ax, 0.22 + train_w + 0.04, y + 0.018, "Validation", size=11, va="center")

    # Results table
    _card(ax, 0.06, 0.15, 0.88, 0.2)
    _text(ax, 0.08, 0.33, "Backtest Stability", size=22, weight="bold", color=ACCENT2)
    _text(ax, 0.15, 0.28, "Model", size=16, weight="bold", color=MUTED)
    _text(ax, 0.40, 0.28, "Median sMAPE", size=16, weight="bold", color=MUTED)
    _text(ax, 0.65, 0.28, "Std Dev", size=16, weight="bold", color=MUTED)

    _text(ax, 0.15, 0.24, "XGBoost", size=18, color=ACCENT)
    _text(ax, 0.40, 0.24, "4.79%", size=18, weight="bold")
    _text(ax, 0.65, 0.24, "2.88", size=18, color=ACCENT2)

    _text(ax, 0.15, 0.20, "LSTM", size=18, color=ACCENT2)
    _text(ax, 0.40, 0.20, "5.76%", size=18, weight="bold")
    _text(ax, 0.65, 0.20, "4.57", size=18, color="#F59E0B")

    return _save(fig, "06_backtest")


def slide_api():
    fig = _fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _header(ax, "REST API & Testing", "Pre-computed predictions, sub-millisecond responses")

    # Endpoints table
    endpoints = [
        ("GET", "/api/v1/health", "Health check"),
        ("GET", "/api/v1/states", "List 43 states"),
        ("GET", "/api/v1/forecast/{state}", "8-week forecast"),
        ("GET", "/api/v1/model-info/{state}", "Model selection details"),
        ("GET", "/api/v1/metrics/summary", "Global leaderboard"),
    ]
    _text(ax, 0.06, 0.73, "Method", size=15, weight="bold", color=MUTED)
    _text(ax, 0.16, 0.73, "Endpoint", size=15, weight="bold", color=MUTED)
    _text(ax, 0.55, 0.73, "Description", size=15, weight="bold", color=MUTED)
    for i, (method, path, desc) in enumerate(endpoints):
        y = 0.68 - i * 0.045
        _badge(ax, 0.06, y + 0.01, method, bg=ACCENT2, size=11)
        _text(ax, 0.16, y + 0.01, path, size=15, color=TEXT_COLOR, va="center",
              family="monospace")
        _text(ax, 0.55, y + 0.01, desc, size=15, color="#CBD5E1", va="center")

    # JSON response example
    _card(ax, 0.06, 0.15, 0.55, 0.28, color="#0D1117")
    json_lines = [
        '{',
        '  "state": "California",',
        '  "horizon_weeks": 8,',
        '  "forecasts": [',
        '    {"date": "2023-04-01",',
        '     "forecast": 841571584.0,',
        '     "model_used": "ensemble(...)"}',
        '  ]',
        '}',
    ]
    for i, line in enumerate(json_lines):
        _text(ax, 0.08, 0.40 - i * 0.028, line, size=13, color=ACCENT2,
              family="monospace")

    # Test stats
    _card(ax, 0.64, 0.15, 0.30, 0.28)
    _text(ax, 0.66, 0.40, "Test Suite", size=20, weight="bold", color=ACCENT)
    _text(ax, 0.79, 0.33, "57", size=48, weight="bold", color=ACCENT2, ha="center")
    _text(ax, 0.79, 0.26, "tests passing", size=16, color=MUTED, ha="center")
    test_areas = ["Features & leakage", "Metrics & bounds", "Ingestion & dates",
                  "All 4 models", "API endpoints", "Validation guards"]
    for i, area in enumerate(test_areas):
        _text(ax, 0.66, 0.22 - i * 0.025, f"✓ {area}", size=12, color="#CBD5E1")

    return _save(fig, "07_api")


def slide_closing():
    fig = _fig()
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    _text(ax, 0.5, 0.72, "End-to-End System", size=48, weight="bold",
          ha="center", va="center")
    _text(ax, 0.5, 0.63, "QuickHyre Data Science Assignment",
          size=24, color=ACCENT, ha="center", va="center")

    highlights = [
        "Messy data → clean weekly panel (37% imputed, documented)",
        "23 leakage-free features with shift(1) + per-state groupby",
        "4 model families: SARIMA, Prophet, XGBoost, LSTM",
        "Per-state best-model selection with validated ensemble",
        "Walk-forward backtest confirms stability across time",
        "REST API with 6 endpoints, pre-computed predictions",
        "57 tests, HTML report, full reproducibility",
    ]
    for i, h in enumerate(highlights):
        y = 0.52 - i * 0.048
        _text(ax, 0.15, y, f"→  {h}", size=18, color="#CBD5E1")

    _text(ax, 0.5, 0.12, "Thank you for watching",
          size=28, color=MUTED, ha="center", va="center")

    return _save(fig, "08_closing")


# ---------------------------------------------------------------------------
# Audio generation
# ---------------------------------------------------------------------------

async def generate_audio():
    """Generate TTS audio for each scene using edge-tts."""
    import edge_tts

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    durations = {}

    for scene in SCENES:
        out_path = AUDIO_DIR / f"{scene['id']}.mp3"
        if out_path.exists():
            continue

        communicate = edge_tts.Communicate(scene["narration"], VOICE, rate="-5%")
        await communicate.save(str(out_path))
        print(f"  Audio: {scene['id']}.mp3")

    return durations


# ---------------------------------------------------------------------------
# Video compositing
# ---------------------------------------------------------------------------

def compose_video():
    """Assemble slides + audio into final video using moviepy."""
    from moviepy import (
        ImageClip, AudioFileClip, CompositeVideoClip,
        TextClip, concatenate_videoclips,
    )

    slide_generators = {
        "title": slide_title,
        "dataset": slide_dataset,
        "features": slide_features,
        "models": slide_models,
        "results": slide_results,
        "backtest": slide_backtest,
        "api": slide_api,
        "closing": slide_closing,
    }

    clips = []
    srt_entries = []
    cumulative_time = 0.0

    for scene in SCENES:
        sid = scene["id"]

        # Generate slide
        slide_path = slide_generators[sid]()
        print(f"  Slide: {sid}")

        # Load audio
        audio_path = AUDIO_DIR / f"{sid}.mp3"
        audio_clip = AudioFileClip(str(audio_path))
        duration = audio_clip.duration + 1.5  # 1.5s padding after speech

        # Create image clip with audio
        img_clip = (
            ImageClip(str(slide_path))
            .with_duration(duration)
            .with_audio(audio_clip)
            .resized((WIDTH, HEIGHT))
        )

        clips.append(img_clip)

        # SRT subtitle entry
        start_ts = _srt_timestamp(cumulative_time)
        end_ts = _srt_timestamp(cumulative_time + audio_clip.duration)
        srt_entries.append(
            f"{len(srt_entries) + 1}\n{start_ts} --> {end_ts}\n{scene['narration']}\n"
        )
        cumulative_time += duration

    # Concatenate with crossfade
    final = concatenate_videoclips(clips, method="compose")

    # Export
    output_path = DEMO_DIR / "Sales_Forecasting_Demo.mp4"
    print(f"\nRendering video ({cumulative_time:.0f}s) ...")
    final.write_videofile(
        str(output_path),
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=4,
        logger="bar",
    )
    print(f"Video saved: {output_path}")

    # Write SRT
    srt_path = DEMO_DIR / "subtitles.srt"
    srt_path.write_text("\n".join(srt_entries), encoding="utf-8")
    print(f"Subtitles saved: {srt_path}")

    # Generate thumbnail (just copy title slide)
    thumb_src = SLIDES_DIR / "01_title.png"
    thumb_dst = DEMO_DIR / "thumbnail.png"
    if thumb_src.exists():
        import shutil
        shutil.copy2(thumb_src, thumb_dst)
        print(f"Thumbnail saved: {thumb_dst}")

    return output_path


def _srt_timestamp(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SLIDES_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Generating TTS Audio ===")
    asyncio.run(generate_audio())

    print("\n=== Generating Slides & Compositing Video ===")
    compose_video()

    # Write narration script
    script_path = DEMO_DIR / "narration_script.md"
    lines = ["# Narration Script\n"]
    for scene in SCENES:
        lines.append(f"## Scene: {scene['id'].title()}\n")
        lines.append(f"{scene['narration']}\n")
    script_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Script saved: {script_path}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
