# Beverage Sales Forecasting System

**QuickHyre Assignment — End-to-End Time Series Forecasting System with API**

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Business Objective](#business-objective)
3. [Dataset Overview](#dataset-overview)
4. [Project Architecture](#project-architecture)
5. [Repository Structure](#repository-structure)
6. [Preprocessing](#preprocessing)
7. [Feature Engineering](#feature-engineering)
8. [Models Implemented](#models-implemented)
9. [Evaluation Methodology](#evaluation-methodology)
10. [Best-Model Selection](#best-model-selection)
11. [API Endpoints](#api-endpoints)
12. [How to Run Locally](#how-to-run-locally)
13. [Sample Outputs](#sample-outputs)
14. [Limitations](#limitations)
15. [Future Improvements](#future-improvements)
16. [Video Demo](#video-demo)
17. [Submission Links](#submission-links)

---

## Problem Statement

Given weekly beverage sales data across 43 US states, build a forecasting system that:
- Preprocesses messy real-world data (mixed date formats, irregular spacing, missing values)
- Engineers meaningful time series features with documented leakage prevention
- Trains multiple forecasting models from different paradigms
- Selects the best model per state based on validation performance
- Generates 8-week-ahead forecasts
- Serves predictions through a REST API

---

## Business Objective

Provide demand planning teams with accurate 8-week-ahead sales forecasts per US state. Accurate short-term forecasts support:

- **Inventory allocation** — position stock where it will sell, reduce carrying costs
- **Production scheduling** — align manufacturing output with anticipated demand
- **Regional strategy** — identify states with diverging demand patterns

The system selects the best forecasting approach per state rather than forcing a single model, because demand patterns differ across states.

---

## Dataset Overview

**Source**: `Forecasting Case- Study.xlsx`

| Property | Value |
|----------|-------|
| Raw rows | 8,084 |
| States | 43 |
| Category | Beverages (single, constant — dropped) |
| Date range | 2019-01-12 to 2023-03-25 |
| Frequency | Weekly (W-SAT after resampling) |
| Cleaned rows | 11,051 (43 states × 257 weeks) |
| Imputation rate | 37.4% of the weekly grid |

The raw data has irregularly spaced observations with a sparse period in 2019. After resampling to a regular weekly grid, 37.4% of cells required imputation via linear interpolation. The source file also mixes `YYYY-MM-DD` and `DD-MM-YYYY` date formats within the same column.

---

## Project Architecture

```
Forecasting Case- Study.xlsx
        │
        ▼
  ┌───────────┐     ┌──────────┐     ┌─────────┐     ┌──────────────────┐
  │  Ingest   │ ──▶ │ Features │ ──▶ │  Split  │ ──▶ │ Train (4 models) │
  │  parse,   │     │ 23 cols  │     │ temporal│     │ SARIMA, Prophet  │
  │  resample,│     │ lag/roll │     │         │     │ XGBoost, LSTM    │
  │  impute   │     │ cyclical │     │         │     │                  │
  └───────────┘     └──────────┘     └─────────┘     └──────────────────┘
                                                            │
                                                            ▼
                                    ┌──────────┐     ┌───────────┐
                                    │ Backtest │ ◀── │  Select   │
                                    │ 3-fold   │     │ per-state │
                                    │ walk-fwd │     │ + ensemble│
                                    └──────────┘     └───────────┘
                                                            │
                                                            ▼
                                                    ┌───────────┐
                                                    │ Forecast  │
                                                    │ 8-week    │
                                                    │ recursive │
                                                    └───────────┘
                                                            │
                                                            ▼
                                                      ┌─────────┐
                                                      │  API    │
                                                      │ FastAPI │
                                                      │ /api/v1 │
                                                      └─────────┘
```

**Data flow**: Raw Excel → cleaned Parquet → feature-enriched Parquet → temporal splits → trained model artifacts + per-state metrics → model selection table → pre-computed forecast CSVs → API (serves from memory).

---

## Repository Structure

```
config/
  settings.py              Central configuration (paths, columns, hyperparams)

src/
  ingest.py                Load Excel, parse mixed dates, resample to W-SAT, impute
  features.py              23-column feature engineering pipeline
  split.py                 Temporal train/val/test split
  dataset.py               Model-specific data prep (XGBoost arrays, LSTM sequences, StateScaler)
  validate.py              Data validation guards — fail fast with clear messages
  evaluate.py              Metrics: MAE, RMSE, MAPE, sMAPE, MASE
  baseline.py              Naive and seasonal-naive baselines for benchmarking
  selector.py              Best-model selection per state + ensemble logic
  forecast.py              8-week recursive forecast generation
  backtest.py              Walk-forward backtesting (expanding window)
  plots.py                 Summary visualizations
  report.py                Self-contained HTML report generator
  train_classical.py       Per-state SARIMA/Prophet training loop
  models/
    base.py                Abstract ForecastModel interface
    sarima.py              SARIMA via pmdarima auto_arima (m=52)
    prophet_model.py       Prophet with US holidays, multiplicative seasonality
    xgboost_model.py       Global XGBoost with recursive multi-step prediction
    lstm_model.py          Global PyTorch LSTM with recursive prediction

scripts/
  train_all.py             Unified orchestrator (preprocess → train → select → backtest → report)
  preprocess.py            Standalone preprocessing
  train_classical.py       CLI for SARIMA/Prophet training
  train_xgboost.py         CLI for XGBoost training
  train_lstm.py            CLI for LSTM training
  serve.py                 FastAPI server launcher

api/
  app.py                   FastAPI app with lifespan startup, middleware, exception handling
  routes.py                6 API endpoints
  schemas.py               Pydantic response models
  data.py                  In-memory ForecastStore (loads CSVs at startup)

tests/                     57 tests covering features, metrics, ingestion, models, API, validation
  conftest.py              Shared fixtures
  test_features.py         Feature engineering: leakage, correctness, completeness
  test_evaluate.py         Metric functions: edge cases, symmetry, bounds
  test_ingest.py           Date parsing, resampling, imputation
  test_api.py              All 6 API endpoints with mock store
  test_pipeline.py         End-to-end: features → split → dataset prep → scaler roundtrip
  test_validate.py         Validation guards
  test_models.py           Model smoke tests: fit/predict/save/load for XGBoost and LSTM

docs/
  assumptions.md           17 documented assumptions across data, modeling, evaluation
  architecture.md          Key engineering decisions with reasoning and trade-offs

artifacts/                 Generated by pipeline (gitignored, reproducible via `make train`)
  models/                  Saved model artifacts (.pkl, .pt)
  metrics/                 Per-model and aggregate metrics CSVs
  forecasts/               8-week forecast CSVs (per-state and combined)
  plots/                   Summary visualizations (PNG)
  report.html              Self-contained HTML summary report
```

---

## Preprocessing

Pipeline: `src/ingest.py`

1. **Load**: read Excel, normalize column headers
2. **Parse dates**: handle mixed `YYYY-MM-DD` / `DD-MM-YYYY` formats with `pd.to_datetime(format="mixed", dayfirst=True)`
3. **Validate**: drop constant `category` column, remove duplicate `(state, date)` rows, check for negative/null sales
4. **Resample**: snap all dates to a `W-SAT` weekly grid, creating a complete time axis per state
5. **Impute**: linear interpolation for interior gaps, forward/back-fill for edges (37.4% imputation rate)
6. **Post-validate**: confirm zero nulls remain, verify weekly regularity

Output: `data/processed/cleaned_weekly.parquet` (11,051 rows).

---

## Feature Engineering

Pipeline: `src/features.py` — 23 features, all computed per-state via `groupby` to prevent cross-state leakage.

### Calendar (11 features)
`month`, `week_of_year`, `year`, `quarter`, `day_of_week`, `month_sin`, `month_cos`, `week_of_year_sin`, `week_of_year_cos`, `is_holiday_week`, `week_index`

- Cyclical sin/cos encodings so Dec→Jan is smooth, not a cliff from 12→1
- `day_of_week` is constant (Saturday=5) on weekly data — included for assignment compliance, documented as no-signal
- `is_holiday_week` flags weeks containing a US federal holiday (vectorized via set membership)

### Lags (4 features)
`lag_1` (1 week), `lag_7` (7 weeks), `lag_30` (30 weeks), `lag_52` (52 weeks / 1 year)

The assignment specifies t-1, t-7, t-30. On weekly data, these are 1/7/30-week lookbacks. `lag_52` is added for yearly seasonality.

### Rolling Statistics (6 features)
`roll_mean_4`, `roll_std_4`, `roll_mean_8`, `roll_std_8`, `roll_mean_13`, `roll_std_13`

**Leakage prevention**: computed on `shift(1)` data — the current row's target never participates in its own rolling window.

### Derived (2 features)
`pct_change_1` (week-over-week change, shifted), `state_expanding_mean` (per-state expanding mean, shifted)

---

## Models Implemented

### SARIMA (`src/models/sarima.py`)
- Per-state `auto_arima` via pmdarima, `m=52` (yearly cycle)
- Stepwise search with `max_P=1, max_Q=1`
- One `.pkl` artifact per state

### Prophet (`src/models/prophet_model.py`)
- Per-state, `yearly_seasonality=True`, `weekly_seasonality=False`, `seasonality_mode='multiplicative'`
- US holidays added via Prophet's built-in support

### XGBoost (`src/models/xgboost_model.py`)
- **Global model** pooling all 43 states (6,106 training rows)
- Why global: 142 rows per state is insufficient for 23+ features
- State identity via label-encoded `state_encoded` feature
- 500 trees, `max_depth=6`, `lr=0.05`, `early_stopping_rounds=30`
- Recursive multi-step prediction: rebuilds all features from a rolling sales buffer at each step
- Top features by importance: `roll_mean_13` (46.2%), `roll_mean_8` (30.8%), `roll_mean_4` (12.4%)

### LSTM (`src/models/lstm_model.py`)
- **Global model**, PyTorch, 2-layer LSTM with `hidden_size=64`, `dropout=0.2`
- Input: 12-step sliding windows × 18 features (excludes raw calendar integers — sin/cos captures the same info)
- Per-state MinMax target scaling, global MinMax feature scaling (fit on train only)
- AdamW + CosineAnnealingLR, gradient clipping at 1.0, early stopping (`patience=15`)
- Recursive multi-step prediction matching XGBoost's approach

### Results

| Model | Median sMAPE | States Won |
|-------|-------------|-----------|
| LSTM | 5.08% | 27 / 43 |
| XGBoost | 6.27% | 15 / 43 |
| SARIMA | 12.46% | 1 / 43 |
| Prophet | 18.52% | 0 / 43 |

For context: naive baseline (repeat last value) scores **5.58%** median sMAPE, seasonal naive (same week last year) scores **12.38%**. LSTM beats both; XGBoost beats seasonal naive but narrowly loses to naive on sMAPE.

26 of 43 states use an ensemble blend of the top-2 models (inverse-sMAPE weighted). Ensemble improved 19/26 eligible states with an average 1.39 percentage point sMAPE reduction.

---

## Evaluation Methodology

### Metrics (`src/evaluate.py`)

| Metric | Role |
|--------|------|
| **sMAPE** | Primary ranking metric — scale-invariant, symmetric, bounded [0, 200] |
| **MASE** | Tiebreaker — compares against seasonal naive baseline (period=52) |
| **MAE** | Absolute error in original units |
| **RMSE** | Penalizes large errors |
| **MAPE** | Percentage error |

**Why sMAPE**: California ($800M/week) and Wyoming ($8M/week) differ by 100×. Raw MAE would always rank Wyoming as "easy" regardless of forecast quality.

### Temporal Split

| Split | Date Range | Purpose |
|-------|-----------|---------|
| Train | Start → 2022-09-30 | Model fitting |
| Validation | 2022-10-01 → 2023-03-31 | Model selection |
| Test | 2023-04-01 → end | Held out (not used for any decisions) |

### Walk-Forward Backtest (`src/backtest.py`)

3-fold expanding window, 13-week horizon per fold. Validates that model advantages hold across multiple time periods, not just one lucky split. Only XGBoost and LSTM are backtested (top-2 models).

| Model | Median sMAPE | Std |
|-------|-------------|-----|
| XGBoost | 4.79% | 2.88 |
| LSTM | 5.76% | 4.57 |

XGBoost is more stable across folds. LSTM's single-split advantage partly reflects a favorable validation window — its backtest median is closer to XGBoost's. This validates the ensemble approach.

---

## Best-Model Selection

Pipeline: `src/selector.py`

1. For each state, pick the model with the lowest sMAPE on validation. Tiebreak by MASE.
2. If the 2nd-best model is within 50% of the best (`ratio ≤ 1.5`), mark as ensemble-eligible.
3. Ensemble blending uses inverse-sMAPE weights: `w = (1/sMAPE) / sum(1/sMAPE)`.

**Result**: 26/43 states use ensemble blending. 17 have a clear single winner.

---

## API Endpoints

The API serves pre-computed forecasts from memory. No model inference at request time — sub-millisecond responses.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Health check with status and version |
| `GET` | `/api/v1/states` | List all 43 available states |
| `GET` | `/api/v1/forecast/{state}` | 8-week forecast for a specific state |
| `GET` | `/api/v1/forecast?state=Texas` | Query parameter alternative |
| `GET` | `/api/v1/model-info/{state}` | Model selection details for a state |
| `GET` | `/api/v1/metrics/summary` | Global model leaderboard |

State names are case-insensitive (`california`, `CALIFORNIA`, `California` all work).

### Example Requests

```bash
# Start the server
python scripts/serve.py

# Health check
curl http://localhost:8000/api/v1/health

# List available states
curl http://localhost:8000/api/v1/states

# Get 8-week forecast for California
curl http://localhost:8000/api/v1/forecast/California

# Get model selection details for Texas
curl http://localhost:8000/api/v1/model-info/Texas

# Global model leaderboard
curl http://localhost:8000/api/v1/metrics/summary
```

### Example Response: `/api/v1/forecast/California`

```json
{
  "state": "California",
  "horizon_weeks": 8,
  "forecasts": [
    {"date": "2023-04-01", "forecast": 841571584.0, "model_used": "ensemble(lstm:0.57+xgboost:0.43)"},
    {"date": "2023-04-08", "forecast": 826849152.0, "model_used": "ensemble(lstm:0.57+xgboost:0.43)"},
    ...
  ],
  "generated_at": "2026-05-10T..."
}
```

Interactive Swagger docs available at `http://localhost:8000/docs`.

---

## How to Run Locally

### Prerequisites

- Python 3.11+
- ~4 GB disk space for virtual environment and artifacts

### Setup

```bash
# Clone the repository
git clone https://github.com/om-gorakhia/quickhyre-forecasting-system.git
cd quickhyre-forecasting-system

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

# Install dependencies
pip install -e ".[dev]"
```

### Train Models

```bash
# Full pipeline (~30 min): all 4 models + baselines + selection + backtest + report
python scripts/train_all.py

# Fast mode (~5 min): skip SARIMA/Prophet
python scripts/train_all.py --skip-classical

# Re-run selection + forecasting only (no retraining)
python scripts/train_all.py --select-only

# Force re-run everything from scratch
python scripts/train_all.py --force
```

### Start API

```bash
python scripts/serve.py                 # default: port 8000
python scripts/serve.py --port 8080     # custom port
python scripts/serve.py --reload        # auto-reload for development
```

### Run Tests

```bash
pytest                     # 57 tests
pytest -v                  # verbose
pytest tests/test_api.py   # single file
```

### Makefile Shortcuts

```bash
make setup       # pip install -e ".[dev]"
make train       # full pipeline
make train-fast  # skip classical models
make test        # run tests
make serve       # start API
make clean       # remove artifacts and processed data
```

---

## Sample Outputs

### Model Leaderboard

```
            median_smape  mean_smape  median_mae   median_mase
lstm               5.08        5.86   8,007,588         1.16
xgboost            6.27        6.58   7,529,504         1.41
sarima            12.46       11.63  20,470,518         3.16
prophet           18.52       18.49  33,350,220         4.89
```

### Baseline Comparison

| Baseline | Median sMAPE |
|----------|-------------|
| Naive (repeat last value) | 5.58% |
| Seasonal naive (same week last year) | 12.38% |

### Hardest States to Forecast

| State | Best Model | sMAPE |
|-------|-----------|-------|
| West Virginia | LSTM | 8.45% |
| Nebraska | XGBoost | 7.90% |
| South Carolina | LSTM | 6.96% |
| Mississippi | XGBoost | 6.95% |
| Arkansas | XGBoost | 6.92% |

### XGBoost Feature Importance (Top 5)

| Feature | Importance |
|---------|-----------|
| `roll_mean_13` | 46.2% |
| `roll_mean_8` | 30.8% |
| `roll_mean_4` | 12.4% |
| `lag_1` | 8.8% |
| `state_expanding_mean` | 0.9% |

Rolling means dominate — the signal is primarily in smoothed recent history.

### Generated Artifacts

After running the full pipeline, the `artifacts/` directory contains:
- **88 model files** (43 SARIMA + 43 Prophet + 1 XGBoost + 1 LSTM)
- **15 metrics CSVs** (per-model, aggregate, baselines, backtest, ensemble validation)
- **44 forecast CSVs** (per-state + combined)
- **8 plots** (leaderboard, state comparison, winner distribution, training curves, sample forecasts)
- **1 HTML report** (self-contained, embeds plots as base64)

---

## Limitations

1. **High imputation rate (37.4%)** — over a third of the weekly grid was filled via interpolation. Models partially learn from imputed data, not ground truth.

2. **Recursive forecast degradation** — multi-step forecasts feed predictions back as inputs. Errors compound; week 8 is less reliable than week 1.

3. **No exogenous variables** — the models use only endogenous features. External drivers (pricing, promotions, weather, economic indicators) are not available in the dataset.

4. **No confidence intervals** — forecasts are point estimates. Prediction intervals would require bootstrapping or quantile regression.

5. **Weekly granularity only** — the entire pipeline is built around weekly data. Sub-weekly or monthly frequencies would require re-engineering.

6. **Global model trade-off** — XGBoost and LSTM are trained on pooled data from all 43 states. States with highly idiosyncratic patterns may be underserved.

---

## Future Improvements

1. **Exogenous features** — integrate pricing, promotional calendars, economic indicators, or weather data.
2. **Probabilistic forecasting** — prediction intervals via quantile regression (XGBoost) or Monte Carlo dropout (LSTM).
3. **Temporal Fusion Transformer** — replace LSTM with TFT for better multi-horizon forecasting with built-in interpretability.
4. **Hyperparameter tuning** — systematic search via Optuna/Ray Tune instead of hand-tuned defaults.
5. **Online learning** — incremental model updates as new weekly data arrives.
6. **Monitoring** — prediction drift detection with automated retraining triggers.

---

## Video Demo

[![Watch the demo](https://github.com/om-gorakhia/quickhyre-forecasting-system/releases/download/v1.0.0/demo_thumbnail.png)](https://github.com/om-gorakhia/quickhyre-forecasting-system/releases/download/v1.0.0/QuickHyre_Forecasting_Demo.mp4)

[Download demo video (MP4)](https://github.com/om-gorakhia/quickhyre-forecasting-system/releases/download/v1.0.0/QuickHyre_Forecasting_Demo.mp4)

---

## Submission Links

| Item | Link |
|------|------|
| **Code Repository** | [github.com/om-gorakhia/quickhyre-forecasting-system](https://github.com/om-gorakhia/quickhyre-forecasting-system) |
| **Demo Video** | [QuickHyre Forecasting Demo (MP4)](https://github.com/om-gorakhia/quickhyre-forecasting-system/releases/download/v1.0.0/QuickHyre_Forecasting_Demo.mp4) |
| **Documentation** | [docs/](https://github.com/om-gorakhia/quickhyre-forecasting-system/tree/main/docs) |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Data | pandas, NumPy, openpyxl |
| Classical models | pmdarima (SARIMA), Prophet |
| ML model | XGBoost |
| Deep learning | PyTorch (LSTM) |
| API | FastAPI, uvicorn, Pydantic |
| Testing | pytest (57 tests) |
| Visualization | matplotlib |
