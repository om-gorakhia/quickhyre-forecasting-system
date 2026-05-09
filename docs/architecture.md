# Architecture & Design Decisions

Key engineering decisions in the forecasting system, the reasoning behind them, and trade-offs considered.

---

## 1. Global vs. Per-State Models

**Decision**: XGBoost and LSTM are trained as single global models pooling all 43 states. SARIMA and Prophet are trained per-state.

**Reasoning**: After temporal splitting and feature warm-up (dropping NaN lag rows), each state has approximately 142 usable training rows. For 23+ features, this creates a high-dimensional, low-sample regime where overfitting is likely. Pooling all states yields 6,106 training rows — a 43x increase. State identity is preserved via a label-encoded `state_encoded` feature (XGBoost) and per-state MinMax scaling (LSTM).

SARIMA and Prophet are inherently univariate and per-series, so global training is not applicable.

**Trade-off**: global models assume some transferable structure across states. States with highly atypical demand patterns may be underserved. If more data were available (3+ years per state), per-state fine-tuning from a globally pre-trained model would be the natural next step.

---

## 2. Recursive Multi-Step Forecasting

**Decision**: XGBoost and LSTM generate 8-week forecasts recursively — each predicted value is fed back as input for the next step.

**Reasoning**: direct multi-output models would require 8 separate output heads or a fundamentally different architecture. Recursive forecasting uses the same single-step model for all horizons and naturally incorporates the lag/rolling feature structure. At each step, the `_build_forecast_row()` method reconstructs all features from scratch using a rolling sales history buffer.

**Trade-off**: errors compound — each step's prediction error becomes part of the next step's input. This is a known limitation of recursive forecasting. In practice, the effect is moderate over an 8-week horizon: the dominant features (rolling means) average over multiple weeks and are resistant to single-step noise.

---

## 3. sMAPE as Primary Metric

**Decision**: model selection and ranking use Symmetric Mean Absolute Percentage Error (sMAPE).

**Reasoning**: comparing forecast quality across states requires a scale-invariant metric. California's sales (~$800M/week) and Wyoming's (~$8M/week) differ by 100x — raw MAE would always rank Wyoming as "easy" regardless of forecast quality. sMAPE is bounded [0, 200], symmetric (penalizes over- and under-prediction equally), and handles near-zero values better than MAPE.

MASE is used as a tiebreaker because it benchmarks against the naive forecast, answering "is this model actually useful, or would repeating last week's value work just as well?"

---

## 4. Leakage Prevention

Three layers of leakage prevention:

1. **Feature engineering**: all rolling statistics use `shift(1)` before computing the window. The current row's target never appears in its own features. All transforms are per-state via `groupby` — no information flows between states.

2. **Temporal split**: train/val/test boundaries are strictly date-based. No shuffling, no random sampling. Every state uses the same cut dates.

3. **Scaler fitting**: the `StateScaler` is fit exclusively on training data. Validation and test data are transformed using training-derived statistics. This prevents future distribution information from leaking into scaled features.

---

## 5. Ensemble Strategy

**Decision**: simple inverse-sMAPE weighted average of the top-2 models, with a 50% threshold gate.

**Reasoning**: ensembling reduces variance from recursive forecasting noise. The inverse-sMAPE weighting gives more influence to the better model while still incorporating the second model's signal. The 50% threshold (`ENSEMBLE_THRESHOLD = 1.5`) ensures that a distant second model (e.g., Prophet at 18% vs. LSTM at 5%) doesn't dilute the winner.

No stacking or meta-learner is used. With only 4 models and ~142 validation rows per state, a meta-learner would overfit. The weighted average is transparent and easy to explain.

**Result**: 26 of 43 states (60%) qualify for ensemble blending.

---

## 6. API Architecture

**Decision**: pre-compute all forecasts at training time and serve from memory. No model inference at request time.

**Reasoning**:

- **Determinism**: every API call for the same state returns the same forecast. No randomness from model loading or initialization.
- **Speed**: responses are O(1) lookups into a pandas DataFrame. Latency is sub-millisecond.
- **Simplicity**: the API has no dependency on PyTorch, XGBoost, or any model library. It reads CSVs.
- **Reliability**: no GPU/CPU spikes, no OOM risk, no model loading failures at request time.

**Trade-off**: forecasts go stale until the pipeline is re-run. In production, this would be addressed by scheduling retraining (daily/weekly cron), but for a demo system, pre-computation is the right call.

---

## 7. Walk-Forward Backtest

**Decision**: 3-fold expanding-window backtest with 13-week horizon per fold, testing only XGBoost and LSTM.

**Reasoning**: a single train/val split can be misleading. A model might score well on one particular time window due to favorable conditions (e.g., stable demand during its validation period). The backtest checks whether performance holds across three different time windows.

Only the top-2 models (XGBoost, LSTM) are backtested. SARIMA takes ~25 minutes per full evaluation pass, and Prophet already lost the single-split comparison decisively. Spending 1+ hour on repeated SARIMA retraining would not change the selection outcome.

**Finding**: XGBoost is more stable across folds (std 2.88 vs. 4.57), confirming it as the more reliable baseline. LSTM's single-split advantage (5.08% vs. 6.27%) partially reflects a favorable validation window — its backtest median is 5.76%, closer to XGBoost's 4.79%. This validates the ensemble approach: LSTM adds value on specific states and time periods, but XGBoost provides the consistent foundation.

---

## 8. Validation Guards

**Decision**: add explicit data validation at pipeline boundaries that fails fast with clear error messages.

**Reasoning**: the full pipeline takes ~30 minutes. Discovering at minute 28 that the input file had missing columns wastes time. Validation checks run in milliseconds and catch:

- Missing/renamed columns in the raw Excel
- Empty dataframes after filtering
- Null targets surviving imputation
- Non-weekly gaps after resampling
- Train/val temporal overlap
- Missing artifacts before API startup

Each check raises a `ValidationError` with a message explaining what went wrong and how to fix it — not a cryptic `KeyError` deep in a model training loop.
