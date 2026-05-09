# Assumptions

Assumptions made during the design and implementation of this forecasting system, documented for reviewer transparency.

---

## Data Assumptions

1. **Single category**: the `category` column is always "Beverages". The pipeline drops it as a constant. If future data includes multiple categories, the pipeline would need a category dimension.

2. **Weekly granularity is appropriate**: the raw data's irregular timestamps are snapped to a weekly grid (W-SAT). This assumes that weekly is the right granularity for demand planning. Sub-weekly patterns are lost.

3. **Linear interpolation is acceptable for imputed weeks**: 37.4% of the weekly grid was missing. Linear interpolation assumes smooth transitions between observed points. This may understate volatility during gap periods (especially the sparse 2019 data).

4. **No structural breaks**: the pipeline treats the full time range as one regime. If there was a fundamental shift in demand patterns (e.g., COVID impact in 2020), the models absorb it as training data rather than adapting to it explicitly.

5. **State boundaries are stable**: all 43 states are present throughout the dataset. No states appear or disappear mid-series.

6. **Sales values are in consistent units**: the `total` column represents the same metric across all states and dates (assumed: dollar revenue). No currency conversion or unit normalization is needed.

---

## Modeling Assumptions

7. **Lag naming convention**: the assignment specifies features at t-1, t-7, t-30. On weekly data, t-7 means "7 weeks ago" (not "7 days ago"). This is the correct interpretation for weekly-resampled data.

8. **day_of_week is a no-op feature**: after resampling to W-SAT, every row lands on Saturday (dayofweek=5). The feature is constant and carries zero predictive signal. It is included for assignment compliance and documented as such.

9. **Global models generalize across states**: XGBoost and LSTM are trained on pooled data from all 43 states. This assumes there is transferable structure (seasonality patterns, trend behavior) across states. The state-encoded feature and per-state scaling preserve identity.

10. **52-week seasonality is dominant**: SARIMA uses m=52. This assumes annual seasonality is the primary cyclical pattern. If there were meaningful sub-annual cycles (e.g., monthly promotions), they would need explicit handling.

11. **Recursive forecasting is acceptable for 8 steps**: error compounding over 8 recursive steps is assumed to be manageable. For longer horizons (26+ weeks), direct multi-output or encoder-decoder architectures would be more appropriate.

---

## Evaluation Assumptions

12. **sMAPE is the right primary metric**: this assumes that scale-invariant, symmetric error measurement is more important than absolute dollar accuracy. For states where dollar accuracy matters most, MAE would be a better primary metric.

13. **Validation window is representative**: the val period (Oct 2022 - Mar 2023) is assumed to be representative of future demand patterns. If this period was atypical (e.g., post-COVID recovery effects), model rankings might not generalize.

14. **MASE uses seasonal period=52**: MASE compares model errors against a seasonal naive baseline (same week last year). For series shorter than 52 observations, it falls back to period=1.

---

## System Assumptions

15. **Forecasts are regenerated offline**: the API serves pre-computed predictions. This assumes forecasts are regenerated periodically (e.g., weekly after new data arrives) via re-running the pipeline.

16. **No concurrent writes**: the pipeline is single-user. Multiple simultaneous pipeline runs writing to the same artifacts directory could corrupt outputs.

17. **Python 3.11+ environment**: the codebase uses `X | Y` union types and `list[str]` generics that require Python 3.11+.
