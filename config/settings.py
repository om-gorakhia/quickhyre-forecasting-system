from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

RAW_EXCEL = PROJECT_ROOT / "Forecasting Case- Study.xlsx"
CLEANED_PARQUET = PROCESSED_DATA_DIR / "cleaned_weekly.parquet"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"

# Data schema
DATE_COL = "date"
STATE_COL = "state"
TARGET_COL = "sales"
FREQ = "W-SAT"  # weekly anchor on Saturday (closest to dominant pattern)

# Temporal split boundaries
VAL_START = "2022-10-01"
TEST_START = "2023-04-01"

# Forecast horizon
FORECAST_WEEKS = 8

# --- Feature engineering config ---

# Lag features (in weekly periods).
# Assignment asks for t-1, t-7, t-30. On weekly data these mean
# 1 week, 7 weeks (~1.6 months), and 30 weeks (~7 months) back.
LAG_PERIODS = [1, 7, 30]

# Seasonal lag (yearly)
SEASONAL_LAG = 52

# Rolling window sizes (weeks)
ROLLING_WINDOWS = [4, 8, 13]

# LSTM sequence length (how many past weeks the network sees)
LSTM_SEQ_LEN = 12
