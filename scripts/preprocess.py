"""Run the full preprocessing pipeline: ingest -> features -> split -> dataset prep.

Usage:
    python scripts/preprocess.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import run_ingestion
from src.features import build_features, ALL_FEATURE_COLS
from src.split import temporal_split, save_splits
from src.dataset import build_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preprocess")


def main():
    logger.info("=== Step 1: Ingestion ===")
    df = run_ingestion()

    logger.info("=== Step 2: Feature Engineering ===")
    df = build_features(df)

    logger.info("=== Step 3: Train/Val/Test Split ===")
    splits = temporal_split(df)
    save_splits(splits)

    logger.info("=== Step 4: Dataset Preparation ===")
    datasets = build_datasets(splits)

    # Summary
    logger.info("=== Pipeline Complete ===")
    logger.info("Feature columns (%d): %s", len(ALL_FEATURE_COLS), ALL_FEATURE_COLS)
    for split_name in ["train", "val", "test"]:
        X, y, meta = datasets["xgboost"][split_name]
        logger.info("XGBoost %-5s: X=%s  y=%s", split_name, X.shape, y.shape)
    for split_name in ["train", "val", "test"]:
        X, y, meta = datasets["lstm"][split_name]
        logger.info("LSTM    %-5s: X=%s  y=%s", split_name, X.shape, y.shape)

    logger.info("Outputs:")
    logger.info("  data/processed/cleaned_weekly.parquet")
    logger.info("  data/processed/features.parquet")
    logger.info("  data/processed/splits/{train,val,test}.parquet")
    logger.info("  data/processed/missing_weeks_report.csv")
    logger.info("  artifacts/scaler.pkl")


if __name__ == "__main__":
    main()
