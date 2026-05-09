"""Train SARIMA and Prophet on all states, evaluate, and save results.

Usage:
    python scripts/train_classical.py
    python scripts/train_classical.py --model sarima
    python scripts/train_classical.py --model prophet
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import ARTIFACTS_DIR
from src.split import load_splits
from src.models.sarima import SARIMAModel
from src.models.prophet_model import ProphetModel
from src.train_classical import train_model_all_states
from src.evaluate import format_metrics

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_classical")

MODELS = {
    "sarima": (SARIMAModel, {}),
    "prophet": (ProphetModel, {}),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS.keys()), default=None,
                        help="Train a single model (default: all)")
    args = parser.parse_args()

    splits = load_splits()
    train_df, val_df = splits["train"], splits["val"]

    (ARTIFACTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)

    models_to_run = {args.model: MODELS[args.model]} if args.model else MODELS
    all_results = []

    for model_name, (model_cls, kwargs) in models_to_run.items():
        metrics_df = train_model_all_states(model_cls, train_df, val_df, kwargs)
        all_results.append(metrics_df)

    if len(all_results) > 1:
        combined = pd.concat(all_results, ignore_index=True)
        summary = combined.groupby("model")[["mae", "rmse", "smape", "mase"]].agg(["mean", "median"])
        logger.info("\n=== Cross-Model Comparison ===\n%s", summary.to_string())
        combined.to_csv(ARTIFACTS_DIR / "metrics" / "classical_combined.csv", index=False)


if __name__ == "__main__":
    main()
