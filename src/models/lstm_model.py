"""LSTM forecaster with recursive multi-step prediction.

Architecture: deliberately small.
- 2-layer LSTM, hidden_size=64, dropout=0.2
- Linear head → scalar output (scaled sales)
- ~5600 training sequences with 18 features is a small dataset for deep learning.
  A bigger network would overfit. This architecture is sized to the data.

Global model (same rationale as XGBoost):
  Sequences from all 43 states are pooled. The features already encode
  state-level context via state_expanding_mean, lag values, and rolling stats
  which carry state-specific magnitude information through the scaler.

Training:
- AdamW with cosine annealing LR schedule
- Early stopping on validation loss (patience=15)
- Gradient clipping at 1.0 to prevent exploding gradients
- Reproducible via seeded RNG

When LSTM underperforms on this data:
  Weekly business time series with <200 observations per state is a worst case
  for LSTMs. They shine on long, high-frequency, complex-pattern sequences
  (10k+ steps, sub-daily). Here the signal is dominated by simple momentum
  (rolling means explain ~90% of variance in XGBoost). An LSTM has to learn
  that same momentum from raw sequences, with far fewer parameters to spare
  for it, while also fighting the overhead of sequential training. Expect it
  to be competitive but unlikely to beat XGBoost on this dataset.
"""

import logging
import pickle
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config.settings import (
    DATE_COL, STATE_COL, TARGET_COL,
    LAG_PERIODS, SEASONAL_LAG, ROLLING_WINDOWS,
    FORECAST_WEEKS, ARTIFACTS_DIR, LSTM_SEQ_LEN,
)
from src.dataset import (
    StateScaler, LSTM_FEATURES,
    prepare_lstm_sequences,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters (all in one place for easy tuning)
# ---------------------------------------------------------------------------
HPARAMS = {
    "hidden_size": 64,
    "num_layers": 2,
    "dropout": 0.2,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "batch_size": 64,
    "max_epochs": 150,
    "patience": 15,
    "grad_clip": 1.0,
    "seed": 42,
}


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        out, _ = self.lstm(x)
        # Take the last timestep's hidden state
        last = out[:, -1, :]
        return self.head(last).squeeze(-1)


# ---------------------------------------------------------------------------
# Forecaster
# ---------------------------------------------------------------------------

class LSTMForecaster:
    """Global LSTM model with recursive multi-step prediction."""

    def __init__(self, hparams: dict | None = None):
        self.hp = {**HPARAMS, **(hparams or {})}
        self.net: _LSTMNet | None = None
        self.scaler: StateScaler | None = None
        self.feature_cols = LSTM_FEATURES
        self.seq_len = LSTM_SEQ_LEN
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    @property
    def name(self) -> str:
        return "lstm"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        scaler: StateScaler,
    ) -> "LSTMForecaster":
        seed_everything(self.hp["seed"])
        self.scaler = scaler

        n_features = X_train.shape[2]
        self.net = _LSTMNet(
            input_size=n_features,
            hidden_size=self.hp["hidden_size"],
            num_layers=self.hp["num_layers"],
            dropout=self.hp["dropout"],
        ).to(self.device)

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        )
        val_ds = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        )
        train_dl = DataLoader(train_ds, batch_size=self.hp["batch_size"], shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=self.hp["batch_size"])

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hp["lr"],
            weight_decay=self.hp["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hp["max_epochs"],
        )
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(1, self.hp["max_epochs"] + 1):
            # Train
            self.net.train()
            epoch_loss = 0.0
            for xb, yb in train_dl:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                pred = self.net(xb)
                loss = criterion(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.hp["grad_clip"])
                optimizer.step()
                epoch_loss += loss.item() * len(xb)
            train_loss = epoch_loss / len(train_ds)
            self.train_losses.append(train_loss)

            # Validate
            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.net(xb)
                    val_loss += criterion(pred, yb).item() * len(xb)
            val_loss /= len(val_ds)
            self.val_losses.append(val_loss)

            scheduler.step()

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.net.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch % 10 == 0 or epoch == 1 or patience_counter == 0:
                lr = optimizer.param_groups[0]["lr"]
                marker = " *" if patience_counter == 0 else ""
                logger.info(
                    "Epoch %3d | train=%.6f | val=%.6f | lr=%.2e%s",
                    epoch, train_loss, val_loss, lr, marker,
                )

            if patience_counter >= self.hp["patience"]:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, self.hp["patience"])
                break

        # Restore best weights
        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()

        logger.info("Best val loss: %.6f", best_val_loss)
        return self

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------

    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Raw prediction on scaled sequences. Returns scaled output."""
        self.net.eval()
        with torch.no_grad():
            t = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.net(t).cpu().numpy()

    def predict_sequences(self, X: np.ndarray, meta: list[dict]) -> pd.DataFrame:
        """Predict on pre-built sequences, inverse-transform per state."""
        raw = self._predict_batch(X)
        rows = []
        for i, m in enumerate(meta):
            scaled_val = float(raw[i])
            actual_val = float(self.scaler.inverse_target(
                pd.Series([scaled_val]), m["state"]
            ).iloc[0])
            rows.append({"state": m["state"], "date": m["date"], "forecast": actual_val})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Recursive multi-step prediction (per state)
    # ------------------------------------------------------------------

    def predict_recursive(
        self,
        history_df: pd.DataFrame,
        state: str,
        horizon: int = FORECAST_WEEKS,
    ) -> pd.DataFrame:
        """Forecast `horizon` weeks ahead for one state.

        Builds the last sequence from history, predicts one step,
        shifts the window forward (appending the prediction), and repeats.

        The sequence window contains scaled feature vectors. For each new step
        we must reconstruct the feature row from the raw sales history,
        scale it, and append it to the sliding window.
        """
        state_df = (
            history_df[history_df[STATE_COL] == state]
            .sort_values(DATE_COL)
            .copy()
        )
        # Drop NaN feature rows
        valid = state_df.dropna(subset=self.feature_cols).reset_index(drop=True)
        if len(valid) < self.seq_len:
            logger.warning("[lstm][%s] Not enough valid rows for sequence", state)
            return pd.DataFrame(columns=["date", "forecast"])

        # Build the initial scaled sequence window
        scaled_df = self.scaler.transform_features(valid, self.feature_cols)
        window = scaled_df[self.feature_cols].values[-self.seq_len:].astype(np.float32)

        # Maintain raw sales buffer for feature reconstruction
        sales_buffer = valid[TARGET_COL].values.tolist()
        last_date = valid[DATE_COL].max()

        all_lags = sorted(set(LAG_PERIODS + [SEASONAL_LAG]))
        forecasts = []

        for step in range(horizon):
            # Predict one step from current window
            pred_scaled = self._predict_batch(window[np.newaxis, :, :])[0]
            pred_raw = float(self.scaler.inverse_target(
                pd.Series([pred_scaled]), state,
            ).iloc[0])

            next_date = last_date + pd.Timedelta(weeks=step + 1)
            forecasts.append({"date": next_date, "forecast": pred_raw})
            sales_buffer.append(pred_raw)

            # Build the next feature row from raw sales history
            raw_row = self._build_feature_row(sales_buffer, next_date, all_lags)
            # Scale it
            row_df = pd.DataFrame([raw_row], columns=self.feature_cols)
            scaled_row = self.scaler.transform_features(row_df, self.feature_cols)
            new_vec = scaled_row[self.feature_cols].values[0].astype(np.float32)

            # Shift window
            window = np.vstack([window[1:], new_vec[np.newaxis, :]])

        return pd.DataFrame(forecasts)

    def _build_feature_row(
        self,
        sales_history: list[float],
        date: pd.Timestamp,
        all_lags: list[int],
    ) -> dict:
        """Reconstruct a raw (unscaled) feature row for a future date."""
        row = {}

        # Cyclical calendar (LSTM features exclude raw month/week_of_year/etc.)
        row["month_sin"] = np.sin(2 * np.pi * date.month / 12)
        row["month_cos"] = np.cos(2 * np.pi * date.month / 12)
        woy = date.isocalendar().week
        row["week_of_year_sin"] = np.sin(2 * np.pi * woy / 52)
        row["week_of_year_cos"] = np.cos(2 * np.pi * woy / 52)

        # Holiday
        try:
            import holidays as _hol
            hset = set(_hol.US(years=date.year).keys())
            row["is_holiday_week"] = int(any(
                (date - pd.Timedelta(days=i)).date() in hset for i in range(7)
            ))
        except ImportError:
            row["is_holiday_week"] = 0

        row["week_index"] = len(sales_history)

        # Lags
        for lag in all_lags:
            if lag <= len(sales_history):
                row[f"lag_{lag}"] = sales_history[-lag]
            else:
                row[f"lag_{lag}"] = sales_history[-1]  # fallback to most recent available

        # Rolling
        for w in ROLLING_WINDOWS:
            recent = sales_history[-w:] if len(sales_history) >= w else sales_history[:]
            row[f"roll_mean_{w}"] = np.mean(recent)
            row[f"roll_std_{w}"] = np.std(recent, ddof=1) if len(recent) > 1 else 0.0

        # Derived
        if len(sales_history) >= 2 and sales_history[-2] != 0:
            row["pct_change_1"] = (sales_history[-1] - sales_history[-2]) / sales_history[-2]
        else:
            row["pct_change_1"] = 0.0

        row["state_expanding_mean"] = np.mean(sales_history)

        return row

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "net_state": self.net.state_dict(),
            "hp": self.hp,
            "feature_cols": self.feature_cols,
            "seq_len": self.seq_len,
            "n_features": self.net.lstm.input_size,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }, path)

    @classmethod
    def load(cls, path: Path, scaler: StateScaler) -> "LSTMForecaster":
        data = torch.load(path, map_location="cpu", weights_only=False)
        obj = cls(hparams=data["hp"])
        obj.scaler = scaler
        obj.feature_cols = data["feature_cols"]
        obj.seq_len = data["seq_len"]
        obj.train_losses = data.get("train_losses", [])
        obj.val_losses = data.get("val_losses", [])
        obj.net = _LSTMNet(
            input_size=data["n_features"],
            hidden_size=data["hp"]["hidden_size"],
            num_layers=data["hp"]["num_layers"],
            dropout=0.0,  # no dropout at inference
        ).to(obj.device)
        obj.net.load_state_dict(data["net_state"])
        obj.net.eval()
        return obj
