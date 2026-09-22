"""
Module 2 — Hyperparameter Tuning & Model Comparison Harness

Trains and compares several forecasting approaches on the SAME synthetic
corridor series, using the SAME train/val split, so results are directly
comparable:

  1. Naive persistence      (tomorrow = today)          -- sanity floor
  2. Seasonal naive (lag=7) (tomorrow = same day last wk) -- sanity floor
  3. GRU  (current architecture, varying hyperparams)
  4. LSTM (same shape, different recurrent cell)
  5. LightGBM on lag/calendar features                    -- non-RNN comparison

Any model that can't beat the naive baselines is a signal that either the
signal-to-noise ratio in the data is too low, or the model/features need
work -- not that you need a fancier architecture.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path

# Import from your actual pipeline file, whatever it's named, as long as
# it's sitting in the SAME directory as this script. Adjust MODULE_FILE
# below if your pipeline script has a different filename/module name.
MODULE_FILE = "demand_pipeline_module2"
sys.path.insert(0, str(Path(__file__).resolve().parent))

_pipeline = __import__(MODULE_FILE)
load_annual_dataset = _pipeline.load_annual_dataset
load_recent_dataset = _pipeline.load_recent_dataset
load_monthly_dataset = _pipeline.load_monthly_dataset
fit_commodity_growth_rates = _pipeline.fit_commodity_growth_rates
fit_monthly_seasonal_index = _pipeline.fit_monthly_seasonal_index
fetch_chennai_corridors_from_graph = _pipeline.fetch_chennai_corridors_from_graph
FALLBACK_CORRIDORS = _pipeline.FALLBACK_CORRIDORS

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Fixed corridor-weight bug from the uploaded script (root cause of the
# quantized/flat predictions): don't shrink the signal by /n_corridors.
# ---------------------------------------------------------------------------

def synthesize_corridor_series_fixed(corridors, growth_rates, seasonal_index, n_years=3):
    base_date = pd.Timestamp.today().normalize() - pd.Timedelta(days=365 * n_years)
    dates = pd.date_range(base_date, periods=365 * n_years, freq="D")
    base_year = dates[0].year
    records = []
    for origin, dest, commodity, corridor_weight in corridors:
        g_rate = growth_rates.get(commodity, 0.03)
        s_index = seasonal_index.get(commodity, {m: 1.0 for m in range(1, 13)})
        base_tonnage = 1200.0
        for d in dates:
            years_elapsed = (d.year + d.month / 12.0) - base_year
            trend = base_tonnage * math.exp(g_rate * years_elapsed)
            seasonal_mult = s_index.get(d.month, 1.0)
            weekday_mult = 0.75 if d.dayofweek >= 5 else 1.0
            noise = np.random.normal(loc=1.0, scale=0.08)
            disruption = 1.0
            if random.random() < 0.01:
                disruption = random.choice([0.3, 2.5])
            tonnage = max(0.0, corridor_weight * trend * seasonal_mult * weekday_mult * noise * disruption)
            records.append({
                "date": d, "origin": origin, "dest": dest, "commodity": commodity,
                "tonnage": tonnage, "month": d.month,
            })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Windowed dataset (shared by GRU/LSTM)
# ---------------------------------------------------------------------------

@dataclass
class HyperParams:
    lookback: int = 30
    horizon: int = 14
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0
    lr: float = 1e-3
    batch_size: int = 16
    epochs: int = 30
    underestimate_penalty: float = 2.0
    weight_decay: float = 0.0
    cell: str = "GRU"  # "GRU" or "LSTM"


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, series: pd.DataFrame, cfg: HyperParams):
        series = series.sort_values("date").reset_index(drop=True)
        tonnage = series["tonnage"].values.astype(np.float32)
        self.mean, self.std = tonnage.mean(), tonnage.std() + 1e-6
        tonnage_norm = (tonnage - self.mean) / self.std

        dow = series["date"].dt.dayofweek.values.astype(np.float32) / 6.0
        month_sin = np.sin(2 * np.pi * series["month"].values / 12).astype(np.float32)
        month_cos = np.cos(2 * np.pi * series["month"].values / 12).astype(np.float32)
        # Extra lag features beyond the base pipeline: 7-day and 14-day lag,
        # which give the model an explicit shortcut to weekly seasonality
        # instead of having to learn it purely from the recurrent state.
        lag7 = np.roll(tonnage_norm, 7)
        lag7[:7] = tonnage_norm[0]
        lag14 = np.roll(tonnage_norm, 14)
        lag14[:14] = tonnage_norm[0]

        features = np.stack([tonnage_norm, dow, month_sin, month_cos, lag7, lag14], axis=1)
        self.n_features = features.shape[1]

        self.X, self.y = [], []
        L, H = cfg.lookback, cfg.horizon
        for i in range(len(features) - L - H + 1):
            self.X.append(features[i:i + L])
            self.y.append(tonnage_norm[i + L:i + L + H])
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

    def denormalize(self, arr):
        return arr * self.std + self.mean


class RecurrentForecaster(nn.Module):
    def __init__(self, n_features: int, cfg: HyperParams):
        super().__init__()
        rnn_cls = nn.GRU if cfg.cell == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            n_features, cfg.hidden_size, num_layers=cfg.num_layers,
            batch_first=True, dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(cfg.hidden_size, cfg.horizon)
        self.cell = cfg.cell

    def forward(self, x):
        out = self.rnn(x)
        h_n = out[1] if self.cell == "GRU" else out[1][0]  # LSTM returns (h_n, c_n)
        h_last = h_n[-1]  # last layer's final hidden state
        return self.head(h_last)


class CostMatrixLoss(nn.Module):
    def __init__(self, underestimate_penalty: float = 2.0):
        super().__init__()
        self.p = underestimate_penalty

    def forward(self, y_pred, y_actual):
        error = y_actual - y_pred
        weight = torch.where(error > 0, self.p, 1.0)
        return torch.mean(weight * error ** 2)


def train_recurrent(series: pd.DataFrame, cfg: HyperParams):
    dataset = WindowDataset(series, cfg)
    split = int(len(dataset) * 0.85)
    train_ds = Subset(dataset, range(0, split))
    val_ds = Subset(dataset, range(split, len(dataset)))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = RecurrentForecaster(dataset.n_features, cfg)
    loss_fn = CostMatrixLoss(cfg.underestimate_penalty)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"train": [], "val": []}
    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= max(1, len(train_ds))

        model.eval()
        val_loss, val_mae_sum, n = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item() * xb.size(0)
                pred_dn = dataset.denormalize(pred.numpy())
                yb_dn = dataset.denormalize(yb.numpy())
                val_mae_sum += np.abs(pred_dn - yb_dn).sum()
                n += pred_dn.size
        val_loss /= max(1, len(val_ds))
        val_mae = val_mae_sum / max(1, n)
        history["train"].append(tr_loss)
        history["val"].append(val_loss)

    return model, dataset, history, val_mae


# ---------------------------------------------------------------------------
# Baselines: naive persistence, seasonal naive, LightGBM on lag features
# ---------------------------------------------------------------------------

def eval_naive_persistence(series: pd.DataFrame, horizon: int = 14) -> float:
    """Predicts every future day = the last known value. Classic sanity floor."""
    tonnage = series.sort_values("date")["tonnage"].values
    split = int(len(tonnage) * 0.85)
    errors = []
    for i in range(split, len(tonnage) - horizon):
        pred = np.full(horizon, tonnage[i])
        actual = tonnage[i + 1:i + 1 + horizon]
        errors.append(np.abs(pred - actual).mean())
    return float(np.mean(errors)) if errors else float("nan")


def eval_seasonal_naive(series: pd.DataFrame, horizon: int = 14, lag: int = 7) -> float:
    """Predicts day t+k = value from (t+k - lag). Captures weekly pattern
    for free, no training -- if this beats your trained model, the model
    isn't earning its complexity."""
    tonnage = series.sort_values("date")["tonnage"].values
    split = int(len(tonnage) * 0.85)
    errors = []
    for i in range(split, len(tonnage) - horizon):
        pred = np.array([tonnage[i + k - lag] if i + k - lag >= 0 else tonnage[i] for k in range(1, horizon + 1)])
        actual = tonnage[i + 1:i + 1 + horizon]
        errors.append(np.abs(pred - actual).mean())
    return float(np.mean(errors)) if errors else float("nan")


def eval_lightgbm(series: pd.DataFrame, horizon: int = 14, lookback: int = 30) -> float:
    """Tabular gradient-boosted-tree baseline on lag + calendar features.
    Trains a SEPARATE model per horizon step (standard multi-horizon
    approach for tree models, since they don't natively output sequences)."""
    series = series.sort_values("date").reset_index(drop=True)
    tonnage = series["tonnage"].values.astype(np.float64)
    dow = series["date"].dt.dayofweek.values
    month = series["date"].dt.month.values

    rows = []
    for i in range(lookback, len(tonnage) - horizon):
        feat = {
            "lag1": tonnage[i - 1], "lag7": tonnage[i - 7], "lag14": tonnage[i - 14],
            "roll7_mean": tonnage[i - 7:i].mean(), "roll30_mean": tonnage[i - lookback:i].mean(),
            "dow": dow[i], "month": month[i],
        }
        for h in range(horizon):
            feat[f"target_{h}"] = tonnage[i + h]
        rows.append(feat)
    df = pd.DataFrame(rows)

    split = int(len(df) * 0.85)
    train_df, val_df = df.iloc[:split], df.iloc[split:]
    feature_cols = ["lag1", "lag7", "lag14", "roll7_mean", "roll30_mean", "dow", "month"]

    horizon_maes = []
    for h in range(horizon):
        model = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, verbose=-1)
        model.fit(train_df[feature_cols], train_df[f"target_{h}"])
        pred = model.predict(val_df[feature_cols])
        horizon_maes.append(np.abs(pred - val_df[f"target_{h}"].values).mean())
    return float(np.mean(horizon_maes))


# ---------------------------------------------------------------------------
# Run comparison
# ---------------------------------------------------------------------------

def main():
    print("Loading real datasets and fitting priors...")
    annual_df = load_annual_dataset()
    monthly_df = load_monthly_dataset()
    recent_df = load_recent_dataset()
    growth_rates = fit_commodity_growth_rates(annual_df, recent_df)
    seasonal_index = fit_monthly_seasonal_index(monthly_df)

    # Pulls REAL station pairs from your freight_network AGE graph via
    # db.py, weighted by real FY2023 commodity earnings share. Falls back
    # to the placeholder list only if db.py/the DB connection isn't
    # reachable from wherever this script is run.
    corridors = fetch_chennai_corridors_from_graph(n_corridors=8)
    series_df = synthesize_corridor_series_fixed(corridors, growth_rates, seasonal_index, n_years=3)

    # Pick one representative corridor for the comparison.
    origin, dest, commodity, _ = corridors[0]
    sub = series_df[(series_df.origin == origin) & (series_df.dest == dest) & (series_df.commodity == commodity)]
    print(f"\nComparing models on corridor: {origin} -> {dest} [{commodity}]")
    print(f"Series length: {len(sub)} days, mean tonnage: {sub['tonnage'].mean():.2f}, std: {sub['tonnage'].std():.2f}")

    results = []

    # --- Baselines ---
    mae_persist = eval_naive_persistence(sub)
    mae_seasonal = eval_seasonal_naive(sub)
    mae_lgb = eval_lightgbm(sub)
    results.append({"model": "Naive Persistence", "val_mae": mae_persist, "notes": "sanity floor"})
    results.append({"model": "Seasonal Naive (lag=7)", "val_mae": mae_seasonal, "notes": "sanity floor"})
    results.append({"model": "LightGBM (lag features)", "val_mae": mae_lgb, "notes": "per-horizon-step trees"})

    # --- Recurrent model sweep: vary a few key hyperparameters ---
    configs = [
        ("GRU baseline",        HyperParams(cell="GRU",  hidden_size=64, num_layers=1, epochs=30, underestimate_penalty=2.0)),
        ("GRU, 2 layers+dropout", HyperParams(cell="GRU",  hidden_size=64, num_layers=2, dropout=0.2, epochs=30, underestimate_penalty=2.0)),
        ("GRU, larger hidden",  HyperParams(cell="GRU",  hidden_size=128, num_layers=1, epochs=30, underestimate_penalty=2.0)),
        ("GRU, low penalty",    HyperParams(cell="GRU",  hidden_size=64, num_layers=1, epochs=30, underestimate_penalty=1.0)),
        ("GRU, high penalty",   HyperParams(cell="GRU",  hidden_size=64, num_layers=1, epochs=30, underestimate_penalty=4.0)),
        ("LSTM baseline",       HyperParams(cell="LSTM", hidden_size=64, num_layers=1, epochs=30, underestimate_penalty=2.0)),
    ]

    histories = {}
    for name, cfg in configs:
        print(f"\nTraining: {name} ...")
        model, dataset, history, val_mae = train_recurrent(sub, cfg)
        results.append({"model": name, "val_mae": val_mae, "notes": f"final val_loss={history['val'][-1]:.4f}"})
        histories[name] = history["val"]

    # --- Report ---
    results_df = pd.DataFrame(results).sort_values("val_mae")
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (lower val_mae = better; units = tons/day)")
    print("=" * 70)
    print(results_df.to_string(index=False))

    # --- Plot validation loss curves for the recurrent sweep ---
    plt.figure(figsize=(9, 5))
    for name, val_hist in histories.items():
        plt.plot(val_hist, label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss (Cost-Matrix)")
    plt.title(f"Validation Loss by Hyperparameter Config\n{origin} -> {dest} [{commodity}]")
    plt.legend(fontsize=8)
    plt.tight_layout()
    out_dir = Path(__file__).resolve().parent
    plt.savefig(out_dir / "val_loss_comparison.png", dpi=120)
    print(f"\nSaved validation loss comparison plot to {out_dir / 'val_loss_comparison.png'}")

    results_df.to_csv(out_dir / "model_comparison_results.csv", index=False)
    print(f"Saved results table to {out_dir / 'model_comparison_results.csv'}")


if __name__ == "__main__":
    main()