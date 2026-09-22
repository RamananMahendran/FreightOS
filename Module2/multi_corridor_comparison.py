"""
Module 2 — Multi-Corridor Model Comparison Harness (v2)

Improvements over the single-corridor version:
  1. Sweeps ALL real corridors pulled from your freight_network graph, not
     just one -- so "best model" reflects robustness, not a lucky corridor.
  2. Tracks the BEST validation-MAE checkpoint per run, not just whatever
     the final epoch happened to land on (the previous run showed some
     configs spiking late and getting lucky/unlucky on the last epoch).
  3. Early stopping on val MAE (patience-based) so the extra corridors
     don't blow up total runtime.
  4. Aggregates results by RANK per corridor, then averages ranks across
     corridors -- this is more meaningful than averaging raw MAE, since
     different corridors have very different demand scales (Coal vs.
     Fertilizers aren't on the same tonnage scale, so their MAEs aren't
     directly comparable, but their WITHIN-corridor rankings are).
"""

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path

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

MODULE_FILE = "demand_pipeline_module2"  # <- change if your pipeline script has a different filename
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
# Synthetic series generation (corridor_weight bug already fixed upstream
# in demand_pipeline_module2.py -- this just uses whatever weight the
# corridor tuple carries, as-is).
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
# Windowed dataset + models
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
    cell: str = "GRU"
    early_stop_patience: int = 8


class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, series: pd.DataFrame, cfg: HyperParams):
        series = series.sort_values("date").reset_index(drop=True)
        tonnage = series["tonnage"].values.astype(np.float32)
        self.mean, self.std = tonnage.mean(), tonnage.std() + 1e-6
        tonnage_norm = (tonnage - self.mean) / self.std

        dow = series["date"].dt.dayofweek.values.astype(np.float32) / 6.0
        month_sin = np.sin(2 * np.pi * series["month"].values / 12).astype(np.float32)
        month_cos = np.cos(2 * np.pi * series["month"].values / 12).astype(np.float32)
        lag7 = np.roll(tonnage_norm, 7); lag7[:7] = tonnage_norm[0]
        lag14 = np.roll(tonnage_norm, 14); lag14[:14] = tonnage_norm[0]

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
        h_n = out[1] if self.cell == "GRU" else out[1][0]
        return self.head(h_n[-1])


class CostMatrixLoss(nn.Module):
    def __init__(self, underestimate_penalty: float = 2.0):
        super().__init__()
        self.p = underestimate_penalty

    def forward(self, y_pred, y_actual):
        error = y_actual - y_pred
        weight = torch.where(error > 0, self.p, 1.0)
        return torch.mean(weight * error ** 2)


def train_recurrent_best_checkpoint(series: pd.DataFrame, cfg: HyperParams):
    """Trains with early stopping on val MAE, and returns the BEST
    checkpoint's val_mae (not whatever the final epoch happened to be)."""
    dataset = WindowDataset(series, cfg)
    split = int(len(dataset) * 0.85)
    train_ds = Subset(dataset, range(0, split))
    val_ds = Subset(dataset, range(split, len(dataset)))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = RecurrentForecaster(dataset.n_features, cfg)
    loss_fn = CostMatrixLoss(cfg.underestimate_penalty)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_mae = float("inf")
    best_epoch = -1
    best_state = None
    epochs_since_improve = 0
    val_mae_history = []

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        mae_sum, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                pred_dn = dataset.denormalize(pred.numpy())
                yb_dn = dataset.denormalize(yb.numpy())
                mae_sum += np.abs(pred_dn - yb_dn).sum()
                n += pred_dn.size
        val_mae = mae_sum / max(1, n)
        val_mae_history.append(val_mae)

        if val_mae < best_val_mae - 1e-6:
            best_val_mae = val_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= cfg.early_stop_patience:
                break

    return best_val_mae, best_epoch, val_mae_history


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def eval_naive_persistence(series: pd.DataFrame, horizon: int = 14) -> float:
    tonnage = series.sort_values("date")["tonnage"].values
    split = int(len(tonnage) * 0.85)
    errors = []
    for i in range(split, len(tonnage) - horizon):
        pred = np.full(horizon, tonnage[i])
        actual = tonnage[i + 1:i + 1 + horizon]
        errors.append(np.abs(pred - actual).mean())
    return float(np.mean(errors)) if errors else float("nan")


def eval_seasonal_naive(series: pd.DataFrame, horizon: int = 14, lag: int = 7) -> float:
    tonnage = series.sort_values("date")["tonnage"].values
    split = int(len(tonnage) * 0.85)
    errors = []
    for i in range(split, len(tonnage) - horizon):
        pred = np.array([tonnage[i + k - lag] if i + k - lag >= 0 else tonnage[i] for k in range(1, horizon + 1)])
        actual = tonnage[i + 1:i + 1 + horizon]
        errors.append(np.abs(pred - actual).mean())
    return float(np.mean(errors)) if errors else float("nan")


def eval_lightgbm(series: pd.DataFrame, horizon: int = 14, lookback: int = 30) -> float:
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
# Multi-corridor sweep + rank aggregation
# ---------------------------------------------------------------------------

CONFIGS = [
    ("GRU baseline",          HyperParams(cell="GRU",  hidden_size=64,  num_layers=1, underestimate_penalty=2.0)),
    ("GRU, 2 layers+dropout", HyperParams(cell="GRU",  hidden_size=64,  num_layers=2, dropout=0.2, underestimate_penalty=2.0)),
    ("GRU, larger hidden",    HyperParams(cell="GRU",  hidden_size=128, num_layers=1, underestimate_penalty=2.0)),
    ("GRU, low penalty",      HyperParams(cell="GRU",  hidden_size=64,  num_layers=1, underestimate_penalty=1.0)),
    ("GRU, high penalty",     HyperParams(cell="GRU",  hidden_size=64,  num_layers=1, underestimate_penalty=4.0)),
    ("LSTM baseline",         HyperParams(cell="LSTM", hidden_size=64,  num_layers=1, underestimate_penalty=2.0)),
]


def main():
    print("Loading real datasets and fitting priors...")
    annual_df = load_annual_dataset()
    monthly_df = load_monthly_dataset()
    recent_df = load_recent_dataset()
    growth_rates = fit_commodity_growth_rates(annual_df, recent_df)
    seasonal_index = fit_monthly_seasonal_index(monthly_df)

    corridors = fetch_chennai_corridors_from_graph(n_corridors=8)
    series_df = synthesize_corridor_series_fixed(corridors, growth_rates, seasonal_index, n_years=3)

    all_rows = []          # long-format: corridor, model, val_mae
    all_val_histories = {} # {corridor_label: {model_name: [val_mae per epoch]}}

    for origin, dest, commodity, _ in corridors:
        corridor_label = f"{origin[-6:]}->{dest[-6:]} [{commodity}]"
        sub = series_df[
            (series_df.origin == origin) & (series_df.dest == dest) & (series_df.commodity == commodity)
        ]
        if len(sub) < 100:
            continue

        print(f"\n{'=' * 70}\nCorridor: {corridor_label}  (mean={sub['tonnage'].mean():.2f}, std={sub['tonnage'].std():.2f})\n{'=' * 70}")

        all_rows.append({"corridor": corridor_label, "model": "Naive Persistence", "val_mae": eval_naive_persistence(sub)})
        all_rows.append({"corridor": corridor_label, "model": "Seasonal Naive (lag=7)", "val_mae": eval_seasonal_naive(sub)})
        all_rows.append({"corridor": corridor_label, "model": "LightGBM (lag features)", "val_mae": eval_lightgbm(sub)})

        histories = {}
        for name, cfg in CONFIGS:
            best_val_mae, best_epoch, val_hist = train_recurrent_best_checkpoint(sub, cfg)
            print(f"  {name:28s} best_val_mae={best_val_mae:.4f} at epoch {best_epoch} (of {len(val_hist)} run)")
            all_rows.append({"corridor": corridor_label, "model": name, "val_mae": best_val_mae})
            histories[name] = val_hist
        all_val_histories[corridor_label] = histories

    results_df = pd.DataFrame(all_rows)

    # --- Rank-based aggregation: rank models WITHIN each corridor (1=best),
    # then average rank across corridors. This is fair across corridors
    # with very different tonnage scales, unlike averaging raw MAE. ---
    results_df["rank"] = results_df.groupby("corridor")["val_mae"].rank(method="min")
    summary = (
        results_df.groupby("model")
        .agg(avg_rank=("rank", "mean"), avg_val_mae=("val_mae", "mean"), n_corridors=("corridor", "nunique"))
        .sort_values("avg_rank")
    )

    print("\n" + "=" * 70)
    print("PER-CORRIDOR RESULTS")
    print("=" * 70)
    pivot = results_df.pivot(index="model", columns="corridor", values="val_mae")
    print(pivot.round(2).to_string())

    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY (ranked by AVERAGE RANK across corridors -- lower is better/more consistent)")
    print("=" * 70)
    print(summary.round(3).to_string())

    out_dir = Path(__file__).resolve().parent
    results_df.to_csv(out_dir / "multi_corridor_results_long.csv", index=False)
    pivot.to_csv(out_dir / "multi_corridor_results_pivot.csv")
    summary.to_csv(out_dir / "multi_corridor_robustness_summary.csv")
    print(f"\nSaved: multi_corridor_results_long.csv, multi_corridor_results_pivot.csv, multi_corridor_robustness_summary.csv")

    # --- Plot: one subplot per corridor, all recurrent models overlaid ---
    n_corridors_plotted = len(all_val_histories)
    ncols = 2
    nrows = math.ceil(n_corridors_plotted / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    for idx, (corridor_label, histories) in enumerate(all_val_histories.items()):
        ax = axes[idx // ncols][idx % ncols]
        for name, val_hist in histories.items():
            ax.plot(val_hist, label=name, linewidth=1.2)
        ax.set_title(corridor_label, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val MAE")
        if idx == 0:
            ax.legend(fontsize=6)
    for idx in range(n_corridors_plotted, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    plt.suptitle("Validation MAE per Epoch, by Corridor (early-stopped)")
    plt.tight_layout()
    plt.savefig(out_dir / "multi_corridor_val_mae.png", dpi=120)
    print("Saved multi_corridor_val_mae.png")


if __name__ == "__main__":
    main()
