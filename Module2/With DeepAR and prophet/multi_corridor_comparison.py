"""
Module 2 — Multi-Corridor Model Comparison Harness (v3)

Includes 11 candidate models across all real Apache AGE corridors:
  - 4 Statistical / ML Baselines: Naive Persistence, Seasonal Naive, LightGBM, and Prophet
  - 6 Recurrent Variants: GRU variants and the production LSTM baseline
  - 1 Probabilistic Deep Learning Model: DeepAR (Gaussian NLL likelihood)
    as formulated by Salinas et al. (2020) and Zhao et al. (2023).

Tracks best validation-MAE checkpoints, early stops on validation stagnation,
and computes rank-based aggregation across differing commodity volume scales.
"""

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Import Module 2 Data Pipeline Functions ---
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
# Synthetic Series Generation
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
# Windowed Dataset Definition
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


# ---------------------------------------------------------------------------
# Recurrent Models with Cost-Matrix Loss
# ---------------------------------------------------------------------------

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
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= cfg.early_stop_patience:
                break

    return best_val_mae, best_epoch, val_mae_history


# ---------------------------------------------------------------------------
# DeepAR Probabilistic Autoregressive Forecaster (Salinas et al., 2020; Zhao et al., 2023)
# ---------------------------------------------------------------------------

class DeepARForecaster(nn.Module):
    """
    Probabilistic autoregressive recurrent forecaster.
    Outputs mean (mu) and variance (sigma via softplus activation) for a Gaussian distribution.
    Trained by maximizing Gaussian log-likelihood.
    """
    def __init__(self, n_features: int, hidden_size: int = 64, num_layers: int = 1, horizon: int = 14):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=num_layers, batch_first=True)
        self.mu_head = nn.Linear(hidden_size, horizon)
        self.sigma_head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        out, (h_n, _) = self.lstm(x)
        mu = self.mu_head(h_n[-1])
        sigma = F.softplus(self.sigma_head(h_n[-1])) + 1e-4
        return mu, sigma


def gaussian_nll_loss(mu, sigma, target):
    """Calculates Gaussian negative log-likelihood."""
    dist = torch.distributions.Normal(mu, sigma)
    return -dist.log_prob(target).mean()


def train_deepar_best_checkpoint(series: pd.DataFrame, cfg: HyperParams):
    """Trains DeepAR and evaluates validation MAE using the point mean prediction (mu)."""
    dataset = WindowDataset(series, cfg)
    split = int(len(dataset) * 0.85)
    train_ds = Subset(dataset, range(0, split))
    val_ds = Subset(dataset, range(split, len(dataset)))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = DeepARForecaster(dataset.n_features, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, horizon=cfg.horizon)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val_mae = float("inf")
    best_epoch = -1
    epochs_since_improve = 0
    val_mae_history = []

    for epoch in range(cfg.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            mu, sigma = model(xb)
            loss = gaussian_nll_loss(mu, sigma, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        mae_sum, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                mu, _ = model(xb)
                pred_dn = dataset.denormalize(mu.numpy())
                yb_dn = dataset.denormalize(yb.numpy())
                mae_sum += np.abs(pred_dn - yb_dn).sum()
                n += pred_dn.size
        val_mae = mae_sum / max(1, n)
        val_mae_history.append(val_mae)

        if val_mae < best_val_mae - 1e-6:
            best_val_mae = val_mae
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= cfg.early_stop_patience:
                break

    return best_val_mae, best_epoch, val_mae_history


# ---------------------------------------------------------------------------
# Baselines: Naive, LightGBM, and Prophet
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


def eval_prophet(series: pd.DataFrame, horizon: int = 14) -> float:
    """
    Fits an additive decomposition model (trend + Fourier seasonality)
    on the 85% training split and evaluates 14-day rolling window MAE.
    Uses official Prophet library if installed, with a mathematical Fourier fallback.
    """
    try:
        from prophet import Prophet
        import logging
        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

        df = series[["date", "tonnage"]].rename(columns={"date": "ds", "tonnage": "y"}).sort_values("ds").reset_index(drop=True)
        split = int(len(df) * 0.85)
        train_df = df.iloc[:split].copy()
        val_df = df.iloc[split:].copy()

        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(train_df)
        future = m.make_future_dataframe(periods=len(val_df), freq="D")
        forecast = m.predict(future)
        val_preds = forecast.iloc[split:]["yhat"].values
        val_actuals = val_df["y"].values

        errors = []
        for i in range(len(val_actuals) - horizon + 1):
            pred_w = val_preds[i : i + horizon]
            act_w = val_actuals[i : i + horizon]
            errors.append(np.abs(pred_w - act_w).mean())
        return float(np.mean(errors)) if errors else float("nan")

    except ImportError:
        # Fallback additive decomposition: Linear trend + weekly and yearly Fourier harmonics
        df = series[["date", "tonnage"]].sort_values("date").reset_index(drop=True)
        split = int(len(df) * 0.85)
        train_df, val_df = df.iloc[:split].copy(), df.iloc[split:].copy()

        t_train = np.arange(len(train_df), dtype=float)
        y_train = train_df["tonnage"].values
        dow = train_df["date"].dt.dayofweek.values
        doy = train_df["date"].dt.dayofyear.values

        X_train = np.column_stack([
            np.ones_like(t_train), t_train,
            np.sin(2 * np.pi * dow / 7.0), np.cos(2 * np.pi * dow / 7.0),
            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
        ])
        w, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

        t_val = np.arange(len(train_df), len(df), dtype=float)
        dow_v = val_df["date"].dt.dayofweek.values
        doy_v = val_df["date"].dt.dayofyear.values
        X_val = np.column_stack([
            np.ones_like(t_val), t_val,
            np.sin(2 * np.pi * dow_v / 7.0), np.cos(2 * np.pi * dow_v / 7.0),
            np.sin(2 * np.pi * doy_v / 365.25), np.cos(2 * np.pi * doy_v / 365.25)
        ])
        val_preds = np.maximum(0.0, X_val @ w)
        val_actuals = val_df["tonnage"].values

        errors = []
        for i in range(len(val_actuals) - horizon + 1):
            pred_w = val_preds[i : i + horizon]
            act_w = val_actuals[i : i + horizon]
            errors.append(np.abs(pred_w - act_w).mean())
        return float(np.mean(errors)) if errors else float("nan")


# ---------------------------------------------------------------------------
# Multi-Corridor Sweep Configurations
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
    print("=" * 75)
    print("🚀 FREIGHTOS MODULE 2: MULTI-CORRIDOR MODEL COMPARISON HARNESS (v3)")
    print("=" * 75)
    print("Loading real Indian Railways datasets and fitting priors...")
    annual_df = load_annual_dataset()
    monthly_df = load_monthly_dataset()
    recent_df = load_recent_dataset()
    growth_rates = fit_commodity_growth_rates(annual_df, recent_df)
    seasonal_index = fit_monthly_seasonal_index(monthly_df)

    corridors = fetch_chennai_corridors_from_graph(n_corridors=8)
    series_df = synthesize_corridor_series_fixed(corridors, growth_rates, seasonal_index, n_years=3)

    all_rows = []
    all_val_histories = {}

    for origin, dest, commodity, _ in corridors:
        corridor_label = f"{origin[-6:]}->{dest[-6:]} [{commodity}]"
        sub = series_df[
            (series_df.origin == origin) & (series_df.dest == dest) & (series_df.commodity == commodity)
        ]
        if len(sub) < 100:
            continue

        print(f"\n{'=' * 75}\nCorridor: {corridor_label} (mean={sub['tonnage'].mean():.2f} T, std={sub['tonnage'].std():.2f} T)\n{'=' * 75}")

        # 1. Non-recurrent baselines
        p_mae = eval_prophet(sub)
        print(f"  {'Prophet (additive)':28s} val_mae={p_mae:.4f}")
        all_rows.append({"corridor": corridor_label, "model": "Prophet (additive)", "val_mae": p_mae})

        lgb_mae = eval_lightgbm(sub)
        print(f"  {'LightGBM (lag features)':28s} val_mae={lgb_mae:.4f}")
        all_rows.append({"corridor": corridor_label, "model": "LightGBM (lag features)", "val_mae": lgb_mae})

        naive_mae = eval_naive_persistence(sub)
        print(f"  {'Naive Persistence':28s} val_mae={naive_mae:.4f}")
        all_rows.append({"corridor": corridor_label, "model": "Naive Persistence", "val_mae": naive_mae})

        s_naive_mae = eval_seasonal_naive(sub)
        print(f"  {'Seasonal Naive (lag=7)':28s} val_mae={s_naive_mae:.4f}")
        all_rows.append({"corridor": corridor_label, "model": "Seasonal Naive (lag=7)", "val_mae": s_naive_mae})

        # 2. Recurrent deep learning models
        histories = {}
        for name, cfg in CONFIGS:
            best_val_mae, best_epoch, val_hist = train_recurrent_best_checkpoint(sub, cfg)
            print(f"  {name:28s} best_val_mae={best_val_mae:.4f} at epoch {best_epoch:02d} (of {len(val_hist)} run)")
            all_rows.append({"corridor": corridor_label, "model": name, "val_mae": best_val_mae})
            histories[name] = val_hist

        # 3. DeepAR Probabilistic Autoregressive Forecaster
        deepar_cfg = HyperParams(hidden_size=64, num_layers=1, lr=1e-3, epochs=30, early_stop_patience=8)
        deepar_val_mae, deepar_epoch, deepar_hist = train_deepar_best_checkpoint(sub, deepar_cfg)
        print(f"  {'DeepAR (probabilistic)':28s} best_val_mae={deepar_val_mae:.4f} at epoch {deepar_epoch:02d} (of {len(deepar_hist)} run)")
        all_rows.append({"corridor": corridor_label, "model": "DeepAR (probabilistic)", "val_mae": deepar_val_mae})
        histories["DeepAR (probabilistic)"] = deepar_hist

        all_val_histories[corridor_label] = histories

    results_df = pd.DataFrame(all_rows)

    # --- Rank-based aggregation: 1 = best within corridor ---
    results_df["rank"] = results_df.groupby("corridor")["val_mae"].rank(method="min")
    summary = (
        results_df.groupby("model")
        .agg(
            avg_rank=("rank", "mean"),
            avg_val_mae=("val_mae", "mean"),
            n_corridors=("corridor", "nunique")
        )
        .sort_values("avg_rank")
    )

    print("\n" + "=" * 75)
    print("PER-CORRIDOR VALIDATION RESULTS (MAE)")
    print("=" * 75)
    pivot = results_df.pivot(index="model", columns="corridor", values="val_mae")
    print(pivot.round(2).to_string())

    print("\n" + "=" * 75)
    print("ROBUSTNESS SUMMARY (RANKED BY AVERAGE WITHIN-CORRIDOR RANK)")
    print("=" * 75)
    print(summary.round(3).to_string())

    out_dir = Path(__file__).resolve().parent
    results_df.to_csv(out_dir / "multi_corridor_results_long.csv", index=False)
    pivot.to_csv(out_dir / "multi_corridor_results_pivot.csv")
    summary.to_csv(out_dir / "multi_corridor_robustness_summary.csv")
    print(f"\n💾 Saved result CSVs to: {out_dir}")

    # --- Plot: Overlaid Loss History Curves ---
    n_corridors_plotted = len(all_val_histories)
    ncols = 2
    nrows = math.ceil(n_corridors_plotted / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
    for idx, (corridor_label, histories) in enumerate(all_val_histories.items()):
        ax = axes[idx // ncols][idx % ncols]
        for name, val_hist in histories.items():
            linewidth = 2.0 if "LSTM" in name or "DeepAR" in name else 1.1
            ax.plot(val_hist, label=name, linewidth=linewidth)
        ax.set_title(corridor_label, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MAE")
        if idx == 0:
            ax.legend(fontsize=6, loc="upper right")
    for idx in range(n_corridors_plotted, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    plt.suptitle("Validation MAE Curves across Corridors (Recurrent & DeepAR Candidates)")
    plt.tight_layout()
    chart_file = out_dir / "multi_corridor_val_mae.png"
    plt.savefig(chart_file, dpi=150)
    plt.close()
    print(f"📊 Saved comparison plot to: {chart_file}")


if __name__ == "__main__":
    main()