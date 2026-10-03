#!/usr/bin/env python3
"""
FreightOS Module 2 -- Revised Evaluation Harness (v4)
=====================================================

Replaces multi_corridor_comparison.py (v3). What changed and why
----------------------------------------------------------------
1. Three-way chronological split (70% train / 15% val / 15% test).
   Neural checkpoints are selected on VAL; every reported number is on TEST,
   so checkpoint selection no longer inflates the neural models' scores.
   (v3 also never restored the best weights -- it only stored the best MAE
   number. v4 reloads the best state_dict.)
2. Rolling-origin evaluation for EVERY model on the same TEST origins
   (one origin every --stride days, 14-day horizon):
     - Naive / Seasonal-naive : trivial, use latest data at each origin
     - LightGBM               : direct multi-horizon, refit every --refit-every origins
                                on an expanding window
     - Prophet                : refit every --refit-every origins on an expanding window
     - GRU / LSTM / DeepAR    : trained once on TRAIN (best epoch chosen on VAL);
                                weights frozen, but the 30-day input window is
                                refreshed with the latest actuals at each origin.
   NOTE: classical models see train+val+test-so-far at refit time, while the
   neural models are not retrained during the test span. That asymmetry favours
   the classical models under regime shifts -- keep it in mind when reading the
   'hard' scenario.
3. Multiple seeds (default 5). The seed controls BOTH the synthetic series draw
   and the network initialisation, so variance reflects data noise as well.
4. Two synthetic scenarios:
     baseline : same structure as v3 (exp. trend x monthly index x weekday x
                i.i.d. 8% noise x 1% single-day disruptions)
     hard     : AR(1) log-noise (phi=0.7), persistent level/regime shifts,
                multi-day disruption/surge episodes
   Prophet's structural assumptions fit 'baseline' almost perfectly by
   construction; 'hard' is the fairer test of history-aware models.
5. Metrics reported together (all on the TEST span):
     mae, nmae, bias_pct, under_rate,
     asym_cost    : the Cost-Matrix objective (squared error, c_under=2, c_over=1),
                    scale-normalised so corridors can be pooled
     pinball_q67  : pinball loss at q = c_under/(c_under+c_over) = 2/3
6. DeepAR-style probabilistic scoring: pinball@0.67, CRPS (quantile-grid
   approximation), weighted quantile loss, quantile-grid calibration error,
   central-interval coverage (50/80/90%) and sharpness. Reported both for the raw
   Gaussian head and for a VAL-recalibrated version (empirical z-scores).
7. Decision forecasts derived from DeepAR's distribution:
     q=0.67 quantile  -> optimal under LINEAR asymmetric cost (pinball)
     0.67 expectile   -> optimal under the SQUARED asymmetric cost actually used
                         by the Cost-Matrix loss
   (A squared 2:1 cost is minimised by the 2/3-EXPECTILE, not the 2/3-quantile;
   for a Gaussian the expectile offset is smaller: ~+0.30 sigma vs ~+0.43 sigma.)
8. Rank tests: Friedman + Nemenyi critical difference on (seed x corridor)
   blocks, for a 'core' model set and a 'decision' model set. Blocks share
   corridor configurations, so treat p-values as indicative, not exact.
9. The Prophet backend actually used is recorded. If the `prophet` package is
   missing the harness uses a clearly-labelled Fourier-OLS fallback (never
   reported as "Prophet"); use --require-prophet to make that a hard error.

Naming caveat: the "DeepAR-style" model is an LSTM with a Gaussian (mu, sigma)
head producing all 14 horizons directly from the last hidden state, trained by
Gaussian NLL. It is NOT the autoregressive-sampling DeepAR of Salinas et al.

Usage
-----
  python module2_eval_harness.py                       # full run (5 seeds x 2 scenarios)
  python module2_eval_harness.py --smoke               # quick plumbing check
  python module2_eval_harness.py --no-neural           # classical models only
  python module2_eval_harness.py --require-prophet --prophet-multiplicative
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq

try:
    import torch
    import torch.nn as nn

    HAVE_TORCH = True
except Exception:  # missing OR broken torch install; allows --no-neural runs
    HAVE_TORCH = False

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
HORIZON = 14
LOOKBACK = 30
C_UNDER, C_OVER = 2.0, 1.0
TAU = C_UNDER / (C_UNDER + C_OVER)  # 2/3: optimal quantile (linear) / expectile (squared)
QUANTILE_GRID = np.round(np.arange(0.025, 0.9751, 0.025), 4)  # 39 levels
INTERVAL_LEVELS = (0.5, 0.8, 0.9)
LGB_MIN_T = 29  # first origin index with a full 30-day feature history

METRICS = ["mae", "nmae", "bias_pct", "under_rate", "asym_cost", "pinball_q67", "npinball_q67"]

DA_MEAN = "DeepAR-style (mean)"
DA_Q_G = "DeepAR-style (q=0.67, Gaussian)"
DA_E_G = "DeepAR-style (expectile, Gaussian)"
DA_Q_R = "DeepAR-style (q=0.67, recal)"
DA_E_R = "DeepAR-style (expectile, recal)"
LGB_NAME = "LightGBM (rolling refit)"
NAIVE, SNAIVE = "Naive Persistence", "Seasonal Naive (lag=7)"


# --------------------------------------------------------------------------- #
# Splits and origins
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Splits:
    a: int  # first VAL index; TRAIN = [0, a)
    b: int  # first TEST index; VAL = [a, b)
    n: int


def make_splits(n: int, train_frac: float = 0.70, val_frac: float = 0.15) -> Splits:
    return Splits(int(n * train_frac), int(n * (train_frac + val_frac)), n)


def origins_train(s: Splits, L: int, H: int) -> list[int]:
    return list(range(L - 1, s.a - H))  # targets t+1..t+H all inside TRAIN


def origins_val(s: Splits, H: int) -> list[int]:
    return list(range(s.a - 1, s.b - H))  # targets all inside VAL


def origins_test(s: Splits, H: int, stride: int) -> list[int]:
    return list(range(s.b - 1, s.n - H, stride))  # targets all inside TEST


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #
def _hard_multiplier(rng: np.random.Generator, n: int) -> np.ndarray:
    """AR(1) log-noise x persistent level shifts x multi-day disruption episodes."""
    phi, sig = 0.7, 0.08
    innov = rng.normal(0.0, sig * math.sqrt(1 - phi**2), n)
    e = np.empty(n)
    e[0] = rng.normal(0.0, sig)
    for i in range(1, n):
        e[i] = phi * e[i - 1] + innov[i]
    level = np.ones(n)
    n_shifts = max(1, int(rng.poisson(2.5)))
    for cp in np.sort(rng.integers(int(0.15 * n), int(0.95 * n), size=n_shifts)):
        level[cp:] *= math.exp(rng.normal(0.0, 0.25))
    disr = np.ones(n)
    for s in np.flatnonzero(rng.random(n) < 0.01):
        dur = int(rng.integers(3, 11))
        mult = rng.uniform(0.3, 0.6) if rng.random() < 0.5 else rng.uniform(1.5, 2.5)
        disr[s : s + dur] *= mult
    return np.exp(e) * level * disr


def synthesize_series(corridors, growth_rates, seasonal_index, scenario: str, seed: int,
                      n_years: int = 3, start: str = "2023-01-01"):
    """Returns {label: (dates, y)}. Fixed start date + seeded Generator => reproducible."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=int(365 * n_years), freq="D")
    n = len(dates)
    t_years = np.arange(n) / 365.25
    month, dow = dates.month.to_numpy(), dates.dayofweek.to_numpy()
    out = {}
    for origin, dest, commodity, weight in corridors:
        g = growth_rates.get(commodity, 0.03)
        s_idx = seasonal_index.get(commodity, {m: 1.0 for m in range(1, 13)})
        seas = np.array([s_idx.get(m, 1.0) for m in range(1, 13)])[month - 1]
        base = weight * 1200.0 * np.exp(g * t_years) * seas * np.where(dow >= 5, 0.75, 1.0)
        if scenario == "baseline":
            noise = rng.normal(1.0, 0.08, n)
            disr = np.ones(n)
            hit = rng.random(n) < 0.01
            disr[hit] = rng.choice([0.3, 2.5], int(hit.sum()))
            mult = noise * disr
        elif scenario == "hard":
            mult = _hard_multiplier(rng, n)
        else:
            raise ValueError(f"unknown scenario {scenario!r}")
        label = f"{str(origin)[-6:]}->{str(dest)[-6:]} [{commodity}]"
        out[label] = (dates, np.maximum(0.0, base * mult))
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def point_metrics(pred: np.ndarray, act: np.ndarray, scale: float) -> dict:
    err = act - pred
    w = np.where(err > 0, C_UNDER, C_OVER)
    pin = np.maximum(TAU * err, (TAU - 1.0) * err)
    mae = float(np.abs(err).mean())
    return dict(
        mae=mae,
        nmae=mae / scale,
        bias_pct=float(100.0 * (-err).mean() / scale),  # >0 = over-forecast
        under_rate=float((err > 0).mean()),
        asym_cost=float((w * (err / scale) ** 2).mean()),
        pinball_q67=float(pin.mean()),
        npinball_q67=float(pin.mean() / scale),
    )


def expectile_std_normal(tau: float) -> float:
    f = lambda e: tau * (stats.norm.pdf(e) - e * stats.norm.sf(e)) \
        - (1 - tau) * (stats.norm.pdf(e) + e * stats.norm.cdf(e))
    return float(brentq(f, -5.0, 5.0))


def empirical_expectile(z: np.ndarray, tau: float) -> float:
    f = lambda e: tau * np.maximum(z - e, 0.0).mean() - (1 - tau) * np.maximum(e - z, 0.0).mean()
    return float(brentq(f, float(z.min()), float(z.max())))


def quantile_array(mu: np.ndarray, sigma: np.ndarray, z_levels: np.ndarray) -> np.ndarray:
    """mu, sigma: (n, H) in tonnes; z_levels: (Q,) -> (n, H, Q)."""
    return mu[..., None] + sigma[..., None] * z_levels


def prob_metrics(Q: np.ndarray, act: np.ndarray, levels: np.ndarray, scale: float) -> dict:
    """Quantile-array scoring. CRPS ~= 2 * mean pinball over the (central) quantile grid."""
    diff = act[..., None] - Q
    pin = np.maximum(levels * diff, (levels - 1.0) * diff)
    hit = (act[..., None] <= Q).mean(axis=(0, 1))
    out = dict(
        crps_q=float(2.0 * pin.mean() / scale),
        wql=float(pin.mean() / scale),
        ace=float(np.abs(hit - levels).mean()),  # mean |empirical - nominal| quantile coverage
    )
    for cl in INTERVAL_LEVELS:
        lo_q = (1 - cl) / 2
        i_lo, i_hi = int(np.argmin(np.abs(levels - lo_q))), int(np.argmin(np.abs(levels - (1 - lo_q))))
        lo, hi = Q[..., i_lo], Q[..., i_hi]
        tag = int(cl * 100)
        out[f"cov{tag}"] = float(((act >= lo) & (act <= hi)).mean())
        out[f"width{tag}"] = float((hi - lo).mean() / scale)
    return out


# --------------------------------------------------------------------------- #
# Classical models (rolling origin)
# --------------------------------------------------------------------------- #
def forecast_naive(y, origins, H):
    return np.stack([np.full(H, y[t]) for t in origins])


def forecast_seasonal_naive(y, origins, H, lag=7):
    k = np.arange(1, H + 1)
    off = k - lag * np.ceil(k / lag).astype(int)  # always <= 0 -> only past data
    return np.stack([y[t + off] for t in origins])


def _lgb_features(y, dates):
    s = pd.Series(y)
    nxt = dates + pd.Timedelta(days=1)
    F = np.column_stack([
        y, s.shift(6), s.shift(13), s.rolling(7).mean(), s.rolling(30).mean(),
        nxt.dayofweek.to_numpy(), nxt.month.to_numpy(),
    ]).astype(float)
    return F  # row t uses data up to and including t; valid for t >= 29


def forecast_lightgbm_rolling(y, dates, origins, H, refit_every, seed):
    F = _lgb_features(y, dates)
    preds = np.empty((len(origins), H))
    for j0 in range(0, len(origins), refit_every):
        block = origins[j0:j0 + refit_every]
        t0 = block[0]
        rows = np.arange(LGB_MIN_T, t0 - H + 1)  # targets <= t0: no look-ahead
        Xb = F[block]
        for h in range(H):
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  random_state=seed, verbose=-1, n_jobs=1)
            m.fit(F[rows], y[rows + 1 + h])
            preds[j0:j0 + len(block), h] = m.predict(Xb)
    return preds


def detect_prophet_backend() -> str:
    try:
        import prophet  # noqa: F401
        return "prophet"
    except Exception:
        return "fallback"


def _prophet_fit_predict(train_dates, train_y, fut_dates, mode, backend):
    if backend == "prophet":
        from prophet import Prophet
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    seasonality_mode=mode, uncertainty_samples=0)
        m.fit(pd.DataFrame({"ds": train_dates, "y": train_y}))
        return m.predict(pd.DataFrame({"ds": fut_dates}))["yhat"].to_numpy()

    def design(d):
        t = np.asarray((d - train_dates[0]).days, dtype=float)
        dow, doy = d.dayofweek.to_numpy(), d.dayofyear.to_numpy()
        return np.column_stack([np.ones_like(t), t,
                                np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
                                np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])
    w, *_ = np.linalg.lstsq(design(train_dates), train_y, rcond=None)
    return np.maximum(0.0, design(fut_dates) @ w)


def forecast_prophet_rolling(y, dates, origins, H, refit_every, mode, backend):
    preds = np.empty((len(origins), H))
    for j0 in range(0, len(origins), refit_every):
        block = origins[j0:j0 + refit_every]
        t0, t_last = block[0], block[-1]
        yhat = _prophet_fit_predict(dates[:t0 + 1], y[:t0 + 1], dates[t0 + 1:t_last + H + 1], mode, backend)
        for k, t in enumerate(block):
            preds[j0 + k] = yhat[t - t0: t - t0 + H]
    return preds


# --------------------------------------------------------------------------- #
# Neural models
# --------------------------------------------------------------------------- #
@dataclass
class NNConfig:
    name: str
    cell: str = "GRU"
    hidden: int = 64
    layers: int = 1
    dropout: float = 0.0
    c_under: float = 2.0


NN_CONFIGS = [
    NNConfig("GRU baseline", "GRU"),
    NNConfig("GRU, 2 layers+dropout", "GRU", layers=2, dropout=0.2),
    NNConfig("GRU, larger hidden", "GRU", hidden=128),
    NNConfig("GRU, low penalty", "GRU", c_under=1.0),
    NNConfig("GRU, high penalty", "GRU", c_under=4.0),
    NNConfig("LSTM baseline", "LSTM"),
]

if HAVE_TORCH:
    class RecurrentForecaster(nn.Module):
        def __init__(self, n_features: int, cfg: NNConfig, horizon: int):
            super().__init__()
            rnn = nn.GRU if cfg.cell == "GRU" else nn.LSTM
            self.rnn = rnn(n_features, cfg.hidden, num_layers=cfg.layers, batch_first=True,
                           dropout=cfg.dropout if cfg.layers > 1 else 0.0)
            self.head = nn.Linear(cfg.hidden, horizon)
            self.is_gru = cfg.cell == "GRU"

        def forward(self, x):
            out = self.rnn(x)
            h_n = out[1] if self.is_gru else out[1][0]
            return self.head(h_n[-1])

    class GaussianHeadForecaster(nn.Module):
        """'DeepAR-style': LSTM + direct multi-horizon (mu, sigma) head."""
        def __init__(self, n_features: int, hidden: int, horizon: int):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
            self.mu_head = nn.Linear(hidden, horizon)
            self.sigma_head = nn.Linear(hidden, horizon)

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            h = h_n[-1]
            return self.mu_head(h), torch.nn.functional.softplus(self.sigma_head(h)) + 1e-4

    def cost_matrix_loss(pred, target, c_under):
        err = target - pred
        w = torch.where(err > 0, torch.full_like(err, c_under), torch.ones_like(err))
        return (w * err ** 2).mean()

    def gaussian_nll(mu, sigma, target):
        return (0.5 * torch.log(2 * math.pi * sigma ** 2) + (target - mu) ** 2 / (2 * sigma ** 2)).mean()

    def train_best_checkpoint(model, loss_fn, point_fn, Xtr, Ytr, Xva, Yva, sd, args, seed):
        """Trains on TRAIN, selects the best epoch on VAL MAE, RESTORES those weights."""
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        gen = torch.Generator().manual_seed(seed)
        best, best_state, best_ep, stale = math.inf, None, -1, 0
        for ep in range(args.epochs):
            model.train()
            perm = torch.randperm(len(Xtr), generator=gen)
            for i in range(0, len(perm), args.batch_size):
                idx = perm[i:i + args.batch_size]
                opt.zero_grad()
                loss_fn(model, Xtr[idx], Ytr[idx]).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.eval()
            with torch.no_grad():
                v = (point_fn(model, Xva) - Yva).abs().mean().item() * sd
            if v < best - 1e-9:
                best, best_ep, stale = v, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= args.patience:
                    break
        model.load_state_dict(best_state)
        model.eval()
        return best, best_ep


def build_features(yn: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    """Same 6 features as Module 2: [y_norm, dow, sin(m), cos(m), lag7, lag14]."""
    m = dates.month.to_numpy()
    lag7 = np.concatenate([np.full(7, yn[0]), yn[:-7]])
    lag14 = np.concatenate([np.full(14, yn[0]), yn[:-14]])
    return np.stack([yn, dates.dayofweek.to_numpy() / 6.0, np.sin(2 * np.pi * m / 12),
                     np.cos(2 * np.pi * m / 12), lag7, lag14], axis=1).astype(np.float32)


def make_windows(feats, yn, origins, L, H):
    X = np.stack([feats[t - L + 1:t + 1] for t in origins]).astype(np.float32)
    Y = np.stack([yn[t + 1:t + 1 + H] for t in origins]).astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(Y)


def run_neural(feats, yn, S, o_tr, o_va, o_te, mu0, sd, seed, args):
    """Returns (point_preds {name: (n,H) tonnes}, deepar dict)."""
    Xtr, Ytr = make_windows(feats, yn, o_tr, LOOKBACK, HORIZON)
    Xva, Yva = make_windows(feats, yn, o_va, LOOKBACK, HORIZON)
    Xte, _ = make_windows(feats, yn, o_te, LOOKBACK, HORIZON)
    nf = feats.shape[1]
    preds = {}
    for cfg in NN_CONFIGS:
        torch.manual_seed(seed)
        model = RecurrentForecaster(nf, cfg, HORIZON)
        train_best_checkpoint(
            model, lambda m, x, y, c=cfg.c_under: cost_matrix_loss(m(x), y, c),
            lambda m, x: m(x), Xtr, Ytr, Xva, Yva, sd, args, seed)
        with torch.no_grad():
            preds[cfg.name] = model(Xte).numpy() * sd + mu0

    torch.manual_seed(seed)
    da = GaussianHeadForecaster(nf, 64, HORIZON)
    train_best_checkpoint(da, lambda m, x, y: gaussian_nll(*m(x), y), lambda m, x: m(x)[0],
                          Xtr, Ytr, Xva, Yva, sd, args, seed)
    with torch.no_grad():
        mu_va, sg_va = (t.numpy() for t in da(Xva))
        mu_te, sg_te = (t.numpy() for t in da(Xte))
    z_val = ((Yva.numpy() - mu_va) / sg_va).ravel()  # scale-free standardized residuals
    return preds, dict(mu=mu_te * sd + mu0, sigma=sg_te * sd, z_val=z_val)


def deepar_outputs(da: dict, act: np.ndarray, scale: float):
    """Decision forecasts (point rows) and probabilistic-score rows from the (mu, sigma) head."""
    mu, sg, zv = da["mu"], da["sigma"], da["z_val"]
    variants = {
        "Gaussian": dict(z_grid=stats.norm.ppf(QUANTILE_GRID), zq=float(stats.norm.ppf(TAU)),
                         ze=expectile_std_normal(TAU)),
        "recal": dict(z_grid=np.quantile(zv, QUANTILE_GRID), zq=float(np.quantile(zv, TAU)),
                      ze=empirical_expectile(zv, TAU)),
    }
    point_preds = {DA_MEAN: mu}
    prob_rows = []
    for tag, v in variants.items():
        q_pred = mu + sg * v["zq"]
        e_pred = mu + sg * v["ze"]
        point_preds[DA_Q_G if tag == "Gaussian" else DA_Q_R] = q_pred
        point_preds[DA_E_G if tag == "Gaussian" else DA_E_R] = e_pred
        pm = prob_metrics(quantile_array(mu, sg, v["z_grid"]), act, QUANTILE_GRID, scale)
        pm["hit_q67"] = float((act <= q_pred).mean())  # should be ~0.667 if calibrated
        pm["z_q67"], pm["z_expectile67"] = v["zq"], v["ze"]
        prob_rows.append(dict(model=f"DeepAR-style ({tag})", **pm))
    return point_preds, prob_rows


# --------------------------------------------------------------------------- #
# Per-series evaluation
# --------------------------------------------------------------------------- #
def evaluate_series(label, dates, y, scenario, seed, args, prophet_labels, backend):
    S = make_splits(len(y))
    o_tr, o_va = origins_train(S, LOOKBACK, HORIZON), origins_val(S, HORIZON)
    o_te = origins_test(S, HORIZON, args.stride)
    act = np.stack([y[t + 1:t + 1 + HORIZON] for t in o_te])
    scale = float(act.mean())
    mu0, sd = float(y[:S.a].mean()), float(y[:S.a].std() + 1e-6)

    preds = {NAIVE: forecast_naive(y, o_te, HORIZON), SNAIVE: forecast_seasonal_naive(y, o_te, HORIZON),
             LGB_NAME: forecast_lightgbm_rolling(y, dates, o_te, HORIZON, args.refit_every, seed)}
    if not args.skip_prophet:
        for mode, name in prophet_labels.items():
            preds[name] = forecast_prophet_rolling(y, dates, o_te, HORIZON, args.refit_every, mode, backend)

    prob_rows = []
    if not args.no_neural:
        yn = (y - mu0) / sd
        nn_preds, da = run_neural(build_features(yn, dates), yn, S, o_tr, o_va, o_te, mu0, sd, seed, args)
        preds.update(nn_preds)
        da_preds, prob_rows = deepar_outputs(da, act, scale)
        preds.update(da_preds)

    base = dict(scenario=scenario, seed=seed, corridor=label)
    rows = [dict(**base, model=name, backend=backend if name.startswith("Prophet") else "",
                 **point_metrics(p, act, scale)) for name, p in preds.items()]
    prob_rows = [dict(**base, **r) for r in prob_rows]
    return rows, prob_rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    for m in ("mae", "asym_cost", "pinball_q67"):
        df[f"rank_{m}"] = df.groupby(["scenario", "seed", "corridor"])[m].rank(method="average")
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = {m: ["mean", "std"] for m in METRICS}
    agg.update({"rank_mae": "mean", "rank_asym_cost": "mean", "rank_pinball_q67": "mean"})
    s = df.groupby(["scenario", "model"]).agg(agg)
    s.columns = ["_".join(c) if c[1] in ("mean", "std") and not c[0].startswith("rank_") else c[0]
                 for c in s.columns]
    return s.reset_index().sort_values(["scenario", "rank_mae"])


def rank_test(df, scenario, models, metric, alpha=0.05):
    piv = df[df.scenario == scenario].pivot_table(index=["seed", "corridor"], columns="model", values=metric)
    cols = [m for m in models if m in piv.columns]
    if len(cols) < 3:
        return None
    piv = piv[cols].dropna()
    n, k = piv.shape
    mean_ranks = piv.rank(axis=1).mean().sort_values()
    chi2, p = stats.friedmanchisquare(*[piv[c].to_numpy() for c in cols])
    try:
        cd = float(stats.studentized_range.ppf(1 - alpha, k, 10_000) / math.sqrt(2) * math.sqrt(k * (k + 1) / (6 * n)))
    except Exception:
        cd = float("nan")
    return dict(scenario=scenario, metric=metric, n_blocks=n, k=k, friedman_chi2=float(chi2),
                friedman_p=float(p), nemenyi_cd=cd, mean_ranks=mean_ranks)


def paired(df, scenario, a, b, metric):
    piv = df[df.scenario == scenario].pivot_table(index=["seed", "corridor"], columns="model", values=metric)
    if a not in piv.columns or b not in piv.columns:
        return None
    d = (piv[a] - piv[b]).dropna()
    p = float("nan")
    if len(d) >= 6 and (d != 0).any():
        p = float(stats.wilcoxon(d).pvalue)
    return dict(scenario=scenario, a=a, b=b, metric=metric, n=len(d),
                a_wins=int((d < 0).sum()), median_diff=float(d.median()), wilcoxon_p=p)


def print_report(df, prob_df, prophet_name, out_dir):
    core = [NAIVE, SNAIVE, LGB_NAME, prophet_name, "GRU baseline", "LSTM baseline", DA_MEAN]
    decision = [prophet_name, LGB_NAME, "LSTM baseline", "GRU, low penalty", "GRU, high penalty",
                DA_MEAN, DA_Q_R, DA_E_R]
    plan = [("core", core, "mae"), ("core", core, "asym_cost"),
            ("decision", decision, "asym_cost"), ("decision", decision, "pinball_q67")]
    pairs = [(prophet_name, "LSTM baseline", "nmae"), (DA_MEAN, "LSTM baseline", "nmae"),
             (prophet_name, "LSTM baseline", "asym_cost"),
             (DA_E_R, "LSTM baseline", "asym_cost"), (DA_Q_R, "LSTM baseline", "npinball_q67")]
    pd.set_option("display.width", 200)
    test_rows = []
    for sc in df.scenario.unique():
        sub = df[df.scenario == sc]
        print(f"\n{'=' * 100}\nSCENARIO: {sc}   (blocks = seeds x corridors = "
              f"{sub.groupby(['seed', 'corridor']).ngroups})\n{'=' * 100}")
        tab = summarize(df[df.scenario == sc])
        show = tab[["model", "nmae_mean", "bias_pct_mean", "under_rate_mean", "asym_cost_mean",
                    "npinball_q67_mean", "rank_mae", "rank_asym_cost", "rank_pinball_q67"]]
        print(show.round(4).to_string(index=False))
        for set_name, models, metric in plan:
            r = rank_test(df, sc, models, metric)
            if r is None:
                continue
            best = r["mean_ranks"].iloc[0]
            print(f"\n[{set_name} set | {metric}] Friedman chi2={r['friedman_chi2']:.1f} p={r['friedman_p']:.3g} | "
                  f"Nemenyi CD={r['nemenyi_cd']:.2f} (n={r['n_blocks']}, k={r['k']})")
            for m, rk in r["mean_ranks"].items():
                tie = "  <- tied with best" if (rk - best) < r["nemenyi_cd"] and rk != best else ""
                print(f"    {rk:6.2f}  {m}{tie}")
            for m, rk in r["mean_ranks"].items():
                test_rows.append(dict(scenario=sc, model_set=set_name, metric=metric, model=m, mean_rank=rk,
                                      friedman_p=r["friedman_p"], nemenyi_cd=r["nemenyi_cd"],
                                      n_blocks=r["n_blocks"], tied_with_best=bool((rk - best) < r["nemenyi_cd"])))
        print("\nPaired comparisons (a_wins = # blocks where a beats b; lower metric is better):")
        for a, b, metric in pairs:
            r = paired(df, sc, a, b, metric)
            if r:
                print(f"    {a} vs {b} [{metric}]: wins {r['a_wins']}/{r['n']}, "
                      f"median diff {r['median_diff']:+.4f}, Wilcoxon p={r['wilcoxon_p']:.3g}")
    if len(prob_df):
        print(f"\n{'=' * 100}\nDEEPAR-STYLE PROBABILISTIC SCORES (TEST, mean over blocks)\n{'=' * 100}")
        cols = ["crps_q", "wql", "ace", "cov50", "cov80", "cov90", "width80", "hit_q67"]
        print("Nominal: cov50=0.50  cov80=0.80  cov90=0.90  hit_q67=0.667  (ace, crps, wql: lower is better)")
        print(prob_df.groupby(["scenario", "model"])[cols].mean().round(4).to_string())
    pd.DataFrame(test_rows).to_csv(out_dir / "rank_tests.csv", index=False)


def make_plots(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    scen = list(df.scenario.unique())
    metrics = [("nmae", "Normalised MAE"), ("asym_cost", "Asymmetric cost (c_u=2)"), ("bias_pct", "Bias % (+ = over)")]
    fig, axes = plt.subplots(len(scen), 3, figsize=(18, 5.5 * len(scen)), squeeze=False)
    for i, sc in enumerate(scen):
        sub = df[df.scenario == sc]
        order = sub.groupby("model")["nmae"].mean().sort_values().index
        for j, (m, title) in enumerate(metrics):
            g = sub.groupby("model")[m].agg(["mean", "std"]).loc[order]
            ax = axes[i][j]
            ax.barh(range(len(g)), g["mean"], xerr=g["std"], alpha=0.8)
            ax.set_yticks(range(len(g)))
            ax.set_yticklabels(g.index if j == 0 else [], fontsize=7)
            ax.invert_yaxis()
            ax.set_title(f"{sc}: {title}", fontsize=10)
            if m == "bias_pct":
                ax.axvline(0, color="k", lw=0.8)
    plt.tight_layout()
    plt.savefig(out_dir / "eval_summary.png", dpi=140)
    plt.close()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", nargs="+", default=["baseline", "hard"], choices=["baseline", "hard"])
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--n-corridors", type=int, default=8)
    p.add_argument("--n-years", type=int, default=3)
    p.add_argument("--stride", type=int, default=5, help="days between rolling test origins (avoid multiples of 7: origins would all share one weekday)")
    p.add_argument("--refit-every", type=int, default=4, help="origins between Prophet/LightGBM refits")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--skip-prophet", action="store_true")
    p.add_argument("--prophet-multiplicative", action="store_true", help="also evaluate multiplicative Prophet")
    p.add_argument("--require-prophet", action="store_true", help="error instead of using the Fourier fallback")
    p.add_argument("--no-neural", action="store_true")
    p.add_argument("--plots", action="store_true")
    p.add_argument("--smoke", action="store_true", help="1 seed, 2 corridors, 3 epochs")
    p.add_argument("--out-dir", default="module2_eval_out")
    a = p.parse_args()
    if a.smoke:
        a.n_seeds, a.n_corridors, a.epochs = 1, 2, 3
    return a


def main():
    args = parse_args()
    if not args.no_neural and not HAVE_TORCH:
        sys.exit("PyTorch is not installed. Install it or pass --no-neural.")
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
    logging.getLogger("prophet").setLevel(logging.ERROR)

    backend = detect_prophet_backend()
    if backend == "fallback" and not args.skip_prophet:
        msg = "`prophet` package not importable -> using Fourier-OLS FALLBACK (labelled as such)."
        if args.require_prophet:
            sys.exit("ERROR: " + msg)
        print("WARNING:", msg)
    if backend == "prophet":
        prophet_labels = {"additive": "Prophet (additive)"}
        if args.prophet_multiplicative:
            prophet_labels["multiplicative"] = "Prophet (multiplicative)"
    else:
        prophet_labels = {"additive": "Prophet-fallback (Fourier OLS)"}
    prophet_name = next(iter(prophet_labels.values()))

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    pipe = importlib.import_module("demand_pipeline_module2")
    growth = pipe.fit_commodity_growth_rates(pipe.load_annual_dataset(), pipe.load_recent_dataset())
    seasonal = pipe.fit_monthly_seasonal_index(pipe.load_monthly_dataset())
    try:
        corridors = pipe.fetch_chennai_corridors_from_graph(n_corridors=args.n_corridors)
    except Exception as exc:  # AGE database unreachable
        print(f"WARNING: AGE graph unavailable ({exc}); using FALLBACK_CORRIDORS.")
        corridors = pipe.FALLBACK_CORRIDORS
    corridors = list(corridors)[:args.n_corridors]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, prob_rows = [], []
    t_start = time.time()
    for scenario in args.scenarios:
        for si in range(args.n_seeds):
            seed = args.seed0 + si
            random.seed(seed)
            np.random.seed(seed)
            series = synthesize_series(corridors, growth, seasonal, scenario, seed, args.n_years)
            for label, (dates, y) in series.items():
                r, pr = evaluate_series(label, dates, y, scenario, seed, args, prophet_labels, backend)
                rows += r
                prob_rows += pr
                print(f"[{time.time() - t_start:7.0f}s] {scenario:8s} seed={seed} {label}")

    df = add_ranks(pd.DataFrame(rows))
    prob_df = pd.DataFrame(prob_rows)
    df.to_csv(out_dir / "results_long.csv", index=False)
    summarize(df).to_csv(out_dir / "summary_by_scenario.csv", index=False)
    prob_df.to_csv(out_dir / "deepar_probabilistic.csv", index=False)
    (out_dir / "run_config.json").write_text(json.dumps(
        dict(args=vars(args), prophet_backend=backend, prophet_name=prophet_name,
             split="70/15/15", horizon=HORIZON, lookback=LOOKBACK, tau=TAU,
             torch=getattr(torch, "__version__", None) if HAVE_TORCH else None), indent=2, default=str))
    print_report(df, prob_df, prophet_name, out_dir)
    if args.plots:
        make_plots(df, out_dir)
    print(f"\nSaved results to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
