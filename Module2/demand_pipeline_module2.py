"""
FreightOS Module 2: Cognitive Demand & Predictive Analytics Engine
Integrates Indian Railways historical freight statistics with Apache AGE
Topological Gravity Ranking for corridor candidate selection.
"""

import os
import re
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Local dataset paths (auto-resolves from current directory or data subfolder)
ANNUAL_CSV = os.getenv("ANNUAL_CSV", "Railway_Key_Statistics_1950-51_to_2013-14.CSV")
MONTHLY_CSV = os.getenv("MONTHLY_CSV", "mothly_Railway_Traffic_Earnings_Freight_Revenue_upto_may_2014.csv")
RECENT_CSV = os.getenv("RECENT_CSV", "ramanan2312001_17867751255402272.csv")

COMMODITY_MAP = {
    "annual": {
        "Cement": "Cement",
        "Food grains": "Foodgrains",
        "Fertilisers": "Fertilizers",
        "Mineral oil (P.O.L.)": "Petroleum, Oil and Lubricant",
        "COAL for- Total Rev.Coal": "Coal",
        "Container Services- Total": "Container Service",
        "Raw mat. for steel plants": "Raw Material for Steel Plants",
        "Pig iron & finished steel- Total": "Pig Iron and Finished Steel",
        "Iron ore- Total": "Iron Ore",
        "Other Goods": "Others",
    },
    "monthly": {
        "Cement": "Cement",
        "Foodgrains": "Foodgrains",
        "Fertilizers": "Fertilizers",
        "Petroleum, Oil and Lubricant": "Petroleum, Oil and Lubricant",
        "Coal": "Coal",
        "Container Service": "Container Service",
        "Raw Material for Steel Plants": "Raw Material for Steel Plants",
        "Pig Iron and Finished Steel": "Pig Iron and Finished Steel",
        "Iron Ore and Steel": "Iron Ore",
        "Others": "Others",
    },
    "recent": {
        "Cement": "Cement",
        "Food grains": "Foodgrains",
        "Fertilizers": "Fertilizers",
        "Mineral oil ( POL )": "Petroleum, Oil and Lubricant",
        "Coal": "Coal",
        "Container Services": "Container Service",
        "Raw material for steel plants except iron ore": "Raw Material for Steel Plants",
        "Pig Iron and finished steel": "Pig Iron and Finished Steel",
        "Iron ore": "Iron Ore",
        "Balance other goods": "Others",
    },
}

DB_MODULE_AVAILABLE = True
try:
    import sys
    sys.path.append(os.path.abspath("../Module1"))
    from db import execute_cypher
except ImportError:
    try:
        from db import execute_cypher
    except ImportError:
        DB_MODULE_AVAILABLE = False

FALLBACK_CORRIDORS = [
    ("OSM_NODE_248545108", "OSM_NODE_261716087", "Container Service", 0.015),
    ("OSM_NODE_248545108", "OSM_NODE_2048191334", "Coal", 0.012),
    ("OSM_NODE_261716087", "OSM_NODE_262308384", "Cement", 0.008),
    ("OSM_NODE_248545108", "OSM_NODE_304278622", "Foodgrains", 0.007),
]

# ---------------------------------------------------------------------------
# Stage 1: Load & Clean Real Indian Railways Datasets
# ---------------------------------------------------------------------------

def _clean_numeric(val) -> float:
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.upper() == "NA" or s == "":
        return np.nan
    match = re.match(r"^-?\d+(\.\d+)?", s)
    return float(match.group()) if match else np.nan

def load_annual_dataset() -> pd.DataFrame:
    df = pd.read_csv(ANNUAL_CSV)
    df["Year"] = df["Year"].astype(str).str.extract(r"(\d{4})")[0].astype(int)
    records = []
    for annual_col, canonical in COMMODITY_MAP["annual"].items():
        if annual_col not in df.columns:
            continue
        for _, row in df.iterrows():
            val = _clean_numeric(row[annual_col])
            if not np.isnan(val):
                records.append({"year": row["Year"], "commodity": canonical, "value": val})
    return pd.DataFrame(records)

def load_recent_dataset() -> pd.DataFrame:
    df = pd.read_csv(RECENT_CSV)
    df.columns = [c.strip() for c in df.columns]
    tonnes_col = [c for c in df.columns if c.startswith("Tonnes Originating")][0]
    ntkm_col = [c for c in df.columns if c.startswith("Net Tonne Kilometers")][0]
    earnings_col = [c for c in df.columns if c.startswith("Earnings")][0]

    records = []
    for _, row in df.iterrows():
        commodity_raw = str(row["Commodity Or Commodity Group"]).strip()
        canonical = COMMODITY_MAP["recent"].get(commodity_raw)
        if canonical is None:
            continue
        year = int(re.search(r"(\d{4})", str(row["Year"])).group(1))
        records.append({
            "year": year,
            "commodity": canonical,
            "tonnes_million": _clean_numeric(row[tonnes_col]),
            "ntkm_million": _clean_numeric(row[ntkm_col]),
            "earnings_crore": _clean_numeric(row[earnings_col]),
        })
    return pd.DataFrame(records)

def load_monthly_dataset() -> pd.DataFrame:
    df = pd.read_csv(MONTHLY_CSV)
    df.columns = [c.strip() for c in df.columns]
    month_cols = [c for c in df.columns if c != "Particulars"]

    records = []
    for _, row in df.iterrows():
        commodity_raw = str(row["Particulars"]).strip()
        canonical = COMMODITY_MAP["monthly"].get(commodity_raw)
        if canonical is None:
            continue
        for col in month_cols:
            month_str, year_str = col.strip().split("-")
            val = _clean_numeric(row[col])
            if not np.isnan(val):
                records.append({
                    "year": int(year_str), "month": int(month_str),
                    "commodity": canonical, "value": val
                })
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Stage 2 & 3: Fit Growth Rates and Monthly Seasonal Indices
# ---------------------------------------------------------------------------

def fit_commodity_growth_rates(annual_df: pd.DataFrame, recent_df: pd.DataFrame = None) -> dict:
    combined = annual_df.copy()
    if recent_df is not None and not recent_df.empty:
        extra = recent_df[["year", "commodity", "ntkm_million"]].rename(
            columns={"ntkm_million": "value"}
        ).dropna()
        combined = pd.concat([combined, extra], ignore_index=True)

    growth_rates = {}
    for commodity, group in combined.groupby("commodity"):
        group = group.sort_values("year")
        years = group["year"].values.astype(float)
        values = np.clip(group["value"].values.astype(float), 1e-6, None)
        if len(years) < 5:
            growth_rates[commodity] = 0.03
            continue
        b, _ = np.polyfit(years, np.log(values), 1)
        growth_rates[commodity] = float(np.clip(b, -0.05, 0.15))
    return growth_rates

def fit_monthly_seasonal_index(monthly_df: pd.DataFrame) -> dict:
    seasonal = {}
    for commodity, group in monthly_df.groupby("commodity"):
        monthly_mean = group.groupby("month")["value"].mean()
        overall_mean = monthly_mean.mean()
        index = (monthly_mean / overall_mean).to_dict()
        seasonal[commodity] = {m: index.get(m, 1.0) for m in range(1, 13)}
    return seasonal

# ---------------------------------------------------------------------------
# Stage 3.5: Meaningful Hub-Degree & Gravity-Ranked Corridor Selection
# ---------------------------------------------------------------------------

def fetch_chennai_corridors_from_graph(n_corridors: int = 8) -> list:
    """
    Ranks candidate station pairs by topological gravity (hub degrees + distance span)
    instead of arbitrary node ID ordering.
    """
    CHENNAI_SHARE_OF_NATIONAL = 0.02

    if not DB_MODULE_AVAILABLE:
        print("  ⚠️ db.py not importable -- using fallback corridors.")
        return FALLBACK_CORRIDORS

    # Query 1: Extract all named stations with their degree of track connectivity
    hub_degree_query = """
        MATCH (s:Station)-[r]-()
        WHERE s.name IS NOT NULL AND NOT s.name STARTS WITH 'Node_'
        RETURN s.id, s.name, count(r) as degree, s.latitude, s.longitude
    """
    try:
        hub_rows = execute_cypher(
            hub_degree_query,
            cols=["id agtype", "name agtype", "deg agtype", "lat agtype", "lon agtype"]
        )
    except Exception as e:
        print(f"  ⚠️ Hub degree query failed ({e}) -- using fallback corridors.")
        return FALLBACK_CORRIDORS

    if not hub_rows:
        return FALLBACK_CORRIDORS

    # Parse stations into structured hub records
    station_hubs = {}
    for row in hub_rows:
        st_id = str(row[0]).strip('"')
        st_name = str(row[1]).strip('"')
        deg = int(row[2]) if row[2] is not None else 1
        lat = float(row[3]) if row[3] is not None else 0.0
        lon = float(row[4]) if row[4] is not None else 0.0
        station_hubs[st_id] = {
            "name": st_name, "degree": deg, "lat": lat, "lon": lon
        }

    # Haversine distance calculator for gravity spatial constraints
    def _dist_km(lat1, lon1, lat2, lon2):
        r = 6371.0
        dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        return r * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

    # Pairwise Gravity Scoring across active stations
    candidate_pairs = []
    hub_ids = list(station_hubs.keys())

    for i in range(len(hub_ids)):
        for j in range(i + 1, len(hub_ids)):
            u, v = hub_ids[i], hub_ids[j]
            s1, s2 = station_hubs[u], station_hubs[v]
            
            d_km = _dist_km(s1["lat"], s1["lon"], s2["lat"], s2["lon"])
            
            # Filter: Target realistic freight corridors (10 km to 120 km)
            if 10.0 <= d_km <= 120.0:
                # Gravity Formulation: Hub Connectivity Degree Product scaled by Corridor Span
                gravity_score = (math.sqrt(s1["degree"] * s2["degree"])) * math.log(1.0 + d_km)
                candidate_pairs.append((gravity_score, u, s1["name"], v, s2["name"], d_km))

    # Sort descending by topological gravity score
    candidate_pairs.sort(key=lambda x: x[0], reverse=True)

    if not candidate_pairs:
        return FALLBACK_CORRIDORS

    # National Commodity Weights from FY2023 actuals
    try:
        recent_df = load_recent_dataset()
        latest = recent_df[recent_df["year"] == recent_df["year"].max()]
        tot = latest["earnings_crore"].sum()
        commodity_shares = dict(zip(latest["commodity"], latest["earnings_crore"] / tot))
    except Exception:
        commodity_shares = {c: 0.11 for c in ["Coal", "Cement", "Foodgrains", "Fertilizers", "Container Service"]}

    commodities = list(commodity_shares.keys())
    weights = list(commodity_shares.values())

    selected_corridors = []
    print(f"📊 Top {n_corridors} Corridors Ranked by Hub Degree Gravity:")
    for score, s1_id, s1_name, s2_id, s2_name, d_km in candidate_pairs[:n_corridors]:
        comm = random.choices(commodities, weights=weights, k=1)[0]
        share = commodity_shares.get(comm, 0.1)
        corridor_weight = (share * CHENNAI_SHARE_OF_NATIONAL) / max(1, n_corridors)
        selected_corridors.append((s1_id, s2_id, comm, corridor_weight))
        print(f"  • {s1_name} <──({d_km:.1f} km, Score: {score:.2f})──> {s2_name} | Assigned: [{comm}]")

    return selected_corridors

# ---------------------------------------------------------------------------
# Stage 4: Synthesize Daily Time-Series Demand
# ---------------------------------------------------------------------------

def synthesize_corridor_series(
    corridors: list, growth_rates: dict, seasonal_index: dict, n_years: int = 3
) -> pd.DataFrame:
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
                "tonnage": tonnage, "is_weekend": d.dayofweek >= 5,
                "month": d.month, "disrupted": disruption != 1.0,
            })
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Stage 5: GRU Forecaster with Asymmetric Cost-Matrix Loss
# ---------------------------------------------------------------------------

class CostMatrixLoss(nn.Module):
    def __init__(self, underestimate_penalty: float = 4.0):
        super().__init__()
        self.underestimate_penalty = underestimate_penalty

    def forward(self, y_pred, y_actual):
        error = y_actual - y_pred
        weight = torch.where(error > 0, self.underestimate_penalty, 1.0)
        return torch.mean(weight * error ** 2)

class DemandGRU(nn.Module):
    def __init__(self, n_features: int = 4, hidden_size: int = 64, horizon: int = 14):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.head(h_n.squeeze(0))

@dataclass
class WindowConfig:
    lookback: int = 30
    horizon: int = 14

class CorridorDemandDataset(Dataset):
    def __init__(self, series_df: pd.DataFrame, corridor_key: tuple, cfg: WindowConfig):
        origin, dest, commodity = corridor_key
        sub = series_df[
            (series_df.origin == origin) & (series_df.dest == dest) & (series_df.commodity == commodity)
        ].sort_values("date").reset_index(drop=True)

        tonnage = sub["tonnage"].values.astype(np.float32)
        self.mean, self.std = tonnage.mean(), tonnage.std() + 1e-6
        tonnage_norm = (tonnage - self.mean) / self.std

        dow = sub["date"].dt.dayofweek.values.astype(np.float32) / 6.0
        month_sin = np.sin(2 * np.pi * sub["month"].values / 12).astype(np.float32)
        month_cos = np.cos(2 * np.pi * sub["month"].values / 12).astype(np.float32)

        features = np.stack([tonnage_norm, dow, month_sin, month_cos], axis=1)

        self.X, self.y = [], []
        for i in range(len(features) - cfg.lookback - cfg.horizon + 1):
            self.X.append(features[i:i + cfg.lookback])
            self.y.append(tonnage_norm[i + cfg.lookback:i + cfg.lookback + cfg.horizon])
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

    def denormalize(self, arr):
        return arr * self.std + self.mean

def train_corridor_model(series_df: pd.DataFrame, corridor_key: tuple, epochs: int = 15):
    cfg = WindowConfig()
    dataset = CorridorDemandDataset(series_df, corridor_key, cfg)
    if len(dataset) < 10:
        return None, None

    split = int(len(dataset) * 0.85)
    train_ds = torch.utils.data.Subset(dataset, range(0, split))
    val_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    model = DemandGRU(n_features=4, horizon=cfg.horizon)
    loss_fn = CostMatrixLoss(underestimate_penalty=4.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= max(1, len(train_ds))

        if epoch % 5 == 0 or epoch == epochs - 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += loss_fn(model(xb), yb).item() * xb.size(0)
            val_loss /= max(1, len(val_ds))
            print(f"    Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return model, dataset

# ---------------------------------------------------------------------------
# Stage 6: Module 3 Compatible Cargo Struct Generation
# ---------------------------------------------------------------------------

def to_module3_cargo_format(corridor_key: tuple, forecast_tonnage: np.ndarray, start_date: pd.Timestamp) -> list:
    origin, dest, commodity = corridor_key
    cargos = []
    for i, tonnage in enumerate(forecast_tonnage):
        due_date = start_date + pd.Timedelta(days=i + 1)
        cargos.append({
            "id": f"{origin}_{dest}_{commodity}_{due_date.strftime('%Y%m%d')}",
            "origin": origin,
            "destination": dest,
            "commodity": commodity,
            "weight": round(float(max(0.0, tonnage)), 2),
            "priority": 0.8 if commodity in ["Coal", "Petroleum, Oil and Lubricant"] else 0.5,
            "due_date": due_date,
        })
    return cargos

# ---------------------------------------------------------------------------
# Main Pipeline Entrypoint
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("STAGE 1-3: Loading IR Historical Statistics & Fitting Priors")
    print("=" * 70)
    annual_df = load_annual_dataset()
    monthly_df = load_monthly_dataset()
    recent_df = load_recent_dataset()

    growth_rates = fit_commodity_growth_rates(annual_df, recent_df)
    seasonal_index = fit_monthly_seasonal_index(monthly_df)

    print("\n" + "=" * 70)
    print("STAGE 3.5: Fetching Gravity-Ranked Corridors from Apache AGE Graph")
    print("=" * 70)
    corridors = fetch_chennai_corridors_from_graph(n_corridors=8)

    print("\n" + "=" * 70)
    print("STAGE 4: Synthesizing Multi-Year Daily Demand Series")
    print("=" * 70)
    series_df = synthesize_corridor_series(corridors, growth_rates, seasonal_index, n_years=3)
    print(f"Generated {len(series_df)} records across {len(corridors)} corridors.")

    print("\n" + "=" * 70)
    print("STAGE 5-6: Training Forecasters & Emitting Module 3 Cargo Intent")
    print("=" * 70)
    all_cargos = []
    for origin, dest, commodity, _ in corridors:
        key = (origin, dest, commodity)
        print(f"\n▶ Training Forecaster for: {origin} -> {dest} [{commodity}]")
        model, dataset = train_corridor_model(series_df, key, epochs=15)
        if model is None:
            continue
        
        # Predict 14-day demand window
        model.eval()
        last_window = torch.from_numpy(dataset.X[-1:]).float()
        with torch.no_grad():
            pred_norm = model(last_window).numpy().flatten()
        forecast = dataset.denormalize(pred_norm)

        last_date = series_df["date"].max()
        cargos = to_module3_cargo_format(key, forecast, last_date)
        all_cargos.extend(cargos)
        print(f"    Predicted 14-Day Demand (Tons/Day): {np.round(forecast, 1).tolist()}")

    print("\n" + "=" * 70)
    print(f"✅ COMPLETE: Emitted {len(all_cargos)} cargo allocations ready for Module 3.")
    if all_cargos:
        print("Sample Cargo Payload for Optimization Solver:", all_cargos[0])
    print("=" * 70)

if __name__ == "__main__":
    main()