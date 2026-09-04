"""
Shared preprocessing for PM2.5-GNN, AirPhyNet, and STGNP.

Reads raw Taiwan EPA AQI CSV and produces:
  features.npy      [T, N, F]  Z-score normalized
  features_raw.npy  [T, N, F]  physical units (pm2.5 in µg/m³, wind in m/s)
  adj_mx.npy        [N, N]     Gaussian-kernel geographic adjacency
  adj_wind.npy      [N, N]     Wind-aware directed adjacency (for PM2.5-GNN)
  dist_mx.npy       [N, N]     Pairwise Haversine distances (km)
  stations.csv                 Station metadata (name, county, lon, lat, siteid)
  timestamps.csv               UTC+8 hourly timestamps
  scalers.pkl                  {mean, std, feature_names} for inverse transform
  split.pkl                    {train, val, test} -> (start, end) index tuples

Feature order (F=10):
  0  pm2.5     µg/m³
  1  pm10      µg/m³
  2  so2       ppb
  3  co        ppm
  4  o3        ppb
  5  no2       ppb
  6  nox       ppb
  7  no        ppb
  8  wind_u    m/s  (eastward; derived from windspeed × winddirec)
  9  wind_v    m/s  (northward; derived from windspeed × winddirec)

Wind convention: Taiwan EPA reports direction the wind comes FROM (degrees CW from north).
  U = -speed × sin(dir_rad)
  V = -speed × cos(dir_rad)
"""

import os
import pickle
from math import atan2, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────

RAW_FEATURE_COLS = [
    "pm2.5", "pm10", "so2", "co", "o3", "no2", "nox", "no",
    "windspeed", "winddirec",
]
OUTPUT_FEATURE_NAMES = [
    "pm2.5", "pm10", "so2", "co", "o3", "no2", "nox", "no",
    "wind_u", "wind_v",
]

# Gaussian kernel parameters for geographic adjacency
DIST_SIGMA_KM = None   # None → auto (std of all pairwise distances)
ADJ_THRESHOLD = 0.1    # zero-out edges below this weight

# Train / val / test split ratios (chronological)
SPLIT_RATIOS = (0.7, 0.1, 0.2)

# Physical validity bounds for each raw feature.
# Values outside [lo, hi] are replaced with NaN and filled by interpolation.
# Small negatives due to sensor noise are clipped to lo (typically 0).
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "pm2.5":     (0.0,  500.0),
    "pm10":      (0.0,  800.0),
    "so2":       (0.0,  200.0),
    "co":        (0.0,   50.0),
    "o3":        (0.0,  200.0),
    "no2":       (0.0,  300.0),
    "nox":       (0.0,  500.0),
    "no":        (0.0,  500.0),
    "windspeed": (0.0,   50.0),
    "winddirec": (0.0,  360.0),
}


# ── geometry ─────────────────────────────────────────────────────────────────

def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def build_distance_matrix(stations: pd.DataFrame) -> np.ndarray:
    """Pairwise Haversine distance matrix [N, N] in km."""
    lons = stations["longitude"].values
    lats = stations["latitude"].values
    n = len(stations)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(lons[i], lats[i], lons[j], lats[j])
            dist[i, j] = dist[j, i] = d
    return dist


def dist_to_adj(dist_mx: np.ndarray, sigma: float | None = None,
                threshold: float = ADJ_THRESHOLD) -> np.ndarray:
    """
    Symmetric Gaussian-kernel adjacency matrix (no self-loops).
    adj[i,j] = exp(-d²/σ²), zeroed below threshold.
    """
    if sigma is None:
        sigma = float(dist_mx[dist_mx > 0].std())
    adj = np.exp(-(dist_mx ** 2) / (sigma ** 2)).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    adj[adj < threshold] = 0.0
    return adj


def build_wind_adj(dist_mx: np.ndarray, stations: pd.DataFrame,
                   mean_wind_u: np.ndarray, mean_wind_v: np.ndarray,
                   sigma: float | None = None,
                   threshold: float = ADJ_THRESHOLD) -> np.ndarray:
    """
    Directed wind-aware adjacency for PM2.5-GNN domain knowledge.

    Edge weight adj[i, j] amplifies the geographic weight when the time-averaged
    wind at station i blows towards station j:
        adj_wind[i,j] = adj_geo[i,j] × max(0, cos(α))
    where α = angle between wind vector at i and the vector from i→j.

    A directed graph (adj_wind is asymmetric) is returned.
    """
    lons = stations["longitude"].values
    lats = stations["latitude"].values
    n = len(stations)

    adj_geo = dist_to_adj(dist_mx, sigma=sigma, threshold=0.0)

    # Unit direction vector from i to j (in lon/lat space, approximate)
    adj_wind = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        wind_mag = np.sqrt(mean_wind_u[i] ** 2 + mean_wind_v[i] ** 2)
        if wind_mag < 1e-6:
            adj_wind[i] = adj_geo[i]
            continue
        wu = mean_wind_u[i] / wind_mag
        wv = mean_wind_v[i] / wind_mag
        for j in range(n):
            if i == j:
                continue
            dx = lons[j] - lons[i]
            dy = lats[j] - lats[i]
            d = sqrt(dx ** 2 + dy ** 2)
            if d < 1e-9:
                continue
            cos_alpha = (wu * dx + wv * dy) / d
            adj_wind[i, j] = adj_geo[i, j] * max(0.0, cos_alpha)

    adj_wind[adj_wind < threshold] = 0.0
    return adj_wind


# ── main ─────────────────────────────────────────────────────────────────────

def preprocess(raw_path: str, output_dir: str) -> None:
    print(f"Reading {raw_path} ...")
    df = pd.read_csv(raw_path)
    df["datacreationdate"] = pd.to_datetime(df["datacreationdate"])

    # Coerce '-' and other non-numeric sentinels to NaN
    for col in RAW_FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply physical bounds: clip small sensor-noise negatives to 0,
    # then set values outside [lo, hi] to NaN (filled later by interpolation).
    for col, (lo, hi) in PHYSICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        v = df[col]
        df[col] = v.where(v >= lo, 0.0)       # small negatives → 0
        df[col] = df[col].where(df[col] <= hi, float("nan"))  # extreme highs → NaN

    # Station metadata with deterministic ordering.
    # Prefer siteid sort; fall back to county+sitename when siteid is mostly absent.
    sta_cols = ["sitename", "county", "longitude", "latitude"]
    if "siteid" in df.columns:
        sta_cols.append("siteid")
    stations = df[sta_cols].drop_duplicates("sitename")
    if "siteid" in stations.columns and stations["siteid"].notna().mean() > 0.5:
        stations = stations.sort_values("siteid")
    else:
        stations = stations.sort_values(["county", "sitename"])
    stations = stations.reset_index(drop=True)
    site_order = stations["sitename"].tolist()
    N = len(stations)
    print(f"Stations: {N}")

    # Build a unified hourly timestamp index (union of all feature timestamps)
    # so every pivot table has the same shape regardless of per-feature gaps.
    all_ts = pd.DatetimeIndex(sorted(df["datacreationdate"].unique()))

    # Pivot each raw feature → [T, N], fill missing values
    pivot_arrays = []
    for col in RAW_FEATURE_COLS:
        piv = df.pivot_table(
            index="datacreationdate", columns="sitename",
            values=col, aggfunc="first",
        )
        # Align to the full timestamp index and all stations
        piv = piv.reindex(index=all_ts, columns=site_order)
        # Linear interpolation along time axis, then boundary fill
        piv = piv.interpolate(method="linear", axis=0, limit_direction="both")
        piv = piv.ffill().bfill()
        pivot_arrays.append(piv.values.astype(np.float32))

    timestamps = all_ts  # DatetimeIndex (unified across all features)
    T = len(timestamps)
    F = len(RAW_FEATURE_COLS)
    features_raw = np.stack(pivot_arrays, axis=-1)  # [T, N, F]
    print(f"Timestamps: {T}  [{timestamps[0]} → {timestamps[-1]}]")

    # Convert windspeed + winddirec → wind_u, wind_v (meteorological convention)
    ws_idx = RAW_FEATURE_COLS.index("windspeed")
    wd_idx = RAW_FEATURE_COLS.index("winddirec")
    ws = features_raw[..., ws_idx]
    wd_rad = np.radians(features_raw[..., wd_idx])
    features_raw[..., ws_idx] = (-ws * np.sin(wd_rad)).astype(np.float32)  # wind_u
    features_raw[..., wd_idx] = (-ws * np.cos(wd_rad)).astype(np.float32)  # wind_v

    # Stations with entirely missing wind data → treat as calm (0 m/s)
    features_raw = np.where(np.isnan(features_raw), 0.0, features_raw).astype(np.float32)

    # Z-score normalization: per feature, pooled across T and N
    # Use nanmean/nanstd to be safe even if residual NaNs exist after filling
    flat = features_raw.reshape(-1, F)
    mean = np.nanmean(flat, axis=0).astype(np.float32)
    std = np.nanstd(flat, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0  # avoid division by zero for constant features
    features_norm = ((features_raw - mean) / std).astype(np.float32)

    # Adjacency matrices
    print("Building adjacency matrices ...")
    dist_mx = build_distance_matrix(stations)
    adj_mx = dist_to_adj(dist_mx, sigma=DIST_SIGMA_KM)

    mean_wind_u = features_raw[..., ws_idx].mean(axis=0)  # [N]
    mean_wind_v = features_raw[..., wd_idx].mean(axis=0)  # [N]
    adj_wind = build_wind_adj(dist_mx, stations, mean_wind_u, mean_wind_v)

    # Train / val / test split (chronological)
    train_end = int(T * SPLIT_RATIOS[0])
    val_end = int(T * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))
    split_indices = {
        "train":      (0, train_end),
        "val":        (train_end, val_end),
        "test":       (val_end, T),
        "train_dates": (str(timestamps[0]),          str(timestamps[train_end - 1])),
        "val_dates":   (str(timestamps[train_end]),  str(timestamps[val_end - 1])),
        "test_dates":  (str(timestamps[val_end]),    str(timestamps[-1])),
    }

    # Save
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "features.npy"), features_norm)
    np.save(os.path.join(output_dir, "features_raw.npy"), features_raw)
    np.save(os.path.join(output_dir, "adj_mx.npy"), adj_mx)
    np.save(os.path.join(output_dir, "adj_wind.npy"), adj_wind)
    np.save(os.path.join(output_dir, "dist_mx.npy"), dist_mx)
    stations.to_csv(os.path.join(output_dir, "stations.csv"), index=False)
    pd.DataFrame({"timestamp": timestamps}).to_csv(
        os.path.join(output_dir, "timestamps.csv"), index=False
    )
    scalers = {"mean": mean, "std": std, "feature_names": OUTPUT_FEATURE_NAMES}
    with open(os.path.join(output_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    with open(os.path.join(output_dir, "split.pkl"), "wb") as f:
        pickle.dump(split_indices, f)

    # Summary
    print()
    print("─" * 50)
    print(f"  features.npy      {features_norm.shape}  (normalized)")
    print(f"  features_raw.npy  {features_raw.shape}  (physical units)")
    print(f"  adj_mx.npy        {adj_mx.shape}  edges={int((adj_mx > 0).sum())}")
    print(f"  adj_wind.npy      {adj_wind.shape}  edges={int((adj_wind > 0).sum())}")
    print(f"  dist_mx.npy       {dist_mx.shape}")
    print(f"  Feature names:    {OUTPUT_FEATURE_NAMES}")
    print(f"  Train  [{train_end:6d} steps]:  {split_indices['train_dates'][0]} → {split_indices['train_dates'][1]}")
    print(f"  Val    [{val_end-train_end:6d} steps]:  {split_indices['val_dates'][0]} → {split_indices['val_dates'][1]}")
    print(f"  Test   [{T-val_end:6d} steps]:  {split_indices['test_dates'][0]} → {split_indices['test_dates'][1]}")
    print(f"  Output → {output_dir}")
    print("─" * 50)


if __name__ == "__main__":
    import sys

    raw = sys.argv[1] if len(sys.argv) > 1 else "airphynet/data/aqi_all.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "data/processed"
    preprocess(raw, out)
