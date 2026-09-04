"""
preprocess_knowair.py
=======================
把 KnowAir-V2 的 netCDF 資料轉成跟原本 aqi/preprocess.py 完全相容的格式，
這樣現有的 train_dcrnn.py / train_agcrn.py / train_gts.py / train_airphynet.py
不需要改任何一行程式碼，只要把 --data 指向這裡輸出的資料夾即可。

輸入（KnowAir-V2）：
    dataset_yrd.nc      長江三角洲（Yangtze River Delta）
    dataset_bthsa.nc     京津冀及周邊（Beijing-Tianjin-Hebei & Surrounding Area）
    stations_yrd.csv
    stations_bthsa.csv

實際變數（已依你本機執行結果確認）：
    污染物: PM2.5, O3   (KnowAir-V2 只有這兩種污染物，不像原版 KnowAir 有 8 種)
    氣象:   t2m, d2m, sp, tp, blh, msdwswrf
    風場:   u100, v100

輸出（跟原本 preprocess.py schema 一致）：
    features.npy / features_raw.npy / adj_mx.npy / adj_wind.npy / dist_mx.npy
    stations.csv / timestamps.csv / scalers.pkl / split.pkl

Usage:
    python preprocess_knowair.py --region yrd   --raw_dir "../KnowAir-V2/data"
    python preprocess_knowair.py --region bthsa --raw_dir "../KnowAir-V2/data"
"""

import argparse
import os
import pickle
from math import atan2, cos, radians, sin, sqrt

import numpy as np
import pandas as pd
import xarray as xr

# ── 變數名稱對照表（依你本機實際 .nc 檔案內容確認過）───────────────────────
VAR_MAP_CANDIDATES = {
    "pm2.5":       ["PM2.5", "pm2.5", "pm25", "pm2p5"],
    "o3":          ["O3", "o3"],
    "t2m":         ["t2m"],
    "d2m":         ["d2m"],
    "sp":          ["sp"],
    "tp":          ["tp"],
    "blh":         ["blh"],
    "msdwswrf":    ["msdwswrf"],
    "windspeed_u": ["u100", "u_component_of_wind+950", "u10", "u"],
    "windspeed_v": ["v100", "v_component_of_wind+950", "v10", "v"],
}

# KnowAir-V2 實際只有 PM2.5 / O3 兩種污染物（原始 KnowAir 論文的其餘 6 種污染物
# 在 V2 中被移除，改為提供更豐富的氣象變數：t2m, d2m, sp, tp, blh, msdwswrf）
RAW_FEATURE_ORDER = ["pm2.5", "o3", "t2m", "d2m", "sp", "tp", "blh", "msdwswrf"]
OUTPUT_FEATURE_NAMES = ["pm2.5", "o3", "t2m", "d2m", "sp", "tp", "blh", "msdwswrf", "wind_u", "wind_v"]

SPLIT_RATIOS = (0.7, 0.1, 0.2)
ADJ_THRESHOLD = 0.1


# ── geometry ─────────────────────────────────────────────────────────────────

def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    phi1, phi2 = radians(lat1), radians(lat2)
    dphi = radians(lat2 - lat1)
    dlam = radians(lon2 - lon1)
    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def build_distance_matrix(stations: pd.DataFrame) -> np.ndarray:
    lons = stations["longitude"].values
    lats = stations["latitude"].values
    n = len(stations)
    dist = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(lons[i], lats[i], lons[j], lats[j])
            dist[i, j] = dist[j, i] = d
    return dist


def dist_to_adj(dist_mx, sigma=None, threshold=ADJ_THRESHOLD):
    if sigma is None:
        sigma = float(dist_mx[dist_mx > 0].std())
    adj = np.exp(-(dist_mx ** 2) / (sigma ** 2)).astype(np.float32)
    np.fill_diagonal(adj, 0.0)
    adj[adj < threshold] = 0.0
    return adj


def build_wind_adj(dist_mx, stations, mean_wind_u, mean_wind_v,
                   sigma=None, threshold=ADJ_THRESHOLD):
    lons = stations["longitude"].values
    lats = stations["latitude"].values
    n = len(stations)
    adj_geo = dist_to_adj(dist_mx, sigma=sigma, threshold=0.0)

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


# ── netCDF 讀取與變數自動偵測 ──────────────────────────────────────────────

def detect_variable(ds: xr.Dataset, candidates: list) -> str:
    available = set(ds.data_vars.keys())
    for c in candidates:
        if c in available:
            return c
    lower_map = {v.lower(): v for v in available}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def load_knowair_nc(nc_path: str, stations_csv: str):
    print(f"開啟 {nc_path} ...")
    ds = xr.open_dataset(nc_path)

    print(f"\n{'='*60}")
    print(f"  .nc 檔案內容診斷")
    print(f"{'='*60}")
    print(f"維度 (dims): {dict(ds.sizes)}")
    print(f"座標 (coords): {list(ds.coords.keys())}")
    print(f"變數 (data_vars): {list(ds.data_vars.keys())}")
    print(f"{'='*60}\n")

    stations = pd.read_csv(stations_csv)
    print(f"stations csv 欄位: {stations.columns.tolist()}")
    print(f"stations 筆數: {len(stations)}")

    time_dim = None
    for cand in ["time", "datetime", "date"]:
        if cand in ds.dims:
            time_dim = cand
            break
    if time_dim is None:
        raise ValueError(f"找不到時間維度，實際 dims 是: {dict(ds.sizes)}")

    station_dim = None
    for cand in ["station", "site", "node", "city"]:
        if cand in ds.dims:
            station_dim = cand
            break
    if station_dim is None:
        raise ValueError(f"找不到測站維度，實際 dims 是: {dict(ds.sizes)}")

    print(f"偵測到時間維度: '{time_dim}'，測站維度: '{station_dim}'")

    timestamps = pd.to_datetime(ds[time_dim].values)
    T = len(timestamps)
    N = len(ds[station_dim])
    print(f"T={T} 個時間點, N={N} 個測站")

    return ds, stations, timestamps, time_dim, station_dim


def preprocess(nc_path: str, stations_csv: str, output_dir: str, region_name: str):
    ds, stations, timestamps, time_dim, station_dim = load_knowair_nc(nc_path, stations_csv)

    N = len(stations)
    T = len(timestamps)

    pivot_arrays = []
    resolved_names = {}

    for feat in RAW_FEATURE_ORDER:
        var_name = detect_variable(ds, VAR_MAP_CANDIDATES.get(feat, [feat]))
        if var_name is None:
            print(f"  ⚠ 找不到 '{feat}' 對應的變數，填 0")
            arr = np.zeros((T, N), dtype=np.float32)
        else:
            arr = ds[var_name].values.astype(np.float32)
            resolved_names[feat] = var_name
        pivot_arrays.append(arr)

    u_name = detect_variable(ds, VAR_MAP_CANDIDATES["windspeed_u"])
    v_name = detect_variable(ds, VAR_MAP_CANDIDATES["windspeed_v"])
    if u_name is None or v_name is None:
        print(f"  ⚠ 找不到風場 u/v 分量，填 0")
        wind_u = np.zeros((T, N), dtype=np.float32)
        wind_v = np.zeros((T, N), dtype=np.float32)
    else:
        wind_u = ds[u_name].values.astype(np.float32)
        wind_v = ds[v_name].values.astype(np.float32)
        resolved_names["wind_u"] = u_name
        resolved_names["wind_v"] = v_name

    print(f"\n成功對應的變數: {resolved_names}\n")

    pivot_arrays.append(wind_u)
    pivot_arrays.append(wind_v)
    features_raw = np.stack(pivot_arrays, axis=-1)   # [T, N, F=10]

    F = features_raw.shape[-1]
    for f in range(F):
        col = pd.DataFrame(features_raw[:, :, f])
        col = col.interpolate(method="linear", axis=0, limit_direction="both")
        col = col.ffill().bfill()
        features_raw[:, :, f] = col.values.astype(np.float32)
    features_raw = np.nan_to_num(features_raw, nan=0.0).astype(np.float32)

    flat = features_raw.reshape(-1, F)
    mean = np.nanmean(flat, axis=0).astype(np.float32)
    std = np.nanstd(flat, axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    features_norm = ((features_raw - mean) / std).astype(np.float32)

    if "sitename" not in stations.columns:
        name_col = next((c for c in stations.columns if "name" in c.lower()), stations.columns[0])
        stations = stations.rename(columns={name_col: "sitename"})
    if "county" not in stations.columns:
        city_col = next((c for c in stations.columns if c.lower() == "city"), None)
        if city_col:
            stations = stations.rename(columns={city_col: "county"})
        else:
            stations["county"] = region_name
    lon_col = next((c for c in stations.columns if "lon" in c.lower()), None)
    lat_col = next((c for c in stations.columns if "lat" in c.lower()), None)
    if lon_col != "longitude":
        stations = stations.rename(columns={lon_col: "longitude"})
    if lat_col != "latitude":
        stations = stations.rename(columns={lat_col: "latitude"})

    print("建構鄰接矩陣 ...")
    dist_mx = build_distance_matrix(stations)
    adj_mx = dist_to_adj(dist_mx)

    mean_wind_u = features_raw[..., -2].mean(axis=0)
    mean_wind_v = features_raw[..., -1].mean(axis=0)
    adj_wind = build_wind_adj(dist_mx, stations, mean_wind_u, mean_wind_v)

    train_end = int(T * SPLIT_RATIOS[0])
    val_end = int(T * (SPLIT_RATIOS[0] + SPLIT_RATIOS[1]))
    split_indices = {
        "train": (0, train_end),
        "val":   (train_end, val_end),
        "test":  (val_end, T),
        "train_dates": (str(timestamps[0]), str(timestamps[train_end - 1])),
        "val_dates":   (str(timestamps[train_end]), str(timestamps[val_end - 1])),
        "test_dates":  (str(timestamps[val_end]), str(timestamps[-1])),
    }

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "features.npy"), features_norm)
    np.save(os.path.join(output_dir, "features_raw.npy"), features_raw)
    np.save(os.path.join(output_dir, "adj_mx.npy"), adj_mx)
    np.save(os.path.join(output_dir, "adj_wind.npy"), adj_wind)
    np.save(os.path.join(output_dir, "dist_mx.npy"), dist_mx)
    stations.to_csv(os.path.join(output_dir, "stations.csv"), index=False)
    pd.DataFrame({"timestamp": timestamps}).to_csv(
        os.path.join(output_dir, "timestamps.csv"), index=False)

    scalers = {"mean": mean, "std": std, "feature_names": OUTPUT_FEATURE_NAMES}
    with open(os.path.join(output_dir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    with open(os.path.join(output_dir, "split.pkl"), "wb") as f:
        pickle.dump(split_indices, f)

    print(f"\n{'─'*55}")
    print(f"  region:           {region_name}")
    print(f"  features.npy:     {features_norm.shape}")
    print(f"  adj_mx.npy:       {adj_mx.shape}  edges={int((adj_mx>0).sum())}")
    print(f"  adj_wind.npy:     {adj_wind.shape}  edges={int((adj_wind>0).sum())}")
    print(f"  Train  [{train_end:6d} steps]: {split_indices['train_dates'][0]} → {split_indices['train_dates'][1]}")
    print(f"  Val    [{val_end-train_end:6d} steps]: {split_indices['val_dates'][0]} → {split_indices['val_dates'][1]}")
    print(f"  Test   [{T-val_end:6d} steps]: {split_indices['test_dates'][0]} → {split_indices['test_dates'][1]}")
    print(f"  輸出至 → {output_dir}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--region", required=True, choices=["yrd", "bthsa"])
    p.add_argument("--raw_dir", default="raw")
    p.add_argument("--output_dir", default=None)
    args = p.parse_args()

    nc_file = os.path.join(args.raw_dir, f"dataset_{args.region}.nc")
    csv_file = os.path.join(args.raw_dir, f"stations_{args.region}.csv")
    out_dir = args.output_dir or f"data/processed_{args.region}"

    preprocess(nc_file, csv_file, out_dir, args.region)