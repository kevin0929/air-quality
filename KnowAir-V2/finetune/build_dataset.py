"""
把 fetch_era5.py / fetch_cnemc.py 抓下來的原始資料組裝成最終資料集:
  1. ERA5 網格內插到每個站點座標
  2. 依 200km 地理距離閾值建立站點鄰接圖
  3. 缺值補值(短缺值時間內插、長缺值空間鄰站平均)
  4. 輸出成 (time, station, feature) 的 dataset.npy,搭配 metadata

用法:
    python build_dataset.py --config config.yaml
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EARTH_RADIUS_KM = 6371.0


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def haversine_km(lat1, lon1, lat2, lon2):
    """向量化計算兩組經緯度之間的地理距離(公里),輸入輸出皆為 numpy array。"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return EARTH_RADIUS_KM * c


def build_graph(stations: pd.DataFrame, threshold_km: float, edge_weight: str):
    """建立 N x N 鄰接矩陣。stations 必須含 lat, lon 欄,列的順序就是圖節點的順序。"""
    n = len(stations)
    lat = stations["lat"].to_numpy()
    lon = stations["lon"].to_numpy()

    dist = np.zeros((n, n))
    for i in range(n):
        dist[i, :] = haversine_km(lat[i], lon[i], lat, lon)

    within = dist <= threshold_km
    np.fill_diagonal(within, False)  # 不留自環,GNN 訓練程式碼通常自己處理自環

    if edge_weight == "binary":
        adj = within.astype(np.float32)
    elif edge_weight == "inverse_distance":
        adj = np.zeros((n, n), dtype=np.float32)
        adj[within] = 1.0 / np.maximum(dist[within], 1e-3)
    else:
        raise ValueError(f"未知的 edge_weight: {edge_weight}")

    isolated = np.where(within.sum(axis=1) == 0)[0]
    if len(isolated) > 0:
        names = stations.iloc[isolated]["station_code"].tolist()
        print(
            f"[warn] 有 {len(isolated)} 個站點在 {threshold_km}km 閾值下沒有任何鄰居: {names}"
            f"(圖會是不連通的,訓練時部分模型可能需要至少一條邊,請確認是否要調高閾值)"
        )
    return adj, dist


def interpolate_era5_to_stations(cfg: dict, stations: pd.DataFrame, time_index: pd.DatetimeIndex):
    """讀取 raw_era5_dir 底下每個月份資料夾裡的 nc 檔,把每個變數內插到站點座標,
    回傳 dict[short_name] -> (time, station) array。

    需要 xarray + netCDF4,實際檔案要先跑過 fetch_era5.py 才會存在。CDS 有時會把同一個月
    拆成 instant/accum 兩個 nc 檔(見 fetch_era5.py 開頭說明),這裡用 xr.merge 合併,
    不用 open_mfdataset,避免額外依賴 dask。
    """
    import xarray as xr

    era5_dir = Path(cfg["paths"]["raw_era5_dir"])
    nc_files = sorted(era5_dir.glob("*/*.nc"))
    if not nc_files:
        raise SystemExit(f"{era5_dir} 底下沒有任何 */*.nc 檔,請先跑 fetch_era5.py")

    print(f"[info] 讀取 {len(nc_files)} 個 ERA5 nc 檔案")
    datasets = [xr.open_dataset(f) for f in nc_files]
    # 注意:compat 必須用 "no_conflicts",不能用 "override"——
    # override 會在座標做 outer join 展開後,對每個變數只保留「第一個」資料集裡的值,
    # 其餘沒被選中的資料集裡的值全部被丟棄變成 NaN。這曾經導致合併 12 個月份時
    # 只有第一個月的資料是真的,其餘 11 個月全部變成缺值(對應到之前看到的 91.5% 缺值率,
    # 即 11/12 個月的資料在合併時被悄悄丟掉)。no_conflicts 才會正確地在不重疊的座標範圍
    # (例如不同月份)分別保留各自資料集的真實數值。
    ds = xr.merge(datasets, compat="no_conflicts", join="outer")

    # 新版 CDS 匯出常用 valid_time 而非 time,兩種都相容
    if "valid_time" in ds.coords and "time" not in ds.coords:
        ds = ds.rename({"valid_time": "time"})

    method = cfg["era5"]["interpolation_method"]
    valid_methods = {"linear", "nearest", "slinear", "cubic", "quintic", "pchip"}
    if method not in valid_methods:
        raise SystemExit(
            f"config.yaml 的 era5.interpolation_method = '{method}' 不是 xarray/scipy 認得的方法名稱。"
            f"可用選項: {sorted(valid_methods)}。"
            f"注意:2D 網格上的雙線性內插,參數名稱要寫成 'linear',不是 'bilinear'。"
        )
    lat_da = xr.DataArray(stations["lat"].to_numpy(), dims="station")
    lon_da = xr.DataArray(stations["lon"].to_numpy(), dims="station")

    out = {}
    for var in cfg["era5"]["variables"]:
        short = var["short_name"]
        candidates = [short] + list(var.get("aliases", []))
        matched = next((name for name in candidates if name in ds.variables), None)
        if matched is None:
            print(
                f"[warn] ERA5 檔案裡找不到變數 {short}"
                f"{f'(也試過候選名稱 {candidates[1:]})' if len(candidates) > 1 else ''}"
                f",請確認下載內容或調整這裡的變數名稱對應"
            )
            continue
        if matched != short:
            print(f"[info] {short} 用候選名稱 '{matched}' 找到了(ECMWF 端改過這個變數的簡稱)")
        da = ds[matched].interp(latitude=lat_da, longitude=lon_da, method=method)
        da = da.reindex(time=time_index, method="nearest", tolerance=pd.Timedelta("1h"))
        out[short] = da.transpose("time", "station").to_numpy()

    return out


def pivot_cnemc(cfg: dict, stations: pd.DataFrame, time_index: pd.DatetimeIndex):
    """讀取 fetch_cnemc.py 輸出的 parquet,轉成 dict[pollutant] -> (time, station) array,並對齊時區到 UTC。"""
    cnemc_file = Path(cfg["paths"]["raw_cnemc_dir"]) / "cnemc_combined.parquet"
    if not cnemc_file.exists():
        raise SystemExit(f"找不到 {cnemc_file},請先跑 fetch_cnemc.py")

    df = pd.read_parquet(cnemc_file)
    # CNEMC 發布時間是北京時間(UTC+8),轉成 UTC 跟 ERA5 對齊
    df["datetime_utc"] = df["datetime_utc8"] - pd.Timedelta(hours=8)

    station_order = stations["station_code"].astype(str).tolist()
    out = {}
    for pollutant in cfg["cnemc"]["pollutants"]:
        sub = df[df["pollutant"] == pollutant]
        wide = sub.pivot_table(index="datetime_utc", columns="station_code", values="value", aggfunc="mean")
        wide = wide.reindex(index=time_index, columns=station_order)
        out[pollutant] = wide.to_numpy()
    return out


def report_missingness(all_arrays: dict, stations: pd.DataFrame, out_dir: Path, warn_threshold: float = 0.5):
    """補值之前先算每個站點、每個 feature 的缺值比例,存成報表,並對缺很多的站點提出警告。

    warn_threshold: 缺值比例超過這個門檻(預設 50%)就警告——這種站點補值後幾乎全是
    鄰居合成值,拿去當 fine-tune 的訓練/驗證目標會失真,建議考慮從 stations.csv 移除。
    """
    rows = []
    for i, code in enumerate(stations["station_code"]):
        row = {"station_code": code}
        for name, arr in all_arrays.items():
            n = arr.shape[0]
            missing = np.isnan(arr[:, i]).sum()
            row[f"{name}_missing_pct"] = missing / n
        rows.append(row)
    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "station_completeness.csv", index=False)

    pct_cols = [c for c in report.columns if c.endswith("_missing_pct")]
    worst = report[pct_cols].max(axis=1)
    bad = report[worst >= warn_threshold]
    if len(bad) > 0:
        print(f"[warn] 有 {len(bad)} 個站點至少一個 feature 缺值比例 >= {warn_threshold*100:.0f}%,"
              f"補值後這些站點大部分數值會是鄰居合成值,建議檢查是否要從 stations.csv 移除:")
        for _, r in bad.iterrows():
            worst_feat = max(pct_cols, key=lambda c: r[c])
            print(f"    {r['station_code']}: 最嚴重 {worst_feat} = {r[worst_feat]*100:.1f}%")
    return report


def impute(arr: np.ndarray, adj: np.ndarray, max_gap_steps: int, k_neighbors: int):
    """arr shape = (time, station)。三層補值策略:
    1. 短缺值:逐站沿時間線性內插,限制最大連續補值長度
    2. 長缺值:用圖上最近的 k 個鄰居在同一時間點的平均值補
    3. 最後保險:如果鄰居當下也剛好全缺(例如區域性回報中斷),
       改用「當下所有站點」的平均值補,確保輸出不會殘留任何 NaN。
       這一層動用的範圍已經超出地理鄰近性,補進去的值品質更低,
       只是為了不讓 NaN 流進訓練流程,不代表這些時間點的資料是可信的。
    """
    df = pd.DataFrame(arr)

    filled = df.interpolate(method="linear", axis=0, limit=max_gap_steps, limit_area="inside")

    still_missing = filled.isna()
    if still_missing.to_numpy().any():
        filled_vals = filled.to_numpy().copy()
        for station_idx in range(arr.shape[1]):
            missing_times = np.where(still_missing.iloc[:, station_idx])[0]
            if len(missing_times) == 0:
                continue
            neighbor_order = np.argsort(-adj[station_idx])
            neighbor_idx = [j for j in neighbor_order if adj[station_idx, j] > 0][:k_neighbors]
            if not neighbor_idx:
                continue
            neighbor_vals = filled_vals[np.ix_(missing_times, neighbor_idx)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                row_means = np.nanmean(neighbor_vals, axis=1)
            filled_vals[missing_times, station_idx] = row_means
        filled = pd.DataFrame(filled_vals)

    # 第三層保險:鄰居當下也全缺的情況,退而求其次用全站當下平均補
    remaining = filled.isna()
    n_remaining = int(remaining.to_numpy().sum())
    if n_remaining > 0:
        vals = filled.to_numpy().copy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            global_mean_per_time = np.nanmean(vals, axis=1)  # shape (time,)
        rows, cols = np.where(np.isnan(vals))
        vals[rows, cols] = global_mean_per_time[rows]
        filled = pd.DataFrame(vals)
        print(f"[warn] 有 {n_remaining} 個缺值連鄰居當下都缺資料,已退一步用「全站當下平均」補,"
              f"這些點的資料可信度更低,建議之後檢查是否集中在特定時段/區域(可能代表區域性回報中斷)")

    # 極端保底:如果連「全站當下平均」都是 NaN(代表那個時間點所有站點全部缺資料),
    # 用時間軸前後值填補,避免任何 NaN 流出這個函式
    if filled.isna().to_numpy().any():
        n_extreme = int(filled.isna().to_numpy().sum())
        filled = filled.ffill().bfill()
        print(f"[warn] 有 {n_extreme} 個缺值是「全站同時缺資料」的極端情況,已用前後時間點填補,"
              f"強烈建議檢查這段時間是不是資料源本身有大規模中斷")

    return filled.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    stations_csv = cfg["paths"]["stations_csv"]
    if not Path(stations_csv).exists():
        raise SystemExit(f"找不到 {stations_csv},請先依 README Step 1 準備好站點清單。")
    stations = pd.read_csv(stations_csv, comment="#").reset_index(drop=True)
    print(f"[info] 讀到 {len(stations)} 個站點")

    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    start = cfg["date_range"]["start"]
    end = cfg["date_range"]["end"]
    time_index = pd.date_range(start=start, end=end + " 23:00:00", freq="h", tz=None)

    # 1. 建圖
    adj, dist = build_graph(stations, cfg["graph"]["distance_threshold_km"], cfg["graph"]["edge_weight"])
    np.save(out_dir / "graph_adj.npy", adj)

    # 2. ERA5 內插
    era5_arrays = interpolate_era5_to_stations(cfg, stations, time_index)

    # 3. CNEMC 對齊
    cnemc_arrays = pivot_cnemc(cfg, stations, time_index)

    feature_names = list(cfg["cnemc"]["pollutants"]) + [v["short_name"] for v in cfg["era5"]["variables"]]
    all_arrays = {**cnemc_arrays, **era5_arrays}

    # 4. 補值前先出缺值報表,揪出「幾乎全缺、補值後等於偽站點」的情況
    report_missingness(all_arrays, stations, out_dir, warn_threshold=cfg["imputation"].get("warn_missing_pct", 0.5))

    max_gap_steps = cfg["imputation"]["max_time_gap_hours"]
    k_neighbors = cfg["imputation"]["spatial_fallback_k"]

    tensors = []
    for name in feature_names:
        if name not in all_arrays:
            raise SystemExit(f"缺少 feature: {name},請檢查前面 ERA5/CNEMC 步驟是否都成功")
        imputed = impute(all_arrays[name], adj, max_gap_steps, k_neighbors)
        n_missing = int(np.isnan(imputed).sum())
        if n_missing > 0:
            # impute() 內部已有三層保險(時間內插 -> 鄰居平均 -> 全站平均 -> 前後值填補),
            # 理論上跑到這裡不應該還有 NaN;如果還有,代表某個 feature 整個時間軸上
            # 所有站點全部都缺(否則 ffill/bfill 一定補得起來),這是資料源層級的嚴重問題。
            raise SystemExit(
                f"{name} 補值後仍有 {n_missing} 個缺值,理論上不該發生——"
                f"代表這個 feature 在某些時間點『所有站點』都沒有資料,請檢查原始 ERA5/CNEMC 資料是否完整。"
            )
        tensors.append(imputed)

    dataset = np.stack(tensors, axis=-1)  # (time, station, feature)
    np.save(out_dir / "dataset.npy", dataset)

    # 5. 輸出 metadata
    pd.Series(time_index).to_frame("time_utc").to_csv(out_dir / "time_index.csv", index=False)
    stations.to_csv(out_dir / "stations_used.csv", index=False)
    with open(out_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    print(f"[done] dataset.npy shape = {dataset.shape} (time, station, feature)")
    print(f"[done] feature 順序: {feature_names}")
    print(f"[done] 已輸出至 {out_dir}/")


if __name__ == "__main__":
    main()