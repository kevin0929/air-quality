"""
prepare_finetune_data.py
=========================
把 knowair_extend/build_dataset.py 的輸出(dataset.npy [time,station,feature] +
stations_used.csv + time_index.csv)轉成跟 preprocess_knowair.py 完全相容的格式
(features.npy + adj_mx.npy + scalers.pkl + split.pkl),讓 train_dcrnn.py /
finetune_dcrnn.py 可以直接讀取這個資料夾,不用改任何 dataloader 邏輯。

跟原始 preprocess_knowair.py 保持一致的地方(逐一對照過,不是憑空假設):
  - feature 順序:  ['pm2.5','o3','t2m','d2m','sp','tp','blh','msdwswrf','wind_u','wind_v']
                    (你的 build_dataset.py 輸出是 tp/sp 對調,這裡用「按名稱對應」
                    重新排列,不是硬編碼位置交換,更不容易再出錯)
  - 標準化:        一定用原本預訓練的 scalers.pkl 裡的 mean/std,不在這裡重新計算
                    (沿用同一把尺,新舊時段的數值才有可比性)
  - adj_mx 公式:    dist_to_adj() 完全照抄——高斯核 exp(-dist^2/sigma^2),
                    sigma = 非零距離的標準差,閾值 0.1 卡在高斯值上(不是公里數)

用法:
    python prepare_finetune_data.py \
        --dataset_dir  ../finetune/output \
        --orig_scalers ../data/processed_yrd/scalers.pkl \
        --output_dir   data/processed_yrd_finetune
"""
import argparse
import json
import pickle
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

ADJ_THRESHOLD = 0.1  # 跟 preprocess_knowair.py 完全一致,卡在高斯值上


# ── 跟 preprocess_knowair.py 逐字一致的公式 ──────────────────────────────

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


# ── feature 名稱對應:把 build_dataset.py 的命名/順序轉成 scalers.pkl 那套 ──

# build_dataset.py 輸出名稱 -> preprocess_knowair.py OUTPUT_FEATURE_NAMES 名稱
NAME_MAP = {
    "PM2.5": "pm2.5",
    "O3": "o3",
    "t2m": "t2m",
    "d2m": "d2m",
    "sp": "sp",
    "tp": "tp",
    "blh": "blh",
    "msdwswrf": "msdwswrf",
    "u100": "wind_u",
    "v100": "wind_v",
}


def reorder_features(dataset: np.ndarray, src_names: list, target_names: list) -> np.ndarray:
    """把 (time, station, feature) 的 feature 軸,依名稱從 src_names 的順序重排成 target_names 的順序。"""
    mapped_src_names = [NAME_MAP.get(n, n) for n in src_names]
    missing = set(target_names) - set(mapped_src_names)
    if missing:
        raise SystemExit(
            f"目標 feature 清單裡有 {missing} 這幾個,但 build_dataset.py 的輸出對應不到。"
            f"來源(對應後): {mapped_src_names}  目標: {target_names}"
        )
    index_lookup = {name: i for i, name in enumerate(mapped_src_names)}
    order = [index_lookup[name] for name in target_names]
    if order == list(range(len(target_names))):
        print("[info] feature 順序本來就一致,不需要重排(但仍走一次流程確保萬無一失)")
    else:
        print(f"[info] feature 重新排序: {[src_names[i] for i in order]} -> {target_names}")
    return dataset[:, :, order]


def monthly_stratified_split(time_index: pd.Series, train_ratio: float, val_ratio: float):
    """依月份分層切分:每個月自己先切一刀(前段 train、中段 val、後段 test),
    再把各月同一組的區段收集起來——train/val/test 三邊都會涵蓋全年每個月,
    不會出現「train 缺冬天、test 全冬天」這種季節分布不對稱的狀況。

    回傳: {"train": [(s1,e1),(s2,e2),...], "val": [...], "test": [...]}
    每個 tuple 是一段連續的區間,供 DCRNNDataset 在區間內部滑窗取樣
    (不會跨月份邊界取樣,避免不同月份中間的斷點被誤當成連續時間)。
    """
    months = time_index.dt.to_period("M")
    train_ranges, val_ranges, test_ranges = [], [], []

    for period in sorted(months.unique()):
        idx = np.where(months == period)[0]
        m_start, m_end = idx[0], idx[-1] + 1   # [start, end) 半開區間
        n = m_end - m_start
        t_end = m_start + int(n * train_ratio)
        v_end = m_start + int(n * (train_ratio + val_ratio))
        if t_end > m_start:
            train_ranges.append((int(m_start), int(t_end)))
        if v_end > t_end:
            val_ranges.append((int(t_end), int(v_end)))
        if m_end > v_end:
            test_ranges.append((int(v_end), int(m_end)))

    return {"train": train_ranges, "val": val_ranges, "test": test_ranges}


def ranges_to_date_strings(ranges: list, time_index: pd.Series):
    """把一組區間轉成 (最早日期, 最晚日期) 字串,方便印出來人工確認涵蓋範圍。"""
    if not ranges:
        return ("(空)", "(空)")
    starts = [time_index.iloc[s] for s, _ in ranges]
    ends = [time_index.iloc[e - 1] for _, e in ranges]
    return (str(min(starts)), str(max(ends)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", required=True, help="build_dataset.py 的 output/ 資料夾")
    ap.add_argument("--orig_scalers", required=True, help="原本預訓練用的 scalers.pkl 路徑")
    ap.add_argument("--output_dir", required=True, help="輸出資料夾,給 train_dcrnn.py --data 指到這裡")
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--split_strategy", choices=["chronological", "monthly_stratified"],
                    default="monthly_stratified",
                    help="chronological: 整段時間一刀切(train/val/test 季節分布可能不對稱)。"
                         "monthly_stratified(預設): 每個月自己先切,train/val/test 都涵蓋全年每個月。")
    # test_ratio 用剩下的補滿,不用另外指定
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 讀原本的 scalers,決定目標 feature 順序 ──
    with open(args.orig_scalers, "rb") as f:
        orig_scalers = pickle.load(f)
    target_names = list(orig_scalers["feature_names"])
    mean = np.asarray(orig_scalers["mean"], dtype=np.float32)
    std = np.asarray(orig_scalers["std"], dtype=np.float32)
    print(f"[info] 原始 scalers.pkl: feature_names={target_names}")
    print(f"[info] mean={mean}")
    print(f"[info] std={std}")

    # ── 讀 build_dataset.py 的輸出 ──
    dataset = np.load(dataset_dir / "dataset.npy")
    with open(dataset_dir / "feature_names.json", encoding="utf-8") as f:
        src_names = json.load(f)
    stations = pd.read_csv(dataset_dir / "stations_used.csv")
    time_index = pd.read_csv(dataset_dir / "time_index.csv")["time_utc"].astype("datetime64[ns]")

    if len(target_names) != dataset.shape[-1]:
        raise SystemExit(
            f"feature 數量對不上:原始 scalers 是 {len(target_names)} 維,"
            f"新資料是 {dataset.shape[-1]} 維,無法直接沿用同一個預訓練模型。"
        )

    # ── 重排 feature 順序,確保跟原本語義一致 ──
    dataset_reordered = reorder_features(dataset, src_names, target_names)

    # ── 用原始 mean/std 標準化(絕對不能用新資料重新算)──
    features_norm = (dataset_reordered - mean) / std
    n_nan = int(np.isnan(features_norm).sum())
    if n_nan > 0:
        raise SystemExit(f"標準化後出現 {n_nan} 個 NaN,通常代表 std 裡有 0,請檢查原始 scalers.pkl")

    # ── 建 adj_mx(高斯核公式跟 preprocess_knowair.py 完全一致)──
    stations_geo = stations.rename(columns={"lat": "latitude", "lon": "longitude"})
    dist_mx = build_distance_matrix(stations_geo)
    adj_mx = dist_to_adj(dist_mx)
    isolated = np.where((adj_mx > 0).sum(axis=1) == 0)[0]
    if len(isolated) > 0:
        codes = stations.iloc[isolated]["station_code"].tolist()
        print(f"[warn] 高斯核 + threshold={ADJ_THRESHOLD} 之下有 {len(isolated)} 個孤立站點: {codes}")

    # ── 切 train/val/test ──
    T = features_norm.shape[0]
    if args.split_strategy == "chronological":
        train_end = int(T * args.train_ratio)
        val_end = int(T * (args.train_ratio + args.val_ratio))
        split_indices = {
            "train": (0, train_end),
            "val": (train_end, val_end),
            "test": (val_end, T),
            "train_dates": (str(time_index.iloc[0]), str(time_index.iloc[train_end - 1])),
            "val_dates": (str(time_index.iloc[train_end]), str(time_index.iloc[val_end - 1])),
            "test_dates": (str(time_index.iloc[val_end]), str(time_index.iloc[-1])),
        }
    else:  # monthly_stratified
        ranges = monthly_stratified_split(time_index, args.train_ratio, args.val_ratio)
        split_indices = {
            "train": ranges["train"],
            "val": ranges["val"],
            "test": ranges["test"],
            "train_dates": ranges_to_date_strings(ranges["train"], time_index),
            "val_dates": ranges_to_date_strings(ranges["val"], time_index),
            "test_dates": ranges_to_date_strings(ranges["test"], time_index),
        }
        n_train = sum(e - s for s, e in ranges["train"])
        n_val = sum(e - s for s, e in ranges["val"])
        n_test = sum(e - s for s, e in ranges["test"])
        train_end, val_end = n_train, n_train + n_val  # 只用來印總筆數,不是真的索引邊界

    # ── 輸出(跟 preprocess_knowair.py 同一套 schema,scalers 原封不動沿用)──
    np.save(output_dir / "features.npy", features_norm.astype(np.float32))
    np.save(output_dir / "features_raw.npy", dataset_reordered.astype(np.float32))
    np.save(output_dir / "adj_mx.npy", adj_mx)
    stations.to_csv(output_dir / "stations.csv", index=False)
    pd.DataFrame({"timestamp": time_index}).to_csv(output_dir / "timestamps.csv", index=False)
    with open(output_dir / "scalers.pkl", "wb") as f:
        pickle.dump(orig_scalers, f)   # 原封不動沿用,不是重新算
    with open(output_dir / "split.pkl", "wb") as f:
        pickle.dump(split_indices, f)

    print(f"\n{'─'*55}")
    print(f"  切分策略: {args.split_strategy}")
    print(f"  features.npy: {features_norm.shape}")
    print(f"  adj_mx.npy:   {adj_mx.shape}  edges={int((adj_mx>0).sum())}")
    if args.split_strategy == "chronological":
        print(f"  Train [{train_end:6d} steps]: {split_indices['train_dates'][0]} -> {split_indices['train_dates'][1]}")
        print(f"  Val   [{val_end-train_end:6d} steps]: {split_indices['val_dates'][0]} -> {split_indices['val_dates'][1]}")
        print(f"  Test  [{T-val_end:6d} steps]: {split_indices['test_dates'][0]} -> {split_indices['test_dates'][1]}")
    else:
        print(f"  Train [{train_end:6d} steps, {len(split_indices['train'])} 段]: "
              f"{split_indices['train_dates'][0]} -> {split_indices['train_dates'][1]} (橫跨全年)")
        print(f"  Val   [{val_end-train_end:6d} steps, {len(split_indices['val'])} 段]: "
              f"{split_indices['val_dates'][0]} -> {split_indices['val_dates'][1]} (橫跨全年)")
        print(f"  Test  [{T-val_end:6d} steps, {len(split_indices['test'])} 段]: "
              f"{split_indices['test_dates'][0]} -> {split_indices['test_dates'][1]} (橫跨全年)")
    print(f"  輸出至 -> {output_dir}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()