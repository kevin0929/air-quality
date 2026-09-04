"""
驗證 build_dataset.py 的輸出品質,在真的拿去 fine-tune 之前先抓出明顯異常。

用法:
    python verify_dataset.py --config config.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 各 feature 的合理數值範圍(超出這個範圍不代表一定錯,但值得人工確認)
SANITY_RANGES = {
    "PM2.5": (0, 1000),       # ug/m3,重污染日偶爾會超過 500,但超過 1000 大概率是單位或資料錯誤
    "O3": (0, 500),           # ug/m3
    "t2m": (200, 330),        # K,約 -73°C ~ 57°C,涵蓋所有陸地合理氣溫
    "d2m": (200, 320),        # K
    "tp": (0, 0.5),           # m,單小時累積降水,0.5m/hr 已經是極端暴雨
    "sp": (50000, 110000),    # Pa
    "blh": (0, 5000),         # m
    "msdwswrf": (0, 1500),    # W/m^2,地表太陽輻射通量理論上限約 1361 W/m^2(太陽常數)
    "u100": (-100, 100),      # m/s
    "v100": (-100, 100),      # m/s
}


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    out_dir = Path(cfg["paths"]["output_dir"])
    dataset = np.load(out_dir / "dataset.npy")
    with open(out_dir / "feature_names.json", encoding="utf-8") as f:
        feature_names = json.load(f)
    time_index = pd.read_csv(out_dir / "time_index.csv")["time_utc"].astype("datetime64[ns]")
    stations = pd.read_csv(out_dir / "stations_used.csv")
    adj = np.load(out_dir / "graph_adj.npy")

    n_time, n_station, n_feature = dataset.shape
    print(f"=== 基本形狀檢查 ===")
    print(f"dataset.npy shape = {dataset.shape} (time={n_time}, station={n_station}, feature={n_feature})")
    ok = True

    if n_station != len(stations):
        print(f"❌ station 維度({n_station})跟 stations_used.csv 列數({len(stations)})對不上")
        ok = False
    if n_time != len(time_index):
        print(f"❌ time 維度({n_time})跟 time_index.csv 列數({len(time_index)})對不上")
        ok = False
    if n_feature != len(feature_names):
        print(f"❌ feature 維度({n_feature})跟 feature_names.json 長度({len(feature_names)})對不上")
        ok = False
    if adj.shape != (n_station, n_station):
        print(f"❌ graph_adj.npy shape {adj.shape} 應該要是 ({n_station}, {n_station})")
        ok = False
    if ok:
        print("✅ 各檔案維度互相一致")

    print(f"\n=== 缺值檢查 ===")
    n_nan = int(np.isnan(dataset).sum())
    if n_nan > 0:
        print(f"❌ 還有 {n_nan} 個 NaN(理論上 build_dataset.py 已經保證沒有殘留 NaN,出現代表用的是舊版腳本或流程中斷過)")
        ok = False
    else:
        print("✅ 沒有任何 NaN")

    print(f"\n=== 時間軸連續性檢查 ===")
    diffs = time_index.diff().dropna()
    expected = pd.Timedelta(hours=1)
    bad_gaps = diffs[diffs != expected]
    if len(bad_gaps) > 0:
        print(f"❌ 時間軸有 {len(bad_gaps)} 處不是剛好間隔 1 小時,請檢查 build_dataset.py 的 time_index 產生邏輯")
        ok = False
    else:
        print(f"✅ 時間軸從 {time_index.iloc[0]} 到 {time_index.iloc[-1]},每小時一筆,完全連續")

    print(f"\n=== 數值範圍檢查(每個 feature)===")
    for i, name in enumerate(feature_names):
        vals = dataset[:, :, i]
        vmin, vmax, vmean = np.nanmin(vals), np.nanmax(vals), np.nanmean(vals)
        flag = ""
        if name in SANITY_RANGES:
            lo, hi = SANITY_RANGES[name]
            n_out = int(((vals < lo) | (vals > hi)).sum())
            if n_out > 0:
                pct = n_out / vals.size * 100
                flag = f"  ⚠️ 有 {n_out} 筆({pct:.2f}%)超出合理範圍 [{lo}, {hi}]"
                if pct > 1.0:  # 超過 1% 才真的算警訊,個位數的極端天氣事件是正常的
                    ok = False
        print(f"  {name:10s}: min={vmin:10.2f}  max={vmax:10.2f}  mean={vmean:10.2f}{flag}")

    print(f"\n=== 圖連通性檢查 ===")
    degrees = adj.sum(axis=1)
    isolated = np.where(degrees == 0)[0]
    print(f"  平均每站鄰居數: {degrees.mean():.1f}  (最少 {int(degrees.min())} / 最多 {int(degrees.max())})")
    if len(isolated) > 0:
        codes = stations.iloc[isolated]["station_code"].tolist()
        print(f"  ⚠️ 有 {len(isolated)} 個孤立站點(0 個鄰居): {codes}")

    print(f"\n=== 各站點資料完整度分佈(來自 station_completeness.csv,若存在)===")
    completeness_file = out_dir / "station_completeness.csv"
    if completeness_file.exists():
        comp = pd.read_csv(completeness_file)
        pct_cols = [c for c in comp.columns if c.endswith("_missing_pct")]
        worst = comp[pct_cols].max(axis=1)
        print(f"  補值前最嚴重 feature 缺值比例: 平均 {worst.mean()*100:.2f}%, 最大 {worst.max()*100:.2f}%")
        bad_stations = comp[worst >= 0.3]
        if len(bad_stations) > 0:
            print(f"  ⚠️ 有 {len(bad_stations)} 個站點原始缺值比例 >= 30%,這些站點的 dataset.npy 數值有較大比例是補值/合成的:")
            for _, r in bad_stations.iterrows():
                wf = max(pct_cols, key=lambda c: r[c])
                print(f"      {r['station_code']}: {wf} = {r[wf]*100:.1f}%")
    else:
        print("  找不到 station_completeness.csv,略過")

    print(f"\n{'='*50}")
    print("✅ 整體檢查通過,可以進入下一步" if ok else "❌ 有項目需要人工確認,建議處理後再進入 fine-tune")


if __name__ == "__main__":
    main()