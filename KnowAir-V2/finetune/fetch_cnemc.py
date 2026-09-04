"""
下載 CNEMC 空氣品質站點資料。

預設走 quotsoft.net 的免申請歷史 CSV 鏡像(全國國控監測點資料,2014/05/13 起,
每日凌晨 2 點後有前一天的完整資料)。若 config.yaml 裡 cnemc.source 設為 "aliyun",
改走阿里雲市場的授權 API(需先依 README Step 0-B 開通並填入 AppCode)。

用法:
    python fetch_cnemc.py --config config.yaml
"""
import argparse
import io
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

QUOTSOFT_URL_TMPL = "{base}/china_sites_{date}.csv"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def date_list(start: str, end: str):
    d0 = datetime.strptime(start, "%Y-%m-%d").date()
    d1 = datetime.strptime(end, "%Y-%m-%d").date()
    out = []
    cur = d0
    while cur <= d1:
        out.append(cur)
        cur += timedelta(days=1)
    return out


def fetch_one_day_quotsoft(day, base_url: str, station_codes: set, pollutants: list, session: requests.Session):
    """下載單日 china_sites_YYYYMMDD.csv,篩選出目標站點與污染物,轉成長格式 DataFrame。

    回傳 columns: [datetime_utc8, station_code, pollutant, value]
    若當天資料下載失敗或不存在,回傳 None(呼叫端要記錄缺失日期,交給後續補值處理)。
    """
    url = QUOTSOFT_URL_TMPL.format(base=base_url, date=day.strftime("%Y%m%d"))
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"[warn] {day} 下載失敗({e}),先跳過,之後會被記錄為缺值")
        return None

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as e:
        print(f"[warn] {day} CSV 解析失敗({e}),先跳過")
        return None

    if "type" not in df.columns:
        print(f"[warn] {day} 檔案格式不符預期(缺少 type 欄位),先跳過")
        return None

    # 只留下我們要的污染物 type,且只留下 config 站點清單裡有的欄位
    df = df[df["type"].isin(pollutants)]
    keep_cols = ["date", "hour", "type"] + [c for c in df.columns if c in station_codes]
    if len(keep_cols) <= 3:
        print(f"[warn] {day} 找不到任何目標站點欄位,請確認 stations.csv 的 station_code 是否跟 CNEMC 編碼一致")
        return None
    df = df[keep_cols]

    long_df = df.melt(id_vars=["date", "hour", "type"], var_name="station_code", value_name="value")
    long_df["datetime_utc8"] = pd.to_datetime(
        long_df["date"].astype(str), format="%Y%m%d"
    ) + pd.to_timedelta(long_df["hour"], unit="h")
    long_df = long_df.rename(columns={"type": "pollutant"})
    long_df = long_df[["datetime_utc8", "station_code", "pollutant", "value"]]
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    return long_df


def fetch_range_quotsoft(cfg: dict, station_codes: set):
    base = cfg["cnemc"]["quotsoft_base_url"]
    pollutants = cfg["cnemc"]["pollutants"]
    days = date_list(cfg["date_range"]["start"], cfg["date_range"]["end"])

    session = requests.Session()
    frames = []
    for day in tqdm(days, desc="下載 CNEMC(quotsoft)"):
        f = fetch_one_day_quotsoft(day, base, station_codes, pollutants, session)
        if f is not None:
            frames.append(f)
        time.sleep(0.5)  # 放慢速度,不要對免費鏡像站造成太大負擔

    if not frames:
        raise SystemExit("整段時間都沒抓到任何資料,請檢查日期範圍、站點代碼、網路連線")
    return pd.concat(frames, ignore_index=True)


def fetch_range_aliyun(cfg: dict, station_codes: set):
    """阿里雲授權 API 的骨架,需依你實際購買方案的 API 文件調整 request 參數與回傳解析。"""
    appcode = cfg["cnemc"]["aliyun"]["appcode"]
    endpoint = cfg["cnemc"]["aliyun"]["endpoint"]
    if not appcode:
        raise SystemExit(
            "cnemc.source 設為 aliyun,但 config.yaml 裡 cnemc.aliyun.appcode 是空的。"
            "請先依 README Step 0-B 開通並填入 AppCode。"
        )

    headers = {"Authorization": f"APPCODE {appcode}"}
    days = date_list(cfg["date_range"]["start"], cfg["date_range"]["end"])
    frames = []
    for day in tqdm(days, desc="下載 CNEMC(阿里雲 API)"):
        # TODO: 這裡的 params 需要照你購買方案的實際 API 文件填,
        # 每家方案的參數名稱、回傳格式都不太一樣,以下只是示意結構。
        params = {"date": day.strftime("%Y-%m-%d")}
        try:
            resp = requests.get(endpoint, headers=headers, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[warn] {day} 阿里雲 API 請求失敗({e}),先跳過")
            continue
        # TODO: 依實際回傳 JSON 結構解析成跟 quotsoft 版本一樣的長格式 DataFrame
        # frames.append(parsed_df)

    if not frames:
        raise SystemExit(
            "阿里雲 API 骨架尚未接上實際回傳格式解析邏輯,請對照你購買方案的 API 文件補上 TODO 部分。"
        )
    return pd.concat(frames, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    stations_csv = cfg["paths"]["stations_csv"]
    if not Path(stations_csv).exists():
        raise SystemExit(f"找不到 {stations_csv},請先依 README Step 1 準備好站點清單。")
    stations = pd.read_csv(stations_csv, comment="#")
    station_codes = set(stations["station_code"].astype(str))

    out_dir = Path(cfg["paths"]["raw_cnemc_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    source = cfg["cnemc"].get("source", "quotsoft")
    if source == "quotsoft":
        df = fetch_range_quotsoft(cfg, station_codes)
    elif source == "aliyun":
        df = fetch_range_aliyun(cfg, station_codes)
    else:
        raise SystemExit(f"未知的 cnemc.source: {source},請設為 quotsoft 或 aliyun")

    missing_codes = station_codes - set(df["station_code"].unique())
    if missing_codes:
        print(
            f"[warn] 有 {len(missing_codes)} 個站點代碼在整段期間完全沒抓到資料"
            f"(可能站點停用、編碼對不上,或該時段資料源缺失): {sorted(missing_codes)[:10]}..."
        )

    out_file = out_dir / "cnemc_combined.parquet"
    df.to_parquet(out_file, index=False)
    print(f"[done] 共 {len(df):,} 筆記錄,已存至 {out_file}")


if __name__ == "__main__":
    main()