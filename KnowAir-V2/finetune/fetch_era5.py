"""
從 Copernicus CDS 下載 ERA5(或 ERA5T)再分析氣象資料,涵蓋 stations.csv 站點範圍。

前置需求:
    1. ~/.cdsapirc 已設定好(見 README Step 0-A)
    2. pip install -r requirements.txt

用法:
    python fetch_era5.py --config config.yaml

已知 CDS 行為(2024 年起持續存在,2026 年官方論壇仍有回報):當請求裡同時包含「瞬時值」類
變數(如 t2m、u100)跟「累積值/平均率」類變數(如 tp、msdwswrf)時,CDS 會把輸出拆成
data_stream-oper_stepType-instant.nc / data_stream-oper_stepType-accum.nc 兩個檔案,
包成 zip——即使明確指定 download_format: unarchived 也一樣,官方目前沒有把這行為納入
該參數的控制範圍。這支程式會自動偵測並解壓縮,下游 build_dataset.py 也已改成可以合併
讀取一個月份底下的多個 nc 檔案。
"""
import argparse
import calendar
import shutil
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_bbox(stations_csv: str, buffer_deg: float):
    """從站點清單自動算出涵蓋範圍的 bounding box(北, 西, 南, 東),外加緩衝degree。"""
    df = pd.read_csv(stations_csv, comment="#")
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError(
            f"{stations_csv} 缺少 lat/lon 欄位,請確認已照 stations_template.csv 的格式整理好。"
        )
    north = df["lat"].max() + buffer_deg
    south = df["lat"].min() - buffer_deg
    east = df["lon"].max() + buffer_deg
    west = df["lon"].min() - buffer_deg
    return [north, west, south, east]  # CDS API 要求的 area 格式:N/W/S/E


def month_range(start: date, end: date):
    """把日期區間拆成 (year, month) 清單,ERA5 CDS 請求通常以月為單位比較穩定。"""
    cur = date(start.year, start.month, 1)
    out = []
    while cur <= end:
        out.append((cur.year, cur.month))
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return out


def is_near_real_time(year: int, month: int) -> bool:
    """粗略判斷是否落在 ERA5 正式版尚未釋出的近期範圍(預設用 3 個月當緩衝)。"""
    cutoff = date.today().replace(day=1) - timedelta(days=90)
    return date(year, month, 1) >= date(cutoff.year, cutoff.month, 1)


def fetch_month(client, cfg: dict, year: int, month: int, area, out_dir: Path):
    """下載單一月份,回傳該月份的資料夾路徑(裡面可能是 1 個或多個 nc 檔,視 CDS 是否拆檔而定)。"""
    days = [f"{d:02d}" for d in range(1, calendar.monthrange(year, month)[1] + 1)]
    hours = [f"{h:02d}:00" for h in range(24)]
    var_names = [v["name"] for v in cfg["era5"]["variables"]]

    nrt = is_near_real_time(year, month)
    tag = "era5t" if nrt else "era5"
    month_dir = out_dir / f"{year}{month:02d}_{tag}"
    done_marker = month_dir / ".complete"

    if done_marker.exists():
        print(f"[skip] {month_dir} 已存在且完整")
        return month_dir

    month_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_dir / f"_tmp_{year}{month:02d}.download"

    request = {
        "product_type": ["reanalysis"],
        "variable": var_names,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": days,
        "time": hours,
        "area": area,  # [North, West, South, East]
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    dataset = cfg["era5"]["dataset_near_real_time"] if nrt else cfg["era5"]["dataset"]
    print(f"[fetch] {year}-{month:02d} ({'ERA5T,近即時,之後可能被官方修正' if nrt else 'ERA5 正式版'})")
    client.retrieve(dataset, request).download(str(tmp_path))

    if zipfile.is_zipfile(tmp_path):
        print(f"[info] {year}-{month:02d} CDS 回傳的是 zip(混合瞬時值/累積值變數時的已知行為),自動解壓縮")
        with zipfile.ZipFile(tmp_path) as zf:
            zf.extractall(month_dir)
        tmp_path.unlink()
    else:
        # 單一 netCDF 檔案,搬進同一套資料夾結構,方便下游統一用 glob 讀取
        shutil.move(str(tmp_path), str(month_dir / f"era5_{year}{month:02d}_{tag}.nc"))

    nc_count = len(list(month_dir.glob("*.nc")))
    if nc_count == 0:
        raise SystemExit(f"{year}-{month:02d} 下載完成但資料夾裡沒有任何 .nc 檔,請檢查 {month_dir}")
    print(f"[info] {year}-{month:02d} 共 {nc_count} 個 nc 檔案")

    done_marker.touch()
    return month_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)

    lock_file = Path(".fetch_era5.lock")
    if lock_file.exists():
        raise SystemExit(
            f"偵測到 {lock_file} 已存在,代表可能已經有另一個 fetch_era5.py 正在執行"
            f"(這就是上次出現兩個重複 CDS 請求的原因——同一個月份被開了兩個視窗各送一次)。\n"
            f"確認真的沒有其他執行中的程序後,手動刪除這個檔案再重跑:rm {lock_file}"
        )
    lock_file.touch()

    try:
        _run(cfg)
    finally:
        lock_file.unlink(missing_ok=True)


def _run(cfg: dict):
    try:
        import cdsapi
    except ImportError as e:
        raise SystemExit(
            "找不到 cdsapi,請先 `pip install cdsapi`,並依 README Step 0-A 設定 ~/.cdsapirc"
        ) from e

    stations_csv = cfg["paths"]["stations_csv"]
    if not Path(stations_csv).exists():
        raise SystemExit(
            f"找不到 {stations_csv}。請先依 README Step 1 準備好站點清單,"
            f"可以先複製 stations_template.csv 改成 stations.csv 再填入真實資料。"
        )

    area = compute_bbox(stations_csv, cfg["era5"]["bbox_buffer_deg"])
    print(f"[info] 依站點清單算出的下載範圍(N/W/S/E): {area}")

    out_dir = Path(cfg["paths"]["raw_era5_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(cfg["date_range"]["start"], "%Y-%m-%d").date()
    end = datetime.strptime(cfg["date_range"]["end"], "%Y-%m-%d").date()

    client = cdsapi.Client()
    downloaded = []
    for year, month in month_range(start, end):
        f = fetch_month(client, cfg, year, month, area, out_dir)
        downloaded.append(f)

    print(f"[done] 共下載 {len(downloaded)} 個月份的 ERA5 資料至 {out_dir}/")


if __name__ == "__main__":
    main()