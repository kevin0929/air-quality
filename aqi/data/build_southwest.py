"""
Build the southwest Taiwan PM2.5 dataset.

Reads all monthly EPA CSV files from raw_data/ (2016-11 → 2026-05),
filters to 6 target counties, normalises station names, selects stations
with sufficient temporal coverage, backfills coordinates, then writes:

  combined_southwest.csv   raw merged data (hourly, long format)
  stations_southwest.csv   station metadata with coordinates

Usage:
    python data/build_southwest.py               # default settings
    python data/build_southwest.py --coverage 0.85
"""

import argparse
import glob
import os

import pandas as pd

# ── configuration ─────────────────────────────────────────────────────────────

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw_data")
OUT_DIR = os.path.dirname(__file__)

TARGET_COUNTIES = {"嘉義市", "嘉義縣", "屏東縣", "雲林縣", "高雄市", "臺南市"}

# Minimum fraction of total hourly timesteps a station must cover to be kept.
# 0.85 keeps stations running since roughly 2019-Q1 and earlier.
DEFAULT_MIN_COVERAGE = 0.85

# Explicit name aliases after full-width bracket normalisation.
# Maps non-standard name → canonical name.
NAME_ALIASES: dict[str, str] = {
    "湖內": "高雄(湖內)",           # brief alternative spelling 2021-09
}

# Short-lived / transient stations to drop explicitly (appear < 3 months total).
DROP_STATIONS = {
    "鹿林山",                        # high-altitude background, only 2016-11
    "楠梓加工出口區",                 # industrial zone, brief 2019-12 to 2020-01
    "永康區中山南路(臺南大學附屬高級中學)",  # one-off mobile measurement 2019-02
    "枋寮",                          # superseded by 屏東(枋寮) same month
    "屏東(建興)",                    # pilot station, 2 months only
    "嘉義(水上)",                    # one-off 2024-02
    "嘉義(東區)",                    # trial station, 3 months
    "嘉義(建國路)",                  # one-off 2026-03
    "屏東縣(泰武)",                  # 2 months only
    "高雄(林園)",                    # one-off 2024-04
    "臺南(南化)",                    # short-term, 2025-03 to 2026-01
    "林森",                          # new 2025-12, too short
}


# ── helpers ───────────────────────────────────────────────────────────────────

def normalise_name(name: str) -> str:
    """Convert full-width brackets to ASCII and apply known aliases."""
    name = str(name).replace("（", "(").replace("）", ")")
    return NAME_ALIASES.get(name, name)


def read_monthly_file(path: str) -> pd.DataFrame | None:
    """Read one monthly CSV; return None on empty / unrecognised format."""
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None

    # Some later files have quoted column names, e.g. '"sitename"'
    df.columns = [c.strip('"') for c in df.columns]

    if "county" not in df.columns or df.empty:
        return None

    return df


# ── main ─────────────────────────────────────────────────────────────────────

def build(min_coverage: float = DEFAULT_MIN_COVERAGE) -> None:
    pattern = os.path.join(RAW_DIR, "空氣品質指標(AQI)(歷史資料)*.csv")
    all_files = sorted(glob.glob(pattern))

    # Separate the no-date file (current partial month) from monthly archives
    monthly = [f for f in all_files if "(" in os.path.basename(f) and "20" in os.path.basename(f)]
    nodate  = [f for f in all_files if f not in monthly]

    print(f"Monthly archive files : {len(monthly)}")
    print(f"Current-month file(s) : {nodate}")

    # ── 1. load and concat ───────────────────────────────────────────────────
    chunks: list[pd.DataFrame] = []

    for path in monthly + nodate:
        df = read_monthly_file(path)
        if df is None:
            print(f"  [skip] {os.path.basename(path)}")
            continue

        # Filter counties first (cheap operation)
        # Normalise county names: 台 → 臺
        df["county"] = df["county"].str.replace("台", "臺", regex=False)
        df = df[df["county"].isin(TARGET_COUNTIES)].copy()
        if df.empty:
            continue

        chunks.append(df)

    print(f"\nLoaded {len(chunks)} non-empty files.")
    raw = pd.concat(chunks, ignore_index=True)
    print(f"Raw rows after county filter: {len(raw):,}")

    # ── 2. clean ─────────────────────────────────────────────────────────────
    raw["datacreationdate"] = pd.to_datetime(raw["datacreationdate"], errors="coerce")
    raw = raw.dropna(subset=["datacreationdate"])

    raw["sitename"] = raw["sitename"].map(normalise_name)

    # Drop explicitly unwanted stations
    raw = raw[~raw["sitename"].isin(DROP_STATIONS)]

    # Coerce numeric columns (sentinel '-' → NaN)
    NUM_COLS = ["so2", "so2_avg", "co", "co_8hr", "o3", "o3_8hr",
                "pm10", "pm10_avg", "pm2.5", "pm2.5_avg",
                "no2", "nox", "no", "windspeed", "winddirec",
                "longitude", "latitude"]
    for col in NUM_COLS:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # ── 3. deduplicate ───────────────────────────────────────────────────────
    raw = raw.drop_duplicates(subset=["sitename", "datacreationdate"], keep="last")
    raw = raw.sort_values(["datacreationdate", "sitename"]).reset_index(drop=True)

    total_timestamps = raw["datacreationdate"].nunique()
    print(f"Unique timestamps      : {total_timestamps:,}")
    print(f"Date range             : {raw['datacreationdate'].min()} → {raw['datacreationdate'].max()}")

    # ── 4. coverage-based station filtering ──────────────────────────────────
    counts = raw.groupby("sitename")["datacreationdate"].nunique()
    coverage = counts / total_timestamps
    kept = coverage[coverage >= min_coverage].index
    dropped_low = coverage[coverage < min_coverage]

    print(f"\nStation coverage (min={min_coverage:.0%}):")
    print(f"  Kept   : {len(kept)} stations")
    print(f"  Dropped: {len(dropped_low)} stations")
    if len(dropped_low):
        for s, c in dropped_low.sort_values(ascending=False).items():
            print(f"    {s:30s}  {c:.1%}")

    raw = raw[raw["sitename"].isin(kept)].copy()

    # ── 5. backfill lon / lat ─────────────────────────────────────────────────
    # Early files (pre-2020) have NaN coordinates; take the most recent known value
    # per station from any record that has coordinates.
    coord_ref = (
        raw.dropna(subset=["longitude", "latitude"])
           .sort_values("datacreationdate", ascending=False)
           .drop_duplicates("sitename")[["sitename", "county", "longitude", "latitude"]]
           .set_index("sitename")
    )
    missing_coords = raw["longitude"].isna()
    if missing_coords.any():
        raw.loc[missing_coords, "longitude"] = raw.loc[missing_coords, "sitename"].map(coord_ref["longitude"])
        raw.loc[missing_coords, "latitude"]  = raw.loc[missing_coords, "sitename"].map(coord_ref["latitude"])

    still_missing = raw["longitude"].isna().sum()
    if still_missing:
        print(f"\nWarning: {still_missing} rows still missing coordinates after backfill.")

    # ── 6. station metadata ──────────────────────────────────────────────────
    stations = (
        raw.dropna(subset=["longitude", "latitude"])
           .sort_values("datacreationdate", ascending=False)
           .drop_duplicates("sitename")[["sitename", "county", "longitude", "latitude"]]
           .sort_values("county")
           .reset_index(drop=True)
    )

    # ── 7. save ───────────────────────────────────────────────────────────────
    out_csv = os.path.join(OUT_DIR, "combined_southwest.csv")
    out_sta = os.path.join(OUT_DIR, "stations_southwest.csv")

    raw.to_csv(out_csv, index=False)
    stations.to_csv(out_sta, index=False)

    print(f"\n{'─'*55}")
    print(f"  Stations kept        : {raw['sitename'].nunique()}")
    print(f"  Unique timestamps    : {raw['datacreationdate'].nunique():,}")
    print(f"  Total rows           : {len(raw):,}")
    print(f"  combined_southwest.csv → {out_csv}")
    print(f"  stations_southwest.csv → {out_sta}")
    print(f"{'─'*55}")
    print("\nFinal station list:")
    for _, row in stations.iterrows():
        print(f"  {row['county']:6s}  {row['sitename']:20s}  "
              f"lon={row['longitude']:.4f}  lat={row['latitude']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=float, default=DEFAULT_MIN_COVERAGE,
                        help="Min fraction of timesteps a station must have (default 0.85)")
    args = parser.parse_args()
    build(min_coverage=args.coverage)
