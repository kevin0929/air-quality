import numpy as np
import pandas as pd

features = np.load("data/processed_southwest/features_raw.npy")  # [T, N, F]
stations = pd.read_csv("data/processed_southwest/stations.csv")
timestamps = pd.read_csv("data/processed_southwest/timestamps.csv")

feature_names = ["pm2.5","pm10","so2","co","o3","no2","nox","no","wind_u","wind_v"]

# 攤平成長表格 (long format) 方便看
records = []
for t_idx in range(5):  # 只看前 5 個時間點示範
    for n_idx, station in stations.iterrows():
        row = {"timestamp": timestamps.iloc[t_idx]["timestamp"], "station": station["sitename"]}
        row.update({feature_names[f]: features[t_idx, n_idx, f] for f in range(len(feature_names))})
        records.append(row)

training_df = pd.DataFrame(records)
print(training_df.head())