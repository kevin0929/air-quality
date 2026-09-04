"""
pm25gnn_dataset.py
==================
Dataset adapter for PM2.5-GNN that converts preprocess.py output
(features.npy, adj_wind.npy, scalers.pkl, split.pkl) into the
original model interface:

    pm25_hist : [B, hist_len, N, 1]   normalised PM2.5
    feature   : [B, hist_len+pred_len, N, F]   all node features
    pm25_label: [B, pred_len, N, 1]   normalised PM2.5 ground truth

Teacher-forcing: future steps in `feature` are ground-truth observations,
not NWP forecasts. Document this clearly in your thesis.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from training.aqi.models.pm25gnn import build_edge_index_attr


# ── Dataset ────────────────────────────────────────────────────────────────────

class PM25GNNDataset(Dataset):
    """
    Sliding-window dataset for PM2.5-GNN.

    Each sample:
        pm25_hist  [hist_len, N, 1]
        feature    [hist_len + pred_len, N, F]
        pm25_label [pred_len, N, 1]

    F = total features (e.g. 10), PM2.5 is feature index 0.
    """

    PM25_IDX = 0   # index of pm2.5 in features.npy

    def __init__(self, features: np.ndarray, start: int, end: int,
                 hist_len: int, pred_len: int):
        """
        features : [T, N, F]  full normalised feature array
        start    : first valid index (inclusive)
        end      : last valid index (exclusive)
        """
        self.hist_len  = hist_len
        self.pred_len  = pred_len
        self.features  = features   # [T, N, F]

        # Valid start indices: need hist_len + pred_len steps ahead
        window = hist_len + pred_len
        self.indices = list(range(start, end - window + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        window = self.hist_len + self.pred_len

        # Full window: [hist_len + pred_len, N, F]
        full = self.features[t: t + window]

        # PM2.5 history: [hist_len, N, 1]
        pm25_hist = full[:self.hist_len, :, self.PM25_IDX: self.PM25_IDX + 1]

        # Feature window (hist + future): [hist_len + pred_len, N, F]
        # Teacher-forcing: future meteorology is ground-truth, but future PM2.5
        # (index 0) is masked to 0 to prevent data leakage.
        # The model must predict PM2.5; it only sees future weather as context.
        feature = full.copy()
        feature[self.hist_len:, :, self.PM25_IDX] = 0.0   # mask future PM2.5

        # PM2.5 labels: [pred_len, N, 1]
        pm25_label = full[self.hist_len:, :, self.PM25_IDX: self.PM25_IDX + 1]

        return (
            torch.FloatTensor(pm25_hist),   # [hist_len, N, 1]
            torch.FloatTensor(feature),     # [hist_len+pred_len, N, F]
            torch.FloatTensor(pm25_label),  # [pred_len, N, 1]
        )


# ── Loader factory ─────────────────────────────────────────────────────────────

def get_pm25gnn_loaders(
    data_dir: str,
    hist_len: int = 24,
    pred_len: int = 24,
    batch_size: int = 32,
    num_workers: int = 0,
):
    """
    Returns:
        loaders      : dict  {train, val, test} → DataLoader
        edge_index   : np.ndarray [2, M]
        edge_attr    : np.ndarray [M, 2]  (city_dist km, city_direc rad)
        scalers      : dict  {mean, std, feature_names}
        wind_mean    : np.ndarray [2]  physical mean of (windspeed, winddirec)
        wind_std     : np.ndarray [2]  physical std
        city_num     : int  N
        in_dim       : int  F (full feature dim, PM2.5 included at idx 0)
    """
    # ── load preprocessed arrays ──────────────────────────────────────────
    features = np.load(os.path.join(data_dir, "features.npy"))        # [T, N, F]
    adj_wind = np.load(os.path.join(data_dir, "adj_wind.npy"))        # [N, N]
    dist_mx  = np.load(os.path.join(data_dir, "dist_mx.npy"))         # [N, N]

    with open(os.path.join(data_dir, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)

    with open(os.path.join(data_dir, "split.pkl"), "rb") as f:
        split = pickle.load(f)

    stations = pd.read_csv(os.path.join(data_dir, "stations.csv"))

    # ── build edge_index and edge_attr ────────────────────────────────────
    station_lons = stations["longitude"].values
    station_lats = stations["latitude"].values

    edge_index, edge_attr = build_edge_index_attr(
        adj_wind, dist_mx, station_lons, station_lats
    )

    # ── recover wind physical stats from scalers ──────────────────────────
    # feature_names order: pm2.5, pm10, so2, co, o3, no2, nox, no, wind_u, wind_v
    feat_names = scalers["feature_names"]
    mean_all   = scalers["mean"]    # [F]
    std_all    = scalers["std"]     # [F]

    # windspeed and winddirec stats (wind_u=idx8, wind_v=idx9)
    # We pass the last-two-feature stats to GraphGNN for denormalization
    wind_u_idx = feat_names.index("wind_u")
    wind_v_idx = feat_names.index("wind_v")
    wind_mean  = np.array([mean_all[wind_u_idx], mean_all[wind_v_idx]], dtype=np.float32)
    wind_std   = np.array([std_all[wind_u_idx],  std_all[wind_v_idx]],  dtype=np.float32)

    # ── build datasets ────────────────────────────────────────────────────
    T, N, F = features.shape
    train_start, train_end = split["train"]
    val_start,   val_end   = split["val"]
    test_start,  test_end  = split["test"]

    train_ds = PM25GNNDataset(features, train_start, train_end, hist_len, pred_len)
    val_ds   = PM25GNNDataset(features, val_start,   val_end,   hist_len, pred_len)
    test_ds  = PM25GNNDataset(features, test_start,  test_end,  hist_len, pred_len)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size,
                            shuffle=True,  num_workers=num_workers, drop_last=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, drop_last=True),
        "test":  DataLoader(test_ds,  batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, drop_last=True),
    }

    print(f"  Stations (N)  : {N}")
    print(f"  Features (F)  : {F}  {feat_names}")
    print(f"  Edges (M)     : {edge_index.shape[1]}")
    print(f"  Train samples : {len(train_ds):,}")
    print(f"  Val   samples : {len(val_ds):,}")
    print(f"  Test  samples : {len(test_ds):,}")

    return loaders, edge_index, edge_attr, scalers, wind_mean, wind_std, N, F