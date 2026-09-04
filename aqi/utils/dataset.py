"""
Shared sliding-window dataset for PM2.5-GNN, AirPhyNet, and STGNP.

Each sample:
    x : [seq_len,  N, F]  – normalised historical features
    y : [pred_len, N, 1]  – normalised PM2.5 (feature index 0) as target
"""

import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class AQIDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,     # [T, N, F]  full normalised feature array
        split: str,               # 'train' | 'val' | 'test'
        split_path: str,
        seq_len: int = 24,
        pred_len: int = 24,
    ):
        with open(split_path, "rb") as f:
            splits = pickle.load(f)
        start, end = splits[split]

        self.data = features[start:end].astype(np.float32)  # [T_split, N, F]
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_samples = len(self.data) - seq_len - pred_len + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.seq_len]                             # [seq_len, N, F]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, :, :1]  # [pred_len, N, 1]
        return torch.from_numpy(x), torch.from_numpy(y)


def get_dataloaders(
    data_dir: str,
    seq_len: int = 24,
    pred_len: int = 24,
    batch_size: int = 32,
    num_workers: int = 0,
) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """
    Load processed arrays and return train/val/test DataLoaders
    plus adjacency matrices and scalers.

    Returns:
        loaders    : {'train': DataLoader, 'val': ..., 'test': ...}
        adj_mx     : [N, N]  symmetric distance adjacency
        adj_wind   : [N, N]  directed wind-aware adjacency
        scalers    : {'mean': [F], 'std': [F], 'feature_names': [...]}
    """
    features = np.load(os.path.join(data_dir, "features.npy"))
    adj_mx   = np.load(os.path.join(data_dir, "adj_mx.npy"))
    adj_wind = np.load(os.path.join(data_dir, "adj_wind.npy"))
    split_path = os.path.join(data_dir, "split.pkl")

    with open(os.path.join(data_dir, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)

    loaders = {}
    for split in ("train", "val", "test"):
        ds = AQIDataset(features, split=split, split_path=split_path,
                        seq_len=seq_len, pred_len=pred_len)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=False,
            drop_last=(split == "train"),
        )
        print(f"  {split:5s}: {len(ds):>7,} samples  ({len(loaders[split]):>5,} batches)")

    return loaders, adj_mx, adj_wind, scalers
