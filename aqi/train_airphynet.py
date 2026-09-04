"""
train_airphynet.py
==================
Training script for faithful AirPhyNet (ICLR 2024) reproduction.
Uses preprocess.py output (features.npy, adj_mx.npy, adj_wind.npy, scalers.pkl).

Usage:
    pip install torchdiffeq
    python train_airphynet.py
    python train_airphynet.py --latent_dim 8 --n_traj 3 --pred_len 24
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from training.aqi.airphynet_faithful import AirPhyNetModel


# ── device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():        return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── dataset ───────────────────────────────────────────────────────────────────

class AirPhyNetDataset(Dataset):
    """
    Sliding window dataset.
    Each sample: (x [T,N,F], y [pred_len,N,1])
    PM2.5 is feature index 0.
    """
    PM25_IDX = 0

    def __init__(self, features, start, end, hist_len, pred_len):
        self.features = features
        self.hist_len = hist_len
        self.pred_len = pred_len
        window = hist_len + pred_len
        self.indices = list(range(start, end - window + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t    = self.indices[idx]
        win  = self.hist_len + self.pred_len
        full = self.features[t: t + win]               # [win, N, F]
        x    = full[:self.hist_len]                    # [T, N, F]
        y    = full[self.hist_len:, :, self.PM25_IDX:self.PM25_IDX+1]  # [pred, N, 1]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def get_loaders(data_dir, hist_len, pred_len, batch_size, num_workers=0):
    features = np.load(os.path.join(data_dir, "features.npy"))
    adj_mx   = np.load(os.path.join(data_dir, "adj_mx.npy"))
    adj_wind = np.load(os.path.join(data_dir, "adj_wind.npy"))

    with open(os.path.join(data_dir, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    with open(os.path.join(data_dir, "split.pkl"), "rb") as f:
        split = pickle.load(f)

    train_s, train_e = split["train"]
    val_s,   val_e   = split["val"]
    test_s,  test_e  = split["test"]

    train_ds = AirPhyNetDataset(features, train_s, train_e, hist_len, pred_len)
    val_ds   = AirPhyNetDataset(features, val_s,   val_e,   hist_len, pred_len)
    test_ds  = AirPhyNetDataset(features, test_s,  test_e,  hist_len, pred_len)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=True),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=True),
    }

    print(f"  Stations     : {features.shape[1]}")
    print(f"  Features     : {features.shape[2]}")
    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Val   samples: {len(val_ds):,}")
    print(f"  Test  samples: {len(test_ds):,}")

    return loaders, adj_mx, adj_wind, scalers, features.shape[1], features.shape[2]


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred_norm, target_norm, scalers):
    pm25_mean = float(scalers["mean"][0])
    pm25_std  = float(scalers["std"][0])
    pred   = pred_norm.squeeze(-1)   * pm25_std + pm25_mean
    target = target_norm.squeeze(-1) * pm25_std + pm25_mean
    mae  = (pred - target).abs().mean().item()
    rmse = ((pred - target) ** 2).mean().sqrt().item()
    mask = target.abs() > 1.0
    mape = ((pred[mask] - target[mask]).abs() / target[mask].abs()).mean().item() * 100 \
           if mask.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ── train / eval ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, adj_mx, adj_wind, device,
                     kl_weight=0.001):
    model.train()
    total = 0.0
    n_batches = len(loader)
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred, kl = model(x, adj_mx, adj_wind, return_kl=True)   # [B, pred_len, N, 1]
        recon_loss = criterion(pred, y)
        loss = recon_loss + kl_weight * kl
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += recon_loss.item()   # track reconstruction loss only for logging

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_batches - i - 1)
            print(f"    batch {i+1:>5d}/{n_batches}  "
                  f"recon={recon_loss.item():.4f}  kl={kl.item():.4f}  "
                  f"nfe={model.ode_func.nfe}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    return total / n_batches


@torch.no_grad()
def evaluate(model, loader, adj_mx, adj_wind, scalers, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        pred = model(x.to(device), adj_mx, adj_wind)
        preds.append(pred.cpu())
        targets.append(y)
    return compute_metrics(torch.cat(preds), torch.cat(targets), scalers)


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/processed_southwest")
    p.add_argument("--hist_len",   type=int,   default=24)
    p.add_argument("--pred_len",   type=int,   default=24)
    p.add_argument("--hidden_dim", type=int,   default=64)
    p.add_argument("--latent_dim", type=int,   default=4)
    p.add_argument("--n_traj",     type=int,   default=1)
    p.add_argument("--gen_dim",    type=int,   default=64)
    p.add_argument("--ode_method", default="dopri5")
    p.add_argument("--ode_rtol",   type=float, default=1e-2)
    p.add_argument("--ode_atol",   type=float, default=1e-3)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--patience",   type=int,   default=15)
    p.add_argument("--lr_patience",type=int,   default=5)
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--workers",    type=int,   default=0)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()

    print(f"Device   : {device}")
    print(f"Data     : {args.data}")
    print(f"Hist/Pred: {args.hist_len}h / {args.pred_len}h")
    print(f"ODE      : method={args.ode_method}  rtol={args.ode_rtol}  atol={args.ode_atol}")
    print(f"Latent   : dim={args.latent_dim}  n_traj={args.n_traj}\n")

    # ── data ──────────────────────────────────────────────────────
    print("Loading data ...")
    loaders, adj_mx_np, adj_wind_np, scalers, N, F = get_loaders(
        args.data, args.hist_len, args.pred_len,
        args.batch_size, args.workers,
    )

    with open(os.path.join(args.data, "split.pkl"), "rb") as f:
        split = pickle.load(f)
    print(f"\nData split:")
    print(f"  Train: {split['train_dates'][0]} → {split['train_dates'][1]}")
    print(f"  Val  : {split['val_dates'][0]}   → {split['val_dates'][1]}")
    print(f"  Test : {split['test_dates'][0]}  → {split['test_dates'][1]}")

    adj_mx   = torch.FloatTensor(adj_mx_np).to(device)
    adj_wind = torch.FloatTensor(adj_wind_np).to(device)

    # ── model ──────────────────────────────────────────────────────
    print("\nBuilding AirPhyNet ...")
    model = AirPhyNetModel(
        num_nodes   = N,
        in_features = F,
        hidden_dim  = args.hidden_dim,
        pred_len    = args.pred_len,
        latent_dim  = args.latent_dim,
        n_traj      = args.n_traj,
        ode_method  = args.ode_method,
        ode_rtol    = args.ode_rtol,
        ode_atol    = args.ode_atol,
        gen_dim     = args.gen_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── optimiser ──────────────────────────────────────────────────
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=args.lr_patience, min_lr=1e-6,
    )
    criterion = nn.MSELoss()

    # ── training loop ──────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "airphynet_faithful_best.pt")

    best_val_mae = float("inf")
    patience_cnt = 0
    prev_lr      = args.lr

    print("\nEpoch   Train-MSE    Val-MAE   Val-RMSE  Val-MAPE      LR      Time")
    print("─" * 72)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss  = train_one_epoch(model, loaders["train"], optimizer,
                                      criterion, adj_mx, adj_wind, device)
        val_metrics = evaluate(model, loaders["val"],
                               adj_mx, adj_wind, scalers, device)
        scheduler.step(val_metrics["MAE"])

        cur_lr  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        is_best = val_metrics["MAE"] < best_val_mae
        marker  = " ★" if is_best else ""

        print(
            f"{epoch:>5d}   {train_loss:>10.4f}   "
            f"{val_metrics['MAE']:>7.3f}   "
            f"{val_metrics['RMSE']:>8.3f}  "
            f"{val_metrics['MAPE']:>7.2f}%  "
            f"{cur_lr:.2e}  "
            f"{elapsed:>5.1f}s"
            f"{marker}"
        )

        if cur_lr < prev_lr:
            print(f"        ↓ lr: {prev_lr:.2e} → {cur_lr:.2e}")
            prev_lr = cur_lr

        if is_best:
            best_val_mae = val_metrics["MAE"]
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "val_metrics": val_metrics, "args": vars(args),
            }, save_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── test ───────────────────────────────────────────────────────
    print(f"\nBest val MAE: {best_val_mae:.3f} µg/m³  ({save_path})")
    ckpt = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, loaders["test"],
                            adj_mx, adj_wind, scalers, device)
    print("\n── Test Results ──────────────────────────────")
    print(f"  MAE  : {test_metrics['MAE']:.3f} µg/m³")
    print(f"  RMSE : {test_metrics['RMSE']:.3f} µg/m³")
    print(f"  MAPE : {test_metrics['MAPE']:.2f}%")
    print("─────────────────────────────────────────────")


if __name__ == "__main__":
    main()