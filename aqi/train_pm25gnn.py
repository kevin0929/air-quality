"""
train_pm25gnn.py
=================
Training script for the faithful PM2.5-GNN reproduction.
Reads preprocess.py output and trains using teacher-forcing (Option B).

Usage:
    python train_pm25gnn.py
    python train_pm25gnn.py --hist_len 48 --pred_len 24 --epochs 100
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

from training.aqi.models.pm25gnn    import PM25GNN
from training.aqi.pm25gnn_dataset   import get_pm25gnn_loaders


# ── device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():   return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(pred_norm, target_norm, scalers):
    """
    pred_norm, target_norm: [B, pred_len, N, 1] normalised PM2.5
    Returns MAE, RMSE, MAPE in physical units (µg/m³).
    """
    pm25_mean = float(scalers["mean"][0])
    pm25_std  = float(scalers["std"][0])

    pred   = pred_norm.squeeze(-1)   * pm25_std + pm25_mean   # [B, T, N]
    target = target_norm.squeeze(-1) * pm25_std + pm25_mean

    mae  = (pred - target).abs().mean().item()
    rmse = ((pred - target) ** 2).mean().sqrt().item()
    # MAPE: mask out near-zero to avoid explosion
    mask = target.abs() > 1.0
    mape = ((pred[mask] - target[mask]).abs() / target[mask].abs()).mean().item() * 100 \
           if mask.any() else float("nan")

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


# ── training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for pm25_hist, feature, pm25_label in loader:
        pm25_hist  = pm25_hist.to(device)
        feature    = feature.to(device)
        pm25_label = pm25_label.to(device)

        optimizer.zero_grad()
        pred = model(pm25_hist, feature)          # [B, pred_len, N, 1]
        loss = criterion(pred, pm25_label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, scalers, device):
    model.eval()
    preds, targets = [], []
    for pm25_hist, feature, pm25_label in loader:
        pred = model(pm25_hist.to(device), feature.to(device))
        preds.append(pred.cpu())
        targets.append(pm25_label)
    preds   = torch.cat(preds,   dim=0)
    targets = torch.cat(targets, dim=0)
    return compute_metrics(preds, targets, scalers)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/processed_southwest")
    p.add_argument("--hist_len",   type=int,   default=24)
    p.add_argument("--pred_len",   type=int,   default=24)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)   # original paper lr
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--patience",   type=int,   default=15)
    p.add_argument("--lr_patience",type=int,   default=5)
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--workers",    type=int,   default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()

    print(f"Device  : {device}")
    print(f"Data    : {args.data}")
    print(f"Hist/Pred: {args.hist_len}h / {args.pred_len}h")
    print(f"LR      : {args.lr}")
    print()

    # ── data ──────────────────────────────────────────────────────────────
    print("Loading data ...")
    (loaders, edge_index, edge_attr,
     scalers, wind_mean, wind_std,
     city_num, in_dim) = get_pm25gnn_loaders(
        args.data,
        hist_len=args.hist_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    # Print split date ranges
    with open(os.path.join(args.data, "split.pkl"), "rb") as f:
        split = pickle.load(f)
    print(f"\nData split:")
    print(f"  Train: {split['train_dates'][0]} → {split['train_dates'][1]}")
    print(f"  Val  : {split['val_dates'][0]}   → {split['val_dates'][1]}")
    print(f"  Test : {split['test_dates'][0]}  → {split['test_dates'][1]}")

    # ── model ──────────────────────────────────────────────────────────────
    print("\nBuilding PM2.5-GNN ...")
    model = PM25GNN(
        hist_len   = args.hist_len,
        pred_len   = args.pred_len,
        in_dim     = in_dim,        # F (full feature dim including PM2.5)
        city_num   = city_num,
        batch_size = args.batch_size,
        device     = device,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        wind_mean  = wind_mean,
        wind_std   = wind_std,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Edges (M)       : {edge_index.shape[1]}")

    # ── optimiser (Adam, lr=1e-3 as in original paper) ────────────────────
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=args.lr_patience, min_lr=1e-6,
    )
    criterion = nn.MSELoss()   # original paper uses MSE

    # ── training ───────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "pm25gnn_faithful_best.pt")

    best_val_mae = float("inf")
    patience_cnt = 0
    prev_lr      = args.lr

    print("\nEpoch   Train-MSE    Val-MAE   Val-RMSE  Val-MAPE      LR      Time")
    print("─" * 72)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss  = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_metrics = evaluate(model, loaders["val"], scalers, device)
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
                "epoch": epoch,
                "model": model.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, save_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── test ───────────────────────────────────────────────────────────────
    print(f"\nBest val MAE: {best_val_mae:.3f} µg/m³  ({save_path})")
    ckpt = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, loaders["test"], scalers, device)

    print("\n── Test Results ──────────────────────────────")
    print(f"  MAE  : {test_metrics['MAE']:.3f} µg/m³")
    print(f"  RMSE : {test_metrics['RMSE']:.3f} µg/m³")
    print(f"  MAPE : {test_metrics['MAPE']:.2f}%")
    print("─────────────────────────────────────────────")


if __name__ == "__main__":
    main()