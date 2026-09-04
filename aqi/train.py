"""
Unified training script for PM2.5-GNN / AirPhyNet / STGNP.

Changes from original:
  1. WeightedHuberLoss  - penalises high-PM2.5 events more heavily
  2. ReduceLROnPlateau  - replaces CosineAnnealingLR; lr halves when val stalls
  3. Seasonal embedding - adds sin/cos month features on-the-fly if --use_season
  4. Better logging     - prints lr each epoch, marks best epoch with ★
  5. split.pkl dates    - printed at startup so you always know train/val/test range

Usage:
    python train.py --model pm25gnn
    python train.py --model pm25gnn --lr 3e-4 --use_season
    python train.py --model airphynet --hidden_dim 128 --epochs 150
    python train.py --model stgnp --pred_len 48 --batch_size 16
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(__file__))

from training.aqi.models.pm25gnn   import PM25GNN
from training.aqi.models.airphynet import AirPhyNet
from training.aqi.models.stgnp     import STGNP
from training.aqi.utils.dataset    import get_dataloaders
from training.aqi.utils.metrics    import compute_metrics


# ── device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── loss ──────────────────────────────────────────────────────────────────────

class WeightedHuberLoss(nn.Module):
    """
    HuberLoss with extra penalty for high-PM2.5 samples.

    target is Z-score normalised, so abs(target) > high_z_threshold
    roughly corresponds to high physical concentrations.

    Default high_z_threshold=1.0 ≈ mean + 1 std  (typically ~25-30 µg/m³
    for Taiwan stations), high_weight=2.0 doubles the loss on those samples.
    """
    def __init__(
        self,
        delta: float = 1.0,
        high_z_threshold: float = 1.0,
        high_weight: float = 2.0,
    ):
        super().__init__()
        self.delta = delta
        self.high_z_threshold = high_z_threshold
        self.high_weight = high_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        huber = nn.functional.huber_loss(
            pred, target, delta=self.delta, reduction="none"
        )
        # Weight high-concentration samples more
        weight = torch.where(
            target.abs() > self.high_z_threshold,
            torch.full_like(target, self.high_weight),
            torch.ones_like(target),
        )
        return (huber * weight).mean()


# ── seasonal embedding ────────────────────────────────────────────────────────

def add_seasonal_features(
    x: torch.Tensor,
    timestamps: np.ndarray,
    sample_indices: torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """
    Append sin/cos month encoding to node features.

    x shape     : [B, seq_len, N, F]
    timestamps  : [T] numpy datetime array
    sample_indices: [B] starting time index of each sample in the batch

    Returns [B, seq_len, N, F+2]
    """
    B, T_in, N, F = x.shape
    sin_cos = torch.zeros(B, T_in, N, 2, device=x.device)

    for b, start_idx in enumerate(sample_indices):
        for t in range(T_in):
            ts = timestamps[start_idx + t]
            month = ts.month if hasattr(ts, "month") else int(
                np.datetime64(ts, "M").astype(int) % 12 + 1
            )
            angle = 2 * np.pi * month / 12
            sin_cos[b, t, :, 0] = np.sin(angle)
            sin_cos[b, t, :, 1] = np.cos(angle)

    return torch.cat([x, sin_cos], dim=-1)


# ── model factory ─────────────────────────────────────────────────────────────

def build_model(
    name: str,
    num_nodes: int,
    in_features: int,
    args: argparse.Namespace,
    station_coords: np.ndarray | None = None,
    dist_mx: np.ndarray | None = None,
) -> nn.Module:
    common = dict(
        num_nodes=num_nodes,
        in_features=in_features,
        hidden_dim=args.hidden_dim,
        pred_len=args.pred_len,
        dropout=args.dropout,
    )
    if name == "pm25gnn":
        return PM25GNN(
            **common,
            num_gnn_layers=args.num_layers,
            station_coords=station_coords,
            dist_mx=dist_mx,
        )
    if name == "airphynet":
        return AirPhyNet(**common, num_physics_steps=args.num_layers)
    if name == "stgnp":
        return STGNP(**common, num_layers=args.num_layers, num_heads=args.num_heads)
    raise ValueError(f"Unknown model: {name}")


# ── training helpers ──────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    adj_mx: torch.Tensor,
    adj_wind: torch.Tensor,
    device: torch.device,
    model_name: str,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        # dataset may return (x, y) or (x, y, indices) if use_season
        x, y = batch[0].to(device), batch[1].to(device)

        optimizer.zero_grad()

        if model_name == "pm25gnn":
            pred = model(x, adj_wind)
            loss = criterion(pred, y)
        elif model_name == "airphynet":
            pred = model(x, adj_mx, adj_wind)
            loss = criterion(pred, y)
        elif model_name == "stgnp":
            loss = model.nll_loss(x, adj_mx, y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    adj_mx: torch.Tensor,
    adj_wind: torch.Tensor,
    device: torch.device,
    model_name: str,
    scalers: dict,
) -> dict[str, float]:
    model.eval()
    preds_list, targets_list = [], []

    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)

        if model_name == "pm25gnn":
            pred = model(x, adj_wind)
        elif model_name == "airphynet":
            pred = model(x, adj_mx, adj_wind)
        elif model_name == "stgnp":
            pred, _ = model(x, adj_mx)

        preds_list.append(pred.cpu())
        targets_list.append(y.cpu())

    preds   = torch.cat(preds_list,   dim=0)
    targets = torch.cat(targets_list, dim=0)
    return compute_metrics(preds, targets, scalers)


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PM2.5-GNN / AirPhyNet / STGNP")
    p.add_argument("--model",      required=True, choices=["pm25gnn", "airphynet", "stgnp"])
    p.add_argument("--data",       default="data/processed_southwest")
    p.add_argument("--seq_len",    type=int,   default=24)
    p.add_argument("--pred_len",   type=int,   default=24)
    p.add_argument("--hidden_dim", type=int,   default=64)
    p.add_argument("--num_layers", type=int,   default=2)
    p.add_argument("--num_heads",  type=int,   default=4)
    p.add_argument("--dropout",    type=float, default=0.1)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=3e-4)   # ← 預設從 1e-3 調低
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--patience",   type=int,   default=15)

    # WeightedHuberLoss params
    p.add_argument("--high_z",     type=float, default=1.0,
                   help="Z-score threshold for high-PM2.5 weighting")
    p.add_argument("--high_w",     type=float, default=2.0,
                   help="Loss weight multiplier for high-PM2.5 samples")

    # ReduceLROnPlateau params
    p.add_argument("--lr_patience",  type=int,   default=5,
                   help="Epochs to wait before halving lr")
    p.add_argument("--lr_factor",    type=float, default=0.5)
    p.add_argument("--lr_min",       type=float, default=1e-6)

    p.add_argument("--use_season",  action="store_true",
                   help="Append sin/cos month features to input")
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--workers",     type=int, default=0)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = get_device()

    print(f"Device  : {device}")
    print(f"Model   : {args.model}")
    print(f"Data    : {args.data}")
    print(f"Seq/Pred: {args.seq_len}h / {args.pred_len}h")
    print(f"LR      : {args.lr}  (ReduceLROnPlateau factor={args.lr_factor} patience={args.lr_patience})")
    print(f"Loss    : WeightedHuber(delta=1.0, high_z>{args.high_z} → ×{args.high_w})")
    print(f"Season  : {'on' if args.use_season else 'off'}")

    # ── print split dates ──────────────────────────────────────────
    split_path = os.path.join(args.data, "split.pkl")
    if os.path.exists(split_path):
        with open(split_path, "rb") as f:
            split_info = pickle.load(f)
        print(f"\nData split:")
        print(f"  Train: {split_info['train_dates'][0]} → {split_info['train_dates'][1]}")
        print(f"  Val  : {split_info['val_dates'][0]}   → {split_info['val_dates'][1]}")
        print(f"  Test : {split_info['test_dates'][0]}  → {split_info['test_dates'][1]}")

    # ── load data ──────────────────────────────────────────────────
    print("\nLoading data …")
    loaders, adj_mx_np, adj_wind_np, scalers = get_dataloaders(
        args.data,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    adj_mx   = torch.FloatTensor(adj_mx_np).to(device)
    adj_wind = torch.FloatTensor(adj_wind_np).to(device)

    features    = np.load(os.path.join(args.data, "features.npy"))
    num_nodes   = features.shape[1]
    in_features = features.shape[2] + (2 if args.use_season else 0)

    for split, loader in loaders.items():
        print(f"  {split:<6}: {len(loader.dataset):>7,} samples  ({len(loader):>5,} batches)")

    # ── load station coords + dist_mx for dynamic adj ────────────
    import pandas as pd
    stations_df = pd.read_csv(os.path.join(args.data, "stations.csv"))
    station_coords_np = stations_df[["longitude", "latitude"]].values  # [N, 2]
    dist_mx_np = np.load(os.path.join(args.data, "dist_mx.npy"))

    # ── model ─────────────────────────────────────────────────────
    model = build_model(
        args.model, num_nodes, in_features, args,
        station_coords=station_coords_np,
        dist_mx=dist_mx_np,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # ── optimiser + scheduler ──────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.lr_min,
    )
    criterion = WeightedHuberLoss(
        delta=1.0,
        high_z_threshold=args.high_z,
        high_weight=args.high_w,
    )

    # ── training loop ──────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"{args.model}_best.pt")

    best_val_mae  = float("inf")
    patience_cnt  = 0
    prev_lr       = args.lr

    print("\nEpoch   Train-loss   Val-MAE   Val-RMSE  Val-MAPE      LR      Time")
    print("─" * 72)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, loaders["train"], optimizer,
            adj_mx, adj_wind, device, args.model, criterion,
        )
        val_metrics = evaluate(
            model, loaders["val"],
            adj_mx, adj_wind, device, args.model, scalers,
        )
        scheduler.step(val_metrics["MAE"])

        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]["lr"]

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

        # lr drop notification
        if cur_lr < prev_lr:
            print(f"        ↓ lr reduced: {prev_lr:.2e} → {cur_lr:.2e}")
            prev_lr = cur_lr

        if is_best:
            best_val_mae = val_metrics["MAE"]
            torch.save(
                {
                    "epoch":       epoch,
                    "model":       model.state_dict(),
                    "args":        vars(args),
                    "val_metrics": val_metrics,
                },
                save_path,
            )
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── test ───────────────────────────────────────────────────────
    print(f"\nBest val MAE: {best_val_mae:.3f} µg/m³  (checkpoint: {save_path})")
    ckpt = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(
        model, loaders["test"],
        adj_mx, adj_wind, device, args.model, scalers,
    )
    print("\n── Test Results ──────────────────────────────")
    print(f"  MAE  : {test_metrics['MAE']:.3f} µg/m³")
    print(f"  RMSE : {test_metrics['RMSE']:.3f} µg/m³")
    print(f"  MAPE : {test_metrics['MAPE']:.2f}%")
    print("─────────────────────────────────────────────")


if __name__ == "__main__":
    main()