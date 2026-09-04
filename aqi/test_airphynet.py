"""
test_airphynet.py
==================
Load the best AirPhyNet checkpoint and evaluate on the test set.

Usage:
    python test_airphynet.py
    python test_airphynet.py --checkpoint checkpoints/airphynet_faithful_best.pt
"""

import argparse
import os
import pickle

import torch

from training.aqi.airphynet_faithful import AirPhyNetModel
from training.aqi.train_airphynet import get_loaders, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/processed_southwest")
    p.add_argument("--checkpoint", default="checkpoints/airphynet_faithful_best.pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers",    type=int, default=0)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device    : {device}")
    print(f"Checkpoint: {args.checkpoint}\n")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    train_args = ckpt["args"]

    print(f"Loaded checkpoint from epoch {ckpt['epoch']}")
    print(f"  Best val metrics at save time: {ckpt['val_metrics']}\n")

    print("Loading data ...")
    loaders, adj_mx_np, adj_wind_np, scalers, N, F = get_loaders(
        args.data,
        hist_len=train_args["hist_len"],
        pred_len=train_args["pred_len"],
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    adj_mx   = torch.FloatTensor(adj_mx_np).to(device)
    adj_wind = torch.FloatTensor(adj_wind_np).to(device)

    print("\nBuilding model ...")
    model = AirPhyNetModel(
        num_nodes   = N,
        in_features = F,
        hidden_dim  = train_args["hidden_dim"],
        pred_len    = train_args["pred_len"],
        latent_dim  = train_args["latent_dim"],
        n_traj      = train_args["n_traj"],
        ode_method  = train_args["ode_method"],
        ode_rtol    = train_args["ode_rtol"],
        ode_atol    = train_args["ode_atol"],
        gen_dim     = train_args["gen_dim"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print("\nEvaluating on test set ...")
    test_metrics = evaluate(model, loaders["test"], adj_mx, adj_wind, scalers, device)

    print("\n── Test Results ──────────────────────────────")
    print(f"  MAE  : {test_metrics['MAE']:.3f} µg/m³")
    print(f"  RMSE : {test_metrics['RMSE']:.3f} µg/m³")
    print(f"  MAPE : {test_metrics['MAPE']:.2f}%")
    print("─────────────────────────────────────────────")


if __name__ == "__main__":
    main()