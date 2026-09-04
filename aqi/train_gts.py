"""
train_gts.py
=============
Training script for faithful GTS (ICLR 2021) reproduction.
Uses preprocess.py output (features.npy, scalers.pkl, split.pkl).
Does NOT use adj_mx.npy / adj_wind.npy -- GTS learns its own graph.

Usage:
    python train_gts.py
    python train_gts.py --temperature 0.5 --epochs 100
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch
torch.backends.cudnn.benchmark = True
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader

from training.aqi.gts_faithful import GTS


# ── device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():        return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── dataset ───────────────────────────────────────────────────────────────────

class GTSDataset(Dataset):
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
        t = self.indices[idx]
        win = self.hist_len + self.pred_len
        full = self.features[t: t + win]
        x = full[:self.hist_len]
        y = full[self.hist_len:, :, self.PM25_IDX:self.PM25_IDX + 1]
        return torch.FloatTensor(x), torch.FloatTensor(y)


def build_node_series(features, train_start, train_end, hist_len):
    """
    Fixed reference PM2.5 series per node, used ONLY by the feature extractor
    to learn the graph structure. Matches the paper's design: the graph
    learner sees a representative sample of each node's dynamics, held fixed
    across all training batches (not re-sampled per batch), to keep the
    learned graph stable during training.

    Uses the first `hist_len` steps of the training split (never touches
    val/test, avoiding leakage into the learned graph).
    """
    pm25_idx = 0
    series = features[train_start:train_start + hist_len, :, pm25_idx]  # [hist_len, N]
    series = series.T[:, None, :]   # [N, 1, hist_len]
    return torch.FloatTensor(series)


def get_loaders(data_dir, hist_len, pred_len, batch_size, num_workers=0):
    features = np.load(os.path.join(data_dir, "features.npy"))
    with open(os.path.join(data_dir, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    with open(os.path.join(data_dir, "split.pkl"), "rb") as f:
        split = pickle.load(f)

    train_s, train_e = split["train"]
    val_s,   val_e   = split["val"]
    test_s,  test_e  = split["test"]

    train_ds = GTSDataset(features, train_s, train_e, hist_len, pred_len)
    val_ds   = GTSDataset(features, val_s,   val_e,   hist_len, pred_len)
    test_ds  = GTSDataset(features, test_s,  test_e,  hist_len, pred_len)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True, pin_memory=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=True, pin_memory=True),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=True, pin_memory=True),
    }

    node_series = build_node_series(features, train_s, train_e, hist_len)

    print(f"  Stations     : {features.shape[1]}")
    print(f"  Features     : {features.shape[2]}")
    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Val   samples: {len(val_ds):,}")
    print(f"  Test  samples: {len(test_ds):,}")

    return loaders, node_series, scalers, features.shape[1], features.shape[2]


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


def compute_sampling_prob(global_step, cl_decay_steps):
    return cl_decay_steps / (cl_decay_steps + np.exp(global_step / cl_decay_steps))


# ── train / eval ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, node_series, optimizer, criterion, device,
                     global_step, cl_decay_steps, kl_weight):
    model.train()
    total_recon = 0.0
    n_batches = len(loader)
    t0 = time.time()

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        tf_ratio = compute_sampling_prob(global_step, cl_decay_steps)
        pred, kl, _ = model(x, node_series, y_true=y, teacher_forcing_ratio=tf_ratio)
        recon_loss = criterion(pred, y)
        loss = recon_loss + kl_weight * kl

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_recon += recon_loss.item()
        global_step += 1

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_batches - i - 1)
            print(f"    batch {i+1:>5d}/{n_batches}  recon={recon_loss.item():.4f}  "
                  f"kl={kl.item():.4f}  tf_ratio={tf_ratio:.3f}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    return total_recon / n_batches, global_step


@torch.no_grad()
def evaluate(model, loader, node_series, scalers, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        pred, _, _ = model(x.to(device), node_series, y_true=None, teacher_forcing_ratio=0.0)
        preds.append(pred.cpu())
        targets.append(y)
    return compute_metrics(torch.cat(preds), torch.cat(targets), scalers)


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/processed_southwest")
    p.add_argument("--hist_len",   type=int,   default=24)
    p.add_argument("--pred_len",   type=int,   default=24)
    p.add_argument("--hid_dim",    type=int,   default=64)
    p.add_argument("--num_layers", type=int,   default=2)
    p.add_argument("--max_diffusion_step", type=int, default=2)
    p.add_argument("--embed_dim",  type=int,   default=100)
    p.add_argument("--temperature",type=float, default=0.5)
    p.add_argument("--kl_weight",  type=float, default=0.01)
    p.add_argument("--filter_type", default="dual_random_walk",
                   choices=["random_walk", "dual_random_walk"])
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-2)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--patience",   type=int,   default=15)
    p.add_argument("--cl_decay_steps", type=int, default=2000)
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--workers",    type=int,   default=4)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()

    print(f"Device      : {device}")
    print(f"Data        : {args.data}")
    print(f"Hist/Pred   : {args.hist_len}h / {args.pred_len}h")
    print(f"LR          : {args.lr}")
    print(f"Temperature : {args.temperature}  (Gumbel-Softmax)")
    print(f"KL weight   : {args.kl_weight}\n")

    print("Loading data ...")
    loaders, node_series, scalers, N, F = get_loaders(
        args.data, args.hist_len, args.pred_len,
        args.batch_size, args.workers,
    )
    node_series = node_series.to(device)

    with open(os.path.join(args.data, "split.pkl"), "rb") as f:
        split = pickle.load(f)
    print(f"\nData split:")
    print(f"  Train: {split['train_dates'][0]} → {split['train_dates'][1]}")
    print(f"  Val  : {split['val_dates'][0]}   → {split['val_dates'][1]}")
    print(f"  Test : {split['test_dates'][0]}  → {split['test_dates'][1]}")

    print("\nBuilding GTS ...")
    model = GTS(
        num_nodes=N, input_dim=F, output_dim=1,
        hid_dim=args.hid_dim, num_layers=args.num_layers,
        max_diffusion_step=args.max_diffusion_step,
        pred_len=args.pred_len, train_seq_len=args.hist_len,
        embed_dim=args.embed_dim, temperature=args.temperature,
        filter_type=args.filter_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr, eps=1e-3)
    scheduler = MultiStepLR(optimizer, milestones=[20, 30, 40, 50], gamma=0.1)
    criterion = nn.L1Loss()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "gts_faithful_best.pt")

    best_val_mae = float("inf")
    patience_cnt = 0
    global_step  = 0

    print("\nEpoch   Train-MAE    Val-MAE   Val-RMSE  Val-MAPE      LR      Time")
    print("─" * 72)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, global_step = train_one_epoch(
            model, loaders["train"], node_series, optimizer, criterion, device,
            global_step, args.cl_decay_steps, args.kl_weight,
        )
        val_metrics = evaluate(model, loaders["val"], node_series, scalers, device)
        scheduler.step()

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

    print(f"\nBest val MAE: {best_val_mae:.3f} µg/m³  ({save_path})")
    ckpt = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    test_metrics = evaluate(model, loaders["test"], node_series, scalers, device)
    print("\n── Test Results ──────────────────────────────")
    print(f"  MAE  : {test_metrics['MAE']:.3f} µg/m³")
    print(f"  RMSE : {test_metrics['RMSE']:.3f} µg/m³")
    print(f"  MAPE : {test_metrics['MAPE']:.2f}%")
    print("─────────────────────────────────────────────")


if __name__ == "__main__":
    main()