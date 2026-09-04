"""
train_dcrnn.py
===============
Training script for faithful DCRNN (ICLR 2018) reproduction.
Uses preprocess.py output (features.npy, adj_mx.npy, scalers.pkl, split.pkl).

Logging
-------
Training/validation curves are written as TensorBoard event files under
``--log_dir/--exp_name`` (default: ``runs/<timestamp>_h<hist>_p<pred>``).
After (or during) training, view them with:

    tensorboard --logdir runs

and open http://localhost:6006 in a browser.

Optionally, pass --wandb to *also* mirror every scalar to Weights & Biases
(requires `pip install wandb` and being logged in). This is not required —
TensorBoard alone satisfies the "must be able to open a local GUI" demo
requirement — but is provided for convenience if you prefer W&B's dashboard.

Usage:
    python train_dcrnn.py
    python train_dcrnn.py --epochs 100 --lr 1e-2
    python train_dcrnn.py --hist_len 72 --pred_len 72 --exp_name yrd_72h
    python train_dcrnn.py --wandb --wandb_project dcrnn-knowair
"""

import argparse
import io
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
from torch.utils.tensorboard import SummaryWriter

from dcrnn_faithful import DCRNN


# ── device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():        return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── dataset ───────────────────────────────────────────────────────────────────

class DCRNNDataset(Dataset):
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


def get_loaders(data_dir, hist_len, pred_len, batch_size, num_workers=0):
    features = np.load(os.path.join(data_dir, "features.npy"))
    adj_mx   = np.load(os.path.join(data_dir, "adj_mx.npy"))
    with open(os.path.join(data_dir, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    with open(os.path.join(data_dir, "split.pkl"), "rb") as f:
        split = pickle.load(f)

    train_s, train_e = split["train"]
    val_s,   val_e   = split["val"]
    test_s,  test_e  = split["test"]

    train_ds = DCRNNDataset(features, train_s, train_e, hist_len, pred_len)
    val_ds   = DCRNNDataset(features, val_s,   val_e,   hist_len, pred_len)
    test_ds  = DCRNNDataset(features, test_s,  test_e,  hist_len, pred_len)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True, pin_memory=True),
        "val":   DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=True, pin_memory=True),
        "test":  DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, drop_last=True, pin_memory=True),
    }

    print(f"  Stations     : {features.shape[1]}")
    print(f"  Features     : {features.shape[2]}")
    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Val   samples: {len(val_ds):,}")
    print(f"  Test  samples: {len(test_ds):,}")

    return loaders, adj_mx, scalers, features.shape[1], features.shape[2]


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
    """Inverse sigmoid decay for scheduled sampling, matches original paper Eq. 7."""
    return cl_decay_steps / (cl_decay_steps + np.exp(global_step / cl_decay_steps))


# ── logging helpers ──────────────────────────────────────────────────────────

def log_scalar(writer, wandb_run, tag, value, step):
    writer.add_scalar(tag, value, step)
    if wandb_run is not None:
        wandb_run.log({tag: value}, step=step)


@torch.no_grad()
def log_prediction_figure(writer, wandb_run, model, loader, scalers, device, epoch, node_idx=0):
    """Log a predicted-vs-actual PM2.5 curve for one station/one batch to TensorBoard.

    Purely illustrative (useful for demo recordings) — does not affect training.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)
    pred = model(x, y_true=None, teacher_forcing_ratio=0.0)

    pm25_mean = float(scalers["mean"][0])
    pm25_std  = float(scalers["std"][0])
    pred_denorm   = pred[0, :, node_idx, 0].cpu().numpy() * pm25_std + pm25_mean
    target_denorm = y[0, :, node_idx, 0].numpy() * pm25_std + pm25_mean

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(target_denorm, label="Ground truth", marker="o", markersize=3)
    ax.plot(pred_denorm,   label="Prediction",   marker="x", markersize=3)
    ax.set_xlabel("Horizon step (h)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title(f"Epoch {epoch} — station idx {node_idx}, val sample")
    ax.legend()
    fig.tight_layout()

    writer.add_figure("val/prediction_vs_truth", fig, global_step=epoch)
    if wandb_run is not None:
        import wandb
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        wandb_run.log({"val/prediction_vs_truth": wandb.Image(buf.getvalue())}, step=epoch)
    plt.close(fig)


# ── train / eval ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, global_step, cl_decay_steps,
                     writer=None, wandb_run=None, tb_log_every=10):
    model.train()
    total = 0.0
    n_batches = len(loader)
    t0 = time.time()

    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        tf_ratio = compute_sampling_prob(global_step, cl_decay_steps)
        pred = model(x, y_true=y, teacher_forcing_ratio=tf_ratio)
        loss = criterion(pred, y)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item()
        global_step += 1

        if writer is not None and (global_step % tb_log_every == 0):
            log_scalar(writer, wandb_run, "train/loss_step", loss.item(), global_step)
            log_scalar(writer, wandb_run, "train/teacher_forcing_ratio", tf_ratio, global_step)
            log_scalar(writer, wandb_run, "train/grad_norm", float(grad_norm), global_step)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_batches - i - 1)
            print(f"    batch {i+1:>5d}/{n_batches}  loss={loss.item():.4f}  "
                  f"tf_ratio={tf_ratio:.3f}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    return total / n_batches, global_step


@torch.no_grad()
def evaluate(model, loader, scalers, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        pred = model(x.to(device), y_true=None, teacher_forcing_ratio=0.0)
        preds.append(pred.cpu())
        targets.append(y)
    return compute_metrics(torch.cat(preds), torch.cat(targets), scalers)


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       default="data/processed_yrd")
    p.add_argument("--hist_len",   type=int,   default=24)
    p.add_argument("--pred_len",   type=int,   default=24)
    p.add_argument("--hid_dim",    type=int,   default=64)
    p.add_argument("--num_layers", type=int,   default=2)
    p.add_argument("--max_diffusion_step", type=int, default=2)
    p.add_argument("--filter_type", default="dual_random_walk",
                   choices=["random_walk", "dual_random_walk"])
    p.add_argument("--batch_size", type=int,   default=64)
    p.add_argument("--lr",         type=float, default=1e-2)   # DCRNN paper default
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--patience",   type=int,   default=15)
    p.add_argument("--cl_decay_steps", type=int, default=2000)  # scheduled sampling decay
    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--workers",    type=int,   default=4)

    # ── logging ──
    p.add_argument("--log_dir",      default="runs",
                    help="Root directory for TensorBoard event files")
    p.add_argument("--exp_name",     default=None,
                    help="Experiment name (subfolder under --log_dir). "
                         "Default: timestamp + hist/pred lengths, e.g. 20260831-142200_h24_p24")
    p.add_argument("--tb_log_every", type=int, default=10,
                    help="Log a training scalar every N optimizer steps")
    p.add_argument("--plot_every",   type=int, default=5,
                    help="Log a prediction-vs-truth figure every N epochs (0 to disable)")
    p.add_argument("--wandb",         action="store_true",
                    help="Also mirror scalars to Weights & Biases (optional)")
    p.add_argument("--wandb_project", default="dcrnn-knowair")
    p.add_argument("--wandb_entity",  default=None)
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()

    if args.exp_name is None:
        args.exp_name = f"{time.strftime('%Y%m%d-%H%M%S')}_h{args.hist_len}_p{args.pred_len}"
    log_path = os.path.join(args.log_dir, args.exp_name)
    writer = SummaryWriter(log_dir=log_path)

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                                name=args.exp_name, config=vars(args))

    print(f"Device   : {device}")
    print(f"Data     : {args.data}")
    print(f"Hist/Pred: {args.hist_len}h / {args.pred_len}h")
    print(f"LR       : {args.lr}")
    print(f"TB logdir: {log_path}   (run: tensorboard --logdir {args.log_dir})\n")

    writer.add_text("config", "  \n".join(f"**{k}**: {v}" for k, v in vars(args).items()), 0)

    print("Loading data ...")
    loaders, adj_mx, scalers, N, F = get_loaders(
        args.data, args.hist_len, args.pred_len,
        args.batch_size, args.workers,
    )

    with open(os.path.join(args.data, "split.pkl"), "rb") as f:
        split = pickle.load(f)
    print(f"\nData split:")
    print(f"  Train: {split['train_dates'][0]} → {split['train_dates'][1]}")
    print(f"  Val  : {split['val_dates'][0]}   → {split['val_dates'][1]}")
    print(f"  Test : {split['test_dates'][0]}  → {split['test_dates'][1]}")

    print("\nBuilding DCRNN ...")
    model = DCRNN(
        num_nodes=N, input_dim=F, output_dim=1,
        hid_dim=args.hid_dim, num_layers=args.num_layers,
        max_diffusion_step=args.max_diffusion_step,
        pred_len=args.pred_len, adj_mx=adj_mx, filter_type=args.filter_type,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    writer.add_text("config", f"**n_params**: {n_params:,}", 1)

    optimizer = Adam(model.parameters(), lr=args.lr, eps=1e-3)
    # DCRNN paper uses MultiStepLR: lr decays at specific epochs
    scheduler = MultiStepLR(optimizer, milestones=[20, 30, 40, 50], gamma=0.1)
    criterion = nn.L1Loss()   # original paper uses MAE loss

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"dcrnn_faithful_{args.exp_name}_best.pt")

    best_val_mae = float("inf")
    patience_cnt = 0
    global_step  = 0

    print("\nEpoch   Train-MAE    Val-MAE   Val-RMSE  Val-MAPE      LR      Time")
    print("─" * 72)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, global_step = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device,
            global_step, args.cl_decay_steps,
            writer=writer, wandb_run=wandb_run, tb_log_every=args.tb_log_every,
        )
        val_metrics = evaluate(model, loaders["val"], scalers, device)
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

        # ── per-epoch TensorBoard / W&B logging ──
        log_scalar(writer, wandb_run, "epoch/train_MAE", train_loss, epoch)
        log_scalar(writer, wandb_run, "epoch/val_MAE",   val_metrics["MAE"],  epoch)
        log_scalar(writer, wandb_run, "epoch/val_RMSE",  val_metrics["RMSE"], epoch)
        log_scalar(writer, wandb_run, "epoch/val_MAPE",  val_metrics["MAPE"], epoch)
        log_scalar(writer, wandb_run, "epoch/lr",        cur_lr, epoch)
        log_scalar(writer, wandb_run, "epoch/time_sec",  elapsed, epoch)

        if args.plot_every and epoch % args.plot_every == 0:
            log_prediction_figure(writer, wandb_run, model, loaders["val"], scalers, device, epoch)

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
    test_metrics = evaluate(model, loaders["test"], scalers, device)
    print("\n── Test Results ──────────────────────────────")
    print(f"  MAE  : {test_metrics['MAE']:.3f} µg/m³")
    print(f"  RMSE : {test_metrics['RMSE']:.3f} µg/m³")
    print(f"  MAPE : {test_metrics['MAPE']:.2f}%")
    print("─────────────────────────────────────────────")

    # ── final hparams summary (lets you compare 24h vs 72h runs side-by-side
    #     in TensorBoard's HPARAMS tab) ──
    hparam_dict = {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))}
    metric_dict = {
        "hparam/test_MAE": test_metrics["MAE"],
        "hparam/test_RMSE": test_metrics["RMSE"],
        "hparam/test_MAPE": test_metrics["MAPE"],
        "hparam/best_val_MAE": best_val_mae,
    }
    writer.add_hparams(hparam_dict, metric_dict)

    if wandb_run is not None:
        wandb_run.log(metric_dict)
        wandb_run.summary.update(metric_dict)
        wandb_run.finish()
    writer.close()


if __name__ == "__main__":
    main()