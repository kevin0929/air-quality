"""
test_gts.py
===========
只做測試（inference）：載入已訓練好的 gts_faithful_best.pt，
在 val / test split 上算 MAE / RMSE / MAPE，不做任何訓練。

模型超參數直接從 checkpoint 內的 args 還原，
所以不需要重新手動指定 hid_dim / embed_dim / temperature 等等。

Usage:
    python test_gts.py                              # 預設跑 val + test
    python test_gts.py --per_horizon                # 加印每個時步的指標
    python test_gts.py --splits test                # 只跑 test
    python test_gts.py --splits train val test      # 三個都跑
"""

import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from gts_faithful import GTS
from train_gts import GTSDataset, build_node_series, compute_metrics, get_device


# ── data ──────────────────────────────────────────────────────────────────────

def load_split_data(data_dir, split_names, hist_len, pred_len, batch_size, workers):
    """
    重建指定 split 的 loader。與訓練時的差別只有 drop_last=False，
    純推論沒理由丟掉最後一個不滿 batch 的樣本。
    """
    features = np.load(os.path.join(data_dir, "features.npy"))
    with open(os.path.join(data_dir, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    with open(os.path.join(data_dir, "split.pkl"), "rb") as f:
        split = pickle.load(f)

    loaders = {}
    for name in split_names:
        s, e = split[name]
        ds = GTSDataset(features, s, e, hist_len, pred_len)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                   num_workers=workers, drop_last=False,
                                   pin_memory=True)
        print(f"  {name:<5}: {len(ds):>7,} samples   "
              f"{split[name + '_dates'][0]} → {split[name + '_dates'][1]}")

    # 圖結構的參考序列必須和訓練時一模一樣：訓練 split 的前 hist_len 步
    train_s, train_e = split["train"]
    node_series = build_node_series(features, train_s, train_e, hist_len)

    print(f"  Stations: {features.shape[1]}   Features: {features.shape[2]}")

    return loaders, node_series, scalers, features.shape[1], features.shape[2]


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader, node_series, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        pred, _, _ = model(x.to(device), node_series,
                           y_true=None, teacher_forcing_ratio=0.0)
        preds.append(pred.cpu())
        targets.append(y)
    return torch.cat(preds), torch.cat(targets)   # [B, pred_len, N, 1]


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",  default="checkpoints/gts_faithful_best.pt")
    p.add_argument("--data",  default=None,
                   help="預設沿用 checkpoint 內記錄的 data 路徑")
    p.add_argument("--splits", nargs="+", default=["val", "test"],
                   choices=["train", "val", "test"])
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--per_horizon", action="store_true",
                   help="額外印出每個預測時步 (1h ~ pred_len h) 的指標")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cli    = parse_args()
    device = get_device()

    ckpt = torch.load(cli.ckpt, map_location=device, weights_only=True)
    targs = ckpt["args"]          # 訓練時的完整超參數

    data_dir   = cli.data       or targs["data"]
    batch_size = cli.batch_size or targs["batch_size"]

    print(f"Device      : {device}")
    print(f"Checkpoint  : {cli.ckpt}  (epoch {ckpt['epoch']})")
    print(f"Data        : {data_dir}")
    print(f"Hist/Pred   : {targs['hist_len']}h / {targs['pred_len']}h")
    print(f"訓練時記錄的 val: MAE {ckpt['val_metrics']['MAE']:.3f} | "
          f"RMSE {ckpt['val_metrics']['RMSE']:.3f} | "
          f"MAPE {ckpt['val_metrics']['MAPE']:.2f}%\n")

    print("Loading data ...")
    loaders, node_series, scalers, N, F = load_split_data(
        data_dir, cli.splits, targs["hist_len"], targs["pred_len"],
        batch_size, cli.workers,
    )
    node_series = node_series.to(device)

    print("\nBuilding GTS ...")
    model = GTS(
        num_nodes=N, input_dim=F, output_dim=1,
        hid_dim=targs["hid_dim"], num_layers=targs["num_layers"],
        max_diffusion_step=targs["max_diffusion_step"],
        pred_len=targs["pred_len"], train_seq_len=targs["hist_len"],
        embed_dim=targs["embed_dim"], temperature=targs["temperature"],
        filter_type=targs["filter_type"],
    ).to(device)
    model.load_state_dict(ckpt["model"])

    results = {}
    for name in cli.splits:
        print(f"Running inference on {name} ...")
        pred, target = predict(model, loaders[name], node_series, device)
        results[name] = (compute_metrics(pred, target, scalers), pred, target)

    print("\n── Results ───────────────────────────────────")
    print("Split       MAE      RMSE      MAPE")
    print("─" * 45)
    for name in cli.splits:
        m = results[name][0]
        print(f"{name:<6}  {m['MAE']:>7.3f}  {m['RMSE']:>8.3f}  {m['MAPE']:>7.2f}%")
    print("─" * 45)
    print("(MAE / RMSE 單位為 µg/m³)")

    if cli.per_horizon:
        for name in cli.splits:
            _, pred, target = results[name]
            print(f"\n── {name} per-horizon ──────────────")
            print("Horizon    MAE     RMSE     MAPE")
            print("─" * 36)
            for h in range(targs["pred_len"]):
                m = compute_metrics(pred[:, h:h + 1], target[:, h:h + 1], scalers)
                print(f"{h + 1:>4d}h  {m['MAE']:>7.3f}  {m['RMSE']:>7.3f}  "
                      f"{m['MAPE']:>6.2f}%")


if __name__ == "__main__":
    main()