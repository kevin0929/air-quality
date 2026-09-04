"""
test_models.py
==============
只做推論（不訓練）：載入已訓練好的 checkpoint，在 val / test split 上
算 MAE / RMSE / MAPE。支援 agcrn / dcrnn / gts / airphynet。

模型超參數全部從 checkpoint 內的 args 還原，不需要手動重打一次。
model 相關的 import 是延遲的：跑 agcrn 不會去 import torchdiffeq。

Usage:
    python test_models.py --model agcrn
    python test_models.py --model dcrnn --per_horizon
    python test_models.py --model gts --splits test
    python test_models.py --model airphynet --ckpt checkpoints/xxx.pt
"""

import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


DEFAULT_CKPT = {
    "agcrn":     "checkpoints_yrd/agcrn_faithful_best.pt",
    "dcrnn":     "checkpoints_yrd/dcrnn_faithful_best.pt",
    "gts":       "checkpoints_yrd/gts_faithful_best.pt",
    "airphynet": "checkpoints_yrd/airphynet_faithful_best.pt",
}


# ── device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── dataset / metrics（與四支訓練腳本中的實作完全一致）────────────────────────

class WindowDataset(Dataset):
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


# ── data ──────────────────────────────────────────────────────────────────────

def load_split_data(data_dir, split_names, hist_len, pred_len, batch_size, workers):
    """
    與訓練時唯一的差別是 drop_last=False：純推論沒有理由丟掉
    最後一個不滿 batch 的樣本。
    """
    features = np.load(os.path.join(data_dir, "features.npy"))
    with open(os.path.join(data_dir, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    with open(os.path.join(data_dir, "split.pkl"), "rb") as f:
        split = pickle.load(f)

    loaders = {}
    for name in split_names:
        s, e = split[name]
        ds = WindowDataset(features, s, e, hist_len, pred_len)
        loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False,
                                   num_workers=workers, drop_last=False,
                                   pin_memory=True)
        print(f"  {name:<5}: {len(ds):>7,} samples   "
              f"{split[name + '_dates'][0]} → {split[name + '_dates'][1]}")

    print(f"  Stations: {features.shape[1]}   Features: {features.shape[2]}")
    return loaders, features, split, scalers, features.shape[1], features.shape[2]


def build_node_series(features, train_start, hist_len):
    """
    GTS 專用：feature extractor 用來學圖結構的固定參考序列。
    必須和 train_gts.py 的 build_node_series 完全一致
    ——訓練 split 的前 hist_len 步——否則學到的圖對不上權重。
    """
    series = features[train_start:train_start + hist_len, :, 0]   # [hist_len, N]
    series = series.T[:, None, :]                                 # [N, 1, hist_len]
    return torch.FloatTensor(series)


# ── model 建構（每個模型只差在這裡）────────────────────────────────────────────

def build_model(name, targs, N, F, data_dir, features, split, device):
    """回傳 (model, forward_fn)。forward_fn(x) -> pred [B, pred_len, N, 1]"""

    if name == "agcrn":
        from agcrn_faithful import AGCRN
        model = AGCRN(
            num_nodes=N, in_features=F, out_dim=1,
            hidden_dim=targs["hidden_dim"], embed_dim=targs["embed_dim"],
            cheb_k=targs["cheb_k"], num_layers=targs["num_layers"],
            pred_len=targs["pred_len"],
        ).to(device)
        forward = lambda x: model(x)

    elif name == "dcrnn":
        from dcrnn_faithful import DCRNN
        adj_mx = np.load(os.path.join(data_dir, "adj_mx.npy"))
        model = DCRNN(
            num_nodes=N, input_dim=F, output_dim=1,
            hid_dim=targs["hid_dim"], num_layers=targs["num_layers"],
            max_diffusion_step=targs["max_diffusion_step"],
            pred_len=targs["pred_len"], adj_mx=adj_mx,
            filter_type=targs["filter_type"],
        ).to(device)
        forward = lambda x: model(x, y_true=None, teacher_forcing_ratio=0.0)

    elif name == "gts":
        from gts_faithful import GTS
        model = GTS(
            num_nodes=N, input_dim=F, output_dim=1,
            hid_dim=targs["hid_dim"], num_layers=targs["num_layers"],
            max_diffusion_step=targs["max_diffusion_step"],
            pred_len=targs["pred_len"], train_seq_len=targs["hist_len"],
            embed_dim=targs["embed_dim"], temperature=targs["temperature"],
            filter_type=targs["filter_type"],
        ).to(device)
        node_series = build_node_series(features, split["train"][0],
                                        targs["hist_len"]).to(device)
        forward = lambda x: model(x, node_series, y_true=None,
                                  teacher_forcing_ratio=0.0)[0]

    elif name == "airphynet":
        from airphynet_faithful import AirPhyNetModel
        adj_mx = torch.FloatTensor(
            np.load(os.path.join(data_dir, "adj_mx.npy"))).to(device)
        adj_wind = torch.FloatTensor(
            np.load(os.path.join(data_dir, "adj_wind.npy"))).to(device)
        model = AirPhyNetModel(
            num_nodes=N, in_features=F, hidden_dim=targs["hidden_dim"],
            pred_len=targs["pred_len"], latent_dim=targs["latent_dim"],
            n_traj=targs["n_traj"], ode_method=targs["ode_method"],
            ode_rtol=targs["ode_rtol"], ode_atol=targs["ode_atol"],
            gen_dim=targs["gen_dim"],
        ).to(device)
        forward = lambda x: model(x, adj_mx, adj_wind)

    else:
        raise ValueError(f"unknown model: {name}")

    return model, forward


# ── inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, forward, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        preds.append(forward(x.to(device)).cpu())
        targets.append(y)
    return torch.cat(preds), torch.cat(targets)   # [B, pred_len, N, 1]


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["agcrn", "dcrnn", "gts", "airphynet"])
    p.add_argument("--ckpt", default=None,
                   help=f"預設 checkpoints/<model>_faithful_best.pt")
    p.add_argument("--data", default=None,
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

    ckpt_path = cli.ckpt or DEFAULT_CKPT[cli.model]
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=True)
    targs = ckpt["args"]                     # 訓練時的完整超參數

    data_dir   = cli.data       or targs["data"]
    batch_size = cli.batch_size or targs["batch_size"]

    print(f"Model       : {cli.model}")
    print(f"Device      : {device}")
    print(f"Checkpoint  : {ckpt_path}  (epoch {ckpt['epoch']})")
    print(f"Data        : {data_dir}")
    print(f"Hist/Pred   : {targs['hist_len']}h / {targs['pred_len']}h")
    v = ckpt["val_metrics"]
    print(f"訓練時記錄的 val: MAE {v['MAE']:.3f} | RMSE {v['RMSE']:.3f} | "
          f"MAPE {v['MAPE']:.2f}%\n")

    print("Loading data ...")
    loaders, features, split, scalers, N, F = load_split_data(
        data_dir, cli.splits, targs["hist_len"], targs["pred_len"],
        batch_size, cli.workers,
    )

    print(f"\nBuilding {cli.model} ...")
    model, forward = build_model(cli.model, targs, N, F,
                                 data_dir, features, split, device)
    model.load_state_dict(ckpt["model"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    results = {}
    for name in cli.splits:
        print(f"Running inference on {name} ...", flush=True)
        pred, target = predict(model, forward, loaders[name], device)
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