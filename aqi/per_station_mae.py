"""
per_station_mae.py
===================
計算「每個測站」各自的 Test-MAE / RMSE（不是全站平均），並標記台南目標站。

支援全部五個已訓練模型：pm25gnn, dcrnn, agcrn, airphynet, gts

Usage:
    python per_station_mae.py --model pm25gnn
    python per_station_mae.py --model dcrnn
    python per_station_mae.py --model agcrn
    python per_station_mae.py --model airphynet
    python per_station_mae.py --model gts
    python per_station_mae.py --model all        # 五個模型都跑，輸出合併表
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))

# 台南目標測站（跟 fetch_aqi_southwest.py 一致）
TAINAN_SITES = {"善化", "安南", "新營", "林森", "臺南"}

DATA_DIR = "data/processed_southwest"
CKPT_DIR = "checkpoints"


# ── 共用工具 ────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():        return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_station_meta(data_dir):
    """回傳 station 名稱 list（跟 features.npy 的 N 維度順序一致）+ is_target 遮罩"""
    stations = pd.read_csv(os.path.join(data_dir, "stations.csv"))
    names = stations["sitename"].tolist()
    is_target = np.array([1 if s in TAINAN_SITES else 0 for s in names])
    return names, is_target


def per_station_metrics(preds_norm, targets_norm, pm25_mean, pm25_std):
    """
    preds_norm, targets_norm: [total_samples, pred_len, N, 1]  (normalized)
    Returns: mae[N], rmse[N]  (physical units, averaged over samples & pred_len)
    """
    pred   = preds_norm.squeeze(-1)   * pm25_std + pm25_mean   # [S, T, N]
    target = targets_norm.squeeze(-1) * pm25_std + pm25_mean

    err = pred - target                       # [S, T, N]
    mae  = err.abs().mean(dim=(0, 1)).numpy()          # [N]
    rmse = (err ** 2).mean(dim=(0, 1)).sqrt().numpy()   # [N]
    return mae, rmse


def print_table(model_name, names, is_target, mae, rmse):
    print(f"\n{'='*55}")
    print(f"  {model_name} — 各測站 Test 表現")
    print(f"{'='*55}")

    order = np.argsort(mae)   # 由準到不準排序
    print(f"{'測站':<8}{'台南目標':<10}{'MAE (µg/m³)':>14}{'RMSE (µg/m³)':>15}")
    print("-" * 55)
    for idx in order:
        tag = "★ 台南" if is_target[idx] else ""
        print(f"{names[idx]:<8}{tag:<10}{mae[idx]:>14.3f}{rmse[idx]:>15.3f}")

    target_mask = is_target.astype(bool)
    print("-" * 55)
    print(f"{'全站平均':<8}{'':<10}{mae.mean():>14.3f}{rmse.mean():>15.3f}")
    print(f"{'台南5站平均':<8}{'':<10}{mae[target_mask].mean():>14.3f}{rmse[target_mask].mean():>15.3f}")
    print(f"{'其他站平均':<8}{'':<10}{mae[~target_mask].mean():>14.3f}{rmse[~target_mask].mean():>15.3f}")


# ── 各模型的載入 + 推論邏輯 ──────────────────────────────────────────────────

@torch.no_grad()
def run_pm25gnn(device):
    from training.aqi.models.pm25gnn import PM25GNN
    from training.aqi.pm25gnn_dataset import get_pm25gnn_loaders

    ckpt = torch.load(os.path.join(CKPT_DIR, "pm25gnn_faithful_best.pt"),
                       map_location=device, weights_only=True)
    args = ckpt["args"]

    loaders, edge_index, edge_attr, scalers, wind_mean, wind_std, N, F = \
        get_pm25gnn_loaders(DATA_DIR, hist_len=args["hist_len"], pred_len=args["pred_len"],
                            batch_size=args["batch_size"])

    model = PM25GNN(
        hist_len=args["hist_len"], pred_len=args["pred_len"], in_dim=F,
        city_num=N, batch_size=args["batch_size"], device=device,
        edge_index=edge_index, edge_attr=edge_attr,
        wind_mean=wind_mean, wind_std=wind_std,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, targets = [], []
    for pm25_hist, feature, pm25_label in loaders["test"]:
        pred = model(pm25_hist.to(device), feature.to(device))
        preds.append(pred.cpu())
        targets.append(pm25_label)

    pm25_mean, pm25_std = float(scalers["mean"][0]), float(scalers["std"][0])
    return torch.cat(preds), torch.cat(targets), pm25_mean, pm25_std


@torch.no_grad()
def run_dcrnn(device):
    from training.aqi.dcrnn_faithful import DCRNN
    from training.aqi.train_dcrnn import get_loaders

    ckpt = torch.load(os.path.join(CKPT_DIR, "dcrnn_faithful_best.pt"),
                       map_location=device, weights_only=True)
    args = ckpt["args"]

    loaders, adj_mx, scalers, N, F = get_loaders(
        DATA_DIR, args["hist_len"], args["pred_len"], args["batch_size"])

    model = DCRNN(
        num_nodes=N, input_dim=F, output_dim=1,
        hid_dim=args["hid_dim"], num_layers=args["num_layers"],
        max_diffusion_step=args["max_diffusion_step"],
        pred_len=args["pred_len"], adj_mx=adj_mx, filter_type=args["filter_type"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, targets = [], []
    for x, y in loaders["test"]:
        pred = model(x.to(device), y_true=None, teacher_forcing_ratio=0.0)
        preds.append(pred.cpu())
        targets.append(y)

    pm25_mean, pm25_std = float(scalers["mean"][0]), float(scalers["std"][0])
    return torch.cat(preds), torch.cat(targets), pm25_mean, pm25_std


@torch.no_grad()
def run_agcrn(device):
    from training.aqi.agcrn_faithful import AGCRN
    from training.aqi.train_agcrn import get_loaders

    ckpt = torch.load(os.path.join(CKPT_DIR, "agcrn_faithful_best.pt"),
                       map_location=device, weights_only=True)
    args = ckpt["args"]

    loaders, scalers, N, F = get_loaders(
        DATA_DIR, args["hist_len"], args["pred_len"], args["batch_size"])

    model = AGCRN(
        num_nodes=N, in_features=F, out_dim=1,
        hidden_dim=args["hidden_dim"], embed_dim=args["embed_dim"],
        cheb_k=args["cheb_k"], num_layers=args["num_layers"],
        pred_len=args["pred_len"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, targets = [], []
    for x, y in loaders["test"]:
        pred = model(x.to(device))
        preds.append(pred.cpu())
        targets.append(y)

    pm25_mean, pm25_std = float(scalers["mean"][0]), float(scalers["std"][0])
    return torch.cat(preds), torch.cat(targets), pm25_mean, pm25_std


@torch.no_grad()
def run_airphynet(device):
    from training.aqi.airphynet_faithful import AirPhyNetModel
    from training.aqi.train_airphynet import get_loaders

    ckpt = torch.load(os.path.join(CKPT_DIR, "airphynet_faithful_best.pt"),
                       map_location=device, weights_only=True)
    args = ckpt["args"]

    loaders, adj_mx_np, adj_wind_np, scalers, N, F = get_loaders(
        DATA_DIR, args["hist_len"], args["pred_len"], args["batch_size"])
    adj_mx   = torch.FloatTensor(adj_mx_np).to(device)
    adj_wind = torch.FloatTensor(adj_wind_np).to(device)

    model = AirPhyNetModel(
        num_nodes=N, in_features=F, hidden_dim=args["hidden_dim"],
        pred_len=args["pred_len"], latent_dim=args["latent_dim"],
        n_traj=args["n_traj"], ode_method=args["ode_method"],
        ode_rtol=args["ode_rtol"], ode_atol=args["ode_atol"],
        gen_dim=args["gen_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, targets = [], []
    for x, y in loaders["test"]:
        pred = model(x.to(device), adj_mx, adj_wind)
        preds.append(pred.cpu())
        targets.append(y)

    pm25_mean, pm25_std = float(scalers["mean"][0]), float(scalers["std"][0])
    return torch.cat(preds), torch.cat(targets), pm25_mean, pm25_std


@torch.no_grad()
def run_gts(device):
    from training.aqi.gts_faithful import GTS
    from training.aqi.train_gts import get_loaders

    ckpt = torch.load(os.path.join(CKPT_DIR, "gts_faithful_best.pt"),
                       map_location=device, weights_only=True)
    args = ckpt["args"]

    loaders, node_series, scalers, N, F = get_loaders(
        DATA_DIR, args["hist_len"], args["pred_len"], args["batch_size"])
    node_series = node_series.to(device)

    model = GTS(
        num_nodes=N, input_dim=F, output_dim=1,
        hid_dim=args["hid_dim"], num_layers=args["num_layers"],
        max_diffusion_step=args["max_diffusion_step"],
        pred_len=args["pred_len"], train_seq_len=args["hist_len"],
        embed_dim=args["embed_dim"], temperature=args["temperature"],
        filter_type=args["filter_type"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    preds, targets = [], []
    for x, y in loaders["test"]:
        pred, _, _ = model(x.to(device), node_series, y_true=None, teacher_forcing_ratio=0.0)
        preds.append(pred.cpu())
        targets.append(y)

    pm25_mean, pm25_std = float(scalers["mean"][0]), float(scalers["std"][0])
    return torch.cat(preds), torch.cat(targets), pm25_mean, pm25_std


RUNNERS = {
    "pm25gnn":   ("PM2.5-GNN", run_pm25gnn),
    "dcrnn":     ("DCRNN",     run_dcrnn),
    "agcrn":     ("AGCRN",     run_agcrn),
    "airphynet": ("AirPhyNet", run_airphynet),
    "gts":       ("GTS",       run_gts),
}


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=list(RUNNERS.keys()) + ["all"])
    args = p.parse_args()

    device = get_device()
    print(f"Device: {device}")

    names, is_target = load_station_meta(DATA_DIR)

    to_run = list(RUNNERS.keys()) if args.model == "all" else [args.model]

    summary = {}
    for key in to_run:
        display_name, runner_fn = RUNNERS[key]
        print(f"\n載入並評估 {display_name} ...")
        try:
            preds, targets, pm25_mean, pm25_std = runner_fn(device)
        except FileNotFoundError as e:
            print(f"  ⚠ 找不到 checkpoint，跳過：{e}")
            continue

        mae, rmse = per_station_metrics(preds, targets, pm25_mean, pm25_std)
        print_table(display_name, names, is_target, mae, rmse)

        summary[display_name] = {
            "全站平均MAE": mae.mean(),
            "台南5站MAE": mae[is_target.astype(bool)].mean(),
        }

    if len(summary) > 1:
        print(f"\n{'='*55}")
        print("  五模型摘要比較（全站 vs 台南5站）")
        print(f"{'='*55}")
        print(f"{'模型':<14}{'全站平均MAE':>14}{'台南5站MAE':>14}")
        print("-" * 55)
        for name, m in summary.items():
            print(f"{name:<14}{m['全站平均MAE']:>14.3f}{m['台南5站MAE']:>14.3f}")


if __name__ == "__main__":
    main()