"""
diagnose_seasonal_error.py
===========================
把 test set 的預測誤差按月份拆開來看,驗證「fine-tune 後在 test 上變差,
是不是因為 train 跟 test 的季節分布不對稱」這個假設。

用法:
    python diagnose_seasonal_error.py \
        --pretrained_ckpt ../checkpoints_yrd/dcrnn_faithful_best.pt \
        --finetuned_ckpt checkpoints_finetune/dcrnn_finetuned_xxx_best.pt \
        --data data/processed_yrd_finetune
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import torch

from dcrnn_faithful import DCRNN
from finetune_dcrnn import DCRNNDataset, compute_metrics, get_device


def load_model(ckpt_path, adj_mx, N, F, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    a = ckpt["args"]
    model = DCRNN(
        num_nodes=N, input_dim=F, output_dim=1,
        hid_dim=a["hid_dim"], num_layers=a["num_layers"],
        max_diffusion_step=a["max_diffusion_step"],
        pred_len=a["pred_len"], adj_mx=adj_mx, filter_type=a["filter_type"],
    ).to(device)
    filtered = {k: v for k, v in ckpt["model"].items() if not k.endswith("supports_stack")}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model, a["hist_len"], a["pred_len"]


@torch.no_grad()
def predict_all(model, loader, device):
    preds, targets = [], []
    for x, y in loader:
        pred = model(x.to(device), y_true=None, teacher_forcing_ratio=0.0)
        preds.append(pred.cpu())
        targets.append(y)
    return torch.cat(preds), torch.cat(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--finetuned_ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    device = get_device()
    features = np.load(os.path.join(args.data, "features.npy"))
    adj_mx = np.load(os.path.join(args.data, "adj_mx.npy"))
    with open(os.path.join(args.data, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    with open(os.path.join(args.data, "split.pkl"), "rb") as f:
        split = pickle.load(f)
    timestamps = pd.read_csv(os.path.join(args.data, "timestamps.csv"))["timestamp"].astype("datetime64[ns]")

    N, F = features.shape[1], features.shape[2]

    pre_model, hist_len, pred_len = load_model(args.pretrained_ckpt, adj_mx, N, F, device)
    ft_model, _, _ = load_model(args.finetuned_ckpt, adj_mx, N, F, device)

    from torch.utils.data import DataLoader
    test_ds = DCRNNDataset(features, split["test"], hist_len, pred_len)
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 每個 sample 對應的月份,用該 sample 預測窗口第一個時間點的月份代表
    sample_months = [timestamps.iloc[test_ds.indices[i] + hist_len].month for i in range(len(test_ds))]

    print("計算 fine-tune 前的預測 ...")
    pre_pred, pre_target = predict_all(pre_model, loader, device)
    print("計算 fine-tune 後的預測 ...")
    ft_pred, ft_target = predict_all(ft_model, loader, device)

    pm25_mean = float(scalers["mean"][0])
    pm25_std = float(scalers["std"][0])

    def denorm(t):
        return t.squeeze(-1) * pm25_std + pm25_mean

    pre_mae_per_sample = (denorm(pre_pred) - denorm(pre_target)).abs().mean(dim=(1, 2))
    ft_mae_per_sample = (denorm(ft_pred) - denorm(ft_target)).abs().mean(dim=(1, 2))

    df = pd.DataFrame({
        "month": sample_months,
        "mae_before": pre_mae_per_sample.numpy(),
        "mae_after": ft_mae_per_sample.numpy(),
    })
    by_month = df.groupby("month").agg(
        n_samples=("mae_before", "size"),
        mae_before=("mae_before", "mean"),
        mae_after=("mae_after", "mean"),
    )
    by_month["change_pct"] = (by_month["mae_before"] - by_month["mae_after"]) / by_month["mae_before"] * 100

    print(f"\n{'='*60}")
    print("  Test set 按月份拆解:fine-tune 前後 MAE 對照")
    print(f"{'='*60}")
    print(f"{'月份':6s}{'樣本數':>8s}{'Fine-tune前':>14s}{'Fine-tune後':>14s}{'變化':>10s}")
    for month, row in by_month.iterrows():
        arrow = "改善" if row["change_pct"] > 0 else "變差"
        print(f"{month:>4d}  {int(row['n_samples']):>8d}"
              f"{row['mae_before']:>14.3f}{row['mae_after']:>14.3f}"
              f"  {arrow}{abs(row['change_pct']):5.1f}%")
    print(f"{'='*60}")

    winter_months = {11, 12, 1, 2}
    is_winter = by_month.index.isin(winter_months)
    if is_winter.any() and (~is_winter).any():
        winter_change = by_month.loc[is_winter, "change_pct"].mean()
        other_change = by_month.loc[~is_winter, "change_pct"].mean()
        print(f"\n冬季月份(11-2月)平均變化: {winter_change:+.1f}%")
        print(f"其餘月份平均變化:         {other_change:+.1f}%")
        if winter_change < other_change - 2:
            print("\n→ 冬季惡化明顯比其他月份嚴重,支持「train/test 季節分布不對稱」這個假設。")
        else:
            print("\n→ 冬季跟其他月份的變化幅度差不多,惡化可能不是單純季節性問題,建議檢查學習率/資料量。")


if __name__ == "__main__":
    main()