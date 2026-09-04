"""
weight_interpolation.py
=========================
WiSE-FT 風格的權重空間內插。用「完全沒動過的預訓練權重」(theta0)跟「已經
fine-tune 過的權重」(theta1),依比例 alpha 內插: theta = (1-alpha)*theta0 + alpha*theta1,
掃過一輪 alpha 找出在 val 上最好的混合比例,再看它在 test 上跟純預訓練模型(alpha=0)
的對照。

supports_stack(圖本身)不參與內插——它不是學到的權重,是站點的地理事實,兩邊模型
只要 N 相同就會是同一個值,直接用當前模型自己建構出來的即可。

不需要重新訓練,直接拿你已經產生的 checkpoint 掃描,很便宜(每個 alpha 只是
forward 過一次 val set)。

用法:
    python weight_interpolation.py \
        --pretrained_ckpt ../checkpoints_yrd/dcrnn_faithful_best.pt \
        --finetuned_ckpt checkpoints_finetune/dcrnn_finetuned_xxx_best.pt \
        --data data/processed_yrd_finetune
"""
import argparse
import os

import torch

from dcrnn_faithful import DCRNN
from finetune_dcrnn import get_loaders, evaluate, get_device


def interpolate_state_dicts(theta0, theta1, alpha):
    """回傳 (1-alpha)*theta0 + alpha*theta1,跳過 supports_stack(圖本身,不是學到的權重)。"""
    out = {}
    for k in theta1.keys():
        if k.endswith("supports_stack"):
            continue
        if k not in theta0:
            raise KeyError(f"{k} 在 pretrained checkpoint 裡找不到,兩個 checkpoint 架構可能不一致")
        out[k] = (1 - alpha) * theta0[k] + alpha * theta1[k]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_ckpt", required=True)
    ap.add_argument("--finetuned_ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ap.add_argument("--save_dir", default="checkpoints_finetune")
    args = ap.parse_args()

    device = get_device()

    ckpt0 = torch.load(args.pretrained_ckpt, map_location=device, weights_only=True)
    ckpt1 = torch.load(args.finetuned_ckpt, map_location=device, weights_only=True)
    a1 = ckpt1["args"]

    hist_len, pred_len = a1["hist_len"], a1["pred_len"]
    loaders, adj_mx, scalers, N, F = get_loaders(args.data, hist_len, pred_len, args.batch_size)
    print(f"站點數 N={N}, feature 數 F={F}")

    def build_model():
        return DCRNN(
            num_nodes=N, input_dim=F, output_dim=1,
            hid_dim=a1["hid_dim"], num_layers=a1["num_layers"],
            max_diffusion_step=a1["max_diffusion_step"],
            pred_len=pred_len, adj_mx=adj_mx, filter_type=a1["filter_type"],
        ).to(device)

    print(f"\n{'alpha':>8s}{'Val MAE':>12s}{'Val RMSE':>12s}   說明")
    print("─" * 55)
    results = []
    for alpha in args.alphas:
        model = build_model()
        interp = interpolate_state_dicts(ckpt0["model"], ckpt1["model"], alpha)
        missing, unexpected = model.load_state_dict(interp, strict=False)
        bad = [k for k in list(missing) + list(unexpected) if not k.endswith("supports_stack")]
        if bad:
            raise RuntimeError(f"權重載入出現非預期缺漏: {bad}")

        val_metrics = evaluate(model, loaders["val"], scalers, device)
        note = "(= 原始預訓練模型)" if alpha == 0.0 else ("(= 純 fine-tune 後模型)" if alpha == 1.0 else "")
        results.append((alpha, val_metrics, {k: v.clone() for k, v in model.state_dict().items()}))
        print(f"{alpha:>8.2f}{val_metrics['MAE']:>12.3f}{val_metrics['RMSE']:>12.3f}   {note}")

    best_alpha, best_val, best_state = min(results, key=lambda r: r[1]["MAE"])
    baseline_result = next(r for r in results if r[0] == 0.0)
    print(f"\n最佳 alpha = {best_alpha:.2f}  (val MAE={best_val['MAE']:.3f}, "
          f"原始模型 val MAE={baseline_result[1]['MAE']:.3f})")

    if best_alpha == 0.0:
        print("\n⚠️  alpha=0(完全不用 fine-tune 的方向)本身就是 val 上最好的選擇,"
              "代表這份 fine-tune checkpoint 學到的方向對這批資料沒有任何幫助,"
              "內插也救不回來——建議換一份 fine-tune checkpoint 試試看,或直接放棄 fine-tune。")
        return

    model = build_model()
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, loaders["test"], scalers, device)

    model0 = build_model()
    model0.load_state_dict(baseline_result[2])
    baseline_test = evaluate(model0, loaders["test"], scalers, device)

    mae_improve = (baseline_test["MAE"] - test_metrics["MAE"]) / baseline_test["MAE"] * 100
    rmse_improve = (baseline_test["RMSE"] - test_metrics["RMSE"]) / baseline_test["RMSE"] * 100

    print("\n" + "═" * 55)
    print(f"  alpha={best_alpha:.2f} vs 原始預訓練模型(alpha=0),同一份 test set 對照")
    print("═" * 55)
    print(f"{'':14s}{'MAE':>10s}{'RMSE':>10s}")
    print(f"{'alpha=0(原始)':14s}{baseline_test['MAE']:>10.3f}{baseline_test['RMSE']:>10.3f}")
    print(f"{'alpha='+f'{best_alpha:.2f}':14s}{test_metrics['MAE']:>10.3f}{test_metrics['RMSE']:>10.3f}")
    print("─" * 55)
    print(f"MAE  {'改善' if mae_improve > 0 else '變差'} {abs(mae_improve):.1f}%"
          f"  |  RMSE {'改善' if rmse_improve > 0 else '變差'} {abs(rmse_improve):.1f}%")
    print("═" * 55)

    if mae_improve > 0:
        save_path = os.path.join(args.save_dir, f"dcrnn_interpolated_alpha{best_alpha:.2f}.pt")
        torch.save({
            "model": best_state, "alpha": best_alpha, "args": a1,
            "val_metrics": best_val, "test_metrics": test_metrics,
            "pretrained_ckpt": args.pretrained_ckpt, "finetuned_ckpt": args.finetuned_ckpt,
        }, save_path)
        print(f"\n✅ 已存至 {save_path}")
    else:
        print("\n⚠️  這個 alpha 在 val 上比原始模型好,但搬到 test 上沒有跟著改善,"
              "代表可能是 val 選到的雜訊,不是真的改善,不建議直接採用。")


if __name__ == "__main__":
    main()