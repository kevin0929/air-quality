"""
finetune_dcrnn.py
==================
Fine-tune 一個已經在 KnowAir-V2(355 站)上訓練好的 DCRNN checkpoint,
接上 prepare_finetune_data.py 產生的新時段資料(格式跟 preprocess_knowair.py
輸出完全相容:features.npy / adj_mx.npy / scalers.pkl / split.pkl)。

核心原理(前面已經用真實 dcrnn_faithful.py 驗證過):
  DCRNN 的可學習權重(gconv_gate / gconv_candidate / output_layer)
  形狀只跟 hid_dim / input_dim / num_units 有關,跟 num_nodes 無關。
  唯一跟站點數綁定的是 supports_stack 這個 buffer(存的是圖本身),
  fine-tune 新站點集合時這個 buffer 本來就該用新的圖重建,不該沿用舊的。
  所以載入 checkpoint 時只需要濾掉 supports_stack,其餘權重原樣繼承。

用法:
    python finetune_dcrnn.py \
        --pretrained_ckpt checkpoints/dcrnn_faithful_xxx_best.pt \
        --data data/processed_yrd_finetune \
        --lr 1e-3 --epochs 30

    # 算力有限、只想輕量微調的話:
    python finetune_dcrnn.py --pretrained_ckpt ... --data ... --freeze_encoder
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dcrnn_faithful import DCRNN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DCRNNDataset(Dataset):
    PM25_IDX = 0

    def __init__(self, features, span, hist_len, pred_len):
        """span 可以是單一 (start, end) tuple(舊版一刀切格式),
        也可以是 [(s1,e1), (s2,e2), ...] 這種多段區間列表(月度分層切分格式)。
        兩種都支援,滑窗取樣不會跨越不同區間之間的邊界。
        """
        self.features = features
        self.hist_len = hist_len
        self.pred_len = pred_len
        window = hist_len + pred_len

        ranges = [span] if isinstance(span, tuple) else list(span)
        self.indices = []
        for start, end in ranges:
            self.indices.extend(range(start, end - window + 1))

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
    adj_mx = np.load(os.path.join(data_dir, "adj_mx.npy"))
    with open(os.path.join(data_dir, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    with open(os.path.join(data_dir, "split.pkl"), "rb") as f:
        split = pickle.load(f)

    train_ds = DCRNNDataset(features, split["train"], hist_len, pred_len)
    val_ds = DCRNNDataset(features, split["val"], hist_len, pred_len)
    test_ds = DCRNNDataset(features, split["test"], hist_len, pred_len)

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True, pin_memory=True),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                          num_workers=num_workers, drop_last=True, pin_memory=True),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, drop_last=True, pin_memory=True),
    }
    return loaders, adj_mx, scalers, features.shape[1], features.shape[2]


def compute_metrics(pred_norm, target_norm, scalers):
    pm25_mean = float(scalers["mean"][0])
    pm25_std = float(scalers["std"][0])
    pred = pred_norm.squeeze(-1) * pm25_std + pm25_mean
    target = target_norm.squeeze(-1) * pm25_std + pm25_mean
    mae = (pred - target).abs().mean().item()
    rmse = ((pred - target) ** 2).mean().sqrt().item()
    mask = target.abs() > 1.0
    mape = ((pred[mask] - target[mask]).abs() / target[mask].abs()).mean().item() * 100 \
           if mask.any() else float("nan")
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def load_pretrained_dcrnn(model, ckpt_state_dict, strict_check=True):
    """濾掉跟站點數綁定的 supports_stack,其餘權重原樣載入。

    已用真實 dcrnn_faithful.py + 兩個不同站點數的模型實測驗證過:
    missing/unexpected keys 應該剛好就是 4 個 supports_stack,不多不少。
    如果出現其他 key 對不上,代表模型架構本身不一致(不是站點數問題),直接報錯。
    """
    filtered = {k: v for k, v in ckpt_state_dict.items() if not k.endswith("supports_stack")}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    bad_missing = [k for k in missing if not k.endswith("supports_stack")]
    bad_unexpected = [k for k in unexpected if not k.endswith("supports_stack")]
    if strict_check and (bad_missing or bad_unexpected):
        raise RuntimeError(
            f"載入時出現非預期的 key 不對齊,不只是站點數問題,請檢查模型架構是否一致:\n"
            f"  missing (非 supports_stack): {bad_missing}\n"
            f"  unexpected (非 supports_stack): {bad_unexpected}"
        )
    return missing, unexpected


def compute_sampling_prob(global_step, cl_decay_steps):
    return cl_decay_steps / (cl_decay_steps + np.exp(global_step / cl_decay_steps))


def train_one_epoch(model, loader, optimizer, criterion, device, global_step, cl_decay_steps):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        tf_ratio = compute_sampling_prob(global_step, cl_decay_steps)
        pred = model(x, y_true=y, teacher_forcing_ratio=tf_ratio)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total += loss.item()
        global_step += 1
    return total / len(loader), global_step


@torch.no_grad()
def evaluate(model, loader, scalers, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        pred = model(x.to(device), y_true=None, teacher_forcing_ratio=0.0)
        preds.append(pred.cpu())
        targets.append(y)
    return compute_metrics(torch.cat(preds), torch.cat(targets), scalers)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_ckpt", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--hist_len", type=int, default=None,
                   help="預設沿用 checkpoint 存的 args,不用重複指定")
    p.add_argument("--pred_len", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="fine-tune 建議比原本預訓練的 lr(通常 1e-2)小,預設 1e-3")
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="L2 正則化,目前訓練曲線是「train loss 慢慢降、val 沒跟著改善」的典型過擬合徵兆,"
                        "加一點 weight decay(例如 1e-4)通常能緩解,讓權重不要偏離預訓練起點太多")
    p.add_argument("--swa_start_frac", type=float, default=0.5,
                   help="SWA(隨機權重平均):從訓練過程的第幾個比例開始,把每個 epoch 的權重平均起來"
                        "當作額外候選模型(常常比任何單一 epoch 的 checkpoint 更穩定)。"
                        "預設 0.5(後半段開始平均),設 >1 可以關掉這個功能。")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--cl_decay_steps", type=int, default=2000)
    p.add_argument("--save_dir", default="checkpoints_finetune")
    p.add_argument("--exp_name", default=None)
    p.add_argument("--log_dir", default="runs_finetune")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="只微調 decoder 的 output_layer,其餘凍結(算力有限時用)")
    p.add_argument("--save_last", action="store_true",
                   help="就算全程沒有任何一個 epoch 贏過 fine-tune 前的 val baseline,"
                        "也強制存最後一個 epoch 的權重並繼續產出前後對照——"
                        "用在你不在乎有沒有真的變好、只是想看/錄一段 fine-tune 前後數字變化的情境"
                        "(例如刻意用大 lr 讓它過擬合)。平常做真正的 fine-tune 不要加這個。")
    p.add_argument("--show_train_val", action="store_true",
                   help="預設只印/只比較 test 的結果(乾淨、聚焦)。加這個才會額外印出"
                        "train/val 的 baseline 數值,以及訓練後 train set 的前後對照。")
    p.add_argument("--no_interpolate", action="store_true",
                   help="訓練跑完後,預設會自動拿最終候選權重跟原始預訓練權重做一次內插掃描"
                        "(跟 weight_interpolation.py 同一套邏輯),取 val 上最好的比例當最終模型——"
                        "這通常比單純 fine-tune 的結果更穩定、更容易在 test 上也贏。加這個關掉這一步。")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print(f"載入預訓練 checkpoint: {args.pretrained_ckpt}")
    ckpt = torch.load(args.pretrained_ckpt, map_location=device, weights_only=True)
    orig_args = ckpt.get("args", {})
    if not orig_args:
        raise SystemExit("checkpoint 裡沒有存 'args',沒辦法自動還原模型架構,請手動確認 hid_dim/num_layers/...")

    hist_len = args.hist_len or orig_args["hist_len"]
    pred_len = args.pred_len or orig_args["pred_len"]
    hid_dim = orig_args["hid_dim"]
    num_layers = orig_args["num_layers"]
    max_diffusion_step = orig_args["max_diffusion_step"]
    filter_type = orig_args["filter_type"]

    print(f"從 checkpoint 還原架構: hid_dim={hid_dim}, num_layers={num_layers}, "
          f"max_diffusion_step={max_diffusion_step}, filter_type={filter_type}")
    print(f"hist_len={hist_len}, pred_len={pred_len} "
          f"({'沿用 checkpoint' if args.hist_len is None else '使用者指定覆蓋'})")

    print(f"\n載入 fine-tune 資料: {args.data}")
    loaders, adj_mx, scalers, N, F = get_loaders(args.data, hist_len, pred_len, args.batch_size)
    print(f"  站點數 N={N}, feature 數 F={F}")
    print(f"  Train samples: {len(loaders['train'].dataset):,}")
    print(f"  Val   samples: {len(loaders['val'].dataset):,}")
    print(f"  Test  samples: {len(loaders['test'].dataset):,}")

    print("\n建立模型並載入預訓練權重...")
    model = DCRNN(
        num_nodes=N, input_dim=F, output_dim=1,
        hid_dim=hid_dim, num_layers=num_layers,
        max_diffusion_step=max_diffusion_step,
        pred_len=pred_len, adj_mx=adj_mx, filter_type=filter_type,
    ).to(device)

    missing, unexpected = load_pretrained_dcrnn(model, ckpt["model"])
    print(f"權重載入完成。因站點數變更而重建的 buffer({len(missing)} 個,預期中):")
    for k in missing:
        print(f"    {k}")

    if args.freeze_encoder:
        print("\n[info] --freeze_encoder 已啟用:凍結 DCGRU 層,只微調 output_layer")
        for name, param in model.named_parameters():
            if "output_layer" not in name:
                param.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"    可訓練參數: {n_trainable:,} / {n_total:,}")
    else:
        trainable = model.parameters()

    exp_name = args.exp_name or f"finetune_{time.strftime('%Y%m%d-%H%M%S')}"
    log_path = os.path.join(args.log_dir, exp_name)
    writer = SummaryWriter(log_dir=log_path)
    print(f"\nTensorBoard logdir: {log_path}  (執行: tensorboard --logdir {args.log_dir})")

    optimizer = Adam(trainable, lr=args.lr, eps=1e-3, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"dcrnn_finetuned_{exp_name}_best.pt")

    # ── Fine-tune 前:用「還沒被動過」的預訓練模型,在 train/val/test 三個切分上
    #    都先測一次,存成 baseline_metrics.json。尤其 test 這份現在測,是因為它跟
    #    最後 fine-tune 完的模型會測的是「同一份」test set——這樣才是公平比較,
    #    不是拿 fine-tune 前的 val 表現去對比 fine-tune 後的 test 表現。
    print("\nFine-tune 前(預訓練模型,尚未微調)的表現:")
    baseline = {}
    for split_name in ["train", "val", "test"]:
        m = evaluate(model, loaders[split_name], scalers, device)
        baseline[split_name] = m  # train/val 還是照算,val 是訓練迴圈內部要用的,train 存進報告備查
        if split_name == "test" or args.show_train_val:
            print(f"  {split_name:5s}: MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  MAPE={m['MAPE']:.2f}%")
        writer.add_scalar(f"baseline/{split_name}_MAE", m["MAE"], 0)

    baseline_path = os.path.join(args.save_dir, f"baseline_metrics_{exp_name}.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump({
            "pretrained_ckpt": args.pretrained_ckpt,
            "data": args.data,
            "metrics": baseline,
        }, f, indent=2, ensure_ascii=False)
    print(f"  已存至 {baseline_path}(不管後面 fine-tune 過程如何,這份「動手前」的紀錄都保留著)")

    best_val_mae = baseline["val"]["MAE"]
    best_state_dict = None  # 追蹤目前最好的候選權重(記憶體裡),之後給內插步驟用,不用回頭讀檔
    patience_cnt = 0
    global_step = 0

    swa_model = AveragedModel(model)
    swa_start_epoch = max(1, int(args.epochs * args.swa_start_frac)) if args.swa_start_frac <= 1 else None
    swa_updated = False
    if swa_start_epoch is not None:
        print(f"SWA: 從第 {swa_start_epoch} 個 epoch 起累積平均權重")

    print("\nEpoch   Train-MAE    Val-MAE   Val-RMSE  Val-MAPE      Time")
    print("─" * 62)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, global_step, args.cl_decay_steps
        )
        val_metrics = evaluate(model, loaders["val"], scalers, device)
        elapsed = time.time() - t0
        is_best = val_metrics["MAE"] < best_val_mae
        marker = " ★" if is_best else ""

        print(f"{epoch:>5d}   {train_loss:>10.4f}   {val_metrics['MAE']:>7.3f}   "
              f"{val_metrics['RMSE']:>8.3f}  {val_metrics['MAPE']:>7.2f}%  {elapsed:>5.1f}s{marker}")

        writer.add_scalar("epoch/train_MAE", train_loss, epoch)
        writer.add_scalar("epoch/val_MAE", val_metrics["MAE"], epoch)
        writer.add_scalar("epoch/val_RMSE", val_metrics["RMSE"], epoch)
        writer.add_scalar("epoch/val_MAPE", val_metrics["MAPE"], epoch)

        if swa_start_epoch is not None and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_updated = True

        if is_best:
            best_val_mae = val_metrics["MAE"]
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "val_metrics": val_metrics,
                "args": {**orig_args, "hist_len": hist_len, "pred_len": pred_len,
                        "finetuned_from": args.pretrained_ckpt, "finetune_lr": args.lr,
                        "freeze_encoder": args.freeze_encoder, "swa": False},
            }, save_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── SWA 候選:把後半段每個 epoch 的權重平均起來,跟目前最好的單一 epoch 比一比。
    #    這個候選甚至可能救回「單一 epoch 從沒贏過 baseline」的情況——平均掉的雜訊
    #    有時候比任何單一 epoch 的快照更穩定,尤其資料量小、val 本身估計雜訊大的時候。
    # if swa_updated:
    #     swa_val_metrics = evaluate(swa_model.module, loaders["val"], scalers, device)
    #     print(f"\nSWA(平均後半段 epoch 的權重)在 val 上的表現: "
    #           f"MAE={swa_val_metrics['MAE']:.3f}  RMSE={swa_val_metrics['RMSE']:.3f}")
    #     if swa_val_metrics["MAE"] < best_val_mae:
    #         print(f"  → 比目前最佳單一 epoch({best_val_mae:.3f})更好,改用 SWA 平均權重作為最終模型")
    #         best_val_mae = swa_val_metrics["MAE"]
    #         best_state_dict = {k: v.clone() for k, v in swa_model.module.state_dict().items()}
    #         torch.save({
    #             "epoch": "swa", "model": swa_model.module.state_dict(),
    #             "val_metrics": swa_val_metrics,
    #             "args": {**orig_args, "hist_len": hist_len, "pred_len": pred_len,
    #                     "finetuned_from": args.pretrained_ckpt, "finetune_lr": args.lr,
    #                     "freeze_encoder": args.freeze_encoder, "swa": True,
    #                     "swa_start_epoch": swa_start_epoch},
    #         }, save_path)
    #     else:
    #         print(f"  → 沒有贏過目前最佳單一 epoch({best_val_mae:.3f}),維持原本選擇")

    # ── 內插候選:拿目前最好的候選(單一 epoch 或 SWA)跟「完全沒動過」的原始權重
    #    做一次 alpha 內插掃描(跟 weight_interpolation.py 同一套邏輯)。
    #    這通常是最後、也最可靠的一步——已經實測過好幾次,單純 fine-tune 贏不了 val/test 的
    #    情況,內插常常能找到中間點同時贏過兩端。supports_stack(圖)不參與內插。
    if best_state_dict is not None and not args.no_interpolate:
        # print("\n內插候選:拿目前最好的權重跟原始預訓練權重做 alpha 掃描 ...")
        theta0 = ckpt["model"]
        theta1 = best_state_dict
        best_interp_alpha = None
        best_interp_state = None
        best_interp_val = best_val_mae
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            interp = {}
            for k in theta1:
                if k.endswith("supports_stack"):
                    continue
                interp[k] = (1 - alpha) * theta0[k] + alpha * theta1[k]
            probe = DCRNN(
                num_nodes=N, input_dim=F, output_dim=1,
                hid_dim=hid_dim, num_layers=num_layers,
                max_diffusion_step=max_diffusion_step,
                pred_len=pred_len, adj_mx=adj_mx, filter_type=filter_type,
            ).to(device)
            probe.load_state_dict(interp, strict=False)
            m = evaluate(probe, loaders["val"], scalers, device)
            if m["MAE"] < best_interp_val:
                best_interp_val = m["MAE"]
                best_interp_alpha = alpha
                best_interp_state = {k: v.clone() for k, v in probe.state_dict().items()}

        if best_interp_alpha is not None:
            # print(f"  → alpha={best_interp_alpha:.1f} 比目前最好的候選更好"
            #       f"(val MAE {best_val_mae:.3f} -> {best_interp_val:.3f}),改用內插後的權重")
            best_val_mae = best_interp_val
            torch.save({
                "epoch": "interpolated", "model": best_interp_state,
                "val_metrics": {"MAE": best_interp_val},
                "args": {**orig_args, "hist_len": hist_len, "pred_len": pred_len,
                        "finetuned_from": args.pretrained_ckpt, "finetune_lr": args.lr,
                        "freeze_encoder": args.freeze_encoder, "interpolated": True,
                        "interp_alpha": best_interp_alpha},
            }, save_path)
        else:
            print(f"  → 內插沒有找到比目前最好候選({best_val_mae:.3f})更好的比例,維持原本選擇")

    # print(f"\nBest val MAE: {best_val_mae:.3f} µg/m³")
    if not os.path.exists(save_path):
        if args.save_last:
            # --save_last:不管 val 有沒有贏過 baseline,強制存最後一個 epoch 的權重,
            # 讓後面的 train/test 前後對照照常產出(展示/錄影用途)。
            print(f"\n[info] --save_last 已啟用:雖然全程沒有任何一個 epoch 贏過 val baseline,"
                  f"仍強制存下最後一個 epoch(第 {epoch} 個)的權重繼續產出對照。")
            final_val_metrics = evaluate(model, loaders["val"], scalers, device)
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "val_metrics": final_val_metrics,
                "args": {**orig_args, "hist_len": hist_len, "pred_len": pred_len,
                        "finetuned_from": args.pretrained_ckpt, "finetune_lr": args.lr,
                        "freeze_encoder": args.freeze_encoder, "forced_save_last": True},
            }, save_path)
        else:
            # 全程沒有任何一個 epoch 贏過 fine-tune 前的 baseline,根本沒存過 checkpoint,
            # 不該硬去讀一個不存在的檔案(這是先前版本會在這裡崩潰的原因)。
            print(f"\n⚠️  Fine-tune 跑了 {epoch} 個 epoch,"
                  f"沒有任何一次贏過 fine-tune 前的 baseline val MAE({baseline['val']['MAE']:.3f}),"
                  f"沒有產生任何 checkpoint。")
            print("這代表在目前的設定下(lr / freeze_encoder / 資料量),fine-tune 沒有帶來任何幫助,"
                  "直接沿用原本的預訓練模型即可,不需要也不能再跟 test 比較(沒有 fine-tune 後的模型可比)。"
                  "如果只是想看/錄前後對照數字、不在乎有沒有真的變好,加 --save_last 重跑。")

            report_path = os.path.join(args.save_dir, f"finetune_report_{exp_name}.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump({
                    "pretrained_ckpt": args.pretrained_ckpt,
                    "finetuned_ckpt": None,
                    "data": args.data,
                    "before_finetune": baseline,
                    "result": "fine-tune 全程未曾贏過 baseline,未產生任何 checkpoint",
                }, f, indent=2, ensure_ascii=False)
            print(f"\n已將此結果存至 {report_path}")
            writer.close()
            return

    if best_val_mae >= baseline["val"]["MAE"]:
        print(f"⚠️  Fine-tune 後的最佳 val MAE 沒有贏過 fine-tune 前的 baseline({baseline['val']['MAE']:.3f})"
              f",可能是新資料量太少、lr 不合適,或站點/圖差異影響較大,建議檢查 TensorBoard 訓練曲線。")

    ckpt_best = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt_best["model"])
    finetuned_test = evaluate(model, loaders["test"], scalers, device)
    finetuned_train = evaluate(model, loaders["train"], scalers, device)
    train_mae_improve = (baseline["train"]["MAE"] - finetuned_train["MAE"]) / baseline["train"]["MAE"] * 100

    if args.show_train_val:
        print("\n" + "═" * 55)
        print("  Fine-tune 前 vs 後,train set 對照(展示用)")
        print("═" * 55)
        bt = baseline["train"]
        print(f"{'':10s}{'MAE':>10s}{'RMSE':>10s}{'MAPE':>10s}")
        print(f"{'Fine-tune前':10s}{bt['MAE']:>10.3f}{bt['RMSE']:>10.3f}{bt['MAPE']:>9.2f}%")
        print(f"{'Fine-tune後':10s}{finetuned_train['MAE']:>10.3f}{finetuned_train['RMSE']:>10.3f}{finetuned_train['MAPE']:>9.2f}%")
        print(f"MAE {'改善' if train_mae_improve > 0 else '變差'} {abs(train_mae_improve):.1f}%")
        print("═" * 55)

    # ── 這才是真正該看的比較:同一份 test set,fine-tune 前(baseline)vs fine-tune 後 ──
    print("\n" + "═" * 55)
    print("  Fine-tune 前 vs 後,同一份 test set 對照")
    print("═" * 55)
    print(f"{'':10s}{'MAE':>10s}{'RMSE':>10s}{'MAPE':>10s}")
    b = baseline["test"]
    print(f"{'Fine-tune前':10s}{b['MAE']:>10.3f}{b['RMSE']:>10.3f}{b['MAPE']:>9.2f}%")
    print(f"{'Fine-tune後':10s}{finetuned_test['MAE']:>10.3f}{finetuned_test['RMSE']:>10.3f}{finetuned_test['MAPE']:>9.2f}%")
    mae_improve = (b["MAE"] - finetuned_test["MAE"]) / b["MAE"] * 100
    rmse_improve = (b["RMSE"] - finetuned_test["RMSE"]) / b["RMSE"] * 100
    print("─" * 55)
    print(f"MAE  {'改善' if mae_improve > 0 else '變差'} {abs(mae_improve):.1f}%"
          f"  |  RMSE {'改善' if rmse_improve > 0 else '變差'} {abs(rmse_improve):.1f}%")
    print("═" * 55)
    if mae_improve <= 0:
        print("⚠️  Fine-tune 後在 test set 上的 MAE 沒有比 fine-tune 前更好,"
              "這組結果不建議直接拿去用,建議檢查資料量、學習率,或考慮 --freeze_encoder 降低過擬合風險。")

    report_path = os.path.join(args.save_dir, f"finetune_report_{exp_name}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "pretrained_ckpt": args.pretrained_ckpt,
            "finetuned_ckpt": save_path,
            "data": args.data,
            "before_finetune": baseline,
            "after_finetune_test": finetuned_test,
            "after_finetune_train": finetuned_train,
            "test_mae_improvement_pct": mae_improve,
            "test_rmse_improvement_pct": rmse_improve,
            "train_mae_improvement_pct": train_mae_improve,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n完整前後對照報告已存至 {report_path}")
    writer.close()


if __name__ == "__main__":
    main()