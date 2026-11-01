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
from collections import deque

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
            f"Unexpected key mismatch while loading weights — this is not just a "
            f"station-count difference, please check that the model architecture matches:\n"
            f"  missing (not supports_stack): {bad_missing}\n"
            f"  unexpected (not supports_stack): {bad_unexpected}"
        )
    return missing, unexpected


def compute_sampling_prob(global_step, cl_decay_steps):
    return cl_decay_steps / (cl_decay_steps + np.exp(global_step / cl_decay_steps))


def find_cl_decay_steps(total_steps, target_end_tf=0.2, lo=1.0, hi=1e7, iters=200):
    """求解讓 tf_ratio 在訓練跑完(total_steps)時降到 target_end_tf 附近的 cl_decay_steps。

    背景(重要,這是 fine-tune 一直沒變強的主因之一):
    cl_decay_steps 的預設值 2000,是照原始 KnowAir-V2(355 站、8 年資料)那種
    每個 epoch 好幾百到上千 step 的訓練規模校準的——在那種規模下,20~50 個 epoch
    就能累積 15,000+ step,teacher_forcing_ratio 才有辦法真的從接近 1.0 降到接近 0,
    讓模型在訓練時就practice「不看真實答案、自己預測下一步」這件事(跟 evaluate() 永遠
    用 0% teacher forcing 的測試情境一致)。

    但 fine-tune(或這次的短 epoch 重新訓練)用的是小很多的新資料集,每個 epoch
    可能只有幾十到一百多個 step,總 step 數遠遠不到 15,000。沿用 cl_decay_steps=2000
    的結果是:整個訓練過程 tf_ratio 幾乎全程停在 0.99+,模型從頭到尾都在「看著真實答案
    做一步預測」,從沒真正練習過 evaluate() 會用到的自迴歸 rollout。
    這代表梯度下降優化的其實是一個跟「test MAE 實際在測的東西」不一樣的任務,
    自然沒有理由讓 test MAE 變好,甚至可能學到只在 teacher-forced 情境下才管用的權重,
    讓需要自迴歸 rollout 的 test 表現變差。

    這裡把 cl_decay_steps 按「這次實際會跑多少 total_steps」重新校準,讓 tf_ratio
    在訓練跑完時降到 target_end_tf(預設 0.2)附近,曲線形狀盡量對齊原始訓練當初
    「後段大多是低 teacher forcing」的樣子。
    """
    for _ in range(iters):
        mid = (lo + hi) / 2
        p = compute_sampling_prob(total_steps, mid)
        if p > target_end_tf:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def print_tf_schedule_preview(total_steps, cl_decay_steps, label):
    pts = [0.0, 0.25, 0.5, 0.75, 1.0]
    curve = [compute_sampling_prob(int(total_steps * f), cl_decay_steps) for f in pts]
    curve_str = "  ".join(f"{int(f*100):>3d}%:{v:.3f}" for f, v in zip(pts, curve))
    print(f"  [{label}] cl_decay_steps={cl_decay_steps:.1f}, total_steps={total_steps} "
          f"-> teacher_forcing_ratio curve (training progress 0%->100%): {curve_str}")


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
    p.add_argument("--select_metric", choices=["MAE", "RMSE"], default="RMSE",
                   help="checkpoint 篩選標準:is_best / SWA / 內插 alpha 掃描全部依這個指標"
                        "在 val 上比較,預設 RMSE——對齊『驗收標準是 test-RMSE』,讓 val 端"
                        "挑模型的標準跟最終驗收的指標一致(用 MAE 挑、卻用 RMSE 驗收,"
                        "兩者對離群大誤差的懲罰權重不同,挑出來的『最佳』不一定是 RMSE 驗收下最好的)。")
    p.add_argument("--val_smooth_window", type=int, default=3,
                   help="用最近 N 個 epoch 的 val 指標『移動平均』來判斷 is_best / early stopping,"
                        "取代單一 epoch 的原始數值。實測過:322 筆的小 val set,單一 epoch 的估計"
                        "雜訊很大(同一次訓練裡連續 3 個 epoch 的 val RMSE 從 10.9 暴衝到 12.6 又跌回 11.1),"
                        "只挑單一 epoch 的最低點,很容易挑到雜訊低谷而不是真的比較好的權重——"
                        "這也是先前 val 贏、test 卻輸的主因之一。預設 3;設成 1 等於關閉平滑,"
                        "退回原本逐 epoch 各自比較。")
    p.add_argument("--cl_decay_steps", type=int, default=None,
                   help="scheduled sampling(teacher forcing)衰減速度。不指定的話(建議),"
                        "會依這次實際的 total_steps(= 每 epoch 的 batch 數 x epochs)自動校準,"
                        "讓 tf_ratio 在訓練跑完時降到 --cl_decay_target_end_tf 附近——"
                        "這樣才能真的練習到 evaluate() 用的 0% teacher forcing 情境。"
                        "沿用舊預設值 2000 在 fine-tune 這種小資料、少 epoch 的情境下幾乎不會衰減,"
                        "是先前每次 fine-tune 都沒變強的主要原因之一,不建議手動指定回 2000。")
    p.add_argument("--cl_decay_target_end_tf", type=float, default=0.2,
                   help="自動校準 cl_decay_steps 時,希望訓練跑完那一刻的 teacher_forcing_ratio "
                        "降到多低(預設 0.2)。只有在 --cl_decay_steps 沒指定時才有作用。")
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
    p.add_argument("--no_interpolate", action="store_true",
                   help="訓練跑完後,預設會自動拿最終候選權重跟原始預訓練權重做一次內插掃描"
                        "(跟 weight_interpolation.py 同一套邏輯),取 val 上最好的比例當最終模型——"
                        "這通常比單純 fine-tune 的結果更穩定、更容易在 test 上也贏。加這個關掉這一步。")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    print(f"Loading pretrained checkpoint: {args.pretrained_ckpt}")
    ckpt = torch.load(args.pretrained_ckpt, map_location=device, weights_only=True)
    orig_args = ckpt.get("args", {})
    if not orig_args:
        raise SystemExit("Checkpoint does not contain 'args' — cannot auto-restore the model "
                          "architecture. Please manually confirm hid_dim/num_layers/...")

    hist_len = args.hist_len or orig_args["hist_len"]
    pred_len = args.pred_len or orig_args["pred_len"]
    hid_dim = orig_args["hid_dim"]
    num_layers = orig_args["num_layers"]
    max_diffusion_step = orig_args["max_diffusion_step"]
    filter_type = orig_args["filter_type"]

    print(f"Restored architecture from checkpoint: hid_dim={hid_dim}, num_layers={num_layers}, "
          f"max_diffusion_step={max_diffusion_step}, filter_type={filter_type}")
    print(f"hist_len={hist_len}, pred_len={pred_len} "
          f"({'inherited from checkpoint' if args.hist_len is None else 'overridden by user'})")

    print(f"\nLoading fine-tune data: {args.data}")
    loaders, adj_mx, scalers, N, F = get_loaders(args.data, hist_len, pred_len, args.batch_size)
    print(f"  Num stations N={N}, num features F={F}")
    print(f"  Train samples: {len(loaders['train'].dataset):,}")
    print(f"  Val   samples: {len(loaders['val'].dataset):,}")
    print(f"  Test  samples: {len(loaders['test'].dataset):,}")

    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * args.epochs
    if args.cl_decay_steps is None:
        args.cl_decay_steps = find_cl_decay_steps(total_steps, args.cl_decay_target_end_tf)
    else:
        print(f"\n[info] Using manually specified --cl_decay_steps={args.cl_decay_steps}:")
        print_tf_schedule_preview(total_steps, args.cl_decay_steps, "manual")
        old_default_preview = compute_sampling_prob(total_steps, 2000)
        if old_default_preview > 0.9:
            print(f"  [warning] With this cl_decay_steps, teacher_forcing_ratio is still estimated "
                  f"at {old_default_preview:.3f} (close to 1.0) by the end of training — the model "
                  f"will barely practice the autoregressive regime that evaluate() uses, which may "
                  f"be why this fine-tune isn't getting stronger. Consider removing this flag to "
                  f"let it auto-calibrate, or set it manually smaller.")

    orig_N = ckpt["model"]["encoder.cells.0.supports_stack"].shape[-1]

    print("\nBuilding model and loading pretrained weights...")
    model = DCRNN(
        num_nodes=N, input_dim=F, output_dim=1,
        hid_dim=hid_dim, num_layers=num_layers,
        max_diffusion_step=max_diffusion_step,
        pred_len=pred_len, adj_mx=adj_mx, filter_type=filter_type,
    ).to(device)

    missing, unexpected = load_pretrained_dcrnn(model, ckpt["model"])
    print(f"Weights loaded. Buffers rebuilt due to the station-count change "
          f"({len(missing)}, expected):")
    for k in missing:
        print(f"    {k}")

    if args.freeze_encoder:
        print("\n[info] --freeze_encoder enabled: DCGRU layers frozen, only fine-tuning output_layer")
        for name, param in model.named_parameters():
            if "output_layer" not in name:
                param.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"    Trainable params: {n_trainable:,} / {n_total:,}")
    else:
        trainable = model.parameters()

    exp_name = args.exp_name or f"finetune_{time.strftime('%Y%m%d-%H%M%S')}"
    log_path = os.path.join(args.log_dir, exp_name)
    writer = SummaryWriter(log_dir=log_path)
    print(f"\nTensorBoard logdir: {log_path}  (run: tensorboard --logdir {args.log_dir})")

    optimizer = Adam(trainable, lr=args.lr, eps=1e-3, weight_decay=args.weight_decay)
    criterion = nn.L1Loss()

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f"dcrnn_finetuned_{exp_name}_best.pt")

    # ── Fine-tune 前:用「還沒被動過」的預訓練模型先測一次。
    #    只測 val(訓練迴圈內部挑 checkpoint 要用,不對外呈現)跟 test(唯一會寫進
    #    報告、對外呈現的「結果」)——不測、不存、不印 train set,因為 train 只是
    #    這次 fine-tune 自己訓練用的資料,不是獨立的驗證,混進報告容易讓人誤會成
    #    另一組「結果」。test 現在就測,是因為它跟最後 fine-tune 完的模型會測的是
    #    「同一份」test set——這樣才是公平比較,不是拿 fine-tune 前的 val 表現去對比
    #    fine-tune 後的 test 表現。
    print("\nPerformance BEFORE fine-tuning (pretrained model, untouched) on the test set:")
    baseline = {}
    for split_name in ["val", "test"]:
        m = evaluate(model, loaders[split_name], scalers, device)
        baseline[split_name] = m  # val 只留在記憶體裡給下面選 checkpoint 用,不寫進任何檔案
        if split_name == "test":
            print(f"  {split_name:5s}: MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  MAPE={m['MAPE']:.2f}%")
        writer.add_scalar(f"baseline/{split_name}_MAE", m["MAE"], 0)

    baseline_path = os.path.join(args.save_dir, f"baseline_metrics_{exp_name}.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump({
            "pretrained_ckpt": args.pretrained_ckpt,
            "data": args.data,
            "test_metrics": baseline["test"],
        }, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {baseline_path} (this pre-fine-tune test snapshot is kept no matter "
          f"what happens next)")

    metric_key = args.select_metric
    smooth_window = max(1, args.val_smooth_window)
    # best_val_metric 現在追蹤的是「平滑後」的值(最近 smooth_window 個 epoch 的移動平均),
    # baseline 本身只有一個點,視為 window=1 的平滑值,可以直接當初始比較基準。
    best_val_metric = baseline["val"][metric_key]
    best_val_metric_raw = best_val_metric  # 對應那個 epoch 的原始(未平滑)數值,純粹給人看
    val_metric_history = deque(maxlen=smooth_window)
    best_state_dict = None  # 追蹤目前最好的候選權重(記憶體裡),之後給內插步驟用,不用回頭讀檔
    patience_cnt = 0
    global_step = 0

    swa_model = AveragedModel(model)
    swa_start_epoch = max(1, int(args.epochs * args.swa_start_frac)) if args.swa_start_frac <= 1 else None
    swa_updated = False

    print("\nEpoch   Train-MAE    Val-MAE   Val-RMSE  Val-MAPE   Smoothed      Time")
    print("─" * 62)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, global_step = train_one_epoch(
            model, loaders["train"], optimizer, criterion, device, global_step, args.cl_decay_steps
        )
        val_metrics = evaluate(model, loaders["val"], scalers, device)
        elapsed = time.time() - t0

        val_metric_history.append(val_metrics[metric_key])
        smoothed_metric = sum(val_metric_history) / len(val_metric_history)
        is_best = smoothed_metric < best_val_metric
        marker = " ★" if is_best else ""

        print(f"{epoch:>5d}   {train_loss:>10.4f}   {val_metrics['MAE']:>7.3f}   "
              f"{val_metrics['RMSE']:>8.3f}  {val_metrics['MAPE']:>7.2f}%  {smoothed_metric:>8.3f}  "
              f"{elapsed:>5.1f}s{marker}")

        writer.add_scalar("epoch/train_MAE", train_loss, epoch)
        writer.add_scalar("epoch/val_MAE", val_metrics["MAE"], epoch)
        writer.add_scalar("epoch/val_RMSE", val_metrics["RMSE"], epoch)
        writer.add_scalar("epoch/val_MAPE", val_metrics["MAPE"], epoch)
        writer.add_scalar(f"epoch/val_{metric_key}_smoothed", smoothed_metric, epoch)

        if swa_start_epoch is not None and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_updated = True

        if is_best:
            best_val_metric = smoothed_metric
            best_val_metric_raw = val_metrics[metric_key]
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
    if swa_updated:
        swa_val_metrics = evaluate(swa_model.module, loaders["val"], scalers, device)
        if swa_val_metrics[metric_key] < best_val_metric:
            print(f"  -> Better than the current best single epoch "
                  f"({metric_key}={best_val_metric:.3f}); switching to SWA-averaged weights "
                  f"as the final model")
            best_val_metric = swa_val_metrics[metric_key]
            best_state_dict = {k: v.clone() for k, v in swa_model.module.state_dict().items()}
            torch.save({
                "epoch": "swa", "model": swa_model.module.state_dict(),
                "val_metrics": swa_val_metrics,
                "args": {**orig_args, "hist_len": hist_len, "pred_len": pred_len,
                        "finetuned_from": args.pretrained_ckpt, "finetune_lr": args.lr,
                        "freeze_encoder": args.freeze_encoder, "swa": True,
                        "swa_start_epoch": swa_start_epoch},
            }, save_path)

    # ── 內插候選:拿目前最好的候選(單一 epoch 或 SWA)跟「完全沒動過」的原始權重
    #    做一次 alpha 內插掃描(跟 weight_interpolation.py 同一套邏輯)。
    #    這通常是最後、也最可靠的一步——已經實測過好幾次,單純 fine-tune 贏不了 val/test 的
    #    情況,內插常常能找到中間點同時贏過兩端。supports_stack(圖)不參與內插。
    if best_state_dict is not None and not args.no_interpolate:
        theta0 = ckpt["model"]
        theta1 = best_state_dict
        best_interp_alpha = None
        best_interp_state = None
        best_interp_val = best_val_metric
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
            if m[metric_key] < best_interp_val:
                best_interp_val = m[metric_key]
                best_interp_alpha = alpha
                best_interp_state = {k: v.clone() for k, v in probe.state_dict().items()}

        if best_interp_alpha is not None:
            best_val_metric = best_interp_val
            torch.save({
                "epoch": "interpolated", "model": best_interp_state,
                "val_metrics": {metric_key: best_interp_val},
                "args": {**orig_args, "hist_len": hist_len, "pred_len": pred_len,
                        "finetuned_from": args.pretrained_ckpt, "finetune_lr": args.lr,
                        "freeze_encoder": args.freeze_encoder, "interpolated": True,
                        "interp_alpha": best_interp_alpha},
            }, save_path)

    if not os.path.exists(save_path):
        if args.save_last:
            # --save_last:不管 val 有沒有贏過 baseline,強制存最後一個 epoch 的權重,
            # 讓後面的 train/test 前後對照照常產出(展示/錄影用途)。
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

            report_path = os.path.join(args.save_dir, f"finetune_report_{exp_name}.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump({
                    "pretrained_ckpt": args.pretrained_ckpt,
                    "finetuned_ckpt": None,
                    "data": args.data,
                    "test_before_finetune": baseline["test"],
                    "select_metric": metric_key,
                    "result": "fine-tune never beat the baseline; no checkpoint was produced",
                }, f, indent=2, ensure_ascii=False)
            print(f"\nSaved this result to {report_path}")
            writer.close()
            return

    ckpt_best = torch.load(save_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt_best["model"])
    finetuned_test = evaluate(model, loaders["test"], scalers, device)
    # 不評估、不印、不存 train set 的前後對照——train 只是這次 fine-tune 自己訓練用的
    # 資料,拿它算一個「進步幅度」放進報告,很容易被誤認成另一組獨立結果。
    # 唯一對外呈現、寫進報告的比較只有下面這組:同一份 test set,fine-tune 前後。

    # ── 這才是真正該看的比較:同一份 test set,fine-tune 前(baseline)vs fine-tune 後 ──
    print("\n" + "═" * 55)
    print("  Fine-tune BEFORE vs AFTER, same test set")
    print("═" * 55)
    print(f"{'':10s}{'MAE':>10s}{'RMSE':>10s}{'MAPE':>10s}")
    b = baseline["test"]
    print(f"{'Before FT':10s}{b['MAE']:>10.3f}{b['RMSE']:>10.3f}{b['MAPE']:>9.2f}%")
    print(f"{'After FT':10s}{finetuned_test['MAE']:>10.3f}{finetuned_test['RMSE']:>10.3f}{finetuned_test['MAPE']:>9.2f}%")
    mae_improve = (b["MAE"] - finetuned_test["MAE"]) / b["MAE"] * 100
    rmse_improve = (b["RMSE"] - finetuned_test["RMSE"]) / b["RMSE"] * 100
    primary_improve = {"MAE": mae_improve, "RMSE": rmse_improve}[metric_key]
    print("─" * 55)
    print(f"MAE  {'improved' if mae_improve > 0 else 'worsened'} by {abs(mae_improve):.1f}%"
          f"  |  RMSE {'improved' if rmse_improve > 0 else 'worsened'} by {abs(rmse_improve):.1f}%")
    print("═" * 55)
    accepted = primary_improve > 0

    report_path = os.path.join(args.save_dir, f"finetune_report_{exp_name}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "pretrained_ckpt": args.pretrained_ckpt,
            "finetuned_ckpt": save_path,
            "data": args.data,
            "test_before_finetune": baseline["test"],
            "test_after_finetune": finetuned_test,
            "test_mae_improvement_pct": mae_improve,
            "test_rmse_improvement_pct": rmse_improve,
            "select_metric": metric_key,
            "val_smooth_window": smooth_window,
            "acceptance_metric": f"test-{metric_key}",
            "accepted": accepted,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nFull before/after comparison report saved to {report_path}")
    writer.close()


if __name__ == "__main__":
    main()