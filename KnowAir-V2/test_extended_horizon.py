"""
test_extended_horizon.py
=========================
Evaluate an already-trained DCRNN checkpoint (e.g. trained with
--hist_len 24 --pred_len 24) at a LONGER forecast horizon (e.g. 72h),
WITHOUT any retraining.

Why this is valid (see dcrnn_faithful.py's module docstring for detail):
  - `pred_len` only controls how many times DCRNNDecoder.forward() loops.
    No nn.Linear / nn.Parameter shape depends on it.
  - The decoder is weight-tied across every decode step, and its per-step
    input is only the PM2.5 channel (the model's own previous prediction),
    never the raw multi-feature exogenous input — so there's no missing
    "future weather" the model needs that it can't get from itself.
  - Therefore the 24h checkpoint's state_dict loads unchanged, and calling
    model(x, pred_len=72) simply asks the same trained weights to
    autoregress for more steps than they were trained on.

Honest caveat this script is DESIGNED to surface, not hide:
  - The model was never trained past hist_len+pred_len=24 steps of
    autoregression, so accuracy should be expected to degrade for the far
    end of the extended horizon (e.g. hours 49-72). This script reports
    metrics broken down into 24h blocks (h1-24 / h25-48 / h49-72 / ...)
    specifically so that degradation is visible and honestly reportable in
    your demo, rather than only reporting one blended overall number.

Usage:
    python test_extended_horizon.py \
        --checkpoint checkpoints/dcrnn_faithful_yrd_24h_best.pt \
        --data data/processed_southwest \
        --pred_len 72
"""

import argparse
import json
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dcrnn_faithful import DCRNN
from train_dcrnn import DCRNNDataset, compute_metrics, get_device


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Path to a .pt saved by train_dcrnn.py (contains 'model' and 'args')")
    p.add_argument("--data",       default="data/processed_southwest")
    p.add_argument("--pred_len",   type=int, default=72,
                   help="Horizon to evaluate at. Can be shorter, equal to, or LONGER than "
                        "the horizon the checkpoint was trained with.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--log_dir",    default="runs")
    p.add_argument("--exp_name",   default=None,
                   help="TensorBoard subfolder name. Default: eval72_<checkpoint filename>")
    p.add_argument("--out_json",   default=None,
                   help="Where to save a machine-readable results summary. "
                        "Default: alongside the checkpoint.")
    p.add_argument("--node_idx",   type=int, default=0,
                   help="Station index used for the sample prediction-vs-truth plot")
    return p.parse_args()


# ── evaluation with per-horizon-step breakdown ────────────────────────────────

@torch.no_grad()
def evaluate_extended(model, loader, scalers, device, pred_len):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        pred = model(x.to(device), y_true=None, teacher_forcing_ratio=0.0, pred_len=pred_len)
        preds.append(pred.cpu())
        targets.append(y)
    preds = torch.cat(preds)      # [B_total, pred_len, N, 1]
    targets = torch.cat(targets)

    overall = compute_metrics(preds, targets, scalers)

    pm25_mean = float(scalers["mean"][0])
    pm25_std  = float(scalers["std"][0])
    pred_denorm   = preds.squeeze(-1)   * pm25_std + pm25_mean   # [B_total, pred_len, N]
    target_denorm = targets.squeeze(-1) * pm25_std + pm25_mean

    per_step_mae  = (pred_denorm - target_denorm).abs().mean(dim=(0, 2))       # [pred_len]
    per_step_rmse = ((pred_denorm - target_denorm) ** 2).mean(dim=(0, 2)).sqrt()

    blocks = {}
    for b in range(0, pred_len, 24):
        end = min(b + 24, pred_len)
        blk_pred = pred_denorm[:, b:end]
        blk_tgt  = target_denorm[:, b:end]
        blocks[f"h{b + 1}-{end}"] = {
            "MAE":  (blk_pred - blk_tgt).abs().mean().item(),
            "RMSE": ((blk_pred - blk_tgt) ** 2).mean().sqrt().item(),
        }

    return overall, per_step_mae.numpy(), per_step_rmse.numpy(), blocks, pred_denorm, target_denorm


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device()

    print(f"Device    : {device}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    train_args = ckpt["args"]
    print(f"Checkpoint: {args.checkpoint}")
    print(f"  Trained with hist_len={train_args['hist_len']}, pred_len={train_args['pred_len']}")
    print(f"  Best epoch={ckpt['epoch']}, val MAE at save time={ckpt['val_metrics']['MAE']:.3f}")
    print(f"  Evaluating now at pred_len={args.pred_len} (hist_len stays {train_args['hist_len']})\n")

    features = np.load(os.path.join(args.data, "features.npy"))
    adj_mx   = np.load(os.path.join(args.data, "adj_mx.npy"))
    with open(os.path.join(args.data, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)
    with open(os.path.join(args.data, "split.pkl"), "rb") as f:
        split = pickle.load(f)

    hist_len = train_args["hist_len"]
    test_s, test_e = split["test"]
    test_ds = DCRNNDataset(features, test_s, test_e, hist_len, args.pred_len)
    print(f"Test samples (hist={hist_len}, pred={args.pred_len}, window={hist_len + args.pred_len}h): "
          f"{len(test_ds):,}")
    if len(test_ds) == 0:
        raise RuntimeError(
            f"Test split has no windows of length hist_len+pred_len={hist_len + args.pred_len}h. "
            f"Your test split range is shorter than that — either shrink --pred_len or "
            f"extend the test date range in split.pkl."
        )
    # drop_last=False: at this longer window length, samples are scarcer than
    # at training time, so keep every sample we can get for evaluation.
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, drop_last=False, pin_memory=True)

    N, F = features.shape[1], features.shape[2]
    model = DCRNN(
        num_nodes=N, input_dim=F, output_dim=1,
        hid_dim=train_args["hid_dim"], num_layers=train_args["num_layers"],
        max_diffusion_step=train_args["max_diffusion_step"],
        pred_len=train_args["pred_len"],     # keep this equal to how it was TRAINED
        adj_mx=adj_mx, filter_type=train_args["filter_type"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print("Checkpoint weights loaded (architecture is pred_len-independent, so this always matches).\n")

    overall, per_step_mae, per_step_rmse, blocks, pred_denorm, target_denorm = evaluate_extended(
        model, test_loader, scalers, device, args.pred_len
    )

    print("── Extended-horizon Test Results ──────────────────")
    print(f"  Overall MAE  : {overall['MAE']:.3f} µg/m³")
    print(f"  Overall RMSE : {overall['RMSE']:.3f} µg/m³")
    print(f"  Overall MAPE : {overall['MAPE']:.2f}%\n")
    print(f"  {'Block':<10s}  {'MAE':>8s}  {'RMSE':>8s}")
    for name, m in blocks.items():
        print(f"  {name:<10s}  {m['MAE']:>8.3f}  {m['RMSE']:>8.3f}")
    print("─────────────────────────────────────────────────")

    # ── TensorBoard: per-step degradation curve + one sample prediction plot ──
    if args.exp_name is None:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        args.exp_name = f"eval{args.pred_len}h_{ckpt_name}"
    log_path = os.path.join(args.log_dir, args.exp_name)
    writer = SummaryWriter(log_dir=log_path)

    for t, (mae_t, rmse_t) in enumerate(zip(per_step_mae, per_step_rmse), start=1):
        writer.add_scalar("extended_eval/MAE_by_horizon_step",  float(mae_t),  t)
        writer.add_scalar("extended_eval/RMSE_by_horizon_step", float(rmse_t), t)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(target_denorm[0, :, args.node_idx].numpy(), label="Ground truth", marker="o", markersize=3)
        ax.plot(pred_denorm[0, :, args.node_idx].numpy(),   label="Prediction",   marker="x", markersize=3)
        ax.axvline(train_args["pred_len"] - 1, color="gray", linestyle="--",
                    label=f"trained horizon ends (h{train_args['pred_len']})")
        ax.set_xlabel("Horizon step (h)")
        ax.set_ylabel("PM2.5 (µg/m³)")
        ax.set_title(f"{args.pred_len}h extended forecast — station idx {args.node_idx}, test sample")
        ax.legend()
        fig.tight_layout()
        writer.add_figure("extended_eval/prediction_vs_truth", fig, global_step=0)
        plt.close(fig)
    except ImportError:
        pass

    writer.add_hparams(
        {"checkpoint": args.checkpoint, "eval_pred_len": args.pred_len, "hist_len": hist_len,
         "trained_pred_len": train_args["pred_len"]},
        {"hparam/overall_MAE": overall["MAE"], "hparam/overall_RMSE": overall["RMSE"],
         "hparam/overall_MAPE": overall["MAPE"]},
    )
    writer.close()
    print(f"\nTensorBoard log: {log_path}   (run: tensorboard --logdir {args.log_dir})")

    # ── machine-readable summary for README / report tables ──
    if args.out_json is None:
        args.out_json = os.path.join(os.path.dirname(args.checkpoint) or ".",
                                      f"{args.exp_name}_results.json")
    with open(args.out_json, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "hist_len": hist_len,
            "trained_pred_len": train_args["pred_len"],
            "eval_pred_len": args.pred_len,
            "overall": overall,
            "blocks": blocks,
            "per_step_mae": per_step_mae.tolist(),
            "per_step_rmse": per_step_rmse.tolist(),
        }, f, indent=2, ensure_ascii=False)
    print(f"Results JSON   : {args.out_json}")


if __name__ == "__main__":
    main()