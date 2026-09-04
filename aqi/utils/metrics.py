"""
Evaluation metrics for PM2.5 forecasting.
All metrics operate on tensors in physical units (µg/m³).
"""

import torch
import numpy as np


def mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def rmse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2))


def mape(pred: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """MAPE with epsilon floor to avoid division by near-zero targets."""
    return torch.mean(torch.abs(pred - target) / (torch.abs(target) + eps)) * 100.0


def inverse_transform(
    y_norm: torch.Tensor,
    scalers: dict,
    feature_idx: int = 0,
) -> torch.Tensor:
    """Convert normalised PM2.5 back to µg/m³."""
    mean = float(scalers["mean"][feature_idx])
    std  = float(scalers["std"][feature_idx])
    return y_norm * std + mean


@torch.no_grad()
def compute_metrics(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    scalers: dict,
) -> dict[str, float]:
    """
    Compute MAE / RMSE / MAPE in physical units (µg/m³).

    Args:
        pred_norm   : [B, T_out, N, 1]  normalised predictions
        target_norm : [B, T_out, N, 1]  normalised ground truth
        scalers     : dict from scalers.pkl
    """
    pred   = inverse_transform(pred_norm.cpu().float(),   scalers)
    target = inverse_transform(target_norm.cpu().float(), scalers)
    return {
        "MAE":  mae(pred, target).item(),
        "RMSE": rmse(pred, target).item(),
        "MAPE": mape(pred, target).item(),
    }
