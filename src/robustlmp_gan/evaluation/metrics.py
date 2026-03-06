"""
Evaluation utilities for RobustLMP-GAN.

Provides:
    - PGD adversarial attack
    - Randomized smoothing inference
    - Model evaluation with metrics
    - Market Vulnerability Score computation
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from robustlmp_gan.models import LSTMForecaster

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# PGD Attack
# ─────────────────────────────────────────────────────────────────────────────

def pgd_attack(
    model: LSTMForecaster,
    x_seq: torch.Tensor,
    y_rt: torch.Tensor,
    y_da: torch.Tensor,
    epsilon: float,
    alpha: float = 0.005,
    steps: int = 10,
) -> torch.Tensor:
    """Projected Gradient Descent (PGD) adversarial attack.

    Iteratively perturbs ``x_seq`` to maximise forecasting loss within
    an L-inf ball of radius ``epsilon``.

    Args:
        model: LSTM forecaster being attacked.
        x_seq: Clean input sequences ``(batch, seq_len, n_features)``.
        y_rt: Real-time LMP targets ``(batch, 1)``.
        y_da: Day-ahead LMP targets ``(batch, 1)``.
        epsilon: L-inf perturbation budget (in normalised scale).
        alpha: Per-step attack step size.
        steps: Number of PGD iterations.

    Returns:
        Adversarially perturbed input tensor (detached).
    """
    loss_fn = nn.HuberLoss()
    x_adv = x_seq.clone().detach()
    x_adv = (x_adv + torch.empty_like(x_adv).uniform_(-epsilon, epsilon)).clamp(0, 1)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        pred_rt, pred_da = model(x_adv)
        loss = loss_fn(pred_rt, y_rt) + loss_fn(pred_da, y_da)
        loss.backward()

        with torch.no_grad():
            x_adv = x_adv + alpha * x_adv.grad.sign()
            delta = (x_adv - x_seq).clamp(-epsilon, epsilon)
            x_adv = (x_seq + delta).clamp(0, 1)

    return x_adv.detach()


# ─────────────────────────────────────────────────────────────────────────────
# Randomized Smoothing
# ─────────────────────────────────────────────────────────────────────────────

def smoothed_predict(
    model: LSTMForecaster,
    x_seq: torch.Tensor,
    sigma: float = 0.05,
    n_samples: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomized smoothing inference via median aggregation.

    Adds Gaussian noise to the input ``n_samples`` times and returns
    the coordinate-wise median prediction across all samples. This
    provides certified robustness at inference time.

    Args:
        model: LSTM forecaster.
        x_seq: Input sequences ``(batch, seq_len, n_features)``.
        sigma: Standard deviation of additive Gaussian noise.
        n_samples: Number of noisy forward passes to aggregate.

    Returns:
        Tuple of ``(pred_rt, pred_da)`` median tensors, each
        ``(batch, 1)``.
    """
    model.eval()
    preds_rt: list[torch.Tensor] = []
    preds_da: list[torch.Tensor] = []

    with torch.no_grad():
        for _ in range(n_samples):
            noise = torch.randn_like(x_seq) * sigma
            x_noisy = (x_seq + noise).clamp(0, 1)
            p_rt, p_da = model(x_noisy)
            preds_rt.append(p_rt)
            preds_da.append(p_da)

    pred_rt = torch.stack(preds_rt, dim=0).median(dim=0).values
    pred_da = torch.stack(preds_da, dim=0).median(dim=0).values
    return pred_rt, pred_da


# ─────────────────────────────────────────────────────────────────────────────
# Model Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model: LSTMForecaster,
    loader: DataLoader,
    scaler_rt: MinMaxScaler,
    scaler_da: MinMaxScaler,
    use_smoothing: bool = False,
    sigma: float = 0.05,
    n_smooth: int = 64,
    attack_eps: float | None = None,
    pgd_alpha: float = 0.005,
    pgd_steps: int = 10,
    tag: str = "",
) -> dict[str, Any]:
    """Evaluate an LSTMForecaster and return MAPE/RMSE metrics.

    Supports clean evaluation, PGD-attacked evaluation, and
    randomized-smoothed evaluation.

    Args:
        model: Trained LSTM forecaster.
        loader: Test DataLoader.
        scaler_rt: MinMaxScaler fitted on RT-LMP (used for inverse transform).
        scaler_da: MinMaxScaler fitted on DA-LMP (used for inverse transform).
        use_smoothing: If True, use :func:`smoothed_predict` instead of
            a single forward pass.
        sigma: Smoothing noise standard deviation.
        n_smooth: Number of smoothing samples.
        attack_eps: PGD attack epsilon. ``None`` means clean evaluation.
        pgd_alpha: PGD step size.
        pgd_steps: PGD iteration count.
        tag: Human-readable label for this evaluation run.

    Returns:
        Dictionary with keys: ``tag``, ``rt_mape``, ``da_mape``,
        ``rt_rmse``, ``da_rmse``, ``rt_true``, ``rt_pred``,
        ``da_true``, ``da_pred``.
    """
    model.eval()
    all_rt_true, all_rt_pred = [], []
    all_da_true, all_da_pred = [], []

    for x_seq, y_rt, y_da in tqdm(loader, desc=f"  Evaluating {tag}", leave=False):
        if attack_eps is not None:
            x_seq = pgd_attack(model, x_seq, y_rt, y_da, attack_eps, pgd_alpha, pgd_steps)

        if use_smoothing:
            pred_rt, pred_da = smoothed_predict(model, x_seq, sigma, n_smooth)
        else:
            with torch.no_grad():
                pred_rt, pred_da = model(x_seq)

        all_rt_true.append(y_rt.numpy())
        all_rt_pred.append(pred_rt.detach().numpy())
        all_da_true.append(y_da.numpy())
        all_da_pred.append(pred_da.detach().numpy())

    # Inverse-transform predictions to $/MWh
    rt_true = scaler_rt.inverse_transform(np.concatenate(all_rt_true))
    rt_pred = scaler_rt.inverse_transform(np.concatenate(all_rt_pred))
    da_true = scaler_da.inverse_transform(np.concatenate(all_da_true))
    da_pred = scaler_da.inverse_transform(np.concatenate(all_da_pred))

    rt_mape = float(mean_absolute_percentage_error(rt_true, rt_pred) * 100)
    da_mape = float(mean_absolute_percentage_error(da_true, da_pred) * 100)
    rt_rmse = float(np.sqrt(mean_squared_error(rt_true, rt_pred)))
    da_rmse = float(np.sqrt(mean_squared_error(da_true, da_pred)))

    logger.info(
        "\n%s\n  RT MAPE=%.2f%% RMSE=%.2f | DA MAPE=%.2f%% RMSE=%.2f",
        tag,
        rt_mape,
        rt_rmse,
        da_mape,
        da_rmse,
    )

    return {
        "tag": tag,
        "rt_mape": rt_mape,
        "da_mape": da_mape,
        "rt_rmse": rt_rmse,
        "da_rmse": da_rmse,
        "rt_true": rt_true,
        "rt_pred": rt_pred,
        "da_true": da_true,
        "da_pred": da_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Market Vulnerability Score
# ─────────────────────────────────────────────────────────────────────────────

def compute_mvs(
    lmp_true: np.ndarray,
    lmp_pred: np.ndarray,
    avg_daily_volume_mwh: float = 1000.0,
) -> float:
    """Compute the Market Vulnerability Score (MVS).

    MVS = E[|LMP_biased - LMP_true|] × V_day

    Quantifies the expected daily dollar value of forecast manipulation
    at a given node.

    Args:
        lmp_true: True LMP values ($/MWh).
        lmp_pred: Biased/predicted LMP values ($/MWh).
        avg_daily_volume_mwh: Average day-ahead market volume at the
            node (MWh). Defaults to 1000 MWh as a placeholder.

    Returns:
        MVS in dollars per day.
    """
    bias = np.abs(lmp_pred - lmp_true)
    return float(bias.mean()) * avg_daily_volume_mwh
