"""
WGAN-GP training loop for adversarial sample generation.

Trains the Generator and Discriminator using the Wasserstein distance
with gradient penalty (Gulrajani et al., 2017).
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from robustlmp_gan.models import Generator, Discriminator

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Gradient Penalty
# ─────────────────────────────────────────────────────────────────────────────

def gradient_penalty(
    disc: Discriminator,
    real: torch.Tensor,
    fake: torch.Tensor,
) -> torch.Tensor:
    """Compute the WGAN-GP gradient penalty.

    Interpolates between real and fake samples and penalises the
    discriminator for having gradient norms deviating from 1.

    Args:
        disc: Discriminator model.
        real: Real data tensor ``(batch, seq_len, n_features)``.
        fake: Generated data tensor ``(batch, seq_len, n_features)``.

    Returns:
        Scalar gradient penalty tensor.
    """
    batch = real.size(0)
    alpha = torch.rand(batch, 1, 1).expand_as(real)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)

    d_interp = disc(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.reshape(batch, -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(
    epoch: int,
    G: Generator,
    D: Discriminator,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
    g_losses: list[float],
    d_losses: list[float],
    path: str,
) -> None:
    """Save WGAN-GP training checkpoint.

    Args:
        epoch: Current epoch index (0-based).
        G: Generator model.
        D: Discriminator model.
        opt_G: Generator optimiser.
        opt_D: Discriminator optimiser.
        g_losses: Generator loss history.
        d_losses: Discriminator loss history.
        path: Output ``.pt`` file path.
    """
    torch.save(
        {
            "epoch": epoch,
            "G_state": G.state_dict(),
            "D_state": D.state_dict(),
            "optG": opt_G.state_dict(),
            "optD": opt_D.state_dict(),
            "g_losses": g_losses,
            "d_losses": d_losses,
        },
        path,
    )
    logger.info("Checkpoint saved: %s", path)


def load_checkpoint(
    path: str,
    G: Generator,
    D: Discriminator,
    opt_G: torch.optim.Optimizer,
    opt_D: torch.optim.Optimizer,
) -> tuple[int, list[float], list[float]]:
    """Load WGAN-GP training checkpoint.

    Args:
        path: Path to ``.pt`` checkpoint file.
        G: Generator model (mutated in-place).
        D: Discriminator model (mutated in-place).
        opt_G: Generator optimiser (mutated in-place).
        opt_D: Discriminator optimiser (mutated in-place).

    Returns:
        Tuple of ``(start_epoch, g_losses, d_losses)``.
    """
    ckpt = torch.load(path)
    G.load_state_dict(ckpt["G_state"])
    D.load_state_dict(ckpt["D_state"])
    opt_G.load_state_dict(ckpt["optG"])
    opt_D.load_state_dict(ckpt["optD"])
    logger.info("Resumed from epoch %d", ckpt["epoch"] + 1)
    return ckpt["epoch"] + 1, ckpt["g_losses"], ckpt["d_losses"]


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_wgan(
    G: Generator,
    D: Discriminator,
    train_loader: DataLoader,
    noise_dim: int,
    pgd_eps: list[float],
    n_critic: int = 3,
    gp_lambda: float = 10.0,
    gen_lr: float = 2e-4,
    disc_lr: float = 2e-4,
    epochs: int = 30,
    checkpoint_every: int = 5,
    output_dir: str = "robustlmp_outputs",
    resume: bool = True,
) -> tuple[Generator, list[float], list[float]]:
    """Train the WGAN-GP.

    Uses the middle epsilon value (index 1) from ``pgd_eps`` for
    perturbation scaling during training.

    Args:
        G: Generator model.
        D: Discriminator model.
        train_loader: Training DataLoader.
        noise_dim: Generator input noise dimensionality.
        pgd_eps: List of epsilon values; ``pgd_eps[1]`` is used for
            adversarial augmentation.
        n_critic: Discriminator updates per generator update.
        gp_lambda: Gradient penalty weight.
        gen_lr: Generator learning rate.
        disc_lr: Discriminator learning rate.
        epochs: Total training epochs.
        checkpoint_every: Save a checkpoint every N epochs.
        output_dir: Directory for saving outputs.
        resume: Whether to resume from the latest checkpoint.

    Returns:
        Tuple of ``(trained_Generator, g_losses, d_losses)``.
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    opt_G = torch.optim.Adam(G.parameters(), lr=gen_lr, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=disc_lr, betas=(0.0, 0.9))

    start_epoch = 0
    g_losses: list[float] = []
    d_losses: list[float] = []

    # Optionally resume from checkpoint
    if resume:
        ckpt_files = sorted(
            f for f in os.listdir(checkpoint_dir)
            if f.startswith("wgan_epoch_") and f.endswith(".pt")
        )
        if ckpt_files:
            latest = os.path.join(checkpoint_dir, ckpt_files[-1])
            start_epoch, g_losses, d_losses = load_checkpoint(
                latest, G, D, opt_G, opt_D
            )

    eps_train = pgd_eps[1]  # middle epsilon for training perturbations
    logger.info(
        "WGAN-GP training: epochs=%d, n_critic=%d, gp_lambda=%.1f, eps=%.3f",
        epochs,
        n_critic,
        gp_lambda,
        eps_train,
    )

    start_time = time.time()

    for epoch in tqdm(
        range(start_epoch, epochs),
        desc="WGAN-GP",
        unit="epoch",
        initial=start_epoch,
        total=epochs,
    ):
        G.train()
        D.train()
        epoch_d: list[float] = []
        epoch_g: list[float] = []

        for x_real, _, _ in train_loader:

            # ── Discriminator update ──────────────────────────────────
            for _ in range(n_critic):
                opt_D.zero_grad()
                z = torch.randn(x_real.size(0), noise_dim)
                delta = G(z)
                x_fake = (x_real + delta * eps_train).detach().clamp(0, 1)

                d_real = D(x_real)
                d_fake = D(x_fake)
                gp = gradient_penalty(D, x_real, x_fake)
                d_loss = d_fake.mean() - d_real.mean() + gp_lambda * gp
                d_loss.backward()
                opt_D.step()
                epoch_d.append(d_loss.item())

            # ── Generator update ──────────────────────────────────────
            opt_G.zero_grad()
            z = torch.randn(x_real.size(0), noise_dim)
            delta = G(z)
            x_fake = (x_real + delta * eps_train).clamp(0, 1)
            g_loss = -D(x_fake).mean()
            g_loss.backward()
            opt_G.step()
            epoch_g.append(g_loss.item())

        g_losses.append(float(np.mean(epoch_g)))
        d_losses.append(float(np.mean(epoch_d)))

        elapsed = time.time() - start_time
        per_epoch = elapsed / (epoch - start_epoch + 1)
        remaining = per_epoch * (epochs - epoch - 1)

        logger.info(
            "Epoch [%d/%d] D=%.4f G=%.4f elapsed=%.1fm remaining=%.1fm",
            epoch + 1,
            epochs,
            d_losses[-1],
            g_losses[-1],
            elapsed / 60,
            remaining / 60,
        )

        if (epoch + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(
                checkpoint_dir, f"wgan_epoch_{epoch + 1:03d}.pt"
            )
            save_checkpoint(epoch, G, D, opt_G, opt_D, g_losses, d_losses, ckpt_path)

    # Final save
    torch.save(G.state_dict(), os.path.join(output_dir, "generator.pt"))
    torch.save(D.state_dict(), os.path.join(output_dir, "discriminator.pt"))
    np.save(os.path.join(output_dir, "g_losses.npy"), np.array(g_losses))
    np.save(os.path.join(output_dir, "d_losses.npy"), np.array(d_losses))
    logger.info("WGAN-GP training complete")
    return G, g_losses, d_losses
