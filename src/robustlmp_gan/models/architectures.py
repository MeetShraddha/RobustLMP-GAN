"""
Neural network architectures for RobustLMP-GAN.

Contains:
    - :class:`Generator`  — WGAN-GP generator producing adversarial
      perturbation sequences.
    - :class:`Discriminator` — WGAN-GP discriminator scoring sequence
      realism.
    - :class:`LSTMForecaster` — Dual-output LSTM predicting RT and DA
      LMP one hour ahead.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Generator(nn.Module):
    """WGAN-GP Generator.

    Maps Gaussian noise to a (seq_len, n_features) perturbation
    delta that will be added to real input sequences.

    Args:
        noise_dim: Dimensionality of the input noise vector.
        n_features: Number of input features (output feature dim).
        seq_len: Sequence length of the generated output.
        hidden_dim: Hidden layer width.
    """

    def __init__(
        self,
        noise_dim: int,
        n_features: int,
        seq_len: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features

        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, seq_len * n_features),
            nn.Tanh(),  # output in [-1, 1]; scaled by epsilon outside
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            z: Noise tensor of shape ``(batch, noise_dim)``.

        Returns:
            Perturbation tensor of shape ``(batch, seq_len, n_features)``.
        """
        out = self.net(z)
        return out.view(-1, self.seq_len, self.n_features)


class Discriminator(nn.Module):
    """WGAN-GP Discriminator (Critic).

    Scores the realism of a ``(seq_len, n_features)`` sequence.
    No sigmoid — outputs raw Wasserstein scores.

    Args:
        n_features: Input feature dimensionality.
        seq_len: Sequence length.
        hidden_dim: Hidden layer width.
    """

    def __init__(
        self,
        n_features: int,
        seq_len: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(seq_len * n_features, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, n_features)``.

        Returns:
            Wasserstein score tensor of shape ``(batch, 1)``.
        """
        return self.net(x.reshape(x.size(0), -1))


class LSTMForecaster(nn.Module):
    """Bidirectional LSTM for joint RT and DA LMP forecasting.

    Predicts one-step-ahead real-time and day-ahead LMP values from
    a 24-hour feature window.

    Args:
        n_features: Number of input features.
        hidden_size: LSTM hidden state dimensionality.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability between LSTM layers.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head_rt = nn.Linear(hidden_size, 1)
        self.head_da = nn.Linear(hidden_size, 1)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(batch, seq_len, n_features)``.

        Returns:
            Tuple of ``(pred_rt, pred_da)``, each of shape
            ``(batch, 1)``.
        """
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # take the final time-step hidden state
        return self.head_rt(last), self.head_da(last)
