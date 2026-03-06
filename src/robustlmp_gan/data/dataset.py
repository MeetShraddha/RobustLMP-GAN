"""
PyTorch Dataset for LMP time-series sequences.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class LMPDataset(Dataset):
    """Sliding-window dataset for LMP forecasting.

    Yields tuples of ``(x_seq, y_rt, y_da)`` where:
        - ``x_seq`` is a ``(seq_len, n_features)`` feature window.
        - ``y_rt`` is the real-time LMP target at ``t + seq_len``.
        - ``y_da`` is the day-ahead LMP target at ``t + seq_len``.

    Args:
        X: Feature array of shape ``(n_timesteps, n_features)``.
        y_rt: Real-time LMP array of shape ``(n_timesteps, 1)``.
        y_da: Day-ahead LMP array of shape ``(n_timesteps, 1)``.
        seq_len: Number of look-back timesteps per sample.
    """

    def __init__(
        self,
        X: "np.ndarray",
        y_rt: "np.ndarray",
        y_da: "np.ndarray",
        seq_len: int = 24,
    ) -> None:
        self.X = torch.FloatTensor(X)
        self.y_rt = torch.FloatTensor(y_rt)
        self.y_da = torch.FloatTensor(y_da)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int):
        x_seq = self.X[idx : idx + self.seq_len]          # (seq_len, n_features)
        y_rt = self.y_rt[idx + self.seq_len]               # (1,)
        y_da = self.y_da[idx + self.seq_len]               # (1,)
        return x_seq, y_rt, y_da
