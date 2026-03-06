"""Unit tests for evaluation functions."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robustlmp_gan.models import LSTMForecaster
from robustlmp_gan.evaluation import pgd_attack, smoothed_predict, compute_mvs


N_FEATURES = 10
SEQ_LEN = 12
BATCH = 4


@pytest.fixture
def small_model():
    return LSTMForecaster(N_FEATURES, hidden_size=16, num_layers=1, dropout=0.0)


@pytest.fixture
def sample_batch():
    x = torch.rand(BATCH, SEQ_LEN, N_FEATURES)
    y_rt = torch.rand(BATCH, 1)
    y_da = torch.rand(BATCH, 1)
    return x, y_rt, y_da


def test_pgd_attack_bounded(small_model, sample_batch):
    x, y_rt, y_da = sample_batch
    epsilon = 0.05
    x_adv = pgd_attack(small_model, x, y_rt, y_da, epsilon)
    delta = (x_adv - x).abs()
    assert delta.max().item() <= epsilon + 1e-5
    assert x_adv.min().item() >= 0.0 - 1e-5
    assert x_adv.max().item() <= 1.0 + 1e-5


def test_pgd_attack_different_from_clean(small_model, sample_batch):
    x, y_rt, y_da = sample_batch
    x_adv = pgd_attack(small_model, x, y_rt, y_da, epsilon=0.1, steps=5)
    assert not torch.allclose(x, x_adv)


def test_pgd_attack_no_nan(small_model, sample_batch):
    x, y_rt, y_da = sample_batch
    x_adv = pgd_attack(small_model, x, y_rt, y_da, epsilon=0.05)
    assert not torch.isnan(x_adv).any()


def test_smoothed_predict_shape(small_model, sample_batch):
    x, _, _ = sample_batch
    pred_rt, pred_da = smoothed_predict(small_model, x, sigma=0.05, n_samples=8)
    assert pred_rt.shape == (BATCH, 1)
    assert pred_da.shape == (BATCH, 1)


def test_smoothed_predict_no_nan(small_model, sample_batch):
    x, _, _ = sample_batch
    pred_rt, pred_da = smoothed_predict(small_model, x, sigma=0.05, n_samples=8)
    assert not torch.isnan(pred_rt).any()
    assert not torch.isnan(pred_da).any()


def test_compute_mvs_positive():
    true = np.array([[30.0], [35.0], [40.0]])
    pred = np.array([[32.0], [33.0], [42.0]])
    mvs = compute_mvs(true, pred, avg_daily_volume_mwh=1000.0)
    assert mvs > 0
    assert isinstance(mvs, float)


def test_compute_mvs_zero_for_perfect():
    arr = np.array([[30.0], [35.0]])
    mvs = compute_mvs(arr, arr, avg_daily_volume_mwh=1000.0)
    assert mvs == pytest.approx(0.0)
