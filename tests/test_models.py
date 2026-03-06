"""Unit tests for model architectures."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from robustlmp_gan.models import Generator, Discriminator, LSTMForecaster


N_FEATURES = 20
SEQ_LEN = 24
NOISE_DIM = 16
BATCH = 8


def test_generator_output_shape():
    G = Generator(NOISE_DIM, N_FEATURES, SEQ_LEN, hidden_dim=32)
    z = torch.randn(BATCH, NOISE_DIM)
    out = G(z)
    assert out.shape == (BATCH, SEQ_LEN, N_FEATURES)


def test_generator_output_in_tanh_range():
    G = Generator(NOISE_DIM, N_FEATURES, SEQ_LEN, hidden_dim=32)
    z = torch.randn(BATCH, NOISE_DIM)
    out = G(z)
    assert out.min().item() >= -1.0 - 1e-5
    assert out.max().item() <= 1.0 + 1e-5


def test_discriminator_output_shape():
    D = Discriminator(N_FEATURES, SEQ_LEN, hidden_dim=32)
    x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
    out = D(x)
    assert out.shape == (BATCH, 1)


def test_lstm_forecaster_output_shapes():
    model = LSTMForecaster(N_FEATURES, hidden_size=32, num_layers=2, dropout=0.1)
    x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
    pred_rt, pred_da = model(x)
    assert pred_rt.shape == (BATCH, 1)
    assert pred_da.shape == (BATCH, 1)


def test_lstm_forecaster_gradient_flows():
    model = LSTMForecaster(N_FEATURES, hidden_size=32, num_layers=2, dropout=0.0)
    x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
    pred_rt, pred_da = model(x)
    loss = pred_rt.sum() + pred_da.sum()
    loss.backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"


def test_generator_no_nan():
    G = Generator(NOISE_DIM, N_FEATURES, SEQ_LEN, hidden_dim=32)
    z = torch.randn(BATCH, NOISE_DIM)
    out = G(z)
    assert not torch.isnan(out).any()


def test_discriminator_no_nan():
    D = Discriminator(N_FEATURES, SEQ_LEN, hidden_dim=32)
    x = torch.randn(BATCH, SEQ_LEN, N_FEATURES)
    out = D(x)
    assert not torch.isnan(out).any()
