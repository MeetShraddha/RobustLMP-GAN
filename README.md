# RobustLMP-GAN

**Certified Defense-Augmented GANs for Day-Ahead LMP Forecasting Under Data Poisoning**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

RobustLMP-GAN is a production-grade forecasting framework that combines:

- **WGAN-GP** adversarial augmentation — a Wasserstein GAN with gradient penalty that generates realistic adversarial perturbations of energy market features, used to harden the forecasting model during training.
- **Dual-output LSTM** — forecasts both real-time (RT) and day-ahead (DA) Locational Marginal Prices (LMP) one hour ahead.
- **Randomized Smoothing** — certified L2 robustness at inference via Gaussian noise averaging (Cohen et al., ICML 2019), providing provable guarantees against bounded input perturbations.
- **Market Vulnerability Score (MVS)** — a novel metric translating forecast bias into expected daily dollar impact of adversarial manipulation.

The framework is trained on five years (2019–2023) of PJM hourly LMP data across 50 high-congestion pricing nodes, augmented with EIA natural gas prices, NOAA weather data, and EIA 930 net interchange.

### Key Results

| Model | RT MAPE (Clean) | RT MAPE (PGD ε=0.05) | Certified R | MVS Reduction |
|---|---|---|---|---|
| Baseline LSTM | 4.2% | 28.1% | — | 0% |
| Adversarial LSTM | 4.6% | 18.4% | — | 34% |
| **RobustLMP-GAN** | **4.5%** | **8.3%** | **0.047** | **67%** |

---

## Project Structure

```
robustlmp_gan/
│
├── README.md
├── requirements.txt
├── setup.py
├── Dockerfile
├── .gitignore
│
├── src/
│   └── robustlmp_gan/
│       ├── __init__.py
│       ├── main.py                  ← Pipeline orchestrator (entry point)
│       │
│       ├── config/
│       │   ├── config.yaml          ← All hyperparameters & paths
│       │   └── settings.py          ← Config loader
│       │
│       ├── data/
│       │   ├── loader.py            ← PJM, EIA, NOAA data loading
│       │   ├── features.py          ← Feature engineering
│       │   └── dataset.py           ← PyTorch Dataset
│       │
│       ├── models/
│       │   └── architectures.py     ← Generator, Discriminator, LSTMForecaster
│       │
│       ├── training/
│       │   ├── wgan_trainer.py      ← WGAN-GP training loop + checkpointing
│       │   └── lstm_trainer.py      ← LSTM training (baseline & robust)
│       │
│       ├── evaluation/
│       │   └── metrics.py           ← PGD attack, smoothed inference, MAPE/RMSE, MVS
│       │
│       └── utils/
│           └── helpers.py           ← Seeding, logging, device, temporal split
│
├── tests/
│   ├── test_features.py
│   ├── test_models.py
│   └── test_evaluation.py
│
├── scripts/
│   ├── train.py                     ← CLI training script
│   ├── evaluate.py                  ← CLI evaluation script
│   ├── download_interchange.py      ← EIA 930 API download
│   └── download_weather.py          ← NOAA CDO API download
│
└── notebooks/
    └── da.ipynb                     ← Original research notebook (reference)
```

---

## Installation

### Option 1 — pip (recommended)

```bash
git clone https://github.com/your-org/robustlmp-gan.git
cd robustlmp-gan

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -e ".[viz,tracking,dev]"
```

### Option 2 — Docker

```bash
docker build -t robustlmp-gan .

# Mount your data directory and run
docker run -v $(pwd)/data:/app/data robustlmp-gan
```

### Verify installation

```bash
python -c "import robustlmp_gan; print(robustlmp_gan.__version__)"
pytest tests/ -v
```

---

## Data Setup

You need three external data sources. Download them before running the pipeline.

### 1. PJM Hourly LMP Data (required)

Download from the [PJM DataMiner2 portal](https://dataminer2.pjm.com/feed/rt_da_monthly_lmps/definition):

- Navigate to **rt_da_monthly_lmps** feed
- Download all monthly CSV files for **2019–2023**
- Place them in:

```
data/PJM hourly LMP and load data 2019-2023/rt_da_monthly_lmps_20*.csv
```

### 2. EIA Natural Gas Prices (required)

Download from [EIA Natural Gas Prices](https://www.eia.gov/naturalgas/):

- Download the Henry Hub spot price Excel file
- Place at: `data/eia_natural_gas_prices.xls`

### 3. EIA 930 Interchange (optional but recommended)

Run the provided download script with your EIA API key (free at [eia.gov](https://www.eia.gov/opendata/)):

```bash
python scripts/download_interchange.py --api-key YOUR_EIA_KEY
```

### 4. NOAA Weather Data (optional but recommended)

Run the download script with your NOAA CDO API token (free at [ncdc.noaa.gov](https://www.ncdc.noaa.gov/cdo-web/token)):

```bash
python scripts/download_weather.py --token YOUR_NOAA_TOKEN
```

---

## Running the Pipeline

All pipeline stages are controlled by `--stage`. Stages can be run independently once upstream outputs exist.

### Full pipeline (data → WGAN → LSTM → evaluate)

```bash
python -m robustlmp_gan.main
# or
python scripts/train.py
```

### Individual stages

```bash
# Stage 1: Data ingestion and feature engineering only
python -m robustlmp_gan.main --stage data

# Stage 2: Train WGAN-GP adversarial generator
python -m robustlmp_gan.main --stage wgan

# Stage 3: Train LSTM forecasters (baseline + robust)
python -m robustlmp_gan.main --stage lstm

# Stage 4: Evaluate all models under clean and PGD attack
python -m robustlmp_gan.main --stage evaluate
# or
python scripts/evaluate.py
```

### Custom config

```bash
python -m robustlmp_gan.main --config path/to/my_config.yaml --stage all
```

---

## Configuration

All hyperparameters, paths, and API settings live in a single YAML file:

```
src/robustlmp_gan/config/config.yaml
```

Key sections:

```yaml
wgan:
  epochs: 30
  noise_dim: 32
  n_critic: 3           # discriminator updates per generator update
  gp_lambda: 10

lstm:
  epochs: 25
  hidden_size: 64
  seq_len: 24           # 24-hour look-back window
  aug_fraction: 0.3     # fraction of batch augmented with GAN samples

pgd:
  epsilons: [0.01, 0.05, 0.10]   # attack strengths for evaluation

smoothing:
  sigma: 0.05           # certified defence noise level
  n_samples: 64         # smoothing passes at inference
```

You can override any value by creating a copy of the YAML and passing `--config`.

---

## Outputs

After a full pipeline run, the following files are written to `robustlmp_outputs/`:

| File | Description |
|---|---|
| `generator.pt` | Trained WGAN-GP generator weights |
| `discriminator.pt` | Trained WGAN-GP discriminator weights |
| `lstm_baseline_best.pt` | Best baseline LSTM weights |
| `lstm_robust_best.pt` | Best GAN-augmented LSTM weights |
| `g_losses.npy` | Generator loss curve (numpy array) |
| `d_losses.npy` | Discriminator loss curve |
| `results_summary.csv` | MAPE/RMSE table for all models/conditions |
| `checkpoints/wgan_epoch_*.pt` | WGAN-GP checkpoints every 5 epochs |

Intermediate feature CSVs written to the working directory:

| File | Description |
|---|---|
| `pjm_lmp_top50_2019_2023.csv` | Filtered top-50 node LMP data |
| `pjm_features_engineered.csv` | Full feature set (pre-weather) |
| `pjm_features_engineered_weather.csv` | Final features including weather |

---

## Running Tests

```bash
pytest tests/ -v
# With coverage
pytest tests/ --cov=robustlmp_gan --cov-report=term-missing
```

---

## Architecture Notes

### WGAN-GP Generator
Maps 32-dimensional Gaussian noise → `(seq_len=24, n_features)` perturbation tensor via a 3-layer MLP with LeakyReLU activations and Tanh output. The output is scaled by epsilon and added to real sequences.

### LSTM Forecaster
A 2-layer LSTM with hidden size 64, dropout 0.2, and two linear heads predicting RT and DA LMP simultaneously. The Huber loss provides robustness to extreme price spikes.

### Adversarial Augmentation
During robust training, 30% of each batch is replaced with GAN-perturbed versions. The epsilon value is sampled uniformly from `[0.01, 0.05, 0.10]` each batch, ensuring the model is hardened against a range of attack strengths.

### Randomized Smoothing Certification
At inference, 64 copies of each input are perturbed with Gaussian noise (σ=0.05) and the model is run on each. The coordinate-wise median prediction is returned. This provides an L2 certified robustness radius of `R = σ · Φ⁻¹(p_A)`.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{robustlmpgan2026,
  title={Certified Defense-Augmented GANs for Day-Ahead LMP Forecasting Under Data Poisoning},
  year={2026}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
