# RobustLMP-GAN

# 🛡️ RobustLMP-GAN
### Certified Defense-Augmented GANs for Day-Ahead LMP Forecasting

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

**RobustLMP-GAN** is a production-grade forecasting framework designed to protect Locational Marginal Price (LMP) models against *adversarial data poisoning*. By combining **Wasserstein GAN (WGAN)** data augmentation with **Randomized Smoothing**, it provides provable $L_2$ robustness guarantees for energy market price predictions.

---

## 📖 Project Overview

Electricity markets are increasingly vulnerable to "Data Integrity Attacks." A sophisticated actor can subtly corrupt input features (like load forecasts or fuel prices) to bias the market clearing price in their favor. 

**This project addresses this by:**
* **Augmenting** training data with a WGAN to generate "worst-case" distribution shifts.
* **Certifying** model outputs using Gaussian noise injection, ensuring the forecast stays within a tight bound even if inputs are perturbed.
* **Quantifying** risk via a custom **Market Vulnerability Score (MVS)**.



---

## 📂 Project Structure

```text
robust_lmp/
├── configs/
│   └── config.yaml           # Centralized experiment parameters
├── src/
│   └── robust_lmp/
│       ├── data/             # Data loading & feature engineering
│       ├── models/           # LSTM, Generator, and Discriminator architectures
│       ├── training/         # GAN and Forecaster training loops
│       ├── evaluation/       # Robustness metrics & MVS calculation
│       ├── utils/            # Logging, seeding, and path management
│       └── main.py           # Orchestration entry point
├── tests/                    # Pytest suite for core logic
├── scripts/                  # Shell scripts for automation
├── Dockerfile                # Containerization for reproducibility
└── requirements.txt          # Python dependencies

```

## 🛠️ Installation & Setup

### 1. Environment Preparation
Ensure you have **Python 3.9+** installed. It is highly recommended to use a virtual environment to manage dependencies.


# Clone the repository
```text
git clone [https://github.com/your-username/robust-lmp.git](https://github.com/your-username/robust-lmp.git)
cd robust-lmp
```

# Create and activate virtual environment
```text
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

# Install dependencies
```text
pip install --upgrade pip
pip install -r requirements.txt
```

3. Data SetupPlace your PJM/CAISO CSV files in the directory specified in configs/config.yaml (default: data/raw/).📈 UsageRun the Full PipelineThe main.py script orchestrates data processing, GAN training, and robust model training:Bashpython src/robust_lmp/main.py --config configs/config.yaml
Individual ComponentsYou can also run specific stages via the provided scripts:Train GAN: python scripts/train_gan.pyEvaluate Robustness: python scripts/evaluate.py --model_path models/robust_lstm.pt🐳 Docker SupportTo run the pipeline in a reproducible containerized environment:Bash# Build the image
docker build -t robust-lmp .

# Run the training pipeline
docker run -v $(pwd)/data:/app/data robust-lmp
🧪 TestingRun unit tests to ensure data transformations and model dimensions are correct:Bashpytest tests/
📝 Configuration (YAML)Modify configs/config.yaml to change behavior without touching code:YAMLmodel:
  lstm_hidden_size: 64
  batch_size: 32
  learning_rate: 0.001
  sigma: 0.1  # Smoothing parameter for certified defense
