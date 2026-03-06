# RobustLMP-GAN
RobustLMP: Certified Defense-Augmented GAN for LMP ForecastingRobustLMP is a production-grade framework designed to defend Locational Marginal Price (LMP) forecasting models against adversarial data poisoning. It combines a Wasserstein GAN (WGAN) for robust data augmentation with Randomized Smoothing to provide certifiable $L_2$ robustness guarantees.This project refactors a research-oriented Jupyter notebook into a modular, scalable, and maintainable Python package.🚀 FeaturesModular Pipeline: Clear separation between data engineering, model definition, training, and evaluation.WGAN-Augmented Training: Uses a GAN to generate synthetic "adversarial-like" edge cases to improve model generalization.Certified Robustness: Implements randomized smoothing to guarantee that small perturbations in input features (e.g., manipulated load data) won't drastically flip the price forecast.Configuration-Driven: All hyperparameters, file paths, and model settings are stored in configs/config.yaml.Production Ready: Includes logging, type hinting, unit tests, and Docker support.📂 Project StructurePlaintextrobust_lmp/
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
🛠 Installation1. Clone the RepositoryBashgit clone https://github.com/your-username/robust_lmp.git
cd robust_lmp
2. Set Up EnvironmentIt is recommended to use a virtual environment:Bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
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
