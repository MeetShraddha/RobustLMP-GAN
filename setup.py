"""Package setup for RobustLMP-GAN."""

from setuptools import setup, find_packages

setup(
    name="robustlmp_gan",
    version="0.1.0",
    description="Certified defense-augmented GANs for LMP forecasting under data poisoning",
    author="RobustLMP-GAN Authors",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "robustlmp_gan": ["config/config.yaml"],
    },
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "openpyxl>=3.1.0",
        "xlrd>=2.0.1",
        "torch>=2.1.0",
        "scikit-learn>=1.3.0",
        "requests>=2.31.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "viz": ["matplotlib>=3.7.0", "seaborn>=0.12.0"],
        "tracking": ["mlflow>=2.8.0"],
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0"],
    },
    entry_points={
        "console_scripts": [
            "robustlmp-train=robustlmp_gan.main:main",
        ],
    },
)
