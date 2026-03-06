"""
RobustLMP-GAN — End-to-end pipeline orchestrator.

Runs the full pipeline in order:
    1. Load and preprocess PJM LMP data
    2. Feature engineering (lags, spreads, weather, interchange, gas prices)
    3. Train WGAN-GP adversarial generator
    4. Train baseline and robust LSTM forecasters
    5. Evaluate under clean and PGD attack conditions
    6. Save results summary CSV

Usage::

    python -m robustlmp_gan.main
    python -m robustlmp_gan.main --config path/to/config.yaml
    python -m robustlmp_gan.main --stage data        # data prep only
    python -m robustlmp_gan.main --stage wgan        # WGAN training only
    python -m robustlmp_gan.main --stage lstm        # LSTM training only
    python -m robustlmp_gan.main --stage evaluate    # evaluation only
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from robustlmp_gan.config import get_cfg
from robustlmp_gan.data import (
    load_pjm_lmp,
    filter_top_congestion_nodes,
    load_eia_natural_gas,
    load_interchange_csv,
    process_weather_to_hourly,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    add_spread_features,
    add_fuel_price_features,
    add_interchange_features,
    add_weather_features,
    clip_lmp_outliers,
    drop_lag_nans,
    build_feature_columns,
)
from robustlmp_gan.data.dataset import LMPDataset
from robustlmp_gan.models import Generator, Discriminator, LSTMForecaster
from robustlmp_gan.training import train_wgan, train_forecaster
from robustlmp_gan.evaluation import evaluate_model
from robustlmp_gan.utils import set_seed, setup_logging, build_temporal_splits

logger = logging.getLogger(__name__)


def run_data_pipeline(cfg: dict) -> pd.DataFrame:
    """Execute data loading and feature engineering stages.

    Args:
        cfg: Loaded YAML configuration dictionary.

    Returns:
        Fully engineered feature DataFrame ready for model training.
    """
    dcfg = cfg["data"]
    fcfg = cfg["features"]

    logger.info("=== Stage 1: Loading PJM LMP data ===")
    df = load_pjm_lmp(
        raw_dir=dcfg["raw_dir"],
        pattern=dcfg["lmp_pattern"],
        date_start=dcfg["date_start"],
        date_end=dcfg["date_end"],
    )
    df = filter_top_congestion_nodes(
        df, node_types=dcfg["node_types"], top_n=dcfg["top_n_nodes"]
    )
    df.to_csv(dcfg["top50_csv"], index=False)
    logger.info("Saved: %s  shape=%s", dcfg["top50_csv"], df.shape)

    logger.info("=== Stage 2: Feature engineering ===")
    df = df.sort_values(["pnode_name", "datetime_beginning_ept"]).reset_index(drop=True)
    df = add_calendar_features(df)
    df = add_lag_features(
        df,
        lag_hours=fcfg["lag_hours"],
        seasonal_lags=fcfg["seasonal_lags"],
    )
    df = add_rolling_features(df, windows=fcfg["rolling_windows"])
    df = add_spread_features(df)

    # Fuel prices
    df_ng = load_eia_natural_gas(
        path=dcfg["eia_gas_path"],
        sheet_name=dcfg["eia_gas_sheet"],
        skiprows=dcfg["eia_gas_skiprows"],
        date_start=dcfg["date_start"],
        date_end=dcfg["date_end"],
    )
    df = add_fuel_price_features(df, df_ng, heat_rate=dcfg["heat_rate"])

    # Interchange
    interchange_csv = os.path.join(
        dcfg["eia_interchange_dir"], "pjm_interchange_2019_2023.csv"
    )
    if os.path.exists(interchange_csv):
        df_ic = load_interchange_csv(interchange_csv)
        df = add_interchange_features(df, df_ic)
    else:
        logger.warning(
            "Interchange CSV not found: %s — skipping. "
            "Run scripts/download_interchange.py first.",
            interchange_csv,
        )

    df = drop_lag_nans(df)
    df.to_csv(dcfg["features_csv"], index=False)
    logger.info("Saved: %s  shape=%s", dcfg["features_csv"], df.shape)

    # Weather
    weather_hourly_csv = os.path.join(
        dcfg["noaa_weather_dir"], "pjm_weather_hourly_2019_2023.csv"
    )
    if os.path.exists(weather_hourly_csv):
        df_w = pd.read_csv(weather_hourly_csv)
        df_w["datetime_beginning_ept"] = pd.to_datetime(df_w["datetime_beginning_ept"])
        df = add_weather_features(df, df_w)
    else:
        logger.warning(
            "Weather hourly CSV not found: %s — skipping. "
            "Run scripts/download_weather.py first.",
            weather_hourly_csv,
        )

    df.to_csv(dcfg["weather_features_csv"], index=False)
    logger.info("Final features saved: %s  shape=%s", dcfg["weather_features_csv"], df.shape)
    return df


def build_loaders(
    df: pd.DataFrame,
    cfg: dict,
    clip: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], MinMaxScaler, MinMaxScaler, MinMaxScaler]:
    """Build train/val/test DataLoaders and fit scalers.

    Args:
        df: Engineered feature DataFrame.
        cfg: Configuration dictionary.
        clip: If True, clip LMP outliers before scaling.

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader, feature_cols,
        scaler_X, scaler_rt, scaler_da)``.
    """
    if clip:
        df = clip_lmp_outliers(
            df,
            low_pct=cfg["data"]["lmp_clip_low"],
            high_pct=cfg["data"]["lmp_clip_high"],
        )

    feature_cols = build_feature_columns(df, exclude_cols=cfg["data"]["exclude_cols"])
    df_train, df_val, df_test = build_temporal_splits(
        df,
        train_end=cfg["split"]["train_end"],
        val_end=cfg["split"]["val_end"],
    )

    scaler_X = MinMaxScaler()
    scaler_rt = MinMaxScaler()
    scaler_da = MinMaxScaler()

    X_train = scaler_X.fit_transform(df_train[feature_cols].fillna(0))
    X_val = scaler_X.transform(df_val[feature_cols].fillna(0))
    X_test = scaler_X.transform(df_test[feature_cols].fillna(0))

    y_train_rt = scaler_rt.fit_transform(df_train[["total_lmp_rt"]])
    y_val_rt = scaler_rt.transform(df_val[["total_lmp_rt"]])
    y_test_rt = scaler_rt.transform(df_test[["total_lmp_rt"]])

    y_train_da = scaler_da.fit_transform(df_train[["total_lmp_da"]])
    y_val_da = scaler_da.transform(df_val[["total_lmp_da"]])
    y_test_da = scaler_da.transform(df_test[["total_lmp_da"]])

    seq_len = cfg["lstm"]["seq_len"]
    batch = cfg["lstm"]["batch_size"]

    train_ds = LMPDataset(X_train, y_train_rt, y_train_da, seq_len)
    val_ds = LMPDataset(X_val, y_val_rt, y_val_da, seq_len)
    test_ds = LMPDataset(X_test, y_test_rt, y_test_da, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch, shuffle=False)

    return train_loader, val_loader, test_loader, feature_cols, scaler_X, scaler_rt, scaler_da


def run_pipeline(cfg: dict, stage: str = "all") -> None:
    """Execute the full RobustLMP-GAN pipeline.

    Args:
        cfg: Loaded YAML configuration dictionary.
        stage: Which pipeline stage to run. One of ``"all"``,
            ``"data"``, ``"wgan"``, ``"lstm"``, ``"evaluate"``.
    """
    set_seed(cfg["project"]["seed"])
    output_dir = cfg["project"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ── Stage 1: Data ─────────────────────────────────────────────────
    weather_csv = cfg["data"]["weather_features_csv"]
    if stage in ("all", "data") or not os.path.exists(weather_csv):
        df = run_data_pipeline(cfg)
    else:
        logger.info("Loading cached features from %s", weather_csv)
        df = pd.read_csv(weather_csv)
        df["datetime_beginning_ept"] = pd.to_datetime(df["datetime_beginning_ept"])

    if stage == "data":
        logger.info("Data stage complete.")
        return

    # ── Build loaders ─────────────────────────────────────────────────
    (
        train_loader, val_loader, test_loader,
        feature_cols, scaler_X, scaler_rt, scaler_da,
    ) = build_loaders(df, cfg)
    n_features = len(feature_cols)
    logger.info("n_features: %d", n_features)

    wcfg = cfg["wgan"]
    lcfg = cfg["lstm"]
    pgd_eps = cfg["pgd"]["epsilons"]

    # ── Stage 2: WGAN-GP ──────────────────────────────────────────────
    G = Generator(wcfg["noise_dim"], n_features, lcfg["seq_len"], wcfg["gen_hidden"])
    D = Discriminator(n_features, lcfg["seq_len"], wcfg["disc_hidden"])

    if stage in ("all", "wgan"):
        logger.info("=== Stage 3: WGAN-GP Training ===")
        G, _, _ = train_wgan(
            G=G,
            D=D,
            train_loader=train_loader,
            noise_dim=wcfg["noise_dim"],
            pgd_eps=pgd_eps,
            n_critic=wcfg["n_critic"],
            gp_lambda=wcfg["gp_lambda"],
            gen_lr=wcfg["gen_lr"],
            disc_lr=wcfg["disc_lr"],
            epochs=wcfg["epochs"],
            checkpoint_every=wcfg["checkpoint_every"],
            output_dir=output_dir,
        )
    else:
        gen_path = os.path.join(output_dir, "generator.pt")
        if os.path.exists(gen_path):
            G.load_state_dict(torch.load(gen_path))
            logger.info("Loaded generator weights from %s", gen_path)

    if stage == "wgan":
        logger.info("WGAN stage complete.")
        return

    # ── Stage 3: LSTM Training ─────────────────────────────────────────
    if stage in ("all", "lstm"):
        logger.info("=== Stage 4a: Baseline LSTM ===")
        model_base, _, _ = train_forecaster(
            train_loader=train_loader,
            val_loader=val_loader,
            n_features=n_features,
            lstm_hidden=lcfg["hidden_size"],
            lstm_layers=lcfg["num_layers"],
            dropout=lcfg["dropout"],
            lr=lcfg["learning_rate"],
            epochs=lcfg["epochs"],
            augment=False,
            output_dir=output_dir,
            tag="baseline",
        )

        logger.info("=== Stage 4b: Robust LSTM (GAN-augmented) ===")
        model_robust, _, _ = train_forecaster(
            train_loader=train_loader,
            val_loader=val_loader,
            n_features=n_features,
            G=G,
            noise_dim=wcfg["noise_dim"],
            lstm_hidden=lcfg["hidden_size"],
            lstm_layers=lcfg["num_layers"],
            dropout=lcfg["dropout"],
            lr=lcfg["learning_rate"],
            epochs=lcfg["epochs"],
            augment=True,
            aug_eps_range=pgd_eps,
            aug_fraction=lcfg["aug_fraction"],
            output_dir=output_dir,
            tag="robust",
        )
    else:
        # Load saved weights
        model_base = LSTMForecaster(n_features, lcfg["hidden_size"], lcfg["num_layers"], lcfg["dropout"])
        model_robust = LSTMForecaster(n_features, lcfg["hidden_size"], lcfg["num_layers"], lcfg["dropout"])
        model_base.load_state_dict(torch.load(os.path.join(output_dir, "lstm_baseline_best.pt")))
        model_robust.load_state_dict(torch.load(os.path.join(output_dir, "lstm_robust_best.pt")))
        logger.info("Loaded saved LSTM weights")

    if stage == "lstm":
        logger.info("LSTM stage complete.")
        return

    # ── Stage 4: Evaluation ────────────────────────────────────────────
    logger.info("=== Stage 5: Evaluation ===")
    smcfg = cfg["smoothing"]
    pgd_cfg = cfg["pgd"]
    results: dict[str, dict] = {}

    eval_kwargs = dict(
        loader=test_loader,
        scaler_rt=scaler_rt,
        scaler_da=scaler_da,
        sigma=smcfg["sigma"],
        n_smooth=smcfg["n_samples"],
        pgd_alpha=pgd_cfg["alpha"],
        pgd_steps=pgd_cfg["steps"],
    )

    results["base_clean"] = evaluate_model(model_base, tag="Baseline LSTM — Clean", **eval_kwargs)

    for eps in pgd_eps:
        results[f"base_pgd_{eps}"] = evaluate_model(
            model_base, attack_eps=eps, tag=f"Baseline LSTM — PGD eps={eps}", **eval_kwargs
        )

    results["robust_clean"] = evaluate_model(model_robust, tag="RobustLMP-GAN — Clean", **eval_kwargs)

    for eps in pgd_eps:
        results[f"robust_pgd_{eps}"] = evaluate_model(
            model_robust, attack_eps=eps, tag=f"RobustLMP-GAN — PGD eps={eps}", **eval_kwargs
        )
        results[f"smooth_pgd_{eps}"] = evaluate_model(
            model_robust, use_smoothing=True, attack_eps=eps,
            tag=f"RobustLMP-GAN+Smooth — PGD eps={eps}", **eval_kwargs
        )

    # Print and save summary
    print("\n\nFINAL RESULTS SUMMARY")
    print("=" * 75)
    print(f"{'Model':<45} {'RT MAPE':>8} {'DA MAPE':>8} {'RT RMSE':>8} {'DA RMSE':>8}")
    print("-" * 75)
    for res in results.values():
        print(
            f"{res['tag']:<45} {res['rt_mape']:>7.2f}% "
            f"{res['da_mape']:>7.2f}% {res['rt_rmse']:>7.2f}  {res['da_rmse']:>7.2f}"
        )

    results_df = pd.DataFrame(
        [
            {
                "model": res["tag"],
                "rt_mape": res["rt_mape"],
                "da_mape": res["da_mape"],
                "rt_rmse": res["rt_rmse"],
                "da_rmse": res["da_rmse"],
            }
            for res in results.values()
        ]
    )
    out_csv = os.path.join(output_dir, "results_summary.csv")
    results_df.to_csv(out_csv, index=False)
    logger.info("Results saved: %s", out_csv)


def main() -> None:
    """CLI entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="RobustLMP-GAN Pipeline")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (default: bundled config.yaml)",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "data", "wgan", "lstm", "evaluate"],
        default="all",
        help="Which pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    args = parser.parse_args()

    setup_logging(level=getattr(logging, args.log_level))
    cfg = get_cfg(args.config)
    run_pipeline(cfg, stage=args.stage)


if __name__ == "__main__":
    main()
