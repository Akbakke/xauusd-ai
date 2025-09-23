#!/usr/bin/env python3
"""Feature generation CLI script.

This script reads OHLCV data and generates feature sets for trading signal prediction.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import read_ohlcv_csv
from features import build_all_features


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        logging.warning(f"Config file {config_path} not found, using defaults")
        return {}

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        logging.info(f"Loaded config from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        return {}


def validate_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean features DataFrame.

    Args:
        features_df: Raw features DataFrame

    Returns:
        Cleaned features DataFrame

    Raises:
        ValueError: If features contain inf values or result is empty
    """
    if features_df.empty:
        raise ValueError("Features DataFrame is empty")

    # Check for infinite values
    if np.isinf(features_df.select_dtypes(include=[np.number]).values).any():
        raise ValueError("Features contain infinite values")

    # Drop leading NaN rows (due to lagging/shifting)
    # Keep rows where at least one feature has a valid value
    features_df = features_df.dropna(how="all")

    if features_df.empty:
        raise ValueError("All features are NaN after dropping leading rows")

    logging.info(f"Dropped leading NaN rows, keeping {len(features_df)} rows")

    return features_df


def main():
    """Main entry point."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Generate trading features from OHLCV data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", default="configs/data.yaml", help="Path to configuration file"
    )
    parser.add_argument("--input", help="Input CSV file path (overrides config)")
    parser.add_argument(
        "--outdir", default="data/features", help="Output directory (overrides config)"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Determine input and output paths
        input_path = args.input or config.get("input_csv", "data/raw/ohlcv.csv")
        output_dir = args.outdir or config.get("output_dir", "data/features")

        logging.info(f"Input CSV: {input_path}")
        logging.info(f"Output directory: {output_dir}")

        # Validate input file exists
        if not os.path.exists(input_path):
            logging.error(f"Input file not found: {input_path}")
            sys.exit(1)

        # Load OHLCV data
        logging.info("Loading OHLCV data...")
        df_m1 = read_ohlcv_csv(input_path)
        logging.info(f"Loaded {len(df_m1)} rows of M1 data")
        logging.info(f"Date range: {df_m1.index[0]} to {df_m1.index[-1]}")

        # Build features
        logging.info("Building features...")
        features_df = build_all_features(df_m1)

        # Validate and clean features
        features_df = validate_features(features_df)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Write features to Parquet
        output_path = os.path.join(output_dir, "features.parquet")
        features_df.to_parquet(output_path)

        # Log summary
        logging.info("Feature generation completed successfully!")
        logging.info(f"Output file: {output_path}")
        logging.info(f"Rows: {len(features_df):,}")
        logging.info(f"Columns: {len(features_df.columns)}")
        logging.info(f"Date range: {features_df.index[0]} to {features_df.index[-1]}")

        # Show sample column names
        sample_cols = sorted(features_df.columns)[:5]
        logging.info(f"Sample columns: {sample_cols}")

        # Log some basic statistics
        logging.info(
            f"Memory usage: {features_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB"
        )

        sys.exit(0)

    except Exception as e:
        logging.error(f"Feature generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
