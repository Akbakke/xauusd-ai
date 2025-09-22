"""CLI script for generating multi-timeframe feature datasets."""

import argparse
import yaml


def main():
    """Generate MTF feature dataset from raw OHLCV data.

    Process:
    1. Read raw CSV data
    2. Build MTF features (M1,M5,M15,H1,H4,D1)
    3. Align all to M1 base timeframe
    4. Apply shift(1) to prevent look-ahead
    5. Save to parquet format
    6. Validate no NaN/inf values
    7. Print dataset statistics
    """
    parser = argparse.ArgumentParser(description="Generate MTF feature dataset")
    parser.add_argument(
        "--config", default="configs/data.yaml", help="Config file path"
    )
    parser.add_argument("--input", help="Input CSV file path")
    parser.add_argument("--output", help="Output parquet file path")

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Processing {config['symbol']} data...")

    # TODO: Implement data processing pipeline
    # TODO: Read raw CSV with read_ohlcv_csv()
    # TODO: Generate MTF features for each timeframe
    # TODO: Align to M1 base with align_to_base()
    # TODO: Apply shift(1) to all features
    # TODO: Validate no NaN/inf values
    # TODO: Save to parquet format
    # TODO: Print column count and time range

    print("Dataset generation complete!")


if __name__ == "__main__":
    main()
