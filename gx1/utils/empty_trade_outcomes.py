"""
TRUTH-safe empty trade_outcomes schema.

Contract: Deterministic empty parquet with canonical columns for 0-trades runs.
Used when n_trades==0 to satisfy SSoT (file must exist).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# TRUTH contract: required columns for trade_outcomes (merge expects these)
TRADE_OUTCOMES_REQUIRED_COLUMNS = [
    "trade_uid",
    "trade_id",
    "entry_time",
    "exit_time",
    "pnl_bps",
    "mae_bps",
    "mfe_bps",
    "time_to_mae_bars",
    "time_to_mfe_bars",
    "close_to_mfe_bps",
    "exit_efficiency",
    "post_exit_mfe_12b_bps",
    "post_exit_mae_12b_bps",
    "duration_bars",
    "side",
    "session",
    "exit_reason",
    "entry_bid",
    "entry_ask",
    "exit_bid",
    "exit_ask",
    "entry_spread_bps",
    "exit_spread_bps",
    "entry_price_used",
    "exit_price_used",
]


def empty_trade_outcomes_dataframe() -> pd.DataFrame:
    """Return empty DataFrame with canonical trade_outcomes schema."""
    return pd.DataFrame({
        "trade_uid": pd.Series(dtype="string"),
        "trade_id": pd.Series(dtype="string"),
        "entry_time": pd.Series(dtype="datetime64[ns, UTC]"),
        "exit_time": pd.Series(dtype="datetime64[ns, UTC]"),
        "pnl_bps": pd.Series(dtype="float64"),
        "mae_bps": pd.Series(dtype="float64"),
        "mfe_bps": pd.Series(dtype="float64"),
        "time_to_mae_bars": pd.Series(dtype="Int64"),
        "time_to_mfe_bars": pd.Series(dtype="Int64"),
        "close_to_mfe_bps": pd.Series(dtype="float64"),
        "exit_efficiency": pd.Series(dtype="float64"),
        "post_exit_mfe_12b_bps": pd.Series(dtype="float64"),
        "post_exit_mae_12b_bps": pd.Series(dtype="float64"),
        "duration_bars": pd.Series(dtype="int64"),
        "side": pd.Series(dtype="string"),
        "session": pd.Series(dtype="string"),
        "exit_reason": pd.Series(dtype="string"),
        "entry_bid": pd.Series(dtype="float64"),
        "entry_ask": pd.Series(dtype="float64"),
        "exit_bid": pd.Series(dtype="float64"),
        "exit_ask": pd.Series(dtype="float64"),
        "entry_spread_bps": pd.Series(dtype="float64"),
        "exit_spread_bps": pd.Series(dtype="float64"),
        "entry_price_used": pd.Series(dtype="float64"),
        "exit_price_used": pd.Series(dtype="float64"),
    })


def write_empty_trade_outcomes_parquet(path: Path, run_id: Optional[str] = None) -> None:
    """
    Write deterministic empty trade_outcomes parquet with canonical schema.

    Deterministic: same schema, no timestamps in payload, stable sort.
    run_id: optional, for logging only.
    """
    df = empty_trade_outcomes_dataframe()
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        str(path),
        compression="snappy",
        use_dictionary=True,
        write_statistics=True,
        row_group_size=65536,
        data_page_size=1048576,
        use_byte_stream_split=False,
        compression_level=None,
        write_page_index=True,
        store_schema=True,
    )
