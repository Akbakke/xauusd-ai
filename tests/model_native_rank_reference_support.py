from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gx1.contracts.entry_model_native_state_v2 import (
    TrainRankReferenceV2,
    load_train_rank_reference_v2,
)
from gx1.scripts.materialize_model_native_train_rank_reference_v2 import run


def materialize_test_rank_reference(
    root: Path,
    *,
    run_id: str,
    history_start: str,
    fit_start: str,
    fit_end: str,
    source_path: Path | None = None,
) -> tuple[Path, TrainRankReferenceV2]:
    """Create a real, schema-valid rank reference for contract tests."""

    root.mkdir(parents=True, exist_ok=True)
    if source_path is None:
        source_path = root / "rank_source.parquet"
        times = pd.DatetimeIndex(
            pd.to_datetime([history_start, fit_start, fit_end], utc=True)
        ).drop_duplicates()
        close = pd.Series(range(len(times)), dtype="float64") + 1800.0
        frame = pd.DataFrame(
            {
                "time": times,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "bid_close": close - 0.05,
                "ask_close": close + 0.05,
            }
        )
        frame.to_parquet(source_path, index=False)
    out = root / "model_native_train_rank_reference_v4.npz"
    run(
        argparse.Namespace(
            source_parquet=source_path,
            out=out,
            history_start=history_start,
            fit_start=fit_start,
            fit_end=fit_end,
            min_rows=0,
            run_id=run_id,
        )
    )
    return source_path.resolve(), load_train_rank_reference_v2(out)
