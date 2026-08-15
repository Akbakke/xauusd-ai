"""Genuine synthetic TRAIN-fit support for squeeze integration tests."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from gx1.features.volatility_squeeze_state_v1 import (
    VOLATILITY_SQUEEZE_CLOCKS,
    VOLATILITY_SQUEEZE_CLOCK_CONTRACT,
    fit_volatility_squeeze_artifact_manifest,
    load_volatility_squeeze_artifact_manifest,
    volatility_squeeze_bar_grid,
)


_FREQ = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1D",
}


def synthetic_closed_ohlcv(timeframe: str, rows: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(20260814 + VOLATILITY_SQUEEZE_CLOCKS.index(timeframe))
    block = (np.arange(rows) // 50) % 2
    sigma = np.where(block == 0, 0.00008, 0.0012)
    close = 2_000.0 * np.exp(np.cumsum(rng.normal(0.0, sigma)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    wick = np.maximum(np.abs(close - open_), close * sigma) + 0.01
    index = pd.date_range(
        "2022-01-02T22:00:00Z",
        periods=rows,
        freq=_FREQ[timeframe],
    )
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "volume": 100 + (np.arange(rows) % 37),
        },
        index=index,
    )


def make_volatility_squeeze_artifact_set(
    tmp_path: Path,
    *,
    frames_by_clock: Mapping[str, pd.DataFrame] | None = None,
    declared_train_window_start: pd.Timestamp | None = None,
    declared_train_window_end: pd.Timestamp | None = None,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bound = Path(__file__).resolve(strict=True)
    bound_sha = hashlib.sha256(bound.read_bytes()).hexdigest()
    frames = (
        {
            clock: synthetic_closed_ohlcv(clock)
            for clock in VOLATILITY_SQUEEZE_CLOCKS
        }
        if frames_by_clock is None
        else {
            clock: frames_by_clock[clock].copy()
            for clock in VOLATILITY_SQUEEZE_CLOCKS
        }
    )
    common = {
        "tape_manifest_artifact": str(bound),
        "tape_manifest_sha256": bound_sha,
        "split_manifest_artifact": str(bound),
        "split_manifest_sha256": bound_sha,
        "pair_manifest_artifact": str(bound),
        "pair_manifest_sha256": bound_sha,
        "pair_generation_id": "synthetic-test-pair-generation-v1",
        "pair_symbol": "XAUUSD",
        "train_split_id": "TRAIN",
        "clock_contract": VOLATILITY_SQUEEZE_CLOCK_CONTRACT,
    }
    provenance = {
        clock: {
            **common,
            "source_artifact": str(bound),
            "source_sha256": bound_sha,
            "source_schema_version": "synthetic_closed_ohlcv_v1",
            "source_lane": clock,
            "bar_grid": volatility_squeeze_bar_grid(clock),
        }
        for clock in VOLATILITY_SQUEEZE_CLOCKS
    }
    output = (tmp_path / "squeeze-artifacts").resolve()
    output.mkdir()
    manifest = fit_volatility_squeeze_artifact_manifest(
        frames,
        declared_train_window_start=(
            declared_train_window_start
            if declared_train_window_start is not None
            else min(frame.index[0] for frame in frames.values())
        ),
        declared_train_window_end=(
            declared_train_window_end
            if declared_train_window_end is not None
            else max(frame.index[-1] for frame in frames.values())
        ),
        source_provenance_by_clock=provenance,
        output_dir=output,
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return load_volatility_squeeze_artifact_manifest(
        manifest,
        expected_sha256=manifest_sha,
    )
