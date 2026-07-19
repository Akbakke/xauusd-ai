"""Shared multi-timeframe stub for model-native EntryV10CtxDataset tests.

One truth: every dataset test that needs a deterministic five-timeframe window
uses this helper instead of a private copy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


def aux_head_target_contract() -> dict:
    """Exact aux-head target contract block for dataset manifests."""
    return {
        "schema_version": trainer.MODEL_NATIVE_AUX_TARGET_SCHEMA_VERSION,
        "columns": list(trainer.MODEL_NATIVE_AUX_TARGET_COLUMNS),
        "future_horizon_bars_by_column": {
            name: int(horizon)
            for name, horizon in trainer.MODEL_NATIVE_AUX_TARGET_HORIZON_BY_COLUMN.items()
        },
        "max_future_horizon_bars": trainer.MODEL_NATIVE_AUX_MAX_FUTURE_HORIZON_BARS,
        "spread_aware_risk_magnitudes_required": True,
        "mid_price_timing_reference_only": True,
        "incomplete_value": "NaN_before_emission_only",
        "incomplete_rows_may_be_emitted": False,
    }


def install_multi_tf_stub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    seq_len: int = 2,
) -> Path:
    """Install a deterministic five-TF window stub and return the m5 path."""
    m5_path = tmp_path / "xau_m5_prebuilt.parquet"
    m5_path.write_bytes(b"test-cache-binding")
    index = pd.DatetimeIndex([pd.Timestamp("2025-01-01", tz="UTC")])
    frames = {
        tf: pd.DataFrame(np.zeros((1, 25), dtype=np.float32), index=index)
        for tf in ("M5", "M15", "H1", "H4", "D1")
    }
    cache_key = f"{m5_path.resolve()}|contract=V2_CAUSAL"
    monkeypatch.setitem(trainer._MULTI_TF_CACHE, cache_key, frames)
    monkeypatch.setattr(
        trainer.EntryV10CtxDataset,
        "_get_multi_tf_window",
        lambda self, target_ts: {
            f"seq_{tf}": np.zeros((seq_len, 25), dtype=np.float32)
            for tf in ("m5", "m15", "h1", "h4", "d1")
        },
    )
    return m5_path
