"""Shared multi-timeframe stub for model-native EntryV10CtxDataset tests.

One truth: every dataset test that needs a deterministic five-timeframe window
uses this helper instead of a private copy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_aux_targets_v3 import (
    model_native_aux_target_contract_metadata,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer


def aux_head_target_contract() -> dict:
    """Exact aux-head target contract block for dataset manifests."""
    return model_native_aux_target_contract_metadata()


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
