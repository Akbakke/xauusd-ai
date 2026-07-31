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
    return {
        **model_native_aux_target_contract_metadata(),
        "incomplete_tail_rows_total": 96,
        "candidate_rows_before_completeness": 100,
        "incomplete_candidate_rows_excluded": 96,
        "complete_rows_emitted": 4,
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
    # The stub declares which per-bar contract it stands for, exactly as a real
    # cache does. Undeclared frames now fail closed, which is the point: the
    # Dataset reads the declaration instead of assuming V2.
    from gx1.features.htf_features import (
        HTF_V4_MATRIX_CONTRACT,
        MULTI_TF_PER_BAR_FEATURES_V4,
    )

    frames = {}
    for tf in ("M5", "M15", "H1", "H4", "D1"):
        frame = pd.DataFrame(
            np.zeros((1, len(MULTI_TF_PER_BAR_FEATURES_V4)), dtype=np.float32),
            index=index,
            columns=list(MULTI_TF_PER_BAR_FEATURES_V4),
        )
        frame.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
        frames[tf] = frame
    cache_key = trainer._multi_tf_cache_key(m5_path)
    monkeypatch.setitem(trainer._MULTI_TF_CACHE, cache_key, frames)
    monkeypatch.setattr(
        trainer.EntryV10CtxDataset,
        "_get_multi_tf_window",
        lambda self, target_ts: {
            f"seq_{tf}": np.zeros(
                (seq_len, len(MULTI_TF_PER_BAR_FEATURES_V4)),
                dtype=np.float32,
            )
            for tf in ("m5", "m15", "h1", "h4", "d1")
        },
    )
    return m5_path
