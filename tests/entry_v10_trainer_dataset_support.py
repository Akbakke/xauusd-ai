"""Shared multi-timeframe stub for model-native EntryV10CtxDataset tests.

One truth: every dataset test that needs a deterministic five-timeframe window
uses this helper instead of a private copy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_aux_targets_v3 import (
    model_native_aux_target_contract_metadata,
)
from gx1.contracts.entry_exit_production_architecture_v1 import (
    PRODUCTION_MTF_PER_TF_WINDOW_BARS,
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
    per_tf_seq_lens: Mapping[str, int] | None = None,
) -> Path:
    """Install a deterministic five-TF window stub and return the m5 path."""
    exact_seq_lens = dict(
        PRODUCTION_MTF_PER_TF_WINDOW_BARS
        if per_tf_seq_lens is None
        else per_tf_seq_lens.items()
    )
    m5_path = tmp_path / "xau_m5_prebuilt.parquet"
    m5_path.write_bytes(b"test-cache-binding")
    cache_dir = tmp_path / "v4-cache"
    cache_dir.mkdir()
    monkeypatch.setenv(trainer._TRAIN_MULTI_TF_CACHE_ENV, str(cache_dir))
    index = pd.DatetimeIndex([pd.Timestamp("2025-01-01", tz="UTC")])
    # The stub declares which per-bar contract it stands for, exactly as a real
    # cache does. Undeclared frames now fail closed, which is the point: the
    # Dataset reads the declaration instead of assuming V2.
    from gx1.features.htf_features import (
        HTF_V4_MATRIX_CONTRACT,
        MULTI_TF_PER_BAR_FEATURES_V4,
        MultiTFV4DiskCache,
    )
    from gx1.contracts.entry_exit_feature_base_v1 import (
        ENTRY_MTF_CONTEXT_TIMEFRAMES,
    )

    from tests.htf_v29_registry_test_support import (
        synthetic_v29_registry_constants,
    )

    frames = MultiTFV4DiskCache(
        cache_identity_sha256="0" * 64,
        manifest_sha256="1" * 64,
        m5_prebuilt_source=str(m5_path.resolve()),
        m5_prebuilt_source_sha256=trainer._sha256_file(m5_path),
        v29_registry_constants=synthetic_v29_registry_constants(),
    )
    for tf in ("M5", "M15", "H1", "H4", "D1"):
        frame = pd.DataFrame(
            np.zeros((1, len(MULTI_TF_PER_BAR_FEATURES_V4)), dtype=np.float32),
            index=index,
            columns=list(MULTI_TF_PER_BAR_FEATURES_V4),
        )
        frame.attrs["htf_feature_contract"] = HTF_V4_MATRIX_CONTRACT
        frames[tf] = frame
    def _load_stub(observed_path: Path) -> MultiTFV4DiskCache:
        if Path(observed_path).resolve() != m5_path.resolve():
            raise AssertionError(f"unexpected M5 test source: {observed_path}")
        return frames

    monkeypatch.setattr(
        trainer,
        "_prebuild_multi_tf_features_once",
        _load_stub,
    )
    monkeypatch.setattr(
        trainer.EntryV10CtxDataset,
        "_get_multi_tf_window",
        lambda self, target_ts, **_kwargs: {
            f"seq_{tf}": np.zeros(
                (exact_seq_lens[tf.upper()], len(MULTI_TF_PER_BAR_FEATURES_V4)),
                dtype=np.float32,
            )
            for tf in (
                timeframe.lower() for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES
            )
        },
    )
    return m5_path
