from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.execution.v12_smart520_state_live import (
    Smart520StateContract,
    Smart520StateBuilder,
    compute_bucket_ctx_cat_full_frame,
    compute_htf_ctx_full_frame,
)


def test_smart520_full_frame_helpers_require_explicit_state_contract() -> None:
    frame = pd.DataFrame(index=pd.date_range("2026-07-08T18:00:00Z", periods=2, freq="5min"))

    with pytest.raises(RuntimeError, match="explicit smart520 state contract required"):
        compute_bucket_ctx_cat_full_frame(frame)

    with pytest.raises(RuntimeError, match="explicit smart520 state contract required"):
        compute_htf_ctx_full_frame(frame)


def test_smart520_state_builder_requires_explicit_state_contract() -> None:
    with pytest.raises(TypeError, match="state_contract"):
        Smart520StateBuilder(ordered_signal_names=[])


def test_smart520_state_contract_verifies_rank_reference_sha(tmp_path: Path) -> None:
    rank_ref = tmp_path / "smart520_rank_reference_xau_direction_repair.npz"
    np.savez_compressed(
        rank_ref,
        time_ns=np.asarray([pd.Timestamp("2026-05-21T00:00:00Z").value], dtype=np.int64),
        vol_regime_id=np.asarray([2], dtype=np.int64),
        spread_bucket=np.asarray([0], dtype=np.int64),
        atr_pinned=np.asarray([1.0], dtype=np.float64),
        atr_bps_sorted=np.asarray([10.0], dtype=np.float64),
        spread_bps_sorted=np.asarray([1.0], dtype=np.float64),
    )
    digest = hashlib.sha256(rank_ref.read_bytes()).hexdigest()
    rank_ref.with_suffix(rank_ref.suffix + ".json").write_text(
        json.dumps({"out_npz_sha256": digest}),
        encoding="utf-8",
    )
    raw = {
        "schema_version": "smart520_state_contract_v1",
        "frame_anchor_utc": "2026-05-21T00:00:00Z",
        "model_range_start_utc": "2020-11-09T00:00:00Z",
        "rank_reference_end_utc": "2026-05-22T00:00:00Z",
        "rank_reference_npz": str(rank_ref),
        "rank_reference_npz_sha256": "0" * 64,
    }

    with pytest.raises(RuntimeError, match="rank_reference_npz_sha256 mismatch"):
        Smart520StateContract.from_metadata(raw, require_xau_direction_repair=True)

    raw["rank_reference_npz_sha256"] = digest
    contract = Smart520StateContract.from_metadata(raw, require_xau_direction_repair=True)
    assert contract.rank_reference_npz == rank_ref.resolve()
