from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    model_native_signal_contract_metadata,
)
from gx1.contracts.entry_model_native_state_v2 import (
    MODEL_NATIVE_HISTORY_MODE,
    MODEL_NATIVE_RANK_TRANSFORM,
    MODEL_NATIVE_STATE_SCHEMA_VERSION,
    MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
    TrainRankReferenceV2,
)
from gx1.execution.v12_model_native_state_live import (
    ModelNativeStateBuilder,
    ModelNativeStateContract,
    compute_bucket_ctx_cat_full_frame,
    compute_htf_ctx_full_frame,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields
from gx1.scripts.materialize_model_native_train_rank_reference_v2 import run as materialize_rank


def test_model_native_full_frame_helpers_require_explicit_state_contract() -> None:
    frame = pd.DataFrame(index=pd.date_range("2026-07-08T18:00:00Z", periods=2, freq="5min"))

    with pytest.raises(RuntimeError, match="explicit model-native state contract required"):
        compute_bucket_ctx_cat_full_frame(frame)

    with pytest.raises(RuntimeError, match="explicit model-native state contract required"):
        compute_htf_ctx_full_frame(frame)


def test_model_native_state_builder_requires_explicit_contracts() -> None:
    with pytest.raises(TypeError, match="state_contract"):
        ModelNativeStateBuilder(ordered_signal_names=[])


def _early_validation_builder(tmp_path: Path) -> ModelNativeStateBuilder:
    signal_contract = model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.state_contract_fixture"
        )
    )
    fit_start = pd.Timestamp("2026-01-01T00:00:00Z")
    fit_end = pd.Timestamp("2026-01-02T00:00:00Z")
    reference = TrainRankReferenceV2(
        path=tmp_path / "unused_for_early_validation.npz",
        sha256="0" * 64,
        sidecar={},
        fit_start_utc=fit_start,
        fit_end_utc=fit_end,
        fit_row_count=1,
        atr_bps_sorted=np.asarray([10.0], dtype=np.float64),
        spread_bps_sorted=np.asarray([1.0], dtype=np.float64),
    )
    state_contract = ModelNativeStateContract(
        feature_history_start_utc=pd.Timestamp("2025-12-01T00:00:00Z"),
        rank_fit_start_utc=fit_start,
        rank_fit_end_utc=fit_end,
        rank_reference_npz=reference.path,
        rank_reference=reference,
        raw={},
    )
    return ModelNativeStateBuilder(
        ordered_signal_names=list(signal_contract["fields"]),
        state_contract=state_contract,
        signal_contract=signal_contract,
    )


@pytest.mark.parametrize(
    ("times", "message"),
    [
        (["2026-07-01T00:00:00Z", pd.NaT], "missing/invalid timestamps"),
        (
            ["2026-07-01T00:00:00Z", "2026-07-01T00:00:00Z"],
            "timestamps are not unique",
        ),
        (
            ["2026-07-01T00:05:00Z", "2026-07-01T00:00:00Z"],
            "not strictly chronological",
        ),
    ],
)
def test_model_native_state_rejects_invalid_time_before_feature_build(
    tmp_path: Path, times: list[object], message: str
) -> None:
    builder = _early_validation_builder(tmp_path)
    with pytest.raises(RuntimeError, match=message):
        builder.prepare_frame(pd.DataFrame({"time": times}))


def _rank_artifact(tmp_path: Path) -> tuple[Path, dict]:
    source = tmp_path / "fresh_xau_source.parquet"
    times = pd.date_range("2026-05-20T23:00:00Z", periods=40, freq="5min")
    close = 2300.0 + np.linspace(0.0, 7.8, len(times))
    pd.DataFrame(
        {
            "time": times,
            "high": close + 0.8,
            "low": close - 0.7,
            "close": close,
            "bid_close": close - 0.05,
            "ask_close": close + 0.05,
        }
    ).to_parquet(source, index=False)
    out = tmp_path / "model_native_train_rank_reference_v2.npz"
    report = materialize_rank(
        argparse.Namespace(
            source_parquet=source,
            out=out,
            history_start="2026-05-20T23:00:00Z",
            fit_start="2026-05-21T00:00:00Z",
            fit_end="2026-05-21T01:00:00Z",
            min_rows=1,
            vedtak="MODEL_NATIVE_STATE_CONTRACT_PYTEST",
        )
    )
    return out, report


def _state_metadata(rank_ref: Path, report: dict, *, sha256: str | None = None) -> dict:
    return {
        "schema_version": MODEL_NATIVE_STATE_SCHEMA_VERSION,
        "feature_history_start_utc": "2026-05-20T23:00:00Z",
        "rank_fit_start_utc": "2026-05-21T00:00:00Z",
        "rank_fit_end_utc": "2026-05-21T01:00:00Z",
        "rank_reference_npz": str(rank_ref),
        "rank_reference_npz_sha256": sha256 or report["out_npz_sha256"],
        "rank_reference_schema_version": MODEL_NATIVE_TRAIN_RANK_SCHEMA_VERSION,
        "normalization_fit_scope": "train_only",
        "rank_transform": MODEL_NATIVE_RANK_TRANSFORM,
        "feature_history_mode": MODEL_NATIVE_HISTORY_MODE,
        "split_reset_allowed": False,
        "post_fit_rows_in_rank_reference": False,
        "runtime_rule_free": True,
        "explicit_vedtak_id": "MODEL_NATIVE_STATE_CONTRACT_PYTEST",
    }


def test_model_native_state_contract_verifies_train_rank_reference_sha(tmp_path: Path) -> None:
    rank_ref, report = _rank_artifact(tmp_path)
    raw = _state_metadata(rank_ref, report, sha256="0" * 64)

    with pytest.raises(RuntimeError, match="TRAIN_RANK_REFERENCE_SHA_MISMATCH"):
        ModelNativeStateContract.from_metadata(raw)

    raw["rank_reference_npz_sha256"] = report["out_npz_sha256"]
    contract = ModelNativeStateContract.from_metadata(raw)
    assert contract.rank_reference_npz == rank_ref.resolve()
    assert contract.rank_reference.fit_row_count == report["fit_row_count"]


@pytest.mark.parametrize(
    "retired_schema",
    ["model_native_state_contract_v1", "model_native_state_contract_v2"],
)
def test_model_native_state_contract_rejects_retired_schema_and_stale_fields(
    tmp_path: Path,
    retired_schema: str,
) -> None:
    rank_ref, report = _rank_artifact(tmp_path)
    raw = _state_metadata(rank_ref, report)
    raw["schema_version"] = retired_schema
    raw["frame_anchor_utc"] = "2026-05-21T00:00:00Z"

    with pytest.raises(RuntimeError, match="STATE_SCHEMA_INVALID"):
        ModelNativeStateContract.from_metadata(raw)
