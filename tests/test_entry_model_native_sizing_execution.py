from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gx1.contracts.entry_model_native_sizing_execution_v1 import (
    ModelNativeSizingExecutionContractError,
    load_bound_joint_exit_sizing_proof,
    load_bound_runtime_sizing_parity,
    recompute_joint_exit_replay_coverage,
    recompute_runtime_sizing_parity_coverage,
)
from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    learned_sizing_authority_contract_metadata,
    prepare_model_native_sizing_authority,
)
from tests.model_native_sizing_support import (
    write_passing_runtime_sizing_parity,
)


def test_joint_active_exit_sizing_proof_is_row_recomputed_and_registry_bound(
    tmp_path: Path,
) -> None:
    evidence = write_passing_runtime_sizing_parity(tmp_path)
    proof, binding = load_bound_joint_exit_sizing_proof(
        evidence["joint_exit_proof_artifact"],
        context="UNIT_JOINT_EXIT_SIZING",
        verify_source_files=True,
    )

    assert binding == evidence["joint_exit_proof_artifact"]
    assert proof["decision"] == "PASS"
    assert proof["exit_replay_coverage"]["failed_rows"] == 0
    assert proof["exit_replay_coverage"]["trade_rows"] >= 128
    assert proof["paired_oos_utility"]["decision"] == "PASS"

    authority = learned_sizing_authority_contract_metadata(
        adoption_artifact=evidence["adoption_artifact"]
    )
    snapshot = prepare_model_native_sizing_authority(
        authority,
        context="UNIT_ADOPTED_JOINT_EXIT_SIZING",
    )
    assert evidence["adoption"]["joint_exit_sizing_proof_artifact"] == binding
    assert snapshot.joint_proof["exit_replay_coverage"]["failed_rows"] == 0

    runtime, runtime_binding = load_bound_runtime_sizing_parity(
        evidence["runtime_sizing_parity_artifact"],
        adoption=evidence["adoption"],
        calibration=evidence["calibration"],
        adoption_artifact=evidence["adoption_artifact"],
        context="UNIT_RUNTIME_SIZING_PARITY",
        verify_source_files=True,
        now_utc=evidence["runtime_sizing_parity"]["created_utc"],
    )
    assert runtime_binding == evidence["runtime_sizing_parity_artifact"]
    assert runtime["coverage"]["rows"] == 36
    assert runtime["coverage"]["direction_mismatch_count"] == 0
    assert runtime["coverage"]["order_submission_count"] == 0
    stale_now = pd.Timestamp(runtime["created_utc"]) + pd.Timedelta(days=2)
    with pytest.raises(ModelNativeSizingExecutionContractError, match="age_seconds"):
        load_bound_runtime_sizing_parity(
            evidence["runtime_sizing_parity_artifact"],
            adoption=evidence["adoption"],
            calibration=evidence["calibration"],
            adoption_artifact=evidence["adoption_artifact"],
            context="UNIT_STALE_RUNTIME_SIZING_PARITY",
            verify_source_files=True,
            now_utc=stale_now,
        )

    observations = pd.read_parquet(evidence["runtime_sizing_observations_path"])
    non_range_index = observations.copy()
    non_range_index.index = range(100, 100 + len(non_range_index))
    assert recompute_runtime_sizing_parity_coverage(
        non_range_index,
        calibration=evidence["calibration"],
        adoption=evidence["adoption"],
        adoption_sha256=evidence["adoption_artifact"]["sha256"],
        event_created_utc=runtime["created_utc"],
        context="UNIT_RUNTIME_NON_RANGE_INDEX",
    ) == runtime["coverage"]
    mutated_direction = observations.copy()
    mutated_direction.loc[0, "direction_after_sizing"] = 1
    with pytest.raises(ModelNativeSizingExecutionContractError, match="parity mismatch"):
        recompute_runtime_sizing_parity_coverage(
            mutated_direction,
            calibration=evidence["calibration"],
            adoption=evidence["adoption"],
            adoption_sha256=evidence["adoption_artifact"]["sha256"],
            event_created_utc=runtime["created_utc"],
            context="UNIT_RUNTIME_DIRECTION_MUTATION",
        )

    submitted_order = observations.copy()
    submitted_order.loc[0, "order_submitted"] = True
    with pytest.raises(ModelNativeSizingExecutionContractError, match="submit no order"):
        recompute_runtime_sizing_parity_coverage(
            submitted_order,
            calibration=evidence["calibration"],
            adoption=evidence["adoption"],
            adoption_sha256=evidence["adoption_artifact"]["sha256"],
            event_created_utc=runtime["created_utc"],
            context="UNIT_RUNTIME_ORDER_SUBMISSION",
        )

    rows = pd.read_parquet(evidence["joint_replay_rows_path"])
    exit_trace_rows = pd.read_parquet(evidence["joint_exit_trace_rows_path"])
    broken_status = rows.copy()
    broken_status.loc[0, "exit_replay_status"] = "HORIZON_CAP"
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="every non-FLAT row must reach active Exit EXIT_NOW",
    ):
        recompute_joint_exit_replay_coverage(
            broken_status,
            exit_trace_rows=exit_trace_rows,
            registry_sha256=proof["artifact_registry"]["sha256"],
            context="UNIT_HORIZON_CAP",
        )

    broken_trace = rows.copy()
    broken_trace.loc[0, "exit_trace_sha256"] = "not-a-hash"
    with pytest.raises(ModelNativeSizingExecutionContractError, match="SHA-256"):
        recompute_joint_exit_replay_coverage(
            broken_trace,
            exit_trace_rows=exit_trace_rows,
            registry_sha256=proof["artifact_registry"]["sha256"],
            context="UNIT_TRACE_HASH",
        )

    broken_trace_steps = exit_trace_rows.drop(exit_trace_rows.index[0]).copy()
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="steps are not contiguous",
    ):
        recompute_joint_exit_replay_coverage(
            rows,
            exit_trace_rows=broken_trace_steps,
            registry_sha256=proof["artifact_registry"]["sha256"],
            context="UNIT_TRACE_STEP_GAP",
        )

    evidence["artifact_registry_path"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(ModelNativeSizingExecutionContractError, match="hash mismatch"):
        load_bound_joint_exit_sizing_proof(
            evidence["joint_exit_proof_artifact"],
            context="UNIT_MUTATED_REGISTRY",
            verify_source_files=True,
        )
