from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_sizing_execution_v1 import (
    ModelNativeSizingExecutionContractError,
    joint_exit_trace_sha256,
    load_bound_joint_exit_sizing_proof,
    load_bound_runtime_sizing_parity,
    read_bound_parquet_exact,
    recompute_joint_exit_replay_coverage,
    recompute_runtime_sizing_parity_coverage,
    require_joint_exit_portfolio_capacity,
    require_joint_replay_extends_canonical_oos_rows,
)
from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    ModelNativeSizingUnavailable,
    learned_sizing_authority_contract_metadata,
    prepare_model_native_sizing_authority,
)
from tests.model_native_sizing_support import (
    write_passing_runtime_sizing_parity,
)


def test_bound_parquet_rejects_same_hash_path_identity_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "bound.parquet"
    replacement = tmp_path / "replacement.parquet"
    pd.DataFrame({"value": [1, 2, 3]}).to_parquet(source, index=False)
    payload = source.read_bytes()
    replacement.write_bytes(payload)
    binding = {
        "path": str(source.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    real_read = os.read
    swapped = False

    def swap_after_first_read(fd: int, size: int) -> bytes:
        nonlocal swapped
        chunk = real_read(fd, size)
        if not swapped:
            swapped = True
            os.replace(replacement, source)
        return chunk

    monkeypatch.setattr(
        "gx1.contracts.entry_model_native_sizing_execution_v1.os.read",
        swap_after_first_read,
    )
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="changed while being read",
    ):
        read_bound_parquet_exact(binding, context="UNIT_SAME_HASH_PATH_SWAP")


def test_joint_active_exit_sizing_proof_is_row_recomputed_and_exit_projection_bound(
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
    assert set(proof["active_exit_artifact_manifests"]) == {
        "xgb",
        "v3_exit",
        "exit_iql",
    }

    authority = learned_sizing_authority_contract_metadata(
        adoption_artifact=evidence["adoption_artifact"]
    )
    snapshot = prepare_model_native_sizing_authority(
        authority,
        context="UNIT_ADOPTED_JOINT_EXIT_SIZING",
    )
    assert evidence["adoption"]["joint_exit_sizing_proof_artifact"] == binding
    assert snapshot.joint_proof["exit_replay_coverage"]["failed_rows"] == 0
    portfolio = require_joint_exit_portfolio_capacity(
        proof,
        max_trades=1,
        context="UNIT_JOINT_EXIT_PORTFOLIO_CAPACITY",
    )
    assert portfolio["max_trades"] == 1
    assert portfolio["admitted_trade_rows"] >= 128
    assert portfolio["mean_long_realized_pnl_bps"] > 0.0
    assert portfolio["mean_short_realized_pnl_bps"] > 0.0
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="outside the portfolio replay contract",
    ):
        require_joint_exit_portfolio_capacity(
            proof,
            max_trades=2,
            context="UNIT_UNPROVEN_PORTFOLIO_CAPACITY",
        )

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
    trade_rows = rows["model_direction_index"].isin([0, 1])
    assert not np.array_equal(
        rows.loc[trade_rows, "active_exit_fill_bid"].to_numpy(),
        rows.loc[trade_rows, "exit_bid"].to_numpy(),
    )
    final_trace = exit_trace_rows.sort_values("step").groupby(
        "reference_row_id",
        sort=False,
    ).tail(1)
    assert not np.array_equal(
        final_trace["state_bid"].to_numpy(),
        final_trace["fresh_quote_bid"].to_numpy(),
    )
    hard_stop_trace_rows = exit_trace_rows.copy()
    hard_stop_reference = str(hard_stop_trace_rows.iloc[0]["reference_row_id"])
    hard_stop_mask = (
        hard_stop_trace_rows["reference_row_id"].astype(str)
        == hard_stop_reference
    )
    hard_stop_index = hard_stop_trace_rows.index[hard_stop_mask][-1]
    hard_stop_trace_rows.loc[hard_stop_index, "bar_committed"] = False
    hard_stop_trace_rows.loc[hard_stop_index, "closed_bar_time"] = pd.NaT
    hard_stop_trace_rows.loc[hard_stop_index, "state_bid"] = (
        hard_stop_trace_rows.loc[hard_stop_index, "fresh_quote_bid"]
    )
    hard_stop_trace_rows.loc[hard_stop_index, "state_ask"] = (
        hard_stop_trace_rows.loc[hard_stop_index, "fresh_quote_ask"]
    )
    hard_stop_row_mask = (
        rows["reference_row_id"].astype(str) == hard_stop_reference
    )
    hard_stop_direction = int(
        rows.loc[hard_stop_row_mask, "model_direction_index"].iloc[0]
    )
    if hard_stop_direction == 0:
        hard_stop_pnl = (
            float(
                hard_stop_trace_rows.loc[
                    hard_stop_index,
                    "state_bid",
                ]
            )
            - float(rows.loc[hard_stop_row_mask, "entry_ask"].iloc[0])
        ) / float(
            rows.loc[hard_stop_row_mask, "entry_ask"].iloc[0]
        ) * 10_000.0
    else:
        hard_stop_pnl = (
            float(rows.loc[hard_stop_row_mask, "entry_bid"].iloc[0])
            - float(
                hard_stop_trace_rows.loc[
                    hard_stop_index,
                    "state_ask",
                ]
            )
        ) / float(
            rows.loc[hard_stop_row_mask, "entry_bid"].iloc[0]
        ) * 10_000.0
    hard_stop_trace_rows.loc[hard_stop_index, "state_pnl_bps"] = hard_stop_pnl
    hard_stop_rows = rows.copy()
    hard_stop_rows.loc[
        hard_stop_row_mask,
        "active_exit_decision_bar_time",
    ] = None
    hard_stop_trace = (
        hard_stop_trace_rows.loc[hard_stop_mask]
        .sort_values("step", kind="mergesort")
        .reset_index(drop=True)
    )
    hard_stop_rows.loc[
        hard_stop_row_mask,
        "exit_trace_sha256",
    ] = joint_exit_trace_sha256(
        hard_stop_trace,
        context="UNIT_PRECANONICAL_HARD_STOP_TRACE",
    )
    hard_stop_coverage = recompute_joint_exit_replay_coverage(
        hard_stop_rows,
        exit_trace_rows=hard_stop_trace_rows,
        exit_authority_sha256=proof[
            "active_exit_registry_projection"
        ]["projection_sha256"],
        context="UNIT_PRECANONICAL_HARD_STOP",
    )
    assert hard_stop_coverage["failed_rows"] == 0
    broken_status = rows.copy()
    broken_status.loc[0, "exit_replay_status"] = "HORIZON_CAP"
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="every non-FLAT row must reach active Exit EXIT_NOW",
    ):
        recompute_joint_exit_replay_coverage(
            broken_status,
            exit_trace_rows=exit_trace_rows,
            exit_authority_sha256=proof[
                "active_exit_registry_projection"
            ]["projection_sha256"],
            context="UNIT_HORIZON_CAP",
        )

    broken_trace = rows.copy()
    broken_trace.loc[0, "exit_trace_sha256"] = "not-a-hash"
    with pytest.raises(ModelNativeSizingExecutionContractError, match="SHA-256"):
        recompute_joint_exit_replay_coverage(
            broken_trace,
            exit_trace_rows=exit_trace_rows,
            exit_authority_sha256=proof[
                "active_exit_registry_projection"
            ]["projection_sha256"],
            context="UNIT_TRACE_HASH",
        )

    wrong_fill_time = rows.copy()
    wrong_fill_time.loc[0, "entry_fill_time"] = wrong_fill_time.loc[0, "time"]
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="entry_fill_time must be exactly decision time \\+ 5m",
    ):
        recompute_joint_exit_replay_coverage(
            wrong_fill_time,
            exit_trace_rows=exit_trace_rows,
            exit_authority_sha256=proof[
                "active_exit_registry_projection"
            ]["projection_sha256"],
            context="UNIT_PRE_FILL_EXIT_REJECTED",
        )

    forged_oos_identity = rows.copy()
    forged_oos_identity["session"] = (
        "FORGED_" + forged_oos_identity["session"].astype(str)
    )
    canonical_oos_rows = pd.read_parquet(
        evidence["oos_source"]["source_bindings"]["oos_rows"]["path"]
    )
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="differ from the exact canonical OOS TEST rows",
    ):
        require_joint_replay_extends_canonical_oos_rows(
            canonical_oos_rows=canonical_oos_rows,
            replay_rows=forged_oos_identity,
            context="UNIT_FORGED_CANONICAL_OOS_IDENTITY",
        )

    forged_trace_pnl = exit_trace_rows.copy()
    first_reference = str(forged_trace_pnl.iloc[0]["reference_row_id"])
    first_mask = forged_trace_pnl["reference_row_id"].astype(str) == first_reference
    first_index = forged_trace_pnl.index[first_mask][0]
    forged_trace_pnl.loc[first_index, "state_pnl_bps"] = 123456.0
    forged_trace = (
        forged_trace_pnl.loc[first_mask]
        .sort_values("step", kind="mergesort")
        .reset_index(drop=True)
    )
    forged_rows = rows.copy()
    forged_rows.loc[
        forged_rows["reference_row_id"].astype(str) == first_reference,
        "exit_trace_sha256",
    ] = joint_exit_trace_sha256(
        forged_trace,
        context="UNIT_FORGED_INTERMEDIATE_PNL_HASH",
    )
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="state PnL differs from closed state prices",
    ):
        recompute_joint_exit_replay_coverage(
            forged_rows,
            exit_trace_rows=forged_trace_pnl,
            exit_authority_sha256=proof[
                "active_exit_registry_projection"
            ]["projection_sha256"],
            context="UNIT_FORGED_INTERMEDIATE_PNL",
        )

    broken_trace_steps = exit_trace_rows.drop(exit_trace_rows.index[0]).copy()
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="steps are not contiguous",
    ):
        recompute_joint_exit_replay_coverage(
            rows,
            exit_trace_rows=broken_trace_steps,
            exit_authority_sha256=proof[
                "active_exit_registry_projection"
            ]["projection_sha256"],
            context="UNIT_TRACE_STEP_GAP",
        )

    gapped_trace_rows = exit_trace_rows.copy()
    first_reference = str(gapped_trace_rows.iloc[0]["reference_row_id"])
    first_mask = gapped_trace_rows["reference_row_id"].astype(str) == first_reference
    first_indices = gapped_trace_rows.index[first_mask]
    gapped_trace_rows.loc[first_indices[1], "fresh_quote_time"] = (
        pd.Timestamp(
            gapped_trace_rows.loc[first_indices[1], "fresh_quote_time"]
        )
        + pd.Timedelta(minutes=1)
    )
    gapped_rows = rows.copy()
    gapped_trace = (
        gapped_trace_rows.loc[first_mask]
        .sort_values("step", kind="mergesort")
        .reset_index(drop=True)
    )
    gapped_rows.loc[
        gapped_rows["reference_row_id"].astype(str) == first_reference,
        "exit_trace_sha256",
    ] = joint_exit_trace_sha256(
        gapped_trace,
        context="UNIT_GAPPED_M1_TRACE_HASH",
    )
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="Exit trace time binding is invalid",
    ):
        recompute_joint_exit_replay_coverage(
            gapped_rows,
            exit_trace_rows=gapped_trace_rows,
            exit_authority_sha256=proof[
                "active_exit_registry_projection"
            ]["projection_sha256"],
            context="UNIT_GAPPED_M1_TRACE",
        )

    exit_identity = (
        Path(proof["active_exit_artifact_manifests"]["v3_exit"]["root_path"])
        / "identity.bin"
    )
    original_identity = exit_identity.read_bytes()
    exit_identity.write_bytes(b"mutated")
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="artifact bytes differ",
    ):
        load_bound_joint_exit_sizing_proof(
            evidence["joint_exit_proof_artifact"],
            context="UNIT_MUTATED_ACTIVE_EXIT_BYTES",
            verify_source_files=True,
        )
    exit_identity.write_bytes(original_identity)

    registry_path = evidence["artifact_registry_path"]
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["active"]["v10_entry"] = {
        "path": "/tmp/unrelated-entry",
        "status": "ACTIVE",
        "in_sample_only": False,
    }
    registry_path.write_text(
        json.dumps(registry, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    load_bound_joint_exit_sizing_proof(
        evidence["joint_exit_proof_artifact"],
        context="UNIT_ENTRY_REGISTRY_CHANGE_DOES_NOT_REWRITE_EXIT",
        verify_source_files=True,
    )
    registry["active"]["v3_exit"]["status"] = "RETIRED"
    registry_path.write_text(
        json.dumps(registry, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="projection changed",
    ):
        load_bound_joint_exit_sizing_proof(
            evidence["joint_exit_proof_artifact"],
            context="UNIT_MUTATED_EXIT_REGISTRY",
            verify_source_files=True,
        )


def test_cached_sizing_authority_rehashes_same_stat_bytes(tmp_path: Path) -> None:
    evidence = write_passing_runtime_sizing_parity(tmp_path)
    authority = learned_sizing_authority_contract_metadata(
        adoption_artifact=evidence["adoption_artifact"]
    )
    snapshot = prepare_model_native_sizing_authority(
        authority,
        context="UNIT_CACHED_SIZING_HASH_BASELINE",
    )
    exit_identity = next(
        Path(path)
        for label, path, _sha256 in snapshot.content_hash_key
        if label.startswith("active_exit.") and Path(path).name == "identity.bin"
    )
    original = exit_identity.read_bytes()
    stat = exit_identity.stat()
    mutated = bytes([original[0] ^ 0x01]) + original[1:]
    exit_identity.write_bytes(mutated)
    os.utime(
        exit_identity,
        ns=(stat.st_atime_ns, stat.st_mtime_ns),
    )
    assert exit_identity.stat().st_ino == stat.st_ino
    assert exit_identity.stat().st_size == stat.st_size
    assert exit_identity.stat().st_mtime_ns == stat.st_mtime_ns
    with pytest.raises(
        ModelNativeSizingUnavailable,
        match="hash changed before snapshot",
    ):
        prepare_model_native_sizing_authority(
            authority,
            context="UNIT_CACHED_SIZING_HASH_MUTATION",
        )
