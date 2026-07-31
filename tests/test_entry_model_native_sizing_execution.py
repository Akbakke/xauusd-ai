from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_sizing_execution_v1 import (
    ModelNativeSizingExecutionContractError,
    canonical_unified_replay_source_code_files,
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
from gx1.scripts.finalize_entry_model_native_sizing_v1 import (
    SizingFinalizationError,
    produce_canonical_unified_joint_sizing_proof,
)
from tests.model_native_sizing_support import (
    write_passing_runtime_sizing_parity,
)


def test_canonical_unified_source_inventory_covers_local_import_closure() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pending = [
        "gx1/execution/v12_model_native_state_live.py",
        "gx1/execution/v12_pipeline.py",
        "gx1/execution/v12_trade_state.py",
        "gx1/scripts/finalize_entry_model_native_sizing_v1.py",
    ]
    observed: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in observed:
            continue
        source_path = repo_root / relative
        if not source_path.is_file():
            continue
        observed.add(relative)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
            elif isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
        for module in modules:
            if not module.startswith(("gx1", "gx1_guards")):
                continue
            module_path = Path(*module.split("."))
            candidate = module_path.with_suffix(".py")
            package_init = module_path / "__init__.py"
            if (repo_root / candidate).is_file():
                pending.append(candidate.as_posix())
            elif (repo_root / package_init).is_file():
                pending.append(package_init.as_posix())

    assert observed.issubset(canonical_unified_replay_source_code_files())


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


def test_joint_unified_exit_sizing_proof_is_row_recomputed_and_candidate_bound(
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
    assert (
        proof["candidate_bundle_authority"]["bundle_dir"]
        == str(evidence["bundle_dir"])
    )

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
        rows.loc[trade_rows, "model_exit_fill_bid"].to_numpy(),
        rows.loc[trade_rows, "exit_bid"].to_numpy(),
    )
    final_trace = exit_trace_rows.sort_values(
        ["reference_row_id", "step"]
    ).groupby(
        "reference_row_id",
        sort=False,
    ).tail(1)
    expected_final = rows.loc[
        trade_rows,
        ["reference_row_id", "model_exit_fill_bid"],
    ].set_index("reference_row_id")
    observed_final = final_trace.set_index("reference_row_id")
    assert expected_final.index.equals(observed_final.index)
    assert np.array_equal(
        observed_final["state_bid"].to_numpy(),
        expected_final["model_exit_fill_bid"].to_numpy(),
    )
    assert set(exit_trace_rows["decision_source"]) == {"unified_model"}
    broken_status = rows.copy()
    broken_status.loc[0, "exit_replay_status"] = "HORIZON_CAP"
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="every non-FLAT row must reach unified Exit EXIT_NOW",
    ):
        recompute_joint_exit_replay_coverage(
            broken_status,
            exit_trace_rows=exit_trace_rows,
            candidate_bundle_sha256=proof["candidate_bundle_authority"][
                "bundle_commit_sha256"
            ],
            context="UNIT_HORIZON_CAP",
        )

    broken_trace = rows.copy()
    broken_trace.loc[0, "exit_trace_sha256"] = "not-a-hash"
    with pytest.raises(ModelNativeSizingExecutionContractError, match="SHA-256"):
        recompute_joint_exit_replay_coverage(
            broken_trace,
            exit_trace_rows=exit_trace_rows,
            candidate_bundle_sha256=proof["candidate_bundle_authority"][
                "bundle_commit_sha256"
            ],
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
            candidate_bundle_sha256=proof["candidate_bundle_authority"][
                "bundle_commit_sha256"
            ],
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
            candidate_bundle_sha256=proof["candidate_bundle_authority"][
                "bundle_commit_sha256"
            ],
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
            candidate_bundle_sha256=proof["candidate_bundle_authority"][
                "bundle_commit_sha256"
            ],
            context="UNIT_TRACE_STEP_GAP",
        )

    gapped_trace_rows = exit_trace_rows.copy()
    first_reference = str(gapped_trace_rows.iloc[0]["reference_row_id"])
    first_mask = gapped_trace_rows["reference_row_id"].astype(str) == first_reference
    first_indices = gapped_trace_rows.index[first_mask]
    gapped_trace_rows.loc[first_indices[1], "closed_bar_time"] = (
        pd.Timestamp(
            gapped_trace_rows.loc[first_indices[1], "closed_bar_time"]
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
            candidate_bundle_sha256=proof["candidate_bundle_authority"][
                "bundle_commit_sha256"
            ],
            context="UNIT_GAPPED_M1_TRACE",
        )

    candidate_model = (
        Path(proof["candidate_bundle_authority"]["bundle_dir"])
        / "model_state_dict.pt"
    )
    original_model = candidate_model.read_bytes()
    candidate_model.write_bytes(b"mutated candidate bytes")
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="model_state_dict.pt hash mismatch",
    ):
        load_bound_joint_exit_sizing_proof(
            evidence["joint_exit_proof_artifact"],
            context="UNIT_MUTATED_CANDIDATE_BYTES",
            verify_source_files=True,
        )
    candidate_model.write_bytes(original_model)


def test_canonical_replay_producer_fails_closed_on_missing_chain(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        SizingFinalizationError,
        match="source file missing",
    ):
        produce_canonical_unified_joint_sizing_proof(
            calibration_path=tmp_path / "calibration.json",
            proof_path=tmp_path / "proof.json",
            source_tape_path=tmp_path / "tape.parquet",
            prebuilt_pair_manifest_path=tmp_path / "pair.json",
            prebuilt_generation_root=tmp_path / "generations",
            train_rank_reference_npz=tmp_path / "rank.npz",
            train_rank_reference_sha256="0" * 64,
            authority_root=tmp_path,
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
    candidate_model = next(
        Path(path)
        for label, path, _sha256 in snapshot.content_hash_key
        if label == "candidate_bundle.model_state_dict.pt"
    )
    original = candidate_model.read_bytes()
    stat = candidate_model.stat()
    mutated = bytes([original[0] ^ 0x01]) + original[1:]
    candidate_model.write_bytes(mutated)
    os.utime(
        candidate_model,
        ns=(stat.st_atime_ns, stat.st_mtime_ns),
    )
    assert candidate_model.stat().st_ino == stat.st_ino
    assert candidate_model.stat().st_size == stat.st_size
    assert candidate_model.stat().st_mtime_ns == stat.st_mtime_ns
    with pytest.raises(
        ModelNativeSizingUnavailable,
        match="hash changed before snapshot",
    ):
        prepare_model_native_sizing_authority(
            authority,
            context="UNIT_CACHED_SIZING_HASH_MUTATION",
        )
