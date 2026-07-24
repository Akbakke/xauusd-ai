from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_sizing_execution_v1 import (
    ModelNativeSizingExecutionContractError,
    canonical_active_exit_replay_source_code_files,
    joint_exit_trace_sha256,
    load_bound_joint_exit_sizing_proof,
    load_bound_runtime_sizing_parity,
    read_bound_parquet_exact,
    recompute_joint_exit_replay_coverage,
    recompute_runtime_sizing_parity_coverage,
    require_canonical_active_exit_replay_launch_authority,
    require_joint_exit_portfolio_capacity,
    require_joint_replay_extends_canonical_oos_rows,
)
from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    ModelNativeSizingUnavailable,
    learned_sizing_authority_contract_metadata,
    prepare_model_native_sizing_authority,
)
from gx1.execution.v12_pipeline import V12Pipeline
from gx1.execution.v12_state_from_prebuilt import read_prebuilt_pair_manifest
from gx1.scripts.finalize_entry_model_native_sizing_v1 import (
    produce_canonical_active_exit_joint_sizing_proof,
)
from tests.model_native_sizing_support import (
    write_passing_joint_exit_sizing_proof,
    write_passing_runtime_sizing_parity,
)
from tests.test_v12_state_from_prebuilt_refresh import _prebuilt_fixture


def _frozen_pair_identity(
    pair_manifest_path: Path,
    generation_root: Path,
) -> dict[str, object]:
    binding = read_prebuilt_pair_manifest(
        pair_manifest_path,
        generation_root=generation_root,
    )
    return {
        "manifest_path": str(binding.manifest_path),
        "manifest_sha256": binding.manifest_sha256,
        "pair_generation_id": binding.pair_generation_id,
        "canonical_v3": {
            "path": str(binding.canonical_v3.parquet_path),
            "sha256": binding.canonical_v3.parquet_sha256,
            "rows": binding.canonical_v3.rows,
            "cols_total": binding.canonical_v3.cols_total,
        },
        "base28": {
            "path": str(binding.base28.parquet_path),
            "sha256": binding.base28.parquet_sha256,
            "rows": binding.base28.rows,
            "cols_total": binding.base28.cols_total,
        },
        "refresh_enabled": False,
    }


class _FrozenPairStub:
    def __init__(
        self,
        identity: dict[str, object],
        *,
        train_rank_binding: dict[str, object] | None = None,
    ) -> None:
        self._identity = identity
        self._train_rank_binding = train_rank_binding

    def frozen_pair_frames(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
        return pd.DataFrame(), pd.DataFrame(), dict(self._identity)

    def train_rank_reference_binding(self) -> dict[str, object]:
        if self._train_rank_binding is None:
            raise RuntimeError("PREBUILT_TRAIN_RANK_REFERENCE_UNBOUND")
        return dict(self._train_rank_binding)


class _CanonicalActiveExitStub:
    """Exercise the producer loop while isolating heavyweight model loading."""

    def __init__(
        self,
        *,
        tape: object,
        identity: dict[str, object],
        train_rank_binding: dict[str, object] | None = None,
    ) -> None:
        self._tape = tape
        self.prebuilt_loader = _FrozenPairStub(
            identity,
            train_rank_binding=train_rank_binding,
        )

    def make_exit_decision(
        self,
        trade: object,
        now: pd.Timestamp,
        bid: float,
        ask: float,
    ) -> dict[str, object]:
        del bid, ask
        closed = self._tape.get_closed_m1_bar(
            pd.Timestamp(now) - pd.Timedelta(minutes=1)
        )
        trade.update_bar(
            m1_bar_ts=closed["time"],
            bid=closed["bid_close"],
            ask=closed["ask_close"],
            m1_close=(closed["bid_close"] + closed["ask_close"]) / 2.0,
            bid_high=closed["bid_high"],
            bid_low=closed["bid_low"],
            ask_high=closed["ask_high"],
            ask_low=closed["ask_low"],
        )
        terminal = trade.bars_in_trade == 5
        return {
            "action_id": 1 if terminal else 0,
            "decision_source": (
                "UNIT_ACTIVE_EXIT_NOW" if terminal else "UNIT_ACTIVE_EXIT_HOLD"
            ),
        }


def test_canonical_active_exit_source_inventory_covers_local_import_closure() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pending = [
        "gx1/execution/v12_exit_iql_live.py",
        "gx1/execution/v12_model_native_state_live.py",
        "gx1/execution/v12_pipeline.py",
        "gx1/execution/v12_trade_state.py",
        "gx1/runtime/exit_decider_v12_adapter.py",
        "gx1/runtime/exit_iql_v2_adapter.py",
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

    assert observed.issubset(canonical_active_exit_replay_source_code_files())


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


def test_canonical_active_exit_producer_owns_every_full_test_trace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = write_passing_joint_exit_sizing_proof(tmp_path)
    diagnostic_proof, _ = load_bound_joint_exit_sizing_proof(
        evidence["joint_exit_proof_artifact"],
        context="UNIT_DIAGNOSTIC_ACTIVE_EXIT_PROOF",
        verify_source_files=True,
    )
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="caller-supplied replay/trace rows have zero launch authority",
    ):
        require_canonical_active_exit_replay_launch_authority(
            diagnostic_proof,
            context="UNIT_DIAGNOSTIC_ACTIVE_EXIT_LAUNCH",
        )

    prebuilt = _prebuilt_fixture(tmp_path / "prebuilt")
    pair_manifest = Path(prebuilt["pair_manifest"]).resolve()
    generation_root = Path(prebuilt["generation_root"]).resolve()
    identity = _frozen_pair_identity(pair_manifest, generation_root)

    from gx1.contracts.entry_model_native_state_v2 import (
        train_rank_reference_identity_v2,
    )
    from tests.model_native_rank_reference_support import (
        materialize_test_rank_reference,
    )

    _rank_source, rank_reference = materialize_test_rank_reference(
        tmp_path / "rank_reference",
        run_id="UNIT_ACTIVE_EXIT_RANK",
        history_start="2026-07-01T00:00:00Z",
        fit_start="2026-07-02T00:00:00Z",
        fit_end="2026-07-03T00:00:00Z",
    )
    rank_identity = train_rank_reference_identity_v2(rank_reference)

    observed_rank_references: list[object] = []

    def load_stub(
        cls: type[V12Pipeline],
        *,
        closed_m1_provider: object,
        train_rank_reference: object,
        **_kwargs: object,
    ) -> _CanonicalActiveExitStub:
        del cls
        observed_rank_references.append(train_rank_reference)
        return _CanonicalActiveExitStub(
            tape=closed_m1_provider,
            identity=identity,
            train_rank_binding=train_rank_reference_identity_v2(
                train_rank_reference
            ),
        )

    monkeypatch.setattr(
        V12Pipeline,
        "load_active_exit_replay",
        classmethod(load_stub),
    )
    with pytest.raises(TypeError):
        produce_canonical_active_exit_joint_sizing_proof(
            calibration_path=Path(
                evidence["calibration_artifact"]["json_path"]
            ),
            proof_path=Path(evidence["oos_proof_artifact"]["json_path"]),
            artifact_registry_path=Path(evidence["artifact_registry_path"]),
            source_tape_path=Path(
                evidence["oos_source"]["source_tape"]["path"]
            ),
            prebuilt_pair_manifest_path=pair_manifest,
            prebuilt_generation_root=generation_root,
            authority_root=Path(evidence["authority_root"]),
        )
    proof_path, proof = produce_canonical_active_exit_joint_sizing_proof(
        calibration_path=Path(evidence["calibration_artifact"]["json_path"]),
        proof_path=Path(evidence["oos_proof_artifact"]["json_path"]),
        artifact_registry_path=Path(evidence["artifact_registry_path"]),
        source_tape_path=Path(evidence["oos_source"]["source_tape"]["path"]),
        prebuilt_pair_manifest_path=pair_manifest,
        prebuilt_generation_root=generation_root,
        train_rank_reference_npz=rank_reference.path,
        train_rank_reference_sha256=rank_reference.sha256,
        authority_root=Path(evidence["authority_root"]),
    )

    assert proof_path.is_file()
    assert proof["decision"] == "PASS"
    producer = proof["canonical_active_exit_replay_producer"]
    assert producer["decision"] == "PASS"
    assert producer["train_rank_reference"] == rank_identity
    assert len(observed_rank_references) == 1
    assert observed_rank_references[0].sha256 == rank_reference.sha256
    assert producer["rows"] == 360
    assert producer["trade_rows"] == 300
    assert producer["trace_rows"] == 1_500
    assert proof["exit_replay_coverage"]["failed_rows"] == 0
    replay_rows = pd.read_parquet(proof["replay_rows"]["path"])
    replay_directions = pd.to_numeric(
        replay_rows["model_direction_index"]
    ).astype(int)
    assert set(replay_directions) == {0, 1, 2}
    assert set(
        pd.to_numeric(
            replay_rows.loc[replay_directions.isin([0, 1]), "exit_steps"]
        ).astype(int)
    ) == {5}
    flat_rows = replay_rows.loc[replay_directions == 2]
    assert set(flat_rows["exit_replay_status"].astype(str)) == {
        "FLAT_NO_ORDER"
    }
    assert set(pd.to_numeric(flat_rows["exit_steps"]).astype(int)) == {0}
    assert flat_rows["active_exit_fill_time"].isna().all()
    require_canonical_active_exit_replay_launch_authority(
        proof,
        context="UNIT_CANONICAL_ACTIVE_EXIT_LAUNCH",
    )
    forged_output = dict(proof)
    forged_output["canonical_active_exit_replay_producer"] = dict(producer)
    forged_output["canonical_active_exit_replay_producer"]["replay_rows"] = (
        proof["canonical_active_exit_replay_producer"]["canonical_oos_rows"]
    )
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="producer output bindings differ from joint proof",
    ):
        require_canonical_active_exit_replay_launch_authority(
            forged_output,
            context="UNIT_FORGED_CANONICAL_ACTIVE_EXIT_OUTPUT",
        )
    forged_tape = dict(proof)
    forged_tape["canonical_active_exit_replay_producer"] = dict(producer)
    forged_tape["canonical_active_exit_replay_producer"]["source_tape"] = (
        producer["runtime_predictions"]
    )
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="SourceTape binding differs from canonical OOS source",
    ):
        require_canonical_active_exit_replay_launch_authority(
            forged_tape,
            context="UNIT_FORGED_CANONICAL_ACTIVE_EXIT_TAPE",
        )
    forged_rank = dict(proof)
    forged_rank["canonical_active_exit_replay_producer"] = dict(producer)
    forged_rank["canonical_active_exit_replay_producer"][
        "train_rank_reference"
    ] = dict(rank_identity, sha256="0" * 64)
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="train rank reference identity invalid",
    ):
        require_canonical_active_exit_replay_launch_authority(
            forged_rank,
            context="UNIT_FORGED_CANONICAL_ACTIVE_EXIT_RANK",
        )
    dropped_rank = dict(proof)
    dropped_rank["canonical_active_exit_replay_producer"] = {
        key: value
        for key, value in producer.items()
        if key != "train_rank_reference"
    }
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="missing=\\['train_rank_reference'\\]",
    ):
        require_canonical_active_exit_replay_launch_authority(
            dropped_rank,
            context="UNIT_DROPPED_CANONICAL_ACTIVE_EXIT_RANK",
        )

    loaded, _ = load_bound_joint_exit_sizing_proof(
        {
            "json_path": str(proof_path.resolve()),
            "sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
        },
        context="UNIT_CANONICAL_ACTIVE_EXIT_RELOAD",
        verify_source_files=True,
    )
    assert (
        loaded["canonical_active_exit_replay_producer"][
            "producer_source_inventory_sha256"
        ]
        == producer["producer_source_inventory_sha256"]
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
