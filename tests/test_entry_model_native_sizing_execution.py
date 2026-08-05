from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

from gx1.contracts.entry_model_native_sizing_calibration_v1 import (
    recompute_sizing_oos_evidence,
)
from gx1.contracts.entry_model_native_sizing_execution_v1 import (
    MODEL_NATIVE_JOINT_EXIT_SIZING_EXTRA_COLUMNS,
    MODEL_NATIVE_JOINT_EXIT_SIZING_FACT_MODE,
    ModelNativeSizingExecutionContractError,
    canonical_unified_replay_source_code_files,
    read_bound_parquet_exact,
    recompute_joint_exit_replay_coverage,
    recompute_unified_replay_net_pnl,
    require_joint_replay_extends_canonical_oos_rows,
    unified_replay_net_cost_policy_metadata,
)
from gx1.scripts.finalize_entry_model_native_sizing_v1 import (
    SizingFinalizationError,
    produce_canonical_unified_joint_sizing_proof,
)
from tests.model_native_sizing_support import (
    write_passing_unified_replay_fixture,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_unified_source_inventory_is_offline_import_closure() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pending = ["gx1/scripts/finalize_entry_model_native_sizing_v1.py"]
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

    inventory = canonical_unified_replay_source_code_files()
    assert observed == set(inventory)
    forbidden = (
        "oanda",
        "serve_parity",
        "smart_entry_live",
        "sizing_authority_v1",
        "v12_trade_state",
        "finalize_entry_model_native_launch",
    )
    assert "gx1/replay/unified_exit_path_state_v1.py" in inventory
    assert not [path for path in inventory if any(token in path for token in forbidden)]


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


def test_unified_replay_recomputes_hard_direction_and_positive_net_edge(
    tmp_path: Path,
) -> None:
    evidence = write_passing_unified_replay_fixture(tmp_path)
    rows = pd.read_parquet(evidence["joint_replay_rows_path"])
    traces = pd.read_parquet(evidence["joint_exit_trace_rows_path"])
    bundle_sha = evidence["candidate_bundle_authority"]["bundle_commit_sha256"]

    canonical_rows = pd.read_parquet(
        evidence["oos_source"]["source_bindings"]["oos_rows"]["path"]
    )
    require_joint_replay_extends_canonical_oos_rows(
        canonical_oos_rows=canonical_rows,
        replay_rows=rows,
        context="UNIT_CANONICAL_TEST_IDENTITY",
    )
    coverage = recompute_joint_exit_replay_coverage(
        rows,
        exit_trace_rows=traces,
        candidate_bundle_sha256=bundle_sha,
        context="UNIT_FULL_TEST_M5_M1_REPLAY",
    )
    assert coverage["rows"] == 384
    assert coverage["trade_rows"] == 256
    assert coverage["long_rows"] == 128
    assert coverage["short_rows"] == 128
    assert coverage["flat_rows"] == 128
    assert coverage["failed_rows"] == 0

    replay_binding = {
        "path": str(Path(evidence["joint_replay_rows_path"]).resolve()),
        "sha256": _sha(Path(evidence["joint_replay_rows_path"])),
    }
    sizing = recompute_sizing_oos_evidence(
        calibration=evidence["calibration"],
        source_bindings={"oos_rows": replay_binding},
        evaluation_bundle=evidence["proof"]["evaluation_bundle"],
        context="UNIT_FULL_TEST_HARD_EDGE",
        fact_provenance_mode=MODEL_NATIVE_JOINT_EXIT_SIZING_FACT_MODE,
        extra_row_columns=MODEL_NATIVE_JOINT_EXIT_SIZING_EXTRA_COLUMNS,
        outcome_price_mode="model_exit_fill",
    )
    policy = sizing["direction_edge_policy"]["enforced_core"]
    assert policy["min_trade_direction_precision"] == 0.98
    assert sizing["direction_edge_admission"]["decision"] == "PASS"
    assert sizing["direction_edge_admission"]["trade_direction_precision"] == 1.0

    assert unified_replay_net_cost_policy_metadata()[
        "additional_round_trip_cost_bps"
    ] == 1.0
    net = recompute_unified_replay_net_pnl(rows, context="UNIT_FULL_TEST_NET_PNL")
    assert net["decision"] == "PASS"
    assert net["total_net_pnl_usd"] > 0.0
    assert net["mean_net_pnl_bps"] > 0.0


def test_direction_and_net_pnl_cannot_soft_pass_when_metrics_are_computable(
    tmp_path: Path,
) -> None:
    evidence = write_passing_unified_replay_fixture(tmp_path)
    rows = pd.read_parquet(evidence["joint_replay_rows_path"])

    wrong_targets = rows.copy()
    long_indices = wrong_targets.index[
        wrong_targets["model_direction_index"].astype(int) == 0
    ][:10]
    wrong_targets.loc[long_indices, "target_direction_index"] = 1
    wrong_path = tmp_path / "joint_exit_replay_rows_20260717T130000123456Z.parquet"
    wrong_targets.to_parquet(wrong_path, index=False)
    sizing = recompute_sizing_oos_evidence(
        calibration=evidence["calibration"],
        source_bindings={
            "oos_rows": {"path": str(wrong_path.resolve()), "sha256": _sha(wrong_path)}
        },
        evaluation_bundle=evidence["proof"]["evaluation_bundle"],
        context="UNIT_DIRECTION_EDGE_MUST_FAIL",
        fact_provenance_mode=MODEL_NATIVE_JOINT_EXIT_SIZING_FACT_MODE,
        extra_row_columns=MODEL_NATIVE_JOINT_EXIT_SIZING_EXTRA_COLUMNS,
        outcome_price_mode="model_exit_fill",
    )
    assert sizing["direction_edge_admission"]["decision"] == "FAIL"
    assert sizing["direction_edge_admission"]["failures"]

    losing = rows.copy()
    long_mask = losing["model_direction_index"].astype(int) == 0
    short_mask = losing["model_direction_index"].astype(int) == 1
    losing.loc[long_mask, "model_exit_fill_bid"] = (
        losing.loc[long_mask, "entry_ask"].astype(float) - 1.0
    )
    losing.loc[short_mask, "model_exit_fill_ask"] = (
        losing.loc[short_mask, "entry_bid"].astype(float) + 1.0
    )
    net = recompute_unified_replay_net_pnl(losing, context="UNIT_NET_EDGE_MUST_FAIL")
    assert net["decision"] == "FAIL"
    assert net["failures"] == ["net_pnl_not_strictly_positive"]


def test_unified_exit_trace_and_status_mutations_fail_closed(tmp_path: Path) -> None:
    evidence = write_passing_unified_replay_fixture(tmp_path)
    rows = pd.read_parquet(evidence["joint_replay_rows_path"])
    traces = pd.read_parquet(evidence["joint_exit_trace_rows_path"])
    bundle_sha = evidence["candidate_bundle_authority"]["bundle_commit_sha256"]

    broken_status = rows.copy()
    trade_index = broken_status.index[
        broken_status["model_direction_index"].astype(int).isin([0, 1])
    ][0]
    broken_status.loc[trade_index, "exit_replay_status"] = "HORIZON_CAP"
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="every non-FLAT row must reach unified Exit EXIT_NOW",
    ):
        recompute_joint_exit_replay_coverage(
            broken_status,
            exit_trace_rows=traces,
            candidate_bundle_sha256=bundle_sha,
            context="UNIT_HORIZON_CAP",
        )

    broken_trace = traces.drop(traces.index[0]).copy()
    with pytest.raises(
        ModelNativeSizingExecutionContractError,
        match="steps are not contiguous",
    ):
        recompute_joint_exit_replay_coverage(
            rows,
            exit_trace_rows=broken_trace,
            candidate_bundle_sha256=bundle_sha,
            context="UNIT_TRACE_STEP_GAP",
        )


def test_canonical_replay_producer_fails_closed_on_missing_chain(
    tmp_path: Path,
) -> None:
    with pytest.raises(SizingFinalizationError, match="source file missing"):
        produce_canonical_unified_joint_sizing_proof(
            calibration_path=tmp_path / "calibration.json",
            proof_path=tmp_path / "proof.json",
            source_tape_path=tmp_path / "tape.parquet",
            prebuilt_pair_manifest_path=tmp_path / "pair.json",
            prebuilt_generation_root=tmp_path / "generations",
            multi_tf_cache_dir=tmp_path / "multi_tf_cache",
            train_rank_reference_npz=tmp_path / "rank.npz",
            train_rank_reference_sha256="0" * 64,
            authority_root=tmp_path,
        )
