from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from gx1.contracts.entry_execution_causality_v1 import (
    build_entry_execution_causality_audit,
    legacy_same_close_target_contract_failures,
    require_entry_execution_causality_audit,
)
from gx1.contracts.entry_fitted_q_v1 import entry_fitted_q_contract
from gx1.contracts.entry_causal_m1_target_policy_v1 import (
    causal_m1_direction_diagnostic_outcome_contract,
)
from gx1.scripts.audit_entry_execution_causality_v1 import build_report
from tests.entry_direction_target_policy_support import causal_m1_target_policy_fixture


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path.resolve()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy_target_contract() -> dict[str, object]:
    return {
        "target": "train_fitted_executable_pnl_side_margin_bps",
        "long_entry_price": "ask_close_t0",
        "long_exit_price": "bid_close_t_plus_fitted_horizon",
        "short_entry_price": "bid_close_t0",
        "short_exit_price": "ask_close_t_plus_fitted_horizon",
        "target_affects_feature_availability": False,
    }


def _causal_target_contract() -> dict[str, object]:
    return {
        "entry_decision_time": "authoritative_m5_bar_close_available_at",
        "long_entry_price": (
            "ask_open_first_authoritative_m1_at_or_after_entry_decision"
        ),
        "short_entry_price": (
            "bid_open_first_authoritative_m1_at_or_after_entry_decision"
        ),
        "long_exit_price": (
            "bid_open_first_authoritative_m1_at_or_after_fitted_exit_decision"
        ),
        "short_exit_price": (
            "ask_open_first_authoritative_m1_at_or_after_fitted_exit_decision"
        ),
        "entry_fill_binding": "exact_m1_quote_time_and_bid_ask",
        "target_affects_feature_availability": False,
    }


def _audit_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, tuple[Path, Path]]]:
    dataset = (tmp_path / "dataset").resolve()
    dataset.mkdir()
    run_id = "ENTRY_EXECUTION_CAUSALITY_TEST_RUN"
    policy_sha = "a" * 64
    signal = _write(
        tmp_path / "signal.json",
        {
            "entry_run_id": run_id,
            "feature_ranking": {
                "entry_direction_target_policy_sha256": policy_sha,
                "target_contract": _legacy_target_contract(),
            },
        },
    )
    split_paths: dict[str, tuple[Path, Path]] = {}
    for split in ("train", "val"):
        manifest = _write(
            dataset / f"{split}.manifest.json",
            {
                "extra": {
                    "entry_run_id": run_id,
                    "entry_fitted_q": entry_fitted_q_contract(),
                    "diagnostic_outcome_policy_sha256": policy_sha,
                    "diagnostic_outcome_labels": {
                        "diagnostic_outcome_policy_sha256": policy_sha,
                    },
                    "entry_position_size_target_policy": {
                        "entry_direction_target_policy_sha256": policy_sha,
                    },
                }
            },
        )
        lifecycle = _write(
            tmp_path / "lifecycle" / f"{split}.json",
            {
                "decision": "PASS",
                "entry_run_id": run_id,
                "entry_side_selection": (
                    "both_sides_for_every_causal_entry_snapshot"
                ),
                "first_state_post_fill_closed_bars": 1,
                "state_row_timestamp_semantics": "authoritative_m1_bar_start",
                "decision_time_semantics": (
                    "authoritative_m1_bar_close_available_at"
                ),
                "future_outcomes_used_as_model_inputs": False,
                "sample_selection_depends_on_future_target": False,
                "exit_supervision_authority": (
                    "executable_exit_now_reward_plus_train_fitted_q"
                ),
            },
        )
        split_paths[split] = (manifest, lifecycle)
    return dataset, signal, split_paths


def _install_causal_ranking(
    signal: Path, split_paths: dict[str, tuple[Path, Path]]
) -> dict[str, object]:
    """Make the ranker and diagnostic contract genuinely causal for a test.

    The caller deliberately retains the legacy sizing manifest unless it needs
    to replace it.  That lets the audit prove that a causal-looking ranker is
    not enough to authorize a mixed auxiliary surface.
    """

    policy = causal_m1_target_policy_fixture(source_parquet_sha256="a" * 64)
    signal_payload = json.loads(signal.read_text(encoding="utf-8"))
    ranking = signal_payload["feature_ranking"]
    ranking.update(
        {
            "source_sha256": "a" * 64,
            "entry_direction_target_policy": policy,
            "entry_direction_target_policy_sha256": policy["policy_sha256"],
            # The ranker's objective contract and the causal-M1 policy's
            # outcome contract have separate owners.  They overlap on fill
            # semantics, but each carries additional metadata the other must
            # not be required to duplicate.  This is the production shape
            # that the execution-causality audit must accept.
            "target_contract": _causal_target_contract(),
        }
    )
    signal.write_text(json.dumps(signal_payload, sort_keys=True), encoding="utf-8")
    diagnostic = causal_m1_direction_diagnostic_outcome_contract(policy)
    for manifest_path, _ in split_paths.values():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        extra = manifest["extra"]
        extra["diagnostic_outcome_policy_sha256"] = policy["policy_sha256"]
        extra["diagnostic_outcome_labels"] = diagnostic
        extra["entry_position_size_target_policy"] = {
            "entry_direction_target_policy_sha256": policy["policy_sha256"]
        }
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return policy


def test_same_close_target_is_explicitly_rejected() -> None:
    failures = legacy_same_close_target_contract_failures(_legacy_target_contract())

    assert "ENTRY_EXECUTION_CAUSALITY_LONG_SAME_CLOSE_ENTRY" in failures
    assert "ENTRY_EXECUTION_CAUSALITY_SHORT_SAME_CLOSE_ENTRY" in failures
    assert any("M1_FILL_CONTRACT_UNBOUND" in item for item in failures)
    assert legacy_same_close_target_contract_failures(_causal_target_contract()) == []


def test_manifest_audit_blocks_legacy_auxiliary_target_even_when_fitted_q_is_causal(
    tmp_path: Path,
) -> None:
    dataset, signal, split_paths = _audit_fixture(tmp_path)

    report = build_report(
        dataset_dir=dataset,
        signal_manifest_path=signal,
        split_paths=split_paths,
    )

    assert report["decision"] == "BLOCK"
    assert report["entry_fitted_q_m1_fill_lifecycle_bound"] is True
    assert report["active_auxiliary_targets_m1_fill_bound"] is False
    assert report["training_authorized"] is False
    assert report["future_causal_rebuild_required"] is True
    with pytest.raises(
        RuntimeError, match="ENTRY_EXECUTION_CAUSALITY_AUDIT_TRAINING_BLOCKED"
    ):
        require_entry_execution_causality_audit(
            report,
            expected_dataset_dir=str(dataset),
            expected_entry_run_id="ENTRY_EXECUTION_CAUSALITY_TEST_RUN",
            require_training_authorized=True,
        )


def test_only_a_hash_bound_causal_split_report_can_authorize_training() -> None:
    split_rows = [
        {
            "split": split,
            "dataset_manifest_path": f"/immutable/{split}.manifest.json",
            "dataset_manifest_sha256": "b" * 64,
            "lifecycle_manifest_path": f"/immutable/{split}.lifecycle.json",
            "lifecycle_manifest_sha256": "c" * 64,
            "entry_fitted_q_m1_fill_lifecycle_bound": True,
            "active_auxiliary_targets_m1_fill_bound": True,
        }
        for split in ("train", "val")
    ]
    report = build_entry_execution_causality_audit(
        dataset_dir="/immutable/dataset",
        entry_run_id="ENTRY_EXECUTION_CAUSALITY_TEST_RUN",
        signal_manifest_path="/immutable/signal.json",
        signal_manifest_sha256="d" * 64,
        ranking_target_contract=_causal_target_contract(),
        split_rows=split_rows,
    )

    assert report["decision"] == "PASS"
    assert require_entry_execution_causality_audit(
        report, require_training_authorized=True
    )["training_authorized"] is True

    tampered = json.loads(json.dumps(report))
    tampered["splits"][0]["active_auxiliary_targets_m1_fill_bound"] = False
    with pytest.raises(
        RuntimeError, match="ENTRY_EXECUTION_CAUSALITY_AUDIT_SPLIT_OUTCOME_MISMATCH"
    ):
        require_entry_execution_causality_audit(
            tampered, require_training_authorized=False
        )


def test_causal_ranking_refuses_a_legacy_sizing_payload(tmp_path: Path) -> None:
    dataset, signal, split_paths = _audit_fixture(tmp_path)
    _install_causal_ranking(signal, split_paths)

    with pytest.raises(
        RuntimeError, match="ENTRY_CAUSAL_M1_POSITION_SIZE_TARGET_POLICY_SCHEMA_INVALID"
    ):
        build_report(
            dataset_dir=dataset,
            signal_manifest_path=signal,
            split_paths=split_paths,
        )


def test_causal_ranking_requires_exact_m1_diagnostic_contract(tmp_path: Path) -> None:
    dataset, signal, split_paths = _audit_fixture(tmp_path)
    _install_causal_ranking(signal, split_paths)
    for manifest_path, _ in split_paths.values():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["extra"]["diagnostic_outcome_labels"][
            "diagnostic_outcome_label_source"
        ] = "legacy_m5_close_outcome"
        manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="ENTRY_EXECUTION_CAUSALITY_TRAIN_CAUSAL_DIAGNOSTIC_CONTRACT_INVALID",
    ):
        build_report(
            dataset_dir=dataset,
            signal_manifest_path=signal,
            split_paths=split_paths,
        )
