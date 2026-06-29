import argparse
from pathlib import Path

from gx1.scripts.verify_entry_candidate_readiness_v1 import _smoke_edge_checks, run
from gx1.scripts.verify_entry_training_readiness_v1 import EXPECTED_ACTIVE_TRAINING_HEADS, EXPECTED_BLOCKED_HEADS


def _passing_smoke_audit() -> dict:
    split = {
        "rows": 128,
        "direction": {
            "accuracy": 0.46,
            "majority_baseline_accuracy": 0.34,
            "beats_majority_baseline": True,
        },
        "bad_path": {"prob_vs_path_quality_spearman": -0.22},
        "specialist_gate": {
            "finite": True,
            "row_sum_max_abs_error": 1e-7,
            "active_specialist_count_gt_1pct": 6,
            "entropy_mean": 1.0,
            "mean_weight": {
                "structure_swing_encoder": 0.18,
                "smc_liquidity_encoder": 0.17,
                "trend_ema_encoder": 0.16,
                "vol_compression_encoder": 0.16,
                "momentum_flow_encoder": 0.17,
                "session_regime_encoder": 0.16,
            },
        },
    }
    return {
        "decision": "PASS",
        "failures": [],
        "dataset_dir": "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_foundation_seq146_smoke",
        "require_edge": True,
        "require_specialist_fusion": True,
        "required_training_specialists": [
            "structure_swing_encoder",
            "smc_liquidity_encoder",
            "trend_ema_encoder",
            "vol_compression_encoder",
            "momentum_flow_encoder",
            "session_regime_encoder",
        ],
        "min_active_specialists": 6,
        "min_gate_entropy": 0.05,
        "require_head_contract": True,
        "head_contract": {
            "decision": "PASS",
            "failures": [],
            "active_training_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
            "blocked_heads": list(EXPECTED_BLOCKED_HEADS),
        },
        "pretrain_manifest_contract": {
            "decision": "PASS",
            "failures": [],
            "feature_objective_coverage_all_present": True,
            "feature_objective_liveness_all_live": True,
            "feature_source_field_liveness_all_live": True,
            "specialist_objective_routing_all_present_and_expected": True,
            "specialist_input_liveness_all_live": True,
            "specialist_active_heads_match_target": True,
            "specialist_blocked_heads_match_target": True,
            "specialist_model_contract_valid": True,
            "specialist_model_contract_set_exact": True,
            "specialist_model_contract_owned_objectives_match": True,
            "smoke_dataset_audit_provenance_all_artifacts_present": True,
            "smoke_dataset_audit_provenance_all_artifact_hashes_present": True,
            "worktree_critical_gate_review_ok": True,
        },
        "bundle_specialist_model_contract": {
            "decision": "PASS",
            "valid": True,
            "set_exact": True,
            "owned_objectives_match": True,
            "support_heads_match": True,
            "signal_families_match": True,
            "model_roles_match": True,
            "failures": [],
        },
        "data_splits": ["val", "test"],
        "bundle_summary": {
            "sanity_bundle": False,
            "seq_input_dim": 146,
            "snap_input_dim": 146,
            "multi_tf_enabled": True,
            "specialist_fusion_enabled": True,
            "specialist_model_contract_declared_valid": True,
            "specialist_model_contract_valid": True,
            "specialist_model_contract_set_exact": True,
            "specialist_model_contract_owned_objectives_match": True,
            "specialist_model_contract_support_heads_match": True,
            "specialist_model_contract_signal_families_match": True,
            "specialist_model_contract_model_roles_match": True,
            "specialist_groups": [
                "structure_swing_encoder",
                "smc_liquidity_encoder",
                "trend_ema_encoder",
                "vol_compression_encoder",
                "momentum_flow_encoder",
                "session_regime_encoder",
            ],
        },
        "splits": {"val": split, "test": split},
    }


def test_smoke_edge_checks_pass_on_actual_edge_contract() -> None:
    checks = _smoke_edge_checks(_passing_smoke_audit())

    assert all(check["ok"] for check in checks)


def test_smoke_edge_checks_reject_sanity_plumbing_audit() -> None:
    report = _passing_smoke_audit()
    report["bundle_summary"]["sanity_bundle"] = True
    report["require_edge"] = False
    report["splits"]["val"]["direction"]["beats_majority_baseline"] = False
    report["splits"]["val"]["bad_path"]["prob_vs_path_quality_spearman"] = 0.1

    checks = _smoke_edge_checks(report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle audit is from actual train output, not sanity bundle" in failed
    assert "smoke bundle audit was run with require_edge" in failed
    assert "direction beats majority on all audited splits" in failed
    assert "bad_path probability ranks worse path quality higher" in failed


def test_smoke_edge_checks_reject_missing_head_contract() -> None:
    report = _passing_smoke_audit()
    report["require_head_contract"] = False
    report["head_contract"] = {"decision": "FAIL", "failures": ["missing tf_agreement"]}

    checks = _smoke_edge_checks(report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle audit was run with require_head_contract" in failed
    assert "smoke bundle head contract PASS" in failed


def test_smoke_edge_checks_reject_missing_pretrain_manifest_contract() -> None:
    report = _passing_smoke_audit()
    report["pretrain_manifest_contract"] = {
        "decision": "FAIL",
        "failures": ["artifact hash mismatch"],
        "feature_objective_coverage_all_present": True,
        "feature_objective_liveness_all_live": True,
        "feature_source_field_liveness_all_live": True,
        "specialist_objective_routing_all_present_and_expected": True,
        "specialist_input_liveness_all_live": True,
        "specialist_active_heads_match_target": True,
        "specialist_blocked_heads_match_target": True,
        "smoke_dataset_audit_provenance_all_artifacts_present": False,
        "smoke_dataset_audit_provenance_all_artifact_hashes_present": False,
        "worktree_critical_gate_review_ok": False,
    }

    checks = _smoke_edge_checks(report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle audit validated pre-train manifest provenance" in failed


def test_smoke_edge_checks_reject_missing_pretrain_specialist_model_contract() -> None:
    report = _passing_smoke_audit()
    report["pretrain_manifest_contract"]["specialist_model_contract_valid"] = False
    report["pretrain_manifest_contract"]["specialist_model_contract_set_exact"] = False
    report["pretrain_manifest_contract"]["specialist_model_contract_owned_objectives_match"] = False

    checks = _smoke_edge_checks(report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle audit validated pre-train manifest provenance" in failed


def test_smoke_edge_checks_reject_missing_bundle_specialist_model_contract() -> None:
    report = _passing_smoke_audit()
    report["bundle_summary"]["specialist_model_contract_valid"] = False
    report["bundle_specialist_model_contract"]["support_heads_match"] = False
    report["bundle_specialist_model_contract"]["failures"] = ["support heads mismatch"]

    checks = _smoke_edge_checks(report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle specialist model contract is preserved in bundle metadata" in failed


def test_smoke_edge_checks_reject_missing_worktree_critical_gate_proof() -> None:
    report = _passing_smoke_audit()
    report["pretrain_manifest_contract"]["worktree_critical_gate_review_ok"] = False

    checks = _smoke_edge_checks(report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle audit validated pre-train manifest provenance" in failed


def test_smoke_edge_checks_rejects_partial_active_head_contract() -> None:
    report = _passing_smoke_audit()
    report["head_contract"]["active_training_heads"] = ["direction", "path_quality", "tf_agreement"]

    checks = _smoke_edge_checks(report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle head contract PASS" in failed


def test_smoke_edge_checks_reject_loose_specialist_gate_contract() -> None:
    report = _passing_smoke_audit()
    report["require_specialist_fusion"] = False
    report["min_active_specialists"] = 3
    report["min_gate_entropy"] = 0.0
    report["splits"]["val"]["specialist_gate"]["active_specialist_count_gt_1pct"] = 3
    report["splits"]["val"]["specialist_gate"]["entropy_mean"] = 0.0
    report["splits"]["val"]["specialist_gate"]["mean_weight"]["momentum_flow_encoder"] = 0.0

    checks = _smoke_edge_checks(report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle audit was run with specialist-fusion gate contract" in failed
    assert "specialist gate is finite, normalized, non-collapsed, and entropic" in failed
    assert "each required specialist has non-collapsed gate weight" in failed


def test_smoke_edge_checks_rejects_extra_specialist_group() -> None:
    report = _passing_smoke_audit()
    report["required_training_specialists"].append("price_action_candle_encoder")
    report["bundle_summary"]["specialist_groups"].append("price_action_candle_encoder")
    for split in report["splits"].values():
        split["specialist_gate"]["mean_weight"]["price_action_candle_encoder"] = 0.05

    checks = _smoke_edge_checks(report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle audit was run with specialist-fusion gate contract" in failed
    assert "smoke bundle has exact specialist groups" in failed


def test_candidate_readiness_current_artifacts_are_not_ready_without_actual_smoke_train(tmp_path: Path) -> None:
    report = run(
        argparse.Namespace(
            audit_doc="/home/andre2/src/GX1_ENGINE/docs/ENTRY_FOUNDATION_AUDIT_20260628.md",
            smoke_bundle_audit_json="/home/andre2/GX1_DATA/reports/entry_foundation_smoke_bundle_audit_20260628_v1/ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json",
            out_dir=str(tmp_path),
            min_active_specialists=3,
            fail_on_not_ready=False,
            quiet=True,
        )
    )

    assert report["decision"] == "NOT_READY_FOR_CANDIDATE_TRAINING"
    assert report["candidate_training_allowed_with_explicit_vedtak"] is False
    assert report["promotion_shadow_live_allowed"] is False
    assert set(report["artifact_fingerprints"]) == set(report["artifacts"])
    artifact_gate = next(gate for gate in report["gates"] if gate["name"] == "artifact_provenance")
    assert artifact_gate["decision"] == "PASS"
    for name, row in report["artifact_fingerprints"].items():
        assert row["path"] == report["artifacts"][name]
        assert row["exists"] is True
        assert row["size_bytes"] > 0
        assert len(row["sha256"]) == 64
    failed = {failure["check"] for failure in report["failures"]}
    assert "smoke bundle audit is from actual train output, not sanity bundle" in failed
    assert "smoke bundle audit was run with require_edge" in failed
    assert Path(report["json_path"]).exists()
