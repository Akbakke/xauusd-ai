import argparse
import json
from pathlib import Path

from gx1.features.entry_foundation_structure_v1 import (
    FOUNDATION_STRUCTURE_FEATURE_VERSION,
    FOUNDATION_STRUCTURE_SOURCE_FIELDS,
)
from gx1.scripts.verify_entry_training_readiness_v1 import (
    _activation_transition,
    _balanced_label_counts,
    _check,
    run,
)


def test_balanced_label_counts_requires_three_nonzero_equal_classes() -> None:
    assert _balanced_label_counts({"0": 8, "1": 8, "2": 8}) is True
    assert _balanced_label_counts({"0": 8, "1": 8}) is False
    assert _balanced_label_counts({"0": 8, "1": 7, "2": 8}) is False
    assert _balanced_label_counts({"0": 8, "1": 8, "2": 0}) is False


def test_check_returns_machine_readable_gate_row() -> None:
    row = _check("sample", True, {"x": 1})

    assert row == {"name": "sample", "ok": True, "details": {"x": 1}}


def test_activation_transition_routes_to_post_apply_after_alias_switch() -> None:
    transition = _activation_transition(
        foundation_contract_ready=False,
        foundation_activation={
            "adoption_candidate_ready": True,
            "activation_plan_ready": True,
            "activation_apply_ready": False,
            "activation_apply_applied": True,
            "activation_post_apply_completed": False,
            "activation_apply_command": [
                "scripts/entry_next_edge_control.sh",
                "foundation-activation-apply",
                "--apply",
                "--vedtak",
                "<id>",
            ],
            "activation_post_apply_command": [
                "scripts/entry_next_edge_control.sh",
                "foundation-activation-post-apply",
                "--activation-apply-json",
                "/tmp/activation-apply.json",
                "--apply",
                "--vedtak",
                "<id>",
            ],
        },
    )

    assert transition["activation_required_before_smoke"] is True
    assert transition["activation_apply_required_before_smoke"] is False
    assert transition["activation_post_apply_required_before_smoke"] is True
    assert transition["next_allowed_command"].startswith(
        "scripts/entry_next_edge_control.sh foundation-activation-post-apply"
    )


def test_training_readiness_current_artifacts(tmp_path: Path) -> None:
    report = run(
        argparse.Namespace(
            audit_doc="/home/andre2/src/GX1_ENGINE/docs/ENTRY_FOUNDATION_AUDIT_20260628.md",
            out_dir=str(tmp_path),
            fail_on_not_ready=False,
            quiet=True,
        )
    )

    execution_gate = next(gate for gate in report["gates"] if gate["name"] == "execution_hygiene")
    dirty_count = execution_gate["checks"][0]["details"]["dirty_count"]
    feature_gate = next(gate for gate in report["gates"] if gate["name"] == "feature_foundation")
    smoke_bundle_gate = next(gate for gate in report["gates"] if gate["name"] == "reference_smoke_bundle_audit")
    feature_checks = {check["name"]: check for check in feature_gate["checks"]}
    version_check = feature_checks["feature audit foundation structure version matches code"]
    activation = report["foundation_activation"]
    assert activation["training_allowed"] is False
    assert activation["activation_allowed_without_vedtak"] is False
    assert activation["activation_apply_decision"]
    assert activation["activation_post_apply_report"]
    assert activation["activation_post_apply_command"][0:2] == [
        "scripts/entry_next_edge_control.sh",
        "foundation-activation-post-apply",
    ]
    if activation["activation_apply_applied"]:
        assert activation["activation_apply_mutation_performed"] is True
        assert activation["activation_apply_ready"] is False
        assert activation["activation_post_apply_waiting_for_activation"] is False
        if activation["activation_post_apply_completed"]:
            assert activation["activation_post_apply_decision"] == "POST_APPLY_REFRESH_COMPLETED"
            assert activation["activation_post_apply_mutations_performed"] is True
        else:
            assert activation["activation_post_apply_completed"] is False
    else:
        assert activation["activation_apply_ready"] is True
        assert activation["activation_apply_mutation_performed"] is False
        assert activation["activation_post_apply_waiting_for_activation"] is True
        assert activation["activation_post_apply_completed"] is False
        assert activation["activation_post_apply_mutations_performed"] is False
    assert "foundation_activation_required_before_smoke" in report
    assert "foundation_activation_apply_required_before_smoke" in report
    assert "foundation_activation_post_apply_required_before_smoke" in report
    assert version_check["details"]["code_foundation_structure_feature_version"] == FOUNDATION_STRUCTURE_FEATURE_VERSION
    if report["foundation_contract_ready_for_smoke"]:
        assert report["foundation_contract_ready_for_smoke"] is True
        assert report["foundation_activation_required_before_smoke"] is False
        assert report["foundation_activation_apply_required_before_smoke"] is False
        assert report["foundation_activation_post_apply_required_before_smoke"] is False
        if dirty_count:
            assert report["decision"] == "READY_FOR_VEDTAK_SMOKE_TRAIN_AFTER_GIT_CLEAN"
            assert report["smoke_training_allowed_with_explicit_vedtak"] is False
            assert report["execution_blockers"]
            assert report["next_allowed_command"].startswith("clean git worktree")
        else:
            assert report["decision"] == "READY_FOR_VEDTAK_SMOKE_TRAIN"
            assert report["smoke_training_allowed_with_explicit_vedtak"] is True
            assert report["execution_blockers"] == []
        assert "smoke-train --vedtak <id> --require-edge-audit" in report["next_allowed_command"]
        if smoke_bundle_gate["decision"] == "FAIL":
            assert any(
                failure["gate"] == "reference_smoke_bundle_audit"
                for failure in report["diagnostic_failures"]
            )
            assert not any(
                failure["gate"] == "reference_smoke_bundle_audit"
                for failure in report["failures"]
            )
    else:
        assert report["foundation_contract_ready_for_smoke"] is False
        assert report["decision"] == "NOT_READY"
        assert report["smoke_training_allowed_with_explicit_vedtak"] is False
        if report["foundation_activation_required_before_smoke"]:
            assert activation["adoption_candidate_ready"] is True
            assert activation["activation_plan_ready"] is True
            assert activation["activation_plan_report"]
            assert activation["activation_apply_report"]
            assert activation["post_apply_command_count"] >= 6
            if report["foundation_activation_apply_required_before_smoke"]:
                assert activation["activation_apply_ready"] is True
                assert activation["activation_apply_mutation_performed"] is False
                assert report["foundation_activation_post_apply_required_before_smoke"] is False
                assert activation["activation_apply_command"][0:2] == [
                    "scripts/entry_next_edge_control.sh",
                    "foundation-activation-apply",
                ]
                assert "--apply" in activation["activation_apply_command"]
                assert "--vedtak" in activation["activation_apply_command"]
                assert report["next_allowed_command"].startswith(
                    "scripts/entry_next_edge_control.sh foundation-activation-apply"
                )
                assert "--apply --vedtak <id>" in report["next_allowed_command"]
            else:
                assert report["foundation_activation_post_apply_required_before_smoke"] is True
                assert activation["activation_apply_applied"] is True
                assert activation["activation_post_apply_completed"] is False
                assert report["next_allowed_command"].startswith(
                    "scripts/entry_next_edge_control.sh foundation-activation-post-apply"
                )
                assert "--apply --vedtak <id>" in report["next_allowed_command"]
        else:
            assert report["next_allowed_command"].startswith("fix failing readiness gates")
        if feature_gate["decision"] != "PASS":
            assert (
                not version_check["ok"]
                or not feature_checks["feature audit PASS"]["ok"]
                or not feature_checks["feature audit has zero failures"]["ok"]
            )
            if not version_check["ok"]:
                assert "rebuild the foundation seq146 dataset" in version_check["details"]["required_action"]
        else:
            assert any(
                failure["gate"] != "feature_foundation"
                for failure in report["failures"]
            )
    assert report["candidate_training_allowed"] is False
    assert report["promotion_shadow_live_allowed"] is False
    assert "foundation_guardrails" in report["artifacts"]
    assert "worktree_hygiene" in report["artifacts"]
    assert set(report["artifact_fingerprints"]) == set(report["artifacts"])
    for name, row in report["artifact_fingerprints"].items():
        assert row["path"] == report["artifacts"][name]
        assert row["exists"] is True
        assert row["size_bytes"] > 0
        assert isinstance(row["mtime_ns"], int)
        assert isinstance(row["sha256"], str)
        assert len(row["sha256"]) == 64
    assert report["command_proofs"]["foundation_guardrails"]["returncode"] == 0
    assert report["command_proofs"]["smoke_train_dry_run"]["returncode"] == 0
    assert "verify --quiet" in report["command_proofs"]["smoke_train_dry_run"]["stdout"]
    assert "--require-edge" in report["command_proofs"]["smoke_train_dry_run"]["stdout"]
    wrapper_gate = next(gate for gate in report["gates"] if gate["name"] == "control_surface")
    wrapper_checks = {check["name"]: check for check in wrapper_gate["checks"]}
    assert wrapper_checks["smoke train manifest records audit artifact hashes"]["ok"] is True
    assert wrapper_checks["smoke train manifest records feature and specialist contracts"]["ok"] is True
    assert wrapper_checks["smoke train dry-run documents foundation verify preflight"]["ok"] is True
    assert wrapper_checks["post-smoke audit dry-run receives pre-train manifest"]["ok"] is True
    blocked_stderr = (
        report["command_proofs"]["blocked_train"]["stderr"]
        + report["command_proofs"]["blocked_live"]["stderr"]
    )
    assert "entry_next_edge_control.sh worktree-hygiene" in blocked_stderr
    assert "entry_next_edge_control.sh stage-foundation-cleanup --dry-run" in blocked_stderr
    assert any(gate["name"] == "foundation_guardrails" and gate["decision"] == "PASS" for gate in report["gates"])
    guardrail_gate = next(gate for gate in report["gates"] if gate["name"] == "foundation_guardrails")
    guardrail_checks = {check["name"]: check for check in guardrail_gate["checks"]}
    readiness_policy = guardrail_checks["foundation guardrails validate readiness command policy"]
    assert readiness_policy["ok"] is True
    required_policy_checks = set(readiness_policy["details"]["required_policy_checks"])
    assert "readiness_policy_command_set_exact" in required_policy_checks
    assert "readiness_policy_command_schema_complete" in required_policy_checks
    assert "readiness_policy_safe_now_verify" in required_policy_checks
    assert "readiness_policy_adoption_candidate_does_not_activate_without_vedtak" in required_policy_checks
    assert "readiness_policy_safe_now_foundation_activation_plan" in required_policy_checks
    assert "readiness_policy_safe_now_foundation_activation_apply_dry_run" in required_policy_checks
    assert "readiness_policy_safe_now_foundation_activation_post_apply_dry_run" in required_policy_checks
    assert "readiness_policy_safe_now_candidate_readiness_report" in required_policy_checks
    assert "readiness_policy_safe_now_candidate_readiness_smart_report" in required_policy_checks
    assert "readiness_policy_safe_now_replay_readiness_smart_report" in required_policy_checks
    assert "readiness_policy_blocks_foundation_activation_apply" in required_policy_checks
    assert "readiness_policy_blocks_foundation_activation_post_apply" in required_policy_checks
    assert "readiness_policy_blocks_smoke_train" in required_policy_checks
    assert "readiness_policy_blocks_smart_smoke_train" in required_policy_checks
    assert "readiness_policy_blocks_candidate_train" in required_policy_checks
    assert "readiness_policy_blocks_candidate_train_smart" in required_policy_checks
    assert "readiness_policy_smart_smoke_train_declares_ram_edge_smart_contract" in required_policy_checks
    assert "readiness_policy_candidate_train_smart_declares_ram_edge_smart_contract" in required_policy_checks
    assert "readiness_policy_blocks_selective_edge" in required_policy_checks
    assert "readiness_policy_blocks_replay_evidence" in required_policy_checks
    assert "readiness_policy_blocks_iql_distill" in required_policy_checks
    assert "readiness_policy_blocks_iql_replay_evidence" in required_policy_checks
    assert "readiness_policy_blocks_iql_compare" in required_policy_checks
    assert "readiness_policy_blocks_preview_shadow" in required_policy_checks
    assert "readiness_policy_blocks_start_shadow" in required_policy_checks
    assert "readiness_policy_blocks_live" in required_policy_checks
    assert any(gate["name"] == "artifact_provenance" and gate["decision"] == "PASS" for gate in report["gates"])
    assert any(gate["name"] == "execution_hygiene" for gate in report["gates"])
    assert feature_checks["feature audit latest records timestamped artifact paths"]["ok"] is True
    objective_coverage = feature_checks["feature audit covers exact foundation objective features"]
    assert objective_coverage["ok"] is True
    assert set(objective_coverage["details"]["expected_objectives"]) == {
        "hh_hl_lh_ll",
        "bos_choch_age",
        "sweep_reclaim_false_breakout",
        "compression_expansion",
        "impulse_pullback_phase",
        "session_x_structure",
    }
    objective_liveness = feature_checks["feature audit validates exact foundation objective liveness per split"]
    assert objective_liveness["ok"] is True
    assert set(objective_liveness["details"]["expected_objectives"]) == {
        "hh_hl_lh_ll",
        "bos_choch_age",
        "sweep_reclaim_false_breakout",
        "compression_expansion",
        "impulse_pullback_phase",
        "session_x_structure",
    }
    liveness_keys = {
        (row["split"], row["objective"])
        for row in objective_liveness["details"]["objective_liveness"]
    }
    assert {split for split, _ in liveness_keys} == {"train", "val", "test"}
    for objective in objective_liveness["details"]["expected_objectives"]:
        assert all((split, objective) in liveness_keys for split in ("train", "val", "test"))
    source_coverage = feature_checks["feature audit validates foundation source fields per split"]
    assert source_coverage["ok"] is True
    assert source_coverage["details"]["expected_source_field_count"] == len(FOUNDATION_STRUCTURE_SOURCE_FIELDS)
    assert set(source_coverage["details"]["splits"]) == {"train", "val", "test"}
    assert all(
        row["source_missing_count"] == 0
        for row in source_coverage["details"]["splits"].values()
    )
    source_liveness = feature_checks["feature audit validates foundation source-field liveness per split"]
    assert source_liveness["ok"] is True
    source_liveness_keys = {
        (row["split"], row["source_field"])
        for row in source_liveness["details"]["source_field_liveness"]
    }
    assert {split for split, _ in source_liveness_keys} == {"train", "val", "test"}
    for source_field in FOUNDATION_STRUCTURE_SOURCE_FIELDS:
        assert all((split, source_field) in source_liveness_keys for split in ("train", "val", "test"))
    target_gate = next(gate for gate in report["gates"] if gate["name"] == "target_foundation")
    target_checks = {check["name"]: check for check in target_gate["checks"]}
    assert target_checks["target audit latest records timestamped artifact paths"]["ok"] is True
    smoke_dataset_gate = next(gate for gate in report["gates"] if gate["name"] == "smoke_dataset")
    smoke_dataset_checks = {check["name"]: check for check in smoke_dataset_gate["checks"]}
    audit_provenance = smoke_dataset_checks["smoke dataset records active audit artifact provenance"]
    if feature_gate["decision"] == "PASS":
        assert audit_provenance["ok"] is True
    else:
        assert audit_provenance["ok"] is False
    assert set(audit_provenance["details"]["required_audit_artifacts"]) == {
        "feature_audit",
        "target_audit",
        "specialist_audit",
    }
    if audit_provenance["ok"]:
        for name in audit_provenance["details"]["required_audit_artifacts"]:
            manifest_row = audit_provenance["details"]["manifest_artifacts"][name]
            active_row = audit_provenance["details"]["active_artifact_fingerprints"][name]
            assert manifest_row["path"] == active_row["path"]
            assert manifest_row["sha256"] == active_row["sha256"]
            assert len(manifest_row["sha256"]) == 64
    source_manifest_hashes = smoke_dataset_checks["smoke dataset records source split manifest hashes"]
    if report["foundation_contract_ready_for_smoke"]:
        assert source_manifest_hashes["ok"] is True
    else:
        assert source_manifest_hashes["ok"] is False
        assert any(failure["gate"] == "smoke_dataset" for failure in report["failures"])
    assert smoke_dataset_checks["smoke dataset output parquet and manifest hashes match files"]["ok"] is True
    specialist_gate = next(gate for gate in report["gates"] if gate["name"] == "specialist_contract")
    specialist_checks = {check["name"]: check for check in specialist_gate["checks"]}
    assert specialist_checks["specialist audit latest records timestamped artifact paths"]["ok"] is True
    specialist_liveness = specialist_checks["all required specialists have live input features per split"]
    assert specialist_liveness["ok"] is True
    exact_required_specialists = specialist_checks["specialist audit has exact required training specialist set"]
    assert exact_required_specialists["ok"] is True
    expected_specialists = {
        "structure_swing_encoder",
        "smc_liquidity_encoder",
        "trend_ema_encoder",
        "vol_compression_encoder",
        "momentum_flow_encoder",
        "session_regime_encoder",
    }
    assert set(exact_required_specialists["details"]["actual_required_training_specialists"]) == expected_specialists
    architecture_active_heads = specialist_checks["specialist architecture active heads match target training contract"]
    architecture_blocked_heads = specialist_checks["specialist architecture blocked heads match target training contract"]
    assert architecture_active_heads["ok"] is True
    assert architecture_blocked_heads["ok"] is True
    assert "hold_horizon" not in architecture_active_heads["details"]["architecture_active_heads"]
    assert architecture_blocked_heads["details"]["architecture_blocked_heads"] == ["hold_horizon"]
    liveness_keys = {
        (row["split"], row["specialist"])
        for row in specialist_liveness["details"]["specialist_input_liveness"]
    }
    assert {split for split, _ in liveness_keys} == {"train", "val", "test"}
    for specialist in specialist_liveness["details"]["required_specialists"]:
        assert all((split, specialist) in liveness_keys for split in ("train", "val", "test"))
    exact_routing = specialist_checks["all exact foundation objective features are routed to expected specialists"]
    assert exact_routing["ok"] is True
    assert set(exact_routing["details"]["foundation_objective_routing"]) == {
        "hh_hl_lh_ll",
        "bos_choch_age",
        "sweep_reclaim_false_breakout",
        "compression_expansion",
        "impulse_pullback_phase",
        "session_x_structure",
    }
    assert specialist_checks["specialist audit has valid specialist model contract"]["ok"] is True
    assert specialist_checks["specialist model contract has exact trainable specialist set"]["ok"] is True
    assert specialist_checks["specialist model contract owns exact roadmap objectives"]["ok"] is True
    assert specialist_checks["specialist model contract matches registry owned objectives"]["ok"] is True
    assert specialist_checks["specialist model contract support heads are active target heads"]["ok"] is True
    assert specialist_checks["specialist model contract declares signal families for every specialist"]["ok"] is True
    trainer_loader = specialist_checks["trainer specialist-fusion loader accepts current audit contract"]
    assert trainer_loader["ok"] is True
    assert trainer_loader["details"]["meta_signal_field_count"] == 146
    assert trainer_loader["details"]["meta_selected_feature_count"] == 105
    loaded = set(trainer_loader["details"]["loaded_specialists"])
    assert loaded == expected_specialists
    assert set(trainer_loader["details"]["trainable_specialists"]) == expected_specialists
    assert "neutral_bridge_anchor" not in loaded
    assert "unmapped" not in loaded
    assert "price_action_candle_encoder" not in loaded
    assert trainer_loader["details"]["excluded_specialist_groups"]["neutral_bridge_anchor"] == 7
    assert trainer_loader["details"]["excluded_specialist_groups"]["price_action_candle_encoder"] == 3
    exact_loaded = specialist_checks["trainer specialist-fusion loader returns exact trainable specialist set"]
    assert exact_loaded["ok"] is True
    excluded = specialist_checks["trainer specialist-fusion loader excludes non-required specialist groups"]
    assert excluded["ok"] is True
    assert "foundation_cleanup_dirty_count" in execution_gate["checks"][0]["details"]
    assert "review_before_stage_dirty_count" in execution_gate["checks"][0]["details"]
    assert "clean_git_resolution" in execution_gate["checks"][0]["details"]
    assert Path(report["artifacts"]["worktree_hygiene"]).exists()
    hygiene = json.loads(Path(report["artifacts"]["worktree_hygiene"]).read_text(encoding="utf-8"))
    assert Path(hygiene["foundation_stage_paths_txt"]).exists()
    assert Path(hygiene["review_hold_paths_txt"]).exists()
    assert Path(hygiene["foundation_stage_status_tsv"]).exists()
    assert Path(hygiene["review_hold_status_tsv"]).exists()
    assert hygiene["foundation_stage_summary"]["count"] == hygiene["foundation_cleanup_dirty_count"]
    assert hygiene["review_hold_summary"]["count"] == hygiene["review_before_stage_dirty_count"]
    assert hygiene["clean_git_resolution"]["review_hold_count"] == hygiene["review_before_stage_dirty_count"]
    assert Path(hygiene["git_add_dry_run_txt"]).exists()
    assert hygiene["git_add_dry_run"]["cached_unchanged"] is True
    assert hygiene["stage_plan_safe"] is True
    assert hygiene["stage_plan_diagnostics"]["git_add_dry_run_hold_overlap_count"] == 0
    assert hygiene["foundation_cleanup_review_decision"] == "PASS"
    assert hygiene["foundation_cleanup_required_review"]["dirty_missing_from_stage"] == []
    assert hygiene["foundation_cleanup_stage_ready"] is (hygiene["foundation_cleanup_dirty_count"] > 0)
    assert hygiene["foundation_cleanup_stage_command"][0:2] == ["git", "add"]
    post_stage = hygiene["foundation_cleanup_post_stage_verification"]
    assert post_stage["decision"] in {"NOT_STAGED", "PASS_STAGED"}
    if post_stage["decision"] == "NOT_STAGED":
        assert post_stage["cached_count"] == 0
    else:
        assert post_stage["cached_count"] == hygiene["foundation_cleanup_dirty_count"]
        assert post_stage["stage_missing_from_cached_count"] == 0
        assert post_stage["cached_not_in_stage_count"] == 0
        assert post_stage["cached_hold_overlap_count"] == 0
    expected_non_execution_failures = {
        failure["gate"]
        for failure in report["failures"]
        if failure["gate"] != "execution_hygiene"
    }
    if feature_gate["decision"] != "PASS":
        expected_non_execution_failures.update(
            {
                "foundation_state",
                "feature_foundation",
                "smoke_dataset",
                "control_surface",
            }
        )
    if smoke_bundle_gate["decision"] != "PASS":
        expected_non_execution_failures.add("reference_smoke_bundle_audit")
    assert all(
        gate["decision"] == "PASS" or gate["name"] in expected_non_execution_failures
        for gate in report["gates"]
        if gate["name"] != "execution_hygiene"
    )
    assert Path(report["json_path"]).exists()
    assert Path(report["md_path"]).exists()
