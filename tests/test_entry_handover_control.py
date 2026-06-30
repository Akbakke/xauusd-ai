import json
import os
import subprocess
from pathlib import Path


REPO = Path("/home/andre2/src/GX1_ENGINE")
HANDOVER = REPO / "scripts/gx1_handover.sh"
CONTROL = REPO / "scripts/entry_next_edge_control.sh"
FOUNDATION_DOC = REPO / "docs/ENTRY_FOUNDATION_AUDIT_20260628.md"
SPECIALIST_DOC = REPO / "docs/ENTRY_SEQUENTIAL_AI_SPECIALIST_BLUEPRINT_20260628.md"


def test_handover_defaults_to_active_entry_foundation() -> None:
    env = os.environ.copy()
    env.pop("GX1_ALLOW_LEGACY_HANDOVER", None)
    env["GX1_HANDOVER_SKIP_GUARDRAILS"] = "1"
    env["GX1_HANDOVER_SKIP_TRAIN_READINESS"] = "1"

    result = subprocess.run(
        ["bash", str(HANDOVER)],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert "GX1 HANDOVER - active Entry foundation seq146" in result.stdout
    assert "Control surface:" in result.stdout
    assert "scripts/entry_next_edge_control.sh" in result.stdout
    assert "Historical legacy handover is available only with:" in result.stdout
    assert "ACTIVE bundles (one per role" not in result.stdout


def test_control_surface_handover_alias_uses_active_entry_foundation() -> None:
    env = os.environ.copy()
    env.pop("GX1_ALLOW_LEGACY_HANDOVER", None)
    env["GX1_HANDOVER_SKIP_GUARDRAILS"] = "1"
    env["GX1_HANDOVER_SKIP_TRAIN_READINESS"] = "1"

    result = subprocess.run(
        ["bash", str(CONTROL), "handover"],
        cwd=REPO,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert "GX1 HANDOVER - active Entry foundation seq146" in result.stdout
    assert "scripts/entry_next_edge_control.sh" in result.stdout
    assert "ACTIVE bundles (one per role" not in result.stdout


def test_handover_legacy_branch_is_explicitly_env_gated() -> None:
    text = HANDOVER.read_text(encoding="utf-8")

    gate = 'GX1_ALLOW_LEGACY_HANDOVER:-}" != "20260627_ALLOW_LEGACY_HANDOVER"'
    assert gate in text
    assert "GX1 HANDOVER - active Entry foundation seq146" in text
    assert "GX1_ALLOW_LEGACY_HANDOVER=20260627_ALLOW_LEGACY_HANDOVER" in text
    assert "critical gate paths ok:" in text
    assert "adoption-candidate:" in text
    assert "adoption activation without vedtak:" in text
    assert "activation-plan:" in text
    assert "activation-apply dry-run:" in text
    assert "activation-post-apply:" in text
    assert text.index("  exit 0") < text.index("ACTIVE bundles (one per role")


def test_control_surface_routes_verify_to_active_foundation_state() -> None:
    text = CONTROL.read_text(encoding="utf-8")

    assert "scripts/entry_next_edge_control.sh handover" in text
    assert "scripts/entry_next_edge_control.sh readiness-report [--json]" in text
    assert "scripts/entry_next_edge_control.sh foundation-adoption-candidate --dataset-dir <dir>" in text
    assert "scripts/entry_next_edge_control.sh foundation-activation-plan [--adoption-report <json>]" in text
    assert "scripts/entry_next_edge_control.sh foundation-activation-apply --plan-json <json>" in text
    assert "scripts/entry_next_edge_control.sh foundation-activation-post-apply --activation-apply-json <json>" in text
    assert "scripts/entry_next_edge_control.sh smoke-manifest --vedtak <id>" in text
    assert "scripts/entry_next_edge_control.sh smoke-manifest-seq215 --vedtak <id>" in text
    assert "scripts/entry_next_edge_control.sh candidate-readiness-seq215" in text
    assert "scripts/entry_next_edge_control.sh candidate-train-seq215 --vedtak <id>" in text
    assert 'handover)' in text
    assert 'exec "$REPO/scripts/gx1_handover.sh"' in text
    assert 'readiness-report)' in text
    assert 'foundation-adoption-candidate)' in text
    assert 'verify_entry_foundation_adoption_candidate_v1' in text
    assert 'foundation-activation-plan)' in text
    assert 'plan_entry_foundation_activation_v1' in text
    assert 'foundation-activation-apply)' in text
    assert 'apply_entry_foundation_activation_v1' in text
    assert 'foundation-activation-post-apply)' in text
    assert 'run_entry_foundation_activation_post_apply_v1' in text
    assert "report-only: no training, replay, IQL distillation, staging, shadow, or live path was started" in text
    assert 'smoke-manifest)' in text
    assert 'smoke-manifest-seq215)' in text
    assert 'candidate-readiness-seq215)' in text
    assert 'candidate-train-seq215)' in text
    assert 'run_entry_foundation_seq146_smoke_train.sh" --manifest-only' in text
    assert 'run_entry_foundation_seq146_smoke_train.sh" --challenger-seq215 --manifest-only' in text
    assert 'run_entry_foundation_seq146_candidate_train.sh" --challenger-seq215' in text
    assert "verify_entry_foundation_state_v1" in text
    assert "run_entry_foundation_seq146_smoke_train.sh" in text
    assert "run_entry_foundation_iql_distill.sh" in text
    assert "train|retrain|promote|pin|live|start-live|xgb|xgb-train|et|et-train|entry-train|shadow" in text


def test_control_surface_verify_fails_closed_without_traceback() -> None:
    result = subprocess.run(
        ["bash", str(CONTROL), "verify", "--quiet"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert "Traceback" not in result.stderr
    if result.returncode != 0:
        assert "FATAL: foundation verify failed" in result.stderr
        assert "foundation verify error:" in result.stderr


def test_docs_name_handover_as_canonical_fresh_session_entrypoint() -> None:
    foundation_doc = FOUNDATION_DOC.read_text(encoding="utf-8")
    specialist_doc = SPECIALIST_DOC.read_text(encoding="utf-8")

    expected = "scripts/entry_next_edge_control.sh handover"
    report_expected = "scripts/entry_next_edge_control.sh readiness-report"
    manifest_expected = "scripts/entry_next_edge_control.sh smoke-manifest --vedtak <id>"
    bundle_contract_expected = "bundle_specialist_model_contract"
    bundle_provenance_expected = "bundle-specialist-model"
    assert expected in foundation_doc
    assert expected in specialist_doc
    assert report_expected in foundation_doc
    assert report_expected in specialist_doc
    assert manifest_expected in foundation_doc
    assert manifest_expected in specialist_doc
    assert bundle_contract_expected in foundation_doc
    assert bundle_contract_expected in specialist_doc
    assert bundle_provenance_expected in foundation_doc
    assert bundle_provenance_expected in specialist_doc
    assert "active Entry foundation seq146" in foundation_doc
    assert "active Entry foundation seq146" in specialist_doc
    assert "active seq146 foundation is activated and post-apply refreshed" in foundation_doc
    assert "Entry evidence gates" in foundation_doc
    assert "foundation-activation-apply --dry-run" in foundation_doc
    assert "foundation-activation-apply --apply --vedtak <id>" in foundation_doc
    assert "foundation-activation-post-apply --apply --vedtak <id>" in foundation_doc
    assert "foundation_activation_required_before_smoke=true" in specialist_doc
    assert "foundation-activation-apply --apply --vedtak <id>" in specialist_doc
    assert "approve trainer start" in specialist_doc
    active_objective_rule = "Active Objective Rule: all Entry and Exit features with multi-timeframe context"
    assert active_objective_rule in foundation_doc
    assert active_objective_rule in specialist_doc
    assert "replay-proven" in foundation_doc
    assert "replay-proven" in specialist_doc
    assert "isolated feature families" in foundation_doc
    assert "isolated feature families" in specialist_doc
    assert "challenger_seq215` must not reuse the six-gate Exit alignment" in foundation_doc
    assert "all eight specialist-gate weights before any Exit train/replay/IQL step" in foundation_doc
    assert "chart_geometry_encoder` and `price_action_candle_encoder` state" in specialist_doc
    assert "eight specialist-gate weights before any Exit train/replay/IQL step" in specialist_doc


def test_control_surface_readiness_report_is_fail_open_status_only() -> None:
    result = subprocess.run(
        ["bash", str(CONTROL), "readiness-report"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 0
    assert "train-readiness:" in result.stdout
    if "train-readiness: SKIPPED_BY_GUARDRAIL_TEST" not in result.stdout:
        assert "foundation activation required:" in result.stdout
        assert "foundation activation apply required:" in result.stdout
        assert "foundation post-apply required:" in result.stdout
    if "train-readiness: NOT_READY" in result.stdout:
        assert "optional proof: scripts/entry_next_edge_control.sh smoke-manifest --vedtak <id>" not in result.stdout
    else:
        assert "optional proof: scripts/entry_next_edge_control.sh smoke-manifest --vedtak <id>" in result.stdout
    assert "worktree-hygiene:" in result.stdout
    assert "dirty/stage/hold:" in result.stdout
    assert "stage-ready/safe:" in result.stdout
    assert "critical gate paths ok:" in result.stdout
    assert "post-stage:" in result.stdout
    assert "stage paths:" in result.stdout
    assert "hold paths:" in result.stdout
    assert "canonical stage dry-run: scripts/entry_next_edge_control.sh stage-foundation-cleanup --dry-run" in result.stdout
    assert "canonical stage apply: scripts/entry_next_edge_control.sh stage-foundation-cleanup --apply --vedtak <id>" in result.stdout
    assert "raw stage command: git add --pathspec-from-file=" in result.stdout
    assert "candidate-readiness:" in result.stdout
    assert "replay-readiness:" in result.stdout
    assert "iql-distillation-contract:" in result.stdout
    assert "iql-student-trade-log:" in result.stdout
    assert "iql-replay-evidence:" in result.stdout
    assert "iql-replay-comparison:" in result.stdout
    assert "foundation-adoption-candidate:" in result.stdout
    assert "foundation-activation-plan:" in result.stdout
    assert "foundation-activation-apply:" in result.stdout
    assert "foundation-activation-post-apply:" in result.stdout
    assert "allowed now:" in result.stdout
    assert "scripts/entry_next_edge_control.sh verify --quiet" in result.stdout
    assert "scripts/entry_next_edge_control.sh candidate-readiness --quiet --no-fail-on-not-ready" in result.stdout
    assert "scripts/entry_next_edge_control.sh candidate-readiness-seq215 --quiet --no-fail-on-not-ready" in result.stdout
    assert "scripts/entry_next_edge_control.sh replay-readiness --quiet --no-fail-on-not-ready" in result.stdout
    assert "scripts/entry_next_edge_control.sh stage-foundation-cleanup --dry-run" in result.stdout
    if "train-readiness: NOT_READY" in result.stdout:
        assert "optional proof commands:" not in result.stdout
    else:
        assert "optional proof commands:" in result.stdout
        assert "scripts/entry_next_edge_control.sh smoke-manifest --vedtak <id>  # proof only, no trainer start" in result.stdout
    assert "blocked now:" in result.stdout
    assert "scripts/entry_next_edge_control.sh smoke-train --vedtak <id> --require-edge-audit  # needs clean git + explicit vedtak" in result.stdout
    assert "candidate-train, replay, IQL distillation, promotion, shadow and live remain blocked" in result.stdout
    assert "report-only: no training, replay, IQL distillation, staging, shadow, or live path was started" in result.stdout


def test_control_surface_readiness_report_json_is_machine_readable() -> None:
    result = subprocess.run(
        ["bash", str(CONTROL), "readiness-report", "--json"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 0
    assert result.stderr == ""
    assert payload["schema_version"] == "entry_next_edge_readiness_report_v1"
    assert payload["report_only"] is True
    foundation_ready = bool(payload["status_summary"]["foundation_contract_ready_for_smoke"])
    adoption_ready = bool(payload["status_summary"].get("foundation_adoption_candidate_ready"))
    activation_required = bool(payload["status_summary"].get("foundation_activation_required_before_smoke"))
    activation_apply_required = bool(payload["status_summary"].get("foundation_activation_apply_required_before_smoke"))
    activation_post_apply_required = bool(
        payload["status_summary"].get("foundation_activation_post_apply_required_before_smoke")
    )
    activation_plan_ready = bool(payload["status_summary"].get("foundation_activation_plan_ready"))
    activation_apply_ready = bool(payload["status_summary"].get("foundation_activation_apply_ready"))
    activation_apply_applied = (
        payload["reports"]["foundation-activation-apply"]["decision"] == "APPLIED_ALIAS_SWITCH"
        and bool(payload["status_summary"].get("foundation_activation_apply_mutation_performed"))
    )
    activation_post_apply_completed = bool(
        payload["status_summary"].get("foundation_activation_post_apply_completed")
    )
    assert payload["status_summary"]["real_smoke_train_allowed"] is False
    assert payload["status_summary"]["smoke_manifest_proof_allowed"] is foundation_ready
    assert payload["status_summary"]["activation_allowed_without_vedtak"] is False
    if adoption_ready:
        assert payload["status_summary"]["foundation_adoption_candidate_report"]
        assert payload["status_summary"]["foundation_adoption_candidate_dataset_dir"]
        assert payload["status_summary"]["foundation_adoption_candidate_smoke_dataset_dir"]
    if payload["status_summary"].get("foundation_activation_plan_ready"):
        assert payload["status_summary"]["foundation_activation_plan_report"]
        assert payload["status_summary"]["foundation_activation_plan_strategy"] == (
            "canonical_active_alias_then_canonical_audit_refresh"
        )
    if activation_apply_ready:
        assert payload["status_summary"]["foundation_activation_apply_report"]
        assert payload["status_summary"]["foundation_activation_apply_mutation_performed"] is False
        assert payload["status_summary"]["foundation_activation_apply_post_apply_command_count"] >= 6
        assert payload["status_summary"]["foundation_activation_post_apply_report"]
        assert payload["status_summary"]["foundation_activation_post_apply_waiting_for_activation"] is True
        assert payload["status_summary"]["foundation_activation_post_apply_ready"] is False
        assert payload["status_summary"]["foundation_activation_post_apply_completed"] is False
        assert payload["status_summary"]["foundation_activation_post_apply_mutations_performed"] is False
        assert payload["status_summary"]["foundation_activation_post_apply_dry_run_command"][0:2] == [
            "scripts/entry_next_edge_control.sh",
            "foundation-activation-post-apply",
        ]
        assert payload["status_summary"]["foundation_activation_post_apply_command"][0:2] == [
            "scripts/entry_next_edge_control.sh",
            "foundation-activation-post-apply",
        ]
    elif activation_apply_applied:
        assert payload["status_summary"]["foundation_activation_apply_report"]
        assert payload["status_summary"]["foundation_activation_apply_mutation_performed"] is True
        assert payload["status_summary"]["foundation_activation_post_apply_waiting_for_activation"] is False
        assert payload["status_summary"]["foundation_activation_post_apply_dry_run_command"][0:2] == [
            "scripts/entry_next_edge_control.sh",
            "foundation-activation-post-apply",
        ]
        assert payload["status_summary"]["foundation_activation_post_apply_command"][0:2] == [
            "scripts/entry_next_edge_control.sh",
            "foundation-activation-post-apply",
        ]
        if activation_post_apply_completed:
            assert payload["status_summary"]["foundation_activation_post_apply_mutations_performed"] is True
            assert activation_post_apply_required is False
        else:
            assert activation_post_apply_required is True
    if activation_required:
        assert foundation_ready is False
        assert activation_apply_required is True
        assert activation_post_apply_required is False
        assert payload["status_summary"]["foundation_activation_next_command"].startswith(
            "scripts/entry_next_edge_control.sh foundation-activation-apply"
        )
        assert "--apply --vedtak <id>" in payload["status_summary"]["foundation_activation_next_command"]
        assert payload["status_summary"]["foundation_activation_apply_command"][0:2] == [
            "scripts/entry_next_edge_control.sh",
            "foundation-activation-apply",
        ]
        assert "--apply" in payload["status_summary"]["foundation_activation_apply_command"]
        assert "--vedtak" in payload["status_summary"]["foundation_activation_apply_command"]
    if payload["worktree_hygiene"]["dirty_count"] == 0:
        assert payload["status_summary"]["foundation_cleanup_stage_ready"] is False
    else:
        assert payload["status_summary"]["foundation_cleanup_stage_ready"] is True
    assert payload["status_summary"]["stage_plan_safe"] is True
    assert payload["status_summary"]["clean_git_resolution_decision"]
    assert payload["status_summary"]["candidate_training_allowed"] is False
    assert payload["status_summary"]["candidate_training_foundation_seq146_allowed"] == (
        payload["status_summary"]["candidate_training_allowed"]
    )
    assert isinstance(payload["status_summary"]["candidate_training_seq215_allowed"], bool)
    assert payload["status_summary"]["candidate_readiness_seq215_decision"] in {
        "READY_FOR_CANDIDATE_TRAINING_VEDTAK",
        "NOT_READY_FOR_CANDIDATE_TRAINING",
        None,
    }
    if not payload["status_summary"]["candidate_training_seq215_allowed"]:
        assert "smoke-train-seq215" in str(payload["status_summary"]["candidate_readiness_seq215_next"])
        assert (
            "seq215 candidate training requires real seq215 smoke bundle edge audit"
            in payload["status_summary"]["current_blockers"]
        )
    assert isinstance(payload["status_summary"]["smoke_manifest_seq215_proof_allowed"], bool)
    assert isinstance(payload["status_summary"]["real_smoke_train_seq215_allowed"], bool)
    assert isinstance(payload["status_summary"]["iql_distillation_allowed"], bool)
    assert isinstance(payload["status_summary"]["iql_replay_evidence_ready"], bool)
    assert isinstance(payload["status_summary"]["iql_replay_comparison_ready"], bool)
    assert isinstance(payload["status_summary"]["promotion_review_allowed"], bool)
    assert payload["status_summary"]["promotion_shadow_live_allowed"] is False
    if foundation_ready:
        assert "clean git worktree and explicit smoke-train vedtak" in payload["status_summary"]["current_blockers"]
    else:
        assert "foundation contract is not ready for smoke" in payload["status_summary"]["current_blockers"]
    if adoption_ready and not foundation_ready:
        assert "explicit vedtak to switch active foundation dataset/audit paths" in payload["status_summary"]["current_blockers"]
    if not payload["status_summary"]["iql_replay_evidence_ready"]:
        assert "IQL replay evidence requires distillation contract and IQL-student replay trade log" in payload["status_summary"]["current_blockers"]
    if not payload["status_summary"]["iql_replay_comparison_ready"]:
        assert "promotion review requires candidate-vs-IQL replay comparison PASS" in payload["status_summary"]["current_blockers"]
    assert payload["side_effects_started"] == {
        "staging": False,
        "training": False,
        "replay": False,
        "iql_distillation": False,
        "shadow": False,
        "live": False,
    }
    assert payload["reports"]["train-readiness"]["decision"] in {
        "READY_FOR_VEDTAK_SMOKE_TRAIN_AFTER_GIT_CLEAN",
        "NOT_READY",
    }
    assert payload["reports"]["worktree-hygiene"]["decision"] == "BLOCKED_BY_DIRTY_GIT"
    assert "iql-student-trade-log" in payload["reports"]
    assert "iql-replay-evidence" in payload["reports"]
    assert "iql-replay-comparison" in payload["reports"]
    assert isinstance(payload["reports"]["iql-replay-evidence"]["exists"], bool)
    assert payload["reports"]["iql-replay-comparison"]["exists"] is True
    assert payload["reports"]["iql-replay-comparison"]["decision"] in {
        "NOT_READY_FOR_PROMOTION_REVIEW",
        "READY_FOR_PROMOTION_REVIEW_VEDTAK",
    }
    if adoption_ready:
        assert payload["reports"]["foundation-adoption-candidate"]["decision"] == "PASS"
    if activation_apply_ready:
        assert payload["reports"]["foundation-activation-apply"]["decision"] == "READY_FOR_VEDTAK_APPLY"
        assert payload["reports"]["foundation-activation-post-apply"]["decision"] == "WAITING_FOR_ACTIVATION_APPLY"
    elif activation_apply_applied:
        assert payload["reports"]["foundation-activation-apply"]["decision"] == "APPLIED_ALIAS_SWITCH"
        if activation_post_apply_completed:
            assert payload["reports"]["foundation-activation-post-apply"]["decision"] == "POST_APPLY_REFRESH_COMPLETED"
        else:
            assert payload["reports"]["foundation-activation-post-apply"]["decision"] in {
                "READY_FOR_POST_APPLY_REFRESH",
                "WAITING_FOR_ACTIVATION_APPLY",
            }
    assert payload["commands"]["readiness_report_json"]["argv"] == [
        "scripts/entry_next_edge_control.sh",
        "readiness-report",
        "--json",
    ]
    assert payload["commands"]["readiness_report_json"]["allowed"] is True
    assert payload["commands"]["readiness_report_json"]["execution_allowed_now"] is True
    assert payload["commands"]["readiness_report_json"]["allowed_after_explicit_vedtak"] is True
    assert payload["commands"]["readiness_report_json"]["not_executable_now_reason"] is None
    assert payload["commands"]["readiness_report_json"]["requires_vedtak"] is False
    assert payload["commands"]["readiness_report_json"]["mutates_git_index"] is False
    assert payload["commands"]["readiness_report_json"]["starts_trainer"] is False
    assert payload["commands"]["verify"]["execution_allowed_now"] is True
    assert payload["commands"]["verify"]["mode"] == "audit"
    assert payload["commands"]["verify"]["starts_trainer"] is False
    assert payload["commands"]["foundation_guardrails"]["execution_allowed_now"] is True
    assert payload["commands"]["foundation_activation_plan"]["execution_allowed_now"] is True
    assert payload["commands"]["foundation_activation_plan"]["mode"] == "report"
    assert payload["commands"]["foundation_activation_plan"]["requires_vedtak"] is False
    assert payload["commands"]["foundation_activation_plan"]["mutates_git_index"] is False
    assert payload["commands"]["foundation_activation_plan"]["starts_trainer"] is False
    assert payload["commands"]["foundation_activation_apply_dry_run"]["execution_allowed_now"] is True
    assert payload["commands"]["foundation_activation_apply_dry_run"]["mode"] == "dry_run"
    assert payload["commands"]["foundation_activation_apply_dry_run"]["requires_vedtak"] is False
    assert payload["commands"]["foundation_activation_apply_dry_run"]["mutates_foundation_paths"] is False
    assert payload["commands"]["foundation_activation_apply"]["execution_allowed_now"] is False
    expected_activation_apply_after_vedtak = activation_plan_ready and not activation_apply_applied
    assert (
        payload["commands"]["foundation_activation_apply"]["allowed_after_explicit_vedtak"]
        is expected_activation_apply_after_vedtak
    )
    assert payload["commands"]["foundation_activation_apply"]["requires_vedtak"] is True
    assert payload["commands"]["foundation_activation_apply"]["mutates_foundation_paths"] is True
    if activation_required:
        assert payload["commands"]["foundation_activation_apply"]["argv"] == payload["status_summary"]["foundation_activation_apply_command"]
    assert payload["commands"]["foundation_activation_post_apply_dry_run"]["execution_allowed_now"] is True
    assert payload["commands"]["foundation_activation_post_apply_dry_run"]["mode"] == "dry_run"
    assert payload["commands"]["foundation_activation_post_apply_dry_run"]["requires_vedtak"] is False
    assert payload["commands"]["foundation_activation_post_apply_dry_run"]["mutates_foundation_paths"] is False
    assert payload["commands"]["foundation_activation_post_apply_dry_run"]["mutates_foundation_audits"] is False
    assert payload["commands"]["foundation_activation_post_apply"]["execution_allowed_now"] is False
    expected_post_apply_after_vedtak = (
        activation_apply_applied
        and not activation_post_apply_completed
    ) or bool(payload["status_summary"].get("foundation_activation_post_apply_ready"))
    assert (
        payload["commands"]["foundation_activation_post_apply"]["allowed_after_explicit_vedtak"]
        is expected_post_apply_after_vedtak
    )
    assert payload["commands"]["foundation_activation_post_apply"]["requires_vedtak"] is True
    assert payload["commands"]["foundation_activation_post_apply"]["mutates_git_index"] is False
    assert payload["commands"]["foundation_activation_post_apply"]["mutates_foundation_paths"] is False
    assert payload["commands"]["foundation_activation_post_apply"]["mutates_foundation_audits"] is True
    assert payload["commands"]["foundation_activation_post_apply"]["materializes_smoke_dataset"] is True
    assert payload["commands"]["foundation_activation_post_apply"]["starts_trainer"] is False
    assert payload["commands"]["foundation_activation_post_apply"]["starts_replay"] is False
    assert payload["commands"]["foundation_activation_post_apply"]["starts_iql_distillation"] is False
    if activation_post_apply_completed:
        assert "already completed" in payload["commands"]["foundation_activation_post_apply"]["not_executable_now_reason"]
    else:
        assert "APPLIED_ALIAS_SWITCH" in payload["commands"]["foundation_activation_post_apply"]["not_executable_now_reason"]
    assert payload["commands"]["train_readiness_report"]["argv"] == [
        "scripts/entry_next_edge_control.sh",
        "train-readiness",
        "--quiet",
        "--no-fail-on-not-ready",
    ]
    assert payload["commands"]["train_readiness_report"]["execution_allowed_now"] is True
    assert payload["commands"]["candidate_readiness_report"]["execution_allowed_now"] is True
    assert payload["commands"]["replay_readiness_report"]["execution_allowed_now"] is True
    assert payload["commands"]["stage_foundation_cleanup_dry_run"]["allowed"] is True
    assert payload["commands"]["stage_foundation_cleanup_dry_run"]["execution_allowed_now"] is True
    assert payload["commands"]["stage_foundation_cleanup_dry_run"]["allowed_after_explicit_vedtak"] is True
    assert payload["commands"]["stage_foundation_cleanup_dry_run"]["mode"] == "dry_run"
    assert payload["commands"]["stage_foundation_cleanup_dry_run"]["mutates_git_index"] is False
    assert payload["commands"]["stage_foundation_cleanup_apply"]["argv"] == [
        "scripts/entry_next_edge_control.sh",
        "stage-foundation-cleanup",
        "--apply",
        "--vedtak",
        "<id>",
    ]
    assert payload["commands"]["stage_foundation_cleanup_apply"]["allowed"] is True
    assert payload["commands"]["stage_foundation_cleanup_apply"]["execution_allowed_now"] is False
    assert payload["commands"]["stage_foundation_cleanup_apply"]["allowed_after_explicit_vedtak"] is True
    assert payload["commands"]["stage_foundation_cleanup_apply"]["not_executable_now_reason"] == (
        "requires explicit staging vedtak and mutates git index"
    )
    assert payload["commands"]["stage_foundation_cleanup_apply"]["requires_vedtak"] is True
    assert payload["commands"]["stage_foundation_cleanup_apply"]["mutates_git_index"] is True
    assert payload["commands"]["stage_foundation_cleanup_apply"]["starts_trainer"] is False
    assert payload["commands"]["smoke_manifest"]["argv"] == [
        "scripts/entry_next_edge_control.sh",
        "smoke-manifest",
        "--vedtak",
        "<id>",
    ]
    assert payload["commands"]["smoke_manifest"]["allowed"] is foundation_ready
    assert payload["commands"]["smoke_manifest"]["execution_allowed_now"] is False
    assert payload["commands"]["smoke_manifest"]["allowed_after_explicit_vedtak"] is foundation_ready
    assert payload["commands"]["smoke_manifest"]["not_executable_now_reason"] == (
        "requires explicit smoke-manifest vedtak; proof-only no trainer start"
    )
    assert payload["commands"]["smoke_manifest"]["mode"] == "proof_only"
    assert payload["commands"]["smoke_manifest"]["requires_vedtak"] is True
    assert payload["commands"]["smoke_manifest"]["requires_clean_git"] is False
    assert payload["commands"]["smoke_manifest"]["starts_trainer"] is False
    assert payload["commands"]["smoke_manifest_seq215"]["argv"] == [
        "scripts/entry_next_edge_control.sh",
        "smoke-manifest-seq215",
        "--vedtak",
        "<id>",
    ]
    assert payload["commands"]["smoke_manifest_seq215"]["allowed"] is foundation_ready
    assert payload["commands"]["smoke_manifest_seq215"]["execution_allowed_now"] is False
    assert payload["commands"]["smoke_manifest_seq215"]["allowed_after_explicit_vedtak"] is foundation_ready
    assert payload["commands"]["smoke_manifest_seq215"]["not_executable_now_reason"] == (
        "requires explicit SEQ215 smoke-manifest vedtak; proof-only no trainer start"
    )
    assert payload["commands"]["smoke_manifest_seq215"]["mode"] == "proof_only"
    assert payload["commands"]["smoke_manifest_seq215"]["requires_vedtak"] is True
    assert payload["commands"]["smoke_manifest_seq215"]["requires_clean_git"] is False
    assert payload["commands"]["smoke_manifest_seq215"]["starts_trainer"] is False
    assert payload["commands"]["smoke_train"]["allowed"] is False
    assert payload["commands"]["smoke_train"]["execution_allowed_now"] is False
    assert payload["commands"]["smoke_train"]["allowed_after_explicit_vedtak"] is False
    if foundation_ready:
        assert payload["commands"]["smoke_train"]["not_executable_now_reason"] == (
            "requires clean git worktree and explicit smoke-train vedtak"
        )
    else:
        assert payload["commands"]["smoke_train"]["not_executable_now_reason"] == (
            "foundation contract is not ready for smoke"
        )
    assert payload["commands"]["smoke_train"]["mode"] == "train"
    assert payload["commands"]["smoke_train"]["requires_vedtak"] is True
    assert payload["commands"]["smoke_train"]["requires_clean_git"] is True
    assert payload["commands"]["smoke_train"]["starts_trainer"] is True
    assert payload["commands"]["smoke_train"]["touches_shadow_or_live"] is False
    assert payload["commands"]["smoke_train_seq215"]["argv"] == [
        "scripts/entry_next_edge_control.sh",
        "smoke-train-seq215",
        "--vedtak",
        "<id>",
        "--require-edge-audit",
    ]
    assert payload["commands"]["smoke_train_seq215"]["allowed"] is False
    assert payload["commands"]["smoke_train_seq215"]["execution_allowed_now"] is False
    assert payload["commands"]["smoke_train_seq215"]["allowed_after_explicit_vedtak"] is False
    if foundation_ready:
        assert payload["commands"]["smoke_train_seq215"]["not_executable_now_reason"] == (
            "requires clean git worktree and explicit SEQ215 smoke-train vedtak"
        )
    else:
        assert payload["commands"]["smoke_train_seq215"]["not_executable_now_reason"] == (
            "foundation contract is not ready for smoke"
        )
    assert payload["commands"]["smoke_train_seq215"]["mode"] == "train"
    assert payload["commands"]["smoke_train_seq215"]["requires_vedtak"] is True
    assert payload["commands"]["smoke_train_seq215"]["requires_clean_git"] is True
    assert payload["commands"]["smoke_train_seq215"]["starts_trainer"] is True
    assert payload["commands"]["smoke_train_seq215"]["touches_shadow_or_live"] is False
    assert payload["commands"]["candidate_train"]["argv"] == [
        "scripts/entry_next_edge_control.sh",
        "candidate-train",
        "--vedtak",
        "<id>",
    ]
    assert payload["commands"]["candidate_train"]["allowed"] is False
    assert payload["commands"]["candidate_train"]["execution_allowed_now"] is False
    assert payload["commands"]["candidate_train"]["allowed_after_explicit_vedtak"] is False
    assert payload["commands"]["candidate_train"]["requires_vedtak"] is True
    assert payload["commands"]["candidate_train"]["requires_clean_git"] is True
    assert payload["commands"]["candidate_train"]["starts_trainer"] is True
    assert "real smoke bundle edge audit" in payload["commands"]["candidate_train"]["not_executable_now_reason"]
    assert payload["commands"]["selective_edge"]["execution_allowed_now"] is False
    assert payload["commands"]["selective_edge"]["starts_trainer"] is False
    assert payload["commands"]["replay_evidence"]["execution_allowed_now"] is False
    assert payload["commands"]["replay_evidence"]["starts_replay"] is False
    assert payload["commands"]["iql_distill"]["execution_allowed_now"] is False
    assert payload["commands"]["iql_distill"]["requires_vedtak"] is True
    assert payload["commands"]["iql_distill"]["starts_iql_distillation"] is True
    assert "replay-readiness PASS" in payload["commands"]["iql_distill"]["not_executable_now_reason"]
    assert payload["commands"]["iql_replay_evidence"]["execution_allowed_now"] is False
    assert payload["commands"]["iql_compare"]["execution_allowed_now"] is False
    assert payload["commands"]["preview_shadow"]["touches_shadow_or_live"] is True
    assert payload["commands"]["start_shadow"]["touches_shadow_or_live"] is True
    assert payload["commands"]["live"]["touches_shadow_or_live"] is True
    assert payload["commands"]["live"]["execution_allowed_now"] is False
    assert "scripts/entry_next_edge_control.sh readiness-report" in payload["allowed_now"]
    assert "scripts/entry_next_edge_control.sh readiness-report --json" in payload["allowed_now"]
    assert "scripts/entry_next_edge_control.sh verify --quiet" in payload["allowed_now"]
    assert "scripts/entry_next_edge_control.sh foundation-activation-plan" in payload["allowed_now"]
    assert any("foundation-activation-apply" in item and "--dry-run" in item for item in payload["allowed_now"])
    assert any("foundation-activation-post-apply" in item and "--dry-run" in item for item in payload["allowed_now"])
    assert "scripts/entry_next_edge_control.sh candidate-readiness --quiet --no-fail-on-not-ready" in payload["allowed_now"]
    assert "scripts/entry_next_edge_control.sh replay-readiness --quiet --no-fail-on-not-ready" in payload["allowed_now"]
    assert "scripts/entry_next_edge_control.sh stage-foundation-cleanup --dry-run" in payload["allowed_now"]
    assert not any("smoke-manifest" in item for item in payload["allowed_now"])
    if foundation_ready:
        assert payload["optional_proof_commands"] == [
            "scripts/entry_next_edge_control.sh smoke-manifest --vedtak <id>  # proof only, no trainer start",
            "scripts/entry_next_edge_control.sh smoke-manifest-seq215 --vedtak <id>  # proof only, no trainer start",
        ]
    else:
        assert payload["optional_proof_commands"] == []
    if payload["worktree_hygiene"]["dirty_count"] == 0:
        assert payload["worktree_hygiene"]["foundation_cleanup_stage_ready"] is False
    else:
        assert payload["worktree_hygiene"]["foundation_cleanup_stage_ready"] is True
    assert payload["worktree_hygiene"]["stage_plan_safe"] is True
    assert payload["worktree_hygiene"]["critical_gate_ok_count"] == payload["worktree_hygiene"]["critical_gate_path_count"]
    assert payload["worktree_hygiene"]["critical_gate_path_count"] >= 1
    assert payload["worktree_hygiene"]["critical_gate_missing_from_repo"] == []
    assert payload["worktree_hygiene"]["critical_gate_dirty_missing_from_stage"] == []
    assert payload["worktree_hygiene"]["canonical_stage_apply"] == (
        "scripts/entry_next_edge_control.sh stage-foundation-cleanup --apply --vedtak <id>"
    )
    assert (
        payload["worktree_hygiene"]["clean_git_resolution"]["review_hold_count"]
        == payload["worktree_hygiene"]["review_before_stage_dirty_count"]
    )
    assert payload["worktree_hygiene"]["raw_stage_command"].startswith("git add --pathspec-from-file=")
    assert any("smoke-train" in item for item in payload["blocked_now"])


def test_control_surface_readiness_report_rejects_unknown_args() -> None:
    result = subprocess.run(
        ["bash", str(CONTROL), "readiness-report", "--bad-arg"],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "FATAL: unknown readiness-report arg: --bad-arg" in result.stderr
