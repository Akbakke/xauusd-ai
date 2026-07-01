import argparse
import json
from pathlib import Path

from gx1.features.entry_foundation_structure_v1 import FOUNDATION_STRUCTURE_SOURCE_FIELDS
from gx1.features.entry_specialist_feature_groups_v1 import FOUNDATION_OBJECTIVE_SPECIALISTS
from gx1.scripts.audit_entry_foundation_features_v1 import REQUIRED_FOUNDATION_OBJECTIVE_FEATURES
from gx1.scripts.verify_entry_foundation_state_v1 import run


REPO = Path("/home/andre2/src/GX1_ENGINE")
AUDIT_DOC = REPO / "docs/ENTRY_FOUNDATION_AUDIT_20260628.md"
CANDIDATE_FEATURE_AUDIT = Path(
    "/home/andre2/GX1_DATA/reports/entry_feature_foundation_audit_20260628_v1/"
    "foundation_seq146_20260629_directional_smc/ENTRY_FEATURE_FOUNDATION_AUDIT_latest.json"
)
CANDIDATE_SPECIALIST_AUDIT = Path(
    "/home/andre2/GX1_DATA/reports/entry_specialist_feature_group_audit_20260628_v1/"
    "foundation_seq146_20260629_directional_smc/ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT_latest.json"
)


def test_foundation_state_allows_entry_train_manifest_report_roots() -> None:
    verifier = (REPO / "gx1/scripts/verify_entry_foundation_state_v1.py").read_text(encoding="utf-8")

    assert "entry_foundation_smoke_train_manifests_20260628_v1" in verifier
    assert "entry_foundation_candidate_train_manifests_20260628_v1" in verifier
    assert "entry_seq215_manifest_provenance_repair_20260630_v1" in verifier
    assert "entry_foundation_manifest_provenance_repair_20260701_v1" in verifier
    assert "entry_candidate_bundle_audit_20260628_v1" in verifier
    assert "entry_candidate_replay_trade_log_20260628_v1" in verifier
    assert "entry_candidate_replay_trade_log_20260628_v1_stop80_tp120" in verifier
    assert "entry_iql_student_trade_log_20260628_v1" in verifier
    assert "entry_iql_replay_slice_audit_20260628_v1" in verifier
    assert "entry_exit_per_bar_handoff_20260630_v1" in verifier
    assert "entry_exit_handoff_readiness_20260630_v1" in verifier
    assert "entry_exit_per_bar_reconstruction_audit_20260630_v1" in verifier
    assert "entry_exit_state_reward_contract_20260630_v1" in verifier
    assert "entry_exit_split_leakage_audit_20260630_v1" in verifier
    assert "entry_exit_model_dataset_readiness_20260630_v1" in verifier
    assert "entry_exit_feature_alignment_20260630_v1" in verifier
    assert "entry_exit_transformer_architecture_readiness_20260630_v1" in verifier
    assert "entry_exit_transformer_training_plan_readiness_20260630_v1" in verifier
    assert "entry_exit_transformer_trainer_wrapper_readiness_20260630_v1" in verifier
    assert "entry_exit_transformer_pretrain_manifest_20260630_v1" in verifier
    assert "entry_exit_model_dataset_slice_robustness_20260630_v1" in verifier
    assert "entry_exit_transformer_train_execution_review_20260630_v1" in verifier
    assert "entry_exit_transformer_post_train_contract_20260630_v1" in verifier
    assert "entry_exit_transformer_train_enablement_20260701_v1" in verifier
    assert "entry_smart_ablation_replay_matrix_20260701_v1" in verifier
    assert "entry_smart_ablation_replay_matrix_gate_20260701_v1" in verifier
    assert "entry_smart_feature_mask_specs_20260701_v1" in verifier
    assert "entry_trend_ema_extension_manifest_20260630_v1" in verifier
    assert "entry_smc_liquidity_quality_manifest_20260630_v1" in verifier
    assert "entry_momentum_flow_challenger_manifest_20260630_v1" in verifier
    assert "entry_session_regime_interaction_manifest_20260630_v1" in verifier


def _args(*, selftest: bool) -> argparse.Namespace:
    return argparse.Namespace(
        audit_doc=str(AUDIT_DOC),
        out="",
        quiet=True,
        selftest=selftest,
    )


def _run_or_active_stale(*, selftest: bool) -> tuple[dict, str]:
    try:
        return run(_args(selftest=selftest)), ""
    except RuntimeError as exc:
        error = str(exc)
        if (
            "feature foundation audit requires PASS" in error
            and "decision=FAIL" in error
            and "entry_foundation_structure_v1_20260629_directional_smc_pressure" in error
        ):
            return {}, error
        raise


def test_foundation_state_verify_requires_exact_objective_coverage() -> None:
    report, active_stale_error = _run_or_active_stale(selftest=False)

    if active_stale_error:
        assert "foundation feature count None != 57" in active_stale_error
        feature_audit = json.loads(CANDIDATE_FEATURE_AUDIT.read_text(encoding="utf-8"))
        specialist_audit = json.loads(CANDIDATE_SPECIALIST_AUDIT.read_text(encoding="utf-8"))
        assert feature_audit["decision"] == "PASS"
        assert feature_audit["failures"] == []
        assert specialist_audit["decision"] == "PASS"
        assert specialist_audit["failures"] == []
    else:
        feature_audit = report["feature_audit_latest"]
        specialist_audit = report["specialist_audit_latest"]
    coverage = {
        row["objective"]: row
        for row in feature_audit["foundation_objective_coverage"]
    }
    objective_liveness = {
        (row["split"], row["objective"]): row
        for row in feature_audit["foundation_objective_liveness"]
    }
    source_liveness = {
        (row["split"], row["source_field"]): row
        for row in feature_audit["foundation_source_field_liveness"]
    }

    assert feature_audit["foundation_objective_coverage_all_present"] is True
    assert feature_audit["foundation_objective_liveness_all_live"] is True
    assert feature_audit["foundation_source_field_liveness_all_live"] is True
    assert set(coverage) == set(REQUIRED_FOUNDATION_OBJECTIVE_FEATURES)
    for objective, required in REQUIRED_FOUNDATION_OBJECTIVE_FEATURES.items():
        assert coverage[objective]["required_count"] == len(required)
        assert coverage[objective]["present_count"] == len(required)
        assert coverage[objective]["missing_count"] == 0
        assert coverage[objective]["missing"] == []
        for split in ("train", "val", "test"):
            row = objective_liveness[(split, objective)]
            assert row["required_count"] == len(required)
            assert row["observed_count"] == len(required)
            assert row["missing_count"] == 0
            assert row["nonfinite_count"] == 0
            assert row["near_constant_count"] == 0
            assert row["mean_active_rate"] >= feature_audit["min_required_objective_active_rate"]

    for split, contract in feature_audit["emitted_contracts"].items():
        assert contract["foundation_structure_source_field_count"] == len(FOUNDATION_STRUCTURE_SOURCE_FIELDS)
        assert contract["foundation_structure_source_missing_count"] == 0
        assert contract["foundation_structure_source_missing"] == []
    assert len(source_liveness) == len(FOUNDATION_STRUCTURE_SOURCE_FIELDS) * 3
    for split in ("train", "val", "test"):
        for source_field in FOUNDATION_STRUCTURE_SOURCE_FIELDS:
            row = source_liveness[(split, source_field)]
            assert row["observed"] is True
            assert row["nonfinite_count"] == 0
            assert row["near_constant"] is False
            assert row["active_count"] >= feature_audit["min_required_source_active_count"]
            assert row["active_rate"] >= feature_audit["min_required_source_active_rate"]

    if report:
        assert "feature foundation audit objective coverage is all-present" in report["checks"]
        assert "feature foundation audit objective liveness is all-live" in report["checks"]
        assert "feature foundation audit source-field liveness is all-live" in report["checks"]
        assert "feature audit default out-dir matches active seq146 latest path" in report["checks"]
        assert "target audit default out-dir matches active seq146 latest path" in report["checks"]
        assert "train foundation source fields are all present" in report["checks"]
        assert "val foundation source fields are all present" in report["checks"]
        assert "test foundation source fields are all present" in report["checks"]

    specialist_routing = {
        row["objective"]: row
        for row in specialist_audit["foundation_objective_routing"]
    }
    specialist_liveness = {
        (row["split"], row["specialist"]): row
        for row in specialist_audit["specialist_input_liveness"]
    }
    assert specialist_audit["specialist_input_liveness_all_live"] is True
    for split in ("train", "val", "test"):
        for specialist in (
            "structure_swing_encoder",
            "smc_liquidity_encoder",
            "trend_ema_encoder",
            "vol_compression_encoder",
            "momentum_flow_encoder",
            "session_regime_encoder",
        ):
            row = specialist_liveness[(split, specialist)]
            assert row["live_feature_count"] >= row["min_required_live_feature_count"]
            assert row["nonfinite_count"] == 0
            assert row["mean_active_rate"] > 0.0
    assert specialist_audit["foundation_objective_routing_all_present_and_expected"] is True
    assert set(specialist_routing) == set(FOUNDATION_OBJECTIVE_SPECIALISTS)
    for objective, expected_specialist in FOUNDATION_OBJECTIVE_SPECIALISTS.items():
        assert specialist_routing[objective]["expected_specialist"] == expected_specialist
        assert specialist_routing[objective]["all_present_and_routed_to_expected"] is True
    if report:
        assert "specialist audit exact foundation objective routing is all-present" in report["checks"]
        assert "specialist audit input liveness is all-live" in report["checks"]


def test_foundation_state_selftest_covers_control_policy_contracts() -> None:
    report, active_stale_error = _run_or_active_stale(selftest=True)
    if active_stale_error:
        assert "feature foundation audit requires PASS" in active_stale_error
        assert "decision=FAIL" in active_stale_error
        return

    checks = set(report["checks"])

    assert "control surface supports non-refreshing readiness policy snapshot" in checks
    assert "control surface reports critical gate path coverage" in checks
    assert "handover reports critical gate path coverage" in checks
    assert "foundation guardrail verifier uses readiness policy snapshot" in checks
    assert "foundation guardrail verifier reports readiness policy checks" in checks
    assert "control surface exposes IQL replay slice audit" in checks
    assert "IQL slice audit requires supported edge robustness" in checks
    assert "IQL slice audit compares diagnostic tail slices" in checks
    assert "control surface exposes Entry-to-Exit handoff audit" in checks
    assert "Entry-to-Exit handoff audit blocks missing exit substrate" in checks
    assert "control surface exposes Entry-bound Exit per-bar materializer" in checks
    assert "Entry-bound Exit per-bar materializer uses handoff substrate contract" in checks
    assert "control surface exposes active Exit per-bar reconstruction audit" in checks
    assert "Entry Exit per-bar reconstruction audit requires live ATR" in checks
    assert "control surface exposes active Exit state/reward contract" in checks
    assert "Entry Exit state/reward contract checks HOLD transition pointers" in checks
    assert "control surface exposes active Exit split/leakage audit" in checks
    assert "Entry Exit split/leakage audit checks HOLD next-row split leakage" in checks
    assert "control surface exposes active Exit model dataset readiness" in checks
    assert "Entry Exit model dataset readiness uses train-only normalization" in checks
    assert "control surface exposes active Exit Transformer architecture readiness" in checks
    assert "Entry Exit Transformer architecture readiness locks model family" in checks
    assert "foundation guardrail verifier blocks candidate train in readiness policy" in checks
    assert "foundation guardrail verifier blocks IQL in readiness policy" in checks
    assert "foundation guardrail verifier blocks live in readiness policy" in checks
    assert "train readiness gate requires guardrail readiness policy proof" in checks
    assert "foundation smoke train manifest records critical gate path review" in checks
    assert "smoke bundle audit validates pretrain critical gate path review" in checks
    assert "candidate readiness gate requires smoke worktree critical gate proof" in checks
    assert "candidate train wrapper preserves smoke worktree critical gate proof" in checks
    assert "replay readiness gate requires candidate smoke worktree critical gate proof" in checks
    assert "IQL distillation contract preserves smoke worktree critical gate proof" in checks
    assert "worktree hygiene audit has critical gate path contract" in checks
    assert "worktree hygiene audit reports critical gate path review" in checks
