import argparse
import json
from pathlib import Path

import pytest

from gx1.scripts.verify_entry_candidate_readiness_v1 import (
    _mode_candidate_train_command,
    _mode_out_dir,
    _mode_smoke_bundle_audit_path,
    _mode_smoke_train_command,
    _smoke_edge_checks,
    _smart_smoke_benchmark_checks,
    run,
)
from gx1.scripts.verify_entry_training_readiness_v1 import EXPECTED_ACTIVE_TRAINING_HEADS, EXPECTED_BLOCKED_HEADS


SEQ146_SMOKE_DATASET = (
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
    "v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_foundation_seq146_smoke"
)
SEQ215_SMOKE_DATASET = (
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
    "v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_challenger_seq215_smoke_20260630"
)
SMART_SEQ520_SMOKE_DATASET = (
    "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/"
    "v10_6yr_rebuild_20260626_spreadfix/v10_dataset_6yr_smartctx_xau_direction_repair_smoke"
)
SEQ146_SPECIALISTS = [
    "structure_swing_encoder",
    "smc_liquidity_encoder",
    "trend_ema_encoder",
    "vol_compression_encoder",
    "momentum_flow_encoder",
    "session_regime_encoder",
]
SEQ215_SPECIALISTS = [*SEQ146_SPECIALISTS, "chart_geometry_encoder", "price_action_candle_encoder"]


def test_candidate_readiness_seq215_defaults_are_isolated_from_seq146_latest() -> None:
    smoke_path = _mode_smoke_bundle_audit_path("challenger_seq215")
    out_dir = _mode_out_dir("challenger_seq215")

    assert "challenger_seq215_20260630" in str(smoke_path)
    assert "challenger_seq215_20260630" in str(out_dir)
    assert smoke_path.name == "ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json"
    assert out_dir.name == "challenger_seq215_20260630"
    assert _mode_smoke_train_command("challenger_seq215") == (
        "scripts/entry_next_edge_control.sh smoke-train-seq215 --vedtak <id> --require-edge-audit"
    )
    assert _mode_candidate_train_command("challenger_seq215") == (
        "scripts/entry_next_edge_control.sh candidate-train-seq215 --vedtak <id>"
    )


def _passing_smoke_audit(
    *,
    contract_mode: str = "foundation_seq146",
    dataset_dir: str = SEQ146_SMOKE_DATASET,
    signal_dim: int = 146,
    specialists: list[str] | None = None,
) -> dict:
    groups = list(specialists or SEQ146_SPECIALISTS)
    weight = round(1.0 / len(groups), 4)
    active_heads = list(EXPECTED_ACTIVE_TRAINING_HEADS)
    if contract_mode == "smart_seq520_candidate":
        active_heads.extend(["trade_side_hierarchy", "trendline_rail", "side_validity"])
    split = {
        "rows": 128,
        "direction": {
            "rows": 128,
            "accuracy": 0.46,
            "majority_baseline_accuracy": 0.34,
            "beats_majority_baseline": True,
            "label_counts": {"LONG": 43, "SHORT": 41, "FLAT": 44},
            "prediction_counts": {"LONG": 48, "SHORT": 44, "FLAT": 36},
        },
        "direction_slice_contract": {
            "decision": "PASS",
            "min_rows": 64,
            "ctx_cat_names": ["session_id", "vol_regime_id"],
            "audited_slice_count": 2,
            "skipped_slice_count": 0,
            "fields": {
                "session_id": {
                    "finite": True,
                    "slice_count": 1,
                    "slices": {
                        "1": {
                            "decision": "PASS",
                            "rows": 128,
                            "accuracy": 0.46,
                            "majority_baseline_accuracy": 0.34,
                            "beats_majority_baseline": True,
                            "failures": [],
                        }
                    },
                },
                "vol_regime_id": {
                    "finite": True,
                    "slice_count": 1,
                    "slices": {
                        "2": {
                            "decision": "PASS",
                            "rows": 128,
                            "accuracy": 0.46,
                            "majority_baseline_accuracy": 0.34,
                            "beats_majority_baseline": True,
                            "failures": [],
                        }
                    },
                },
            },
            "failures": [],
        },
        "bad_path": {"prob_vs_path_quality_spearman": -0.22},
        "specialist_gate": {
            "finite": True,
            "row_sum_max_abs_error": 1e-7,
            "active_specialist_count_gt_1pct": len(groups),
            "entropy_mean": 1.0,
            "mean_weight": {group: weight for group in groups},
        },
    }
    report = {
        "decision": "PASS",
        "failures": [],
        "dataset_dir": dataset_dir,
        "require_edge": True,
        "require_specialist_fusion": True,
        "required_training_specialists": groups,
        "min_active_specialists": len(groups),
        "min_gate_entropy": 0.05,
        "require_head_contract": True,
        "head_contract": {
            "decision": "PASS",
            "failures": [],
            "active_training_heads": active_heads,
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
            "specialist_required_training_set_exact": True,
            "specialist_trainable_set_exact": True,
            "specialist_model_contract_valid": True,
            "specialist_model_contract_set_exact": True,
            "specialist_model_contract_owned_objectives_match": True,
            "smoke_dataset_audit_provenance_all_artifacts_present": True,
            "smoke_dataset_audit_provenance_all_artifact_hashes_present": True,
            "worktree_critical_gate_review_ok": True,
        },
        "path_calibration_recipe_contract": {
            "decision": "PASS",
            "active_heads": ["bad_path", "direction", "path_quality"],
            "path_quality_active": True,
            "bad_path_active": True,
            "path_quality_rank_full_batch": True,
            "path_quality_rank_weight": 2.0,
            "path_quality_rank_margin": 0.25,
            "path_quality_rank_quantile": 0.25,
            "bad_path_quality_rank_weight": 2.0,
            "bad_path_quality_rank_margin": 0.25,
            "bad_path_quality_rank_quantile": 0.25,
            "failures": [],
        },
        "direction_balance_recipe_contract": {
            "decision": "PASS",
            "active_heads": ["bad_path", "direction", "path_quality"],
            "direction_active": True,
            "pred_balance_alpha": 0.05,
            "pred_balance_target": "label",
            "pred_balance_class_weights": [1.0, 1.0, 1.0],
            "direction_ce_scale": 1.30,
            "ckpt_monitor": "dir_acc",
            "ckpt_class_balance_guard_weight": 0.0,
            "ckpt_class_balance_min_pred_to_label": 0.0,
            "ckpt_class_balance_min_pred_rate": 0.0,
            "failures": [],
        },
        "tail_direction_recipe_contract": {
            "decision": "PASS",
            "active_heads": ["bad_path", "direction", "path_quality"],
            "direction_active": True,
            "tail_direction_ce_weight": 0.35,
            "tail_direction_quality_quantile": 0.70,
            "tail_direction_min_batch": 8,
            "tail_direction_mask": "directional_tradable_clean_path_top_quality",
            "failures": [],
        },
        "symmetric_validation_recipe_contract": {
            "decision": "PASS",
            "active_heads": ["bad_path", "direction", "path_quality"],
            "contract_mode": contract_mode,
            "symmetric_negatives": True,
            "selector_masked_aux": True,
            "validation_objective_matches_train": True,
            "aux_selector_mode": "long_short_union",
            "clean_edge_target_mode": "bidir",
            "survival_target_mode": "bidir",
            "bad_path_ce_in_direction_loss": True,
            "bad_path_prob_penalty_in_validation": True,
            "symmetric_short_prob_penalties": True,
            "symmetric_clean_edge_rank": True,
            "failures": [],
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
            "seq_input_dim": signal_dim,
            "snap_input_dim": signal_dim,
            "multi_tf_enabled": True,
            "specialist_fusion_enabled": True,
            "specialist_model_contract_declared_valid": True,
            "specialist_model_contract_valid": True,
            "specialist_model_contract_set_exact": True,
            "specialist_model_contract_owned_objectives_match": True,
            "specialist_model_contract_support_heads_match": True,
            "specialist_model_contract_signal_families_match": True,
            "specialist_model_contract_model_roles_match": True,
            "specialist_groups": groups,
        },
        "splits": {"val": split, "test": split},
    }
    if contract_mode != "foundation_seq146":
        report["specialist_contract_mode"] = contract_mode
        report["pretrain_manifest_contract"]["specialist_contract_mode"] = contract_mode
        report["bundle_summary"]["specialist_contract_mode"] = contract_mode
    return report


def test_smoke_edge_checks_pass_on_actual_edge_contract() -> None:
    checks = _smoke_edge_checks(_passing_smoke_audit())

    assert all(check["ok"] for check in checks)


def test_smoke_edge_checks_pass_on_challenger_seq215_contract() -> None:
    report = _passing_smoke_audit(
        contract_mode="challenger_seq215",
        dataset_dir=SEQ215_SMOKE_DATASET,
        signal_dim=215,
        specialists=SEQ215_SPECIALISTS,
    )

    checks = _smoke_edge_checks(report, contract_mode="challenger_seq215", min_active_specialists=8)

    assert all(check["ok"] for check in checks)


def test_smoke_edge_checks_reject_seq215_without_challenger_contract() -> None:
    report = _passing_smoke_audit(
        dataset_dir=SEQ215_SMOKE_DATASET,
        signal_dim=215,
        specialists=SEQ215_SPECIALISTS,
    )

    checks = _smoke_edge_checks(report, contract_mode="challenger_seq215", min_active_specialists=8)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle audit specialist contract mode is challenger_seq215" in failed


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


def test_smoke_edge_checks_reject_missing_path_calibration_contract() -> None:
    report = _passing_smoke_audit()
    report["path_calibration_recipe_contract"] = {
        "decision": "FAIL",
        "path_quality_rank_full_batch": False,
        "path_quality_rank_weight": 0.0,
        "bad_path_quality_rank_weight": 2.0,
        "failures": ["path_quality active head requires path_quality_rank_full_batch=true"],
    }

    checks = _smoke_edge_checks(report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle audit path calibration recipe contract PASS" in failed


def test_smoke_edge_checks_reject_missing_direction_balance_contract() -> None:
    report = _passing_smoke_audit()
    report["direction_balance_recipe_contract"] = {
        "decision": "FAIL",
        "direction_active": True,
        "pred_balance_alpha": 0.0,
        "pred_balance_target": "uniform",
        "direction_ce_scale": 1.30,
        "ckpt_monitor": "val_loss",
        "failures": ["direction active head requires pred_balance_alpha >= 0.05"],
    }

    checks = _smoke_edge_checks(report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle audit direction balance recipe contract PASS" in failed


def test_smoke_edge_checks_require_stronger_smart_direction_balance_contract() -> None:
    report = _passing_smoke_audit(
        contract_mode="smart_seq520_candidate",
        dataset_dir=SMART_SEQ520_SMOKE_DATASET,
        signal_dim=520,
        specialists=SEQ215_SPECIALISTS,
    )

    checks = _smoke_edge_checks(
        report,
        contract_mode="smart_seq520_candidate",
        expected_smoke_dataset_dir=SMART_SEQ520_SMOKE_DATASET,
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle audit direction balance recipe contract PASS" in failed


def test_smoke_edge_checks_accept_stronger_smart_direction_balance_contract() -> None:
    report = _passing_smoke_audit(
        contract_mode="smart_seq520_candidate",
        dataset_dir=SMART_SEQ520_SMOKE_DATASET,
        signal_dim=520,
        specialists=SEQ215_SPECIALISTS,
    )
    report["direction_balance_recipe_contract"]["pred_balance_alpha"] = 0.50
    report["direction_balance_recipe_contract"]["pred_balance_class_weights"] = [1.0, 1.0, 4.0]
    report["direction_balance_recipe_contract"]["direction_ce_scale"] = 2.00
    report["direction_balance_recipe_contract"]["hierarchical_entry_heads_enabled"] = True
    report["direction_balance_recipe_contract"]["side_validity_head_enabled"] = True
    report["direction_balance_recipe_contract"]["hier_side_validity_weight"] = 1.50
    report["direction_balance_recipe_contract"]["hier_side_validity_min_utility_bps"] = 15.0
    report["direction_balance_recipe_contract"]["hier_side_validity_pos_weight_cap"] = 8.0
    report["direction_balance_recipe_contract"]["trendline_rail_head_enabled"] = True
    report["direction_balance_recipe_contract"]["trendline_rail_aux_weight"] = 1.00
    report["direction_balance_recipe_contract"]["trendline_rail_wrong_side_weight"] = 1.50
    report["direction_balance_recipe_contract"]["hier_legacy_ce_mult"] = 1.00
    report["direction_balance_recipe_contract"]["anchor_gate_enabled"] = True
    report["direction_balance_recipe_contract"]["anchor_gate_init"] = 0.0
    report["direction_balance_recipe_contract"]["ckpt_class_balance_guard_weight"] = 0.50
    report["direction_balance_recipe_contract"]["ckpt_class_balance_min_pred_to_label"] = 0.35
    report["direction_balance_recipe_contract"]["ckpt_class_balance_min_pred_rate"] = 0.05
    report["direction_balance_recipe_contract"]["direction_min_pred_rate_loss_weight"] = 2.50
    report["direction_balance_recipe_contract"]["direction_min_pred_rate_fraction"] = 0.50
    report["direction_balance_recipe_contract"]["direction_min_pred_rate_floor"] = 0.05
    report["direction_balance_recipe_contract"]["direction_min_pred_rate_softmax_temperature"] = 0.20
    report["direction_balance_recipe_contract"]["direction_slice_accuracy_edge_weight"] = 4.00
    report["direction_balance_recipe_contract"]["direction_slice_accuracy_edge_margin"] = 0.02
    report["direction_balance_recipe_contract"]["direction_slice_confusion_pair_weight"] = 4.00
    report["direction_balance_recipe_contract"]["direction_slice_confusion_pair_margin"] = 0.02
    report["direction_balance_recipe_contract"]["direction_vs_flat_margin_weight"] = 4.00
    report["direction_balance_recipe_contract"]["direction_vs_flat_margin"] = 0.10
    report["direction_balance_recipe_contract"]["direction_utility_margin_weight"] = 4.00
    report["direction_balance_recipe_contract"]["direction_utility_min_gap_bps"] = 15.0
    report["direction_balance_recipe_contract"]["direction_utility_logit_margin"] = 0.10
    report["direction_balance_recipe_contract"]["direction_side_utility_conviction_weight"] = 6.00
    report["direction_balance_recipe_contract"]["direction_side_utility_conviction_min_gap_bps"] = 15.0
    report["direction_balance_recipe_contract"]["direction_side_utility_conviction_logit_margin"] = 0.10
    report["direction_balance_recipe_contract"]["direction_utility_trade_conviction_weight"] = 8.00
    report["direction_balance_recipe_contract"]["direction_utility_trade_conviction_min_gap_bps"] = 15.0
    report["direction_balance_recipe_contract"]["direction_utility_trade_conviction_min_utility_bps"] = 0.0
    report["direction_balance_recipe_contract"]["direction_utility_trade_conviction_max_bad_path"] = 0.50
    report["direction_balance_recipe_contract"]["direction_utility_trade_conviction_logit_margin"] = 0.10
    report["direction_balance_recipe_contract"]["direction_utility_triad_ce_weight"] = 8.00
    report["direction_balance_recipe_contract"]["direction_utility_triad_ce_min_gap_bps"] = 15.0
    report["direction_balance_recipe_contract"]["direction_utility_triad_ce_min_utility_bps"] = 0.0
    report["direction_balance_recipe_contract"]["direction_utility_triad_ce_max_bad_path"] = 0.50
    report["direction_balance_recipe_contract"]["direction_utility_triad_ce_class_weight_cap"] = 4.0
    report["direction_balance_recipe_contract"]["direction_hierarchical_composition"] = True
    report["direction_balance_recipe_contract"]["hier_compose_residual_logit_cap"] = 0.18
    report["direction_balance_recipe_contract"]["hier_compose_residual_side_neutral"] = True
    report["direction_balance_recipe_contract"]["hier_compose_public_flat_from_trade"] = True
    report["direction_balance_recipe_contract"]["hier_public_direction_composition"] = "margin_maxnorm_confidence"
    report["direction_balance_recipe_contract"]["hier_public_trade_head"] = True
    report["direction_balance_recipe_contract"]["hier_public_trade_dir_margin_bridge"] = True
    report["direction_balance_recipe_contract"]["hier_public_trade_dir_margin_bridge_scale"] = 0.50
    report["direction_balance_recipe_contract"]["hier_public_trade_dir_margin_bridge_cap"] = 0.25
    report["direction_balance_recipe_contract"]["hier_public_side_head"] = True
    report["direction_balance_recipe_contract"]["hier_ctx_prior_adapter"] = True
    report["direction_balance_recipe_contract"]["hier_ctx_prior_adapter_scale"] = 0.50
    report["direction_balance_recipe_contract"]["hier_ctx_direction_calibration"] = True
    report["direction_balance_recipe_contract"]["hier_ctx_direction_calibration_scale"] = 0.50
    report["direction_balance_recipe_contract"]["hier_ctx_direction_calibration_cap"] = 0.35
    report["direction_balance_recipe_contract"]["hier_trade_global_prior_match_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_trade_global_prior_match_tolerance"] = 0.02
    report["direction_balance_recipe_contract"]["hier_trade_global_prior_match_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_trade_prior_match_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_slice_trade_prior_match_tolerance"] = 0.02
    report["direction_balance_recipe_contract"]["hier_slice_trade_prior_match_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_trade_prior_match_min_rows"] = 8
    report["direction_balance_recipe_contract"]["hier_slice_trade_accuracy_edge_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_slice_trade_accuracy_edge_margin"] = 0.02
    report["direction_balance_recipe_contract"]["hier_flat_logit_margin_weight"] = 8.00
    report["direction_balance_recipe_contract"]["hier_flat_logit_margin"] = 0.10
    report["direction_balance_recipe_contract"]["hier_flat_logit_margin_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_flat_logit_margin_weight"] = 8.00
    report["direction_balance_recipe_contract"]["hier_slice_flat_logit_margin"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_flat_logit_margin_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_flat_logit_margin_min_rows"] = 8
    report["direction_balance_recipe_contract"]["hier_public_flat_consistency_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_public_flat_consistency_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_public_flat_consistency_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_slice_public_flat_consistency_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_public_flat_consistency_min_rows"] = 8
    report["direction_balance_recipe_contract"]["hier_side_global_prior_match_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_side_global_prior_match_tolerance"] = 0.02
    report["direction_balance_recipe_contract"]["hier_side_global_prior_match_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_side_accuracy_edge_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_slice_side_accuracy_edge_margin"] = 0.02
    report["direction_balance_recipe_contract"]["hier_slice_side_prior_match_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_slice_side_prior_match_tolerance"] = 0.02
    report["direction_balance_recipe_contract"]["hier_slice_side_prior_match_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_side_prior_match_min_rows"] = 8
    report["direction_balance_recipe_contract"]["direction_flat_starvation_weight"] = 8.00
    report["direction_balance_recipe_contract"]["direction_flat_starvation_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["direction_flat_starvation_min_rows"] = 8
    report["direction_balance_recipe_contract"]["direction_flat_starvation_pred_fraction"] = 0.50
    report["direction_balance_recipe_contract"]["direction_flat_starvation_pred_floor"] = 0.10
    report["direction_balance_recipe_contract"]["direction_flat_starvation_logit_margin"] = 0.10
    report["direction_balance_recipe_contract"]["best_direction_balance_guard_ok"] = True

    checks = _smoke_edge_checks(
        report,
        contract_mode="smart_seq520_candidate",
        expected_smoke_dataset_dir=SMART_SEQ520_SMOKE_DATASET,
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle audit direction balance recipe contract PASS" not in failed


def test_smoke_edge_checks_reject_smart_missing_symmetric_validation_contract() -> None:
    report = _passing_smoke_audit(
        contract_mode="smart_seq520_candidate",
        dataset_dir=SMART_SEQ520_SMOKE_DATASET,
        signal_dim=520,
        specialists=SEQ215_SPECIALISTS,
    )
    report["direction_balance_recipe_contract"]["pred_balance_alpha"] = 0.50
    report["direction_balance_recipe_contract"]["pred_balance_class_weights"] = [1.0, 1.0, 4.0]
    report["direction_balance_recipe_contract"]["direction_ce_scale"] = 2.00
    report["direction_balance_recipe_contract"]["hierarchical_entry_heads_enabled"] = True
    report["direction_balance_recipe_contract"]["side_validity_head_enabled"] = True
    report["direction_balance_recipe_contract"]["hier_side_validity_weight"] = 1.50
    report["direction_balance_recipe_contract"]["hier_side_validity_min_utility_bps"] = 15.0
    report["direction_balance_recipe_contract"]["hier_side_validity_pos_weight_cap"] = 8.0
    report["direction_balance_recipe_contract"]["trendline_rail_head_enabled"] = True
    report["direction_balance_recipe_contract"]["trendline_rail_aux_weight"] = 1.00
    report["direction_balance_recipe_contract"]["trendline_rail_wrong_side_weight"] = 1.50
    report["direction_balance_recipe_contract"]["hier_legacy_ce_mult"] = 1.00
    report["direction_balance_recipe_contract"]["anchor_gate_enabled"] = True
    report["direction_balance_recipe_contract"]["anchor_gate_init"] = 0.0
    report["direction_balance_recipe_contract"]["ckpt_class_balance_guard_weight"] = 0.50
    report["direction_balance_recipe_contract"]["ckpt_class_balance_min_pred_to_label"] = 0.35
    report["direction_balance_recipe_contract"]["ckpt_class_balance_min_pred_rate"] = 0.05
    report["direction_balance_recipe_contract"]["direction_min_pred_rate_loss_weight"] = 2.50
    report["direction_balance_recipe_contract"]["direction_min_pred_rate_fraction"] = 0.50
    report["direction_balance_recipe_contract"]["direction_min_pred_rate_floor"] = 0.05
    report["direction_balance_recipe_contract"]["direction_min_pred_rate_softmax_temperature"] = 0.20
    report["direction_balance_recipe_contract"]["direction_slice_accuracy_edge_weight"] = 4.00
    report["direction_balance_recipe_contract"]["direction_slice_accuracy_edge_margin"] = 0.02
    report["direction_balance_recipe_contract"]["direction_slice_confusion_pair_weight"] = 4.00
    report["direction_balance_recipe_contract"]["direction_slice_confusion_pair_margin"] = 0.02
    report["direction_balance_recipe_contract"]["direction_vs_flat_margin_weight"] = 4.00
    report["direction_balance_recipe_contract"]["direction_vs_flat_margin"] = 0.10
    report["direction_balance_recipe_contract"]["direction_utility_margin_weight"] = 4.00
    report["direction_balance_recipe_contract"]["direction_utility_min_gap_bps"] = 15.0
    report["direction_balance_recipe_contract"]["direction_utility_logit_margin"] = 0.10
    report["direction_balance_recipe_contract"]["direction_side_utility_conviction_weight"] = 6.00
    report["direction_balance_recipe_contract"]["direction_side_utility_conviction_min_gap_bps"] = 15.0
    report["direction_balance_recipe_contract"]["direction_side_utility_conviction_logit_margin"] = 0.10
    report["direction_balance_recipe_contract"]["direction_utility_trade_conviction_weight"] = 8.00
    report["direction_balance_recipe_contract"]["direction_utility_trade_conviction_min_gap_bps"] = 15.0
    report["direction_balance_recipe_contract"]["direction_utility_trade_conviction_min_utility_bps"] = 0.0
    report["direction_balance_recipe_contract"]["direction_utility_trade_conviction_max_bad_path"] = 0.50
    report["direction_balance_recipe_contract"]["direction_utility_trade_conviction_logit_margin"] = 0.10
    report["direction_balance_recipe_contract"]["direction_utility_triad_ce_weight"] = 8.00
    report["direction_balance_recipe_contract"]["direction_utility_triad_ce_min_gap_bps"] = 15.0
    report["direction_balance_recipe_contract"]["direction_utility_triad_ce_min_utility_bps"] = 0.0
    report["direction_balance_recipe_contract"]["direction_utility_triad_ce_max_bad_path"] = 0.50
    report["direction_balance_recipe_contract"]["direction_utility_triad_ce_class_weight_cap"] = 4.0
    report["direction_balance_recipe_contract"]["direction_hierarchical_composition"] = True
    report["direction_balance_recipe_contract"]["hier_compose_residual_logit_cap"] = 0.18
    report["direction_balance_recipe_contract"]["hier_compose_residual_side_neutral"] = True
    report["direction_balance_recipe_contract"]["hier_compose_public_flat_from_trade"] = True
    report["direction_balance_recipe_contract"]["hier_public_direction_composition"] = "margin_maxnorm_confidence"
    report["direction_balance_recipe_contract"]["hier_public_trade_head"] = True
    report["direction_balance_recipe_contract"]["hier_public_trade_dir_margin_bridge"] = True
    report["direction_balance_recipe_contract"]["hier_public_trade_dir_margin_bridge_scale"] = 0.50
    report["direction_balance_recipe_contract"]["hier_public_trade_dir_margin_bridge_cap"] = 0.25
    report["direction_balance_recipe_contract"]["hier_public_side_head"] = True
    report["direction_balance_recipe_contract"]["hier_ctx_prior_adapter"] = True
    report["direction_balance_recipe_contract"]["hier_ctx_prior_adapter_scale"] = 0.50
    report["direction_balance_recipe_contract"]["hier_ctx_direction_calibration"] = True
    report["direction_balance_recipe_contract"]["hier_ctx_direction_calibration_scale"] = 0.50
    report["direction_balance_recipe_contract"]["hier_ctx_direction_calibration_cap"] = 0.35
    report["direction_balance_recipe_contract"]["hier_trade_global_prior_match_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_trade_global_prior_match_tolerance"] = 0.02
    report["direction_balance_recipe_contract"]["hier_trade_global_prior_match_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_trade_prior_match_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_slice_trade_prior_match_tolerance"] = 0.02
    report["direction_balance_recipe_contract"]["hier_slice_trade_prior_match_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_trade_prior_match_min_rows"] = 8
    report["direction_balance_recipe_contract"]["hier_slice_trade_accuracy_edge_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_slice_trade_accuracy_edge_margin"] = 0.02
    report["direction_balance_recipe_contract"]["hier_flat_logit_margin_weight"] = 8.00
    report["direction_balance_recipe_contract"]["hier_flat_logit_margin"] = 0.10
    report["direction_balance_recipe_contract"]["hier_flat_logit_margin_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_flat_logit_margin_weight"] = 8.00
    report["direction_balance_recipe_contract"]["hier_slice_flat_logit_margin"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_flat_logit_margin_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_flat_logit_margin_min_rows"] = 8
    report["direction_balance_recipe_contract"]["hier_public_flat_consistency_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_public_flat_consistency_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_public_flat_consistency_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_slice_public_flat_consistency_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_public_flat_consistency_min_rows"] = 8
    report["direction_balance_recipe_contract"]["hier_side_global_prior_match_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_side_global_prior_match_tolerance"] = 0.02
    report["direction_balance_recipe_contract"]["hier_side_global_prior_match_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_side_accuracy_edge_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_slice_side_accuracy_edge_margin"] = 0.02
    report["direction_balance_recipe_contract"]["hier_slice_side_prior_match_weight"] = 4.00
    report["direction_balance_recipe_contract"]["hier_slice_side_prior_match_tolerance"] = 0.02
    report["direction_balance_recipe_contract"]["hier_slice_side_prior_match_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["hier_slice_side_prior_match_min_rows"] = 8
    report["direction_balance_recipe_contract"]["direction_flat_starvation_weight"] = 8.00
    report["direction_balance_recipe_contract"]["direction_flat_starvation_min_label_rate"] = 0.10
    report["direction_balance_recipe_contract"]["direction_flat_starvation_min_rows"] = 8
    report["direction_balance_recipe_contract"]["direction_flat_starvation_pred_fraction"] = 0.50
    report["direction_balance_recipe_contract"]["direction_flat_starvation_pred_floor"] = 0.10
    report["direction_balance_recipe_contract"]["direction_flat_starvation_logit_margin"] = 0.10
    report["symmetric_validation_recipe_contract"] = {
        "decision": "FAIL",
        "active_heads": ["bad_path", "direction", "path_quality"],
        "symmetric_negatives": True,
        "selector_masked_aux": True,
        "validation_objective_matches_train": False,
        "aux_selector_mode": "long_only",
        "clean_edge_target_mode": "long",
        "survival_target_mode": "long",
        "bad_path_ce_in_direction_loss": True,
        "bad_path_prob_penalty_in_validation": False,
        "symmetric_short_prob_penalties": False,
        "failures": ["smart symmetric validation requires validation_objective_matches_train=true"],
    }

    checks = _smoke_edge_checks(
        report,
        contract_mode="smart_seq520_candidate",
        expected_smoke_dataset_dir=SMART_SEQ520_SMOKE_DATASET,
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle audit symmetric validation recipe contract PASS" in failed


def test_smart_smoke_benchmark_checks_accept_matching_baselines() -> None:
    smart = _passing_smoke_audit(
        contract_mode="smart_seq520_candidate",
        dataset_dir=SMART_SEQ520_SMOKE_DATASET,
        signal_dim=520,
        specialists=SEQ215_SPECIALISTS,
    )
    foundation = _passing_smoke_audit()
    seq215 = _passing_smoke_audit(
        contract_mode="challenger_seq215",
        dataset_dir=SEQ215_SMOKE_DATASET,
        signal_dim=215,
        specialists=SEQ215_SPECIALISTS,
    )

    checks = _smart_smoke_benchmark_checks(
        smart,
        foundation_report=foundation,
        seq215_report=seq215,
    )

    assert all(check["ok"] for check in checks)


def test_smart_smoke_benchmark_checks_reject_direction_and_balance_regression() -> None:
    smart = _passing_smoke_audit(
        contract_mode="smart_seq520_candidate",
        dataset_dir=SMART_SEQ520_SMOKE_DATASET,
        signal_dim=520,
        specialists=SEQ215_SPECIALISTS,
    )
    for split in smart["splits"].values():
        split["direction"]["accuracy"] = 0.35
        split["direction"]["prediction_counts"] = {"LONG": 80, "SHORT": 44, "FLAT": 4}
    foundation = _passing_smoke_audit()
    seq215 = _passing_smoke_audit(
        contract_mode="challenger_seq215",
        dataset_dir=SEQ215_SMOKE_DATASET,
        signal_dim=215,
        specialists=SEQ215_SPECIALISTS,
    )

    checks = _smart_smoke_benchmark_checks(
        smart,
        foundation_report=foundation,
        seq215_report=seq215,
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smart smoke direction accuracy does not regress versus foundation/seq215" in failed
    assert "smart smoke class-balance drift does not regress versus foundation/seq215" in failed


def test_smart_smoke_benchmark_checks_reject_non_strict_scope() -> None:
    smart = _passing_smoke_audit(
        contract_mode="smart_seq520_candidate",
        dataset_dir=SMART_SEQ520_SMOKE_DATASET,
        signal_dim=520,
        specialists=SEQ215_SPECIALISTS,
    )
    foundation = _passing_smoke_audit()
    seq215 = _passing_smoke_audit(
        contract_mode="challenger_seq215",
        dataset_dir=SEQ215_SMOKE_DATASET,
        signal_dim=215,
        specialists=SEQ215_SPECIALISTS,
    )

    with pytest.raises(ValueError, match="NO_EDGE_FALLBACK"):
        _smart_smoke_benchmark_checks(
            smart,
            foundation_report=foundation,
            seq215_report=seq215,
            edge_test_scope="smoke",
        )


def test_smoke_edge_checks_reject_missing_tail_direction_contract() -> None:
    report = _passing_smoke_audit()
    report["tail_direction_recipe_contract"] = {
        "decision": "FAIL",
        "direction_active": True,
        "tail_direction_ce_weight": 0.0,
        "tail_direction_quality_quantile": 0.70,
        "tail_direction_min_batch": 8,
        "tail_direction_mask": "directional_tradable_clean_path_top_quality",
        "failures": ["direction active head requires positive tail_direction_ce_weight"],
    }

    checks = _smoke_edge_checks(report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smoke bundle audit tail direction recipe contract PASS" in failed


def test_smoke_edge_checks_reject_direction_distribution_collapse() -> None:
    report = _passing_smoke_audit()
    report["splits"]["test"]["direction"]["label_counts"] = {"LONG": 42, "SHORT": 40, "FLAT": 46}
    report["splits"]["test"]["direction"]["prediction_counts"] = {"LONG": 70, "SHORT": 54, "FLAT": 4}

    checks = _smoke_edge_checks(report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "direction distribution covers active LONG/SHORT/FLAT classes" in failed


def test_smoke_edge_checks_reject_direction_slice_failure() -> None:
    report = _passing_smoke_audit()
    report["splits"]["val"]["direction_slice_contract"] = {
        "decision": "FAIL",
        "audited_slice_count": 1,
        "failures": ["session_id=2: accuracy=0.300000 does not beat majority=0.500000"],
    }

    checks = _smoke_edge_checks(report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "direction context slices pass session/regime bucket diagnostics" in failed


def test_smoke_edge_checks_reject_non_strict_scope() -> None:
    report = _passing_smoke_audit()

    with pytest.raises(ValueError, match="NO_EDGE_FALLBACK"):
        _smoke_edge_checks(report, edge_test_scope="smoke")


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


def test_smoke_edge_checks_reject_missing_pretrain_exact_trainable_specialist_set() -> None:
    report = _passing_smoke_audit(
        contract_mode="challenger_seq215",
        dataset_dir=SEQ215_SMOKE_DATASET,
        signal_dim=215,
        specialists=SEQ215_SPECIALISTS,
    )
    report["pretrain_manifest_contract"]["specialist_required_training_set_exact"] = False
    report["pretrain_manifest_contract"]["specialist_trainable_set_exact"] = False

    checks = _smoke_edge_checks(report, contract_mode="challenger_seq215", min_active_specialists=8)
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


def test_candidate_readiness_seq215_requires_challenger_contract_and_clean_gates(tmp_path: Path, monkeypatch) -> None:
    train_readiness_path = tmp_path / "train_readiness.json"
    train_readiness_path.write_text(
        json.dumps(
            {
                "decision": "READY_FOR_VEDTAK_SMOKE_TRAIN",
                "candidate_training_allowed": False,
                "promotion_shadow_live_allowed": False,
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    specialist_path = tmp_path / "specialist_audit.json"
    specialist_path.write_text(json.dumps({"decision": "PASS"}), encoding="utf-8")

    def fake_train_readiness(_args):
        return {
            "decision": "READY_FOR_VEDTAK_SMOKE_TRAIN",
            "candidate_training_allowed": False,
            "promotion_shadow_live_allowed": False,
            "failures": [],
            "json_path": str(train_readiness_path),
        }

    monkeypatch.setattr("gx1.scripts.verify_entry_candidate_readiness_v1.run_train_readiness", fake_train_readiness)
    smoke_path = tmp_path / "seq215_smoke_audit.json"
    smoke_path.write_text(
        json.dumps(
            _passing_smoke_audit(
                contract_mode="challenger_seq215",
                dataset_dir=SEQ215_SMOKE_DATASET,
                signal_dim=215,
                specialists=SEQ215_SPECIALISTS,
            )
        ),
        encoding="utf-8",
    )

    report = run(
        argparse.Namespace(
            audit_doc="/home/andre2/src/GX1_ENGINE/docs/ENTRY_FOUNDATION_AUDIT_20260628.md",
            smoke_bundle_audit_json=str(smoke_path),
            specialist_audit_json=str(specialist_path),
            contract_mode="challenger_seq215",
            out_dir=str(tmp_path / "out"),
            min_active_specialists=8,
            fail_on_not_ready=False,
            quiet=True,
        )
    )

    assert report["decision"] == "READY_FOR_CANDIDATE_TRAINING_VEDTAK"
    assert report["contract_mode"] == "challenger_seq215"
    assert report["expected_signal_dim"] == 215
    assert report["required_specialist_groups"] == SEQ215_SPECIALISTS
    assert report["candidate_training_allowed_with_explicit_vedtak"] is True
    assert report["promotion_shadow_live_allowed"] is False
    assert report["next_required_gate"] == (
        "scripts/entry_next_edge_control.sh candidate-train-seq215 --vedtak <id> then post-train replay gates"
    )

    smoke_path.write_text(
        json.dumps(
            _passing_smoke_audit(
                contract_mode="foundation_seq146",
                dataset_dir=SEQ215_SMOKE_DATASET,
                signal_dim=215,
                specialists=SEQ215_SPECIALISTS,
            )
        ),
        encoding="utf-8",
    )
    blocked = run(
        argparse.Namespace(
            audit_doc="/home/andre2/src/GX1_ENGINE/docs/ENTRY_FOUNDATION_AUDIT_20260628.md",
            smoke_bundle_audit_json=str(smoke_path),
            specialist_audit_json=str(specialist_path),
            contract_mode="challenger_seq215",
            out_dir=str(tmp_path / "blocked"),
            min_active_specialists=8,
            fail_on_not_ready=False,
            quiet=True,
        )
    )

    assert blocked["decision"] == "NOT_READY_FOR_CANDIDATE_TRAINING"
    assert blocked["candidate_training_allowed_with_explicit_vedtak"] is False
    failed = {failure["check"] for failure in blocked["failures"]}
    assert "smoke bundle audit specialist contract mode is challenger_seq215" in failed


def test_candidate_readiness_seq215_missing_smoke_audit_reports_not_ready(tmp_path: Path, monkeypatch) -> None:
    train_readiness_path = tmp_path / "train_readiness.json"
    train_readiness_path.write_text(
        json.dumps(
            {
                "decision": "READY_FOR_VEDTAK_SMOKE_TRAIN",
                "candidate_training_allowed": False,
                "promotion_shadow_live_allowed": False,
                "failures": [],
            }
        ),
        encoding="utf-8",
    )

    def fake_train_readiness(_args):
        return {
            "decision": "READY_FOR_VEDTAK_SMOKE_TRAIN",
            "candidate_training_allowed": False,
            "promotion_shadow_live_allowed": False,
            "failures": [],
            "json_path": str(train_readiness_path),
        }

    monkeypatch.setattr("gx1.scripts.verify_entry_candidate_readiness_v1.run_train_readiness", fake_train_readiness)
    missing_smoke_path = tmp_path / "missing_seq215_smoke_audit.json"
    report = run(
        argparse.Namespace(
            audit_doc="/home/andre2/src/GX1_ENGINE/docs/ENTRY_FOUNDATION_AUDIT_20260628.md",
            smoke_bundle_audit_json=str(missing_smoke_path),
            specialist_audit_json=None,
            contract_mode="challenger_seq215",
            out_dir=str(tmp_path / "out_missing"),
            min_active_specialists=8,
            fail_on_not_ready=False,
            quiet=True,
        )
    )

    assert report["decision"] == "NOT_READY_FOR_CANDIDATE_TRAINING"
    assert report["candidate_training_allowed_with_explicit_vedtak"] is False
    assert report["next_required_gate"] == (
        "run scripts/entry_next_edge_control.sh smoke-train-seq215 --vedtak <id> --require-edge-audit"
    )
    assert report["smoke_bundle_audit_json"] == str(missing_smoke_path.resolve())
    assert "missing JSON artifact" in str(report["smoke_bundle_audit_load_error"])
    failed = {failure["check"] for failure in report["failures"]}
    assert "smoke bundle audit JSON exists and is readable" in failed
    assert Path(report["json_path"]).exists()


def test_candidate_readiness_smart_seq520_opens_after_contract_and_smoke_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    trainability_path = tmp_path / "smart_trainability.json"
    trainability_path.write_text(
        json.dumps(
            {
                "decision": "READY_FOR_SMART_SEQ520_TRAINABILITY_REVIEW",
                "candidate_training_allowed": False,
                "promotion_shadow_live_allowed": False,
                "failures": [],
            }
        ),
        encoding="utf-8",
    )
    specialist_path = tmp_path / "specialist_audit.json"
    specialist_path.write_text(json.dumps({"decision": "PASS"}), encoding="utf-8")
    smoke_path = tmp_path / "smart_smoke_audit.json"
    smart_smoke_report = _passing_smoke_audit(
        contract_mode="smart_seq520_candidate",
        dataset_dir=SMART_SEQ520_SMOKE_DATASET,
        signal_dim=520,
        specialists=SEQ215_SPECIALISTS,
    )
    smart_smoke_report["direction_balance_recipe_contract"]["pred_balance_alpha"] = 0.50
    smart_smoke_report["direction_balance_recipe_contract"]["pred_balance_class_weights"] = [1.0, 1.0, 4.0]
    smart_smoke_report["direction_balance_recipe_contract"]["direction_ce_scale"] = 2.00
    smart_smoke_report["direction_balance_recipe_contract"]["hierarchical_entry_heads_enabled"] = True
    smart_smoke_report["direction_balance_recipe_contract"]["side_validity_head_enabled"] = True
    smart_smoke_report["direction_balance_recipe_contract"]["hier_side_validity_weight"] = 1.50
    smart_smoke_report["direction_balance_recipe_contract"]["hier_side_validity_min_utility_bps"] = 15.0
    smart_smoke_report["direction_balance_recipe_contract"]["hier_side_validity_pos_weight_cap"] = 8.0
    smart_smoke_report["direction_balance_recipe_contract"]["trendline_rail_head_enabled"] = True
    smart_smoke_report["direction_balance_recipe_contract"]["trendline_rail_aux_weight"] = 1.00
    smart_smoke_report["direction_balance_recipe_contract"]["trendline_rail_wrong_side_weight"] = 1.50
    smart_smoke_report["direction_balance_recipe_contract"]["hier_legacy_ce_mult"] = 1.00
    smart_smoke_report["direction_balance_recipe_contract"]["anchor_gate_enabled"] = True
    smart_smoke_report["direction_balance_recipe_contract"]["anchor_gate_init"] = 0.0
    smart_smoke_report["direction_balance_recipe_contract"]["ckpt_class_balance_guard_weight"] = 0.50
    smart_smoke_report["direction_balance_recipe_contract"]["ckpt_class_balance_min_pred_to_label"] = 0.35
    smart_smoke_report["direction_balance_recipe_contract"]["ckpt_class_balance_min_pred_rate"] = 0.05
    smart_smoke_report["direction_balance_recipe_contract"]["direction_min_pred_rate_loss_weight"] = 2.50
    smart_smoke_report["direction_balance_recipe_contract"]["direction_min_pred_rate_fraction"] = 0.50
    smart_smoke_report["direction_balance_recipe_contract"]["direction_min_pred_rate_floor"] = 0.05
    smart_smoke_report["direction_balance_recipe_contract"]["direction_min_pred_rate_softmax_temperature"] = 0.20
    smart_smoke_report["direction_balance_recipe_contract"]["direction_slice_accuracy_edge_weight"] = 4.00
    smart_smoke_report["direction_balance_recipe_contract"]["direction_slice_accuracy_edge_margin"] = 0.02
    smart_smoke_report["direction_balance_recipe_contract"]["direction_slice_confusion_pair_weight"] = 4.00
    smart_smoke_report["direction_balance_recipe_contract"]["direction_slice_confusion_pair_margin"] = 0.02
    smart_smoke_report["direction_balance_recipe_contract"]["direction_vs_flat_margin_weight"] = 4.00
    smart_smoke_report["direction_balance_recipe_contract"]["direction_vs_flat_margin"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["direction_utility_margin_weight"] = 4.00
    smart_smoke_report["direction_balance_recipe_contract"]["direction_utility_min_gap_bps"] = 15.0
    smart_smoke_report["direction_balance_recipe_contract"]["direction_utility_logit_margin"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["direction_side_utility_conviction_weight"] = 6.00
    smart_smoke_report["direction_balance_recipe_contract"]["direction_side_utility_conviction_min_gap_bps"] = 15.0
    smart_smoke_report["direction_balance_recipe_contract"]["direction_side_utility_conviction_logit_margin"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["direction_utility_trade_conviction_weight"] = 8.00
    smart_smoke_report["direction_balance_recipe_contract"]["direction_utility_trade_conviction_min_gap_bps"] = 15.0
    smart_smoke_report["direction_balance_recipe_contract"]["direction_utility_trade_conviction_min_utility_bps"] = 0.0
    smart_smoke_report["direction_balance_recipe_contract"]["direction_utility_trade_conviction_max_bad_path"] = 0.50
    smart_smoke_report["direction_balance_recipe_contract"]["direction_utility_trade_conviction_logit_margin"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["direction_utility_triad_ce_weight"] = 8.00
    smart_smoke_report["direction_balance_recipe_contract"]["direction_utility_triad_ce_min_gap_bps"] = 15.0
    smart_smoke_report["direction_balance_recipe_contract"]["direction_utility_triad_ce_min_utility_bps"] = 0.0
    smart_smoke_report["direction_balance_recipe_contract"]["direction_utility_triad_ce_max_bad_path"] = 0.50
    smart_smoke_report["direction_balance_recipe_contract"]["direction_utility_triad_ce_class_weight_cap"] = 4.0
    smart_smoke_report["direction_balance_recipe_contract"]["direction_hierarchical_composition"] = True
    smart_smoke_report["direction_balance_recipe_contract"]["hier_compose_residual_logit_cap"] = 0.18
    smart_smoke_report["direction_balance_recipe_contract"]["hier_compose_residual_side_neutral"] = True
    smart_smoke_report["direction_balance_recipe_contract"]["hier_compose_public_flat_from_trade"] = True
    smart_smoke_report["direction_balance_recipe_contract"]["hier_public_direction_composition"] = "margin_maxnorm_confidence"
    smart_smoke_report["direction_balance_recipe_contract"]["hier_public_trade_head"] = True
    smart_smoke_report["direction_balance_recipe_contract"]["hier_public_trade_dir_margin_bridge"] = True
    smart_smoke_report["direction_balance_recipe_contract"]["hier_public_trade_dir_margin_bridge_scale"] = 0.50
    smart_smoke_report["direction_balance_recipe_contract"]["hier_public_trade_dir_margin_bridge_cap"] = 0.25
    smart_smoke_report["direction_balance_recipe_contract"]["hier_public_side_head"] = True
    smart_smoke_report["direction_balance_recipe_contract"]["hier_ctx_prior_adapter"] = True
    smart_smoke_report["direction_balance_recipe_contract"]["hier_ctx_prior_adapter_scale"] = 0.50
    smart_smoke_report["direction_balance_recipe_contract"]["hier_ctx_direction_calibration"] = True
    smart_smoke_report["direction_balance_recipe_contract"]["hier_ctx_direction_calibration_scale"] = 0.50
    smart_smoke_report["direction_balance_recipe_contract"]["hier_ctx_direction_calibration_cap"] = 0.35
    smart_smoke_report["direction_balance_recipe_contract"]["hier_trade_global_prior_match_weight"] = 4.00
    smart_smoke_report["direction_balance_recipe_contract"]["hier_trade_global_prior_match_tolerance"] = 0.02
    smart_smoke_report["direction_balance_recipe_contract"]["hier_trade_global_prior_match_min_label_rate"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_trade_prior_match_weight"] = 4.00
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_trade_prior_match_tolerance"] = 0.02
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_trade_prior_match_min_label_rate"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_trade_prior_match_min_rows"] = 8
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_trade_accuracy_edge_weight"] = 4.00
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_trade_accuracy_edge_margin"] = 0.02
    smart_smoke_report["direction_balance_recipe_contract"]["hier_flat_logit_margin_weight"] = 8.00
    smart_smoke_report["direction_balance_recipe_contract"]["hier_flat_logit_margin"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["hier_flat_logit_margin_min_label_rate"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_flat_logit_margin_weight"] = 8.00
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_flat_logit_margin"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_flat_logit_margin_min_label_rate"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_flat_logit_margin_min_rows"] = 8
    smart_smoke_report["direction_balance_recipe_contract"]["hier_public_flat_consistency_weight"] = 4.00
    smart_smoke_report["direction_balance_recipe_contract"]["hier_public_flat_consistency_min_label_rate"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_public_flat_consistency_weight"] = 4.00
    smart_smoke_report["direction_balance_recipe_contract"][
        "hier_slice_public_flat_consistency_min_label_rate"
    ] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_public_flat_consistency_min_rows"] = 8
    smart_smoke_report["direction_balance_recipe_contract"]["hier_side_global_prior_match_weight"] = 4.00
    smart_smoke_report["direction_balance_recipe_contract"]["hier_side_global_prior_match_tolerance"] = 0.02
    smart_smoke_report["direction_balance_recipe_contract"]["hier_side_global_prior_match_min_label_rate"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_side_accuracy_edge_weight"] = 4.00
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_side_accuracy_edge_margin"] = 0.02
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_side_prior_match_weight"] = 4.00
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_side_prior_match_tolerance"] = 0.02
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_side_prior_match_min_label_rate"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["hier_slice_side_prior_match_min_rows"] = 8
    smart_smoke_report["direction_balance_recipe_contract"]["direction_flat_starvation_weight"] = 8.00
    smart_smoke_report["direction_balance_recipe_contract"]["direction_flat_starvation_min_label_rate"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["direction_flat_starvation_min_rows"] = 8
    smart_smoke_report["direction_balance_recipe_contract"]["direction_flat_starvation_pred_fraction"] = 0.50
    smart_smoke_report["direction_balance_recipe_contract"]["direction_flat_starvation_pred_floor"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["direction_flat_starvation_logit_margin"] = 0.10
    smart_smoke_report["direction_balance_recipe_contract"]["best_direction_balance_guard_ok"] = True
    smoke_path.write_text(json.dumps(smart_smoke_report), encoding="utf-8")
    foundation_smoke_path = tmp_path / "foundation_smoke_audit.json"
    foundation_smoke_path.write_text(json.dumps(_passing_smoke_audit()), encoding="utf-8")
    seq215_smoke_path = tmp_path / "seq215_smoke_audit.json"
    seq215_smoke_path.write_text(
        json.dumps(
            _passing_smoke_audit(
                contract_mode="challenger_seq215",
                dataset_dir=SEQ215_SMOKE_DATASET,
                signal_dim=215,
                specialists=SEQ215_SPECIALISTS,
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "gx1.scripts.verify_entry_candidate_readiness_v1.SMART_SEQ520_TRAINABILITY_READINESS_LATEST",
        trainability_path,
    )
    monkeypatch.setattr(
        "gx1.scripts.verify_entry_candidate_readiness_v1.SMOKE_BUNDLE_AUDIT_LATEST",
        foundation_smoke_path,
    )
    monkeypatch.setattr(
        "gx1.scripts.verify_entry_candidate_readiness_v1.CHALLENGER_SEQ215_SMOKE_BUNDLE_AUDIT_LATEST",
        seq215_smoke_path,
    )

    report = run(
        argparse.Namespace(
            audit_doc="/home/andre2/src/GX1_ENGINE/docs/ENTRY_FOUNDATION_AUDIT_20260628.md",
            smoke_bundle_audit_json=str(smoke_path),
            specialist_audit_json=str(specialist_path),
            contract_mode="smart_seq520_candidate",
            out_dir=str(tmp_path / "out_smart"),
            min_active_specialists=8,
            fail_on_not_ready=False,
            quiet=True,
        )
    )

    assert report["readiness_checks_pass"] is True
    assert report["training_allowed_by_contract"] is True
    assert report["decision"] == "READY_FOR_CANDIDATE_TRAINING_VEDTAK"
    assert report["candidate_training_allowed_with_explicit_vedtak"] is True
    assert report["next_required_gate"] == "scripts/entry_next_edge_control.sh candidate-train-smart --vedtak <id> then post-train replay gates"
    assert report["failures"] == []


def test_candidate_readiness_current_artifacts_are_not_ready(tmp_path: Path) -> None:
    report = run(
        argparse.Namespace(
            audit_doc="/home/andre2/src/GX1_ENGINE/docs/ENTRY_FOUNDATION_AUDIT_20260628.md",
            smoke_bundle_audit_json="/home/andre2/GX1_DATA/reports/entry_foundation_smoke_bundle_audit_20260628_v1/ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json",
            specialist_audit_json=None,
            contract_mode="foundation_seq146",
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
    assert failed
    readiness_blockers = {
        "foundation train-readiness is green",
        "smoke bundle audit PASS",
        "smoke bundle audit has zero failures",
        "smoke bundle audit is from actual train output, not sanity bundle",
        "smoke bundle audit was run with require_edge",
        "direction beats majority on all audited splits",
        "direction distribution covers active LONG/SHORT/FLAT classes",
        "direction context slices pass session/regime bucket diagnostics",
    }
    assert failed & readiness_blockers
    assert Path(report["json_path"]).exists()


def test_candidate_readiness_rejects_non_strict_edge_scope(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="NO_EDGE_FALLBACK"):
        run(
            argparse.Namespace(
                audit_doc="/home/andre2/src/GX1_ENGINE/docs/ENTRY_FOUNDATION_AUDIT_20260628.md",
                smoke_bundle_audit_json=str(tmp_path / "smoke.json"),
                specialist_audit_json=None,
                contract_mode="smart_seq520_candidate",
                edge_test_scope="smoke",
                out_dir=str(tmp_path / "out"),
                min_active_specialists=8,
                fail_on_not_ready=False,
                quiet=True,
            )
        )
