import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.features.entry_specialist_feature_groups_v1 import required_training_specialists_for_mode
from gx1.scripts.verify_entry_replay_readiness_v1 import (
    CHALLENGER_SEQ215_CANDIDATE_BUNDLE_AUDIT,
    CHALLENGER_SEQ215_CANDIDATE_READINESS_LATEST,
    CHALLENGER_SEQ215_REPLAY_DIR,
    CHALLENGER_SEQ215_SELECTIVE_EDGE_DIR,
    FOUNDATION_DATASET_DIR,
    SMART_SEQ520_CANDIDATE_BUNDLE_AUDIT,
    SMART_SEQ520_CANDIDATE_READINESS_LATEST,
    SMART_SEQ520_DATASET_DIR,
    SMART_SEQ520_REPLAY_DIR,
    SMART_SEQ520_SELECTIVE_EDGE_DIR,
    _candidate_bundle_audit_checks,
    _replay_checks,
    _selective_edge_checks,
    _smart_xau_pretrain_audit_checks,
    build_parser,
    run,
)
from gx1.scripts.verify_entry_training_readiness_v1 import EXPECTED_ACTIVE_TRAINING_HEADS, EXPECTED_BLOCKED_HEADS


def _specialist_snapshot(mode: str = "foundation_seq146") -> dict:
    expected = sorted(required_training_specialists_for_mode(mode))
    dim = {"foundation_seq146": 146, "challenger_seq215": 215, "smart_seq520_candidate": 520}[mode]
    return {
        "requested_contract_mode": mode,
        "observed_contract_mode": mode,
        "contract_mode_declared": True,
        "expected_signal_dim": dim,
        "bundle_seq_input_dim": dim,
        "bundle_snap_input_dim": dim,
        "specialist_fusion_enabled": True,
        "expected_specialists": expected,
        "observed_specialists": expected,
        "required_specialists_exact": True,
        "chart_geometry_present": "chart_geometry_encoder" in expected,
        "price_action_candle_present": "price_action_candle_encoder" in expected,
        "specialist_model_contract_valid": True,
        "specialist_model_contract_set_exact": True,
        "specialist_model_contract_owned_objectives_match": True,
        "specialist_model_contract_signal_families_match": True,
        "specialist_model_contract_support_heads_match": True,
        "specialist_model_contract_model_roles_match": True,
        "failures": [],
    }


def _selective_summary() -> dict:
    rows = []
    for model in ("candidate", "candidate_no_xgb"):
        for split in ("val", "test"):
            rows.append(
                {
                    "split": split,
                    "model": model,
                    "top5_all_mean_pnl_bps": 3.2,
                    "top10_all_mean_pnl_bps": 2.1,
                    "top5_all_direction_precision": 0.57,
                    "top10_all_direction_precision": 0.56,
                }
            )
    return {
        "decision": "PASS",
        "failures": [],
        "contract_mode": "foundation_seq146",
        "bundle_dir": "/tmp/candidate_bundle",
        "no_xgb_bundle_dir": "/tmp/candidate_no_xgb_bundle",
        "no_xgb_ablation": {
            "required": True,
            "mode": "bundle",
            "neutralize_signal_bridge": False,
            "neutralized_fields": [],
            "neutral_values": [],
        },
        "no_xgb_ablation_diagnostics": {
            "available": True,
            "candidate_model": "candidate",
            "no_xgb_model": "candidate_no_xgb",
            "splits": {
                "val": {
                    "rows": 10,
                    "comparable": True,
                    "time_match": True,
                    "max_abs_prob_delta": 0.2,
                    "max_abs_edge_score_delta": 0.1,
                    "trade_side_diff_count": 1,
                    "pred_direction_diff_count": 1,
                },
                "test": {
                    "rows": 10,
                    "comparable": True,
                    "time_match": True,
                    "max_abs_prob_delta": 0.2,
                    "max_abs_edge_score_delta": 0.1,
                    "trade_side_diff_count": 1,
                    "pred_direction_diff_count": 1,
                },
            },
        },
        "dataset_dir": str(FOUNDATION_DATASET_DIR),
        "bundle_seq_input_dim": 146,
        "bundle_snap_input_dim": 146,
        "bundle_specialist_contract": _specialist_snapshot("foundation_seq146"),
        "no_xgb_bundle_specialist_contract": _specialist_snapshot("foundation_seq146"),
        "input_bridge_contract": {
            "splits": {
                "val": {"neutral_xgb_bridge": False},
                "test": {"neutral_xgb_bridge": False},
            }
        },
        "splits": ["val", "test"],
        "summaries": rows,
    }


def _selective_metrics() -> pd.DataFrame:
    rows = []
    for split, session in (("val", "EU"), ("test", "US")):
        for top_frac in (0.05, 0.10):
            for group in ("ALL", f"session={session}", "side=LONG", "vol_regime=2"):
                rows.append(
                    {
                        "split": split,
                        "model": "candidate",
                        "scope": "top_score",
                        "top_frac": top_frac,
                        "group": group,
                        "n": 31 if split == "test" else 25,
                        "mean_pnl_bps": 2.5 if split == "test" else 3.0,
                        "win_rate": 0.55 if split == "test" else 0.56,
                        "direction_precision": 0.56 if split == "test" else 0.57,
                    }
                )
    return pd.DataFrame(rows)


def _replay_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "scope": ["aggregate"],
            "policy_id": ["candidate_top5"],
            "n_trades": [42],
            "net_sum_bps": [250.0],
            "win_rate": [0.57],
            "profit_factor": [1.35],
            "max_drawdown_bps": [120.0],
            "max_loss_bps": [-18.0],
        }
    )


def _monthly_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "policy_id": ["candidate_top5", "candidate_top5"],
            "month": ["2026-01", "2026-02"],
            "net_sum_bps": [80.0, 120.0],
        }
    )


def _replay_trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": ["2026-01-03T08:00:00Z", "2026-02-03T10:00:00Z"],
            "policy_id": ["candidate_top5", "candidate_top5"],
            "session": ["EU", "US"],
            "side": ["LONG", "SHORT"],
            "score": [0.8, 0.7],
            "p_long": [0.82, 0.10],
            "p_short": [0.10, 0.78],
            "p_flat": [0.08, 0.12],
            "net_pnl_bps": [120.0, 90.0],
            "mfe_bps": [140.0, 110.0],
            "mae_bps": [10.0, 12.0],
            "held_bars": [12, 10],
        }
    )


def _candidate_bundle_audit() -> dict:
    split = {
        "rows": 128,
        "direction": {"beats_majority_baseline": True},
        "direction_distribution_contract": {"decision": "PASS", "failures": []},
        "direction_slice_contract": {"decision": "PASS", "failures": [], "audited_slice_count": 2},
        "path_quality": {"pred_vs_target_spearman": 0.25},
        "bad_path": {"prob_vs_path_quality_spearman": -0.25},
        "specialist_gate": {
            "mean_weight": {
                "structure_swing_encoder": 0.18,
                "smc_liquidity_encoder": 0.17,
                "trend_ema_encoder": 0.16,
                "vol_compression_encoder": 0.16,
                "momentum_flow_encoder": 0.17,
                "session_regime_encoder": 0.16,
            }
        },
    }
    return {
        "decision": "PASS",
        "failures": [],
        "specialist_contract_mode": "foundation_seq146",
        "dataset_dir": str(FOUNDATION_DATASET_DIR),
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
        "path_calibration_recipe_contract": {
            "decision": "PASS",
            "failures": [],
            "active_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
            "path_quality_active": True,
            "bad_path_active": True,
            "path_quality_rank_full_batch": True,
            "path_quality_rank_weight": 2.0,
            "path_quality_rank_margin": 0.25,
            "path_quality_rank_quantile": 0.25,
            "bad_path_quality_rank_weight": 2.0,
            "bad_path_quality_rank_margin": 0.25,
            "bad_path_quality_rank_quantile": 0.25,
        },
        "direction_balance_recipe_contract": {
            "decision": "PASS",
            "failures": [],
            "active_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
            "direction_active": True,
            "pred_balance_alpha": 0.05,
            "pred_balance_target": "label",
            "pred_balance_class_weights": [1.0, 1.0, 1.0],
            "direction_ce_scale": 1.30,
            "ckpt_monitor": "dir_acc",
        },
        "tail_direction_recipe_contract": {
            "decision": "PASS",
            "failures": [],
            "active_heads": list(EXPECTED_ACTIVE_TRAINING_HEADS),
            "direction_active": True,
            "tail_direction_ce_weight": 0.35,
            "tail_direction_quality_quantile": 0.70,
            "tail_direction_min_batch": 8,
            "tail_direction_mask": "directional_tradable_clean_path_top_quality",
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
            "smoke_edge_required_specialists_exact": True,
            "smoke_edge_specialist_groups_exact": True,
            "smoke_edge_specialist_model_contract_valid": True,
            "smoke_edge_specialist_model_contract_set_exact": True,
            "smoke_edge_specialist_model_contract_owned_objectives_match": True,
            "smoke_dataset_audit_provenance_all_artifacts_present": True,
            "smoke_dataset_audit_provenance_all_artifact_hashes_present": True,
            "smoke_edge_worktree_critical_gate_review_ok": True,
        },
        "bundle_summary": {
            "sanity_bundle": False,
            "contract_mode": "foundation_seq146",
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
        "splits": {"val": split, "test": split},
    }


def _smart_candidate_bundle_audit() -> dict:
    report = _candidate_bundle_audit()
    specialists = list(required_training_specialists_for_mode("smart_seq520_candidate"))
    mean_weight = {name: 1.0 / float(len(specialists)) for name in specialists}
    report["head_contract"]["active_training_heads"] = [
        *list(EXPECTED_ACTIVE_TRAINING_HEADS),
        "trade_side_hierarchy",
        "trendline_rail",
        "side_validity",
    ]
    report["specialist_contract_mode"] = "smart_seq520_candidate"
    report["dataset_dir"] = str(SMART_SEQ520_DATASET_DIR)
    report["required_training_specialists"] = specialists
    report["min_active_specialists"] = len(specialists)
    report["bundle_summary"]["contract_mode"] = "smart_seq520_candidate"
    report["bundle_summary"]["seq_input_dim"] = 520
    report["bundle_summary"]["snap_input_dim"] = 520
    report["bundle_summary"]["specialist_groups"] = specialists
    for split in report["splits"].values():
        split["specialist_gate"]["mean_weight"] = dict(mean_weight)
    return report


def test_selective_edge_checks_pass_on_candidate_contract() -> None:
    checks = _selective_edge_checks(
        _selective_summary(),
        _selective_metrics(),
        model_name="candidate",
        min_top5_mean_pnl_bps=0.0,
        min_top10_mean_pnl_bps=0.0,
        require_no_xgb_ablation=True,
        expected_bundle_dir="/tmp/candidate_bundle",
    )

    assert all(check["ok"] for check in checks)


def test_selective_edge_checks_accept_smart_dataset_contract() -> None:
    summary = _selective_summary()
    summary["contract_mode"] = "smart_seq520_candidate"
    summary["dataset_dir"] = str(SMART_SEQ520_DATASET_DIR)
    summary["bundle_seq_input_dim"] = 520
    summary["bundle_snap_input_dim"] = 520
    summary["selection_score_mode"] = "expected_utility"
    summary["selection_score_threshold"] = 0.0
    summary["bundle_specialist_contract"] = _specialist_snapshot("smart_seq520_candidate")
    summary["no_xgb_bundle_specialist_contract"] = _specialist_snapshot("smart_seq520_candidate")

    checks = _selective_edge_checks(
        summary,
        _selective_metrics(),
        model_name="candidate",
        min_top5_mean_pnl_bps=0.0,
        min_top10_mean_pnl_bps=0.0,
        require_no_xgb_ablation=True,
        expected_bundle_dir="/tmp/candidate_bundle",
        expected_dataset_dir=SMART_SEQ520_DATASET_DIR,
        expected_contract_mode="smart_seq520_candidate",
    )

    assert all(check["ok"] for check in checks)


def test_selective_edge_checks_reject_smart_edge_score_selection_mode() -> None:
    summary = _selective_summary()
    summary["contract_mode"] = "smart_seq520_candidate"
    summary["dataset_dir"] = str(SMART_SEQ520_DATASET_DIR)
    summary["bundle_seq_input_dim"] = 520
    summary["bundle_snap_input_dim"] = 520
    summary["bundle_specialist_contract"] = _specialist_snapshot("smart_seq520_candidate")
    summary["no_xgb_bundle_specialist_contract"] = _specialist_snapshot("smart_seq520_candidate")

    checks = _selective_edge_checks(
        summary,
        _selective_metrics(),
        model_name="candidate",
        min_top5_mean_pnl_bps=0.0,
        min_top10_mean_pnl_bps=0.0,
        require_no_xgb_ablation=True,
        expected_bundle_dir="/tmp/candidate_bundle",
        expected_dataset_dir=SMART_SEQ520_DATASET_DIR,
        expected_contract_mode="smart_seq520_candidate",
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "smart selective-edge uses expected-utility selection mode" in failed


def test_smart_xau_pretrain_audit_gate_fails_when_missing(tmp_path: Path) -> None:
    checks = _smart_xau_pretrain_audit_checks(
        tmp_path / "missing.json",
        {},
        expected_dataset_dir=SMART_SEQ520_DATASET_DIR,
    )

    assert any(check["name"] == "smart XAU pretrain audit artifact exists" and not check["ok"] for check in checks)


def test_selective_edge_checks_reject_low_selected_tail_direction_precision() -> None:
    summary = _selective_summary()
    for row in summary["summaries"]:
        if row["model"] == "candidate" and row["split"] == "test":
            row["top5_all_direction_precision"] = 0.49

    checks = _selective_edge_checks(
        summary,
        _selective_metrics(),
        model_name="candidate",
        min_top5_mean_pnl_bps=0.0,
        min_top10_mean_pnl_bps=0.0,
        require_no_xgb_ablation=True,
        expected_bundle_dir="/tmp/candidate_bundle",
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate top5 selected-tail direction precision clears threshold on val/test" in failed


def test_selective_edge_checks_reject_low_selected_tail_direction_slice() -> None:
    metrics = _selective_metrics()
    metrics.loc[
        (metrics["split"] == "test")
        & (metrics["group"] == "session=US")
        & (metrics["top_frac"] == 0.05),
        "direction_precision",
    ] = 0.49

    checks = _selective_edge_checks(
        _selective_summary(),
        metrics,
        model_name="candidate",
        min_top5_mean_pnl_bps=0.0,
        min_top10_mean_pnl_bps=0.0,
        require_no_xgb_ablation=True,
        expected_bundle_dir="/tmp/candidate_bundle",
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "selective-edge selected-tail direction slices clear threshold" in failed


def test_selective_edge_checks_reject_mismatched_candidate_bundle() -> None:
    checks = _selective_edge_checks(
        _selective_summary(),
        _selective_metrics(),
        model_name="candidate",
        min_top5_mean_pnl_bps=0.0,
        min_top10_mean_pnl_bps=0.0,
        require_no_xgb_ablation=True,
        expected_bundle_dir="/tmp/other_candidate_bundle",
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "selective-edge summary matches candidate bundle audit bundle" in failed


def test_selective_edge_checks_reject_seq215_summary_when_replay_contract_defaults_seq146() -> None:
    summary = _selective_summary()
    summary["contract_mode"] = "challenger_seq215"
    summary["bundle_seq_input_dim"] = 215
    summary["bundle_snap_input_dim"] = 215

    checks = _selective_edge_checks(
        summary,
        _selective_metrics(),
        model_name="candidate",
        min_top5_mean_pnl_bps=0.0,
        min_top10_mean_pnl_bps=0.0,
        require_no_xgb_ablation=True,
        expected_bundle_dir="/tmp/candidate_bundle",
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "selective-edge summary contract mode matches candidate bundle audit" in failed
    assert "selective-edge summary input dimensions match contract mode" in failed


def test_selective_edge_checks_reject_same_bundle_without_neutralized_ablation() -> None:
    summary = _selective_summary()
    summary["no_xgb_bundle_dir"] = summary["bundle_dir"]
    summary["no_xgb_ablation"] = {
        "required": True,
        "mode": "bundle",
        "neutralize_signal_bridge": False,
        "neutralized_fields": [],
        "neutral_values": [],
    }
    checks = _selective_edge_checks(
        summary,
        _selective_metrics(),
        model_name="candidate",
        min_top5_mean_pnl_bps=0.0,
        min_top10_mean_pnl_bps=0.0,
        require_no_xgb_ablation=True,
        expected_bundle_dir="/tmp/candidate_bundle",
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "selective-edge no-XGB ablation provenance is explicit" in failed


def test_selective_edge_checks_accept_same_bundle_with_neutralized_ablation() -> None:
    summary = _selective_summary()
    summary["no_xgb_bundle_dir"] = summary["bundle_dir"]
    summary["no_xgb_ablation"] = {
        "required": True,
        "mode": "neutralize_signal_bridge",
        "neutralize_signal_bridge": True,
        "neutralized_fields": [f"field_{idx}" for idx in range(7)],
        "neutral_values": [0.0] * 7,
    }
    summary["no_xgb_ablation_diagnostics"] = {
        "available": True,
        "candidate_model": "candidate",
        "no_xgb_model": "candidate_no_xgb",
        "splits": {
            "val": {
                "rows": 10,
                "comparable": True,
                "time_match": True,
                "max_abs_prob_delta": 0.0,
                "max_abs_edge_score_delta": 0.0,
                "trade_side_diff_count": 0,
                "pred_direction_diff_count": 0,
            },
            "test": {
                "rows": 10,
                "comparable": True,
                "time_match": True,
                "max_abs_prob_delta": 0.0,
                "max_abs_edge_score_delta": 0.0,
                "trade_side_diff_count": 0,
                "pred_direction_diff_count": 0,
            },
        },
    }
    summary["input_bridge_contract"] = {
        "splits": {
            "val": {"neutral_xgb_bridge": True},
            "test": {"neutral_xgb_bridge": True},
        }
    }
    checks = _selective_edge_checks(
        summary,
        _selective_metrics(),
        model_name="candidate",
        min_top5_mean_pnl_bps=0.0,
        min_top10_mean_pnl_bps=0.0,
        require_no_xgb_ablation=True,
        expected_bundle_dir="/tmp/candidate_bundle",
    )

    assert all(check["ok"] for check in checks)


def test_selective_edge_checks_reject_dead_neutralized_ablation_without_neutral_input() -> None:
    summary = _selective_summary()
    summary["no_xgb_bundle_dir"] = summary["bundle_dir"]
    summary["no_xgb_ablation"] = {
        "required": True,
        "mode": "neutralize_signal_bridge",
        "neutralize_signal_bridge": True,
        "neutralized_fields": [f"field_{idx}" for idx in range(7)],
        "neutral_values": [0.0] * 7,
    }
    for row in summary["no_xgb_ablation_diagnostics"]["splits"].values():
        row["max_abs_prob_delta"] = 0.0
        row["max_abs_edge_score_delta"] = 0.0
        row["trade_side_diff_count"] = 0
        row["pred_direction_diff_count"] = 0
    checks = _selective_edge_checks(
        summary,
        _selective_metrics(),
        model_name="candidate",
        min_top5_mean_pnl_bps=0.0,
        min_top10_mean_pnl_bps=0.0,
        require_no_xgb_ablation=True,
        expected_bundle_dir="/tmp/candidate_bundle",
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "selective-edge no-XGB ablation is live or input bridge already neutral" in failed


def test_candidate_bundle_audit_checks_reject_partial_active_head_contract(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _candidate_bundle_audit()
    report["head_contract"]["active_training_heads"] = ["direction", "path_quality", "tf_agreement"]

    checks = _candidate_bundle_audit_checks(audit_path, report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate bundle head contract PASS" in failed


def test_candidate_bundle_audit_checks_reject_missing_pretrain_provenance(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _candidate_bundle_audit()
    report["pretrain_manifest_contract"]["feature_source_field_liveness_all_live"] = False
    report["pretrain_manifest_contract"]["specialist_active_heads_match_target"] = False
    report["pretrain_manifest_contract"]["smoke_dataset_audit_provenance_all_artifacts_present"] = False
    report["pretrain_manifest_contract"]["smoke_edge_worktree_critical_gate_review_ok"] = False

    checks = _candidate_bundle_audit_checks(audit_path, report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate bundle audit validated pre-train manifest provenance" in failed


def test_candidate_bundle_audit_checks_reject_missing_pretrain_exact_specialist_set(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _candidate_bundle_audit()
    report["pretrain_manifest_contract"]["specialist_required_training_set_exact"] = False
    report["pretrain_manifest_contract"]["specialist_trainable_set_exact"] = False
    report["pretrain_manifest_contract"]["smoke_edge_required_specialists_exact"] = False
    report["pretrain_manifest_contract"]["smoke_edge_specialist_groups_exact"] = False

    checks = _candidate_bundle_audit_checks(audit_path, report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate bundle audit validated pre-train manifest provenance" in failed


def test_candidate_bundle_audit_checks_reject_missing_specialist_model_contract(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _candidate_bundle_audit()
    report["pretrain_manifest_contract"]["specialist_model_contract_valid"] = False
    report["pretrain_manifest_contract"]["specialist_model_contract_set_exact"] = False
    report["pretrain_manifest_contract"]["specialist_model_contract_owned_objectives_match"] = False
    report["pretrain_manifest_contract"]["smoke_edge_specialist_model_contract_valid"] = False
    report["pretrain_manifest_contract"]["smoke_edge_specialist_model_contract_set_exact"] = False
    report["pretrain_manifest_contract"]["smoke_edge_specialist_model_contract_owned_objectives_match"] = False

    checks = _candidate_bundle_audit_checks(audit_path, report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate bundle audit validated pre-train manifest provenance" in failed


def test_candidate_bundle_audit_checks_reject_missing_bundle_specialist_model_contract(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _candidate_bundle_audit()
    report["bundle_summary"]["specialist_model_contract_valid"] = False
    report["bundle_specialist_model_contract"]["owned_objectives_match"] = False
    report["bundle_specialist_model_contract"]["failures"] = ["owned objectives mismatch"]

    checks = _candidate_bundle_audit_checks(audit_path, report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate bundle specialist model contract is preserved in bundle metadata" in failed


def test_candidate_bundle_audit_checks_reject_missing_path_calibration_recipe(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _candidate_bundle_audit()
    report["path_calibration_recipe_contract"] = {
        "decision": "FAIL",
        "path_quality_active": True,
        "path_quality_rank_full_batch": False,
        "path_quality_rank_weight": 0.0,
        "failures": ["path_quality active head requires positive path_quality_rank_weight"],
    }

    checks = _candidate_bundle_audit_checks(audit_path, report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate bundle path calibration recipe contract PASS" in failed


def test_candidate_bundle_audit_checks_reject_missing_require_edge(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _candidate_bundle_audit()
    report["require_edge"] = False

    checks = _candidate_bundle_audit_checks(audit_path, report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate bundle audit was run with require_edge" in failed


def test_candidate_bundle_audit_checks_reject_candidate_direction_collapse(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _candidate_bundle_audit()
    report["splits"]["test"]["direction_distribution_contract"] = {
        "decision": "FAIL",
        "failures": ["FLAT prediction_rate=0.030000 below required 0.120000"],
    }

    checks = _candidate_bundle_audit_checks(audit_path, report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate bundle direction distribution covers active classes" in failed


def test_candidate_bundle_audit_checks_reject_wrong_signed_candidate_path_heads(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _candidate_bundle_audit()
    report["splits"]["val"]["path_quality"]["pred_vs_target_spearman"] = -0.10
    report["splits"]["test"]["bad_path"]["prob_vs_path_quality_spearman"] = 0.10

    checks = _candidate_bundle_audit_checks(audit_path, report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate bundle path_quality ranks realized path quality positively" in failed
    assert "candidate bundle bad_path ranks worse path quality higher" in failed


def test_candidate_bundle_audit_checks_reject_missing_direction_balance_recipe(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _candidate_bundle_audit()
    report["direction_balance_recipe_contract"] = {
        "decision": "FAIL",
        "direction_active": True,
        "pred_balance_alpha": 0.0,
        "pred_balance_target": "uniform",
        "direction_ce_scale": 1.30,
        "ckpt_monitor": "val_loss",
        "failures": ["direction active head requires pred_balance_target=label"],
    }

    checks = _candidate_bundle_audit_checks(audit_path, report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate bundle direction balance recipe contract PASS" in failed


def test_candidate_bundle_audit_checks_reject_smart_weak_flat_repair_recipe(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _smart_candidate_bundle_audit()

    checks = _candidate_bundle_audit_checks(
        audit_path,
        report,
        contract_mode="smart_seq520_candidate",
        expected_dataset_dir=SMART_SEQ520_DATASET_DIR,
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate bundle direction balance recipe contract PASS" in failed


def test_candidate_bundle_audit_checks_accept_smart_flat_repair_recipe(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _smart_candidate_bundle_audit()
    report["direction_balance_recipe_contract"].update(
        {
            "pred_balance_alpha": 0.50,
            "pred_balance_class_weights": [1.0, 1.0, 4.0],
                "direction_ce_scale": 2.00,
                "hierarchical_entry_heads_enabled": True,
                "side_validity_head_enabled": True,
                "hier_side_validity_weight": 1.50,
                "hier_side_validity_min_utility_bps": 15.0,
                "hier_side_validity_pos_weight_cap": 8.0,
                "trendline_rail_head_enabled": True,
            "trendline_rail_aux_weight": 1.00,
            "trendline_rail_wrong_side_weight": 1.50,
            "hier_legacy_ce_mult": 1.00,
            "anchor_gate_enabled": True,
            "anchor_gate_init": 0.0,
            "ckpt_class_balance_guard_weight": 0.50,
            "ckpt_class_balance_min_pred_to_label": 0.35,
            "ckpt_class_balance_min_pred_rate": 0.05,
            "direction_min_pred_rate_loss_weight": 2.50,
            "direction_min_pred_rate_fraction": 0.50,
            "direction_min_pred_rate_floor": 0.05,
            "direction_min_pred_rate_softmax_temperature": 0.20,
            "direction_vs_flat_margin_weight": 4.00,
            "direction_vs_flat_margin": 0.10,
            "direction_utility_margin_weight": 4.00,
            "direction_utility_min_gap_bps": 15.0,
            "direction_utility_logit_margin": 0.10,
            "direction_side_utility_conviction_weight": 6.00,
            "direction_side_utility_conviction_min_gap_bps": 15.0,
            "direction_side_utility_conviction_logit_margin": 0.10,
            "direction_utility_trade_conviction_weight": 8.00,
            "direction_utility_trade_conviction_min_gap_bps": 15.0,
            "direction_utility_trade_conviction_min_utility_bps": 0.0,
            "direction_utility_trade_conviction_max_bad_path": 0.50,
            "direction_utility_trade_conviction_logit_margin": 0.10,
            "direction_utility_triad_ce_weight": 8.00,
            "direction_utility_triad_ce_min_gap_bps": 15.0,
            "direction_utility_triad_ce_min_utility_bps": 0.0,
            "direction_utility_triad_ce_max_bad_path": 0.50,
            "direction_utility_triad_ce_class_weight_cap": 4.0,
            "direction_flat_starvation_weight": 8.00,
            "direction_flat_starvation_min_label_rate": 0.10,
            "direction_flat_starvation_min_rows": 8,
            "direction_flat_starvation_pred_fraction": 0.50,
            "direction_flat_starvation_pred_floor": 0.10,
            "direction_flat_starvation_logit_margin": 0.10,
            "best_direction_balance_guard_ok": True,
        }
    )

    checks = _candidate_bundle_audit_checks(
        audit_path,
        report,
        contract_mode="smart_seq520_candidate",
        expected_dataset_dir=SMART_SEQ520_DATASET_DIR,
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate bundle direction balance recipe contract PASS" not in failed


def test_candidate_bundle_audit_checks_reject_missing_tail_direction_recipe(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _candidate_bundle_audit()
    report["tail_direction_recipe_contract"] = {
        "decision": "FAIL",
        "direction_active": True,
        "tail_direction_ce_weight": 0.0,
        "tail_direction_quality_quantile": 0.70,
        "tail_direction_min_batch": 8,
        "tail_direction_mask": "directional_tradable_clean_path_top_quality",
        "failures": ["direction active head requires positive tail_direction_ce_weight"],
    }

    checks = _candidate_bundle_audit_checks(audit_path, report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate bundle tail direction recipe contract PASS" in failed


def test_candidate_bundle_audit_checks_reject_extra_specialist_group(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _candidate_bundle_audit()
    report["required_training_specialists"].append("price_action_candle_encoder")
    report["bundle_summary"]["specialist_groups"].append("price_action_candle_encoder")
    for split in report["splits"].values():
        split["specialist_gate"]["mean_weight"]["price_action_candle_encoder"] = 0.05

    checks = _candidate_bundle_audit_checks(audit_path, report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate bundle has exact specialist groups" in failed
    assert "candidate bundle audit was run with specialist-fusion gate contract" in failed


def test_candidate_bundle_audit_checks_reject_seq215_bundle_under_default_seq146_contract(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _candidate_bundle_audit()
    report["specialist_contract_mode"] = "challenger_seq215"
    report["bundle_summary"]["contract_mode"] = "challenger_seq215"
    report["bundle_summary"]["seq_input_dim"] = 215
    report["bundle_summary"]["snap_input_dim"] = 215

    checks = _candidate_bundle_audit_checks(audit_path, report)
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "candidate bundle audit contract mode matches requested replay contract" in failed
    assert "candidate bundle input dimensions match contract mode" in failed


def test_candidate_bundle_audit_checks_accept_seq215_with_explicit_contract(tmp_path: Path) -> None:
    audit_path = tmp_path / "candidate_audit.json"
    audit_path.write_text("{}", encoding="utf-8")
    report = _candidate_bundle_audit()
    report["specialist_contract_mode"] = "challenger_seq215"
    report["required_training_specialists"].extend(["chart_geometry_encoder", "price_action_candle_encoder"])
    report["min_active_specialists"] = 8
    report["bundle_summary"]["contract_mode"] = "challenger_seq215"
    report["bundle_summary"]["seq_input_dim"] = 215
    report["bundle_summary"]["snap_input_dim"] = 215
    report["bundle_summary"]["specialist_groups"].extend(["chart_geometry_encoder", "price_action_candle_encoder"])
    for split in report["splits"].values():
        split["specialist_gate"]["mean_weight"]["chart_geometry_encoder"] = 0.05
        split["specialist_gate"]["mean_weight"]["price_action_candle_encoder"] = 0.05

    checks = _candidate_bundle_audit_checks(audit_path, report, contract_mode="challenger_seq215")

    assert all(check["ok"] for check in checks)


def test_replay_checks_pass_on_positive_stable_replay(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    manifest = {
        "decision": "PASS",
        "failures": [],
        "replay_identity_contract": {
            "ready": True,
            "contract_mode": "foundation_seq146",
            "candidate_bundle_dir": "/tmp/candidate_bundle",
            "candidate_specialist_contract": {"ready": True},
            "selective_edge_specialist_contract": {"ready": True},
        },
    }

    checks = _replay_checks(
        replay_dir,
        manifest,
        _replay_metrics(),
        _monthly_metrics(),
        _replay_trades(),
        min_net_sum_bps=0.0,
        min_profit_factor=1.05,
        max_drawdown_bps=650.0,
        expected_candidate_bundle_dir="/tmp/candidate_bundle",
    )

    assert all(check["ok"] for check in checks)


def test_replay_checks_reject_missing_explicit_replay_manifest(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()

    checks = _replay_checks(
        replay_dir,
        {},
        _replay_metrics(),
        _monthly_metrics(),
        _replay_trades(),
        min_net_sum_bps=0.0,
        min_profit_factor=1.05,
        max_drawdown_bps=650.0,
        expected_candidate_bundle_dir="/tmp/candidate_bundle",
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "offline replay manifest PASS" in failed
    assert "offline replay identity contract ready" in failed


def test_replay_checks_reject_missing_specialist_contract_identity(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    manifest = {
        "decision": "PASS",
        "failures": [],
        "replay_identity_contract": {
            "ready": True,
            "contract_mode": "foundation_seq146",
            "candidate_bundle_dir": "/tmp/candidate_bundle",
        },
    }

    checks = _replay_checks(
        replay_dir,
        manifest,
        _replay_metrics(),
        _monthly_metrics(),
        _replay_trades(),
        min_net_sum_bps=0.0,
        min_profit_factor=1.05,
        max_drawdown_bps=650.0,
        expected_candidate_bundle_dir="/tmp/candidate_bundle",
    )
    failed = {check["name"] for check in checks if not check["ok"]}

    assert "offline replay identity preserves candidate specialist contract" in failed
    assert "offline replay identity preserves selective-edge specialist contract" in failed


def test_candidate_bundle_audit_checks_pass_on_strict_candidate_contract(tmp_path: Path) -> None:
    path = tmp_path / "candidate_audit.json"
    path.write_text(json.dumps(_candidate_bundle_audit()), encoding="utf-8")

    checks = _candidate_bundle_audit_checks(path, json.loads(path.read_text(encoding="utf-8")))

    assert all(check["ok"] for check in checks)


def test_candidate_bundle_audit_accepts_resolved_foundation_dataset_path(tmp_path: Path) -> None:
    report = _candidate_bundle_audit()
    report["dataset_dir"] = str(Path(report["dataset_dir"]).resolve(strict=False))
    path = tmp_path / "candidate_audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    checks = {
        check["name"]: check
        for check in _candidate_bundle_audit_checks(
            path,
            report,
            expected_dataset_dir=Path(report["dataset_dir"]),
        )
    }

    assert checks["candidate bundle audit used expected dataset"]["ok"] is True


def test_replay_readiness_current_artifacts_roundtrip_report_contract(tmp_path: Path) -> None:
    report = run(
        argparse.Namespace(
            candidate_readiness_json="/home/andre2/GX1_DATA/reports/entry_candidate_readiness_20260628_v1/ENTRY_CANDIDATE_READINESS_latest.json",
            candidate_bundle_audit_json="/home/andre2/GX1_DATA/reports/entry_candidate_bundle_audit_20260628_v1/ENTRY_FOUNDATION_SMOKE_BUNDLE_AUDIT_latest.json",
            selective_edge_summary_json="/home/andre2/GX1_DATA/reports/entry_candidate_selective_edge_20260628_v1/summary.json",
            selective_edge_metrics_csv="/home/andre2/GX1_DATA/reports/entry_candidate_selective_edge_20260628_v1/selective_edge_metrics.csv",
            replay_dir="/home/andre2/GX1_DATA/reports/entry_candidate_replay_20260628_v1",
            out_dir=str(tmp_path),
            model_name="candidate",
            min_top5_mean_pnl_bps=0.0,
            min_top10_mean_pnl_bps=0.0,
            min_replay_net_sum_bps=0.0,
            min_profit_factor=1.05,
            max_abs_drawdown_bps=650.0,
            require_no_xgb_ablation=True,
            fail_on_not_ready=False,
            quiet=True,
        )
    )

    assert report["decision"] in {"NOT_READY_FOR_IQL_DISTILLATION", "READY_FOR_IQL_DISTILLATION_VEDTAK"}
    assert report["iql_distillation_allowed_with_explicit_vedtak"] is (
        report["decision"] == "READY_FOR_IQL_DISTILLATION_VEDTAK"
    )
    assert report["promotion_shadow_live_allowed"] is False
    assert set(report["artifact_fingerprints"]) == set(report["artifacts"])
    assert any(gate["name"] == "artifact_provenance" for gate in report["gates"])
    failed = {failure["check"] for failure in report["failures"]}
    if report["decision"] == "NOT_READY_FOR_IQL_DISTILLATION":
        assert failed
        assert {
            "candidate-readiness is green",
            "candidate bundle path calibration recipe contract PASS",
            "selective-edge summary has val/test",
            "selective-edge summary input dimensions match contract mode",
            "offline replay dir exists",
        } & failed
    else:
        assert not failed
    assert Path(report["json_path"]).exists()
    assert json.loads(Path(report["json_path"]).read_text())["decision"] == report["decision"]


def test_replay_readiness_parser_defaults_to_infer_contract_mode() -> None:
    args = build_parser().parse_args([])

    assert args.contract_mode is None


def test_replay_readiness_challenger_seq215_rewrites_default_contract_paths(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["--challenger-seq215", "--quiet", "--no-fail-on-not-ready", "--out-dir", str(tmp_path)]
    )
    report = run(args)

    assert report["contract_mode"] == "challenger_seq215"
    assert report["candidate_readiness_json"] == str(CHALLENGER_SEQ215_CANDIDATE_READINESS_LATEST.resolve())
    assert report["candidate_bundle_audit_json"] == str(CHALLENGER_SEQ215_CANDIDATE_BUNDLE_AUDIT.resolve())
    assert report["selective_edge_summary_json"] == str((CHALLENGER_SEQ215_SELECTIVE_EDGE_DIR / "summary.json").resolve())
    assert report["selective_edge_metrics_csv"] == str(
        (CHALLENGER_SEQ215_SELECTIVE_EDGE_DIR / "selective_edge_metrics.csv").resolve()
    )
    assert report["replay_dir"] == str(CHALLENGER_SEQ215_REPLAY_DIR.resolve())
    assert Path(report["json_path"]).parent == tmp_path.resolve()
    assert report["decision"] == "NOT_READY_FOR_IQL_DISTILLATION"
    assert report["promotion_shadow_live_allowed"] is False


def test_replay_readiness_smart_seq520_rewrites_default_contract_paths(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        ["--smart-seq520", "--quiet", "--no-fail-on-not-ready", "--out-dir", str(tmp_path)]
    )
    report = run(args)

    assert report["contract_mode"] == "smart_seq520_candidate"
    assert report["candidate_readiness_json"] == str(SMART_SEQ520_CANDIDATE_READINESS_LATEST.resolve())
    assert report["candidate_bundle_audit_json"] == str(SMART_SEQ520_CANDIDATE_BUNDLE_AUDIT.resolve())
    assert report["selective_edge_summary_json"] == str((SMART_SEQ520_SELECTIVE_EDGE_DIR / "summary.json").resolve())
    assert report["selective_edge_metrics_csv"] == str(
        (SMART_SEQ520_SELECTIVE_EDGE_DIR / "selective_edge_metrics.csv").resolve()
    )
    assert report["replay_dir"] == str(SMART_SEQ520_REPLAY_DIR.resolve())
    assert report["promotion_shadow_live_allowed"] is False
