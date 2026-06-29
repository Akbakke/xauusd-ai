import argparse
import json
from pathlib import Path

import pandas as pd

from gx1.scripts.verify_entry_replay_readiness_v1 import (
    _candidate_bundle_audit_checks,
    _replay_checks,
    _selective_edge_checks,
    run,
)
from gx1.scripts.verify_entry_training_readiness_v1 import EXPECTED_ACTIVE_TRAINING_HEADS, EXPECTED_BLOCKED_HEADS


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
                }
            )
    return {
        "decision": "PASS",
        "failures": [],
        "bundle_dir": "/tmp/candidate_bundle",
        "no_xgb_bundle_dir": "/tmp/candidate_no_xgb_bundle",
        "dataset_dir": "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_foundation_seq146_neutral",
        "splits": ["val", "test"],
        "summaries": rows,
    }


def _selective_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["val", "test"],
            "model": ["candidate", "candidate"],
            "scope": ["top_score", "top_score"],
            "top_frac": [0.05, 0.05],
            "group": ["session=EU", "session=US"],
            "n": [25, 31],
            "mean_pnl_bps": [3.0, 2.5],
            "win_rate": [0.56, 0.55],
            "direction_precision": [0.57, 0.56],
        }
    )


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
        "dataset_dir": "/home/andre2/GX1_DATA/runs/FASE2B_REGIME_V4_20260605/v10_6yr_rebuild_20260628_foundation_seq146/v10_dataset_foundation_seq146_neutral",
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


def test_replay_checks_pass_on_positive_stable_replay(tmp_path: Path) -> None:
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    manifest = {
        "decision": "PASS",
        "failures": [],
        "replay_identity_contract": {
            "ready": True,
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

    assert all(check["ok"] for check in checks)


def test_candidate_bundle_audit_checks_pass_on_strict_candidate_contract(tmp_path: Path) -> None:
    path = tmp_path / "candidate_audit.json"
    path.write_text(json.dumps(_candidate_bundle_audit()), encoding="utf-8")

    checks = _candidate_bundle_audit_checks(path, json.loads(path.read_text(encoding="utf-8")))

    assert all(check["ok"] for check in checks)


def test_replay_readiness_current_artifacts_are_not_ready(tmp_path: Path) -> None:
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

    assert report["decision"] == "NOT_READY_FOR_IQL_DISTILLATION"
    assert report["iql_distillation_allowed_with_explicit_vedtak"] is False
    assert report["promotion_shadow_live_allowed"] is False
    assert set(report["artifact_fingerprints"]) == set(report["artifacts"])
    assert any(gate["name"] == "artifact_provenance" for gate in report["gates"])
    failed = {failure["check"] for failure in report["failures"]}
    assert "candidate-readiness is green" in failed
    assert "candidate bundle audit exists" in failed
    assert "selective-edge summary has val/test" in failed
    assert "offline replay dir exists" in failed
    assert Path(report["json_path"]).exists()
    assert json.loads(Path(report["json_path"]).read_text())["decision"] == "NOT_READY_FOR_IQL_DISTILLATION"
