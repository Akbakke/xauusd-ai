import argparse
import copy
import json
from pathlib import Path

import pandas as pd

from gx1.scripts import verify_entry_smart_ablation_replay_matrix_v1 as matrix
from gx1.scripts.materialize_entry_smart_ablation_replay_plan_gate_v1 import build_required_ablation_plan


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _candidate_audit_path(plan: dict) -> Path:
    return Path(plan["candidate_identity"]["candidate_bundle_audit_json"])


def _write_ready_plan(path: Path) -> dict:
    candidate_audit = path.parent / "candidate_bundle_audit.json"
    _write_json(
        candidate_audit,
        {
            "decision": "PASS",
            "bundle_dir": "/tmp/smart_seq520_candidate_bundle",
            "contract_mode": "smart_seq520_candidate",
            "bundle_summary": {
                "seq_input_dim": matrix.CONTRACT_SEQ_SNAP_WIDTH,
                "snap_input_dim": matrix.CONTRACT_SEQ_SNAP_WIDTH,
            },
        },
    )
    plan = {
        "schema_version": "entry_smart_ablation_replay_plan_gate_v1",
        "decision": "READY_FOR_SMART_ABLATION_REPLAY_PLAN_REVIEW",
        "report_only": True,
        "training_allowed": False,
        "replay_allowed_by_this_gate": False,
        "iql_allowed_by_this_gate": False,
        "shadow_live_promotion_allowed": False,
        "side_effects_started": {
            "training": False,
            "replay": False,
            "iql_distillation": False,
            "shadow": False,
            "live": False,
            "promotion": False,
        },
        "smart_variant": "smart_seq520_candidate",
        "candidate_identity": {
            "candidate_bundle_dir": "/tmp/smart_seq520_candidate_bundle",
            "candidate_bundle_audit_json": str(candidate_audit),
            "seq_input_dim": matrix.CONTRACT_SEQ_SNAP_WIDTH,
            "snap_input_dim": matrix.CONTRACT_SEQ_SNAP_WIDTH,
            "manifest_variant": "smart_seq520_candidate",
        },
        "required_ablation_plan": build_required_ablation_plan(),
        "json_path": str(path),
        "md_path": str(path.with_suffix(".md")),
    }
    _write_json(path, plan)
    return plan


def _required_arm(plan: dict, ablation_id: str) -> dict:
    return next(
        row
        for row in plan["required_ablation_plan"]["required_ablations"]
        if row["ablation_id"] == ablation_id
    )


def _valid_no_xgb_summary_fields() -> dict:
    bridge_fields = list(matrix.REQUIRED_NO_XGB_BRIDGE_FIELDS)
    split_diagnostics = {
        "comparable": True,
        "time_match": True,
        "identical_predictions": True,
        "max_abs_prob_delta": 0.0,
        "max_abs_edge_score_delta": 0.0,
        "pred_direction_diff_count": 0,
        "trade_side_diff_count": 0,
    }
    split_bridge = {
        "neutral_xgb_bridge": True,
        "bridge_source": "neutral_uniform_proba",
        "bridge_fields": bridge_fields,
        "seq_input_dim": matrix.CONTRACT_SEQ_SNAP_WIDTH,
        "snap_input_dim": matrix.CONTRACT_SEQ_SNAP_WIDTH,
        "fields": [f"signal_{idx}" for idx in range(matrix.CONTRACT_SEQ_SNAP_WIDTH)],
    }
    return {
        "no_xgb_ablation": {
            "mode": "neutralize_signal_bridge",
            "neutralize_signal_bridge": True,
            "required": True,
            "neutralized_fields": bridge_fields,
            "neutral_values": [1.0 / 3.0] * len(bridge_fields),
        },
        "no_xgb_ablation_diagnostics": {
            "available": True,
            "splits": {
                "val": dict(split_diagnostics),
                "test": dict(split_diagnostics),
            },
        },
        "input_bridge_contract": {
            "splits": {
                "val": dict(split_bridge),
                "test": dict(split_bridge),
            },
        },
    }


def _selective_summary(*, include_no_xgb: bool = True) -> dict:
    summary = {
        "decision": "PASS",
        "contract_mode": "smart_seq520_candidate",
        "bundle_dir": "/tmp/smart_seq520_candidate_bundle",
        "bundle_seq_input_dim": matrix.CONTRACT_SEQ_SNAP_WIDTH,
        "bundle_snap_input_dim": matrix.CONTRACT_SEQ_SNAP_WIDTH,
    }
    if include_no_xgb:
        summary.update(_valid_no_xgb_summary_fields())
    return summary


def _feature_mask_for_arm(tmp_path: Path, arm: dict, *, defect: str | None = None) -> dict:
    expected_count = matrix.CONTRACT_SEQ_SNAP_WIDTH - int(arm["expected_seq_snap_width"])
    zero_indices = list(range(expected_count))
    if defect == "count":
        zero_indices = zero_indices[:-1]
    zero_names = [f"masked_feature_{idx}" for idx in zero_indices]
    spec = {
        "schema_version": "entry_smart_feature_mask_spec_v1",
        "mask_mode": "zero_seq_snap_features",
        "zero_value": 0.0,
        "signal_field_count": matrix.CONTRACT_SEQ_SNAP_WIDTH,
        "zero_feature_count": len(zero_indices),
        "zero_indices": zero_indices,
        "zero_feature_names": zero_names,
        "plan_arm": matrix._canonical_arm_signature(arm),
    }
    if defect == "plan":
        spec["plan_arm"] = {**spec["plan_arm"], "expected_seq_snap_width": matrix.CONTRACT_SEQ_SNAP_WIDTH}
    spec_path = tmp_path / f"{arm['ablation_id']}_feature_mask.json"
    _write_json(spec_path, spec)
    spec_sha = matrix._sha256_file(spec_path)
    mask = {
        "enabled": True,
        "ablation_id": arm["ablation_id"],
        "path": str(spec_path),
        "sha256": spec_sha,
        "mask_mode": "zero_seq_snap_features",
        "zero_value": 0.0,
        "signal_field_count": matrix.CONTRACT_SEQ_SNAP_WIDTH,
        "zero_indices": zero_indices,
        "zero_feature_names": zero_names,
    }
    if defect == "hash":
        mask["sha256"] = "0" * 64
    if defect == "mode":
        mask["mask_mode"] = "drop_seq_snap_features"
    return mask


def _write_pass_replay_arm(
    replay_root: Path,
    *,
    plan: dict,
    ablation_id: str,
    policy_id: str = "candidate_top5",
    feature_mask_defect: str | None = None,
    include_no_xgb_summary: bool = True,
    selective_summary_override: dict | None = None,
) -> None:
    arm = _required_arm(plan, ablation_id)
    replay_dir = replay_root / ablation_id
    _write_csv(
        replay_dir / "replay_policy_metrics.csv",
        [
            {
                "scope": "aggregate",
                "fold": "ALL",
                "policy_id": policy_id,
                "n_trades": 4,
                "net_sum_bps": 125.0,
                "profit_factor": 1.8,
                "max_drawdown_bps": 22.0,
                "win_rate": 0.75,
                "mean_mae_bps": 9.0,
            }
        ],
    )
    _write_csv(
        replay_dir / "replay_policy_monthly.csv",
        [
            {"policy_id": policy_id, "entry_month": "2026-01", "net_sum_bps": 80.0},
            {"policy_id": policy_id, "entry_month": "2026-02", "net_sum_bps": 45.0},
        ],
    )
    _write_csv(
        replay_dir / "replay_policy_trades.csv",
        [
            {
                "entry_time": "2026-01-05T08:00:00Z",
                "policy_id": policy_id,
                "side": "LONG",
                "net_pnl_bps": 80.0,
            }
        ],
    )
    _write_csv(
        replay_dir / "replay_policy_slices.csv",
        [
            {
                "slice_family": "session",
                "slice_value": "EU",
                "policy_id": policy_id,
                "n_trades": 1,
                "net_sum_bps": 80.0,
                "profit_factor": 2.0,
                "max_drawdown_bps": 10.0,
            },
            {
                "slice_family": "regime",
                "slice_value": "UP",
                "policy_id": policy_id,
                "n_trades": 1,
                "net_sum_bps": 70.0,
                "profit_factor": 1.9,
                "max_drawdown_bps": 12.0,
            },
            {
                "slice_family": "direction",
                "slice_value": "LONG",
                "policy_id": policy_id,
                "n_trades": 1,
                "net_sum_bps": 95.0,
                "profit_factor": 2.3,
                "max_drawdown_bps": 8.0,
            },
            {
                "slice_family": "tail",
                "slice_value": "normal",
                "policy_id": policy_id,
                "n_trades": 1,
                "net_sum_bps": 60.0,
                "profit_factor": 1.7,
                "max_drawdown_bps": 14.0,
            },
        ],
    )
    selective_summary = (
        copy.deepcopy(selective_summary_override)
        if selective_summary_override is not None
        else _selective_summary(include_no_xgb=include_no_xgb_summary)
    )
    selective_summary_path = replay_dir / "selective_edge_summary.json"
    _write_json(selective_summary_path, selective_summary)
    selective_summary_sha = matrix._sha256_file(selective_summary_path)
    candidate_audit_path = _candidate_audit_path(plan)
    identity = {
        "contract_mode": "smart_seq520_candidate",
        "selective_edge_contract_mode": "smart_seq520_candidate",
        "candidate_bundle_dir": plan["candidate_identity"]["candidate_bundle_dir"],
        "candidate_bundle_audit_json": str(candidate_audit_path),
        "candidate_bundle_audit_sha256": matrix._sha256_file(candidate_audit_path),
        "candidate_bundle_seq_input_dim": matrix.CONTRACT_SEQ_SNAP_WIDTH,
        "candidate_bundle_snap_input_dim": matrix.CONTRACT_SEQ_SNAP_WIDTH,
        "expected_input_dim": matrix.CONTRACT_SEQ_SNAP_WIDTH,
        "selective_edge_seq_input_dim": matrix.CONTRACT_SEQ_SNAP_WIDTH,
        "selective_edge_snap_input_dim": matrix.CONTRACT_SEQ_SNAP_WIDTH,
        "selective_edge_summary_json": str(selective_summary_path),
        "selective_edge_summary_sha256": selective_summary_sha,
    }
    artifact_hashes = {
        name: matrix._sha256_file(replay_dir / name)
        for name in (
            "replay_policy_metrics.csv",
            "replay_policy_monthly.csv",
            "replay_policy_trades.csv",
            "replay_policy_slices.csv",
        )
    }
    manifest = {
        "decision": "PASS",
        "contract_mode": "smart_seq520_candidate",
        "manifest_variant": "smart_seq520_candidate",
        "ablation_id": ablation_id,
        "policy_id": policy_id,
        "policies": [policy_id],
        "best_aggregate_row": {"policy_id": policy_id},
        "artifact_hashes": artifact_hashes,
        "selective_edge_summary_json": str(selective_summary_path),
        "replay_identity_contract": identity,
    }
    if arm["ablation_type"] in {"feature_set_ablation", "drop_smart_family"}:
        manifest["feature_mask_ablation"] = _feature_mask_for_arm(
            replay_dir,
            arm,
            defect=feature_mask_defect,
        )
    _write_json(
        replay_dir / "REPLAY_EVIDENCE_MANIFEST.json",
        manifest,
    )


def _validate_arm(plan: dict, replay_root: Path, ablation_id: str) -> list[dict]:
    _identity, checks = matrix._validate_replay_arm(
        arm=_required_arm(plan, ablation_id),
        plan=plan,
        replay_root=replay_root,
    )
    return checks


def _failed_checks(checks: list[dict]) -> set[str]:
    return {check["name"] for check in checks if not check["ok"]}


def test_matrix_report_includes_edge_summaries_for_pass_arm(tmp_path: Path) -> None:
    plan_path = tmp_path / "ENTRY_SMART_ABLATION_REPLAY_PLAN_GATE_latest.json"
    plan = _write_ready_plan(plan_path)
    pass_arm = plan["required_ablation_plan"]["required_ablations"][0]
    replay_root = tmp_path / "replay_matrix"
    _write_pass_replay_arm(replay_root, plan=plan, ablation_id=pass_arm["ablation_id"])

    report = matrix.run(
        argparse.Namespace(
            plan_json=str(plan_path),
            replay_root=str(replay_root),
            out_dir=str(tmp_path / "out"),
            quiet=True,
            fail_on_not_ready=False,
        )
    )

    arm = next(row for row in report["ablation_results"] if row["ablation_id"] == pass_arm["ablation_id"])
    assert arm["decision"] == "PASS"
    identity = arm["identity"]
    assert identity["metrics_summary"] == {
        "policy_id": "candidate_top5",
        "n_trades": 4,
        "net_sum_bps": 125.0,
        "profit_factor": 1.8,
        "max_drawdown_bps": 22.0,
        "win_rate": 0.75,
        "mean_mae_bps": 9.0,
    }
    assert identity["monthly_summary"] == {
        "months": 2,
        "negative_months": 0,
        "all_months_positive": True,
        "min_month_net_bps": 45.0,
        "worst_month": "2026-02",
    }
    assert identity["slice_edge_summary"]["slices"] == 4
    assert identity["slice_edge_summary"]["negative_slices"] == 0
    assert identity["slice_edge_summary"]["worst_slice"] == {
        "family": "tail",
        "value": "normal",
        "net_sum_bps": 60.0,
    }
    assert set(identity["slice_edge_summary"]["by_family"]) == {"session", "regime", "direction", "tail"}

    persisted = json.loads(Path(report["json_path"]).read_text(encoding="utf-8"))
    persisted_arm = next(
        row for row in persisted["ablation_results"] if row["ablation_id"] == pass_arm["ablation_id"]
    )
    assert persisted_arm["decision"] == "PASS"
    assert persisted_arm["identity"]["metrics_summary"] == identity["metrics_summary"]
    assert persisted_arm["identity"]["monthly_summary"] == identity["monthly_summary"]
    assert persisted_arm["identity"]["slice_edge_summary"] == identity["slice_edge_summary"]


def test_feature_mask_provenance_passes_for_masked_arm(tmp_path: Path) -> None:
    plan_path = tmp_path / "ENTRY_SMART_ABLATION_REPLAY_PLAN_GATE_latest.json"
    plan = _write_ready_plan(plan_path)
    replay_root = tmp_path / "replay_matrix"
    ablation_id = "drop_family_trend_ema_smart_layer"
    _write_pass_replay_arm(replay_root, plan=plan, ablation_id=ablation_id)

    failed = _failed_checks(_validate_arm(plan, replay_root, ablation_id))

    assert not any("feature-mask" in check for check in failed)


def test_feature_mask_provenance_fails_closed_on_plan_hash_mode_and_count_mismatch(tmp_path: Path) -> None:
    plan_path = tmp_path / "ENTRY_SMART_ABLATION_REPLAY_PLAN_GATE_latest.json"
    plan = _write_ready_plan(plan_path)
    replay_root = tmp_path / "replay_matrix"
    ablation_id = "drop_family_trend_ema_smart_layer"

    expected_failure_by_defect = {
        "plan": f"{ablation_id} feature-mask spec parses and canonical plan arm matches",
        "hash": f"{ablation_id} feature-mask hash matches artifact",
        "mode": f"{ablation_id} feature-mask zero contract matches seq520 mask spec",
        "count": f"{ablation_id} feature-mask count matches plan width delta",
    }
    for defect, expected_failure in expected_failure_by_defect.items():
        case_root = replay_root / defect
        _write_pass_replay_arm(
            case_root,
            plan=plan,
            ablation_id=ablation_id,
            feature_mask_defect=defect,
        )

        failed = _failed_checks(_validate_arm(plan, case_root, ablation_id))

        assert expected_failure in failed


def test_no_xgb_provenance_fails_closed_when_missing_or_wrong(tmp_path: Path) -> None:
    plan_path = tmp_path / "ENTRY_SMART_ABLATION_REPLAY_PLAN_GATE_latest.json"
    plan = _write_ready_plan(plan_path)
    replay_root = tmp_path / "replay_matrix"
    ablation_id = "no_xgb"
    expected_failure = f"{ablation_id} no-XGB provenance matches neutralized bridge contract when required"

    _write_pass_replay_arm(
        replay_root / "missing",
        plan=plan,
        ablation_id=ablation_id,
        include_no_xgb_summary=False,
    )
    wrong_summary = _selective_summary()
    wrong_summary["input_bridge_contract"]["splits"]["test"]["bridge_source"] = "live_xgb_proba"
    _write_pass_replay_arm(
        replay_root / "wrong",
        plan=plan,
        ablation_id=ablation_id,
        selective_summary_override=wrong_summary,
    )

    assert expected_failure in _failed_checks(_validate_arm(plan, replay_root / "missing", ablation_id))
    assert expected_failure in _failed_checks(_validate_arm(plan, replay_root / "wrong", ablation_id))
