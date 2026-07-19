import json
import hashlib
from pathlib import Path

import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
    model_native_signal_contract_metadata,
)
from gx1.execution.model_native_entry_replay_v1 import (
    OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
    UNIT_NORMALIZED_PNL_MODE,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    required_training_specialists_for_mode,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    model_direction_decision_contract_metadata,
)
from gx1.scripts.verify_entry_replay_readiness_v1 import (
    _candidate_bundle_audit_checks,
    _normalize_contract_mode,
    _replay_checks,
    _selective_edge_checks,
    _selective_metrics_authority_checks,
    build_parser,
)


def _signal_contract() -> dict:
    selected = [
        *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
        *(
            f"session_regime.replay_fixture_{index:03d}"
            for index in range(MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT)
        ),
    ]
    return model_native_signal_contract_metadata(selected)


def _specialist_snapshot() -> dict:
    expected = sorted(
        required_training_specialists_for_mode(MODEL_NATIVE_CONTRACT_MODE)
    )
    return {
        "requested_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "observed_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "contract_mode_declared": True,
        "expected_signal_dim": 513,
        "bundle_seq_input_dim": 513,
        "bundle_snap_input_dim": 513,
        "specialist_fusion_enabled": True,
        "expected_specialists": expected,
        "observed_specialists": expected,
        "required_specialists_exact": True,
        "chart_geometry_present": True,
        "price_action_candle_present": True,
        "specialist_model_contract_valid": True,
        "specialist_model_contract_set_exact": True,
        "specialist_model_contract_owned_objectives_match": True,
        "specialist_model_contract_signal_families_match": True,
        "specialist_model_contract_support_heads_match": True,
        "specialist_model_contract_model_roles_match": True,
        "failures": [],
    }


def _selective_report(dataset_dir: Path, bundle_dir: Path) -> dict:
    return {
        "decision": "PASS",
        "failures": [],
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "dataset_dir": str(dataset_dir),
        "bundle_dir": str(bundle_dir),
        "bundle_seq_input_dim": 513,
        "bundle_snap_input_dim": 513,
        "feature_mask_ablation": {"enabled": False},
        "model_native_signal_contract": _signal_contract(),
        "direction_decision_contract": model_direction_decision_contract_metadata(),
        "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
        "splits": ["test", "val"],
        "summaries": [
            {
                "split": split,
                "model": "candidate",
                "top5_all_mean_pnl_bps": 10.0,
                "top10_all_mean_pnl_bps": 8.0,
                "top5_all_direction_precision": 0.99,
                "top10_all_direction_precision": 0.97,
            }
            for split in ("val", "test")
        ],
        "bundle_specialist_contract": _specialist_snapshot(),
    }


def _selective_metrics() -> pd.DataFrame:
    rows = []
    for split in ("val", "test"):
        for top_frac in (0.05, 0.10):
            for group in ("ALL", "session=EU"):
                rows.append(
                    {
                        "split": split,
                        "model": "candidate",
                        "scope": "top_score",
                        "top_frac": top_frac,
                        "group": group,
                        "n": 25,
                        "mean_pnl_bps": 8.0,
                        "win_rate": 0.9,
                        "direction_precision": 0.98,
                    }
                )
    return pd.DataFrame(rows)


def test_contract_mode_rejects_all_retired_widths() -> None:
    assert _normalize_contract_mode(None) == MODEL_NATIVE_CONTRACT_MODE
    for retired in ("foundation_seq146", "challenger_seq215", "smart_seq520_candidate"):
        with pytest.raises(RuntimeError, match="retired"):
            _normalize_contract_mode(retired)


def test_selective_edge_checks_require_exact_seq513_and_model_argmax(
    tmp_path: Path,
) -> None:
    report = _selective_report(tmp_path / "dataset", tmp_path / "bundle")
    checks = _selective_edge_checks(
        report,
        _selective_metrics(),
        model_name="candidate",
        min_top5_mean_pnl_bps=0.0,
        min_top10_mean_pnl_bps=0.0,
        min_top_direction_precision=0.95,
        min_direction_slice_precision=0.90,
        min_direction_slice_n=20,
        expected_bundle_dir=str(tmp_path / "bundle"),
        expected_dataset_dir=tmp_path / "dataset",
    )
    assert all(check["ok"] for check in checks)

    report["selection_score_threshold"] = 0.1
    checks = _selective_edge_checks(
        report,
        _selective_metrics(),
        model_name="candidate",
        min_top5_mean_pnl_bps=0.0,
        min_top10_mean_pnl_bps=0.0,
        min_top_direction_precision=0.95,
        min_direction_slice_precision=0.90,
        expected_bundle_dir=str(tmp_path / "bundle"),
        expected_dataset_dir=tmp_path / "dataset",
    )
    failed = {check["name"] for check in checks if not check["ok"]}
    assert "selective-edge uses exact final direction argmax schema" in failed


def test_selective_edge_rejects_soft_or_implicit_precision_bounds(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="explicit high-precision admission bound"):
        _selective_edge_checks(
            _selective_report(tmp_path / "dataset", tmp_path / "bundle"),
            _selective_metrics(),
            model_name="candidate",
            min_top5_mean_pnl_bps=0.0,
            min_top10_mean_pnl_bps=0.0,
            min_top_direction_precision=0.50,
            min_direction_slice_precision=0.90,
            expected_bundle_dir=str(tmp_path / "bundle"),
            expected_dataset_dir=tmp_path / "dataset",
        )


def test_selective_metrics_are_bound_to_timestamped_report(tmp_path: Path) -> None:
    metrics_path = (
        tmp_path / "selective_edge_metrics_20260716T120000123456Z.csv"
    )
    metrics_path.write_text("split,model\nval,candidate\n", encoding="utf-8")
    report = {
        "metrics_path": str(metrics_path),
        "metrics_sha256": hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
    }
    checks = _selective_metrics_authority_checks(report, metrics_path)
    assert all(check["ok"] for check in checks)

    metrics_path.write_text("split,model\ntest,candidate\n", encoding="utf-8")
    checks = _selective_metrics_authority_checks(report, metrics_path)
    assert any(not check["ok"] for check in checks)


def _replay_trades(*, mismatch: bool = False) -> pd.DataFrame:
    side = ["SHORT", "SHORT"] if mismatch else ["LONG", "SHORT"]
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2026-01-03T08:00:00Z", "2026-02-04T09:00:00Z"]
            ),
            "source_split": ["test", "test"],
            "policy_id": ["candidate_replay"] * 2,
            "session": ["EU", "US"],
            "side": side,
            "score": [0.8, 0.7],
            "p_long": [0.8, 0.1],
            "p_short": [0.1, 0.8],
            "p_flat": [0.1, 0.1],
            "net_pnl_bps": [12.0, 10.0],
            "mfe_bps": [20.0, 18.0],
            "mae_bps": [-3.0, -4.0],
            "held_bars": [3, 4],
            "horizon_bars": [3, 4],
            "exit_mode": ["label_horizon", "label_horizon"],
            "row_simulation_mode": ["independent", "independent"],
            "filters_applied": [False, False],
            "offline_only": [True, True],
            "diagnostic_scope": [OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE] * 2,
            "pnl_normalization": [UNIT_NORMALIZED_PNL_MODE] * 2,
            "execution_order_simulation": [False, False],
            "position_size_applied": [False, False],
            "vol_regime": ["normal", "high"],
            "path_quality_pred": [0.75, 0.70],
            "bad_path_prob": [0.10, 0.15],
        }
    )


def _replay_manifest() -> dict:
    snapshot = {"ready": True}
    return {
        "decision": "PASS",
        "failures": [],
        "offline_only": True,
        "diagnostic_scope": OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
        "pnl_normalization": UNIT_NORMALIZED_PNL_MODE,
        "execution_order_simulation": False,
        "position_size_applied": False,
        "trade_log_authority_contract": {
            "ready": True,
            "failures": [],
        },
        "replay_identity_contract": {
            "ready": True,
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "candidate_specialist_contract": snapshot,
            "selective_edge_specialist_contract": snapshot,
        },
    }


def _replay_metrics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scope": "aggregate",
                "policy_id": "candidate_replay",
                "n_trades": 2,
                "net_sum_bps": 22.0,
                "win_rate": 1.0,
                "profit_factor": 3.0,
                "max_drawdown_bps": 0.0,
                "max_loss_bps": 0.0,
            }
        ]
    )


def test_replay_checks_fail_when_trade_side_differs_from_model_argmax(
    tmp_path: Path,
) -> None:
    monthly = pd.DataFrame({"net_sum_bps": [10.0, 12.0]})
    good = _replay_checks(
        tmp_path,
        _replay_manifest(),
        _replay_metrics(),
        monthly,
        _replay_trades(),
        min_net_sum_bps=0.0,
        min_profit_factor=1.05,
        max_drawdown_bps=100.0,
    )
    argmax_check = next(
        row
        for row in good
        if row["name"]
        == "offline replay trade sides equal model LONG/SHORT/FLAT argmax"
    )
    assert argmax_check["ok"] is True

    bad = _replay_checks(
        tmp_path,
        _replay_manifest(),
        _replay_metrics(),
        monthly,
        _replay_trades(mismatch=True),
        min_net_sum_bps=0.0,
        min_profit_factor=1.05,
        max_drawdown_bps=100.0,
    )
    argmax_check = next(
        row
        for row in bad
        if row["name"]
        == "offline replay trade sides equal model LONG/SHORT/FLAT argmax"
    )
    assert argmax_check["ok"] is False


def test_replay_checks_reject_execution_sizing_fields(tmp_path: Path) -> None:
    manifest = _replay_manifest()
    manifest["sizing_authority_contract"] = {"stale": True}
    checks = _replay_checks(
        tmp_path,
        manifest,
        _replay_metrics(),
        pd.DataFrame({"net_sum_bps": [10.0, 12.0]}),
        _replay_trades(),
        min_net_sum_bps=0.0,
        min_profit_factor=1.05,
        max_drawdown_bps=100.0,
    )
    no_sizing = next(
        row
        for row in checks
        if row["name"] == "offline replay exposes no execution sizing authority"
    )
    assert no_sizing["ok"] is False


def test_candidate_bundle_check_reads_exact_bundle_metadata(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bundle_metadata.json").write_text(
        json.dumps(
            {
                "seq_input_dim": 513,
                "snap_input_dim": 513,
                "model_native_signal_contract": _signal_contract(),
                "direction_decision_contract": model_direction_decision_contract_metadata(),
            }
        ),
        encoding="utf-8",
    )
    report = {
        "bundle_dir": str(bundle),
        "bundle_summary": {
            "specialist_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "seq_input_dim": 513,
            "snap_input_dim": 513,
        },
    }
    checks = _candidate_bundle_audit_checks(
        tmp_path / "audit.json",
        report,
        expected_dataset_dir=tmp_path / "dataset",
    )
    contract_check = next(
        row
        for row in checks
        if row["name"]
        == "candidate bundle metadata proves exact model-native direction contract"
    )
    assert contract_check["ok"] is True


def test_parser_requires_explicit_immutable_evidence_paths() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--candidate-readiness-json",
            "/tmp/candidate.json",
            "--candidate-bundle-audit-json",
            "/tmp/bundle-audit.json",
            "--selective-edge-report-json",
            "/tmp/selective.json",
            "--selective-edge-metrics-csv",
            "/tmp/metrics.csv",
            "--replay-evidence-json",
            "/tmp/replay.json",
            "--pretrain-audit-json",
            "/tmp/pretrain.json",
            "--expected-dataset-dir",
            "/tmp/dataset",
            "--out-dir",
            "/tmp/out",
        ]
    )
    assert not hasattr(args, "contract_mode")
    assert not hasattr(args, "min_top_direction_precision")
    assert "fail-on-not-ready" not in parser.format_help()
    assert not hasattr(args, "min_direction_slice_precision")
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--candidate-readiness-json",
                "/tmp/candidate.json",
                "--candidate-bundle-audit-json",
                "/tmp/bundle-audit.json",
                "--selective-edge-report-json",
                "/tmp/selective.json",
                "--selective-edge-metrics-csv",
                "/tmp/metrics.csv",
                "--replay-evidence-json",
                "/tmp/replay.json",
                "--pretrain-audit-json",
                "/tmp/pretrain.json",
                "--expected-dataset-dir",
                "/tmp/dataset",
                "--out-dir",
                "/tmp/out",
                "--smart-seq520",
            ]
        )
