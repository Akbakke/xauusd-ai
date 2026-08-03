import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    model_native_signal_contract_metadata,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.features.entry_specialist_feature_groups_v1 import (
    required_training_specialists_for_mode,
)
from gx1.execution.model_native_entry_replay_v1 import (
    LABEL_HORIZON_EXIT_MODE,
    OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
    UNIT_NORMALIZED_PNL_MODE,
    label_horizon_exit_policy_contract,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    model_direction_decision_contract_metadata,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    atomic_write_parquet_immutable,
    build_prediction_evidence_declaration,
)
from tests.model_native_turning_point_support import (
    turning_point_prediction_columns,
)
from tests.model_native_offline_rl_support import offline_rl_prediction_columns
from gx1.scripts.materialize_entry_candidate_replay_evidence_v1 import (
    REPLAY_REQUIRED_SPLIT,
    REPLAY_REQUIRED_YEAR,
    _identity_contract,
    _trade_log_authority_contract,
    audit_model_native_replay_trades,
    build_parser,
    normalize_trades,
)
from gx1.scripts.materialize_entry_candidate_replay_trade_log_v1 import (
    CANDIDATE_EVENT_PREFIX,
    TRADE_LOG_EVENT_PREFIX,
    TRADE_LOG_SCHEMA_VERSION,
    _direction_policy_contract,
)
from gx1.scripts.verify_entry_foundation_state_v1 import STATE_EVENT_PREFIX
from tests.model_native_signal_support import canonical_model_native_selected_fields


def _signal_contract() -> dict:
    return model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.replay_evidence_fixture"
        )
    )


def _specialist_snapshot() -> dict:
    specialists = sorted(
        required_training_specialists_for_mode(MODEL_NATIVE_CONTRACT_MODE)
    )
    return {
        "requested_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "observed_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "expected_signal_dim": 513,
        "bundle_seq_input_dim": 513,
        "bundle_snap_input_dim": 513,
        "specialist_fusion_enabled": True,
        "expected_specialists": specialists,
        "observed_specialists": specialists,
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


def _write_identity_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    metadata = {
        "seq_input_dim": 513,
        "snap_input_dim": 513,
        "state_dict_sha256": "a" * 64,
        "model_native_signal_contract": _signal_contract(),
        "direction_decision_contract": model_direction_decision_contract_metadata(),
    }
    (bundle / "bundle_metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )

    logits = np.asarray([[3.0, 1.0, 0.0], [0.0, 3.0, 1.0]], dtype=np.float64)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    pair_logits = np.column_stack(
        [np.maximum(logits[:, 0], logits[:, 1]), logits[:, 2]]
    )
    pair_exp = np.exp(pair_logits - pair_logits.max(axis=1, keepdims=True))
    pair_probs = pair_exp / pair_exp.sum(axis=1, keepdims=True)
    frame = pd.DataFrame(
        {
            "split": ["val", "test"],
            "model": ["candidate", "candidate"],
            "time": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z"]
            ),
            "y_direction": [0, 1],
            "pred_direction": np.argmax(logits, axis=1),
            "p_long": probs[:, 0],
            "p_short": probs[:, 1],
            "p_flat": probs[:, 2],
            "selection_score_mode": [MODEL_DIRECTION_SELECTION_MODE] * 2,
            "public_trade_probability": pair_probs[:, 0],
            "public_flat_probability": pair_probs[:, 1],
            "public_trade_flat_margin": pair_logits[:, 0] - pair_logits[:, 1],
            "public_trade_flat_hard_decision": np.argmax(pair_logits, axis=1),
            "direction_logits": [row.tolist() for row in logits],
            "public_trade_flat_decision_logits": [
                row.tolist() for row in pair_logits
            ],
            **turning_point_prediction_columns(2),
            **offline_rl_prediction_columns(2),
        }
    )
    stamp = "20260716T120000123456Z"
    predictions_path = tmp_path / f"selective_edge_predictions_{stamp}.parquet"
    atomic_write_parquet_immutable(frame, predictions_path)
    evidence = build_prediction_evidence_declaration(
        predictions_path=predictions_path,
        bundle_dir=bundle,
        bundle_metadata=metadata,
        requested_splits=["val", "test"],
    )

    report_path = tmp_path / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_{stamp}.json"
    report = {
        "schema_version": "entry_candidate_selective_edge_v1",
        "created_utc": "2026-07-16T12:00:00.123456+00:00",
        "decision": "PASS",
        "failures": [],
        "json_path": str(report_path),
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "bundle_dir": str(bundle),
        "dataset_dir": str(dataset),
        "bundle_seq_input_dim": 513,
        "bundle_snap_input_dim": 513,
        "feature_mask_ablation": {"enabled": False},
        "model_native_signal_contract": _signal_contract(),
        "direction_decision_contract": model_direction_decision_contract_metadata(),
        "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
        "splits": ["test", "val"],
        "models": ["candidate"],
        "summaries": [],
        "bundle_specialist_contract": _specialist_snapshot(),
        "predictions_path": str(predictions_path),
        "prediction_evidence": evidence,
        "bundle_metadata_sha256": evidence["bundle_metadata_sha256"],
        "model_state_dict_sha256": evidence["model_state_dict_sha256"],
    }
    report_path.write_text(
        json.dumps(report, sort_keys=True) + "\n", encoding="utf-8"
    )

    specialists = sorted(
        required_training_specialists_for_mode(MODEL_NATIVE_CONTRACT_MODE)
    )
    candidate_path = tmp_path / "ENTRY_CANDIDATE_BUNDLE_AUDIT_20260716T120000123456Z.json"
    candidate = {
        "decision": "PASS",
        "failures": [],
        "bundle_dir": str(bundle),
        "specialist_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "required_training_specialists": specialists,
        "bundle_summary": {
            "specialist_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "seq_input_dim": 513,
            "snap_input_dim": 513,
            "specialist_groups": specialists,
            "specialist_fusion_enabled": True,
            "specialist_model_contract_valid": True,
            "specialist_model_contract_set_exact": True,
            "specialist_model_contract_owned_objectives_match": True,
            "specialist_model_contract_support_heads_match": True,
            "specialist_model_contract_signal_families_match": True,
            "specialist_model_contract_model_roles_match": True,
        },
        "bundle_specialist_model_contract": {
            "valid": True,
            "set_exact": True,
            "owned_objectives_match": True,
            "support_heads_match": True,
            "signal_families_match": True,
            "model_roles_match": True,
            "failures": [],
        },
    }
    candidate_path.write_text(
        json.dumps(candidate, sort_keys=True) + "\n", encoding="utf-8"
    )
    return candidate_path, report_path, predictions_path, evidence


def test_identity_contract_revalidates_newest_immutable_prediction_event(
    tmp_path: Path,
) -> None:
    candidate_path, report_path, predictions_path, evidence = _write_identity_fixture(
        tmp_path
    )
    identity = _identity_contract(
        candidate_bundle_audit_path=candidate_path,
        selective_edge_report_path=report_path,
        require_identity_artifacts=True,
        requested_contract_mode=MODEL_NATIVE_CONTRACT_MODE,
    )
    assert identity["ready"] is True
    assert identity["expected_input_dim"] == 513
    assert identity["authoritative_predictions_path"] == str(predictions_path)
    assert identity["prediction_evidence"] == evidence

    with pytest.raises(RuntimeError, match="latest"):
        _identity_contract(
            candidate_bundle_audit_path=candidate_path.with_name(
                "ENTRY_CANDIDATE_BUNDLE_AUDIT_latest.json"
            ),
            selective_edge_report_path=report_path,
            require_identity_artifacts=True,
        )


def test_trade_log_authority_requires_hash_and_same_prediction_report(
    tmp_path: Path,
) -> None:
    candidate_path, report_path, _, _ = _write_identity_fixture(tmp_path)
    identity = _identity_contract(
        candidate_bundle_audit_path=candidate_path,
        selective_edge_report_path=report_path,
        require_identity_artifacts=True,
    )
    trades_path = tmp_path / "candidate_replay_trade_log.csv"
    trades_path.write_text("side,p_long,p_short,p_flat\nLONG,0.8,0.1,0.1\n", encoding="utf-8")
    trades_sha = hashlib.sha256(trades_path.read_bytes()).hexdigest()
    counts_path = tmp_path / "candidate_replay_policy_counts.csv"
    counts = {
        "policy_id": "candidate_replay",
        "evaluated_rows": 2,
        "model_flat_rows": 1,
        "non_flat_argmax_rows": 1,
        "expected_trades": 1,
        "trades": 1,
        "trades_equal_non_flat_argmax_rows": True,
        "filters_applied": False,
        "offline_only": True,
        "diagnostic_scope": OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
        "pnl_normalization": UNIT_NORMALIZED_PNL_MODE,
        "execution_order_simulation": False,
        "position_size_applied": False,
        "occupancy_filter_applied": False,
        "cooldown_applied": False,
        "max_trades_per_day_applied": False,
        "daily_loss_limit_applied": False,
        "invalid_path_skip_allowed": False,
    }
    counts_path.write_text(pd.DataFrame([counts]).to_csv(index=False), encoding="utf-8")
    state_path, _ = write_immutable_json_event(
        tmp_path / "state",
        STATE_EVENT_PREFIX,
        {
            "created_utc": "2026-07-16T12:00:01.123456+00:00",
            "decision": "MODEL_NATIVE_SEQ513_STATE_PROVEN_LAUNCH_BLOCKED",
            "failures": [],
        },
    )
    candidate_readiness_path, _ = write_immutable_json_event(
        tmp_path / "candidate_readiness",
        CANDIDATE_EVENT_PREFIX,
        {
            "created_utc": "2026-07-16T12:00:02.123456+00:00",
            "decision": "READY_FOR_CANDIDATE_TRAINING",
            "failures": [],
        },
    )
    manifest = {
        "schema_version": TRADE_LOG_SCHEMA_VERSION,
        "created_utc": "2026-07-16T12:00:03.123456+00:00",
        "decision": "PASS",
        "failures": [],
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "expected_signal_dim": 513,
        "trades_path": str(trades_path),
        "trades_sha256": trades_sha,
        "counts_path": str(counts_path),
        "counts_sha256": hashlib.sha256(counts_path.read_bytes()).hexdigest(),
        "prediction_report_json": str(report_path),
        "prediction_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "prediction_evidence": identity["prediction_evidence"],
        "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
        "direction_policy_contract": _direction_policy_contract(),
        "exit_policy_contract": label_horizon_exit_policy_contract(),
        "model_native_authority": {
            "state_json": str(state_path),
            "state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
            "candidate_readiness_json": str(candidate_readiness_path),
            "candidate_readiness_sha256": hashlib.sha256(
                candidate_readiness_path.read_bytes()
            ).hexdigest(),
        },
        "policy_config": {
            "direction_authority": "argmax(final_calibrated_direction_logits)",
            "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
            "eval_split": REPLAY_REQUIRED_SPLIT,
            "offline_only": True,
            "diagnostic_scope": OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
            "pnl_normalization": UNIT_NORMALIZED_PNL_MODE,
            "execution_order_simulation": False,
            "position_size_applied": False,
            "model_flat_is_only_direction_no_trade": True,
            "one_trade_per_non_flat_argmax_row": True,
            "row_simulation_mode": "independent",
            "occupancy_filter_allowed": False,
            "cooldown_allowed": False,
            "max_trades_per_day_allowed": False,
            "daily_loss_limit_allowed": False,
            "invalid_path_skip_allowed": False,
            "exit_mode": LABEL_HORIZON_EXIT_MODE,
            "filters_applied": False,
        },
        "n_test_rows": 2,
        "n_model_flat_rows": 1,
        "n_non_flat_argmax_rows": 1,
        "n_trades": 1,
        "trades_equal_non_flat_argmax_rows": True,
        "filters_applied": False,
        "offline_only": True,
        "diagnostic_scope": OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
        "pnl_normalization": UNIT_NORMALIZED_PNL_MODE,
        "execution_order_simulation": False,
        "position_size_applied": False,
        "policy_counts": counts,
    }
    manifest_path, manifest = write_immutable_json_event(
        tmp_path / "trade_log",
        TRADE_LOG_EVENT_PREFIX,
        manifest,
    )
    contract = _trade_log_authority_contract(
        manifest_path=manifest_path,
        trades_path=trades_path,
        selective_edge_report_path=report_path,
        identity=identity,
    )
    assert contract["ready"] is True

    tampered_counts = dict(counts)
    tampered_counts["trades"] = 0
    tampered_counts["trades_equal_non_flat_argmax_rows"] = False
    counts_path.write_text(
        pd.DataFrame([tampered_counts]).to_csv(index=False), encoding="utf-8"
    )
    manifest["counts_sha256"] = hashlib.sha256(counts_path.read_bytes()).hexdigest()
    manifest["policy_counts"] = tampered_counts
    manifest["n_trades"] = 0
    manifest["trades_equal_non_flat_argmax_rows"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    contract = _trade_log_authority_contract(
        manifest_path=manifest_path,
        trades_path=trades_path,
        selective_edge_report_path=report_path,
        identity=identity,
    )
    assert contract["ready"] is False
    assert any("non-FLAT" in failure for failure in contract["failures"])

    manifest.pop("trades_sha256")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    contract = _trade_log_authority_contract(
        manifest_path=manifest_path,
        trades_path=trades_path,
        selective_edge_report_path=report_path,
        identity=identity,
    )
    assert contract["ready"] is False
    assert any("SHA-256" in failure for failure in contract["failures"])


def test_model_native_trade_audit_enforces_argmax_and_rejects_flat_trade() -> None:
    base = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(["2026-01-01T00:00:00Z"]),
            "source_split": [REPLAY_REQUIRED_SPLIT],
            "policy_id": ["candidate"],
            "session": ["EU"],
            "side": ["LONG"],
            "score": [0.8],
            "p_long": [0.8],
            "p_short": [0.1],
            "p_flat": [0.1],
            "net_pnl_bps": [10.0],
            "mfe_bps": [15.0],
            "mae_bps": [-3.0],
            "held_bars": [3],
            "horizon_bars": [3],
            "exit_mode": [LABEL_HORIZON_EXIT_MODE],
            "row_simulation_mode": ["independent"],
            "filters_applied": [False],
            "offline_only": [True],
            "diagnostic_scope": [OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE],
            "pnl_normalization": [UNIT_NORMALIZED_PNL_MODE],
            "execution_order_simulation": [False],
            "position_size_applied": [False],
            "vol_regime": ["normal"],
            "path_quality_pred": [0.75],
            "bad_path_prob": [0.10],
        }
    )
    assert audit_model_native_replay_trades(base)["ready"] is True

    wrong = base.copy()
    wrong["side"] = "SHORT"
    report = audit_model_native_replay_trades(wrong)
    assert report["ready"] is False
    assert any("argmax" in failure for failure in report["failures"])

    flat = base.copy()
    flat[["p_long", "p_short", "p_flat"]] = [0.1, 0.1, 0.8]
    flat["side"] = "FLAT"
    report = audit_model_native_replay_trades(flat)
    assert report["ready"] is False
    assert any("model-FLAT" in failure for failure in report["failures"])

    tied = base.copy()
    tied[["p_long", "p_short", "p_flat"]] = [0.45, 0.45, 0.10]
    report = audit_model_native_replay_trades(tied)
    assert report["ready"] is False
    assert any("no unique top class" in failure for failure in report["failures"])


def test_trade_normalization_rejects_threshold_columns_and_missing_full_stack() -> None:
    frame = pd.DataFrame(
        {
            "entry_time": ["2026-01-01T00:00:00Z"],
            "source_split": [REPLAY_REQUIRED_SPLIT],
            "policy_id": ["candidate"],
            "session": ["EU"],
            "side": ["LONG"],
            "score": [0.8],
            "p_long": [0.8],
            "p_short": [0.1],
            "p_flat": [0.1],
            "gross_pnl_bps": [11.0],
            "net_pnl_bps": [10.0],
            "mfe_bps": [15.0],
            "mae_bps": [-3.0],
            "held_bars": [3],
            "horizon_bars": [3],
            "exit_mode": [LABEL_HORIZON_EXIT_MODE],
            "row_simulation_mode": ["independent"],
            "filters_applied": [False],
            "offline_only": [True],
            "diagnostic_scope": [OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE],
            "pnl_normalization": [UNIT_NORMALIZED_PNL_MODE],
            "execution_order_simulation": [False],
            "position_size_applied": [False],
            "vol_regime": ["normal"],
            "path_quality_pred": [0.75],
            "bad_path_prob": [0.10],
            "direction_correct": [True],
        }
    )
    normalized, failures = normalize_trades(
        frame,
        policy_id="candidate",
    )
    assert failures == []
    assert audit_model_native_replay_trades(normalized)["ready"] is True

    _, failures = normalize_trades(
        frame.assign(entry_time=[f"{REPLAY_REQUIRED_YEAR - 1}-01-01T00:00:00Z"]),
        policy_id="candidate",
    )
    assert any("required year" in failure for failure in failures)
    _, failures = normalize_trades(
        frame.assign(source_split=["val"]),
        policy_id="candidate",
    )
    assert any("test-split" in failure for failure in failures)

    with pytest.raises(RuntimeError, match="direction-threshold"):
        normalize_trades(
            frame.assign(score_threshold=None),
            policy_id="candidate",
        )
    with pytest.raises(RuntimeError, match="execution-sizing"):
        normalize_trades(
            frame.assign(applied_size_multiplier=1.0),
            policy_id="candidate",
        )
    with pytest.raises(RuntimeError, match="exact model-native columns"):
        normalize_trades(
            frame.drop(columns=["vol_regime"]),
            policy_id="candidate",
        )


def test_parser_requires_explicit_lineage_and_only_native_contract() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--trades-path",
            "/tmp/trades.csv",
            "--trade-log-manifest-json",
            "/tmp/trade-report.json",
            "--out-dir",
            "/tmp/out",
            "--candidate-bundle-audit-json",
            "/tmp/bundle-audit.json",
            "--selective-edge-report-json",
            "/tmp/selective.json",
        ]
    )
    assert args.policy_id == "candidate_replay"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--trades-path",
                "/tmp/trades.csv",
                "--trade-log-manifest-json",
                "/tmp/trade-report.json",
                "--out-dir",
                "/tmp/out",
                "--candidate-bundle-audit-json",
                "/tmp/bundle-audit.json",
                "--selective-edge-report-json",
                "/tmp/selective.json",
                "--smart-seq520",
            ]
        )
    for retired_flag in (
        "--require-year",
        "--allow-non-2026",
        "--contract-mode",
        "--require-model-native-trade-fields",
        "--no-require-model-native-trade-fields",
        "--require-identity-artifacts",
        "--no-require-identity-artifacts",
    ):
        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "--trades-path",
                    "/tmp/trades.csv",
                    "--trade-log-manifest-json",
                    "/tmp/trade-report.json",
                    "--out-dir",
                    "/tmp/out",
                    "--candidate-bundle-audit-json",
                    "/tmp/bundle-audit.json",
                    "--selective-edge-report-json",
                    "/tmp/selective.json",
                    retired_flag,
                ]
            )
