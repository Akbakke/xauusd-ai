import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
    model_native_signal_contract_metadata,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.execution.model_native_entry_replay_v1 import (
    LABEL_HORIZON_EXIT_MODE,
    OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
    OFFLINE_REPLAY_EXECUTION_CODE_PATH,
    SourceTape,
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
    sha256_file,
)
from tests.model_native_turning_point_support import (
    turning_point_prediction_columns,
)
from tests.model_native_offline_rl_support import offline_rl_prediction_columns
from gx1.scripts.materialize_entry_candidate_replay_trade_log_v1 import (
    CANDIDATE_EVENT_PREFIX,
    TRADE_LOG_EVENT_PREFIX,
    _prepare_predictions,
    _resolve_score_surface,
    build_parser,
    run,
)
from gx1.scripts.verify_entry_foundation_state_v1 import (
    EVIDENCE_SPECS,
    STATE_EVENT_PREFIX,
    STATE_PROVEN_DECISION,
    STATE_SCHEMA_VERSION,
)
from tests.model_native_signal_support import canonical_model_native_selected_fields


def _signal_contract() -> dict:
    return model_native_signal_contract_metadata(
        canonical_model_native_selected_fields(
            remainder_prefix="session_regime.replay_trade_fixture"
        )
    )


def _specialist_snapshot() -> dict:
    return {
        "requested_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "expected_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "bundle_seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "bundle_snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "specialist_fusion_enabled": True,
        "required_specialists_exact": True,
        "specialist_model_contract_valid": True,
        "specialist_model_contract_set_exact": True,
        "specialist_model_contract_owned_objectives_match": True,
        "specialist_model_contract_signal_families_match": True,
        "specialist_model_contract_support_heads_match": True,
        "specialist_model_contract_model_roles_match": True,
        "chart_geometry_present": True,
        "price_action_candle_present": True,
        "failures": [],
    }


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_model_native_authority(tmp_path: Path) -> tuple[Path, Path]:
    artifact = tmp_path / "candidate_input.json"
    artifact.write_text("{}\n", encoding="utf-8")
    candidate_payload = {
        "schema_version": "entry_candidate_readiness_model_native_v1",
        "created_utc": "2026-07-16T12:00:01.123456+00:00",
        "decision": "READY_FOR_CANDIDATE_TRAINING_VEDTAK",
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "expected_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "edge_test_scope": "strict",
        "promotion_shadow_live_allowed": False,
        "artifact_fingerprints": {
            "candidate_input": {
                "path": str(artifact.resolve()),
                "sha256": sha256_file(artifact),
            }
        },
        "failures": [],
    }
    candidate_path, _ = write_immutable_json_event(
        tmp_path / "candidate_authority",
        CANDIDATE_EVENT_PREFIX,
        candidate_payload,
    )

    evidence_rows: list[dict] = []
    for offset, spec in enumerate(EVIDENCE_SPECS):
        if spec.name == "candidate_readiness":
            event_path = candidate_path
        else:
            event_path, _ = write_immutable_json_event(
                tmp_path / "state_inputs" / spec.name,
                spec.event_prefix,
                {
                    "schema_version": spec.schema_version,
                    "created_utc": f"2026-07-16T12:00:{10 + offset:02d}.123456+00:00",
                    "decision": spec.ready_decision,
                    "failures": [],
                },
            )
        evidence_rows.append(
            {
                "name": spec.name,
                "ready": True,
                "path": str(event_path),
                "sha256": sha256_file(event_path),
                "schema_version": spec.schema_version,
                "decision": spec.ready_decision,
                "failures": [],
            }
        )
    evidence_hashes = {row["name"]: row["sha256"] for row in evidence_rows}
    state_path, _ = write_immutable_json_event(
        tmp_path / "state_authority",
        STATE_EVENT_PREFIX,
        {
            "schema_version": STATE_SCHEMA_VERSION,
            "created_utc": "2026-07-16T12:01:00.123456+00:00",
            "decision": STATE_PROVEN_DECISION,
            "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
            "direction_logit_mode": MODEL_NATIVE_DIRECTION_LOGIT_MODE,
            "expected_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
            "model_native_evidence_ready": True,
            "launch_allowed": False,
            "promotion_shadow_live_allowed": False,
            "training_started": False,
            "replay_started": False,
            "live_started": False,
            "evidence": evidence_rows,
            "evidence_sha256": _sha256_json(evidence_hashes),
            "failures": [],
        },
    )
    return state_path, candidate_path


def _write_prediction_event(
    tmp_path: Path,
    predictions: pd.DataFrame,
    dataset_dir: Path,
) -> tuple[Path, Path]:
    frame = predictions.copy()
    probabilities = frame[["p_long", "p_short", "p_flat"]].to_numpy(
        dtype=np.float64
    )
    logits = np.log(np.clip(probabilities, 1e-8, 1.0))
    pred_direction = np.argmax(logits, axis=1).astype(np.int64)
    pair_logits = np.column_stack(
        [np.maximum(logits[:, 0], logits[:, 1]), logits[:, 2]]
    )
    pair_exp = np.exp(pair_logits - pair_logits.max(axis=1, keepdims=True))
    pair_probabilities = pair_exp / pair_exp.sum(axis=1, keepdims=True)
    frame["pred_direction"] = pred_direction
    frame["trade_side"] = pred_direction
    frame["selection_score_mode"] = MODEL_DIRECTION_SELECTION_MODE
    frame["direction_logits"] = [row.tolist() for row in logits]
    frame["public_trade_flat_decision_logits"] = [
        row.tolist() for row in pair_logits
    ]
    frame["public_trade_probability"] = pair_probabilities[:, 0]
    frame["public_flat_probability"] = pair_probabilities[:, 1]
    frame["public_trade_flat_margin"] = pair_logits[:, 0] - pair_logits[:, 1]
    frame["public_trade_flat_hard_decision"] = np.argmax(pair_logits, axis=1)
    for name, values in turning_point_prediction_columns(len(frame)).items():
        frame[name] = values
    for name, values in offline_rl_prediction_columns(len(frame)).items():
        frame[name] = values

    bundle = tmp_path / "bundle"
    event_dir = tmp_path / "prediction_event"
    bundle.mkdir(exist_ok=True)
    event_dir.mkdir(exist_ok=True)
    metadata = {
        "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "state_dict_sha256": "a" * 64,
        "model_native_signal_contract": _signal_contract(),
        "direction_decision_contract": model_direction_decision_contract_metadata(),
    }
    (bundle / "bundle_metadata.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    stamp = "20260716T120000123456Z"
    predictions_path = event_dir / f"selective_edge_predictions_{stamp}.parquet"
    atomic_write_parquet_immutable(frame, predictions_path)
    evidence = build_prediction_evidence_declaration(
        predictions_path=predictions_path,
        bundle_dir=bundle,
        bundle_metadata=metadata,
        requested_splits=["val", "test"],
    )
    dataset_splits: dict[str, dict[str, str]] = {}
    for split in ("val", "test"):
        parquet = (dataset_dir / f"tiny_{split}.parquet").resolve()
        manifest = parquet.with_suffix(".manifest.json")
        manifest.write_text(
            json.dumps({"output_data_path": str(parquet)}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        dataset_splits[split] = {
            "manifest_path": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "parquet_path": str(parquet),
            "parquet_sha256": sha256_file(parquet),
        }
    report_path = event_dir / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_{stamp}.json"
    report = {
        "schema_version": "entry_candidate_selective_edge_v1",
        "created_utc": "2026-07-16T12:00:00.123456+00:00",
        "decision": "PASS",
        "failures": [],
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "bundle_dir": str(bundle.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "splits": ["test", "val"],
        "models": ["candidate"],
        "feature_mask_ablation": {"enabled": False},
        "model_native_signal_contract": _signal_contract(),
        "dataset_signal_contract": {
            "contract": _signal_contract(),
            "splits": dataset_splits,
        },
        "bundle_seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "bundle_snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "bundle_specialist_fusion_enabled": True,
        "bundle_specialist_contract": _specialist_snapshot(),
        "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
        "direction_decision_contract": model_direction_decision_contract_metadata(),
        "prediction_evidence": evidence,
        "predictions_path": str(predictions_path.resolve()),
        "bundle_metadata_sha256": evidence["bundle_metadata_sha256"],
        "model_state_dict_sha256": evidence["model_state_dict_sha256"],
        "promotion_shadow_live_allowed": False,
        "json_path": str(report_path.resolve()),
    }
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    return predictions_path, report_path


def _prediction_frame(val_times: pd.DatetimeIndex, test_times: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["val"] * len(val_times) + ["test"] * len(test_times),
            "model": ["candidate"] * (len(val_times) + len(test_times)),
            "time": list(val_times) + list(test_times),
            "y_direction": [0, 1, 0, 1, 0, 0, 1, 1, 0, 2],
            "session": ["EU"] * 10,
            "vol_regime": ["1"] * 10,
            "p_long": [0.80, 0.10, 0.55, 0.20, 0.84, 0.82, 0.12, 0.13, 0.70, 0.05],
            "p_short": [0.10, 0.80, 0.20, 0.55, 0.10, 0.12, 0.82, 0.81, 0.20, 0.05],
            "p_flat": [0.10, 0.10, 0.25, 0.25, 0.06, 0.06, 0.06, 0.06, 0.10, 0.90],
            "path_quality_pred": [1.0] * 10,
            "bad_path_prob": [0.2] * 10,
            "position_size_pred": [0.25, 0.75] * 5,
        }
    )


def _source_frame(times: list[pd.Timestamp]) -> pd.DataFrame:
    values = np.linspace(100.0, 101.0, len(times))
    return pd.DataFrame(
        {
            "time": times,
            "bid_open": values,
            "ask_open": values + 0.02,
            "bid_close": values,
            "ask_close": values + 0.02,
            "bid_high": values + 0.05,
            "bid_low": values - 0.05,
            "ask_high": values + 0.07,
            "ask_low": values - 0.03,
        }
    )


def _run_args(
    *,
    state_path: Path,
    candidate_path: Path,
    predictions_path: Path,
    prediction_report_path: Path,
    dataset_dir: Path,
    source_path: Path,
    out_dir: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        model_native_state_json=str(state_path),
        candidate_readiness_json=str(candidate_path),
        selective_edge_predictions=str(predictions_path),
        selective_edge_report_json=str(prediction_report_path),
        dataset_dir=str(dataset_dir),
        source_parquet=str(source_path),
        out_dir=str(out_dir),
        model_name="candidate",
        cost_stress_bps=0.0,
        policy_id="candidate_replay",
        slippage_bps=0.0,
        quiet=True,
    )


def test_candidate_replay_parser_requires_all_explicit_lineage() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--selective-edge-predictions", "/tmp/timestamped.parquet"]
        )
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--model-native-state-json",
                "/tmp/state.json",
                "--candidate-readiness-json",
                "/tmp/candidate.json",
                "--selective-edge-predictions",
                "/tmp/predictions.parquet",
                "--selective-edge-report-json",
                "/tmp/report.json",
                "--dataset-dir",
                "/tmp/dataset",
                "--source-parquet",
                "/tmp/source.parquet",
                "--out-dir",
                "/tmp/out",
                "--threshold-top-fracs",
                "1.0",
            ]
        )
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            [
                "--model-native-state-json",
                "/tmp/state.json",
                "--candidate-readiness-json",
                "/tmp/candidate.json",
                "--selective-edge-predictions",
                "/tmp/predictions.parquet",
                "--selective-edge-report-json",
                "/tmp/report.json",
                "--dataset-dir",
                "/tmp/dataset",
                "--source-parquet",
                "/tmp/source.parquet",
                "--out-dir",
                "/tmp/out",
                "--exit-mode",
                "horizon",
            ]
        )


def test_candidate_replay_trade_log_is_unit_normalized_offline_diagnostic(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    val_times = pd.date_range("2025-12-31T23:00:00Z", periods=4, freq="5min")
    test_times = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="5min")
    pd.DataFrame({"time": val_times, "label_horizon_bars": [1] * 4}).to_parquet(
        dataset_dir / "tiny_val.parquet", index=False
    )
    pd.DataFrame({"time": test_times, "label_horizon_bars": [1] * 6}).to_parquet(
        dataset_dir / "tiny_test.parquet", index=False
    )
    predictions_path, report_path = _write_prediction_event(
        tmp_path, _prediction_frame(val_times, test_times), dataset_dir
    )
    state_path, candidate_path = _write_model_native_authority(tmp_path)
    source_path = tmp_path / "source.parquet"
    source_times = list(val_times) + list(test_times) + [
        test_times[-1] + pd.Timedelta(minutes=5),
        test_times[-1] + pd.Timedelta(minutes=10),
    ]
    _source_frame(source_times).to_parquet(source_path, index=False)
    out_dir = tmp_path / "out"

    report = run(
        _run_args(
            state_path=state_path,
            candidate_path=candidate_path,
            predictions_path=predictions_path,
            prediction_report_path=report_path,
            dataset_dir=dataset_dir,
            source_path=source_path,
            out_dir=out_dir,
        )
    )

    trades = pd.read_csv(report["trades_path"])
    assert report["decision"] == "PASS"
    assert Path(report["json_path"]).name.startswith(f"{TRADE_LOG_EVENT_PREFIX}_")
    assert report["trades_sha256"] == sha256_file(Path(report["trades_path"]))
    assert report["direction_policy_contract"]["filters_applied"] is False
    assert report["offline_only"] is True
    assert report["diagnostic_scope"] == OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE
    assert report["pnl_normalization"] == UNIT_NORMALIZED_PNL_MODE
    assert report["execution_order_simulation"] is False
    assert report["position_size_applied"] is False
    assert report["direction_policy_contract"]["offline_only"] is True
    assert report["direction_policy_contract"]["diagnostic_scope"] == (
        OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE
    )
    assert report["direction_policy_contract"]["pnl_normalization"] == (
        UNIT_NORMALIZED_PNL_MODE
    )
    assert report["policy_config"]["offline_only"] is True
    assert report["policy_config"]["execution_order_simulation"] is False
    assert report["exit_policy_contract"]["code_path"] == OFFLINE_REPLAY_EXECUTION_CODE_PATH
    assert report["exit_policy_contract"] == label_horizon_exit_policy_contract()
    assert report["n_test_rows"] == 6
    assert report["n_model_flat_rows"] == 1
    assert report["n_non_flat_argmax_rows"] == 5
    assert report["n_trades"] == 5
    assert report["trades_equal_non_flat_argmax_rows"] is True
    assert report["policy_counts"]["trades"] == 5
    assert report["policy_counts"]["non_flat_argmax_rows"] == 5
    assert report["policy_counts"]["filters_applied"] is False
    assert report["policy_counts"]["invalid_path_skip_allowed"] is False
    assert set(trades["diagnostic_scope"]) == {OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE}
    assert set(trades["pnl_normalization"]) == {UNIT_NORMALIZED_PNL_MODE}
    assert set(trades["offline_only"]) == {True}
    assert set(trades["execution_order_simulation"]) == {False}
    assert set(trades["position_size_applied"]) == {False}
    assert set(trades["exit_mode"]) == {LABEL_HORIZON_EXIT_MODE}
    assert set(trades["exit_reason"]) == {LABEL_HORIZON_EXIT_MODE}
    assert (trades["held_bars"] == trades["horizon_bars"]).all()
    assert set(trades["row_simulation_mode"]) == {"independent"}
    assert "position_size_pred" in trades.columns
    assert {"vol_regime", "path_quality_pred", "bad_path_prob"}.issubset(trades)
    assert not (out_dir / "CANDIDATE_REPLAY_TRADE_LOG_latest.json").exists()
    assert not (out_dir / "CANDIDATE_REPLAY_TRADE_LOG_MANIFEST.json").exists()
    assert not {"threshold_top_frac", "score_threshold"}.intersection(trades.columns)
    assert not {
        "dynamic_sizing_applied",
        "applied_size_multiplier",
        "replay_size_multiplier",
        "sizing_authority_contract",
    }.intersection(trades.columns)
    assert not {
        "dynamic_sizing_applied",
        "applied_size_multiplier",
        "replay_size_multiplier",
        "sizing_authority_contract",
    }.intersection(report)

    with pytest.raises(RuntimeError, match="new/empty"):
        run(
            _run_args(
                state_path=state_path,
                candidate_path=candidate_path,
                predictions_path=predictions_path,
                prediction_report_path=report_path,
                dataset_dir=dataset_dir,
                source_path=source_path,
                out_dir=out_dir,
            )
        )


def test_candidate_replay_requires_full_dataset_prediction_coverage(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    val_times = pd.date_range("2025-12-31T23:00:00Z", periods=4, freq="5min")
    test_times = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="5min")
    pd.DataFrame({"time": val_times, "label_horizon_bars": [1] * 4}).to_parquet(
        dataset_dir / "tiny_val.parquet", index=False
    )
    pd.DataFrame({"time": test_times, "label_horizon_bars": [1] * 6}).to_parquet(
        dataset_dir / "tiny_test.parquet", index=False
    )
    predictions_path, _ = _write_prediction_event(
        tmp_path, _prediction_frame(val_times, test_times), dataset_dir
    )
    incomplete = pd.read_parquet(predictions_path).iloc[:-1].copy()
    incomplete_path = tmp_path / "incomplete_predictions.parquet"
    incomplete.to_parquet(incomplete_path, index=False)

    with pytest.raises(RuntimeError, match="exactly cover all val/test dataset rows"):
        _prepare_predictions(
            incomplete_path,
            {
                "val": dataset_dir / "tiny_val.parquet",
                "test": dataset_dir / "tiny_test.parquet",
            },
            "candidate",
        )


def test_label_horizon_source_tape_fails_closed_on_incomplete_path(
    tmp_path: Path,
) -> None:
    times = list(pd.date_range("2026-01-01T00:00:00Z", periods=3, freq="5min"))
    source_path = tmp_path / "source.parquet"
    _source_frame(times).to_parquet(source_path, index=False)
    tape = SourceTape.load(source_path)

    trade = tape.simulate_trade(start_idx=0, horizon_bars=2, side=0)
    assert trade["entry_price"] == pytest.approx(100.02)
    assert trade["exit_src_idx"] == 2
    assert trade["held_bars"] == 2
    assert trade["exit_reason"] == LABEL_HORIZON_EXIT_MODE

    with pytest.raises(RuntimeError, match="full label horizon"):
        tape.simulate_trade(start_idx=1, horizon_bars=2, side=0)


def test_replay_sources_expose_no_retired_exit_or_filter_cli() -> None:
    root = Path(__file__).resolve().parents[1]
    trade_log_source = (
        root / "gx1/scripts/materialize_entry_candidate_replay_trade_log_v1.py"
    ).read_text(encoding="utf-8")
    execution_source = (root / OFFLINE_REPLAY_EXECUTION_CODE_PATH).read_text(
        encoding="utf-8"
    )

    for retired_cli in (
        "--exit-mode",
        "--take-profit-bps",
        "--stop-loss-bps",
        "--same-bar-policy",
        "--mfe-protect-activation-bps",
        "--cooldown-bars",
        "--max-trades-per-day",
        "--daily-loss-limit-bps",
        "--fail-on-audit-fail",
    ):
        assert retired_cli not in trade_log_source
    for retired_path in (
        "skipped_invalid_path",
        "skipped_open_or_cooldown",
        "unavailable_until_src_idx",
        "daily_trade_count",
        "daily_pnl_bps",
    ):
        assert retired_path not in trade_log_source
    assert "MODEL_NATIVE_APPLIED_SIZE_MULTIPLIER" not in trade_log_source
    assert "model_native_sizing_authority_contract_metadata" not in trade_log_source
    assert "def _split_file" not in trade_log_source
    assert 'glob(f"*_{split}' not in trade_log_source
    assert "def _first_stop_tp_hit" not in execution_source
    assert "def _first_stop_tp_mfe_protect_hit" not in execution_source
    assert "execution_sizing_authority" not in execution_source


def test_candidate_replay_rejects_utility_selection_surface() -> None:
    predictions = pd.DataFrame(
        {
            "selection_score_mode": ["expected_utility"],
            "p_long": [0.8],
            "p_short": [0.1],
            "p_flat": [0.1],
        }
    )
    with pytest.raises(RuntimeError, match="model_direction_argmax"):
        _resolve_score_surface(predictions)


def test_candidate_replay_rejects_stale_state_after_newer_event(tmp_path: Path) -> None:
    state_path, candidate_path = _write_model_native_authority(tmp_path)
    write_immutable_json_event(
        candidate_path.parent,
        CANDIDATE_EVENT_PREFIX,
        {
            "schema_version": "entry_candidate_readiness_model_native_v1",
            "created_utc": "2026-07-16T12:02:01.123456+00:00",
            "decision": "NOT_READY_FOR_CANDIDATE_TRAINING",
            "failures": ["newer failure"],
        },
    )
    from gx1.scripts.materialize_entry_candidate_replay_trade_log_v1 import (
        _validate_model_native_authority,
    )

    with pytest.raises(RuntimeError, match="not the newest"):
        _validate_model_native_authority(state_path, candidate_path)
