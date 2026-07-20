from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    learned_sizing_authority_contract_metadata,
)
from gx1.contracts.entry_model_native_sizing_calibration_v1 import (
    calibrated_sizing_transform,
)
from gx1.contracts.entry_model_native_sizing_execution_v1 import (
    MODEL_NATIVE_JOINT_EXIT_SIZING_FACT_MODE,
    MODEL_NATIVE_JOINT_EXIT_TRACE_COLUMNS,
    joint_exit_trace_sha256,
)
from gx1.contracts.immutable_event_authority_v1 import write_immutable_json_event
from gx1.contracts.model_native_serve_gate_v1 import (
    MODEL_NATIVE_SERVE_GATE_CONTRACT_VERSION,
    MODEL_NATIVE_SERVE_PARITY_SCHEMA_VERSION,
    SERVE_PARITY_ENV_PINS,
    SERVE_PARITY_FORWARD_TOL,
    SERVE_PARITY_SAMPLE_COUNT,
    SERVE_PARITY_SAMPLING_CONTRACT,
    SERVE_PARITY_STATE_TOL,
    UTC_TIME_COVERAGE_SCHEMA_VERSION,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    model_direction_decision_contract_metadata,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    build_prediction_evidence_declaration,
)
from gx1.scripts.finalize_entry_model_native_sizing_v1 import (
    bind_bundle_sizing_calibration,
    capture_oanda_instrument_evidence,
    adopt_learned_sizing,
    finalize_joint_exit_sizing_proof,
    finalize_runtime_sizing_parity,
    finalize_test_sizing_proof,
    fit_train_val_sizing_calibration,
    materialize_test_sizing_oos_source,
)
from tests.model_native_serve_gate_support import (
    passing_serve_parity_liveness_sections,
)
from tests.model_native_turning_point_support import turning_point_prediction_row
from tests.model_native_offline_rl_support import offline_rl_prediction_row


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _binding(path: Path) -> dict[str, str]:
    return {"json_path": str(path.resolve()), "sha256": _sha(path)}


def _source_binding(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": _sha(path)}


class _FakeOandaClient:
    def get_account_summary(self) -> dict[str, Any]:
        return {"account": {"currency": "USD"}, "lastTransactionID": "100"}

    def get_account_instruments(self, instruments: list[str]) -> dict[str, Any]:
        assert instruments == ["XAU_USD"]
        return {
            "instruments": [
                {
                    "name": "XAU_USD",
                    "tradeUnitsPrecision": 0,
                    "minimumTradeSize": "1",
                    "maximumOrderUnits": "100000",
                    "marginRate": "0.05",
                }
            ],
            "lastTransactionID": "101",
        }


def default_runtime_sizing_constraints() -> dict[str, Any]:
    observed = datetime.now(timezone.utc).isoformat()
    return {
        "instrument": "XAU_USD",
        "account_currency": "USD",
        "account_equity": 10_000.0,
        "account_balance": 10_000.0,
        "account_floating_drawdown_bps": 0.0,
        "margin_available": 1_000.0,
        "margin_used": 0.0,
        "mark_price": 2_500.0,
        "margin_rate": 0.05,
        "unit_step": 1,
        "minimum_order_units": 1,
        "maximum_gross_xau_units": 1_000,
        "current_xau_abs_units": 0.0,
        "sizing_decision_utc": observed,
        "account_observed_utc": observed,
        "instrument_observed_utc": observed,
        "exposure_observed_utc": observed,
        "account_last_transaction_id": "snapshot-9001",
        "instrument_last_transaction_id": "snapshot-9001",
        "exposure_last_transaction_id": "snapshot-9001",
        "fact_provenance_mode": "broker_live",
    }


def unverified_learned_sizing_authority() -> dict[str, Any]:
    return learned_sizing_authority_contract_metadata(
        adoption_artifact={
            "json_path": (
                "/tmp/ENTRY_MODEL_NATIVE_SIZING_ADOPTION_"
                "20260717T120200123456Z.json"
            ),
            "sha256": "a" * 64,
        }
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(shifted)
    return exp / exp.sum()


def _prediction_row(
    *,
    timestamp: pd.Timestamp,
    split: str,
    logit: float,
    direction: int,
    target: float,
    session: str,
    regime: str,
) -> dict[str, Any]:
    direction_logits = np.full(3, -2.0, dtype=np.float64)
    direction_logits[direction] = 2.0
    direction_probs = _softmax(direction_logits)
    public_logits = np.asarray(
        [max(direction_logits[0], direction_logits[1]), direction_logits[2]],
        dtype=np.float64,
    )
    public_probs = _softmax(public_logits)
    return {
        "split": split,
        "model": "candidate",
        "time": timestamp.isoformat(),
        "y_direction": direction,
        "pred_direction": direction,
        "p_long": float(direction_probs[0]),
        "p_short": float(direction_probs[1]),
        "p_flat": float(direction_probs[2]),
        "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
        "public_trade_probability": float(public_probs[0]),
        "public_flat_probability": float(public_probs[1]),
        "public_trade_flat_margin": float(public_logits[0] - public_logits[1]),
        "public_trade_flat_hard_decision": int(np.argmax(public_logits)),
        "direction_logits": direction_logits.tolist(),
        "public_trade_flat_decision_logits": public_logits.tolist(),
        "position_size_logit": float(logit),
        "position_size_pred": float(1.0 / (1.0 + np.exp(-logit))),
        "y_position_size_target": float(target),
        "session": session,
        "vol_regime": regime,
        **turning_point_prediction_row(0),
        **offline_rl_prediction_row(0),
    }


def _write_dataset_manifest(
    path: Path,
    *,
    output_data_path: Path,
    coverage: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    source_tape: Path,
) -> None:
    _write_json(
        path,
        {
            "output_data_path": str(output_data_path.resolve()),
            "ts_min_max_by_split": {
                split: {"ts_min": str(bounds[0]), "ts_max": str(bounds[1])}
                for split, bounds in coverage.items()
            },
            "extra": {
                "model_native_state_contract": {
                    "rank_reference_source_parquet": str(source_tape.resolve()),
                    "rank_reference_source_parquet_sha256": _sha(source_tape),
                }
            },
        },
    )


def _write_prediction_event(
    root: Path,
    *,
    frame: pd.DataFrame,
    bundle_dir: Path,
    dataset_dir: Path,
    stamp: str,
) -> tuple[Path, Path, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    predictions = root / f"selective_edge_predictions_{stamp}.parquet"
    frame.to_parquet(predictions, index=False)
    metadata = json.loads((bundle_dir / "bundle_metadata.json").read_text())
    splits = sorted(frame["split"].astype(str).unique())
    evidence = build_prediction_evidence_declaration(
        predictions_path=predictions,
        bundle_dir=bundle_dir,
        bundle_metadata=metadata,
        requested_splits=splits,
    )
    report_path = root / f"ENTRY_CANDIDATE_SELECTIVE_EDGE_{stamp}.json"
    dataset_splits: dict[str, Any] = {}
    for split in splits:
        manifest = dataset_dir / f"entry_model_native_{split}.manifest.json"
        parquet = dataset_dir / f"entry_model_native_{split}.parquet"
        dataset_splits[split] = {
            "manifest_path": str(manifest.resolve()),
            "manifest_sha256": _sha(manifest),
            "parquet_path": str(parquet.resolve()),
            "parquet_sha256": _sha(parquet),
            "seq_input_dim": 513,
            "snap_input_dim": 513,
            "ordered_fields_sha256": "f" * 64,
        }
    report = {
        "schema_version": "entry_candidate_selective_edge_v1",
        "created_utc": datetime.strptime(
            stamp, "%Y%m%dT%H%M%S%fZ"
        ).replace(tzinfo=timezone.utc).isoformat(),
        "decision": "PASS",
        "failures": [],
        "json_path": str(report_path.resolve()),
        "predictions_path": str(predictions.resolve()),
        "prediction_evidence": evidence,
        "bundle_dir": str(bundle_dir.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
        "direction_decision_contract": metadata["direction_decision_contract"],
        "bundle_metadata_sha256": _sha(bundle_dir / "bundle_metadata.json"),
        "model_state_dict_sha256": _sha(bundle_dir / "model_state_dict.pt"),
        "splits": splits,
        "models": ["candidate"],
        "dataset_signal_contract": {"contract": {}, "splits": dataset_splits},
    }
    _write_json(report_path, report)
    return predictions, report_path, evidence


def _coverage(times: pd.DatetimeIndex) -> dict[str, Any]:
    values = np.asarray(times.asi8, dtype="<i8")
    return {
        "schema_version": UTC_TIME_COVERAGE_SCHEMA_VERSION,
        "rows": len(times),
        "first_utc": times[0].isoformat(),
        "last_utc": times[-1].isoformat(),
        "utc_ns_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def _write_model_head_serve_parity(
    root: Path,
    *,
    bundle_dir: Path,
    dataset_dir: Path,
    predictions: Path,
    prediction_report: Path,
    prediction_evidence: dict[str, Any],
    test_times: pd.DatetimeIndex,
) -> Path:
    pick = (
        np.arange(SERVE_PARITY_SAMPLE_COUNT, dtype=np.int64) * (len(test_times) - 1)
    ) // (SERVE_PARITY_SAMPLE_COUNT - 1)
    metadata_sha = _sha(bundle_dir / "bundle_metadata.json")
    lock_sha = _sha(bundle_dir / "MASTER_TRANSFORMER_LOCK.json")
    liveness = passing_serve_parity_liveness_sections(
        len(test_times),
        bundle_dir=str(bundle_dir.resolve()),
        bundle_metadata_sha256=metadata_sha,
        master_transformer_lock_sha256=lock_sha,
    )
    common = {
        "contract_version": MODEL_NATIVE_SERVE_GATE_CONTRACT_VERSION,
        "split": "test",
        "model_name": "candidate",
        "bundle_dir": str(bundle_dir.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_parquet": str(
            (dataset_dir / "entry_model_native_test.parquet").resolve()
        ),
        "dataset_parquet_sha256": _sha(
            dataset_dir / "entry_model_native_test.parquet"
        ),
        "prediction_evidence": prediction_evidence,
        "prediction_report_evidence": _binding(prediction_report),
        "test_coverage": {
            "dataset": _coverage(test_times),
            "predictions": _coverage(test_times),
            "exact_match": True,
        },
    }
    path, _ = write_immutable_json_event(
        root,
        "MODEL_NATIVE_SERVE_PARITY",
        {
            "schema_version": MODEL_NATIVE_SERVE_PARITY_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "decision": "PASS",
            "failures": [],
            **common,
            "pinned_predictions": str(predictions.resolve()),
            "n_bars": SERVE_PARITY_SAMPLE_COUNT,
            "sampling_contract": SERVE_PARITY_SAMPLING_CONTRACT,
            "state_tol": SERVE_PARITY_STATE_TOL,
            "forward_tol": SERVE_PARITY_FORWARD_TOL,
            "env_pins": dict(SERVE_PARITY_ENV_PINS),
            "sampled_test_coverage": _coverage(test_times[pick]),
            "state_parity": {
                "n_compared": SERVE_PARITY_SAMPLE_COUNT,
                "tolerance": SERVE_PARITY_STATE_TOL,
            },
            "forward_parity": {
                "n_compared": SERVE_PARITY_SAMPLE_COUNT,
                "tolerance": SERVE_PARITY_FORWARD_TOL,
                "per_head_tolerance": liveness[
                    "forward_parity_per_head_tolerance"
                ],
            },
            "direction_calibration_parity": liveness[
                "direction_calibration_parity"
            ],
            "test_prediction_liveness": liveness["test_prediction_liveness"],
            "specialist_decision_influence": liveness[
                "specialist_decision_influence"
            ],
            "upstream_context_decision_influence": liveness[
                "upstream_context_decision_influence"
            ],
            "multi_tf_decision_influence": liveness[
                "multi_tf_decision_influence"
            ],
            "direction_evidence_fusion_influence": liveness[
                "direction_evidence_fusion_influence"
            ],
        },
    )
    return path


def write_passing_sizing_calibration_and_proof(root: Path) -> dict[str, Any]:
    root = root.resolve()
    authority_root = root / "authority"
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    source_bundle = root / "source_bundle"
    source_bundle.mkdir(parents=True, exist_ok=True)
    state = source_bundle / "model_state_dict.pt"
    state.write_bytes(b"canonical sizing checkpoint")
    state_sha = _sha(state)
    direction_contract = model_direction_decision_contract_metadata()
    _write_json(
        source_bundle / "bundle_metadata.json",
        {
            "state_dict_sha256": state_sha,
            "direction_decision_contract": direction_contract,
        },
    )
    _write_json(
        source_bundle / "MASTER_TRANSFORMER_LOCK.json",
        {
            "model_path_relative": "model_state_dict.pt",
            "model_sha256": state_sha,
            "direction_decision_contract": direction_contract,
        },
    )

    train_times = pd.date_range("2024-01-01T00:00:00Z", periods=300, freq="12h")
    val_times = pd.date_range("2025-01-01T00:00:00Z", periods=300, freq="12h")
    test_times = pd.date_range("2026-01-01T00:00:00Z", periods=360, freq="12h")
    tape_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    sessions = ("ASIA", "EU", "OVERLAP", "US")
    regimes = ("0", "1", "2")
    for index, ts in enumerate(test_times):
        direction = (index // 2) % 2
        winner = index % 2 == 0
        logit = 2.0 if winner else -2.0
        test_rows.append(
            _prediction_row(
                timestamp=ts,
                split="test",
                logit=logit,
                direction=direction,
                target=float(1.0 / (1.0 + np.exp(-logit))),
                session=sessions[(index // 2) % len(sessions)],
                regime=regimes[(index // 2) % len(regimes)],
            )
        )
        decision = {
            "time": ts,
            "bid_open": 2499.9,
            "ask_open": 2500.1,
            "bid_close": 2499.9,
            "ask_close": 2500.1,
            "bid_high": 2500.0,
            "bid_low": 2499.8,
            "ask_high": 2500.2,
            "ask_low": 2500.0,
        }
        fill = {**decision, "time": ts + pd.Timedelta(minutes=5)}
        if direction == 0:
            exit_bid = 2501.1 if winner else 2499.1
            exit_ask = exit_bid + 0.2
        else:
            exit_ask = 2498.9 if winner else 2500.9
            exit_bid = exit_ask - 0.2
        exit_row = {
            **decision,
            "time": ts + pd.Timedelta(minutes=10),
            "bid_close": exit_bid,
            "ask_close": exit_ask,
            "bid_high": max(exit_bid, exit_ask),
            "bid_low": min(exit_bid, exit_ask),
            "ask_high": max(exit_bid, exit_ask),
            "ask_low": min(exit_bid, exit_ask),
        }
        tape_rows.extend((decision, fill, exit_row))
    tape = root / "canonical_source_tape.parquet"
    pd.DataFrame(tape_rows).to_parquet(tape, index=False)

    split_times = {"train": train_times, "val": val_times, "test": test_times}
    coverage = {
        split: (times[0], times[-1]) for split, times in split_times.items()
    }
    for split, times in split_times.items():
        frame = pd.DataFrame(
            {
                "time": times,
                "y_position_size_target": np.linspace(0.1, 0.9, len(times)),
                "label_horizon_bars": np.ones(len(times), dtype=np.int64),
            }
        )
        if split == "test":
            frame["y_position_size_target"] = [
                row["y_position_size_target"] for row in test_rows
            ]
        parquet = dataset_dir / f"entry_model_native_{split}.parquet"
        frame.to_parquet(parquet, index=False)
        _write_dataset_manifest(
            dataset_dir / f"entry_model_native_{split}.manifest.json",
            output_data_path=parquet,
            coverage=coverage,
            source_tape=tape,
        )

    fit_rows: list[dict[str, Any]] = []
    for split, times in (("train", train_times), ("val", val_times)):
        targets = pd.read_parquet(
            dataset_dir / f"entry_model_native_{split}.parquet"
        )["y_position_size_target"].to_numpy(float)
        logits = np.log(np.clip(targets, 1e-6, 1 - 1e-6) / np.clip(1 - targets, 1e-6, 1))
        for index, (ts, logit, target) in enumerate(zip(times, logits, targets, strict=True)):
            fit_rows.append(
                _prediction_row(
                    timestamp=ts,
                    split=split,
                    logit=float(logit),
                    direction=index % 2,
                    target=float(target),
                    session=sessions[index % len(sessions)],
                    regime=regimes[index % len(regimes)],
                )
            )
    fit_predictions, fit_report, _ = _write_prediction_event(
        root / "fit_predictions",
        frame=pd.DataFrame(fit_rows),
        bundle_dir=source_bundle,
        dataset_dir=dataset_dir,
        stamp="20260717T100000123456Z",
    )
    instrument_path, _ = capture_oanda_instrument_evidence(
        authority_root=authority_root, client=_FakeOandaClient()
    )
    calibration_path, calibration = fit_train_val_sizing_calibration(
        predictions_path=fit_predictions,
        prediction_report_path=fit_report,
        bundle_dir=source_bundle,
        dataset_dir=dataset_dir,
        dataset_manifest_path=dataset_dir / "entry_model_native_train.manifest.json",
        instrument_evidence_path=instrument_path,
        authority_root=authority_root,
    )
    final_bundle = root / "final_bundle"
    bind_bundle_sizing_calibration(
        source_bundle_dir=source_bundle,
        output_bundle_dir=final_bundle,
        calibration_path=calibration_path,
    )
    test_predictions, test_report, test_evidence = _write_prediction_event(
        root / "test_predictions",
        frame=pd.DataFrame(test_rows),
        bundle_dir=final_bundle,
        dataset_dir=dataset_dir,
        stamp="20260717T110000123456Z",
    )
    parity_path = _write_model_head_serve_parity(
        root / "model_head_parity",
        bundle_dir=final_bundle,
        dataset_dir=dataset_dir,
        predictions=test_predictions,
        prediction_report=test_report,
        prediction_evidence=test_evidence,
        test_times=test_times,
    )
    oos_path, oos_source = materialize_test_sizing_oos_source(
        calibration_path=calibration_path,
        test_predictions_path=test_predictions,
        test_prediction_report_path=test_report,
        bundle_dir=final_bundle,
        dataset_dir=dataset_dir,
        source_tape_path=tape,
        model_head_serve_parity_path=parity_path,
        authority_root=authority_root,
    )
    proof_path, proof = finalize_test_sizing_proof(
        calibration_path=calibration_path,
        oos_source_path=oos_path,
        authority_root=authority_root,
    )
    return {
        "authority_root": authority_root,
        "bundle_dir": final_bundle,
        "calibration": calibration,
        "calibration_artifact": _binding(calibration_path),
        "proof": proof,
        "oos_proof_artifact": _binding(proof_path),
        "oos_source": oos_source,
        "oos_source_artifact": _binding(oos_path),
        "source_bindings": oos_source["source_bindings"],
        "bundle_calibration": json.loads(
            (final_bundle / "bundle_metadata.json").read_text()
        )["model_native_sizing_calibration"],
        "runtime_constraints": default_runtime_sizing_constraints(),
    }


def write_passing_joint_exit_sizing_proof(root: Path) -> dict[str, Any]:
    """Extend the canonical sizing fixture with strict full-TEST Exit traces."""

    evidence = write_passing_sizing_calibration_and_proof(root)
    active_paths: dict[str, str] = {}
    for role in ("xgb", "v3_exit", "exit_iql"):
        path = root / "active_exit" / role
        path.mkdir(parents=True, exist_ok=True)
        (path / "identity.bin").write_bytes(role.encode("utf-8"))
        active_paths[role] = str(path.resolve())
    registry_path = root / "PROJECT_STATE_artifacts.json"
    _write_json(
        registry_path,
        {
            "schema_version": "gx1_artifact_selection_v2",
            "project": "XAUUSD",
            "updated_utc": "2026-07-17T12:00:00Z",
            "active": {
                role: {
                    "path": active_paths[role],
                    "status": "ACTIVE",
                    "in_sample_only": False,
                }
                for role in ("xgb", "v3_exit", "exit_iql")
            },
        },
    )
    registry_sha = _sha(registry_path)
    source_rows_path = Path(
        evidence["oos_source"]["source_bindings"]["oos_rows"]["path"]
    )
    rows = pd.read_parquet(source_rows_path)
    rows["fact_provenance_mode"] = MODEL_NATIVE_JOINT_EXIT_SIZING_FACT_MODE
    directions = pd.to_numeric(rows["model_direction_index"]).astype(int)
    times = pd.to_datetime(rows["time"], utc=True)
    rows["exit_replay_status"] = np.where(
        directions.isin([0, 1]), "EXIT_NOW", "FLAT_NO_ORDER"
    )
    rows["exit_time"] = [
        (timestamp + pd.Timedelta(minutes=10)).isoformat()
        if direction in (0, 1)
        else None
        for timestamp, direction in zip(times, directions, strict=True)
    ]
    rows["exit_reason"] = np.where(
        directions.isin([0, 1]), "EXIT_IQL_ARGMAX", "MODEL_FLAT"
    )
    rows["exit_steps"] = np.where(directions.isin([0, 1]), 10, 0)
    rows["active_exit_registry_sha256"] = registry_sha
    flat_mask = directions == 2
    rows.loc[flat_mask, "exit_bid"] = rows.loc[flat_mask, "entry_bid"]
    rows.loc[flat_mask, "exit_ask"] = rows.loc[flat_mask, "entry_ask"]
    trace_records: list[dict[str, Any]] = []
    for row_index, row in rows.loc[~flat_mask].iterrows():
        direction = int(row["model_direction_index"])
        entry_time = pd.Timestamp(row["time"])
        for step in range(1, int(row["exit_steps"]) + 1):
            fraction = step / float(row["exit_steps"])
            bid = float(row["entry_bid"]) + fraction * (
                float(row["exit_bid"]) - float(row["entry_bid"])
            )
            ask = float(row["entry_ask"]) + fraction * (
                float(row["exit_ask"]) - float(row["entry_ask"])
            )
            current_pnl_bps = (
                (bid - float(row["entry_ask"]))
                / float(row["entry_ask"])
                * 10_000.0
                if direction == 0
                else (float(row["entry_bid"]) - ask)
                / float(row["entry_bid"])
                * 10_000.0
            )
            trace_records.append(
                {
                    "reference_row_id": str(row["reference_row_id"]),
                    "entry_time": entry_time,
                    "step": step,
                    "bar_time": entry_time + pd.Timedelta(minutes=step),
                    "action_id": 1 if step == int(row["exit_steps"]) else 0,
                    "decision_source": (
                        str(row["exit_reason"])
                        if step == int(row["exit_steps"])
                        else "HOLD"
                    ),
                    "current_pnl_bps": current_pnl_bps,
                    "bid": bid,
                    "ask": ask,
                    "active_exit_registry_sha256": registry_sha,
                }
            )
    exit_trace_rows = pd.DataFrame(
        trace_records, columns=sorted(MODEL_NATIVE_JOINT_EXIT_TRACE_COLUMNS)
    )
    exit_trace_rows_path = (
        root / "joint_exit_trace_rows_20260717T120000123456Z.parquet"
    )
    exit_trace_rows.to_parquet(exit_trace_rows_path, index=False)
    exit_trace_rows = pd.read_parquet(exit_trace_rows_path)
    rows["exit_trace_sha256"] = hashlib.sha256(b"FLAT_NO_ORDER").hexdigest()
    for reference_row_id, trace in exit_trace_rows.groupby(
        "reference_row_id", sort=False
    ):
        rows.loc[
            rows.index[rows["reference_row_id"].astype(str) == str(reference_row_id)],
            "exit_trace_sha256",
        ] = joint_exit_trace_sha256(
            trace.sort_values("step", kind="mergesort").reset_index(drop=True),
            context="UNIT_JOINT_EXIT_TRACE_FIXTURE",
        )
    replay_rows_path = root / "joint_exit_replay_rows_20260717T120000123456Z.parquet"
    rows.to_parquet(replay_rows_path, index=False)
    joint_path, joint_proof = finalize_joint_exit_sizing_proof(
        calibration_path=Path(evidence["calibration_artifact"]["json_path"]),
        proof_path=Path(evidence["oos_proof_artifact"]["json_path"]),
        replay_rows_path=replay_rows_path,
        exit_trace_rows_path=exit_trace_rows_path,
        artifact_registry_path=registry_path,
        authority_root=evidence["authority_root"],
    )
    evidence.update(
        {
            "artifact_registry_path": registry_path,
            "joint_replay_rows_path": replay_rows_path,
            "joint_exit_trace_rows_path": exit_trace_rows_path,
            "joint_exit_proof": joint_proof,
            "joint_exit_proof_artifact": _binding(joint_path),
        }
    )
    return evidence


def write_passing_runtime_sizing_parity(root: Path) -> dict[str, Any]:
    """Extend the joint fixture through adoption and broker-live shadow parity."""

    evidence = write_passing_joint_exit_sizing_proof(root)
    adoption_path, adoption = adopt_learned_sizing(
        bundle_dir=evidence["bundle_dir"],
        calibration_path=Path(evidence["calibration_artifact"]["json_path"]),
        proof_path=Path(evidence["oos_proof_artifact"]["json_path"]),
        joint_exit_proof_path=Path(
            evidence["joint_exit_proof_artifact"]["json_path"]
        ),
        authority_root=evidence["authority_root"],
        accepted_via_vedtak="UNIT_RUNTIME_SIZING_ADOPTION",
    )
    adoption_binding = _binding(adoption_path)
    adoption_created = pd.Timestamp(adoption["created_utc"])
    directions = [index % 3 for index in range(36)]
    logits = np.linspace(-3.0, 3.0, len(directions))
    rows: list[dict[str, Any]] = []
    for index, (direction, logit) in enumerate(zip(directions, logits, strict=True)):
        timestamp = adoption_created + pd.Timedelta(microseconds=index + 1)
        transaction_id = f"broker-snapshot-{index // 12 + 1}"
        constraints = default_runtime_sizing_constraints()
        constraints.update(
            {
                "sizing_decision_utc": timestamp.isoformat(),
                "account_observed_utc": timestamp.isoformat(),
                "instrument_observed_utc": timestamp.isoformat(),
                "exposure_observed_utc": timestamp.isoformat(),
                "account_last_transaction_id": transaction_id,
                "instrument_last_transaction_id": transaction_id,
                "exposure_last_transaction_id": transaction_id,
            }
        )
        transformed = calibrated_sizing_transform(
            calibration=evidence["calibration"],
            position_size_logit=float(logit),
            model_direction_index=direction,
            runtime_constraints=constraints,
            context=f"UNIT_RUNTIME_PARITY_ROW_{index}",
        )
        rows.append(
            {
                "time": timestamp.isoformat(),
                "position_size_logit": float(logit),
                "model_direction_index": direction,
                "direction_after_sizing": direction,
                **constraints,
                **{
                    field: transformed[field]
                    for field in (
                        "calibrated_size_fraction",
                        "applied_size_multiplier",
                        "capacity_units",
                        "reference_pre_round_units",
                        "pre_round_units",
                        "units",
                        "authorized_order",
                        "no_order_reason",
                    )
                },
                "runtime_bundle_metadata_sha256": adoption[
                    "bundle_metadata_sha256"
                ],
                "runtime_model_state_dict_sha256": adoption[
                    "model_state_dict_sha256"
                ],
                "runtime_adoption_sha256": adoption_binding["sha256"],
                "order_submitted": False,
            }
        )
    observations_path = root / "runtime_sizing_observations_20260717T130000123456Z.parquet"
    pd.DataFrame(rows).to_parquet(observations_path, index=False)
    runtime_path, runtime_parity = finalize_runtime_sizing_parity(
        adoption_path=adoption_path,
        observations_path=observations_path,
        authority_root=evidence["authority_root"],
    )
    evidence.update(
        {
            "adoption": adoption,
            "adoption_artifact": adoption_binding,
            "runtime_sizing_observations_path": observations_path,
            "runtime_sizing_parity": runtime_parity,
            "runtime_sizing_parity_artifact": _binding(runtime_path),
        }
    )
    return evidence


def next_created_utc(created_utc: str, *, microseconds: int = 1) -> str:
    parsed = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
    return datetime.fromtimestamp(
        parsed.timestamp() + microseconds / 1_000_000.0, tz=timezone.utc
    ).isoformat()
