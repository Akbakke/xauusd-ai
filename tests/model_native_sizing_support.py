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
from gx1.contracts.entry_model_native_sizing_execution_v1 import (
    MODEL_NATIVE_JOINT_EXIT_SIZING_FACT_MODE,
    MODEL_NATIVE_JOINT_EXIT_TRACE_COLUMNS,
    candidate_bundle_authority,
    joint_exit_trace_sha256,
)
from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    CORE_ARTIFACTS as BUNDLE_COMMIT_CORE_ARTIFACTS,
    write_bundle_commit_manifest,
)
from gx1.contracts.entry_model_native_calibration_v1 import (
    IMMUTABLE_CALIBRATION_EVENT_SCHEMA_VERSION,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    canonical_unified_evidence_sha256,
    model_direction_decision_contract_metadata,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    build_prediction_evidence_declaration,
)
from gx1.scripts.finalize_entry_model_native_sizing_v1 import (
    bind_bundle_sizing_calibration,
    finalize_test_sizing_proof,
    fit_val_sizing_calibration,
    materialize_test_sizing_oos_source,
)
from tests.model_native_turning_point_support import turning_point_prediction_row
from tests.model_native_offline_rl_support import (
    add_test_runtime_calibration_metadata,
    offline_rl_prediction_row,
    runtime_head_prediction_columns,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_SIGNAL_DIM,
)


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


def default_offline_sizing_constraints() -> dict[str, Any]:
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
        "fact_provenance_mode": "canonical_oos_reference",
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
        "session_id": ("ASIA", "EU", "OVERLAP", "US").index(session),
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
    evidence_stage: str,
) -> tuple[Path, Path, dict[str, Any]]:
    root.mkdir(parents=True, exist_ok=True)
    predictions = root / f"selective_edge_predictions_{stamp}.parquet"
    metadata = json.loads((bundle_dir / "bundle_metadata.json").read_text())
    frame = frame.copy()
    if evidence_stage == "runtime_authoritative":
        for name, values in runtime_head_prediction_columns(
            frame,
            metadata,
        ).items():
            frame[name] = values
    frame.to_parquet(predictions, index=False)
    splits = sorted(frame["split"].astype(str).unique())
    evidence = build_prediction_evidence_declaration(
        predictions_path=predictions,
        bundle_dir=bundle_dir,
        bundle_metadata=metadata,
        evidence_stage=evidence_stage,
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
            "seq_input_dim": MODEL_NATIVE_SIGNAL_DIM,
            "snap_input_dim": MODEL_NATIVE_SIGNAL_DIM,
            "ordered_fields_sha256": "f" * 64,
        }
    report = {
        "schema_version": "entry_candidate_selective_edge_v1",
        "created_utc": datetime.strptime(
            stamp, "%Y%m%dT%H%M%S%fZ"
        ).replace(tzinfo=timezone.utc).isoformat(),
        "decision": "PASS",
        "failures": [],
        "evidence_stage": evidence_stage,
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
    run_lineage = {
        "schema_version": "entry_model_native_training_run_lineage_v2",
        "training_run_id": "UNIT_INITIAL_ADAPTATION_ADMISSION",
        "dataset_run_id": "UNIT_DATASET_RUN_20260717",
        "training_profile": "candidate",
        "requested_subsample_rows": 0,
        "physical_train_rows": 300,
        "effective_train_rows": 300,
    }
    source_metadata: dict[str, Any] = {
        "state_dict_sha256": state_sha,
        "direction_decision_contract": direction_contract,
        "run_lineage": run_lineage,
    }
    add_test_runtime_calibration_metadata(source_metadata)
    _write_json(source_bundle / "bundle_metadata.json", source_metadata)
    _write_json(
        source_bundle / "MASTER_TRANSFORMER_LOCK.json",
        {
            "model_path_relative": "model_state_dict.pt",
            "model_sha256": state_sha,
            "direction_decision_contract": direction_contract,
            "run_lineage": run_lineage,
        },
    )
    calibration_event_names = (
        "ENTRY_MODEL_NATIVE_CALIBRATION_20260717T090000123456Z.json",
        "ENTRY_MODEL_NATIVE_CALIBRATION_20260717T093000123456Z.json",
    )
    for head, name in zip(("direction", "path"), calibration_event_names, strict=True):
        _write_json(
            source_bundle / name,
            {
                "schema_version": IMMUTABLE_CALIBRATION_EVENT_SCHEMA_VERSION,
                "decision": "PASS",
                "head": head,
                "calibration": source_metadata[f"{head}_calibration"],
            },
        )
    write_bundle_commit_manifest(
        bundle_dir=source_bundle.resolve(),
        artifact_names=(*BUNDLE_COMMIT_CORE_ARTIFACTS, *calibration_event_names),
        bundle_kind="calibrated",
        created_at_utc="2026-07-17T00:00:00+00:00",
    )

    train_times = pd.date_range("2024-01-01T00:00:00Z", periods=300, freq="12h")
    val_times = pd.date_range("2025-01-01T00:00:00Z", periods=300, freq="12h")
    test_times = pd.date_range("2026-01-01T00:00:00Z", periods=384, freq="12h")
    tape_rows: list[dict[str, Any]] = []
    test_rows: list[dict[str, Any]] = []
    sessions = ("ASIA", "EU", "OVERLAP", "US")
    regimes = ("0", "1", "2")
    for index, ts in enumerate(test_times):
        direction = index % 3
        winner = index % 2 == 0
        logit = 2.0 if winner else -2.0
        test_rows.append(
            _prediction_row(
                timestamp=ts,
                split="test",
                logit=logit,
                direction=direction,
                target=float(1.0 / (1.0 + np.exp(-logit))),
                session=sessions[(index // 6) % len(sessions)],
                regime=regimes[(index // 6) % len(regimes)],
            )
        )
        entry_bid = 2499.9
        entry_ask = 2500.1
        if direction == 0:
            exit_bid = 2501.1 if winner else 2499.1
            exit_ask = exit_bid + 0.2
        else:
            exit_ask = 2498.9 if winner else 2500.9
            exit_bid = exit_ask - 0.2
        for minute in range(11):
            progress = max(0.0, (minute - 5) / 5.0)
            bid = entry_bid + progress * (exit_bid - entry_bid)
            ask = entry_ask + progress * (exit_ask - entry_ask)
            mid = (bid + ask) / 2.0
            tape_rows.append(
                {
                    "time": ts + pd.Timedelta(minutes=minute),
                    "open": mid,
                    "high": mid + 0.1,
                    "low": mid - 0.1,
                    "close": mid,
                    "bid_open": bid,
                    "ask_open": ask,
                    "bid_close": bid,
                    "ask_close": ask,
                    "bid_high": bid + 0.1,
                    "bid_low": bid - 0.1,
                    "ask_high": ask + 0.1,
                    "ask_low": ask - 0.1,
                    "volume": 100 + minute,
                }
            )
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
                # Label horizons are M5 bars even though replay reads the
                # higher-resolution authoritative M1 tape.
                "label_horizon_bars": np.full(len(times), 1, dtype=np.int64),
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
    for split, times in (("val", val_times),):
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
        evidence_stage="pre_calibration",
    )
    calibration_path, calibration = fit_val_sizing_calibration(
        predictions_path=fit_predictions,
        predictions_sha256=_sha(fit_predictions),
        prediction_report_path=fit_report,
        bundle_dir=source_bundle,
        dataset_dir=dataset_dir,
        authority_root=authority_root,
    )
    final_bundle = root / "final_bundle"
    bind_bundle_sizing_calibration(
        source_bundle_dir=source_bundle,
        output_bundle_dir=final_bundle,
        calibration_path=calibration_path,
    )
    test_predictions, test_report, _ = _write_prediction_event(
        root / "test_predictions",
        frame=pd.DataFrame(test_rows),
        bundle_dir=final_bundle,
        dataset_dir=dataset_dir,
        stamp="20260717T110000123456Z",
        evidence_stage="runtime_authoritative",
    )
    oos_path, oos_source = materialize_test_sizing_oos_source(
        calibration_path=calibration_path,
        test_predictions_path=test_predictions,
        test_predictions_sha256=_sha(test_predictions),
        test_prediction_report_path=test_report,
        bundle_dir=final_bundle,
        dataset_dir=dataset_dir,
        source_tape_path=tape,
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
        "offline_constraints": default_offline_sizing_constraints(),
    }


def write_passing_unified_replay_fixture(root: Path) -> dict[str, Any]:
    """Build exact full-TEST Entry M5 plus Exit M1 rows without fake authority."""

    evidence = write_passing_sizing_calibration_and_proof(root)
    bundle_authority = candidate_bundle_authority(
        bundle_dir=Path(evidence["proof"]["evaluation_bundle"]["bundle_dir"]),
        evaluation_bundle=evidence["proof"]["evaluation_bundle"],
        context="UNIT_CANDIDATE_BUNDLE_AUTHORITY",
    )
    candidate_bundle_sha = bundle_authority["bundle_commit_sha256"]
    source_rows_path = Path(
        evidence["oos_source"]["source_bindings"]["oos_rows"]["path"]
    )
    rows = pd.read_parquet(source_rows_path)
    rows["fact_provenance_mode"] = MODEL_NATIVE_JOINT_EXIT_SIZING_FACT_MODE
    directions = pd.to_numeric(rows["model_direction_index"]).astype(int)
    times = pd.to_datetime(rows["time"], utc=True)
    rows["entry_fill_time"] = times + pd.Timedelta(minutes=5)
    rows["exit_replay_status"] = np.where(
        directions.isin([0, 1]), "EXIT_NOW", "FLAT_NO_ORDER"
    )
    rows["model_exit_fill_time"] = [
        (timestamp + pd.Timedelta(minutes=15)).isoformat()
        if direction in (0, 1)
        else None
        for timestamp, direction in zip(times, directions, strict=True)
    ]
    rows["model_exit_decision_bar_time"] = [
        (timestamp + pd.Timedelta(minutes=14)).isoformat()
        if direction in (0, 1)
        else None
        for timestamp, direction in zip(times, directions, strict=True)
    ]
    model_exit_shift = np.where(
        directions == 0,
        0.02,
        np.where(directions == 1, -0.02, np.nan),
    )
    rows["model_exit_fill_bid"] = rows["exit_bid"] + model_exit_shift
    rows["model_exit_fill_ask"] = rows["exit_ask"] + model_exit_shift
    rows["exit_reason"] = np.where(
        directions.isin([0, 1]), "UNIFIED_MODEL_ARGMAX", "MODEL_FLAT"
    )
    rows["exit_steps"] = np.where(directions.isin([0, 1]), 10, 0)
    rows["candidate_bundle_sha256"] = candidate_bundle_sha
    flat_mask = directions == 2
    trace_records: list[dict[str, Any]] = []
    for row_index, row in rows.loc[~flat_mask].iterrows():
        direction = int(row["model_direction_index"])
        entry_time = pd.Timestamp(row["entry_fill_time"])
        entry_snapshot_sha = hashlib.sha256(
            str(row["reference_row_id"]).encode("utf-8")
        ).hexdigest()
        for step in range(1, int(row["exit_steps"]) + 1):
            state_fraction = step / float(row["exit_steps"])
            state_bid = float(row["entry_bid"]) + state_fraction * (
                float(row["model_exit_fill_bid"]) - float(row["entry_bid"])
            )
            state_ask = float(row["entry_ask"]) + state_fraction * (
                float(row["model_exit_fill_ask"]) - float(row["entry_ask"])
            )
            state_pnl_bps = (
                (state_bid - float(row["entry_ask"]))
                / float(row["entry_ask"])
                * 10_000.0
                if direction == 0
                else (float(row["entry_bid"]) - state_ask)
                / float(row["entry_bid"])
                * 10_000.0
            )
            exit_now = step == int(row["exit_steps"])
            logits = [-2.0, 2.0] if exit_now else [2.0, -2.0]
            probs = _softmax(np.asarray(logits, dtype=np.float64)).tolist()
            action_id = 1 if exit_now else 0
            action = "EXIT_NOW" if exit_now else "HOLD"
            path_sha = hashlib.sha256(
                f"{row['reference_row_id']}:{step}".encode("utf-8")
            ).hexdigest()
            output = {
                "exit_action_logits": logits,
                "exit_action_probs": probs,
                "exit_action_index": action_id,
                "action": action,
                "decision_source": "unified_model",
                "bundle_sha256": candidate_bundle_sha,
                "entry_snapshot_sha256": entry_snapshot_sha,
                "exit_path_envelope_sha256": path_sha,
            }
            trace_records.append(
                {
                    "reference_row_id": str(row["reference_row_id"]),
                    "entry_fill_time": entry_time,
                    "step": step,
                    "closed_bar_time": (
                        entry_time + pd.Timedelta(minutes=step - 1)
                    ),
                    "model_exit_fill_time": (
                        entry_time + pd.Timedelta(minutes=step)
                    ),
                    "bar_committed": True,
                    "action_id": action_id,
                    "action": action,
                    "decision_source": "unified_model",
                    "state_pnl_bps": state_pnl_bps,
                    "state_bid": state_bid,
                    "state_ask": state_ask,
                    "exit_hold_logit": logits[0],
                    "exit_now_logit": logits[1],
                    "exit_hold_prob": probs[0],
                    "exit_now_prob": probs[1],
                    "candidate_bundle_sha256": candidate_bundle_sha,
                    "entry_snapshot_sha256": entry_snapshot_sha,
                    "exit_path_envelope_sha256": path_sha,
                    "output_evidence_sha256": (
                        canonical_unified_evidence_sha256(output)
                    ),
                    "closed_m1_source_path": evidence["oos_source"][
                        "source_tape"
                    ]["path"],
                    "closed_m1_source_sha256": evidence["oos_source"][
                        "source_tape"
                    ]["sha256"],
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
    evidence.update(
        {
            "candidate_bundle_authority": bundle_authority,
            "joint_replay_rows_path": replay_rows_path,
            "joint_exit_trace_rows_path": exit_trace_rows_path,
        }
    )
    return evidence


def next_created_utc(created_utc: str, *, microseconds: int = 1) -> str:
    parsed = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
    return datetime.fromtimestamp(
        parsed.timestamp() + microseconds / 1_000_000.0, tz=timezone.utc
    ).isoformat()
