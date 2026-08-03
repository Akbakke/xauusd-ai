#!/usr/bin/env python3
"""Publish an immutable model-native seq513 candidate replay trade log.

This is an offline evidence writer.  The sole LONG/SHORT/FLAT authority is the
persisted argmax of the candidate's final calibrated ``direction_logits``.
Structure, trend, liquidity, volatility, momentum, session, price-action,
path-quality, utility, tradability, and sizing outputs remain learned model
evidence, but none may filter, threshold, resize, or rewrite replay direction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_signal_contract,
)
from gx1.contracts.immutable_event_authority_v1 import (
    require_newest_immutable_event,
    write_immutable_json_event,
)
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    CLASS_ORDER,
)
from gx1.execution.model_native_entry_replay_v1 import (
    LABEL_HORIZON_EXIT_MODE,
    OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
    SourceTape,
    UNIT_NORMALIZED_PNL_MODE,
    _policy_hash,
    label_horizon_exit_policy_config,
    label_horizon_exit_policy_contract,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    require_model_direction_decision_contract,
)
from gx1.scripts.audit_model_native_direction_pockets_v1 import (
    _model_direction_contract_failures,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    atomic_write_text,
    resolve_and_validate_prediction_evidence,
    sha256_file,
)
from gx1.scripts.verify_entry_foundation_state_v1 import (
    EVIDENCE_SPECS,
    STATE_EVENT_PREFIX,
    STATE_PROVEN_DECISION,
    STATE_SCHEMA_VERSION,
)


TRADE_LOG_SCHEMA_VERSION = "entry_candidate_replay_trade_log_model_native_v3"
TRADE_LOG_EVENT_PREFIX = "CANDIDATE_REPLAY_TRADE_LOG"
CANDIDATE_EVENT_PREFIX = "ENTRY_CANDIDATE_READINESS"
CANDIDATE_SCHEMA_VERSION = "entry_candidate_readiness_model_native_v1"
CANDIDATE_READY_DECISION = "READY_FOR_CANDIDATE_TRAINING"

SIDE_LONG = CLASS_ORDER.index("LONG")
SIDE_SHORT = CLASS_ORDER.index("SHORT")
SIDE_FLAT = CLASS_ORDER.index("FLAT")
SIDE_NAMES = dict(enumerate(CLASS_ORDER))

_FORBIDDEN_DIRECTION_KEYS = frozenset(
    {
        "anchor_logits",
        "delta_logits",
        "anchor_gate",
        "neutralize_signal_bridge",
        "expected_utility_side",
        "utility_side",
        "direction_override",
        "side_override",
        "direction_gate",
        "entry_gate",
        "trade_gate",
        "session_allowed",
        "trend_allowed",
        "utility_allowed",
        "path_allowed",
    }
)

# Learned auxiliary heads that are useful for post-hoc slicing.  They are
# copied after the action is fixed and never read by the replay policy.
_SCALAR_MODEL_DIAGNOSTICS = (
    "tradable_prob",
    "clean_edge_prob",
    "survival_prob",
    "tf_agreement_prob",
    "position_size_pred",
    "hold_horizon_pred",
    "mfe_first_n_pred",
    "path_quality_log_var",
    "mtf_long_minus_short",
    "trendline_rail_rising_support_prob",
    "trendline_rail_falling_resistance_prob",
    "trendline_rail_breakout_up_prob",
    "trendline_rail_breakout_down_prob",
    "long_expected_utility_bps",
    "short_expected_utility_bps",
    "long_expected_mae_bps",
    "short_expected_mae_bps",
)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"could not read immutable JSON event {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"immutable JSON event root is not an object: {path}")
    return payload


def _walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _forbidden_direction_keys(payload: dict[str, Any]) -> list[str]:
    found: set[str] = set()
    for row in _walk_dicts(payload):
        found.update(_FORBIDDEN_DIRECTION_KEYS.intersection(row))
    return sorted(found)


def _require_candidate_readiness_event(path: Path) -> dict[str, Any]:
    require_newest_immutable_event(path, CANDIDATE_EVENT_PREFIX)
    payload = _read_json_object(path)
    failures: list[str] = []
    exact = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "decision": CANDIDATE_READY_DECISION,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "expected_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "edge_test_scope": "strict",
        "promotion_shadow_live_allowed": False,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            failures.append(f"candidate readiness {key}={payload.get(key)!r} expected={expected!r}")
    if payload.get("failures"):
        failures.append("candidate readiness declares failures")
    forbidden = _forbidden_direction_keys(payload)
    if forbidden:
        failures.append(f"candidate readiness contains forbidden direction keys: {forbidden}")

    fingerprints = payload.get("artifact_fingerprints")
    fingerprints = fingerprints if isinstance(fingerprints, dict) else {}
    if not fingerprints:
        failures.append("candidate readiness lacks artifact_fingerprints")
    for name, raw_row in fingerprints.items():
        row = raw_row if isinstance(raw_row, dict) else {}
        raw_path = str(row.get("path") or "").strip()
        declared_sha = str(row.get("sha256") or "").strip().lower()
        artifact = Path(raw_path).expanduser().resolve() if raw_path else Path("/")
        if not raw_path or len(declared_sha) != 64:
            failures.append(f"candidate readiness fingerprint is incomplete: {name}")
        elif not artifact.is_file() or sha256_file(artifact) != declared_sha:
            failures.append(f"candidate readiness fingerprint rehash mismatch: {name}")
    if failures:
        raise RuntimeError("candidate-readiness contract failed: " + " | ".join(failures))
    return payload


def _require_model_native_state_event(
    state_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    require_newest_immutable_event(state_path, STATE_EVENT_PREFIX)
    state = _read_json_object(state_path)
    failures: list[str] = []
    exact = {
        "schema_version": STATE_SCHEMA_VERSION,
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
    }
    for key, expected in exact.items():
        if state.get(key) != expected:
            failures.append(f"model-native state {key}={state.get(key)!r} expected={expected!r}")
    if state.get("failures"):
        failures.append("model-native state declares failures")
    forbidden = _forbidden_direction_keys(state)
    if forbidden:
        failures.append(f"model-native state contains forbidden direction keys: {forbidden}")

    rows = state.get("evidence")
    rows = rows if isinstance(rows, list) else []
    by_name = {
        str(row.get("name")): row
        for row in rows
        if isinstance(row, dict) and str(row.get("name") or "")
    }
    expected_names = {spec.name for spec in EVIDENCE_SPECS}
    if set(by_name) != expected_names:
        failures.append(
            "model-native state evidence inventory mismatch: "
            f"observed={sorted(by_name)} expected={sorted(expected_names)}"
        )
    spec_by_name = {spec.name: spec for spec in EVIDENCE_SPECS}
    observed_hashes: dict[str, str | None] = {}
    for name in sorted(expected_names):
        row = by_name.get(name) or {}
        spec = spec_by_name[name]
        raw_path = str(row.get("path") or "").strip()
        declared_sha = str(row.get("sha256") or "").strip().lower()
        event_path = Path(raw_path).expanduser().resolve() if raw_path else Path("/")
        observed_hashes[name] = declared_sha or None
        if row.get("ready") is not True:
            failures.append(f"model-native state evidence is not ready: {name}")
        if not raw_path or len(declared_sha) != 64:
            failures.append(f"model-native state evidence binding is incomplete: {name}")
            continue
        try:
            require_newest_immutable_event(event_path, spec.event_prefix)
        except Exception as exc:
            failures.append(f"model-native state evidence authority failed for {name}: {exc}")
            continue
        if sha256_file(event_path) != declared_sha:
            failures.append(f"model-native state evidence rehash mismatch: {name}")
        if row.get("schema_version") != spec.schema_version:
            failures.append(f"model-native state evidence schema mismatch: {name}")
        if row.get("decision") != spec.ready_decision:
            failures.append(f"model-native state evidence decision mismatch: {name}")

    candidate_row = by_name.get("candidate_readiness") or {}
    if str(candidate_row.get("path") or ""):
        if Path(str(candidate_row["path"])).expanduser().resolve() != candidate_path:
            failures.append("model-native state candidate-readiness path mismatch")
    if str(candidate_row.get("sha256") or "").lower() != sha256_file(candidate_path):
        failures.append("model-native state candidate-readiness hash mismatch")
    if state.get("evidence_sha256") != _sha256_json(observed_hashes):
        failures.append("model-native state evidence_sha256 mismatch")
    if failures:
        raise RuntimeError("model-native state contract failed: " + " | ".join(failures))
    return state


def _validate_model_native_authority(
    state_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    candidate = _require_candidate_readiness_event(candidate_path)
    state = _require_model_native_state_event(state_path, candidate_path)
    return {
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "expected_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "state_json": str(state_path),
        "state_sha256": sha256_file(state_path),
        "state_decision": state["decision"],
        "state_evidence_sha256": state["evidence_sha256"],
        "candidate_readiness_json": str(candidate_path),
        "candidate_readiness_sha256": sha256_file(candidate_path),
        "candidate_readiness_decision": candidate["decision"],
    }


def _validate_prediction_report_contract(
    report: dict[str, Any],
    *,
    bundle_dir: Path,
) -> None:
    failures: list[str] = []
    if report.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE:
        failures.append("prediction report contract_mode is not exact model-native seq513")
    if report.get("bundle_seq_input_dim") != MODEL_NATIVE_SIGNAL_DIM:
        failures.append("prediction report seq_input_dim is not 513")
    if report.get("bundle_snap_input_dim") != MODEL_NATIVE_SIGNAL_DIM:
        failures.append("prediction report snap_input_dim is not 513")
    if report.get("feature_mask_ablation") != {"enabled": False}:
        failures.append("prediction report is feature-masked or lacks exact no-ablation proof")
    if report.get("bundle_specialist_fusion_enabled") is not True:
        failures.append("prediction report does not prove specialist fusion enabled")
    if report.get("promotion_shadow_live_allowed") is not False:
        failures.append("prediction report improperly authorizes promotion/shadow/live")
    forbidden = _forbidden_direction_keys(report)
    if forbidden:
        failures.append(f"prediction report contains forbidden direction keys: {forbidden}")
    try:
        require_model_native_signal_contract(
            report.get("model_native_signal_contract") or {},
            context="CANDIDATE_REPLAY_PREDICTION_REPORT",
        )
        require_model_direction_decision_contract(
            {"direction_decision_contract": report.get("direction_decision_contract")},
            context="candidate replay prediction report",
        )
    except RuntimeError as exc:
        failures.append(str(exc))

    metadata_path = bundle_dir / "bundle_metadata.json"
    metadata = _read_json_object(metadata_path)
    try:
        require_model_native_signal_contract(
            metadata.get("model_native_signal_contract") or {},
            context="CANDIDATE_REPLAY_BUNDLE",
        )
    except RuntimeError as exc:
        failures.append(str(exc))
    if metadata.get("model_native_signal_contract") != report.get("model_native_signal_contract"):
        failures.append("prediction report and bundle model-native signal contracts differ")
    if metadata.get("seq_input_dim") != MODEL_NATIVE_SIGNAL_DIM or metadata.get(
        "snap_input_dim"
    ) != MODEL_NATIVE_SIGNAL_DIM:
        failures.append("prediction bundle does not expose exact seq/snap width 513")

    specialist = report.get("bundle_specialist_contract")
    specialist = specialist if isinstance(specialist, dict) else {}
    for key in (
        "specialist_fusion_enabled",
        "required_specialists_exact",
        "specialist_model_contract_valid",
        "specialist_model_contract_set_exact",
        "specialist_model_contract_owned_objectives_match",
        "specialist_model_contract_signal_families_match",
        "specialist_model_contract_support_heads_match",
        "specialist_model_contract_model_roles_match",
        "chart_geometry_present",
        "price_action_candle_present",
    ):
        if specialist.get(key) is not True:
            failures.append(f"prediction specialist contract {key} is not true")
    if specialist.get("failures"):
        failures.append("prediction specialist contract declares failures")
    if failures:
        raise RuntimeError("prediction report model-native contract failed: " + " | ".join(failures))


def _prediction_report_split_artifacts(
    report: dict[str, Any],
    dataset_dir: Path,
) -> dict[str, Path]:
    contract = report.get("dataset_signal_contract")
    rows = contract.get("splits") if isinstance(contract, dict) else None
    if not isinstance(rows, dict) or set(rows) != {"val", "test"}:
        raise RuntimeError(
            "prediction report lacks exact val/test dataset artifact bindings"
        )
    parquets: dict[str, Path] = {}
    for split in ("val", "test"):
        row = rows[split]
        if not isinstance(row, dict):
            raise RuntimeError(
                f"prediction report dataset binding is invalid: split={split}"
            )
        for kind, suffix in (
            ("manifest", f"_{split}.manifest.json"),
            ("parquet", f"_{split}.parquet"),
        ):
            path = Path(str(row.get(f"{kind}_path") or "")).expanduser()
            expected_sha = str(row.get(f"{kind}_sha256") or "").strip().lower()
            if (
                not path.is_absolute()
                or path.is_symlink()
                or not path.is_file()
                or path.resolve() != path
                or path.parent != dataset_dir
                or not path.name.endswith(suffix)
                or any("latest" in part.lower() for part in path.parts)
            ):
                raise RuntimeError(
                    f"prediction report dataset artifact is invalid: {split}_{kind}={path}"
                )
            if len(expected_sha) != 64 or any(
                character not in "0123456789abcdef" for character in expected_sha
            ):
                raise RuntimeError(
                    f"prediction report dataset artifact lacks SHA-256: {split}_{kind}"
                )
            observed_sha = sha256_file(path)
            if observed_sha != expected_sha:
                raise RuntimeError(
                    f"prediction report dataset artifact hash mismatch: {split}_{kind}"
                )
            if kind == "parquet":
                parquets[split] = path
        manifest = json.loads(Path(row["manifest_path"]).read_text(encoding="utf-8"))
        if Path(str(manifest.get("output_data_path") or "")).expanduser() != parquets[split]:
            raise RuntimeError(
                f"prediction report dataset manifest self-path mismatch: split={split}"
            )
    return parquets


def _load_horizons(split_parquets: dict[str, Path], splits: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split in splits:
        path = split_parquets[split]
        frame = pd.read_parquet(path, columns=["time", "label_horizon_bars"])
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        if frame["time"].isna().any():
            raise RuntimeError(f"{split} dataset has unparsable time rows: {path}")
        frame["split"] = str(split)
        frame["label_horizon_bars"] = pd.to_numeric(
            frame["label_horizon_bars"], errors="coerce"
        )
        horizons = frame["label_horizon_bars"].to_numpy(np.float64)
        if (
            not np.isfinite(horizons).all()
            or bool((horizons <= 0.0).any())
            or not bool((horizons == np.floor(horizons)).all())
        ):
            raise RuntimeError(
                f"{split} dataset contains invalid label_horizon_bars: {path}"
            )
        frame["label_horizon_bars"] = horizons.astype(np.int64)
        frames.append(frame[["split", "time", "label_horizon_bars"]])
    out = pd.concat(frames, ignore_index=True)
    if out.duplicated(["split", "time"]).any():
        raise RuntimeError("dataset horizon join keys contain duplicate split/time rows")
    return out


def _prepare_predictions(
    predictions_path: Path,
    split_parquets: dict[str, Path],
    model_name: str,
) -> pd.DataFrame:
    required = {
        "split",
        "model",
        "time",
        "y_direction",
        "pred_direction",
        "session",
        "vol_regime",
        "p_long",
        "p_short",
        "p_flat",
        "selection_score_mode",
        "path_quality_pred",
        "bad_path_prob",
        "public_trade_probability",
        "public_flat_probability",
        "public_trade_flat_margin",
        "public_trade_flat_hard_decision",
        "direction_logits",
        "public_trade_flat_decision_logits",
    }
    predictions = pd.read_parquet(predictions_path)
    missing = sorted(required - set(str(column) for column in predictions.columns))
    if missing:
        raise RuntimeError(f"selective-edge predictions missing columns: {missing}")
    forbidden = sorted(_FORBIDDEN_DIRECTION_KEYS.intersection(predictions.columns))
    if forbidden:
        raise RuntimeError(f"selective-edge predictions contain forbidden direction columns: {forbidden}")
    predictions = predictions[predictions["model"].astype(str) == str(model_name)].copy()
    if predictions.empty:
        raise RuntimeError(f"selective-edge predictions contain no rows for model={model_name!r}")

    direction_failures = _model_direction_contract_failures(predictions)
    if direction_failures:
        raise RuntimeError(
            "selective-edge model-direction contract failed: "
            + " | ".join(direction_failures)
        )
    pred_direction = pd.to_numeric(predictions["pred_direction"], errors="raise").astype(np.int64)
    if "trade_side" in predictions.columns:
        redundant_side = pd.to_numeric(predictions["trade_side"], errors="raise").astype(np.int64)
        if not np.array_equal(redundant_side.to_numpy(), pred_direction.to_numpy()):
            raise RuntimeError("prediction trade_side differs from final direction-logits argmax")
    if "side" in predictions.columns:
        expected_names = pred_direction.map(SIDE_NAMES)
        if not predictions["side"].astype(str).str.upper().equals(expected_names):
            raise RuntimeError("prediction side label differs from final direction-logits argmax")
    predictions["trade_side"] = pred_direction
    predictions["model_direction_margin"] = (
        predictions[["p_long", "p_short"]].astype(float).max(axis=1)
        - predictions["p_flat"].astype(float)
    )
    predictions["time"] = pd.to_datetime(predictions["time"], utc=True, errors="coerce")
    if predictions["time"].isna().any():
        raise RuntimeError("selective-edge predictions have unparsable time rows")
    predictions["split"] = predictions["split"].astype(str)
    observed_splits = sorted(predictions["split"].unique())
    if observed_splits != ["test", "val"]:
        raise RuntimeError(
            f"candidate replay requires exact val/test prediction splits; observed={observed_splits}"
        )
    for column in ("path_quality_pred", "bad_path_prob"):
        values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"prediction diagnostic is not fully finite: {column}")
    bad_path = pd.to_numeric(predictions["bad_path_prob"], errors="raise")
    if bool(((bad_path < 0.0) | (bad_path > 1.0)).any()):
        raise RuntimeError("prediction bad_path_prob is outside [0,1]")
    for column in ("session", "vol_regime"):
        values = predictions[column].fillna("").astype(str).str.strip().str.upper()
        if bool(values.isin(["", "UNKNOWN", "NAN", "NONE"]).any()):
            raise RuntimeError(f"prediction {column} contains missing/UNKNOWN state")

    horizons = _load_horizons(split_parquets, ["val", "test"])
    coverage = predictions[["split", "time"]].merge(
        horizons[["split", "time"]],
        on=["split", "time"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    coverage_counts = coverage["_merge"].value_counts().to_dict()
    if not bool((coverage["_merge"] == "both").all()):
        raise RuntimeError(
            "selective-edge predictions do not exactly cover all val/test dataset rows: "
            f"coverage={coverage_counts}"
        )
    merged = predictions.merge(
        horizons,
        on=["split", "time"],
        how="left",
        validate="one_to_one",
    )
    if merged["label_horizon_bars"].isna().any():
        raise RuntimeError("failed to attach label_horizon_bars to all prediction rows")
    return merged.sort_values(["split", "time"], kind="mergesort").reset_index(drop=True)


def _resolve_score_surface(predictions: pd.DataFrame) -> tuple[str, str]:
    """Return a diagnostic margin without granting it selection authority."""

    modes = sorted(
        {
            str(value).strip()
            for value in predictions["selection_score_mode"].dropna().astype(str).unique()
            if str(value).strip()
        }
    )
    if modes != [MODEL_DIRECTION_SELECTION_MODE]:
        raise RuntimeError(
            "replay requires exact model_direction_argmax mode; "
            f"observed={modes or ['<missing>']}"
        )
    if "selection_score_threshold" in predictions.columns:
        raise RuntimeError("replay predictions contain forbidden selection_score_threshold")
    if "model_direction_margin" not in predictions.columns:
        probabilities = predictions[["p_long", "p_short", "p_flat"]].astype(float)
        predictions["model_direction_margin"] = (
            probabilities[["p_long", "p_short"]].max(axis=1)
            - probabilities["p_flat"]
        )
    return "model_direction_margin", MODEL_DIRECTION_SELECTION_MODE


def _run_exact_label_horizon_replay(
    *,
    eval_df: pd.DataFrame,
    tape: SourceTape,
    score_column: str,
    cost_stress_bps: float,
    slippage_bps: float,
    policy_id: str,
    policy_config_hash: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    probabilities = eval_df[["p_long", "p_short", "p_flat"]].astype(float).to_numpy(np.float64)
    scores = pd.to_numeric(eval_df[score_column], errors="coerce").to_numpy(np.float64)
    sides = pd.to_numeric(eval_df["trade_side"], errors="coerce").to_numpy(np.int64)
    labels = pd.to_numeric(eval_df["y_direction"], errors="coerce").to_numpy(np.int64)
    if not np.isfinite(scores).all():
        raise RuntimeError("model direction diagnostic margin contains non-finite values")
    if not bool(np.isin(sides, [SIDE_LONG, SIDE_SHORT, SIDE_FLAT]).all()):
        raise RuntimeError("replay contains a direction outside LONG/SHORT/FLAT")
    winner_counts = np.count_nonzero(
        probabilities == np.max(probabilities, axis=1, keepdims=True),
        axis=1,
    )
    tied_rows = int(np.count_nonzero(winner_counts != 1))
    if tied_rows:
        raise RuntimeError(
            "replay direction probabilities have no unique top class; "
            f"rows={tied_rows}"
        )
    probability_argmax = np.argmax(probabilities, axis=1).astype(np.int64)
    if not np.array_equal(sides, probability_argmax):
        raise RuntimeError("replay trade_side differs from model probability argmax")

    exit_policy = label_horizon_exit_policy_config()
    exit_policy_hash = _policy_hash(exit_policy)
    model_flat_rows = int(np.count_nonzero(sides == SIDE_FLAT))
    non_flat_argmax_rows = int(len(sides) - model_flat_rows)
    counts: dict[str, Any] = {
        "policy_id": policy_id,
        "direction_authority": "argmax(final_calibrated_direction_logits)",
        "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
        "offline_only": True,
        "diagnostic_scope": OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
        "pnl_normalization": UNIT_NORMALIZED_PNL_MODE,
        "execution_order_simulation": False,
        "position_size_applied": False,
        "row_simulation_mode": "independent",
        "one_trade_per_non_flat_argmax_row": True,
        "filters_applied": False,
        "occupancy_filter_applied": False,
        "cooldown_applied": False,
        "max_trades_per_day_applied": False,
        "daily_loss_limit_applied": False,
        "invalid_path_skip_allowed": False,
        "cost_stress_bps": float(cost_stress_bps),
        "slippage_bps": float(slippage_bps),
        "exit_mode": LABEL_HORIZON_EXIT_MODE,
        "exit_policy_config_hash": exit_policy_hash,
        "evaluated_rows": int(len(eval_df)),
        "model_flat_rows": model_flat_rows,
        "non_flat_argmax_rows": non_flat_argmax_rows,
        "expected_trades": non_flat_argmax_rows,
        "trades": 0,
        "trades_equal_non_flat_argmax_rows": False,
    }
    trades: list[dict[str, Any]] = []
    for index, row in enumerate(eval_df.itertuples(index=False)):
        row_score = float(scores[index])
        side = int(sides[index])
        if side == SIDE_FLAT:
            continue
        decision_time = pd.Timestamp(row.time)
        raw_horizon = float(row.label_horizon_bars)
        if not np.isfinite(raw_horizon) or raw_horizon <= 0.0 or not raw_horizon.is_integer():
            raise RuntimeError(
                "non-FLAT row has invalid label_horizon_bars: "
                f"row={index} time={decision_time.isoformat()} value={row.label_horizon_bars!r}"
            )
        horizon_bars = int(raw_horizon)
        try:
            sim = tape.simulate_label_horizon_trade(
                decision_time=decision_time,
                horizon_m5_bars=horizon_bars,
                side=side,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "non-FLAT replay path failed closed: "
                f"row={index} time={decision_time.isoformat()} side={SIDE_NAMES[side]}: {exc}"
            ) from exc
        entry_time = pd.Timestamp(sim["entry_time"])
        day = entry_time.date().isoformat()
        # This is a unit-normalized price-path diagnostic. A learned
        # position_size_pred may be logged below, but is never applied to PnL.
        net_pnl_bps = (
            float(sim["gross_pnl_bps"])
            - float(cost_stress_bps)
            - float(slippage_bps)
        )
        trade: dict[str, Any] = {
            "replay_row_index": int(index),
            "fold": "2026_TEST",
            "policy_id": policy_id,
            "policy_config_hash": policy_config_hash,
            "direction_authority": "argmax(final_calibrated_direction_logits)",
            "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
            "offline_only": True,
            "diagnostic_scope": OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
            "pnl_normalization": UNIT_NORMALIZED_PNL_MODE,
            "execution_order_simulation": False,
            "position_size_applied": False,
            "row_simulation_mode": "independent",
            "filters_applied": False,
            "exit_mode": LABEL_HORIZON_EXIT_MODE,
            "exit_policy_config_hash": exit_policy_hash,
            "cost_stress_bps": float(cost_stress_bps),
            "slippage_bps": float(slippage_bps),
            "source_split": str(row.split),
            "session": str(row.session).upper(),
            "vol_regime": str(row.vol_regime),
            "decision_time": decision_time,
            "entry_day": day,
            "entry_month": entry_time.strftime("%Y-%m"),
            "entry_time": sim["entry_time"],
            "fill_price_source": "m1_open_at_decision_plus_5m",
            "exit_time": sim["exit_time"],
            "side": SIDE_NAMES[side],
            "label": SIDE_NAMES.get(int(labels[index]), str(int(labels[index]))),
            "direction_correct": bool(int(labels[index]) == side),
            "score": row_score,
            "p_long": float(probabilities[index, 0]),
            "p_short": float(probabilities[index, 1]),
            "p_flat": float(probabilities[index, 2]),
            "chosen_prob": float(probabilities[index, side]),
            "path_quality_pred": float(row.path_quality_pred),
            "bad_path_prob": float(row.bad_path_prob),
            "entry_price": float(sim["entry_price"]),
            "exit_price": float(sim["exit_price"]),
            "gross_pnl_bps": float(sim["gross_pnl_bps"]),
            "net_pnl_bps": float(net_pnl_bps),
            "mfe_bps": float(sim["mfe_bps"]),
            "mae_bps": float(sim["mae_bps"]),
            "horizon_bars": horizon_bars,
            "horizon_timeframe": "M5",
            "held_bars": int(sim["held_bars"]),
            "exit_reason": str(sim["exit_reason"]),
        }
        for column in _SCALAR_MODEL_DIAGNOSTICS:
            if column not in eval_df.columns:
                continue
            value = getattr(row, column)
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric):
                trade[column] = numeric
        trades.append(trade)
        counts["trades"] += 1
    counts["trades_equal_non_flat_argmax_rows"] = bool(
        counts["trades"] == counts["non_flat_argmax_rows"]
    )
    if not counts["trades_equal_non_flat_argmax_rows"]:
        raise RuntimeError(
            "exact replay cardinality invariant failed: "
            f"trades={counts['trades']} non_flat={counts['non_flat_argmax_rows']}"
        )
    return trades, counts


def _trade_failures(
    trades: pd.DataFrame,
    *,
    counts: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if counts.get("filters_applied") is not False:
        failures.append("candidate replay counts do not prove filters_applied=false")
    exact_diagnostic_counts = {
        "offline_only": True,
        "diagnostic_scope": OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
        "pnl_normalization": UNIT_NORMALIZED_PNL_MODE,
        "execution_order_simulation": False,
        "position_size_applied": False,
    }
    for key, expected in exact_diagnostic_counts.items():
        if counts.get(key) != expected:
            failures.append(
                f"candidate replay counts {key}={counts.get(key)!r} expected={expected!r}"
            )
    if counts.get("trades") != counts.get("non_flat_argmax_rows") or counts.get(
        "trades_equal_non_flat_argmax_rows"
    ) is not True:
        failures.append("candidate replay counts do not prove trades == non-FLAT rows")
    if trades.empty:
        failures.append("candidate replay trade log produced zero trades")
        return failures
    required = {
        "entry_time",
        "policy_id",
        "session",
        "vol_regime",
        "side",
        "score",
        "p_long",
        "p_short",
        "p_flat",
        "path_quality_pred",
        "bad_path_prob",
        "gross_pnl_bps",
        "net_pnl_bps",
        "mfe_bps",
        "mae_bps",
        "held_bars",
        "horizon_bars",
        "exit_mode",
        "exit_reason",
        "row_simulation_mode",
        "filters_applied",
        "offline_only",
        "diagnostic_scope",
        "pnl_normalization",
        "execution_order_simulation",
        "position_size_applied",
        "direction_correct",
    }
    missing = sorted(required - set(trades.columns))
    if missing:
        return [f"candidate replay trade log missing required columns: {missing}"]
    years = set(pd.to_datetime(trades["entry_time"], utc=True).dt.year.astype(int).unique())
    if years != {2026}:
        failures.append(f"candidate replay trade log contains years outside 2026: {sorted(years)}")
    numeric_columns = (
        "score",
        "p_long",
        "p_short",
        "p_flat",
        "path_quality_pred",
        "bad_path_prob",
        "gross_pnl_bps",
        "net_pnl_bps",
        "mfe_bps",
        "mae_bps",
        "held_bars",
        "horizon_bars",
    )
    for column in numeric_columns:
        values = pd.to_numeric(trades[column], errors="coerce").to_numpy(np.float64)
        if not np.isfinite(values).all():
            failures.append(f"candidate replay numeric column is not fully finite: {column}")
    probabilities = trades[["p_long", "p_short", "p_flat"]].astype(float).to_numpy(np.float64)
    if not bool(((probabilities >= 0.0) & (probabilities <= 1.0)).all()):
        failures.append("candidate replay direction probabilities are outside [0,1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        failures.append("candidate replay direction probabilities do not sum to 1")
    winner_counts = np.count_nonzero(
        probabilities == np.max(probabilities, axis=1, keepdims=True),
        axis=1,
    )
    tied_rows = int(np.count_nonzero(winner_counts != 1))
    if tied_rows:
        failures.append(
            "candidate replay direction probabilities have no unique top class: "
            f"rows={tied_rows}"
        )
    expected_side = np.asarray(CLASS_ORDER)[np.argmax(probabilities, axis=1)]
    observed_side = trades["side"].astype(str).str.upper().to_numpy()
    if not np.array_equal(expected_side, observed_side):
        failures.append("candidate replay side differs from model probability argmax")
    if bool((observed_side == "FLAT").any()):
        failures.append("candidate replay contains a model-FLAT trade")
    if not bool((trades["exit_mode"] == LABEL_HORIZON_EXIT_MODE).all()) or not bool(
        (trades["exit_reason"] == LABEL_HORIZON_EXIT_MODE).all()
    ):
        failures.append("candidate replay contains a non-label-horizon exit")
    held = pd.to_numeric(trades["held_bars"], errors="coerce")
    horizon = pd.to_numeric(trades["horizon_bars"], errors="coerce")
    if not bool((held == horizon).all()):
        failures.append("candidate replay held_bars differs from label horizon")
    if not bool((trades["row_simulation_mode"] == "independent").all()):
        failures.append("candidate replay rows are not marked independent")
    if not bool((trades["filters_applied"] == False).all()):  # noqa: E712
        failures.append("candidate replay trade rows do not prove filters_applied=false")
    exact_diagnostic_columns = {
        "offline_only": True,
        "diagnostic_scope": OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
        "pnl_normalization": UNIT_NORMALIZED_PNL_MODE,
        "execution_order_simulation": False,
        "position_size_applied": False,
    }
    for column, expected in exact_diagnostic_columns.items():
        if not bool((trades[column] == expected).all()):
            failures.append(
                f"candidate replay trade rows {column} are not exactly {expected!r}"
            )
    retired_sizing_columns = {
        "dynamic_sizing_applied",
        "applied_size_multiplier",
        "replay_size_multiplier",
        "sizing_authority_contract",
    }
    unexpected_sizing = sorted(retired_sizing_columns.intersection(trades.columns))
    if unexpected_sizing:
        failures.append(
            f"candidate replay contains execution-sizing fields: {unexpected_sizing}"
        )
    bad_path = pd.to_numeric(trades["bad_path_prob"], errors="coerce")
    if bool(((bad_path < 0.0) | (bad_path > 1.0)).any()):
        failures.append("candidate replay bad_path_prob is outside [0,1]")
    for column in ("session", "vol_regime"):
        values = trades[column].fillna("").astype(str).str.strip().str.upper()
        if bool(values.isin(["", "UNKNOWN", "NAN", "NONE"]).any()):
            failures.append(f"candidate replay {column} contains missing/UNKNOWN state")
    if "threshold_top_frac" in trades.columns or "score_threshold" in trades.columns:
        failures.append("candidate replay contains retired direction-threshold columns")
    return failures


def _direction_policy_contract() -> dict[str, Any]:
    return {
        "schema_version": "entry_candidate_replay_direction_policy_v3",
        "direction_authority": "argmax(final_calibrated_direction_logits)",
        "selection_score_mode": MODEL_DIRECTION_SELECTION_MODE,
        "direction_classes": list(CLASS_ORDER),
        "offline_only": True,
        "diagnostic_scope": OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
        "pnl_normalization": UNIT_NORMALIZED_PNL_MODE,
        "execution_order_simulation": False,
        "position_size_applied": False,
        "model_flat_is_only_direction_no_trade": True,
        "one_trade_per_non_flat_argmax_row": True,
        "row_simulation_mode": "independent",
        "exit_mode": LABEL_HORIZON_EXIT_MODE,
        "filters_applied": False,
        "occupancy_filter_allowed": False,
        "cooldown_allowed": False,
        "max_trades_per_day_allowed": False,
        "daily_loss_limit_allowed": False,
        "invalid_path_skip_allowed": False,
        "direction_thresholds_allowed": False,
        "auxiliary_heads_direction_authority": "none",
        "trend_session_path_utility_direction_filters_allowed": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    state_path = Path(args.model_native_state_json).expanduser().resolve()
    candidate_path = Path(args.candidate_readiness_json).expanduser().resolve()
    requested_predictions_path = Path(args.selective_edge_predictions).expanduser().resolve()
    requested_report_path = Path(args.selective_edge_report_json).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    source_parquet = Path(args.source_parquet).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"candidate replay out-dir must be new/empty for immutable publication: {out_dir}"
        )
    model_native_authority = _validate_model_native_authority(state_path, candidate_path)
    predictions_path, prediction_report, prediction_evidence = (
        resolve_and_validate_prediction_evidence(
            requested_predictions_path,
            prediction_report_path=requested_report_path,
            bundle_dir=None,
            dataset_dir=dataset_dir,
            expected_model=str(args.model_name),
        )
    )
    bundle_dir = Path(str(prediction_report.get("bundle_dir") or "")).expanduser().resolve()
    _validate_prediction_report_contract(prediction_report, bundle_dir=bundle_dir)
    split_parquets = _prediction_report_split_artifacts(
        prediction_report,
        dataset_dir,
    )
    predictions = _prepare_predictions(
        predictions_path,
        split_parquets,
        str(args.model_name),
    )
    score_column, selection_score_mode = _resolve_score_surface(predictions)
    test = predictions[predictions["split"] == "test"].sort_values(
        "time", kind="mergesort"
    ).reset_index(drop=True)
    if not bool((test["time"].dt.year == 2026).all()):
        raise RuntimeError("candidate replay trade-log test split must be entirely 2026")

    cost_stress_bps = float(args.cost_stress_bps)
    slippage_bps = float(args.slippage_bps)
    if not np.isfinite(cost_stress_bps) or cost_stress_bps < 0.0:
        raise RuntimeError("cost_stress_bps must be finite and non-negative")
    if not np.isfinite(slippage_bps) or slippage_bps < 0.0:
        raise RuntimeError("slippage_bps must be finite and non-negative")
    tape = SourceTape.load(source_parquet)

    exit_policy = label_horizon_exit_policy_config()
    policy_config = {
        "schema_version": "entry_candidate_replay_policy_v3",
        "model_name": str(args.model_name),
        "direction_authority": "argmax(final_calibrated_direction_logits)",
        "selection_score_mode": selection_score_mode,
        "score_diagnostic": score_column,
        "direction_classes": list(CLASS_ORDER),
        "offline_only": True,
        "diagnostic_scope": OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
        "pnl_normalization": UNIT_NORMALIZED_PNL_MODE,
        "execution_order_simulation": False,
        "position_size_applied": False,
        "model_flat_is_only_direction_no_trade": True,
        "one_trade_per_non_flat_argmax_row": True,
        "row_simulation_mode": "independent",
        "filters_applied": False,
        "occupancy_filter_allowed": False,
        "cooldown_allowed": False,
        "max_trades_per_day_allowed": False,
        "daily_loss_limit_allowed": False,
        "invalid_path_skip_allowed": False,
        "eval_split": "test",
        "exit_mode": LABEL_HORIZON_EXIT_MODE,
        "exit_policy_config_hash": _policy_hash(exit_policy),
        "exit_policy": exit_policy,
        "cost_stress_bps": cost_stress_bps,
        "slippage_bps": slippage_bps,
    }
    policy_config_hash = _policy_hash(policy_config)
    policy_id = str(args.policy_id).strip() or f"candidate_model_native_{policy_config_hash}"
    trades, counts = _run_exact_label_horizon_replay(
        eval_df=test,
        tape=tape,
        score_column=score_column,
        cost_stress_bps=cost_stress_bps,
        slippage_bps=slippage_bps,
        policy_id=policy_id,
        policy_config_hash=policy_config_hash,
    )
    trades_df = pd.DataFrame(trades)
    failures = _trade_failures(trades_df, counts=counts)

    out_dir.mkdir(parents=True, exist_ok=True)
    event_created_utc = datetime.now(timezone.utc)
    timestamp = event_created_utc.strftime("%Y%m%dT%H%M%S%fZ")
    trades_path = out_dir / f"candidate_replay_trade_log_{timestamp}.csv"
    counts_path = out_dir / f"candidate_replay_policy_counts_{timestamp}.csv"
    atomic_write_text(trades_path, trades_df.to_csv(index=False))
    atomic_write_text(counts_path, pd.DataFrame([counts]).to_csv(index=False))

    report = {
        "schema_version": TRADE_LOG_SCHEMA_VERSION,
        "created_utc": event_created_utc.isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "expected_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "model_native_authority": model_native_authority,
        "selective_edge_predictions": str(predictions_path),
        "requested_selective_edge_predictions": str(requested_predictions_path),
        "prediction_report_json": str(requested_report_path),
        "prediction_report_sha256": sha256_file(requested_report_path),
        "prediction_evidence": prediction_evidence,
        "dataset_dir": str(dataset_dir),
        "source_parquet": str(source_parquet),
        "source_parquet_sha256": sha256_file(source_parquet),
        "out_dir": str(out_dir),
        "direction_policy_contract": _direction_policy_contract(),
        "offline_only": True,
        "diagnostic_scope": OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
        "pnl_normalization": UNIT_NORMALIZED_PNL_MODE,
        "execution_order_simulation": False,
        "position_size_applied": False,
        "score_column": score_column,
        "selection_score_mode": selection_score_mode,
        "policy_id": policy_id,
        "policy_config_hash": policy_config_hash,
        "policy_config": policy_config,
        "cost_stress_bps": cost_stress_bps,
        "trades_path": str(trades_path),
        "trades_sha256": sha256_file(trades_path),
        "counts_path": str(counts_path),
        "counts_sha256": sha256_file(counts_path),
        "n_trades": int(len(trades_df)),
        "n_test_rows": int(counts["evaluated_rows"]),
        "n_model_flat_rows": int(counts["model_flat_rows"]),
        "n_non_flat_argmax_rows": int(counts["non_flat_argmax_rows"]),
        "trades_equal_non_flat_argmax_rows": bool(
            counts["trades_equal_non_flat_argmax_rows"]
        ),
        "filters_applied": False,
        "policy_counts": counts,
        "exit_policy_contract": label_horizon_exit_policy_contract(),
        "offline_trade_simulation_completed": True,
        "trainer_started": False,
        "live_replay_started": False,
        "promotion_shadow_live_allowed": False,
        "failures": failures,
    }
    _, published = write_immutable_json_event(
        out_dir,
        TRADE_LOG_EVENT_PREFIX,
        report,
    )
    if not args.quiet:
        print(json.dumps(published, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(2)
    return published


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-native-state-json",
        required=True,
        help="explicit newest immutable ENTRY_MODEL_NATIVE_SEQ513_STATE event",
    )
    parser.add_argument(
        "--candidate-readiness-json",
        required=True,
        help="explicit newest immutable ENTRY_CANDIDATE_READINESS event bound by state",
    )
    parser.add_argument(
        "--selective-edge-predictions",
        required=True,
        help="explicit timestamped authoritative selective_edge_predictions event",
    )
    parser.add_argument(
        "--selective-edge-report-json",
        required=True,
        help="matching newest immutable ENTRY_CANDIDATE_SELECTIVE_EDGE event",
    )
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-parquet", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--model-name", default="candidate")
    parser.add_argument("--cost-stress-bps", type=float, required=True)
    parser.add_argument("--policy-id", default="candidate_replay")
    parser.add_argument("--slippage-bps", type=float, required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
