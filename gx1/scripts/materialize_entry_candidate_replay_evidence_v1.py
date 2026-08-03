#!/usr/bin/env python3
"""Materialize Entry candidate offline replay evidence.

This script consumes an explicit trade-level replay log and writes timestamped,
immutable metrics and monthly evidence required by Entry replay-readiness. It
does not run replay, train, promote, shadow, live, or select implicit
latest/legacy artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_signal_contract,
)
from gx1.contracts.immutable_event_authority_v1 import require_newest_immutable_event
from gx1.contracts.entry_model_native_direction_evidence_fusion_v1 import (
    CLASS_ORDER,
)
from gx1.features.entry_specialist_feature_groups_v1 import required_training_specialists_for_mode
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    require_model_direction_decision_contract,
)
from gx1.execution.model_native_entry_replay_v1 import (
    LABEL_HORIZON_EXIT_MODE,
    OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
    UNIT_NORMALIZED_PNL_MODE,
    label_horizon_exit_policy_contract,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    atomic_write_text,
    resolve_and_validate_prediction_evidence,
)
from gx1.scripts.materialize_entry_candidate_replay_trade_log_v1 import (
    CANDIDATE_EVENT_PREFIX,
    TRADE_LOG_EVENT_PREFIX,
    TRADE_LOG_SCHEMA_VERSION,
    _direction_policy_contract,
)
from gx1.scripts.verify_entry_foundation_state_v1 import STATE_EVENT_PREFIX
CONTRACT_INPUT_DIMS = {MODEL_NATIVE_CONTRACT_MODE: MODEL_NATIVE_SIGNAL_DIM}
REPLAY_REQUIRED_SPLIT = "test"
REPLAY_REQUIRED_YEAR = 2026
_TIMESTAMPED_JSON_RE = re.compile(r".+_\d{8}T\d{6}(?:\d{6})?Z\.json")


def _is_explicit_timestamped_json(path: Path) -> bool:
    return bool(_TIMESTAMPED_JSON_RE.fullmatch(path.name))

MODEL_NATIVE_REPLAY_REQUIRED_COLUMNS = (
    "entry_time",
    "source_split",
    "policy_id",
    "session",
    "side",
    "score",
    "p_long",
    "p_short",
    "p_flat",
    "net_pnl_bps",
    "mfe_bps",
    "mae_bps",
    "held_bars",
    "vol_regime",
    "path_quality_pred",
    "bad_path_prob",
    "horizon_bars",
    "exit_mode",
    "row_simulation_mode",
    "filters_applied",
    "offline_only",
    "diagnostic_scope",
    "pnl_normalization",
    "execution_order_simulation",
    "position_size_applied",
)

MODEL_NATIVE_REPLAY_NUMERIC_COLUMNS = (
    "score",
    "p_long",
    "p_short",
    "p_flat",
    "net_pnl_bps",
    "mfe_bps",
    "mae_bps",
    "held_bars",
    "path_quality_pred",
    "bad_path_prob",
    "horizon_bars",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        out = float(obj)
        return out if np.isfinite(out) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing replay trades file: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise RuntimeError(f"unsupported replay trades extension: {path.suffix}")


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_specialists(contract_mode: str) -> list[str]:
    if contract_mode != MODEL_NATIVE_CONTRACT_MODE:
        return []
    return sorted(required_training_specialists_for_mode(MODEL_NATIVE_CONTRACT_MODE))


def _candidate_bundle_specialist_contract(candidate_audit: dict[str, Any], contract_mode: str) -> dict[str, Any]:
    bundle = candidate_audit.get("bundle_summary") if isinstance(candidate_audit.get("bundle_summary"), dict) else {}
    specialist_contract = (
        candidate_audit.get("bundle_specialist_model_contract")
        if isinstance(candidate_audit.get("bundle_specialist_model_contract"), dict)
        else {}
    )
    expected = _expected_specialists(contract_mode)
    required = sorted(str(x) for x in candidate_audit.get("required_training_specialists", []) if str(x))
    groups = sorted(str(x) for x in bundle.get("specialist_groups", []) if str(x))
    failures: list[str] = []

    if not expected:
        failures.append(f"unknown specialist contract mode: {contract_mode}")
    if required != expected:
        failures.append(f"candidate audit required specialist set mismatch: observed={required} expected={expected}")
    if groups != expected:
        failures.append(f"candidate bundle specialist group mismatch: observed={groups} expected={expected}")
    if not bool(bundle.get("specialist_fusion_enabled")):
        failures.append("candidate bundle summary specialist_fusion_enabled is not true")
    for flag in (
        "specialist_model_contract_valid",
        "specialist_model_contract_set_exact",
        "specialist_model_contract_owned_objectives_match",
        "specialist_model_contract_support_heads_match",
        "specialist_model_contract_signal_families_match",
        "specialist_model_contract_model_roles_match",
    ):
        if not bool(bundle.get(flag)):
            failures.append(f"candidate bundle summary {flag} is not true")
    for flag in (
        "valid",
        "set_exact",
        "owned_objectives_match",
        "support_heads_match",
        "signal_families_match",
        "model_roles_match",
    ):
        if not bool(specialist_contract.get(flag)):
            failures.append(f"candidate bundle specialist contract {flag} is not true")
    if specialist_contract.get("failures"):
        failures.append(f"candidate bundle specialist contract failures: {specialist_contract.get('failures')}")
    for required_name in ("chart_geometry_encoder", "price_action_candle_encoder"):
        if required_name not in groups:
            failures.append(f"candidate {contract_mode} bundle missing specialist group: {required_name}")
        if required_name not in required:
            failures.append(f"candidate {contract_mode} audit missing required specialist: {required_name}")

    return {
        "ready": not failures,
        "contract_mode": contract_mode,
        "expected_specialists": expected,
        "required_training_specialists": required,
        "bundle_specialist_groups": groups,
        "bundle_specialist_model_contract": specialist_contract,
        "failures": failures,
    }


def _selective_specialist_snapshot_checks(snapshot: dict[str, Any], contract_mode: str, *, label: str) -> list[str]:
    expected = _expected_specialists(contract_mode)
    expected_dim = CONTRACT_INPUT_DIMS.get(contract_mode)
    failures: list[str] = []
    if not snapshot:
        return [f"{label} missing specialist contract snapshot"]
    if snapshot.get("failures"):
        failures.append(f"{label} specialist contract snapshot failures: {snapshot.get('failures')}")
    if str(snapshot.get("requested_contract_mode") or "") != contract_mode:
        failures.append(
            f"{label} specialist contract mode mismatch: "
            f"{snapshot.get('requested_contract_mode')} != {contract_mode}"
        )
    if expected_dim is not None and int(snapshot.get("expected_signal_dim") or 0) != int(expected_dim):
        failures.append(
            f"{label} expected signal dim mismatch: {snapshot.get('expected_signal_dim')} != {expected_dim}"
        )
    if expected_dim is not None and int(snapshot.get("bundle_seq_input_dim") or 0) != int(expected_dim):
        failures.append(
            f"{label} seq_input_dim mismatch: {snapshot.get('bundle_seq_input_dim')} != {expected_dim}"
        )
    if expected_dim is not None and int(snapshot.get("bundle_snap_input_dim") or 0) != int(expected_dim):
        failures.append(
            f"{label} snap_input_dim mismatch: {snapshot.get('bundle_snap_input_dim')} != {expected_dim}"
        )
    if sorted(str(x) for x in snapshot.get("expected_specialists", []) if str(x)) != expected:
        failures.append(f"{label} expected specialist set mismatch")
    if sorted(str(x) for x in snapshot.get("observed_specialists", []) if str(x)) != expected:
        failures.append(f"{label} observed specialist set mismatch")
    if not bool(snapshot.get("specialist_fusion_enabled")):
        failures.append(f"{label} specialist fusion is not enabled")
    if not bool(snapshot.get("required_specialists_exact")):
        failures.append(f"{label} required specialist set is not exact")
    for flag in (
        "specialist_model_contract_valid",
        "specialist_model_contract_set_exact",
        "specialist_model_contract_owned_objectives_match",
        "specialist_model_contract_signal_families_match",
        "specialist_model_contract_support_heads_match",
        "specialist_model_contract_model_roles_match",
    ):
        if not bool(snapshot.get(flag)):
            failures.append(f"{label} {flag} is not true")
    if not bool(snapshot.get("chart_geometry_present")):
        failures.append(f"{label} missing chart_geometry_encoder")
    if not bool(snapshot.get("price_action_candle_present")):
        failures.append(f"{label} missing price_action_candle_encoder")
    return failures


def _selective_edge_specialist_contract(selective_summary: dict[str, Any], contract_mode: str) -> dict[str, Any]:
    candidate_snapshot = (
        selective_summary.get("bundle_specialist_contract")
        if isinstance(selective_summary.get("bundle_specialist_contract"), dict)
        else {}
    )
    failures = _selective_specialist_snapshot_checks(candidate_snapshot, contract_mode, label="selective-edge candidate")
    return {
        "ready": not failures,
        "contract_mode": contract_mode,
        "candidate_bundle_specialist_contract": candidate_snapshot,
        "failures": failures,
    }


def _identity_contract(
    *,
    candidate_bundle_audit_path: Path,
    selective_edge_report_path: Path,
    require_identity_artifacts: bool,
    requested_contract_mode: str | None = None,
) -> dict[str, Any]:
    contract_mode = str(
        requested_contract_mode or MODEL_NATIVE_CONTRACT_MODE
    ).strip()
    if contract_mode != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError(
            f"retired replay evidence contract mode is forbidden: {contract_mode!r}"
        )
    for label, path in (
        ("candidate bundle audit", candidate_bundle_audit_path),
        ("selective-edge report", selective_edge_report_path),
    ):
        if not _is_explicit_timestamped_json(path):
            raise RuntimeError(
                f"{label} must be an explicitly timestamped immutable artifact: {path}"
            )

    candidate_audit = _read_json_if_exists(candidate_bundle_audit_path)
    selective_report = _read_json_if_exists(selective_edge_report_path)
    failures: list[str] = []
    if require_identity_artifacts and not candidate_bundle_audit_path.is_file():
        failures.append(f"missing candidate bundle audit: {candidate_bundle_audit_path}")
    if require_identity_artifacts and not selective_edge_report_path.is_file():
        failures.append(f"missing selective-edge report: {selective_edge_report_path}")
    if candidate_audit and (
        str(candidate_audit.get("decision")) != "PASS"
        or candidate_audit.get("failures")
    ):
        failures.append("candidate bundle audit is not a zero-failure PASS")
    if selective_report and (
        str(selective_report.get("decision")) != "PASS"
        or selective_report.get("failures")
    ):
        failures.append("selective-edge report is not a zero-failure PASS")
    feature_mask_raw = selective_report.get("feature_mask_ablation")
    feature_mask = feature_mask_raw if isinstance(feature_mask_raw, dict) else {}
    if not isinstance(feature_mask_raw, dict) or feature_mask.get("enabled") is not False:
        failures.append(
            "selective-edge report does not prove a complete unmasked feature stack"
        )

    candidate_bundle_dir = str(candidate_audit.get("bundle_dir") or "").strip()
    selective_bundle_dir = str(selective_report.get("bundle_dir") or "").strip()
    if not candidate_bundle_dir or not selective_bundle_dir:
        failures.append("candidate/selective evidence must both declare bundle_dir")
    elif (
        Path(candidate_bundle_dir).expanduser().resolve()
        != Path(selective_bundle_dir).expanduser().resolve()
    ):
        failures.append("selective-edge bundle_dir does not match candidate bundle audit")
    bundle_dir = (
        Path(candidate_bundle_dir).expanduser().resolve()
        if candidate_bundle_dir
        else Path("/")
    )
    metadata_path = bundle_dir / "bundle_metadata.json"
    metadata: dict[str, Any] = {}
    signal_contract: dict[str, Any] = {}
    try:
        metadata = _read_json_if_exists(metadata_path)
        if not metadata:
            raise RuntimeError(f"missing bundle metadata: {metadata_path}")
        raw_signal_contract = metadata.get("model_native_signal_contract")
        if not isinstance(raw_signal_contract, dict):
            raise RuntimeError("bundle metadata lacks model_native_signal_contract")
        require_model_native_signal_contract(
            raw_signal_contract,
            context="REPLAY_EVIDENCE_BUNDLE",
        )
        require_model_direction_decision_contract(
            metadata,
            context="replay-evidence bundle",
        )
        signal_contract = dict(raw_signal_contract)
        if int(metadata.get("seq_input_dim") or -1) != MODEL_NATIVE_SIGNAL_DIM:
            raise RuntimeError("bundle seq_input_dim is not 513")
        if int(metadata.get("snap_input_dim") or -1) != MODEL_NATIVE_SIGNAL_DIM:
            raise RuntimeError("bundle snap_input_dim is not 513")
    except Exception as exc:
        failures.append(str(exc))

    selective_mode = str(selective_report.get("contract_mode") or "").strip()
    if selective_mode != MODEL_NATIVE_CONTRACT_MODE:
        failures.append(
            f"selective-edge contract mode mismatch: {selective_mode!r}"
        )
    selective_signal_contract = selective_report.get("model_native_signal_contract")
    try:
        if not isinstance(selective_signal_contract, dict):
            raise RuntimeError("selective-edge report lacks model_native_signal_contract")
        require_model_native_signal_contract(
            selective_signal_contract,
            context="REPLAY_EVIDENCE_SELECTIVE_REPORT",
        )
        if signal_contract and selective_signal_contract != signal_contract:
            raise RuntimeError(
                "selective-edge and bundle model-native signal contracts differ"
            )
        require_model_direction_decision_contract(
            {
                "direction_decision_contract": selective_report.get(
                    "direction_decision_contract"
                )
            },
            context="replay-evidence selective report",
        )
        if (
            str(selective_report.get("selection_score_mode") or "")
            != MODEL_DIRECTION_SELECTION_MODE
            or "selection_score_threshold" in selective_report
        ):
            raise RuntimeError(
                "selective-edge report does not use the exact model direction argmax schema"
            )
    except Exception as exc:
        failures.append(str(exc))

    dataset_raw = str(selective_report.get("dataset_dir") or "").strip()
    dataset_dir = Path(dataset_raw).expanduser().resolve() if dataset_raw else Path("/")
    prediction_evidence = (
        selective_report.get("prediction_evidence")
        if isinstance(selective_report.get("prediction_evidence"), dict)
        else {}
    )
    prediction_path_raw = str(prediction_evidence.get("path") or "").strip()
    resolved_prediction_path = Path("/")
    validated_prediction_evidence: dict[str, Any] = {}
    if not prediction_path_raw or not dataset_raw or not candidate_bundle_dir:
        failures.append(
            "selective-edge report lacks prediction path, dataset_dir, or bundle_dir"
        )
    else:
        try:
            (
                resolved_prediction_path,
                _validated_report,
                validated_prediction_evidence,
            ) = resolve_and_validate_prediction_evidence(
                Path(prediction_path_raw),
                prediction_report_path=selective_edge_report_path,
                bundle_dir=bundle_dir,
                dataset_dir=dataset_dir,
            )
            forbidden_columns = {
                "anchor_logits",
                "delta_logits",
                "anchor_gate",
            }.intersection(validated_prediction_evidence.get("columns") or [])
            if forbidden_columns:
                raise RuntimeError(
                    f"prediction evidence contains forbidden legacy columns: "
                    f"{sorted(forbidden_columns)}"
                )
        except Exception as exc:
            failures.append(str(exc))

    candidate_specialist_contract = _candidate_bundle_specialist_contract(
        candidate_audit, contract_mode
    )
    selective_specialist_contract = _selective_edge_specialist_contract(
        selective_report, contract_mode
    )
    failures.extend(candidate_specialist_contract["failures"])
    failures.extend(selective_specialist_contract["failures"])

    return {
        "ready": not failures,
        "contract_mode": contract_mode,
        "selective_edge_contract_mode": selective_mode,
        "expected_input_dim": MODEL_NATIVE_SIGNAL_DIM,
        "model_native_signal_contract": signal_contract,
        "candidate_bundle_audit_sha256": (
            _sha256_file(candidate_bundle_audit_path)
            if candidate_bundle_audit_path.is_file()
            else ""
        ),
        "selective_edge_report_sha256": (
            _sha256_file(selective_edge_report_path)
            if selective_edge_report_path.is_file()
            else ""
        ),
        "bundle_metadata_path": str(metadata_path),
        "bundle_metadata_sha256": (
            _sha256_file(metadata_path) if metadata_path.is_file() else ""
        ),
        "candidate_bundle_seq_input_dim": int(metadata.get("seq_input_dim") or 0),
        "candidate_bundle_snap_input_dim": int(metadata.get("snap_input_dim") or 0),
        "selective_edge_seq_input_dim": int(
            selective_report.get("bundle_seq_input_dim") or 0
        ),
        "selective_edge_snap_input_dim": int(
            selective_report.get("bundle_snap_input_dim") or 0
        ),
        "candidate_bundle_audit_json": str(candidate_bundle_audit_path),
        "selective_edge_report_json": str(selective_edge_report_path),
        "candidate_bundle_dir": candidate_bundle_dir,
        "selective_edge_bundle_dir": selective_bundle_dir,
        "dataset_dir": str(dataset_dir) if dataset_raw else "",
        "authoritative_predictions_path": (
            str(resolved_prediction_path) if prediction_path_raw else ""
        ),
        "prediction_evidence": validated_prediction_evidence,
        "selective_edge_feature_mask_ablation": (
            selective_report.get("feature_mask_ablation")
            if isinstance(selective_report.get("feature_mask_ablation"), dict)
            else {}
        ),
        "candidate_audit_decision": str(candidate_audit.get("decision") or ""),
        "selective_edge_decision": str(selective_report.get("decision") or ""),
        "candidate_specialist_contract": candidate_specialist_contract,
        "selective_edge_specialist_contract": selective_specialist_contract,
        "require_identity_artifacts": bool(require_identity_artifacts),
        "failures": failures,
    }

def _safe_mean(values: pd.Series) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else None


def _safe_percentile(values: pd.Series, q: float) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, q)) if vals.size else None


def _profit_factor(values: pd.Series) -> float | None:
    vals = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    gains = float(vals[vals > 0.0].sum())
    losses = float(vals[vals < 0.0].sum())
    if losses == 0.0:
        return None
    return float(gains / abs(losses))


def _max_drawdown(values: pd.Series) -> tuple[float, float]:
    vals = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(np.float64)
    if vals.size == 0:
        return 0.0, 0.0
    equity = np.concatenate([[0.0], np.cumsum(vals)])
    dd = equity - np.maximum.accumulate(equity)
    signed = float(np.min(dd))
    return abs(signed), signed


def normalize_trades(
    raw: pd.DataFrame,
    *,
    policy_id: str,
) -> tuple[pd.DataFrame, list[str]]:
    failures: list[str] = []
    if raw.empty:
        raise RuntimeError("replay trades input is empty")

    forbidden_policy_columns = sorted(
        {"threshold_top_frac", "score_threshold"}.intersection(raw.columns)
    )
    if forbidden_policy_columns:
        raise RuntimeError(
            "replay trades contain retired direction-threshold columns: "
            f"{forbidden_policy_columns}"
        )
    retired_sizing_columns = sorted(
        {
            "dynamic_sizing_applied",
            "applied_size_multiplier",
            "replay_size_multiplier",
            "sizing_authority_contract",
        }.intersection(raw.columns)
    )
    if retired_sizing_columns:
        raise RuntimeError(
            "replay trades contain forbidden execution-sizing columns: "
            f"{retired_sizing_columns}"
        )

    required = {
        *MODEL_NATIVE_REPLAY_REQUIRED_COLUMNS,
        "gross_pnl_bps",
        "direction_correct",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(
            f"replay trades input lacks exact model-native columns: {missing}"
        )

    out = raw.copy()
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    out["net_pnl_bps"] = pd.to_numeric(out["net_pnl_bps"], errors="coerce")
    out["gross_pnl_bps"] = pd.to_numeric(out["gross_pnl_bps"], errors="coerce")
    if out[["entry_time", "net_pnl_bps", "gross_pnl_bps"]].isna().any().any():
        raise RuntimeError("replay trades contain invalid entry_time or PnL values")

    out["policy_id"] = out["policy_id"].astype(str)
    observed_policies = sorted({str(value).strip() for value in out["policy_id"].dropna().astype(str) if str(value).strip()})
    if observed_policies != [str(policy_id)]:
        failures.append(
            "replay trades policy_id mismatch: "
            f"expected={[str(policy_id)]} observed={observed_policies}"
        )
    out["fold"] = out["entry_time"].dt.year.astype(str)
    out["source_split"] = out["source_split"].astype(str).str.strip().str.lower()
    observed_splits = sorted(set(out["source_split"]))
    if observed_splits != [REPLAY_REQUIRED_SPLIT]:
        failures.append(
            "replay trades are not exact test-split rows: "
            f"observed={observed_splits} expected={[REPLAY_REQUIRED_SPLIT]}"
        )
    out["side"] = out["side"].astype(str).str.upper()
    out["session"] = out["session"].astype(str).str.upper()
    out["direction_correct"] = pd.Series(out["direction_correct"]).map(
        lambda x: (str(x).strip().lower() == "true") if str(x).strip().lower() in {"true", "false"} else x
    )
    if not bool(out["direction_correct"].isin([True, False, 0, 1]).all()):
        raise RuntimeError("replay trades contain invalid direction_correct values")
    out["entry_day"] = out["entry_time"].dt.strftime("%Y-%m-%d")
    out["entry_month"] = out["entry_time"].dt.strftime("%Y-%m")
    if "tail_bucket" not in out.columns:
        pnl_values = pd.to_numeric(out["net_pnl_bps"], errors="coerce")
        p05 = float(pnl_values.quantile(0.05)) if len(pnl_values) else 0.0
        p10 = float(pnl_values.quantile(0.10)) if len(pnl_values) else 0.0
        out["tail_bucket"] = np.select(
            [pnl_values <= p05, pnl_values <= p10],
            ["tail_loss_p05", "tail_loss_p10"],
            default="normal",
        )
    if "bad_path_bucket" not in out.columns:
        bad_path = pd.to_numeric(out["bad_path_prob"], errors="coerce")
        p75 = float(bad_path.quantile(0.75)) if bad_path.notna().any() else 0.0
        p90 = float(bad_path.quantile(0.90)) if bad_path.notna().any() else 0.0
        out["bad_path_bucket"] = np.select(
            [bad_path >= p90, bad_path >= p75],
            ["bad_path_p90", "bad_path_p75"],
            default="normal",
        )

    years = set(int(x) for x in out["entry_time"].dt.year.dropna().astype(int).unique())
    if years != {REPLAY_REQUIRED_YEAR}:
        failures.append(
            f"replay trades must contain only required year {REPLAY_REQUIRED_YEAR}: "
            f"observed={sorted(years)}"
        )

    keep = [
        "fold",
        "policy_id",
        "session",
        "entry_day",
        "entry_month",
        "entry_time",
        "source_split",
        "side",
        "direction_correct",
        "score",
        "gross_pnl_bps",
        "net_pnl_bps",
        "mfe_bps",
        "mae_bps",
    ]
    optional_prefixes = (
        "foundation_",
        "specialist_",
        "ctx_",
        "state_",
        "teacher_",
        "candidate_",
        "exit_",
        "mfe_protect_",
    )
    for optional in (
        "candidate_uid",
        "trade_uid",
        "exit_mode",
        "exit_policy_config_hash",
        "entry_price",
        "exit_price",
        "exit_time",
        "exit_reason",
        "held_bars",
        "horizon_bars",
        "row_simulation_mode",
        "vol_regime",
        "tail_bucket",
        "bad_path_bucket",
        "cost_stress_bps",
        "policy_config_hash",
        "p_long",
        "p_short",
        "p_flat",
        "path_quality_pred",
        "bad_path_prob",
        "tradable_prob",
        "clean_edge_prob",
        "survival_prob",
        "tf_agreement_prob",
        "position_size_pred",
        "hold_horizon_pred",
        "direction_authority",
        "selection_score_mode",
        "filters_applied",
        "offline_only",
        "diagnostic_scope",
        "pnl_normalization",
        "execution_order_simulation",
        "position_size_applied",
    ):
        if optional in out.columns and optional not in keep:
            keep.append(optional)
    for col in out.columns:
        if any(str(col).startswith(prefix) for prefix in optional_prefixes) and col not in keep:
            keep.append(str(col))
    return out[[c for c in keep if c in out.columns]].copy(), failures


def _safe_value_counts(frame: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in frame.columns:
        return {}
    return {
        str(k): int(v)
        for k, v in frame[col].fillna("UNKNOWN").astype(str).str.upper().value_counts(dropna=False).to_dict().items()
    }


def audit_model_native_replay_trades(trades: pd.DataFrame) -> dict[str, Any]:
    missing = [
        col for col in MODEL_NATIVE_REPLAY_REQUIRED_COLUMNS if col not in trades.columns
    ]
    failures: list[str] = []
    if missing:
        failures.append(
            f"model-native replay trade log missing required columns: {missing}"
        )

    numeric_status: dict[str, dict[str, Any]] = {}
    for col in MODEL_NATIVE_REPLAY_NUMERIC_COLUMNS:
        if col not in trades.columns:
            continue
        values = pd.to_numeric(trades[col], errors="coerce")
        arr = values.to_numpy(dtype=np.float64)
        finite = bool(np.isfinite(arr).all()) if len(arr) else False
        null_count = int(values.isna().sum())
        numeric_status[col] = {
            "finite": finite,
            "null_count": null_count,
            "min": float(values.min()) if len(values) and values.notna().any() else None,
            "max": float(values.max()) if len(values) and values.notna().any() else None,
        }
        if not finite or null_count > 0:
            failures.append(f"model-native replay numeric column not fully finite: {col}")

    for prob_col in ("p_long", "p_short", "p_flat"):
        if prob_col in trades.columns:
            values = pd.to_numeric(trades[prob_col], errors="coerce")
            if bool(((values < 0.0) | (values > 1.0)).any()):
                failures.append(
                    f"model-native replay probability column outside [0,1]: {prob_col}"
                )

    if {"p_long", "p_short", "p_flat"}.issubset(trades.columns):
        probability_matrix = trades[["p_long", "p_short", "p_flat"]].apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float64)
        prob_sum = (
            pd.to_numeric(trades["p_long"], errors="coerce")
            + pd.to_numeric(trades["p_short"], errors="coerce")
            + pd.to_numeric(trades["p_flat"], errors="coerce")
        )
        probability_sum_max_abs_error = float((prob_sum - 1.0).abs().max()) if len(prob_sum) else float("inf")
        if not np.isfinite(probability_sum_max_abs_error) or probability_sum_max_abs_error > 1e-6:
            failures.append(
                "model-native replay probability sum drifts from 1.0: "
                f"max_abs_error={probability_sum_max_abs_error}"
            )
        if np.isfinite(probability_matrix).all() and "side" in trades.columns:
            winner_counts = np.count_nonzero(
                probability_matrix
                == np.max(probability_matrix, axis=1, keepdims=True),
                axis=1,
            )
            tied_rows = int(np.count_nonzero(winner_counts != 1))
            if tied_rows:
                failures.append(
                    "model-native replay direction probabilities have no unique "
                    f"top class: rows={tied_rows}"
                )
            expected_side = np.asarray(CLASS_ORDER)[
                np.argmax(probability_matrix, axis=1)
            ]
            observed_side = trades["side"].astype(str).str.upper().to_numpy()
            mismatches = int(np.count_nonzero(expected_side != observed_side))
            if mismatches:
                failures.append(
                    "model-native replay side mismatches model LONG/SHORT/FLAT argmax: "
                    f"rows={mismatches}"
                )
            flat_rows = int(np.count_nonzero(expected_side == "FLAT"))
            if flat_rows:
                failures.append(
                    "model-native replay trade log contains model-FLAT actions: "
                    f"rows={flat_rows}"
                )
    else:
        probability_sum_max_abs_error = None

    if "entry_time" in trades.columns:
        parsed_time = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
        if parsed_time.isna().any():
            failures.append("model-native replay entry_time contains invalid timestamps")
        elif set(parsed_time.dt.year.astype(int)) != {REPLAY_REQUIRED_YEAR}:
            failures.append(
                f"model-native replay rows are outside required year {REPLAY_REQUIRED_YEAR}"
            )

    if "source_split" in trades.columns:
        splits = trades["source_split"].astype(str).str.strip().str.lower()
        if set(splits) != {REPLAY_REQUIRED_SPLIT}:
            failures.append("model-native replay rows are not exact test-split rows")
    if "exit_mode" in trades.columns:
        if not bool((trades["exit_mode"].astype(str) == LABEL_HORIZON_EXIT_MODE).all()):
            failures.append("model-native replay contains a non-label-horizon exit")
    if {"held_bars", "horizon_bars"}.issubset(trades.columns):
        held = pd.to_numeric(trades["held_bars"], errors="coerce")
        horizon = pd.to_numeric(trades["horizon_bars"], errors="coerce")
        if not bool((held == horizon).all()):
            failures.append("model-native replay held_bars differs from label horizon")
    if "row_simulation_mode" in trades.columns:
        if not bool((trades["row_simulation_mode"].astype(str) == "independent").all()):
            failures.append("model-native replay rows are not independent")
    for boolean_column, expected in (
        ("filters_applied", False),
        ("offline_only", True),
        ("execution_order_simulation", False),
        ("position_size_applied", False),
    ):
        if boolean_column in trades.columns:
            normalized = trades[boolean_column].map(
                lambda value: str(value).strip().lower()
            )
            if not bool((normalized == str(expected).lower()).all()):
                failures.append(
                    f"model-native replay {boolean_column} is not exactly {expected}"
                )
    for column, expected in (
        ("diagnostic_scope", OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE),
        ("pnl_normalization", UNIT_NORMALIZED_PNL_MODE),
    ):
        if column in trades.columns and not bool(
            (trades[column].astype(str) == expected).all()
        ):
            failures.append(
                f"model-native replay {column} is not exactly {expected!r}"
            )
    retired_sizing_columns = sorted(
        {
            "dynamic_sizing_applied",
            "applied_size_multiplier",
            "replay_size_multiplier",
            "sizing_authority_contract",
        }.intersection(trades.columns)
    )
    if retired_sizing_columns:
        failures.append(
            "model-native replay exposes execution-sizing columns: "
            f"{retired_sizing_columns}"
        )

    session_counts = _safe_value_counts(trades, "session")
    if "session" in trades.columns:
        sessions = trades["session"].fillna("").astype(str).str.strip().str.upper()
        if bool(sessions.isin(["", "UNKNOWN", "NAN", "NONE"]).any()):
            failures.append(
                "model-native replay session state contains missing/UNKNOWN values"
            )

    if "vol_regime" in trades.columns:
        regimes = trades["vol_regime"].fillna("").astype(str).str.strip().str.upper()
        if bool(regimes.isin(["", "UNKNOWN", "NAN", "NONE"]).any()):
            failures.append(
                "model-native replay volatility regime contains missing/UNKNOWN values"
            )

    if "policy_id" in trades.columns:
        policies = trades["policy_id"].fillna("").astype(str).str.strip()
        if bool((policies == "").any()):
            failures.append("model-native replay policy_id contains blank values")

    if "bad_path_prob" in trades.columns:
        bad_path = pd.to_numeric(trades["bad_path_prob"], errors="coerce")
        if bool(((bad_path < 0.0) | (bad_path > 1.0)).any()):
            failures.append("model-native replay bad_path_prob is outside [0,1]")

    side_counts = _safe_value_counts(trades, "side")
    valid_side_rows = 0
    if "side" in trades.columns:
        valid_sides = trades["side"].astype(str).str.upper().isin(["LONG", "SHORT"])
        valid_side_rows = int(valid_sides.sum())
        if not bool(valid_sides.all()):
            failures.append(
                "model-native replay action side contains non-LONG/SHORT values"
            )
        if valid_side_rows <= 0:
            failures.append("model-native replay action side has no LONG/SHORT rows")

    return {
        "ready": not failures,
        "required_columns": list(MODEL_NATIVE_REPLAY_REQUIRED_COLUMNS),
        "missing_columns": missing,
        "numeric_status": numeric_status,
        "probability_sum_max_abs_error": probability_sum_max_abs_error,
        "session_counts": session_counts,
        "side_counts": side_counts,
        "valid_side_rows": valid_side_rows,
        "failures": failures,
    }


def _metrics_row(scope: str, fold: str, policy_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    pnl = pd.to_numeric(frame["net_pnl_bps"], errors="coerce")
    gross = pd.to_numeric(frame["gross_pnl_bps"], errors="coerce")
    dd_abs, dd_signed = _max_drawdown(pnl)
    direction = pd.to_numeric(frame.get("direction_correct", pd.Series(dtype=float)), errors="coerce")
    return {
        "scope": scope,
        "fold": fold,
        "policy_id": policy_id,
        "n_trades": int(len(frame)),
        "n_days": int(frame["entry_day"].nunique()),
        "n_months": int(frame["entry_month"].nunique()),
        "net_sum_bps": float(pnl.sum()),
        "net_mean_bps": _safe_mean(pnl),
        "net_median_bps": _safe_percentile(pnl, 50),
        "net_p10_bps": _safe_percentile(pnl, 10),
        "net_p90_bps": _safe_percentile(pnl, 90),
        "gross_mean_bps": _safe_mean(gross),
        "win_rate": float((pnl > 0.0).mean()) if len(pnl) else None,
        "profit_factor": _profit_factor(pnl),
        "max_win_bps": float(pnl.max()) if len(pnl) else None,
        "max_loss_bps": float(pnl.min()) if len(pnl) else None,
        "max_drawdown_bps": dd_abs,
        "max_drawdown_signed_bps": dd_signed,
        "mean_score": _safe_mean(frame["score"]),
        "mean_mfe_bps": _safe_mean(frame["mfe_bps"]),
        "mean_mae_bps": _safe_mean(frame["mae_bps"]),
        "long_rate": float((frame["side"].astype(str).str.upper() == "LONG").mean()),
        "short_rate": float((frame["side"].astype(str).str.upper() == "SHORT").mean()),
        "direction_precision": _safe_mean(direction),
        "avg_trades_per_day": float(len(frame) / max(frame["entry_day"].nunique(), 1)),
    }


def build_replay_tables(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    for (policy_id, fold), frame in trades.groupby(["policy_id", "fold"], sort=True):
        metric_rows.append(_metrics_row("fold", str(fold), str(policy_id), frame))
    for policy_id, frame in trades.groupby("policy_id", sort=True):
        metric_rows.append(_metrics_row("aggregate", "ALL", str(policy_id), frame))
    metrics = pd.DataFrame(metric_rows)

    daily = (
        trades.groupby(["policy_id", "entry_day"], as_index=False)
        .agg(
            n_trades=("net_pnl_bps", "size"),
            net_sum_bps=("net_pnl_bps", "sum"),
            net_mean_bps=("net_pnl_bps", "mean"),
            wins=("net_pnl_bps", lambda s: int((s > 0.0).sum())),
        )
    )
    daily["win_rate"] = daily["wins"] / daily["n_trades"].clip(lower=1)

    monthly = (
        trades.groupby(["policy_id", "entry_month"], as_index=False)
        .agg(
            n_trades=("net_pnl_bps", "size"),
            net_sum_bps=("net_pnl_bps", "sum"),
            net_mean_bps=("net_pnl_bps", "mean"),
            wins=("net_pnl_bps", lambda s: int((s > 0.0).sum())),
        )
    )
    monthly["month"] = monthly["entry_month"]
    monthly["win_rate"] = monthly["wins"] / monthly["n_trades"].clip(lower=1)
    return metrics, daily, monthly


def _slice_metrics_row(slice_dimension: str, slice_value: str, policy_id: str, frame: pd.DataFrame) -> dict[str, Any]:
    row = _metrics_row("slice", f"{slice_dimension}={slice_value}", policy_id, frame)
    row["slice_family"] = slice_dimension
    row["slice_dimension"] = slice_dimension
    row["slice_value"] = slice_value
    if "bad_path_prob" in frame.columns:
        bad_path = pd.to_numeric(frame["bad_path_prob"], errors="coerce")
        row["mean_bad_path_prob"] = _safe_mean(bad_path)
        row["bad_path_rate"] = float((bad_path >= 0.5).mean()) if len(bad_path) else None
    if "path_quality_pred" in frame.columns:
        row["mean_path_quality_pred"] = _safe_mean(frame["path_quality_pred"])
        row["path_quality_p10"] = _safe_percentile(frame["path_quality_pred"], 10)
    if "mae_bps" in frame.columns:
        row["mae_p95_bps"] = _safe_percentile(frame["mae_bps"], 95)
    return row


def build_replay_slices(trades: pd.DataFrame) -> pd.DataFrame:
    dimensions = [
        ("session", "session"),
        ("regime", "vol_regime"),
        ("direction", "side"),
        ("tail", "tail_bucket"),
        ("bad_path", "bad_path_bucket"),
    ]
    rows: list[dict[str, Any]] = []
    for slice_dimension, column in dimensions:
        if column not in trades.columns:
            continue
        work = trades.copy()
        work[column] = work[column].fillna("UNKNOWN").astype(str)
        for (policy_id, value), frame in work.groupby(["policy_id", column], sort=True):
            rows.append(_slice_metrics_row(slice_dimension, str(value), str(policy_id), frame))
    return pd.DataFrame(rows)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Candidate Replay Evidence",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Trades: `{report['n_trades']}`",
        f"- Out dir: `{report['out_dir']}`",
        f"- Promotion/shadow/live allowed: `{report['promotion_shadow_live_allowed']}`",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        lines.extend(f"- {failure}" for failure in report["failures"])
    else:
        lines.append("- None")
    atomic_write_text(path, "\n".join(lines) + "\n")


def _trade_log_authority_contract(
    *,
    manifest_path: Path,
    trades_path: Path,
    selective_edge_report_path: Path,
    identity: dict[str, Any],
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        require_newest_immutable_event(manifest_path, TRADE_LOG_EVENT_PREFIX)
    except Exception as exc:
        failures.append(f"trade-log immutable authority failed: {exc}")
    manifest = _read_json_if_exists(manifest_path)
    if not manifest:
        failures.append(f"missing trade-log manifest: {manifest_path}")
    if manifest.get("schema_version") != TRADE_LOG_SCHEMA_VERSION:
        failures.append("trade-log manifest schema_version is not model-native")
    if manifest.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE:
        failures.append("trade-log manifest contract_mode is not model-native seq513")
    if manifest.get("expected_signal_dim") != MODEL_NATIVE_SIGNAL_DIM:
        failures.append("trade-log manifest expected_signal_dim is not 513")
    if str(manifest.get("decision") or "") != "PASS" or manifest.get("failures"):
        failures.append("trade-log manifest is not a zero-failure PASS")
    declared_trades = str(manifest.get("trades_path") or "").strip()
    if not declared_trades or Path(declared_trades).expanduser().resolve() != trades_path:
        failures.append("trade-log manifest trades_path mismatch")
    declared_sha = str(manifest.get("trades_sha256") or "").strip().lower()
    observed_sha = _sha256_file(trades_path) if trades_path.is_file() else ""
    if len(declared_sha) != 64 or declared_sha != observed_sha:
        failures.append("trade-log manifest trades SHA-256 is missing or mismatched")
    declared_prediction_report = str(
        manifest.get("prediction_report_json") or ""
    ).strip()
    if (
        not declared_prediction_report
        or Path(declared_prediction_report).expanduser().resolve()
        != selective_edge_report_path
    ):
        failures.append("trade-log manifest prediction report mismatch")
    prediction_evidence = (
        manifest.get("prediction_evidence")
        if isinstance(manifest.get("prediction_evidence"), dict)
        else {}
    )
    if prediction_evidence != identity.get("prediction_evidence"):
        failures.append("trade-log and validated prediction evidence declarations differ")
    declared_report_sha = str(manifest.get("prediction_report_sha256") or "").lower()
    if declared_report_sha != _sha256_file(selective_edge_report_path):
        failures.append("trade-log prediction report SHA-256 is missing or mismatched")
    if str(manifest.get("selection_score_mode") or "") != MODEL_DIRECTION_SELECTION_MODE:
        failures.append("trade-log manifest direction mode is not model_direction_argmax")

    direction_policy = (
        manifest.get("direction_policy_contract")
        if isinstance(manifest.get("direction_policy_contract"), dict)
        else {}
    )
    if direction_policy != _direction_policy_contract():
        failures.append("trade-log direction policy contract is not exact model-native argmax")
    retired_sizing_keys = {
        "dynamic_sizing_applied",
        "applied_size_multiplier",
        "replay_size_multiplier",
        "sizing_authority_contract",
    }
    stale_manifest_sizing = sorted(retired_sizing_keys.intersection(manifest))
    stale_direction_sizing = sorted(
        retired_sizing_keys.intersection(direction_policy)
    )
    if stale_manifest_sizing:
        failures.append(
            f"trade-log manifest exposes execution-sizing fields: {stale_manifest_sizing}"
        )
    if stale_direction_sizing:
        failures.append(
            "trade-log direction policy exposes execution-sizing fields: "
            f"{stale_direction_sizing}"
        )
    exit_policy = (
        manifest.get("exit_policy_contract")
        if isinstance(manifest.get("exit_policy_contract"), dict)
        else {}
    )
    if exit_policy != label_horizon_exit_policy_contract():
        failures.append("trade-log exit policy is not the exact label-horizon contract")
    policy = (
        manifest.get("policy_config")
        if isinstance(manifest.get("policy_config"), dict)
        else {}
    )
    forbidden_policy_keys = sorted(
        {
            "min_direction_prob",
            "min_score_floor",
            "threshold_source",
            "threshold_top_frac",
            "score_threshold",
            "top_fracs",
            "expected_utility_side",
            "utility_side",
            "session_allowed",
            "trend_allowed",
            "path_allowed",
        }.intersection(policy)
    )
    if forbidden_policy_keys:
        failures.append(
            f"trade-log policy contains retired direction selectors: {forbidden_policy_keys}"
        )
    if str(policy.get("selection_score_mode") or "") != MODEL_DIRECTION_SELECTION_MODE:
        failures.append("trade-log policy direction mode is not model_direction_argmax")
    if policy.get("direction_authority") != "argmax(final_calibrated_direction_logits)":
        failures.append("trade-log policy lacks final calibrated logits authority")
    if policy.get("filters_applied") is not False:
        failures.append("trade-log policy does not prove filters_applied=false")
    stale_policy_sizing = sorted(retired_sizing_keys.intersection(policy))
    if stale_policy_sizing:
        failures.append(
            f"trade-log policy exposes execution-sizing fields: {stale_policy_sizing}"
        )
    exact_policy = {
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
    }
    for key, expected in exact_policy.items():
        if policy.get(key) != expected:
            failures.append(
                f"trade-log policy {key}={policy.get(key)!r} expected={expected!r}"
            )

    authority = (
        manifest.get("model_native_authority")
        if isinstance(manifest.get("model_native_authority"), dict)
        else {}
    )
    for label, path_key, hash_key, prefix in (
        ("state", "state_json", "state_sha256", STATE_EVENT_PREFIX),
        (
            "candidate readiness",
            "candidate_readiness_json",
            "candidate_readiness_sha256",
            CANDIDATE_EVENT_PREFIX,
        ),
    ):
        raw_path = str(authority.get(path_key) or "").strip()
        declared_hash = str(authority.get(hash_key) or "").strip().lower()
        event_path = Path(raw_path).expanduser().resolve() if raw_path else Path("/")
        if not raw_path or len(declared_hash) != 64:
            failures.append(f"trade-log {label} authority binding is incomplete")
            continue
        try:
            require_newest_immutable_event(event_path, prefix)
        except Exception as exc:
            failures.append(f"trade-log {label} immutable authority failed: {exc}")
            continue
        if _sha256_file(event_path) != declared_hash:
            failures.append(f"trade-log {label} authority hash mismatch")

    counts_raw = str(manifest.get("counts_path") or "").strip()
    counts_path = Path(counts_raw).expanduser().resolve() if counts_raw else Path("/")
    if (
        not counts_raw
        or not counts_path.is_file()
        or str(manifest.get("counts_sha256") or "").lower()
        != _sha256_file(counts_path)
    ):
        failures.append("trade-log policy-count artifact hash is missing or mismatched")
    else:
        try:
            counts_frame = pd.read_csv(counts_path)
        except Exception as exc:
            failures.append(f"trade-log policy-count artifact is unreadable: {exc}")
        else:
            if len(counts_frame) != 1:
                failures.append("trade-log policy-count artifact must contain exactly one row")
            else:
                counts_row = counts_frame.iloc[0].to_dict()
                manifest_counts = (
                    manifest.get("policy_counts")
                    if isinstance(manifest.get("policy_counts"), dict)
                    else {}
                )

                def _int_value(value: Any) -> int | None:
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        return None
                    if not np.isfinite(numeric) or not numeric.is_integer():
                        return None
                    return int(numeric)

                def _bool_value(value: Any) -> bool | None:
                    normalized = str(value).strip().lower()
                    if normalized == "true":
                        return True
                    if normalized == "false":
                        return False
                    return None

                evaluated = _int_value(counts_row.get("evaluated_rows"))
                flat = _int_value(counts_row.get("model_flat_rows"))
                non_flat = _int_value(counts_row.get("non_flat_argmax_rows"))
                expected_trades = _int_value(counts_row.get("expected_trades"))
                trades = _int_value(counts_row.get("trades"))
                exact_count_proof = (
                    evaluated is not None
                    and flat is not None
                    and non_flat is not None
                    and expected_trades is not None
                    and trades is not None
                    and evaluated == flat + non_flat
                    and expected_trades == non_flat
                    and trades == non_flat
                    and _bool_value(
                        counts_row.get("trades_equal_non_flat_argmax_rows")
                    )
                    is True
                    and _bool_value(counts_row.get("filters_applied")) is False
                    and _bool_value(counts_row.get("offline_only")) is True
                    and str(counts_row.get("diagnostic_scope"))
                    == OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE
                    and str(counts_row.get("pnl_normalization"))
                    == UNIT_NORMALIZED_PNL_MODE
                    and _bool_value(
                        counts_row.get("execution_order_simulation")
                    )
                    is False
                    and _bool_value(counts_row.get("position_size_applied"))
                    is False
                    and _bool_value(counts_row.get("occupancy_filter_applied")) is False
                    and _bool_value(counts_row.get("cooldown_applied")) is False
                    and _bool_value(counts_row.get("max_trades_per_day_applied"))
                    is False
                    and _bool_value(counts_row.get("daily_loss_limit_applied"))
                    is False
                    and _bool_value(counts_row.get("invalid_path_skip_allowed"))
                    is False
                )
                if not exact_count_proof:
                    failures.append(
                        "trade-log policy counts do not prove one independent trade per "
                        "non-FLAT argmax row with filters=false"
                    )
                critical_count_keys = (
                    "evaluated_rows",
                    "model_flat_rows",
                    "non_flat_argmax_rows",
                    "expected_trades",
                    "trades",
                    "trades_equal_non_flat_argmax_rows",
                    "filters_applied",
                    "offline_only",
                    "diagnostic_scope",
                    "pnl_normalization",
                    "execution_order_simulation",
                    "position_size_applied",
                )
                for key in critical_count_keys:
                    if key not in manifest_counts or str(manifest_counts.get(key)) != str(
                        counts_row.get(key)
                    ):
                        failures.append(
                            f"trade-log manifest/counts mismatch for exact replay field: {key}"
                        )

                report_exact = (
                    _int_value(manifest.get("n_test_rows")) == evaluated
                    and _int_value(manifest.get("n_model_flat_rows")) == flat
                    and _int_value(manifest.get("n_non_flat_argmax_rows")) == non_flat
                    and _int_value(manifest.get("n_trades")) == trades
                    and manifest.get("trades_equal_non_flat_argmax_rows") is True
                    and manifest.get("filters_applied") is False
                    and manifest.get("offline_only") is True
                    and manifest.get("diagnostic_scope")
                    == OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE
                    and manifest.get("pnl_normalization")
                    == UNIT_NORMALIZED_PNL_MODE
                    and manifest.get("execution_order_simulation") is False
                    and manifest.get("position_size_applied") is False
                )
                if not report_exact:
                    failures.append(
                        "trade-log report does not mirror exact replay counts and filters=false"
                    )
                try:
                    trade_row_count = len(pd.read_csv(trades_path))
                except Exception as exc:
                    failures.append(f"trade-log CSV is unreadable for cardinality proof: {exc}")
                else:
                    if trades is None or trade_row_count != trades:
                        failures.append(
                            "trade-log CSV row count differs from exact non-FLAT trade count"
                        )
    return {
        "ready": not failures,
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path) if manifest_path.is_file() else "",
        "trades_path": str(trades_path),
        "trades_sha256": observed_sha,
        "selection_score_mode": manifest.get("selection_score_mode"),
        "direction_policy_contract": direction_policy,
        "exit_policy_contract": exit_policy,
        "model_native_authority": authority,
        "failures": failures,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    trades_path = Path(args.trades_path).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    candidate_bundle_audit_path = Path(args.candidate_bundle_audit_json).expanduser().resolve()
    selective_edge_report_path = Path(args.selective_edge_report_json).expanduser().resolve()
    trade_log_manifest_path = Path(args.trade_log_manifest_json).expanduser().resolve()
    ablation_id = ""
    policy_id = str(args.policy_id or "").strip()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"replay evidence out-dir must be new/empty for immutable publication: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = _read_table(trades_path)
    identity = _identity_contract(
        candidate_bundle_audit_path=candidate_bundle_audit_path,
        selective_edge_report_path=selective_edge_report_path,
        require_identity_artifacts=True,
        requested_contract_mode=MODEL_NATIVE_CONTRACT_MODE,
    )
    trade_log_authority = _trade_log_authority_contract(
        manifest_path=trade_log_manifest_path,
        trades_path=trades_path,
        selective_edge_report_path=selective_edge_report_path,
        identity=identity,
    )
    trades, failures = normalize_trades(
        raw,
        policy_id=str(args.policy_id),
    )
    failures.extend(identity["failures"])
    failures.extend(trade_log_authority["failures"])
    metrics, daily, monthly = build_replay_tables(trades)
    slices = build_replay_slices(trades)
    model_native_trade_audit = audit_model_native_replay_trades(trades)
    failures.extend(model_native_trade_audit["failures"])

    best = metrics[metrics["scope"].astype(str).isin(["aggregate", "all", "ALL"])]
    if best.empty:
        failures.append("no aggregate replay metrics were produced")
    else:
        row = best.sort_values("net_sum_bps", ascending=False).iloc[0]
        if int(row.get("n_trades") or 0) <= 0:
            failures.append("aggregate replay metrics have zero trades")

    event_created_utc = datetime.now(timezone.utc)
    timestamp = event_created_utc.strftime("%Y%m%dT%H%M%S%fZ")
    trades_out = out_dir / f"replay_policy_trades_{timestamp}.csv"
    metrics_out = out_dir / f"replay_policy_metrics_{timestamp}.csv"
    daily_out = out_dir / f"replay_policy_daily_{timestamp}.csv"
    monthly_out = out_dir / f"replay_policy_monthly_{timestamp}.csv"
    slices_out = out_dir / f"replay_policy_slices_{timestamp}.csv"
    report_json = out_dir / f"ENTRY_CANDIDATE_REPLAY_EVIDENCE_{timestamp}.json"
    report_md = out_dir / f"ENTRY_CANDIDATE_REPLAY_EVIDENCE_{timestamp}.md"

    for frame, path in (
        (trades, trades_out),
        (metrics, metrics_out),
        (daily, daily_out),
        (monthly, monthly_out),
        (slices, slices_out),
    ):
        atomic_write_text(path, frame.to_csv(index=False))
    artifact_hashes = {
        path.name: _sha256_file(path)
        for path in (trades_out, metrics_out, daily_out, monthly_out, slices_out)
    }

    best_row = best.sort_values("net_sum_bps", ascending=False).iloc[0].to_dict() if not best.empty else {}
    report = {
        "schema_version": "entry_candidate_replay_evidence_v2",
        "created_utc": event_created_utc.isoformat(),
        "decision": "PASS" if not failures else "FAIL",
        "contract_mode": identity["contract_mode"],
        "ablation_id": ablation_id,
        "policy_id": policy_id,
        "trades_path": str(trades_path),
        "out_dir": str(out_dir),
        "candidate_bundle_audit_json": str(candidate_bundle_audit_path),
        "selective_edge_report_json": str(selective_edge_report_path),
        "trade_log_manifest_json": str(trade_log_manifest_path),
        "candidate_bundle_dir": identity["candidate_bundle_dir"],
        "feature_mask_ablation": identity["selective_edge_feature_mask_ablation"],
        "replay_identity_contract": identity,
        "trade_log_authority_contract": trade_log_authority,
        "offline_only": True,
        "diagnostic_scope": OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
        "pnl_normalization": UNIT_NORMALIZED_PNL_MODE,
        "execution_order_simulation": False,
        "position_size_applied": False,
        "required_split": REPLAY_REQUIRED_SPLIT,
        "required_year": REPLAY_REQUIRED_YEAR,
        "n_trades": int(len(trades)),
        "policies": sorted(str(x) for x in trades["policy_id"].unique()),
        "best_aggregate_row": best_row,
        "trades_csv": str(trades_out),
        "metrics_csv": str(metrics_out),
        "daily_csv": str(daily_out),
        "monthly_csv": str(monthly_out),
        "slices_csv": str(slices_out),
        "artifact_hashes": artifact_hashes,
        "model_native_replay_trades_ready": bool(model_native_trade_audit["ready"]),
        "model_native_replay_trade_contract": model_native_trade_audit,
        "json_path": str(report_json),
        "md_path": str(report_md),
        "trainer_started": False,
        "replay_started": False,
        "promotion_shadow_live_allowed": False,
        "failures": failures,
    }
    serialized_report = json.dumps(
        report, indent=2, sort_keys=True, default=_json_default
    ) + "\n"
    atomic_write_text(report_json, serialized_report)
    _write_markdown(report_md, report)
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": failures,
                    "metrics_csv": str(metrics_out),
                    "monthly_csv": str(monthly_out),
                    "json_path": str(report_json),
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    if failures:
        raise SystemExit(2)
    return report


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trades-path", required=True)
    ap.add_argument("--trade-log-manifest-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--candidate-bundle-audit-json", required=True)
    ap.add_argument("--selective-edge-report-json", required=True)
    ap.add_argument("--policy-id", default="candidate_replay")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
