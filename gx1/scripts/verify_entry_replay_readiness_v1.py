#!/usr/bin/env python3
"""Verify immutable model-native Entry candidate replay readiness.

This gate does not run replay, train another direction policy, promote, shadow,
or trade. It checks
that a post-candidate specialist-fusion bundle has selective-edge evidence and
offline replay evidence.  No secondary direction authority is authorized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_SIGNAL_DIM,
    require_model_native_signal_contract,
)
from gx1.contracts.entry_model_native_adaptation_drift_v1 import (
    ModelNativeAdaptationDriftError,
    adaptation_bundle_identity_from_dir,
)
from gx1.contracts.entry_model_native_adaptation_lifecycle_v1 import (
    MODEL_NATIVE_REPLAY_READINESS_REQUIRED_ARTIFACTS,
    adaptation_lifecycle_handoff_metadata,
)
from gx1.execution.model_native_entry_replay_v1 import (
    OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
    UNIT_NORMALIZED_PNL_MODE,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
    MODEL_NATIVE_REQUIRED_SPECIALISTS,
    artifact_fingerprint_checks as _artifact_fingerprint_checks,
    artifact_fingerprints as _artifact_fingerprints,
    model_native_readiness_contract_metadata,
    readiness_check as _check,
    require_model_native_readiness_contract,
)
from gx1.contracts.immutable_event_authority_v1 import (
    next_immutable_event_created_utc,
    write_immutable_json_event,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_SELECTION_MODE,
    require_model_direction_decision_contract,
)
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    atomic_write_text,
    resolve_and_validate_prediction_evidence,
)
from gx1.scripts.materialize_entry_candidate_replay_evidence_v1 import (
    MODEL_NATIVE_REPLAY_REQUIRED_COLUMNS,
    _selective_edge_specialist_contract,
    audit_model_native_replay_trades,
)
from gx1.scripts.verify_entry_candidate_readiness_v1 import (
    _bundle_specialist_model_contract_passes,
    REQUIRED_MIN_GATE_ENTROPY,
)
CONTRACT_INPUT_DIMS = {MODEL_NATIVE_CONTRACT_MODE: MODEL_NATIVE_SIGNAL_DIM}
_TIMESTAMPED_JSON_RE = re.compile(r".+_\d{8}T\d{6}(?:\d{6})?Z\.json")
_TIMESTAMPED_CSV_RE = re.compile(r".+_\d{8}T\d{6}(?:\d{6})?Z\.csv")
READINESS_MODEL_NAME = "candidate"
MIN_TOP5_MEAN_PNL_BPS = 0.0
MIN_TOP10_MEAN_PNL_BPS = 0.0
MIN_TOP_DIRECTION_PRECISION = 0.95
MIN_DIRECTION_SLICE_PRECISION = 0.90
MIN_DIRECTION_SLICE_N = 20
MIN_REPLAY_NET_SUM_BPS = 0.0
MIN_REPLAY_PROFIT_FACTOR = 1.05
MAX_REPLAY_ABS_DRAWDOWN_BPS = 650.0


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_explicit_timestamped_json(path: Path) -> bool:
    return bool(_TIMESTAMPED_JSON_RE.fullmatch(path.name))


def _expected_active_heads_for_mode(contract_mode: str) -> set[str]:
    if str(contract_mode).strip() != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError(f"retired readiness contract mode: {contract_mode!r}")
    return set(MODEL_NATIVE_ACTIVE_HEADS)


def _readiness_contract_is_exact(report: dict[str, Any]) -> bool:
    try:
        require_model_native_readiness_contract(
            report.get("model_native_readiness_contract"),
            context="REPLAY_READINESS_UPSTREAM",
        )
    except RuntimeError:
        return False
    return True


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _all_ok(checks: list[dict[str, Any]]) -> bool:
    return all(bool(check.get("ok")) for check in checks)


def _model_native_pretrain_audit_checks(
    path: Path, report: dict[str, Any], *, expected_dataset_dir: Path
) -> list[dict[str, Any]]:
    exists = path.exists()
    checks = [
        _check(
            "model-native XAU pretrain audit artifact exists",
            exists,
            {"path": str(path)},
        )
    ]
    if not exists:
        return checks
    checks.extend(
        [
            _check(
                "model-native XAU pretrain audit PASS",
                str(report.get("decision")) == "PASS",
                {"decision": report.get("decision"), "failures": report.get("failures")},
            ),
            _check(
                "model-native XAU pretrain audit dataset_dir matches expected pin",
                _same_resolved_path(report.get("dataset_dir"), expected_dataset_dir),
                {"expected_dataset_dir": str(expected_dataset_dir), "audit_dataset_dir": report.get("dataset_dir")},
            ),
        ]
    )
    return checks


def _same_resolved_path(actual: Any, expected: Path) -> bool:
    if actual in (None, ""):
        return False
    try:
        return Path(str(actual)).resolve(strict=False) == expected.resolve(strict=False)
    except (OSError, RuntimeError):
        return False


def _normalize_contract_mode(value: Any) -> str:
    raw = str(value or MODEL_NATIVE_CONTRACT_MODE).strip()
    if raw != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError(
            f"retired replay contract mode is forbidden: {raw!r}"
        )
    return MODEL_NATIVE_CONTRACT_MODE


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


_RETIRED_HIER_PUBLIC_DIRECTION_COMPOSITION_KEY = (
    "hier_public_direction_composition"
)
MODEL_NATIVE_DIRECTION_RECIPE_FORBIDDEN_KEYS = frozenset(
    {
        "anchor_gate_enabled",
        "anchor_gate_init",
        "hier_compose_residual_side_neutral",
        _RETIRED_HIER_PUBLIC_DIRECTION_COMPOSITION_KEY,
        "hier_public_trade_dir_margin_bridge",
        "hier_public_side_dir_margin_bridge",
    }
)


def _direction_balance_contract_passes(
    contract: dict[str, Any],
    *,
    contract_mode: str,
) -> bool:
    if contract_mode != MODEL_NATIVE_CONTRACT_MODE:
        return False
    if MODEL_NATIVE_DIRECTION_RECIPE_FORBIDDEN_KEYS.intersection(contract):
        return False
    weights = contract.get("pred_balance_class_weights")
    try:
        parsed_weights = [float(value) for value in weights]
    except (TypeError, ValueError):
        return False
    alpha = _float_or_zero(contract.get("pred_balance_alpha"))
    return (
        str(contract.get("decision")) == "PASS"
        and not contract.get("failures")
        and bool(contract.get("direction_active"))
        and 0.05 <= alpha <= 0.50
        and str(contract.get("pred_balance_target") or "").strip().lower() == "label"
        and len(parsed_weights) == 3
        and all(value > 0.0 for value in parsed_weights)
        and _float_or_zero(contract.get("direction_ce_scale")) > 0.0
        and str(contract.get("ckpt_monitor") or "").strip().lower() == "dir_acc"
        and _float_or_zero(contract.get("direction_min_pred_rate_loss_weight")) > 0.0
        and _float_or_zero(contract.get("direction_slice_min_pred_rate_loss_weight")) > 0.0
        and bool(contract.get("best_direction_balance_guard_ok"))
    )
def _contract_mode_from_bundle_audit(report: dict[str, Any]) -> str:
    bundle = report.get("bundle_summary") if isinstance(report.get("bundle_summary"), dict) else {}
    return _normalize_contract_mode(
        report.get("specialist_contract_mode")
        or report.get("contract_mode")
        or bundle.get("specialist_contract_mode")
        or bundle.get("contract_mode")
        or bundle.get("audit_contract_mode")
    )


def _contract_mode_from_identity(identity: dict[str, Any]) -> str:
    return _normalize_contract_mode(identity.get("contract_mode"))


def _model_summaries(summary: dict[str, Any], model_name: str) -> list[dict[str, Any]]:
    return [
        row
        for row in summary.get("summaries", [])
        if isinstance(row, dict) and str(row.get("model")) == str(model_name)
    ]


def _summary_by_split(summary: dict[str, Any], model_name: str) -> dict[str, dict[str, Any]]:
    return {str(row.get("split")): row for row in _model_summaries(summary, model_name)}


def _split_metric(summary: dict[str, Any], model_name: str, key: str) -> dict[str, Any]:
    return {split: row.get(key) for split, row in _summary_by_split(summary, model_name).items()}


def _all_split_metric_gt(summary: dict[str, Any], model_name: str, key: str, threshold: float) -> bool:
    values = _split_metric(summary, model_name, key)
    required = [split for split in ("val", "test") if split in values]
    return len(required) == 2 and all(values[split] is not None and float(values[split]) > float(threshold) for split in required)


def _all_split_metric_ge(summary: dict[str, Any], model_name: str, key: str, threshold: float) -> bool:
    values = _split_metric(summary, model_name, key)
    required = [split for split in ("val", "test") if split in values]
    if len(required) != 2:
        return False
    for split in required:
        value = values.get(split)
        if value is None:
            return False
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return False
        if not np.isfinite(numeric) or numeric < float(threshold):
            return False
    return True


def _selected_tail_direction_slice_report(
    metrics: pd.DataFrame,
    *,
    model_name: str,
    min_direction_precision: float,
    min_rows: int,
) -> dict[str, Any]:
    required_columns = {"split", "model", "scope", "top_frac", "group", "n", "direction_precision"}
    metric_columns = set(str(c) for c in metrics.columns)
    if metrics.empty or not required_columns.issubset(metric_columns):
        return {
            "checked_rows": 0,
            "failed_rows": 0,
            "failures": [],
            "missing_columns": sorted(required_columns - metric_columns),
        }
    rows = metrics[
        (metrics["model"].astype(str) == str(model_name))
        & (metrics["scope"].astype(str) == "top_score")
    ].copy()
    if rows.empty:
        return {"checked_rows": 0, "failed_rows": 0, "failures": [], "missing_columns": []}
    groups = rows["group"].astype(str)
    top_frac = pd.to_numeric(rows["top_frac"], errors="coerce")
    n = pd.to_numeric(rows["n"], errors="coerce")
    wanted_group = (
        (groups == "ALL")
        | groups.str.startswith("session=")
        | groups.str.startswith("side=")
        | groups.str.startswith("vol_regime=")
    )
    wanted_frac = np.isclose(top_frac.astype(float), 0.05) | np.isclose(top_frac.astype(float), 0.10)
    eligible = rows[wanted_group & wanted_frac & (n >= int(min_rows))].copy()
    if eligible.empty:
        return {"checked_rows": 0, "failed_rows": 0, "failures": [], "missing_columns": []}
    eligible_direction = pd.to_numeric(eligible["direction_precision"], errors="coerce")
    failed = eligible[(~np.isfinite(eligible_direction)) | (eligible_direction < float(min_direction_precision))].copy()
    failures = []
    for _, row in failed.head(25).iterrows():
        value = pd.to_numeric(pd.Series([row.get("direction_precision")]), errors="coerce").iloc[0]
        failures.append(
            {
                "split": str(row.get("split")),
                "group": str(row.get("group")),
                "top_frac": float(row.get("top_frac")),
                "n": int(row.get("n")),
                "direction_precision": None if pd.isna(value) else float(value),
            }
        )
    return {
        "checked_rows": int(len(eligible)),
        "failed_rows": int(len(failed)),
        "failures": failures,
        "missing_columns": [],
    }


def _selective_edge_checks(
    summary: dict[str, Any],
    metrics: pd.DataFrame,
    *,
    model_name: str,
    min_top5_mean_pnl_bps: float,
    min_top10_mean_pnl_bps: float,
    min_top_direction_precision: float,
    min_direction_slice_precision: float,
    min_direction_slice_n: int = 20,
    expected_bundle_dir: str | None = None,
    expected_dataset_dir: Path,
    expected_contract_mode: str = MODEL_NATIVE_CONTRACT_MODE,
) -> list[dict[str, Any]]:
    if expected_contract_mode != MODEL_NATIVE_CONTRACT_MODE:
        raise RuntimeError(
            f"retired selective-edge contract mode: {expected_contract_mode!r}"
        )
    for name, value in (
        ("min_top_direction_precision", min_top_direction_precision),
        ("min_direction_slice_precision", min_direction_slice_precision),
    ):
        numeric = float(value)
        if not np.isfinite(numeric) or not 0.50 < numeric <= 1.0:
            raise RuntimeError(
                f"{name} must be an explicit high-precision admission bound in "
                f"(0.50, 1.0], got {value!r}"
            )
    splits = set(str(x) for x in summary.get("splits", []))
    models = {
        str(row.get("model"))
        for row in summary.get("summaries", [])
        if isinstance(row, dict)
    }
    summary_bundle_dir = str(summary.get("bundle_dir") or "")
    contract_mode = _normalize_contract_mode(summary.get("contract_mode"))
    observed_seq_input_dim = int(summary.get("bundle_seq_input_dim") or 0)
    observed_snap_input_dim = int(summary.get("bundle_snap_input_dim") or 0)
    selective_specialist_contract = _selective_edge_specialist_contract(
        summary, MODEL_NATIVE_CONTRACT_MODE
    )
    selection_score_mode = str(summary.get("selection_score_mode") or "")
    selection_score_threshold_present = "selection_score_threshold" in summary
    feature_mask = summary.get("feature_mask_ablation")
    full_stack_unmasked = (
        isinstance(feature_mask, dict)
        and feature_mask.get("enabled") is False
    )
    signal_contract = summary.get("model_native_signal_contract")
    signal_contract_ok = True
    signal_contract_error = ""
    try:
        if not isinstance(signal_contract, dict):
            raise RuntimeError("missing model_native_signal_contract")
        require_model_native_signal_contract(
            signal_contract,
            context="REPLAY_READINESS_SELECTIVE_EDGE",
        )
        require_model_direction_decision_contract(
            {
                "direction_decision_contract": summary.get(
                    "direction_decision_contract"
                )
            },
            context="replay-readiness selective-edge",
        )
    except Exception as exc:
        signal_contract_ok = False
        signal_contract_error = str(exc)

    candidate_by_split = _summary_by_split(summary, model_name)
    top5 = _split_metric(summary, model_name, "top5_all_mean_pnl_bps")
    top10 = _split_metric(summary, model_name, "top10_all_mean_pnl_bps")
    top5_direction = _split_metric(
        summary, model_name, "top5_all_direction_precision"
    )
    top10_direction = _split_metric(
        summary, model_name, "top10_all_direction_precision"
    )
    required_columns = {
        "split",
        "model",
        "scope",
        "top_frac",
        "group",
        "n",
        "mean_pnl_bps",
        "win_rate",
        "direction_precision",
    }
    metric_columns = set(str(c) for c in metrics.columns)
    slice_rows = pd.DataFrame()
    if not metrics.empty and required_columns.issubset(metric_columns):
        slice_rows = metrics[
            (metrics["model"].astype(str) == str(model_name))
            & (metrics["scope"].astype(str) == "top_score")
            & (metrics["group"].astype(str).str.startswith("session="))
        ]
    direction_slice_report = _selected_tail_direction_slice_report(
        metrics,
        model_name=model_name,
        min_direction_precision=min_direction_slice_precision,
        min_rows=min_direction_slice_n,
    )
    return [
        _check(
            "selective-edge report PASS",
            str(summary.get("decision")) == "PASS",
            {"failures": summary.get("failures")},
        ),
        _check(
            "selective-edge report has zero failures",
            not summary.get("failures"),
            {"failures": summary.get("failures")},
        ),
        _check(
            "selective-edge report uses expected dataset",
            _same_resolved_path(summary.get("dataset_dir"), expected_dataset_dir),
            {
                "expected_dataset_dir": str(expected_dataset_dir),
                "report_dataset_dir": summary.get("dataset_dir"),
            },
        ),
        _check(
            "selective-edge report matches candidate bundle audit",
            True
            if expected_bundle_dir is None
            else _same_resolved_path(summary_bundle_dir, Path(expected_bundle_dir)),
            {
                "expected_bundle_dir": expected_bundle_dir,
                "report_bundle_dir": summary_bundle_dir,
            },
        ),
        _check(
            "selective-edge contract is exact model-native seq513",
            contract_mode == MODEL_NATIVE_CONTRACT_MODE
            and observed_seq_input_dim == MODEL_NATIVE_SIGNAL_DIM
            and observed_snap_input_dim == MODEL_NATIVE_SIGNAL_DIM
            and signal_contract_ok,
            {
                "contract_mode": contract_mode,
                "seq_input_dim": observed_seq_input_dim,
                "snap_input_dim": observed_snap_input_dim,
                "signal_contract_error": signal_contract_error,
            },
        ),
        _check(
            "selective-edge uses exact final direction argmax schema",
            selection_score_mode == MODEL_DIRECTION_SELECTION_MODE
            and not selection_score_threshold_present,
            {
                "selection_score_mode": selection_score_mode,
                "retired_selection_score_threshold_present": (
                    selection_score_threshold_present
                ),
            },
        ),
        _check(
            "selective-edge authority uses the complete unmasked feature stack",
            full_stack_unmasked,
            {"feature_mask_ablation": feature_mask},
        ),
        _check(
            "selective-edge preserves specialist contract snapshot",
            bool(selective_specialist_contract.get("ready")),
            {"specialist_contract": selective_specialist_contract},
        ),
        _check(
            "selective-edge report has val/test",
            {"val", "test"}.issubset(splits),
            {"splits": sorted(splits)},
        ),
        _check(
            "selective-edge report includes candidate model only",
            models == {model_name},
            {"models": sorted(models)},
        ),
        _check(
            "candidate selective-edge has val/test summaries",
            {"val", "test"}.issubset(set(candidate_by_split)),
            {"candidate_splits": sorted(candidate_by_split)},
        ),
        _check(
            "candidate top5 mean pnl is positive on val/test",
            _all_split_metric_gt(
                summary,
                model_name,
                "top5_all_mean_pnl_bps",
                min_top5_mean_pnl_bps,
            ),
            {"threshold": min_top5_mean_pnl_bps, "values": top5},
        ),
        _check(
            "candidate top10 mean pnl is positive on val/test",
            _all_split_metric_gt(
                summary,
                model_name,
                "top10_all_mean_pnl_bps",
                min_top10_mean_pnl_bps,
            ),
            {"threshold": min_top10_mean_pnl_bps, "values": top10},
        ),
        _check(
            "candidate top5 direction precision clears threshold",
            _all_split_metric_ge(
                summary,
                model_name,
                "top5_all_direction_precision",
                min_top_direction_precision,
            ),
            {"threshold": min_top_direction_precision, "values": top5_direction},
        ),
        _check(
            "candidate top10 direction precision clears threshold",
            _all_split_metric_ge(
                summary,
                model_name,
                "top10_all_direction_precision",
                min_top_direction_precision,
            ),
            {"threshold": min_top_direction_precision, "values": top10_direction},
        ),
        _check(
            "selective-edge metrics exist and have required columns",
            not metrics.empty and required_columns.issubset(metric_columns),
            {
                "rows": int(len(metrics)),
                "missing_columns": sorted(required_columns - metric_columns),
            },
        ),
        _check(
            "selective-edge metrics include session evidence slices",
            len(slice_rows) > 0,
            {"session_slice_rows": int(len(slice_rows))},
        ),
        _check(
            "selected-tail direction slices clear threshold",
            int(direction_slice_report.get("checked_rows") or 0) > 0
            and int(direction_slice_report.get("failed_rows") or 0) == 0,
            {
                "threshold": min_direction_slice_precision,
                "min_rows": min_direction_slice_n,
                **direction_slice_report,
            },
        ),
    ]

def _selective_prediction_evidence_checks(
    summary: dict[str, Any],
    *,
    expected_report_path: Path,
    expected_bundle_dir: str | None,
    expected_dataset_dir: Path,
    model_name: str,
) -> list[dict[str, Any]]:
    """Re-hash the evaluator's timestamped prediction event before replay."""

    requested_raw = str(summary.get("predictions_path") or "").strip()
    report_raw = str(summary.get("json_path") or "").strip()
    bundle_raw = str(expected_bundle_dir or summary.get("bundle_dir") or "").strip()
    details: dict[str, Any] = {
        "requested_predictions_path": requested_raw,
        "requested_prediction_report_json": report_raw,
        "expected_prediction_report_json": str(expected_report_path),
        "expected_bundle_dir": bundle_raw,
        "expected_dataset_dir": str(expected_dataset_dir),
    }
    ok = False
    if requested_raw and report_raw and bundle_raw:
        try:
            if Path(report_raw).expanduser().resolve() != expected_report_path:
                raise RuntimeError(
                    "selective-edge report json_path does not match the explicit report event"
                )
            authoritative, report, evidence = resolve_and_validate_prediction_evidence(
                Path(requested_raw),
                prediction_report_path=Path(report_raw),
                bundle_dir=Path(bundle_raw),
                dataset_dir=expected_dataset_dir,
                expected_model=model_name,
            )
            summary_evidence = summary.get("prediction_evidence")
            if not isinstance(summary_evidence, dict) or summary_evidence != evidence:
                raise RuntimeError(
                    "selective-edge summary prediction_evidence mismatches timestamped report"
                )
            if str(summary.get("bundle_metadata_sha256") or "").lower() != str(
                evidence.get("bundle_metadata_sha256") or ""
            ).lower():
                raise RuntimeError("selective-edge summary bundle metadata SHA-256 mismatch")
            if str(summary.get("model_state_dict_sha256") or "").lower() != str(
                evidence.get("model_state_dict_sha256") or ""
            ).lower():
                raise RuntimeError("selective-edge summary model state SHA-256 mismatch")
            forbidden_columns = {"anchor_logits", "delta_logits", "anchor_gate"}.intersection(
                evidence.get("columns") or []
            )
            if forbidden_columns:
                raise RuntimeError(
                    f"prediction evidence contains forbidden legacy columns: "
                    f"{sorted(forbidden_columns)}"
                )
            details.update(
                {
                    "authoritative_predictions_path": str(authoritative),
                    "prediction_report_json": str(report.get("json_path") or ""),
                    "prediction_sha256": evidence.get("sha256"),
                    "prediction_rows": evidence.get("rows"),
                    "prediction_splits": evidence.get("splits"),
                    "prediction_models": evidence.get("models"),
                }
            )
            ok = True
        except Exception as exc:
            details["error"] = str(exc)
    return [
        _check(
            "selective-edge authoritative predictions rehash and match model contract",
            ok,
            details,
        )
    ]


def _selective_metrics_authority_checks(
    summary: dict[str, Any], explicit_metrics_path: Path
) -> list[dict[str, Any]]:
    declared_raw = str(summary.get("metrics_path") or "").strip()
    declared_sha = str(summary.get("metrics_sha256") or "").strip().lower()
    observed_sha = _sha256_file(explicit_metrics_path)
    path_matches = False
    if declared_raw:
        path_matches = (
            Path(declared_raw).expanduser().resolve() == explicit_metrics_path
        )
    return [
        _check(
            "selective-edge metrics path matches timestamped report",
            path_matches
            and bool(_TIMESTAMPED_CSV_RE.fullmatch(explicit_metrics_path.name)),
            {
                "declared_metrics_path": declared_raw,
                "explicit_metrics_path": str(explicit_metrics_path),
            },
        ),
        _check(
            "selective-edge metrics SHA-256 matches timestamped report",
            len(declared_sha) == 64 and declared_sha == observed_sha,
            {
                "declared_sha256": declared_sha,
                "observed_sha256": observed_sha,
            },
        ),
    ]


def _candidate_bundle_audit_checks(
    path: Path,
    report: dict[str, Any],
    *,
    contract_mode: str = MODEL_NATIVE_CONTRACT_MODE,
    expected_dataset_dir: Path,
) -> list[dict[str, Any]]:
    exists = path.exists()
    bundle = report.get("bundle_summary") if isinstance(report.get("bundle_summary"), dict) else {}
    head_contract = report.get("head_contract") if isinstance(report.get("head_contract"), dict) else {}
    path_calibration = (
        report.get("path_calibration_recipe_contract")
        if isinstance(report.get("path_calibration_recipe_contract"), dict)
        else {}
    )
    direction_balance = (
        report.get("direction_balance_recipe_contract")
        if isinstance(report.get("direction_balance_recipe_contract"), dict)
        else {}
    )
    tail_direction = (
        report.get("tail_direction_recipe_contract")
        if isinstance(report.get("tail_direction_recipe_contract"), dict)
        else {}
    )
    pretrain_manifest = (
        report.get("pretrain_manifest_contract")
        if isinstance(report.get("pretrain_manifest_contract"), dict)
        else {}
    )
    active_heads = set(str(x) for x in head_contract.get("active_training_heads", []) if str(x))
    blocked_heads = set(str(x) for x in head_contract.get("blocked_heads", []) if str(x))
    splits = {
        str(split): row
        for split, row in (report.get("splits") or {}).items()
        if isinstance(row, dict)
    }
    split_rows = {split: int((row or {}).get("rows") or 0) for split, row in splits.items()}
    split_names = tuple(split for split in ("val", "test") if split in splits)
    specialist_groups = set(str(x) for x in bundle.get("specialist_groups", []) if str(x))
    required_specialists = {str(x) for x in report.get("required_training_specialists", []) if str(x)}
    expected_contract_mode = _normalize_contract_mode(contract_mode)
    observed_contract_mode = _contract_mode_from_bundle_audit(report)
    expected_input_dim = CONTRACT_INPUT_DIMS.get(expected_contract_mode)
    expected_specialist_groups = MODEL_NATIVE_REQUIRED_SPECIALISTS
    required_gate_live = True
    for row in splits.values():
        gate = (row or {}).get("specialist_gate") if isinstance(row, dict) else {}
        mean_weight = gate.get("mean_weight") if isinstance(gate, dict) and isinstance(gate.get("mean_weight"), dict) else {}
        for group in expected_specialist_groups:
            if float(mean_weight.get(group) or 0.0) <= 0.01:
                required_gate_live = False
    direction_by_split = {
        split: (splits.get(split) or {}).get("direction") or {}
        for split in split_names
    }
    distribution_by_split = {
        split: (splits.get(split) or {}).get("direction_distribution_contract") or {}
        for split in split_names
    }
    slice_by_split = {
        split: (splits.get(split) or {}).get("direction_slice_contract") or {}
        for split in split_names
    }
    path_quality_rho = {
        split: ((splits.get(split) or {}).get("path_quality") or {}).get("pred_vs_target_spearman")
        for split in split_names
    }
    bad_path_rho = {
        split: ((splits.get(split) or {}).get("bad_path") or {}).get("prob_vs_path_quality_spearman")
        for split in split_names
    }
    expected_active_heads = _expected_active_heads_for_mode(expected_contract_mode)
    readiness_contract_error = ""
    try:
        require_model_native_readiness_contract(
            report.get("model_native_readiness_contract"),
            context="REPLAY_CANDIDATE_BUNDLE_AUDIT",
        )
    except RuntimeError as exc:
        readiness_contract_error = str(exc)
    bundle_contract_ok = True
    bundle_contract_error = ""
    metadata_path = Path(str(report.get("bundle_dir") or "/")).expanduser().resolve() / "bundle_metadata.json"
    try:
        metadata = _read_json(metadata_path)
        signal_contract = metadata.get("model_native_signal_contract")
        if not isinstance(signal_contract, dict):
            raise RuntimeError("bundle metadata lacks model_native_signal_contract")
        require_model_native_signal_contract(
            signal_contract,
            context="REPLAY_READINESS_BUNDLE",
        )
        require_model_direction_decision_contract(
            metadata,
            context="replay-readiness bundle",
        )
        if int(metadata.get("seq_input_dim") or -1) != MODEL_NATIVE_SIGNAL_DIM:
            raise RuntimeError("bundle seq_input_dim is not 513")
        if int(metadata.get("snap_input_dim") or -1) != MODEL_NATIVE_SIGNAL_DIM:
            raise RuntimeError("bundle snap_input_dim is not 513")
    except Exception as exc:
        bundle_contract_ok = False
        bundle_contract_error = str(exc)
    return [
        _check("candidate bundle audit exists", exists, {"path": str(path)}),
        _check("candidate bundle audit PASS", exists and str(report.get("decision")) == "PASS", {"failures": report.get("failures")}),
        _check("candidate bundle audit has zero failures", exists and not report.get("failures"), {"failures": report.get("failures")}),
        _check("candidate bundle audit was run with require_edge", exists and bool(report.get("require_edge"))),
        _check(
            "candidate bundle audit used expected dataset",
            exists and _same_resolved_path(report.get("dataset_dir"), expected_dataset_dir),
            {"expected_dataset_dir": str(expected_dataset_dir), "observed_dataset_dir": report.get("dataset_dir")},
        ),
        _check("candidate bundle audit is from actual train output, not sanity bundle", exists and not bool(bundle.get("sanity_bundle"))),
        _check(
            "candidate bundle audit contract mode matches requested replay contract",
            exists and observed_contract_mode == expected_contract_mode,
            {"expected_contract_mode": expected_contract_mode, "observed_contract_mode": observed_contract_mode},
        ),
        _check(
            "candidate bundle audit carries exact model-native readiness contract",
            not readiness_contract_error,
            {"error": readiness_contract_error},
        ),
        _check(
            "candidate bundle input dimensions match contract mode",
            exists
            and bool(expected_input_dim)
            and int(bundle.get("seq_input_dim") or 0) == expected_input_dim
            and int(bundle.get("snap_input_dim") or 0) == expected_input_dim,
            {
                "expected_contract_mode": expected_contract_mode,
                "expected_input_dim": expected_input_dim,
                "seq_input_dim": bundle.get("seq_input_dim"),
                "snap_input_dim": bundle.get("snap_input_dim"),
            },
        ),
        _check(
            "candidate bundle metadata proves exact model-native direction contract",
            bundle_contract_ok,
            {
                "bundle_metadata_path": str(metadata_path),
                "error": bundle_contract_error,
            },
        ),
        _check("candidate bundle has multi-TF enabled", exists and bool(bundle.get("multi_tf_enabled"))),
        _check("candidate bundle has specialist fusion", exists and bool(bundle.get("specialist_fusion_enabled"))),
        _check(
            "candidate bundle specialist model contract is preserved in bundle metadata",
            exists and _bundle_specialist_model_contract_passes(report),
            {
                "bundle_summary": {
                    "specialist_model_contract_declared_valid": bundle.get("specialist_model_contract_declared_valid"),
                    "specialist_model_contract_valid": bundle.get("specialist_model_contract_valid"),
                    "specialist_model_contract_set_exact": bundle.get("specialist_model_contract_set_exact"),
                    "specialist_model_contract_owned_objectives_match": bundle.get("specialist_model_contract_owned_objectives_match"),
                    "specialist_model_contract_support_heads_match": bundle.get("specialist_model_contract_support_heads_match"),
                    "specialist_model_contract_signal_families_match": bundle.get("specialist_model_contract_signal_families_match"),
                    "specialist_model_contract_model_roles_match": bundle.get("specialist_model_contract_model_roles_match"),
                },
                "bundle_specialist_model_contract": report.get("bundle_specialist_model_contract"),
            },
        ),
        _check(
            "candidate bundle includes required specialist groups",
            exists and set(expected_specialist_groups).issubset(specialist_groups),
            {"specialist_groups": sorted(specialist_groups)},
        ),
        _check(
            "candidate bundle has exact specialist groups",
            exists and specialist_groups == set(expected_specialist_groups),
            {
                "expected_specialist_groups": list(expected_specialist_groups),
                "actual_specialist_groups": sorted(specialist_groups),
            },
        ),
        _check(
            "candidate bundle audit was run with specialist-fusion gate contract",
            exists
            and bool(report.get("require_specialist_fusion"))
            and required_specialists == set(expected_specialist_groups)
            and int(report.get("min_active_specialists") or 0) >= len(expected_specialist_groups)
            and float(report.get("min_gate_entropy") or -1.0) >= float(REQUIRED_MIN_GATE_ENTROPY),
            {
                "required_training_specialists": report.get("required_training_specialists"),
                "min_active_specialists": report.get("min_active_specialists"),
                "min_gate_entropy": report.get("min_gate_entropy"),
            },
        ),
        _check(
            "candidate bundle required specialist gate weights are non-collapsed",
            exists and required_gate_live,
            {"min_mean_weight": 0.01, "required_specialists": list(expected_specialist_groups)},
        ),
        _check("candidate bundle audit was run with require_head_contract", exists and bool(report.get("require_head_contract"))),
        _check(
            "candidate bundle head contract PASS",
            exists
            and str(head_contract.get("decision")) == "PASS"
            and not head_contract.get("failures")
            and active_heads == expected_active_heads
            and blocked_heads == set(MODEL_NATIVE_BLOCKED_HEADS),
            {
                "head_contract": head_contract,
                "expected_active_heads": sorted(expected_active_heads),
                "actual_active_heads": sorted(active_heads),
                "expected_blocked_heads": list(MODEL_NATIVE_BLOCKED_HEADS),
                "actual_blocked_heads": sorted(blocked_heads),
            },
        ),
        _check(
            "candidate bundle direction balance recipe contract PASS",
            exists
            and _direction_balance_contract_passes(
                direction_balance,
                contract_mode=expected_contract_mode,
            ),
            {"direction_balance_recipe_contract": direction_balance},
        ),
        _check(
            "candidate bundle path calibration recipe contract PASS",
            exists
            and str(path_calibration.get("decision")) == "PASS"
            and not path_calibration.get("failures")
            and bool(path_calibration.get("path_quality_active"))
            and bool(path_calibration.get("path_quality_rank_full_batch"))
            and float(path_calibration.get("path_quality_rank_weight") or 0.0) > 0.0,
            {"path_calibration_recipe_contract": path_calibration},
        ),
        _check(
            "candidate bundle tail direction recipe contract PASS",
            exists
            and str(tail_direction.get("decision")) == "PASS"
            and not tail_direction.get("failures")
            and bool(tail_direction.get("direction_active"))
            and float(tail_direction.get("tail_direction_ce_weight") or 0.0) > 0.0
            and 0.50 <= float(tail_direction.get("tail_direction_quality_quantile") or 0.0) <= 0.95
            and int(tail_direction.get("tail_direction_min_batch") or 0) >= 2
            and str(tail_direction.get("tail_direction_mask") or "") == "directional_tradable_clean_path_top_quality",
            {"tail_direction_recipe_contract": tail_direction},
        ),
        _check(
            "candidate bundle direction beats majority on val/test",
            exists
            and bool(split_names)
            and all(bool((direction_by_split.get(split) or {}).get("beats_majority_baseline")) for split in split_names),
            {"direction": direction_by_split},
        ),
        _check(
            "candidate bundle direction distribution covers active classes",
            exists
            and bool(split_names)
            and all(str((distribution_by_split.get(split) or {}).get("decision")) == "PASS" for split in split_names),
            {"direction_distribution": distribution_by_split},
        ),
        _check(
            "candidate bundle direction context slices pass strictly",
            exists
            and bool(split_names)
            and str(report.get("edge_test_scope")) == "strict"
            and all(
                str((slice_by_split.get(split) or {}).get("decision")) == "PASS"
                and int((slice_by_split.get(split) or {}).get("audited_slice_count") or 0) > 0
                for split in split_names
            ),
            {"direction_slices": slice_by_split, "audit_edge_test_scope": report.get("edge_test_scope")},
        ),
        _check(
            "candidate bundle path_quality ranks realized path quality positively",
            exists
            and bool(split_names)
            and all(value is not None and float(value) > 0.0 for value in path_quality_rho.values()),
            {"path_quality_pred_vs_target_spearman": path_quality_rho},
        ),
        _check(
            "candidate bundle bad_path ranks worse path quality higher",
            exists
            and bool(split_names)
            and all(value is not None and float(value) < 0.0 for value in bad_path_rho.values()),
            {"bad_path_prob_vs_path_quality_spearman": bad_path_rho},
        ),
        _check(
            "candidate bundle audit validated pre-train manifest provenance",
            exists
            and str(pretrain_manifest.get("decision")) == "PASS"
            and not pretrain_manifest.get("failures")
            and bool(pretrain_manifest.get("feature_objective_coverage_all_present"))
            and bool(pretrain_manifest.get("feature_objective_liveness_all_live"))
            and bool(pretrain_manifest.get("feature_source_field_liveness_all_live"))
            and bool(pretrain_manifest.get("specialist_objective_routing_all_present_and_expected"))
            and bool(pretrain_manifest.get("specialist_input_liveness_all_live"))
            and bool(pretrain_manifest.get("specialist_active_heads_match_target"))
            and bool(pretrain_manifest.get("specialist_blocked_heads_match_target"))
            and bool(pretrain_manifest.get("specialist_required_training_set_exact"))
            and bool(pretrain_manifest.get("specialist_trainable_set_exact"))
            and bool(pretrain_manifest.get("specialist_model_contract_valid"))
            and bool(pretrain_manifest.get("specialist_model_contract_set_exact"))
            and bool(pretrain_manifest.get("specialist_model_contract_owned_objectives_match"))
            and bool(pretrain_manifest.get("smoke_edge_required_specialists_exact"))
            and bool(pretrain_manifest.get("smoke_edge_specialist_groups_exact"))
            and bool(pretrain_manifest.get("smoke_edge_specialist_model_contract_valid"))
            and bool(pretrain_manifest.get("smoke_edge_specialist_model_contract_set_exact"))
            and bool(pretrain_manifest.get("smoke_edge_specialist_model_contract_owned_objectives_match"))
            and bool(pretrain_manifest.get("smoke_dataset_audit_provenance_all_artifacts_present"))
            and bool(pretrain_manifest.get("smoke_dataset_audit_provenance_all_artifact_hashes_present"))
            and bool(pretrain_manifest.get("smoke_edge_worktree_critical_gate_review_ok")),
            {"pretrain_manifest_contract": pretrain_manifest},
        ),
        _check(
            "candidate bundle audit covered val/test rows",
            exists and split_rows.get("val", 0) > 0 and split_rows.get("test", 0) > 0,
            {"split_rows": split_rows},
        ),
    ]


def _best_replay_row(metrics: pd.DataFrame) -> dict[str, Any] | None:
    if metrics.empty:
        return None
    if "scope" in metrics.columns:
        aggregate = metrics[metrics["scope"].astype(str).isin(["aggregate", "ALL", "overall"])]
        if not aggregate.empty:
            metrics = aggregate
    if "net_sum_bps" in metrics.columns:
        metrics = metrics.sort_values("net_sum_bps", ascending=False)
    return metrics.iloc[0].to_dict()


def _replay_checks(
    replay_dir: Path,
    manifest: dict[str, Any],
    metrics: pd.DataFrame,
    monthly: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    min_net_sum_bps: float,
    min_profit_factor: float,
    max_drawdown_bps: float,
    expected_candidate_bundle_dir: str | None = None,
    expected_contract_mode: str = MODEL_NATIVE_CONTRACT_MODE,
    expected_replay_report_path: Path | None = None,
) -> list[dict[str, Any]]:
    row = _best_replay_row(metrics)
    required_metric_columns = {
        "policy_id",
        "n_trades",
        "net_sum_bps",
        "win_rate",
        "profit_factor",
        "max_drawdown_bps",
        "max_loss_bps",
    }
    metric_columns = set(str(c) for c in metrics.columns)
    monthly_columns = set(str(c) for c in monthly.columns)
    trade_columns = set(str(c) for c in trades.columns)
    model_native_missing_columns = sorted(
        set(MODEL_NATIVE_REPLAY_REQUIRED_COLUMNS) - trade_columns
    )
    model_native_trade_audit = (
        audit_model_native_replay_trades(trades)
        if not trades.empty
        else {
        "ready": False,
        "failures": ["replay trade evidence is empty"],
        }
    )
    identity = manifest.get("replay_identity_contract") if isinstance(manifest.get("replay_identity_contract"), dict) else {}
    identity_contract_mode = _contract_mode_from_identity(identity)
    candidate_specialist_identity = (
        identity.get("candidate_specialist_contract")
        if isinstance(identity.get("candidate_specialist_contract"), dict)
        else {}
    )
    selective_specialist_identity = (
        identity.get("selective_edge_specialist_contract")
        if isinstance(identity.get("selective_edge_specialist_contract"), dict)
        else {}
    )
    row_details = row or {}
    drawdown = row_details.get("max_drawdown_bps")
    drawdown_ok = drawdown is not None and abs(float(drawdown)) <= float(max_drawdown_bps)
    trade_log_authority = (
        manifest.get("trade_log_authority_contract")
        if isinstance(manifest.get("trade_log_authority_contract"), dict)
        else {}
    )
    diagnostic_contract = {
        "offline_only": True,
        "diagnostic_scope": OFFLINE_DIRECTION_DIAGNOSTIC_SCOPE,
        "pnl_normalization": UNIT_NORMALIZED_PNL_MODE,
        "execution_order_simulation": False,
        "position_size_applied": False,
    }
    diagnostic_mismatches = {
        key: {"observed": manifest.get(key), "expected": expected}
        for key, expected in diagnostic_contract.items()
        if manifest.get(key) != expected
    }
    retired_sizing_keys = {
        "dynamic_sizing_applied",
        "applied_size_multiplier",
        "replay_size_multiplier",
        "sizing_authority_contract",
    }
    retired_sizing_present = sorted(
        retired_sizing_keys.intersection(manifest)
        | retired_sizing_keys.intersection(trade_log_authority)
    )
    return [
        _check("offline replay dir exists", replay_dir.exists(), {"replay_dir": str(replay_dir)}),
        _check(
            "offline replay report json_path matches explicit immutable event",
            True
            if expected_replay_report_path is None
            else _same_resolved_path(
                manifest.get("json_path"), expected_replay_report_path
            ),
            {
                "declared_json_path": manifest.get("json_path"),
                "expected_json_path": (
                    str(expected_replay_report_path)
                    if expected_replay_report_path is not None
                    else None
                ),
            },
        ),
        _check("offline replay manifest PASS", str(manifest.get("decision")) == "PASS", {"manifest_failures": manifest.get("failures")}),
        _check("offline replay manifest has zero failures", not manifest.get("failures"), {"manifest_failures": manifest.get("failures")}),
        _check(
            "offline replay is exact unit-normalized direction diagnostic",
            not diagnostic_mismatches,
            {
                "required_contract": diagnostic_contract,
                "mismatches": diagnostic_mismatches,
            },
        ),
        _check(
            "offline replay exposes no execution sizing authority",
            not retired_sizing_present,
            {"retired_sizing_fields": retired_sizing_present},
        ),
        _check(
            "offline replay binds exact trade-log authority",
            bool(trade_log_authority.get("ready"))
            and not trade_log_authority.get("failures"),
            {"trade_log_authority_contract": trade_log_authority},
        ),
        _check("offline replay identity contract ready", bool(identity.get("ready")), {"identity": identity}),
        _check(
            "offline replay identity preserves candidate specialist contract",
            bool(candidate_specialist_identity.get("ready")),
            {"candidate_specialist_contract": candidate_specialist_identity},
        ),
        _check(
            "offline replay identity preserves selective-edge specialist contract",
            bool(selective_specialist_identity.get("ready")),
            {"selective_edge_specialist_contract": selective_specialist_identity},
        ),
        _check(
            "offline replay identity contract mode matches replay-readiness contract",
            identity_contract_mode == expected_contract_mode,
            {"expected_contract_mode": expected_contract_mode, "identity_contract_mode": identity_contract_mode},
        ),
        _check(
            "offline replay identity matches candidate bundle audit",
            True if expected_candidate_bundle_dir is None else str(identity.get("candidate_bundle_dir") or "") == str(expected_candidate_bundle_dir),
            {"expected_candidate_bundle_dir": expected_candidate_bundle_dir, "identity_candidate_bundle_dir": identity.get("candidate_bundle_dir")},
        ),
        _check("offline replay metrics exist and have rows", not metrics.empty, {"rows": int(len(metrics))}),
        _check(
            "offline replay metrics have required columns",
            required_metric_columns.issubset(metric_columns),
            {"missing_columns": sorted(required_metric_columns - metric_columns)},
        ),
        _check("offline replay monthly file has rows", not monthly.empty, {"rows": int(len(monthly))}),
        _check("offline replay monthly file has net_sum_bps", "net_sum_bps" in monthly_columns),
        _check("offline replay trades file has rows", not trades.empty, {"rows": int(len(trades))}),
        _check(
            "offline replay trades have exact model-native evidence columns",
            not model_native_missing_columns,
            {
                "missing_columns": model_native_missing_columns,
                "required_columns": list(MODEL_NATIVE_REPLAY_REQUIRED_COLUMNS),
            },
        ),
        _check(
            "offline replay trade sides equal model LONG/SHORT/FLAT argmax",
            bool(model_native_trade_audit.get("ready")),
            {"model_native_trade_audit": model_native_trade_audit},
        ),
        _check(
            "offline replay best row has enough trades",
            row is not None and int(row_details.get("n_trades") or 0) > 0,
            {"best_row": row_details},
        ),
        _check(
            "offline replay net sum is positive",
            row is not None and float(row_details.get("net_sum_bps") or 0.0) > float(min_net_sum_bps),
            {"threshold": min_net_sum_bps, "best_row": row_details},
        ),
        _check(
            "offline replay profit factor passes",
            row is not None and float(row_details.get("profit_factor") or 0.0) >= float(min_profit_factor),
            {"threshold": min_profit_factor, "best_row": row_details},
        ),
        _check(
            "offline replay drawdown is within bound",
            drawdown_ok,
            {"max_abs_drawdown_bps": max_drawdown_bps, "best_row": row_details},
        ),
        _check(
            "offline replay has no negative months",
            not monthly.empty and "net_sum_bps" in monthly.columns and bool((monthly["net_sum_bps"].astype(float) > 0.0).all()),
            {"monthly_rows": int(len(monthly))},
        ),
    ]


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Entry Replay Readiness",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Model-native replay evidence ready: `{report['model_native_replay_evidence_ready']}`",
        f"- Secondary direction authority allowed: `{report['secondary_direction_authority_allowed']}`",
        f"- Promotion/shadow/live allowed: `{report['promotion_shadow_live_allowed']}`",
        f"- Next required gate: `{report['next_required_gate']}`",
        "",
        "## Gates",
        "",
    ]
    for gate in report["gates"]:
        lines.append(f"- `{gate['name']}`: {gate['decision']} ({gate['passed']}/{gate['total']} checks)")
    lines.extend(["", "## Failures", ""])
    if report["failures"]:
        for failure in report["failures"]:
            lines.append(f"- `{failure['gate']}`: {failure['check']}")
    else:
        lines.append("- None")
    atomic_write_text(path, "\n".join(lines) + "\n")


def _run(args: argparse.Namespace) -> dict[str, Any]:
    contract_mode = MODEL_NATIVE_CONTRACT_MODE
    candidate_readiness_path = Path(
        args.candidate_readiness_json
    ).expanduser().resolve()
    candidate_bundle_audit_path = Path(
        args.candidate_bundle_audit_json
    ).expanduser().resolve()
    selective_report_path = Path(
        args.selective_edge_report_json
    ).expanduser().resolve()
    selective_metrics_path = Path(
        args.selective_edge_metrics_csv
    ).expanduser().resolve()
    replay_report_path = Path(args.replay_evidence_json).expanduser().resolve()
    pretrain_audit_path = Path(args.pretrain_audit_json).expanduser().resolve()
    expected_dataset_dir = Path(args.expected_dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    for label, path in (
        ("candidate readiness", candidate_readiness_path),
        ("candidate bundle audit", candidate_bundle_audit_path),
        ("selective-edge report", selective_report_path),
        ("replay evidence", replay_report_path),
        ("pretrain audit", pretrain_audit_path),
    ):
        if not _is_explicit_timestamped_json(path):
            raise RuntimeError(
                f"{label} must be an explicitly timestamped immutable artifact: {path}"
            )
    if not _TIMESTAMPED_CSV_RE.fullmatch(selective_metrics_path.name):
        raise RuntimeError(
            "selective-edge metrics must be an explicitly timestamped immutable "
            f"artifact: {selective_metrics_path}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_readiness = _read_json(candidate_readiness_path)
    candidate_bundle_audit = _read_json(candidate_bundle_audit_path)
    selective_report = _read_json(selective_report_path)
    selective_metrics = _read_csv_or_empty(selective_metrics_path)
    replay_report = _read_json(replay_report_path)
    pretrain_audit = _read_json(pretrain_audit_path)

    replay_metrics_path = Path(str(replay_report.get("metrics_csv") or "/")).expanduser().resolve()
    replay_monthly_path = Path(str(replay_report.get("monthly_csv") or "/")).expanduser().resolve()
    replay_trades_path = Path(str(replay_report.get("trades_csv") or "/")).expanduser().resolve()
    for label, path in (
        ("replay metrics", replay_metrics_path),
        ("replay monthly", replay_monthly_path),
        ("replay trades", replay_trades_path),
    ):
        if not _TIMESTAMPED_CSV_RE.fullmatch(path.name):
            raise RuntimeError(
                f"{label} must be an explicitly timestamped immutable artifact: {path}"
            )
    replay_metrics = _read_csv_or_empty(replay_metrics_path)
    replay_monthly = _read_csv_or_empty(replay_monthly_path)
    replay_trades = _read_csv_or_empty(replay_trades_path)
    replay_dir = Path(str(replay_report.get("out_dir") or replay_report_path.parent)).expanduser().resolve()

    expected_candidate_bundle_dir = str(
        candidate_bundle_audit.get("bundle_dir") or ""
    ) or None
    replay_identity = (
        replay_report.get("replay_identity_contract")
        if isinstance(replay_report.get("replay_identity_contract"), dict)
        else {}
    )
    candidate_specialist_identity = (
        replay_identity.get("candidate_specialist_contract")
        if isinstance(replay_identity.get("candidate_specialist_contract"), dict)
        else {}
    )
    selective_edge_specialist_identity = (
        replay_identity.get("selective_edge_specialist_contract")
        if isinstance(replay_identity.get("selective_edge_specialist_contract"), dict)
        else {}
    )
    evidence_identity = {
        "candidate_bundle_audit_json": str(candidate_bundle_audit_path),
        "selective_edge_report_json": str(selective_report_path),
        "replay_evidence_json": str(replay_report_path),
        "pretrain_audit_json": str(pretrain_audit_path),
        "candidate_bundle_dir": str(candidate_bundle_audit.get("bundle_dir") or ""),
        "selective_edge_bundle_dir": str(selective_report.get("bundle_dir") or ""),
        "replay_identity_candidate_bundle_dir": str(
            replay_identity.get("candidate_bundle_dir") or ""
        ),
        "replay_identity_ready": bool(replay_identity.get("ready")),
        "contract_mode": contract_mode,
        "model_native_readiness_contract": model_native_readiness_contract_metadata(),
        "candidate_bundle_contract_mode": _contract_mode_from_bundle_audit(
            candidate_bundle_audit
        ),
        "selective_edge_contract_mode": _normalize_contract_mode(
            selective_report.get("contract_mode")
        ),
        "replay_identity_contract_mode": _contract_mode_from_identity(
            replay_identity
        ),
        "candidate_specialist_contract": candidate_specialist_identity,
        "selective_edge_specialist_contract": selective_edge_specialist_identity,
        "candidate_specialist_contract_ready": bool(
            candidate_specialist_identity.get("ready")
        ),
        "selective_edge_specialist_contract_ready": bool(
            selective_edge_specialist_identity.get("ready")
        ),
    }
    artifacts = {
        "candidate_readiness": str(candidate_readiness_path),
        "candidate_bundle_audit": str(candidate_bundle_audit_path),
        "selective_edge_report": str(selective_report_path),
        "selective_edge_metrics": str(selective_metrics_path),
        "candidate_replay_report": str(replay_report_path),
        "candidate_replay_metrics": str(replay_metrics_path),
        "candidate_replay_monthly": str(replay_monthly_path),
        "candidate_replay_trades": str(replay_trades_path),
        "pretrain_audit": str(pretrain_audit_path),
    }
    prediction_evidence = (
        selective_report.get("prediction_evidence")
        if isinstance(selective_report.get("prediction_evidence"), dict)
        else {}
    )
    prediction_path = str(prediction_evidence.get("path") or "").strip()
    if prediction_path:
        artifacts["selective_edge_authoritative_predictions"] = prediction_path
    artifact_fingerprints = _artifact_fingerprints(artifacts)

    declared_hashes = (
        replay_report.get("artifact_hashes")
        if isinstance(replay_report.get("artifact_hashes"), dict)
        else {}
    )
    replay_hash_checks = []
    for path in (replay_metrics_path, replay_monthly_path, replay_trades_path):
        expected_sha = str(declared_hashes.get(path.name) or "").lower()
        observed_sha = _sha256_file(path)
        replay_hash_checks.append(
            _check(
                f"replay artifact {path.name} rehashes",
                len(expected_sha) == 64 and expected_sha == observed_sha,
                {
                    "path": str(path),
                    "expected_sha256": expected_sha,
                    "observed_sha256": observed_sha,
                },
            )
        )

    try:
        bundle_identity = adaptation_bundle_identity_from_dir(
            Path(expected_candidate_bundle_dir),
            context="ENTRY_REPLAY_READINESS_BUNDLE",
        )
        bundle_identity_error = ""
    except ModelNativeAdaptationDriftError as exc:
        bundle_identity = None
        bundle_identity_error = str(exc)

    gate_checks = {
        "candidate_readiness": [
            _check(
                "candidate-readiness is green for exact seq513",
                str(candidate_readiness.get("decision"))
                == "READY_FOR_CANDIDATE_TRAINING"
                and candidate_readiness.get("contract_mode")
                == MODEL_NATIVE_CONTRACT_MODE
                and int(candidate_readiness.get("expected_signal_dim") or 0)
                == MODEL_NATIVE_SIGNAL_DIM
                and not candidate_readiness.get("failures"),
                {
                    "decision": candidate_readiness.get("decision"),
                    "contract_mode": candidate_readiness.get("contract_mode"),
                    "expected_signal_dim": candidate_readiness.get(
                        "expected_signal_dim"
                    ),
                    "failures": candidate_readiness.get("failures"),
                },
            ),
            _check(
                "candidate-readiness still blocks promotion/shadow/live",
                bool(
                    candidate_readiness.get("promotion_shadow_live_allowed")
                )
                is False,
            ),
            _check(
                "candidate-readiness carries exact model-native readiness contract",
                _readiness_contract_is_exact(candidate_readiness),
                {
                    "model_native_readiness_contract": candidate_readiness.get(
                        "model_native_readiness_contract"
                    )
                },
            ),
        ],
        "candidate_bundle_audit": _candidate_bundle_audit_checks(
            candidate_bundle_audit_path,
            candidate_bundle_audit,
            contract_mode=contract_mode,
            expected_dataset_dir=expected_dataset_dir,
        ),
        "selective_edge": [
            *_selective_edge_checks(
                selective_report,
                selective_metrics,
                model_name=READINESS_MODEL_NAME,
                min_top5_mean_pnl_bps=MIN_TOP5_MEAN_PNL_BPS,
                min_top10_mean_pnl_bps=MIN_TOP10_MEAN_PNL_BPS,
                min_top_direction_precision=MIN_TOP_DIRECTION_PRECISION,
                min_direction_slice_precision=MIN_DIRECTION_SLICE_PRECISION,
                min_direction_slice_n=MIN_DIRECTION_SLICE_N,
                expected_bundle_dir=expected_candidate_bundle_dir,
                expected_dataset_dir=expected_dataset_dir,
                expected_contract_mode=contract_mode,
            ),
            *_selective_metrics_authority_checks(
                selective_report, selective_metrics_path
            ),
        ],
        "selective_prediction_evidence": _selective_prediction_evidence_checks(
            selective_report,
            expected_report_path=selective_report_path,
            expected_bundle_dir=expected_candidate_bundle_dir,
            expected_dataset_dir=expected_dataset_dir,
            model_name=READINESS_MODEL_NAME,
        ),
        "offline_replay": [
            *_replay_checks(
                replay_dir,
                replay_report,
                replay_metrics,
                replay_monthly,
                replay_trades,
                min_net_sum_bps=MIN_REPLAY_NET_SUM_BPS,
                min_profit_factor=MIN_REPLAY_PROFIT_FACTOR,
                max_drawdown_bps=MAX_REPLAY_ABS_DRAWDOWN_BPS,
                expected_candidate_bundle_dir=expected_candidate_bundle_dir,
                expected_contract_mode=contract_mode,
                expected_replay_report_path=replay_report_path,
            ),
            *replay_hash_checks,
        ],
        "pretrain_audit": _model_native_pretrain_audit_checks(
            pretrain_audit_path,
            pretrain_audit,
            expected_dataset_dir=expected_dataset_dir,
        ),
        "execution_guard": [
            _check("gate never trains another direction model", True),
            _check("gate never promotes", True),
            _check("gate never starts shadow/live", True),
        ],
        "artifact_provenance": [
            *_artifact_fingerprint_checks(artifact_fingerprints),
            _check(
                "replay-readiness artifact inventory is exact",
                set(artifacts)
                == set(MODEL_NATIVE_REPLAY_READINESS_REQUIRED_ARTIFACTS),
                {
                    "observed": sorted(artifacts),
                    "required": sorted(
                        MODEL_NATIVE_REPLAY_READINESS_REQUIRED_ARTIFACTS
                    ),
                },
            ),
            _check(
                "candidate bundle has exact current byte identity",
                bundle_identity is not None,
                {"error": bundle_identity_error},
            ),
        ],
    }
    gates: list[dict[str, Any]] = []
    for name, checks in gate_checks.items():
        passed = sum(1 for check in checks if check["ok"])
        gates.append(
            {
                "name": name,
                "decision": "PASS" if _all_ok(checks) else "FAIL",
                "passed": int(passed),
                "total": int(len(checks)),
                "checks": checks,
            }
        )
    failures = [
        {
            "gate": gate["name"],
            "check": check["name"],
            "details": check.get("details") or {},
        }
        for gate in gates
        for check in gate["checks"]
        if not check["ok"]
    ]
    ready = not failures
    event_created_utc = next_immutable_event_created_utc(
        out_dir,
        "ENTRY_REPLAY_READINESS",
    )
    timestamp = event_created_utc.strftime("%Y%m%dT%H%M%S%fZ")
    report = {
        "schema_version": "entry_replay_readiness_model_native_v2",
        "created_utc": event_created_utc.isoformat(),
        "contract_mode": contract_mode,
        "model_native_readiness_contract": model_native_readiness_contract_metadata(),
        "decision": (
            "READY_FOR_MODEL_NATIVE_REPLAY_REVIEW"
            if ready
            else "NOT_READY_FOR_MODEL_NATIVE_REPLAY_REVIEW"
        ),
        "model_native_replay_evidence_ready": bool(ready),
        "secondary_direction_authority_allowed": False,
        "promotion_shadow_live_allowed": False,
        "adaptation_lifecycle_handoff": adaptation_lifecycle_handoff_metadata(),
        "next_required_gate": (
            "immutable adaptation lifecycle admission; no direct launch pass-through"
            if ready
            else "repair candidate/selective-edge/replay evidence and rerun"
        ),
        "candidate_readiness_json": str(candidate_readiness_path),
        "candidate_bundle_audit_json": str(candidate_bundle_audit_path),
        "selective_edge_report_json": str(selective_report_path),
        "selective_edge_metrics_csv": str(selective_metrics_path),
        "replay_evidence_json": str(replay_report_path),
        "bundle_identity": bundle_identity,
        "evidence_identity": evidence_identity,
        "artifacts": artifacts,
        "artifact_fingerprints": artifact_fingerprints,
        "gates": gates,
        "failures": failures,
    }
    json_path = out_dir / f"ENTRY_REPLAY_READINESS_{timestamp}.json"
    md_path = out_dir / f"ENTRY_REPLAY_READINESS_{timestamp}.md"
    if json_path.exists() or md_path.exists():
        raise RuntimeError("immutable replay-readiness event already exists")
    report["json_path"] = str(json_path)
    report["md_path"] = str(md_path)
    atomic_write_text(
        json_path,
        json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n",
    )
    _write_markdown(md_path, report)

    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": report["decision"],
                    "failures": report["failures"],
                    "json_path": report["json_path"],
                    "md_path": report["md_path"],
                    "next_required_gate": report["next_required_gate"],
                },
                indent=2,
                sort_keys=True,
                default=_json_default,
            )
        )
    if failures:
        raise SystemExit(2)
    return report


def _publish_terminal_replay_readiness_failure(
    args: argparse.Namespace,
    error: Exception,
) -> None:
    """Invalidate any older READY event when a refresh crashes before evidence."""

    out_dir = Path(args.out_dir).expanduser().resolve()
    created = next_immutable_event_created_utc(
        out_dir,
        "ENTRY_REPLAY_READINESS",
    )
    write_immutable_json_event(
        out_dir,
        "ENTRY_REPLAY_READINESS",
        {
            "schema_version": "entry_replay_readiness_terminal_failure_v1",
            "created_utc": created.isoformat(),
            "decision": "NOT_READY_FOR_MODEL_NATIVE_REPLAY_REVIEW",
            "model_native_replay_evidence_ready": False,
            "secondary_direction_authority_allowed": False,
            "promotion_shadow_live_allowed": False,
            "failures": [
                {
                    "gate": "producer",
                    "check": "replay-readiness refresh completed",
                    "details": {
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                }
            ],
        },
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        return _run(args)
    except SystemExit:
        raise
    except Exception as exc:
        _publish_terminal_replay_readiness_failure(args, exc)
        raise

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-readiness-json", required=True)
    ap.add_argument("--candidate-bundle-audit-json", required=True)
    ap.add_argument("--selective-edge-report-json", required=True)
    ap.add_argument("--selective-edge-metrics-csv", required=True)
    ap.add_argument("--replay-evidence-json", required=True)
    ap.add_argument("--pretrain-audit-json", required=True)
    ap.add_argument("--expected-dataset-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> int:
    run(build_parser().parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
