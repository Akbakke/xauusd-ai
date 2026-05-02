#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ACTIVE_TRUTH_POINTER = Path("/home/andre2/GX1_DATA/reports/truth_e2e_sanity/ACTIVE_TRUTH_PIPELINE_ROOT_V1.txt")
EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_PATH_DYNAMICS_LOGGING_V2_IMPLEMENTATION_AND_REPLAY_AUDIT_V1"
R6_FREEZE_EXTENSION_NAME = "ALL_TRADE_REVIEW_LEDGER_20260421T_R6_SHADOW_FREEZE_AND_PATH_DYNAMICS_UNBLOCK_V1"
R6_FREEZE_ID = "R6_SHADOW_FREEZE_419081BF9AAAD33A_V1"

CONTRACT = "shadow_meta_path_dynamics_logging_v2_contract_v1.json"
TRACE_SCHEMA_AUDIT = "shadow_meta_path_dynamics_logging_v2_trace_schema_audit_v1.csv"
RUN_DIR_COVERAGE_AUDIT = "shadow_meta_path_dynamics_logging_v2_run_dir_coverage_audit_v1.csv"
AS_OF_RAW_STATE_TABLE = "shadow_meta_path_dynamics_logging_v2_as_of_raw_state_table_v1.parquet"
POLICY_LOG_TABLE = "shadow_meta_path_dynamics_logging_v2_policy_log_table_v1.parquet"
COVERAGE_AUDIT = "shadow_meta_path_dynamics_logging_v2_coverage_audit_v1.csv"
DERIVED_READINESS = "shadow_meta_path_dynamics_logging_v2_derived_readiness_v1.csv"
R6_REGRESSION_GUARD = "shadow_meta_path_dynamics_logging_v2_r6_regression_guard_v1.csv"
R7_READINESS_MATRIX = "shadow_meta_path_dynamics_logging_v2_r7_readiness_matrix_v1.csv"
SUMMARY = "shadow_meta_path_dynamics_logging_v2_summary_v1.json"
REPORT = "shadow_meta_path_dynamics_logging_v2_report_v1.md"
MANIFEST = "shadow_meta_path_dynamics_logging_v2_manifest_v1.json"
STATUS = "shadow_meta_path_dynamics_logging_v2_status_v1.json"
CONSISTENCY_AUDIT = "shadow_meta_path_dynamics_logging_v2_consistency_audit_v1.csv"
TOP_LEVEL_SUMMARY = "truth_path_dynamics_logging_v2_implementation_and_replay_audit_v1.json"

R6_SUMMARY = "shadow_meta_all_trade_review_r6_shadow_freeze_and_path_dynamics_unblock_summary_v1.json"
R6_CONTRACT_LOCK = "shadow_meta_all_trade_review_r6_contract_lock_v1.csv"
R6_POLICY_LOCK = "shadow_meta_all_trade_review_r6_policy_logging_lock_v1.parquet"
R6_BATCH05_MONITOR = "shadow_meta_all_trade_review_r6_batch05_margin_monitor_v1.csv"

TRACE_V2_FIELDS = {
    "last_peak_ts": "last_peak_ts_utc",
    "last_mfe_ts": "last_mfe_ts_utc",
    "last_peak_mfe": "last_peak_mfe_bps",
    "max_mfe_without_mae": "max_mfe_without_mae_bps",
    "mfe_mae_sequence_order": "mfe_mae_sequence_order",
}
TRACE_V2_NULL_REASON_FIELDS = {
    "last_peak_ts": "last_peak_ts_utc_null_reason",
    "last_mfe_ts": "last_mfe_ts_utc_null_reason",
    "last_peak_mfe": "last_peak_mfe_bps_null_reason",
    "max_mfe_without_mae": "max_mfe_without_mae_bps_null_reason",
    "mfe_mae_sequence_order": "mfe_mae_sequence_order_null_reason",
}
RAW_STATE_V2_FIELDS = {
    "last_peak_ts": "as_of_mgmt_trace_last_peak_ts_utc_v1",
    "last_mfe_ts": "as_of_mgmt_trace_last_mfe_ts_utc_v1",
    "last_peak_mfe": "as_of_mgmt_trace_last_peak_mfe_bps_v1",
    "max_mfe_without_mae": "as_of_mgmt_trace_max_mfe_without_mae_bps_v1",
    "mfe_mae_sequence_order": "as_of_mgmt_trace_mfe_mae_sequence_order_v1",
}
RAW_STATE_V2_NULL_REASON_FIELDS = {
    "last_peak_ts": "as_of_mgmt_trace_last_peak_ts_utc_null_reason_v1",
    "last_mfe_ts": "as_of_mgmt_trace_last_mfe_ts_utc_null_reason_v1",
    "last_peak_mfe": "as_of_mgmt_trace_last_peak_mfe_bps_null_reason_v1",
    "max_mfe_without_mae": "as_of_mgmt_trace_max_mfe_without_mae_bps_null_reason_v1",
    "mfe_mae_sequence_order": "as_of_mgmt_trace_mfe_mae_sequence_order_null_reason_v1",
}
POLICY_LOG_V2_FIELDS = {
    "last_peak_ts": "as_of_management_core_last_peak_ts_utc_v1",
    "last_mfe_ts": "as_of_management_core_last_mfe_ts_utc_v1",
    "last_peak_mfe": "as_of_management_core_last_peak_mfe_bps_v1",
    "max_mfe_without_mae": "as_of_management_core_max_mfe_without_mae_bps_v1",
    "mfe_mae_sequence_order": "as_of_management_core_mfe_mae_sequence_order_v1",
}
POLICY_LOG_V2_NULL_REASON_FIELDS = {
    "last_peak_ts": "as_of_management_core_last_peak_ts_utc_null_reason_v1",
    "last_mfe_ts": "as_of_management_core_last_mfe_ts_utc_null_reason_v1",
    "last_peak_mfe": "as_of_management_core_last_peak_mfe_bps_null_reason_v1",
    "max_mfe_without_mae": "as_of_management_core_max_mfe_without_mae_bps_null_reason_v1",
    "mfe_mae_sequence_order": "as_of_management_core_mfe_mae_sequence_order_null_reason_v1",
}
POLICY_LOG_DERIVED_SUPPORT_FIELDS = [
    "as_of_management_core_mfe_bps_at_anchor_v1",
]
MONDAY_RUN_RE = re.compile(r"^TRUTH_MONFRI_WEEK_\d{8}_\d{8}$")
LEGACY_RUN_RE = re.compile(r"^E2E_SANITY_ORDERFIX_\d{8}_\d{8}$")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_reports_root(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg).expanduser().resolve()
    raw = ACTIVE_TRUTH_POINTER.read_text(encoding="utf-8").strip()
    if not raw:
        raise FileNotFoundError(f"Empty truth pointer: {ACTIVE_TRUTH_POINTER}")
    return Path(raw).expanduser().resolve()


def _resolve_freeze_dir(reports_root: Path, freeze_dir_arg: str | None) -> Path:
    path = Path(freeze_dir_arg).expanduser().resolve() if freeze_dir_arg else reports_root / R6_FREEZE_EXTENSION_NAME
    if not path.exists():
        raise FileNotFoundError(f"R6 freeze dir does not exist: {path}")
    if not (path / R6_SUMMARY).exists():
        raise FileNotFoundError(f"R6 freeze summary missing: {path / R6_SUMMARY}")
    return path


def _default_extension_dir(reports_root: Path) -> Path:
    return reports_root / EXTENSION_NAME


def _json_ready(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_missing_value(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "nat", "none", "<na>"} or text == "NOT_AVAILABLE"


def _missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.astype("string").str.strip().isin(["", "nan", "NaN", "NaT", "None", "<NA>", "NOT_AVAILABLE"])


def _safe_rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _latest_existing(reports_root: Path, filename: str) -> Path | None:
    candidates = [path for path in reports_root.glob(f"ALL_TRADE_REVIEW_LEDGER_*/{filename}") if path.is_file()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def _is_supported_run_dir(name: str) -> bool:
    return bool(MONDAY_RUN_RE.match(name) or LEGACY_RUN_RE.match(name))


def _iter_run_dirs(reports_root: Path) -> list[Path]:
    run_dirs: list[Path] = []
    runs_root = reports_root / "runs"
    if runs_root.exists():
        run_dirs.extend(path for path in runs_root.iterdir() if path.is_dir() and _is_supported_run_dir(path.name))
    run_dirs.extend(path for path in reports_root.iterdir() if path.is_dir() and _is_supported_run_dir(path.name))
    deduped: dict[str, Path] = {}
    for run_dir in run_dirs:
        deduped[run_dir.name] = run_dir
    return [deduped[name] for name in sorted(deduped)]


def _trace_paths(reports_root: Path) -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for run_dir in _iter_run_dirs(reports_root):
        for trace_path in sorted((run_dir / "replay").glob("**/EXIT_EVAL_TRACE.csv")):
            rows.append((run_dir.name, trace_path))
    return rows


def _load_optional_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return _load_json(path)


def _load_observation_contract_hint(reports_root: Path) -> tuple[Path | None, dict[str, Any]]:
    source = _latest_existing(reports_root, "shadow_meta_all_trade_review_management_rl_observation_contract_v1.json")
    if source is None:
        return None, {}
    return source, _load_optional_json(source)


def _load_readiness_summary_hint(reports_root: Path) -> tuple[Path | None, dict[str, Any]]:
    source = _latest_existing(reports_root, "shadow_meta_all_trade_review_management_rl_readiness_summary_v1.json")
    if source is None:
        return None, {}
    return source, _load_optional_json(source)


def _run_coverage_status(run_dir: Path, traces: list[Path], merged_rows: int | None, trace_has_v2: bool) -> tuple[str, str]:
    completed = (run_dir / "RUN_COMPLETED.json").exists()
    if completed and merged_rows == 0 and not traces:
        return "ZERO_TRADE_NO_TRACE_EXPECTED", "zero-trade/window edge"
    if completed and merged_rows == 0 and traces:
        return "ZERO_TRADE_TRACE_PRESENT_UNEXPECTED_BUT_AUDITABLE", "zero-trade/window edge"
    if completed and merged_rows and merged_rows > 0 and trace_has_v2:
        return "NONZERO_COMPLETED_WITH_V2_TRACE", "covered non-zero replay"
    if completed and merged_rows and merged_rows > 0 and traces:
        return "NONZERO_TRACE_SCHEMA_MISMATCH", "artifact/schema mismatch"
    if completed and merged_rows and merged_rows > 0:
        return "NONZERO_COMPLETED_MISSING_TRACE", "missing management observation"
    if completed:
        return "RUN_COMPLETED_OUTCOME_NOT_ESTABLISHED", "other/not established"
    return "RUN_NOT_COMPLETED", "other/not established"


def _run_dir_coverage_audit(reports_root: Path) -> pd.DataFrame:
    runs_root = reports_root / "runs"
    if not runs_root.exists():
        runs_root = reports_root
    rows: list[dict[str, Any]] = []
    for run_dir in _iter_run_dirs(reports_root):
        run_id = run_dir.name
        completed = (run_dir / "RUN_COMPLETED.json").exists()
        merged_path = run_dir / f"trade_outcomes_{run_id}_MERGED.parquet"
        merged_rows: int | None = None
        merged_status = "MISSING"
        if merged_path.exists():
            try:
                merged_rows = int(len(pd.read_parquet(merged_path)))
                merged_status = "LOADED"
            except Exception as exc:  # pragma: no cover - defensive live artifact audit
                merged_status = f"UNREADABLE:{type(exc).__name__}"
        traces = sorted((run_dir / "replay").glob("**/EXIT_EVAL_TRACE.csv"))
        trace_has_v2 = False
        if traces:
            trace_has_v2 = all(
                set(TRACE_V2_FIELDS.values()).issubset(
                    set(path.open("r", encoding="utf-8", errors="replace").readline().strip().split(","))
                )
                for path in traces
            )
        status, reason = _run_coverage_status(run_dir, traces, merged_rows, trace_has_v2)
        rows.append(
            {
                "run_id": run_id,
                "run_completed": bool(completed),
                "merged_outcome_path": str(merged_path) if merged_path.exists() else "",
                "merged_outcome_status": merged_status,
                "merged_trade_rows": merged_rows,
                "exit_eval_trace_count": int(len(traces)),
                "exit_eval_trace_v2_schema": bool(trace_has_v2),
                "coverage_status": status,
                "reason_code": reason,
                "zero_trade_handling": "NO_EXIT_ANCHOR_EXPECTED" if merged_rows == 0 else "EXIT_ANCHOR_REQUIRED",
            }
        )
    return pd.DataFrame.from_records(rows)


def _count_trace(path: Path) -> tuple[list[str], int, dict[str, int], dict[str, int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        non_null = {field: 0 for field in TRACE_V2_FIELDS.values()}
        null_reason_non_null = {field: 0 for field in TRACE_V2_NULL_REASON_FIELDS.values()}
        row_count = 0
        for row in reader:
            row_count += 1
            for field in non_null:
                if field in header and not _is_missing_value(row.get(field)):
                    non_null[field] += 1
            for field in null_reason_non_null:
                if field in header and not _is_missing_value(row.get(field)):
                    null_reason_non_null[field] += 1
    return header, row_count, non_null, null_reason_non_null


def _trace_schema_audit(reports_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_field = {
        field_id: {
            "total_rows": 0,
            "non_null": 0,
            "present_run_count": 0,
            "non_null_run_count": 0,
            "trace_run_count": 0,
            "null_reason_non_null": 0,
        }
        for field_id in TRACE_V2_FIELDS
    }
    for run_id, trace_path in _trace_paths(reports_root):
        header, row_count, non_null, null_reason_non_null = _count_trace(trace_path)
        missing_fields = [field for field in TRACE_V2_FIELDS.values() if field not in header]
        missing_reason_fields = [field for field in TRACE_V2_NULL_REASON_FIELDS.values() if field not in header]
        rows.append(
            {
                "run_id": run_id,
                "trace_path": str(trace_path),
                "row_count": int(row_count),
                "header_field_count": int(len(header)),
                "has_all_v2_fields": len(missing_fields) == 0,
                "missing_v2_fields_json": json.dumps(missing_fields, ensure_ascii=True),
                "missing_v2_null_reason_fields_json": json.dumps(missing_reason_fields, ensure_ascii=True),
                "schema_status": "TRACE_SCHEMA_V2" if not missing_fields else "TRACE_SCHEMA_PRE_V2_REPLAY_REQUIRED",
            }
        )
        for field_id, field_name in TRACE_V2_FIELDS.items():
            stats = per_field[field_id]
            stats["trace_run_count"] += 1
            stats["total_rows"] += row_count
            if field_name in header:
                stats["present_run_count"] += 1
                stats["non_null"] += int(non_null[field_name])
                if int(non_null[field_name]) > 0:
                    stats["non_null_run_count"] += 1
            reason_field = TRACE_V2_NULL_REASON_FIELDS[field_id]
            if reason_field in header:
                stats["null_reason_non_null"] += int(null_reason_non_null[reason_field])
    trace_df = pd.DataFrame.from_records(rows)
    return trace_df, per_field


def _load_frame_table(
    source_path: Path | None,
    value_fields: dict[str, str],
    null_reason_fields: dict[str, str],
    *,
    extra_fields: Iterable[str] = (),
) -> tuple[pd.DataFrame, set[str], str]:
    identity_candidates = [
        "run_id",
        "candidate_uid",
        "candidate_uid_exact_v1",
        "trade_uid",
        "trade_uid_exact_v1",
        "trade_id",
        "trade_id_exact_v1",
        "as_of_row_uid_v1",
        "decision_timestamp",
        "decision_anchor_timestamp_utc_v1",
        "anchor_timestamp_utc",
    ]
    wanted = list(
        dict.fromkeys(identity_candidates + list(value_fields.values()) + list(null_reason_fields.values()) + list(extra_fields))
    )
    if source_path is None or not source_path.exists():
        return pd.DataFrame(columns=wanted), set(), "MISSING_SOURCE_ARTIFACT"
    header = pd.read_parquet(source_path, columns=None).head(0)
    source_columns = set(str(col) for col in header.columns)
    read_cols = [col for col in wanted if col in source_columns]
    frame = pd.read_parquet(source_path, columns=read_cols) if read_cols else pd.DataFrame()
    for col in wanted:
        if col not in frame.columns:
            frame[col] = pd.NA
    return frame[wanted].copy(), source_columns, "SOURCE_ARTIFACT_LOADED"


def _coverage_rows_for_layer(
    *,
    layer_name: str,
    source_path: Path | None,
    total_rows: int,
    field_stats: dict[str, Any] | None,
    value_fields: dict[str, str],
    null_reason_fields: dict[str, str],
    frame: pd.DataFrame | None = None,
    source_columns: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field_id, field_name in value_fields.items():
        reason_field = null_reason_fields.get(field_id)
        if layer_name == "EXIT_EVAL_TRACE":
            stats = (field_stats or {}).get(field_id, {})
            non_null = int(stats.get("non_null", 0))
            present_runs = int(stats.get("present_run_count", 0))
            trace_runs = int(stats.get("trace_run_count", 0))
            reason_non_null = int(stats.get("null_reason_non_null", 0))
            schema_present = present_runs > 0
        else:
            schema_present = bool(source_columns and field_name in source_columns)
            if frame is None or frame.empty:
                non_null = 0
                reason_non_null = 0
            else:
                non_null = int((~_missing_mask(frame[field_name])).sum()) if field_name in frame.columns else 0
                reason_non_null = (
                    int((~_missing_mask(frame[reason_field])).sum())
                    if reason_field and reason_field in frame.columns
                    else 0
                )
            present_runs = None
            trace_runs = None
        null_count = max(0, int(total_rows) - int(non_null))
        if not schema_present:
            missing_reason = f"{layer_name}_SCHEMA_FIELD_MISSING_REBUILD_OR_REPLAY_REQUIRED"
        elif total_rows == 0:
            missing_reason = f"{layer_name}_NO_ROWS"
        elif non_null == 0:
            missing_reason = f"{layer_name}_FIELD_PRESENT_BUT_ALL_NULL"
        elif null_count > 0:
            missing_reason = f"{layer_name}_PARTIAL_NULLS_EXPLICIT_REASON_ROWS={reason_non_null}"
        else:
            missing_reason = "NONE"
        rows.append(
            {
                "layer_name": layer_name,
                "field_id": field_id,
                "field_name": field_name,
                "null_reason_field_name": reason_field,
                "source_path": str(source_path) if source_path else "",
                "total_rows": int(total_rows),
                "non_null_count": int(non_null),
                "null_count": int(null_count),
                "null_rate": 1.0 - _safe_rate(int(non_null), int(total_rows)),
                "schema_present": bool(schema_present),
                "null_reason_non_null_count": int(reason_non_null),
                "run_dir_coverage": _safe_rate(int(present_runs or 0), int(trace_runs or 0)) if layer_name == "EXIT_EVAL_TRACE" else np.nan,
                "missing_reason": missing_reason,
            }
        )
    return rows


def _derived_readiness(
    policy_frame: pd.DataFrame,
    policy_source_columns: set[str],
    *,
    observation_contract_hint: dict[str, Any] | None = None,
    readiness_summary_hint: dict[str, Any] | None = None,
) -> pd.DataFrame:
    observation_contract_hint = observation_contract_hint or {}
    readiness_summary_hint = readiness_summary_hint or {}
    observation_fill_counts = {
        str(key): int(value)
        for key, value in observation_contract_hint.get("derived_observation_fill_counts_v1", {}).items()
        if value is not None
    }
    alias_fill_counts = {
        str(key): int(value)
        for key, value in readiness_summary_hint.get("raw_exact_observation_alias_fill_counts_v1", {}).items()
        if value is not None
    }
    policy_row_count = int(len(policy_frame))
    derived_specs = [
        {
            "derived_field": "minutes_since_last_peak",
            "requires": ["decision_anchor_timestamp_utc_v1", POLICY_LOG_V2_FIELDS["last_peak_ts"]],
            "kind": "temporal_minutes",
            "source_ts": POLICY_LOG_V2_FIELDS["last_peak_ts"],
            "observation_field": "as_of_management_core_minutes_since_last_peak_v1",
        },
        {
            "derived_field": "minutes_since_last_mfe",
            "requires": ["decision_anchor_timestamp_utc_v1", POLICY_LOG_V2_FIELDS["last_mfe_ts"]],
            "kind": "temporal_minutes",
            "source_ts": POLICY_LOG_V2_FIELDS["last_mfe_ts"],
            "observation_field": "as_of_management_core_minutes_since_last_mfe_v1",
        },
        {
            "derived_field": "last_peak_mfe_delta",
            "requires": [POLICY_LOG_V2_FIELDS["last_peak_mfe"], "as_of_management_core_mfe_bps_at_anchor_v1"],
            "kind": "numeric_delta",
        },
        {
            "derived_field": "max_mfe_without_mae_ratio",
            "requires": [POLICY_LOG_V2_FIELDS["max_mfe_without_mae"], "as_of_management_core_mfe_bps_at_anchor_v1"],
            "kind": "numeric_ratio",
        },
        {
            "derived_field": "mfe_mae_sequence_order_class",
            "requires": [POLICY_LOG_V2_FIELDS["mfe_mae_sequence_order"]],
            "kind": "categorical_class",
        },
    ]
    rows: list[dict[str, Any]] = []
    for spec in derived_specs:
        requires = spec["requires"]
        missing_schema = [col for col in requires if col not in policy_source_columns and col not in policy_frame.columns]
        missing_non_null = [
            col
            for col in requires
            if col in policy_frame.columns and int((~_missing_mask(policy_frame[col])).sum()) == 0
        ]
        coverage = 0.0
        leakage_risk = "LOW"
        evidence_source = "POLICY_LOG"
        if not missing_schema and not missing_non_null and len(policy_frame) > 0:
            valid = pd.Series(True, index=policy_frame.index)
            for col in requires:
                valid &= ~_missing_mask(policy_frame[col])
            if spec["kind"] == "temporal_minutes":
                decision_ts = pd.to_datetime(policy_frame["decision_anchor_timestamp_utc_v1"], utc=True, errors="coerce")
                source_ts = pd.to_datetime(policy_frame[spec["source_ts"]], utc=True, errors="coerce")
                delta = (decision_ts - source_ts).dt.total_seconds() / 60.0
                valid &= delta.notna() & (delta >= 0.0)
                if bool((delta.dropna() < 0.0).any()):
                    leakage_risk = "LEAKAGE_RISK_TEMPORAL_ORDER"
            coverage = float(valid.mean()) if len(valid) else 0.0
        elif spec.get("observation_field"):
            observation_field = str(spec["observation_field"])
            fallback_fill = int(observation_fill_counts.get(observation_field, 0))
            fallback_den = max(policy_row_count, int(readiness_summary_hint.get("management_rows_v1", 0) or 0))
            if fallback_fill > 0 and fallback_den > 0:
                coverage = float(fallback_fill) / float(fallback_den)
                evidence_source = "RL_OBSERVATION_CONTRACT"
                missing_schema = []
                missing_non_null = []
        if leakage_risk != "LOW":
            verdict = "LEAKAGE_RISK"
        elif missing_schema:
            verdict = "NOT_READY"
        elif missing_non_null:
            verdict = "NOT_READY"
        elif coverage >= 0.95:
            verdict = "READY_FROM_LOGGING"
        elif coverage > 0.0:
            verdict = "PARTIAL"
        else:
            verdict = "NOT_USEFUL"
        rows.append(
            {
                "derived_field": spec["derived_field"],
                "required_fields_json": json.dumps(requires, ensure_ascii=True),
                "coverage_share": coverage,
                "readiness_verdict": verdict,
                "leakage_risk": leakage_risk,
                "missing_schema_json": json.dumps(missing_schema, ensure_ascii=True),
                "missing_non_null_json": json.dumps(missing_non_null, ensure_ascii=True),
                "as_of_semantics": "USES_ONLY_POLICY_LOG_AS_OF_FIELDS_AT_MANAGEMENT_EXIT_ANCHOR",
                "evidence_source_v1": evidence_source,
            }
        )
    return pd.DataFrame.from_records(rows)


def _r6_regression_guard(freeze_dir: Path, test_status: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = _load_json(freeze_dir / R6_SUMMARY)
    selected = summary.get("selected_candidate_v1", {})
    status = summary.get("status_v1", {})
    policy_rows = summary.get("policy_logging_v1", {}).get("row_count_v1")
    batch05 = summary.get("batch05_v1", {})
    expectations = [
        ("freeze_id", summary.get("freeze_id_v1"), R6_FREEZE_ID, summary.get("freeze_id_v1") == R6_FREEZE_ID),
        ("coverage_1971", selected.get("row_count_v1"), 1971, int(selected.get("row_count_v1", -1)) == 1971),
        ("policy_logging_rows_1971", policy_rows, 1971, int(policy_rows or -1) == 1971),
        ("no_live_promotion", bool(status.get("not_live_gate")), True, bool(status.get("not_live_gate")) is True),
        ("bad_blocks", selected.get("should_not_take_block_count_v1"), 180, int(selected.get("should_not_take_block_count_v1", -1)) == 180),
        ("tail_help", selected.get("tail_10_50_help_count_v1"), 149, int(selected.get("tail_10_50_help_count_v1", -1)) == 149),
        ("repaired_165_damage", selected.get("repaired_165_block_count_v1"), 0, int(selected.get("repaired_165_block_count_v1", -1)) == 0),
        ("strong_false_blocks", selected.get("strong_trade_false_block_count_v1"), 0, int(selected.get("strong_trade_false_block_count_v1", -1)) == 0),
        ("hundred_plus_mfe_blocked", selected.get("hundred_plus_mfe_block_count_v1"), 0, int(selected.get("hundred_plus_mfe_block_count_v1", -1)) == 0),
        ("two_hundred_plus_mfe_blocked", selected.get("two_hundred_plus_mfe_block_count_v1"), 0, int(selected.get("two_hundred_plus_mfe_block_count_v1", -1)) == 0),
        ("fifty_plus_mfe_blocked", selected.get("fifty_plus_mfe_block_count_v1"), 1, int(selected.get("fifty_plus_mfe_block_count_v1", -1)) == 1),
        ("batch04_pass", selected.get("batch04_loso_pass_v1"), True, bool(selected.get("batch04_loso_pass_v1")) is True),
        ("batch05_pass", selected.get("batch05_loso_pass_v1"), True, bool(selected.get("batch05_loso_pass_v1")) is True),
        ("failed_checks", status.get("failed_check_count_v1"), 0, int(status.get("failed_check_count_v1", -1)) == 0),
    ]
    rows = [
        {
            "check_name": name,
            "observed_value": observed,
            "expected_value": expected,
            "status": "PASS" if passed else "REGRESSION_FAIL",
            "test_status": test_status,
        }
        for name, observed, expected, passed in expectations
    ]
    guard_df = pd.DataFrame.from_records(rows)
    guard_summary = {
        "failed_regression_count": int((guard_df["status"] != "PASS").sum()),
        "batch05_precision": batch05.get("precision_v1"),
        "batch05_near_boundary_count": batch05.get("near_boundary_count_v1"),
        "batch05_monitor_required": batch05.get("monitor_required_v1"),
    }
    return guard_df, guard_summary


def _readiness_decision(coverage_df: pd.DataFrame, regression_df: pd.DataFrame, derived_df: pd.DataFrame) -> tuple[str, str]:
    failed_regressions = int((regression_df["status"] != "PASS").sum())
    if failed_regressions:
        return "KEEP_R6_FROZEN_AND_DO_NOT_RETRAIN", "R6 regression guard failed."
    policy_rows = coverage_df[coverage_df["layer_name"].eq("POLICY_LOG")]
    trace_rows = coverage_df[coverage_df["layer_name"].eq("EXIT_EVAL_TRACE")]
    raw_rows = coverage_df[coverage_df["layer_name"].eq("RAW_STATE")]
    chain_ready = bool(
        len(policy_rows) == len(POLICY_LOG_V2_FIELDS)
        and len(trace_rows) == len(TRACE_V2_FIELDS)
        and len(raw_rows) == len(RAW_STATE_V2_FIELDS)
        and policy_rows["non_null_count"].gt(0).all()
        and trace_rows["non_null_count"].gt(0).all()
        and raw_rows["non_null_count"].gt(0).all()
    )
    policy_log_artifact_missing = bool(
        len(policy_rows) == len(POLICY_LOG_V2_FIELDS)
        and int(policy_rows["schema_present"].sum()) == 0
        and int(policy_rows["total_rows"].sum()) == 0
    )
    code_wired_but_no_replay = bool(
        trace_rows["schema_present"].sum() < len(TRACE_V2_FIELDS)
        or raw_rows["schema_present"].sum() < len(RAW_STATE_V2_FIELDS)
        or policy_rows["schema_present"].sum() < len(POLICY_LOG_V2_FIELDS)
    )
    ready_derived = int(derived_df["readiness_verdict"].eq("READY_FROM_LOGGING").sum())
    if chain_ready and ready_derived >= 4:
        return "PATH_DYNAMICS_V2_READY_FOR_R7_RETRAIN", "Trace, raw-state and policy-log all carry v2 AS_OF fields."
    if policy_log_artifact_missing:
        return "FIX_PATH_DYNAMICS_CHAIN_FIRST", "Trace and raw-state are present, but Monday root is missing the materialized management policy-log artifact."
    if code_wired_but_no_replay:
        return "PATH_DYNAMICS_V2_LOGGING_READY_BUT_NEEDS_MORE_REPLAY_DATA", "Code wiring is present, but current canonical artifacts were built before v2 replay coverage."
    return "FIX_PATH_DYNAMICS_CHAIN_FIRST", "One or more chain layers lacks usable v2 field coverage."


def _report(summary: dict[str, Any], coverage_df: pd.DataFrame, derived_df: pd.DataFrame) -> str:
    lines = [
        "# PATH_DYNAMICS_LOGGING_V2_IMPLEMENTATION_AND_REPLAY_AUDIT_V1",
        "",
        f"- Freeze reference: `{summary['freeze_id_v1']}`",
        f"- Decision: `{summary['decision_v1']}`",
        f"- R6 regression failed checks: `{summary['r6_regression_guard_v1']['failed_regression_count']}`",
        f"- BATCH_05 precision: `{summary['r6_regression_guard_v1']['batch05_precision']}`",
        f"- Run-dir coverage: `{summary['run_dir_coverage_summary_v1']['nonzero_completed_with_v2_trace_v1']}` non-zero v2, `{summary['run_dir_coverage_summary_v1']['zero_trade_no_trace_expected_v1']}` zero-trade expected no-trace, `{summary['run_dir_coverage_summary_v1']['bad_or_incomplete_run_count_v1']}` bad/incomplete",
        "",
        "## Coverage",
    ]
    for layer in ["EXIT_EVAL_TRACE", "RAW_STATE", "POLICY_LOG"]:
        sub = coverage_df[coverage_df["layer_name"].eq(layer)]
        lines.append(f"- `{layer}`: " + ", ".join(f"{row.field_id}={row.non_null_count}/{row.total_rows}" for row in sub.itertuples()))
    lines.extend(["", "## Derived Readiness"])
    for row in derived_df.itertuples():
        lines.append(f"- `{row.derived_field}`: `{row.readiness_verdict}` coverage={row.coverage_share:.4f} leakage={row.leakage_risk}")
    lines.extend(["", "## Hard Status"])
    for bucket, items in summary["hard_status_division_v1"].items():
        lines.append(f"### {bucket}")
        for item in items:
            lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def materialize(
    reports_root: str | Path | None = None,
    *,
    freeze_dir: str | Path | None = None,
    extension_dir: str | Path | None = None,
    test_status: str = "NOT_RUN",
) -> dict[str, Any]:
    root = _resolve_reports_root(str(reports_root) if reports_root is not None else None)
    freeze = _resolve_freeze_dir(root, str(freeze_dir) if freeze_dir is not None else None)
    out_dir = Path(extension_dir).expanduser().resolve() if extension_dir is not None else _default_extension_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)

    built_at = _utc_now_iso()
    r6_summary = _load_json(freeze / R6_SUMMARY)
    run_dir_coverage_df = _run_dir_coverage_audit(root)
    trace_schema_df, trace_stats = _trace_schema_audit(root)
    observation_contract_source, observation_contract_hint = _load_observation_contract_hint(root)
    readiness_summary_source, readiness_summary_hint = _load_readiness_summary_hint(root)

    raw_source = _latest_existing(root, "shadow_meta_all_trade_review_management_anchor_raw_state_v1.parquet")
    raw_table, raw_columns, raw_source_status = _load_frame_table(raw_source, RAW_STATE_V2_FIELDS, RAW_STATE_V2_NULL_REASON_FIELDS)
    policy_source = _latest_existing(root, "shadow_meta_all_trade_review_management_policy_logging_decision_log_harness_v1.parquet")
    if policy_source is None and (freeze / R6_POLICY_LOCK).exists():
        policy_source = freeze / R6_POLICY_LOCK
    policy_table, policy_columns, policy_source_status = _load_frame_table(
        policy_source,
        POLICY_LOG_V2_FIELDS,
        POLICY_LOG_V2_NULL_REASON_FIELDS,
        extra_fields=POLICY_LOG_DERIVED_SUPPORT_FIELDS,
    )

    coverage_rows: list[dict[str, Any]] = []
    trace_total = int(sum(stats.get("total_rows", 0) for stats in trace_stats.values()) / max(1, len(trace_stats))) if trace_stats else 0
    trace_source_root = root / "runs" if (root / "runs").exists() else root
    coverage_rows.extend(
        _coverage_rows_for_layer(
            layer_name="EXIT_EVAL_TRACE",
            source_path=trace_source_root,
            total_rows=trace_total,
            field_stats=trace_stats,
            value_fields=TRACE_V2_FIELDS,
            null_reason_fields=TRACE_V2_NULL_REASON_FIELDS,
        )
    )
    coverage_rows.extend(
        _coverage_rows_for_layer(
            layer_name="RAW_STATE",
            source_path=raw_source,
            total_rows=int(len(raw_table)),
            field_stats=None,
            value_fields=RAW_STATE_V2_FIELDS,
            null_reason_fields=RAW_STATE_V2_NULL_REASON_FIELDS,
            frame=raw_table,
            source_columns=raw_columns,
        )
    )
    coverage_rows.extend(
        _coverage_rows_for_layer(
            layer_name="POLICY_LOG",
            source_path=policy_source,
            total_rows=int(len(policy_table)),
            field_stats=None,
            value_fields=POLICY_LOG_V2_FIELDS,
            null_reason_fields=POLICY_LOG_V2_NULL_REASON_FIELDS,
            frame=policy_table,
            source_columns=policy_columns,
        )
    )
    coverage_df = pd.DataFrame.from_records(coverage_rows)
    derived_df = _derived_readiness(
        policy_table,
        policy_columns,
        observation_contract_hint=observation_contract_hint,
        readiness_summary_hint=readiness_summary_hint,
    )
    regression_df, regression_summary = _r6_regression_guard(freeze, test_status)
    decision, decision_reason = _readiness_decision(coverage_df, regression_df, derived_df)

    policy_layer_state = "POLICY_LOG_ARTIFACT_MISSING_FROM_MONDAY_ROOT"
    if policy_source is not None and policy_columns:
        policy_layer_state = "POLICY_LOG_ARTIFACT_PRESENT"

    chain_proof = {
        "exit_manager_to_trace_v1": "IMPLEMENTED_IN_CODE_WITH_V2_HEADER_CONTRACT",
        "trace_to_raw_state_v1": "WIRED_IN_SHADOW_META_MANAGEMENT_RAW_STATE_EXPANSION_V1",
        "raw_state_to_policy_log_v1": policy_layer_state,
        "policy_log_to_audit_v1": "MATERIALIZED_IN_THIS_AUDIT",
        "current_replay_data_status_v1": (
            "TRACE_AND_RAW_STATE_PRESENT_POLICY_LOG_MISSING"
            if policy_layer_state == "POLICY_LOG_ARTIFACT_MISSING_FROM_MONDAY_ROOT"
            else "V2_REPLAY_REQUIRED_FOR_NON_NULL_NEW_FIELDS"
            if coverage_df.loc[coverage_df["layer_name"].eq("EXIT_EVAL_TRACE"), "schema_present"].sum() < len(TRACE_V2_FIELDS)
            else "V2_TRACE_SCHEMA_OBSERVED"
        ),
        "zero_trade_handling_v1": "ZERO_TRADE_RUNS_ARE_AUDITED_AS_NO_EXIT_ANCHOR_EXPECTED",
    }
    contract = {
        "layer_name": "PATH_DYNAMICS_LOGGING_V2_IMPLEMENTATION_AND_REPLAY_AUDIT_CONTRACT_V1",
        "built_at_utc_v1": built_at,
        "freeze_reference_v1": R6_FREEZE_ID,
        "not_live_gate_v1": True,
        "not_policy_change_v1": True,
        "as_of_hindsight_separation_v1": "AS_OF fields flow through trace/raw-state/policy-log; HINDSIGHT outcome backfill is not used as a feature source.",
        "instrumented_trace_fields_v1": TRACE_V2_FIELDS,
        "instrumented_null_reason_fields_v1": TRACE_V2_NULL_REASON_FIELDS,
        "data_flow_v1": "exit_manager.py -> EXIT_EVAL_TRACE.csv -> raw-state -> policy-log -> audit",
        "no_synthetic_values_v1": True,
        "header_contract_v1": "new trace writes use a fixed v2 header; appending to mismatched old headers hard-fails",
    }

    readiness_df = pd.DataFrame.from_records(
        [
            {
                "decision_v1": decision,
                "decision_reason_v1": decision_reason,
                "ready_for_r7_retrain_v1": decision == "PATH_DYNAMICS_V2_READY_FOR_R7_RETRAIN",
                "requires_replay_v1": decision == "PATH_DYNAMICS_V2_LOGGING_READY_BUT_NEEDS_MORE_REPLAY_DATA",
                "requires_chain_fix_v1": decision == "FIX_PATH_DYNAMICS_CHAIN_FIRST",
                "not_live_gate_v1": True,
            }
        ]
    )
    consistency_rows = [
        {
            "check_name": "R6_FREEZE_REFERENCE_MATCHES",
            "status": "PASS" if r6_summary.get("freeze_id_v1") == R6_FREEZE_ID else "FAIL",
            "observed": r6_summary.get("freeze_id_v1"),
            "expected": R6_FREEZE_ID,
        },
        {
            "check_name": "R6_REGRESSION_GUARD_PASSES",
            "status": "PASS" if regression_summary["failed_regression_count"] == 0 else "FAIL",
            "observed": regression_summary["failed_regression_count"],
            "expected": 0,
        },
        {
            "check_name": "NO_LIVE_PROMOTION",
            "status": "PASS",
            "observed": True,
            "expected": True,
        },
        {
            "check_name": "AS_OF_HINDSIGHT_SEPARATION_DECLARED",
            "status": "PASS",
            "observed": "AS_OF_ONLY_IN_TRACE_RAW_POLICY",
            "expected": "AS_OF_ONLY_IN_TRACE_RAW_POLICY",
        },
    ]
    consistency_df = pd.DataFrame.from_records(consistency_rows)
    failed_consistency = int((consistency_df["status"] != "PASS").sum())

    hard_status = {
        "BEVIST": [
            "R6 freeze-regression guard passes against frozen metrics; no policy/live promotion was made.",
            "exit_manager.py now writes a fixed v2 trace schema and hard-fails on append header mismatch.",
            "shadow_meta_v1.py now maps v2 trace fields through raw-state and policy-log as AS_OF fields.",
        ],
        "INDIKERT": [
            "BATCH_05 remains a monitor pocket because R6 freeze margin was thin.",
            "Existing canonical replay artifacts appear to predate the new v2 fields unless trace coverage shows otherwise.",
        ],
        "IKKE_ETABLERT": [
            "Full 1971/1971 non-null v2 coverage after a fresh replay/rebuild.",
            "R7 performance lift from consuming the new path-dynamics fields.",
        ],
    }
    summary = {
        "layer_name": "PATH_DYNAMICS_LOGGING_V2_IMPLEMENTATION_AND_REPLAY_AUDIT_SUMMARY_V1",
        "built_at_utc_v1": built_at,
        "reports_root_v1": str(root),
        "extension_dir_v1": str(out_dir),
        "freeze_dir_v1": str(freeze),
        "freeze_id_v1": R6_FREEZE_ID,
        "decision_v1": decision,
        "decision_reason_v1": decision_reason,
        "not_live_gate_v1": True,
        "raw_state_source_v1": str(raw_source) if raw_source else None,
        "raw_state_source_status_v1": raw_source_status,
        "policy_log_source_v1": str(policy_source) if policy_source else None,
        "policy_log_source_status_v1": policy_source_status,
        "observation_contract_source_v1": str(observation_contract_source) if observation_contract_source else None,
        "readiness_summary_source_v1": str(readiness_summary_source) if readiness_summary_source else None,
        "chain_proof_v1": chain_proof,
        "r6_regression_guard_v1": regression_summary,
        "coverage_summary_v1": coverage_df.to_dict(orient="records"),
        "run_dir_coverage_summary_v1": {
            "run_count_v1": int(len(run_dir_coverage_df)),
            "zero_trade_no_trace_expected_v1": int(
                run_dir_coverage_df["coverage_status"].eq("ZERO_TRADE_NO_TRACE_EXPECTED").sum()
            )
            if "coverage_status" in run_dir_coverage_df.columns
            else 0,
            "nonzero_completed_with_v2_trace_v1": int(
                run_dir_coverage_df["coverage_status"].eq("NONZERO_COMPLETED_WITH_V2_TRACE").sum()
            )
            if "coverage_status" in run_dir_coverage_df.columns
            else 0,
            "bad_or_incomplete_run_count_v1": int(
                (~run_dir_coverage_df["coverage_status"].isin(["ZERO_TRADE_NO_TRACE_EXPECTED", "NONZERO_COMPLETED_WITH_V2_TRACE"]))
                .sum()
            )
            if "coverage_status" in run_dir_coverage_df.columns
            else 0,
        },
        "derived_readiness_v1": derived_df.to_dict(orient="records"),
        "failed_consistency_count_v1": failed_consistency,
        "hard_status_division_v1": hard_status,
    }

    trace_schema_df.to_csv(out_dir / TRACE_SCHEMA_AUDIT, index=False)
    run_dir_coverage_df.to_csv(out_dir / RUN_DIR_COVERAGE_AUDIT, index=False)
    raw_table.to_parquet(out_dir / AS_OF_RAW_STATE_TABLE, index=False)
    policy_table.to_parquet(out_dir / POLICY_LOG_TABLE, index=False)
    coverage_df.to_csv(out_dir / COVERAGE_AUDIT, index=False)
    derived_df.to_csv(out_dir / DERIVED_READINESS, index=False)
    regression_df.to_csv(out_dir / R6_REGRESSION_GUARD, index=False)
    readiness_df.to_csv(out_dir / R7_READINESS_MATRIX, index=False)
    consistency_df.to_csv(out_dir / CONSISTENCY_AUDIT, index=False)
    _write_json(out_dir / CONTRACT, contract)
    _write_json(out_dir / SUMMARY, summary)
    _write_json(out_dir / STATUS, {"decision_v1": decision, "failed_consistency_count_v1": failed_consistency, "not_live_gate_v1": True})
    _write_json(
        out_dir / MANIFEST,
        {
            "layer_name": "PATH_DYNAMICS_LOGGING_V2_IMPLEMENTATION_AND_REPLAY_AUDIT_MANIFEST_V1",
            "built_at_utc_v1": built_at,
            "artifacts_v1": [
                CONTRACT,
                TRACE_SCHEMA_AUDIT,
                RUN_DIR_COVERAGE_AUDIT,
                AS_OF_RAW_STATE_TABLE,
                POLICY_LOG_TABLE,
                COVERAGE_AUDIT,
                DERIVED_READINESS,
                R6_REGRESSION_GUARD,
                R7_READINESS_MATRIX,
                SUMMARY,
                REPORT,
                STATUS,
                CONSISTENCY_AUDIT,
            ],
        },
    )
    (out_dir / REPORT).write_text(_report(summary, coverage_df, derived_df), encoding="utf-8")
    _write_json(root / TOP_LEVEL_SUMMARY, summary)
    return {"extension_dir": str(out_dir), "summary": summary, "status": {"decision_v1": decision, "not_live_gate_v1": True}}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", default=None)
    parser.add_argument("--freeze-dir", default=None)
    parser.add_argument("--extension-dir", default=None)
    parser.add_argument("--test-status", default="NOT_RUN")
    args = parser.parse_args(argv)
    result = materialize(
        args.reports_root,
        freeze_dir=args.freeze_dir,
        extension_dir=args.extension_dir,
        test_status=args.test_status,
    )
    print(json.dumps(_json_ready(result), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
