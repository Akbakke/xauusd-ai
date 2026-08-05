"""Exact immutable calibration metadata shared by producer and bundle loader."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from gx1.contracts.entry_run_lineage_v1 import require_entry_run_id


IMMUTABLE_CALIBRATION_EVENT_SCHEMA_VERSION = (
    "entry_model_native_immutable_calibration_v3"
)
DIRECTION_CALIBRATION_VERSION = "entry_model_native_direction_calibration_v2"
DIRECTION_CALIBRATION_TRANSFORM = (
    "direction_logits=raw_direction_logits/temperature"
)
DIRECTION_CALIBRATION_TIE_POLICY = "fail_closed"
PATH_CALIBRATION_VERSION = "entry_model_native_path_calibration_v1"
CALIBRATION_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_CALIBRATION_"
CALIBRATION_FIT_SPLITS = ("val",)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STAMP_RE = r"\d{8}T\d{12}Z"
_PREDICTION_NAME_RE = re.compile(
    rf"^selective_edge_predictions_{_STAMP_RE}\.parquet$"
)
_REPORT_NAME_RE = re.compile(
    rf"^ENTRY_CANDIDATE_SELECTIVE_EDGE_{_STAMP_RE}\.json$"
)
_COMMON_KEYS = {
    "enabled",
    "version",
    "fitted_at_utc",
    "fitted_on_split",
    "fitted_rows",
    "model",
    "min_fit_rows",
    "run_id",
    "source_bundle_dir",
    "source_bundle_metadata_sha256",
    "predictions_path",
    "predictions_sha256",
    "prediction_report_path",
    "prediction_report_sha256",
}
_DIRECTION_KEYS = _COMMON_KEYS | {
    "temperature",
    "class_order",
    "transform",
    "argmax_preserving",
    "tie_policy",
}
_PATH_KEYS = _COMMON_KEYS | {
    "path_quality_scale",
    "path_quality_shift",
    "bad_path_temperature",
    "bad_path_bias",
}


def _fail(context: str, message: str) -> None:
    raise RuntimeError(f"[{context}_INVALID] {message}")


def _finite(value: Any, *, context: str, field: str) -> float:
    if isinstance(value, bool):
        _fail(context, f"{field} must be finite numeric")
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"[{context}_INVALID] {field} must be finite numeric"
        ) from exc
    if not math.isfinite(observed):
        _fail(context, f"{field} must be finite numeric")
    return observed


def _absolute_plain_path(
    value: Any,
    *,
    context: str,
    field: str,
) -> Path:
    raw = str(value or "")
    path = Path(raw).expanduser()
    if not raw or not path.is_absolute() or path != Path(raw):
        _fail(context, f"{field} must be an absolute canonical path")
    return path


def require_model_native_calibration_metadata(
    value: Mapping[str, Any] | Any,
    *,
    head: str,
    context: str,
) -> dict[str, Any]:
    """Reject partial, anonymous or non-canonical calibration declarations."""

    if not isinstance(value, Mapping):
        _fail(context, "calibration must be an object")
    observed = dict(value)
    if head == "direction":
        if "bias" in observed:
            _fail(
                context,
                "direction class bias/remap is forbidden; stale calibration schema",
            )
        expected_keys = _DIRECTION_KEYS
        expected_version = DIRECTION_CALIBRATION_VERSION
    elif head == "path":
        expected_keys = _PATH_KEYS
        expected_version = PATH_CALIBRATION_VERSION
    else:
        _fail(context, f"unknown calibration head {head!r}")
    if set(observed) != expected_keys:
        _fail(
            context,
            "calibration fields differ: "
            f"missing={sorted(expected_keys - set(observed))} "
            f"unexpected={sorted(set(observed) - expected_keys)}",
        )
    if observed["enabled"] is not True or observed["version"] != expected_version:
        _fail(context, "enabled/version mismatch")

    fitted_on = observed["fitted_on_split"]
    if fitted_on not in CALIBRATION_FIT_SPLITS:
        _fail(context, "fit split is not held out")
    for field in ("fitted_rows", "min_fit_rows"):
        raw = observed[field]
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
            _fail(context, f"{field} must be a positive integer")
    if observed["fitted_rows"] < observed["min_fit_rows"]:
        _fail(context, "fitted_rows is below min_fit_rows")
    if not isinstance(observed["model"], str) or not observed["model"].strip():
        _fail(context, "model must be non-empty")
    try:
        require_entry_run_id(observed["run_id"])
    except Exception as exc:
        raise RuntimeError(f"[{context}_INVALID] run_id is invalid") from exc
    try:
        fitted_at = datetime.fromisoformat(
            str(observed["fitted_at_utc"]).replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise RuntimeError(f"[{context}_INVALID] fitted_at_utc is invalid") from exc
    if fitted_at.utcoffset() is None or fitted_at.utcoffset().total_seconds() != 0:
        _fail(context, "fitted_at_utc must be UTC")

    source_dir = _absolute_plain_path(
        observed["source_bundle_dir"],
        context=context,
        field="source_bundle_dir",
    )
    predictions = _absolute_plain_path(
        observed["predictions_path"],
        context=context,
        field="predictions_path",
    )
    report = _absolute_plain_path(
        observed["prediction_report_path"],
        context=context,
        field="prediction_report_path",
    )
    if not source_dir.name or _PREDICTION_NAME_RE.fullmatch(predictions.name) is None:
        _fail(context, "prediction artifact name is not immutable/canonical")
    if _REPORT_NAME_RE.fullmatch(report.name) is None:
        _fail(context, "prediction report name is not immutable/canonical")
    for field in (
        "source_bundle_metadata_sha256",
        "predictions_sha256",
        "prediction_report_sha256",
    ):
        if _SHA256_RE.fullmatch(str(observed[field])) is None:
            _fail(context, f"{field} is not SHA-256")

    if head == "direction":
        temperature = _finite(
            observed["temperature"], context=context, field="temperature"
        )
        if temperature <= 0.0:
            _fail(context, "direction temperature must be strictly positive")
        if observed["class_order"] != ["LONG", "SHORT", "FLAT"]:
            _fail(context, "direction class order mismatch")
        if observed["transform"] != DIRECTION_CALIBRATION_TRANSFORM:
            _fail(context, "direction calibration transform mismatch")
        if observed["argmax_preserving"] is not True:
            _fail(context, "direction calibration must preserve raw argmax")
        if observed["tie_policy"] != DIRECTION_CALIBRATION_TIE_POLICY:
            _fail(context, "direction calibration tie policy mismatch")
    else:
        path_scale = _finite(
            observed["path_quality_scale"],
            context=context,
            field="path_quality_scale",
        )
        bad_temperature = _finite(
            observed["bad_path_temperature"],
            context=context,
            field="bad_path_temperature",
        )
        _finite(
            observed["path_quality_shift"],
            context=context,
            field="path_quality_shift",
        )
        _finite(
            observed["bad_path_bias"],
            context=context,
            field="bad_path_bias",
        )
        if path_scale <= 0.0 or bad_temperature <= 0.0:
            _fail(context, "path calibration scales must be positive")
    return observed


__all__ = [
    "CALIBRATION_EVENT_PREFIX",
    "CALIBRATION_FIT_SPLITS",
    "DIRECTION_CALIBRATION_TIE_POLICY",
    "DIRECTION_CALIBRATION_TRANSFORM",
    "DIRECTION_CALIBRATION_VERSION",
    "IMMUTABLE_CALIBRATION_EVENT_SCHEMA_VERSION",
    "PATH_CALIBRATION_VERSION",
    "require_model_native_calibration_metadata",
]
