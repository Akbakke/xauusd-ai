"""Immutable offline calibration and recomputed TEST sizing proof.

The only learned transform is:

``position_size_logit -> calibrated capacity fraction -> reference multiplier
                       -> floor-to-step integer additional units``

Calibration is fit on exact VAL-only pre-calibration evidence. TEST admission is not granted by a
self-reported PASS: this module independently derives and then re-derives one
hash-bound canonical TEST row source from the exact candidate predictions,
dataset manifests, SourceTape, and frozen evaluation bundle. The immutable
0.98/Wilson/class precision policy is recomputed from those same TEST rows.
No broker, live, serve-parity, adoption, or caller compatibility evidence is
accepted anywhere in this contract.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import t as student_t

from gx1.contracts.immutable_event_authority_v1 import (
    ImmutableEventAuthorityError,
    require_newest_immutable_event,
)
from gx1.replay.source_tape_v1 import SourceTape
from gx1.scripts.entry_candidate_prediction_evidence_v1 import (
    resolve_and_validate_prediction_evidence,
)


MODEL_NATIVE_SIZING_CALIBRATION_SCHEMA_VERSION = (
    "entry_model_native_sizing_calibration_v4"
)
MODEL_NATIVE_SIZING_BUNDLE_CALIBRATION_SCHEMA_VERSION = (
    "entry_model_native_sizing_bundle_calibration_v2"
)
MODEL_NATIVE_SIZING_OOS_PROOF_SCHEMA_VERSION = (
    "entry_model_native_sizing_oos_proof_v6"
)
MODEL_NATIVE_SIZING_OOS_SOURCE_SCHEMA_VERSION = (
    "entry_model_native_sizing_oos_source_v4"
)
MODEL_NATIVE_SIZING_TRANSFORM_VERSION = (
    "monotone_logistic_available_margin_capacity_fraction_v2"
)
MODEL_NATIVE_SIZING_RISK_POLICY_SCHEMA_VERSION = (
    "entry_model_native_sizing_risk_policy_v2"
)
MODEL_NATIVE_SIZING_FIT_SCOPE = "EXACT_VAL_ONLY_PRE_CALIBRATION"
MODEL_NATIVE_SIZING_FIT_SPLITS = ("val",)
MODEL_NATIVE_SIZING_HOLDOUT_SPLIT = "test"
MODEL_NATIVE_SIZING_OOS_SCOPE = (
    "FULL_TEST_RUNTIME_AUTHORITATIVE_SIZING_AND_DIRECTION_EDGE_PROOF"
)
MODEL_NATIVE_SIZING_HEAD_VARIATION_EPSILON = 1e-8
MODEL_NATIVE_SIZING_MAX_CAPACITY_FRACTION = 0.75
MODEL_NATIVE_SIZING_MAX_ACCOUNT_MARGIN_FRACTION = 0.10
MODEL_NATIVE_SIZING_MAX_GROSS_XAU_UNITS = 1_000
MODEL_NATIVE_SIZING_MAX_ACCOUNT_FLOATING_DRAWDOWN_BPS = 250.0
MODEL_NATIVE_SIZING_MAX_OOS_CUMULATIVE_DRAWDOWN_BPS = 250.0
MODEL_NATIVE_SIZING_MIN_OOS_ORDERS = 100
MODEL_NATIVE_SIZING_MIN_FIT_ROWS_PER_SPLIT = 256
MODEL_NATIVE_SIZING_MIN_OOS_WEEK_BLOCKS = 12
MODEL_NATIVE_SIZING_MIN_OOS_MONTH_BLOCKS = 6
MODEL_NATIVE_SIZING_MIN_SLICE_WEEK_BLOCKS = 6
_FOUNDATION_AUDIT_POLICY_SCHEMA_RE = re.compile(
    r"entry_foundation_audit_policy_v[1-9][0-9]*"
)
_DIRECTION_EDGE_POLICY_FIELDS = (
    "min_direction_accuracy",
    "min_balanced_accuracy",
    "min_trade_direction_precision",
    "min_class_precision",
    "wilson_confidence_level",
    "wilson_z_score",
    "min_trade_rows",
    "min_prediction_rows_per_class",
    "min_trade_precision_wilson_lower",
    "min_class_precision_wilson_lower",
)
_DIRECTION_EDGE_POLICY_FLOORS = {
    "min_direction_accuracy": 0.90,
    "min_balanced_accuracy": 0.90,
    "min_trade_direction_precision": 0.98,
    "min_class_precision": 0.95,
    "wilson_confidence_level": 0.95,
    "wilson_z_score": 1.959963984540054,
    "min_trade_rows": 200,
    "min_prediction_rows_per_class": 100,
    "min_trade_precision_wilson_lower": 0.95,
    "min_class_precision_wilson_lower": 0.90,
}

_CALIBRATION_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_CALIBRATION"
_OOS_PROOF_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_OOS_PROOF"
_OOS_SOURCE_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_OOS_SOURCE"
_PREDICTION_REPORT_EVENT_PREFIX = "ENTRY_CANDIDATE_SELECTIVE_EDGE"
_IMMUTABLE_JSON_RE = re.compile(
    r"^(?P<prefix>[A-Z0-9_]+)_(?P<stamp>[0-9]{8}T[0-9]{12}Z)\.json$"
)
_IMMUTABLE_SOURCE_STAMP_RE = re.compile(r"[0-9]{8}T[0-9]{12}Z")
_CALIBRATION_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "json_path",
        "fit_scope",
        "fit_splits",
        "holdout_split",
        "source_head",
        "transform_version",
        "monotonic_direction",
        "instrument_constraints",
        "parameters",
        "lineage",
        "fit_prediction_provenance",
        "fit_contract",
    }
)
_BUNDLE_CALIBRATION_KEYS = frozenset(
    {
        "schema_version",
        "source_head",
        "transform_version",
        "fit_scope",
        "calibration_artifact",
        "risk_policy",
    }
)
_INSTRUMENT_KEYS = frozenset(
    {
        "constraint_source",
        "instrument",
        "account_currency",
        "quote_currency",
        "unit_step",
        "minimum_order_units",
        "maximum_gross_xau_units",
        "margin_rate",
    }
)
_PARAMETER_KEYS = frozenset(
    {
        "logit_center",
        "logit_temperature",
        "minimum_capacity_fraction",
        "reference_capacity_fraction",
        "maximum_capacity_fraction",
    }
)
_LINEAGE_KEYS = frozenset(
    {
        "dataset_manifest_path",
        "dataset_manifest_sha256",
        "fit_predictions_path",
        "fit_predictions_sha256",
        "model_checkpoint_path",
        "model_checkpoint_sha256",
    }
)
_PREDICTION_PROVENANCE_KEYS = frozenset(
    {
        "prediction_report_artifact",
        "bundle_dir",
        "dataset_dir",
        "dataset_split_bindings",
    }
)
_DATASET_SPLIT_BINDING_KEYS = frozenset(
    {
        "manifest_path",
        "manifest_sha256",
        "parquet_path",
        "parquet_sha256",
    }
)
_EVALUATION_BUNDLE_KEYS = frozenset(
    {
        "bundle_dir",
        "bundle_metadata_path",
        "bundle_metadata_sha256",
        "master_transformer_lock_path",
        "master_transformer_lock_sha256",
        "model_state_dict_path",
        "model_state_dict_sha256",
    }
)
_RUNTIME_CONSTRAINT_KEYS = frozenset(
    {
        "instrument",
        "account_currency",
        "account_equity",
        "account_balance",
        "account_floating_drawdown_bps",
        "margin_available",
        "margin_used",
        "mark_price",
        "margin_rate",
        "unit_step",
        "minimum_order_units",
        "maximum_gross_xau_units",
        "current_xau_abs_units",
        "fact_provenance_mode",
    }
)
MODEL_NATIVE_SIZING_RUNTIME_CONSTRAINT_KEYS = tuple(
    sorted(_RUNTIME_CONSTRAINT_KEYS)
)
_SOURCE_BINDING_NAMES = (
    "oos_rows",
)
MODEL_NATIVE_SIZING_OOS_PREDICTION_COLUMNS = frozenset(
    {
        "time",
        "split",
        "model",
        "position_size_logit",
        "model_direction_index",
        "target_direction_index",
        "session",
        "vol_regime",
    }
)
MODEL_NATIVE_SIZING_OOS_OUTCOME_COLUMNS = frozenset(
    {
        "time",
        "account_equity",
        "account_balance",
        "account_floating_drawdown_bps",
        "margin_available",
        "margin_used",
        "current_xau_abs_units",
        "mark_price",
        "entry_bid",
        "entry_ask",
        "exit_bid",
        "exit_ask",
        "fact_provenance_mode",
    }
)
MODEL_NATIVE_SIZING_OOS_REPLAY_COLUMNS = frozenset(
    {
        "time",
        "position_size_logit",
        "model_direction_index",
        "calibrated_size_fraction",
        "applied_size_multiplier",
        "capacity_units",
        "reference_pre_round_units",
        "pre_round_units",
        "units",
        "authorized_order",
        "no_order_reason",
    }
)
MODEL_NATIVE_SIZING_OOS_PROVENANCE_COLUMNS = frozenset(
    {
        "bundle_metadata_sha256",
        "model_state_dict_sha256",
        "test_predictions_sha256",
        "source_tape_sha256",
        "reference_row_id",
    }
)
MODEL_NATIVE_SIZING_OOS_ROW_COLUMNS = frozenset(
    MODEL_NATIVE_SIZING_OOS_PREDICTION_COLUMNS
    | MODEL_NATIVE_SIZING_OOS_OUTCOME_COLUMNS
    | MODEL_NATIVE_SIZING_OOS_REPLAY_COLUMNS
    | MODEL_NATIVE_SIZING_OOS_PROVENANCE_COLUMNS
)
_PROOF_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "failures",
        "evaluation_scope",
        "evaluation_split",
        "calibration_artifact_sha256",
        "risk_policy",
        "source_bindings",
        "test_prediction_provenance",
        "evaluation_bundle",
        "oos_source_artifact",
        "full_test_coverage",
        "position_size_head_liveness",
        "monotonicity",
        "exposure_bounds",
        "drawdown_bounds",
        "paired_oos_utility",
        "account_capacity_grid",
        "direction_edge_policy",
        "direction_edge_admission",
        "direction_invariance",
    }
)
_OOS_SOURCE_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "failures",
        "calibration_artifact_sha256",
        "test_predictions",
        "test_prediction_provenance",
        "evaluation_bundle",
        "source_tape",
        "reference_account_policy",
        "source_bindings",
    }
)
_SOURCE_TAPE_KEYS = frozenset(
    {"path", "sha256", "dataset_test_manifest_path", "dataset_test_manifest_sha256"}
)


class ModelNativeSizingContractError(RuntimeError):
    """Immutable learned sizing evidence is absent, stale, or inconsistent."""


def sizing_offline_instrument_constraints_metadata() -> dict[str, Any]:
    """Code-owned XAUUSD research constraints; never broker observations."""

    return {
        "constraint_source": "code_owned_offline_xauusd_v1",
        "instrument": "XAU_USD",
        "account_currency": "USD",
        "quote_currency": "USD",
        "unit_step": 1,
        "minimum_order_units": 1,
        "maximum_gross_xau_units": MODEL_NATIVE_SIZING_MAX_GROSS_XAU_UNITS,
        "margin_rate": 0.05,
    }


def sizing_risk_policy_metadata() -> dict[str, Any]:
    return {
        "schema_version": MODEL_NATIVE_SIZING_RISK_POLICY_SCHEMA_VERSION,
        "maximum_capacity_fraction": MODEL_NATIVE_SIZING_MAX_CAPACITY_FRACTION,
        "maximum_account_margin_fraction": (
            MODEL_NATIVE_SIZING_MAX_ACCOUNT_MARGIN_FRACTION
        ),
        "maximum_gross_xau_units": MODEL_NATIVE_SIZING_MAX_GROSS_XAU_UNITS,
        "maximum_account_floating_drawdown_bps": (
            MODEL_NATIVE_SIZING_MAX_ACCOUNT_FLOATING_DRAWDOWN_BPS
        ),
        "maximum_oos_cumulative_drawdown_bps": (
            MODEL_NATIVE_SIZING_MAX_OOS_CUMULATIVE_DRAWDOWN_BPS
        ),
        "minimum_oos_evaluated_orders": MODEL_NATIVE_SIZING_MIN_OOS_ORDERS,
        "utility_confidence_level": 0.95,
        "utility_interval": "two_sided_student_t_on_independent_time_blocks",
        "utility_formula": (
            "paired_account_net_pnl_bps_after_exact_source_tape_bid_ask_costs;"
            "admission=actual_1_unit_and_equal_total_continuous_allocation_controls_"
            "each_with_ISO_week_month_session_regime_block_lower95_across_all_"
            "code_owned_account_capacity_scenarios;"
            "dense_row_iid_ci_is_diagnostic_only"
        ),
        "historical_control_units": 1,
        "allocation_control": "capacity_capped_equal_total_continuous_units",
        "rounded_allocation_control_role": "diagnostic_only",
        "execution_or_live_authority": False,
        "required_final_evidence": (
            "same_candidate_same_bundle_entry_m5_exit_m1_full_test_replay"
        ),
        "capacity_formula": (
            "floor_step(min((min(margin_available,max(0,equity*0.10-margin_used))"
            "/(mark_price*margin_rate)),maximum_gross_xau_units-current_xau_abs_units))"
        ),
        "account_floating_drawdown_formula": (
            "max(0,(account_balance-account_equity)/account_balance)*10000"
        ),
        "oos_cumulative_drawdown_formula": (
            "max(running_peak(cumulative_pnl_bps)-cumulative_pnl_bps)"
        ),
        "rounding_mode": "floor_to_unit_step",
        "external_runtime_fact_inputs_allowed": False,
    }


def sizing_oos_reference_account_policy_metadata() -> dict[str, Any]:
    """Code-owned account/capacity grid for paired historical OOS sizing."""

    return {
        "schema_version": "historical_reference_account_grid_v1",
        "fact_provenance_mode": "canonical_oos_reference",
        "account_currency": "USD",
        "canonical_row_scenario": "medium",
        "scenario_order": ["small", "medium", "large"],
        "scenarios": {
            "small": {
                "account_equity": 5_000.0,
                "account_balance": 5_000.0,
                "account_floating_drawdown_bps": 0.0,
                "margin_available": 500.0,
                "margin_used": 0.0,
                "current_xau_abs_units": 0.0,
            },
            "medium": {
                "account_equity": 10_000.0,
                "account_balance": 10_000.0,
                "account_floating_drawdown_bps": 0.0,
                "margin_available": 1_000.0,
                "margin_used": 0.0,
                "current_xau_abs_units": 0.0,
            },
            "large": {
                "account_equity": 50_000.0,
                "account_balance": 50_000.0,
                "account_floating_drawdown_bps": 0.0,
                "margin_available": 5_000.0,
                "margin_used": 0.0,
                "current_xau_abs_units": 0.0,
            },
        },
        "row_simulation_mode": "independent_full_test_rows",
        "external_runtime_or_broker_facts_allowed": False,
        "price_path_authority": "hash_bound_source_tape_bid_ask",
    }


def sizing_fit_contract_metadata() -> dict[str, Any]:
    return {
        "schema_version": "entry_model_native_sizing_fit_contract_v1",
        "optimizer": "scipy_least_squares_trf_deterministic",
        "objective": "row_mse_monotone_logistic_capacity_fraction",
        "source_head": "position_size_logit",
        "target": "y_position_size_target_times_maximum_capacity_fraction",
        "required_splits": list(MODEL_NATIVE_SIZING_FIT_SPLITS),
        "required_models": ["candidate"],
        "minimum_rows_per_split": MODEL_NATIVE_SIZING_MIN_FIT_ROWS_PER_SPLIT,
        "max_nfev": 2_000,
        "ftol": 1e-12,
        "xtol": 1e-12,
        "gtol": 1e-12,
        "parameter_recompute_abs_tolerance": 1e-12,
    }


def fit_monotone_sizing_parameters(
    logits: np.ndarray, targets: np.ndarray
) -> dict[str, float]:
    """Shared deterministic VAL-only fit used by producer and verifier."""

    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    high = MODEL_NATIVE_SIZING_MAX_CAPACITY_FRACTION
    if (
        logits.ndim != 1
        or targets.shape != logits.shape
        or not np.isfinite(logits).all()
        or not np.isfinite(targets).all()
        or float(np.std(logits)) <= MODEL_NATIVE_SIZING_HEAD_VARIATION_EPSILON
        or np.any((targets < 0.0) | (targets > high))
    ):
        _fail("SIZING_FIT", "VAL sizing arrays are invalid or inert")
    spread = max(float(np.std(logits)), 1e-3)
    lower = np.asarray(
        [0.0, float(np.min(logits) - 10.0 * spread), math.log(1e-4)],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            high - 1e-6,
            float(np.max(logits) + 10.0 * spread),
            math.log(max(100.0 * spread, 1.0)),
        ],
        dtype=np.float64,
    )

    def residual(theta: np.ndarray) -> np.ndarray:
        low, center, log_temperature = theta
        scaled = np.clip(
            (logits - center) / math.exp(log_temperature), -80.0, 80.0
        )
        sigmoid = 1.0 / (1.0 + np.exp(-scaled))
        return low + (high - low) * sigmoid - targets

    result = least_squares(
        residual,
        x0=np.asarray(
            [0.05, float(np.median(logits)), math.log(spread)], dtype=np.float64
        ),
        bounds=(lower, upper),
        method="trf",
        max_nfev=2_000,
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    if not result.success or not np.isfinite(result.x).all():
        _fail("SIZING_FIT", f"deterministic fit failed: {result.message}")
    low, center, log_temperature = (float(value) for value in result.x)
    return {
        "logit_center": center,
        "logit_temperature": math.exp(log_temperature),
        "minimum_capacity_fraction": low,
        "reference_capacity_fraction": (low + high) / 2.0,
        "maximum_capacity_fraction": high,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fail(context: str, detail: str) -> None:
    raise ModelNativeSizingContractError(
        f"[{str(context).strip() or 'MODEL_NATIVE_SIZING'}_INVALID] {detail}"
    )


def _exact_keys(
    value: Mapping[str, Any] | Any,
    expected: frozenset[str],
    *,
    context: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(context, "expected an object")
    observed = dict(value)
    missing = sorted(expected - set(observed))
    unexpected = sorted(set(observed) - expected)
    if missing or unexpected:
        _fail(context, f"exact keys mismatch: missing={missing} unexpected={unexpected}")
    return observed


def _finite(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        _fail(context, "boolean is not numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        _fail(context, f"not numeric: {value!r}")
    if not math.isfinite(parsed):
        _fail(context, f"not finite: {value!r}")
    return parsed


def _strict_int(value: Any, *, context: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail(context, f"exact JSON integer required: {value!r}")
    try:
        parsed = operator.index(value)
    except TypeError:
        _fail(context, f"exact integer required: {value!r}")
    parsed = int(parsed)
    if parsed < minimum:
        _fail(context, f"integer={parsed} must be >= {minimum}")
    return parsed


def _sha(value: Any, *, context: str) -> str:
    parsed = str(value or "").strip().lower()
    if len(parsed) != 64 or any(ch not in "0123456789abcdef" for ch in parsed):
        _fail(context, "not an exact SHA-256")
    return parsed


def _utc(value: Any, *, context: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        _fail(context, f"invalid UTC timestamp: {value!r}")
    offset = parsed.utcoffset() if parsed.tzinfo is not None else None
    if offset is None or offset.total_seconds() != 0.0:
        _fail(context, "timestamp must be timezone-aware UTC")
    return parsed


def require_immutable_json_binding(
    binding: Mapping[str, Any] | Any,
    *,
    event_prefix: str,
    context: str,
    verify_file: bool,
) -> dict[str, str]:
    observed = _exact_keys(
        binding, frozenset({"json_path", "sha256"}), context=context
    )
    path = Path(str(observed["json_path"] or "")).expanduser()
    if not path.is_absolute():
        _fail(context, f"json_path must be absolute: {path}")
    match = _IMMUTABLE_JSON_RE.fullmatch(path.name)
    if match is None or match.group("prefix") != event_prefix:
        _fail(context, f"json_path must be immutable {event_prefix}_<microstamp>.json")
    expected_sha = _sha(observed["sha256"], context=f"{context}.sha256")
    resolved = path.resolve()
    if verify_file:
        if not resolved.is_file():
            _fail(context, f"bound file missing: {resolved}")
        try:
            require_newest_immutable_event(resolved, event_prefix)
        except ImmutableEventAuthorityError as exc:
            _fail(context, f"bound event is not newest immutable family authority: {exc}")
        actual = sha256_file(resolved)
        if actual != expected_sha:
            _fail(context, f"bound file hash mismatch: declared={expected_sha} actual={actual}")
    return {"json_path": str(resolved), "sha256": expected_sha}


def model_native_sizing_bundle_calibration_metadata(
    *,
    calibration_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind exact VAL calibration into a frozen offline candidate bundle."""

    binding = require_immutable_json_binding(
        calibration_artifact,
        event_prefix=_CALIBRATION_EVENT_PREFIX,
        context="SIZING_BUNDLE_CALIBRATION.calibration_artifact",
        verify_file=False,
    )
    return {
        "schema_version": MODEL_NATIVE_SIZING_BUNDLE_CALIBRATION_SCHEMA_VERSION,
        "source_head": "position_size_logit",
        "transform_version": MODEL_NATIVE_SIZING_TRANSFORM_VERSION,
        "fit_scope": MODEL_NATIVE_SIZING_FIT_SCOPE,
        "calibration_artifact": binding,
        "risk_policy": sizing_risk_policy_metadata(),
    }


def require_model_native_sizing_bundle_calibration(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Require the exact offline calibration declaration bound in a bundle."""

    observed = _exact_keys(value, _BUNDLE_CALIBRATION_KEYS, context=context)
    expected = model_native_sizing_bundle_calibration_metadata(
        calibration_artifact=observed["calibration_artifact"]
    )
    if observed != expected:
        mismatched = sorted(
            key
            for key, expected_value in expected.items()
            if observed.get(key) != expected_value
        )
        _fail(context, f"bundle calibration mismatch: {mismatched}")
    return observed


def _source_binding(
    binding: Mapping[str, Any] | Any,
    *,
    context: str,
    verify_file: bool,
) -> dict[str, str]:
    observed = _exact_keys(binding, frozenset({"path", "sha256"}), context=context)
    path = Path(str(observed["path"] or "")).expanduser()
    if not path.is_absolute() or _IMMUTABLE_SOURCE_STAMP_RE.search(path.name) is None:
        _fail(context, "source path must be absolute and microsecond-stamped")
    expected_sha = _sha(observed["sha256"], context=f"{context}.sha256")
    resolved = path.resolve()
    if verify_file:
        if not resolved.is_file():
            _fail(context, f"source file missing: {resolved}")
        actual = sha256_file(resolved)
        if actual != expected_sha:
            _fail(context, f"source hash mismatch: declared={expected_sha} actual={actual}")
    return {"path": str(resolved), "sha256": expected_sha}


def _lineage_file(
    path_raw: Any,
    sha_raw: Any,
    *,
    context: str,
    verify_file: bool,
) -> None:
    path = Path(str(path_raw or "")).expanduser()
    if not path.is_absolute():
        _fail(context, "lineage path must be absolute")
    expected = _sha(sha_raw, context=f"{context}.sha256")
    if verify_file:
        if not path.is_file() or sha256_file(path) != expected:
            _fail(context, "lineage file missing or hash mismatch")


def _json_file(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail(context, f"JSON unreadable: {path}: {exc}")
    if not isinstance(payload, dict):
        _fail(context, f"JSON root must be an object: {path}")
    return payload


def require_sizing_prediction_provenance(
    value: Mapping[str, Any] | Any,
    *,
    predictions_binding: Mapping[str, Any],
    expected_splits: tuple[str, ...],
    expected_stage: str,
    context: str,
    verify_files: bool,
) -> dict[str, Any]:
    """Revalidate the canonical prediction report/parquet/bundle lineage."""

    observed = _exact_keys(value, _PREDICTION_PROVENANCE_KEYS, context=context)
    report = require_immutable_json_binding(
        observed["prediction_report_artifact"],
        event_prefix=_PREDICTION_REPORT_EVENT_PREFIX,
        context=f"{context}.prediction_report_artifact",
        verify_file=verify_files,
    )
    bundle_dir = Path(str(observed["bundle_dir"] or "")).expanduser()
    dataset_dir = Path(str(observed["dataset_dir"] or "")).expanduser()
    if not bundle_dir.is_absolute() or not dataset_dir.is_absolute():
        _fail(context, "bundle_dir and dataset_dir must be absolute")
    split_bindings_raw = observed["dataset_split_bindings"]
    if not isinstance(split_bindings_raw, Mapping) or set(split_bindings_raw) != set(
        expected_splits
    ):
        _fail(context, "dataset split bindings must exactly match prediction splits")
    split_bindings: dict[str, dict[str, str]] = {}
    for split in expected_splits:
        row = _exact_keys(
            split_bindings_raw[split],
            _DATASET_SPLIT_BINDING_KEYS,
            context=f"{context}.dataset_split_bindings.{split}",
        )
        manifest_path = Path(str(row["manifest_path"] or "")).expanduser()
        parquet_path = Path(str(row["parquet_path"] or "")).expanduser()
        if not manifest_path.is_absolute() or not parquet_path.is_absolute():
            _fail(context, "dataset manifest/parquet paths must be absolute")
        if (
            manifest_path.resolve() != manifest_path
            or parquet_path.resolve() != parquet_path
            or manifest_path.is_symlink()
            or parquet_path.is_symlink()
            or any("latest" in part.lower() for part in manifest_path.parts)
            or any("latest" in part.lower() for part in parquet_path.parts)
            or not manifest_path.name.endswith(f"_{split}.manifest.json")
            or not parquet_path.name.endswith(f"_{split}.parquet")
        ):
            _fail(context, f"{split} dataset artifact identity is not immutable")
        split_bindings[split] = {
            "manifest_path": str(manifest_path.resolve()),
            "manifest_sha256": _sha(
                row["manifest_sha256"],
                context=f"{context}.{split}.manifest_sha256",
            ),
            "parquet_path": str(parquet_path.resolve()),
            "parquet_sha256": _sha(
                row["parquet_sha256"],
                context=f"{context}.{split}.parquet_sha256",
            ),
        }
    canonical = {
        "prediction_report_artifact": report,
        "bundle_dir": str(bundle_dir.resolve()),
        "dataset_dir": str(dataset_dir.resolve()),
        "dataset_split_bindings": split_bindings,
    }
    if canonical != observed:
        _fail(context, "prediction provenance path canonicalization mismatch")
    source = _source_binding(
        predictions_binding,
        context=f"{context}.predictions",
        verify_file=verify_files,
    )
    if not verify_files:
        return observed
    if not bundle_dir.is_dir() or not dataset_dir.is_dir():
        _fail(context, "prediction provenance bundle/dataset directory is missing")
    try:
        authoritative, report_payload, evidence = resolve_and_validate_prediction_evidence(
            Path(source["path"]),
            expected_sha256=source["sha256"],
            prediction_report_path=Path(report["json_path"]),
            bundle_dir=bundle_dir,
            dataset_dir=dataset_dir,
            expected_stage=expected_stage,
            expected_splits=expected_splits,
            expected_model="candidate",
        )
    except Exception as exc:
        _fail(context, f"canonical prediction evidence validation failed: {exc}")
    if tuple(str(item) for item in evidence.get("splits") or []) != expected_splits:
        _fail(context, "prediction evidence split set is not exact")
    expected_authority = expected_stage == "runtime_authoritative"
    if (
        evidence.get("evidence_stage") != expected_stage
        or evidence.get("authoritative") is not expected_authority
        or evidence.get("runtime_head_evidence_authoritative")
        is not expected_authority
    ):
        _fail(context, "prediction evidence stage/authority is not exact")
    if list(evidence.get("models") or []) != ["candidate"]:
        _fail(context, "prediction evidence model set must be exactly candidate")
    if authoritative.resolve() != Path(source["path"]).resolve():
        _fail(context, "canonical prediction path differs from source binding")
    if str(evidence.get("sha256") or "").lower() != source["sha256"]:
        _fail(context, "prediction declaration hash differs from source binding")
    checkpoint = bundle_dir.resolve() / "model_state_dict.pt"
    state_sha = _sha(
        evidence.get("model_state_dict_sha256"),
        context=f"{context}.model_state_dict_sha256",
    )
    if not checkpoint.is_file() or sha256_file(checkpoint) != state_sha:
        _fail(context, "prediction bundle model_state_dict.pt hash mismatch")
    report_dataset = report_payload.get("dataset_signal_contract")
    report_splits = (
        report_dataset.get("splits") if isinstance(report_dataset, Mapping) else None
    )
    if not isinstance(report_splits, Mapping) or set(report_splits) != set(
        expected_splits
    ):
        _fail(context, "prediction report dataset split contract mismatch")
    prediction_columns = [
        "time",
        "split",
        "model",
        "position_size_logit",
        "position_size_pred",
    ]
    if expected_stage == "pre_calibration":
        prediction_columns.append("y_position_size_target")
    prediction_frame = pd.read_parquet(authoritative, columns=prediction_columns)
    sizing_logits = pd.to_numeric(
        prediction_frame["position_size_logit"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    sizing_predictions = pd.to_numeric(
        prediction_frame["position_size_pred"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    expected_predictions = 1.0 / (
        1.0 + np.exp(-np.clip(sizing_logits, -80.0, 80.0))
    )
    if (
        not np.isfinite(sizing_logits).all()
        or not np.isfinite(sizing_predictions).all()
        or not np.allclose(
            sizing_predictions, expected_predictions, rtol=1e-5, atol=1e-6
        )
    ):
        _fail(context, "position_size_pred differs from sigmoid(position_size_logit)")
    for split in expected_splits:
        bound = split_bindings[split]
        manifest_path = Path(bound["manifest_path"])
        parquet_path = Path(bound["parquet_path"])
        if manifest_path.parent != dataset_dir.resolve() or parquet_path.parent != dataset_dir.resolve():
            _fail(context, f"{split} dataset files are outside proven dataset_dir")
        if (
            not manifest_path.is_file()
            or not parquet_path.is_file()
            or sha256_file(manifest_path) != bound["manifest_sha256"]
            or sha256_file(parquet_path) != bound["parquet_sha256"]
        ):
            _fail(context, f"{split} dataset manifest/parquet hash mismatch")
        manifest = _json_file(
            manifest_path,
            context=f"{context}.{split}.dataset_manifest",
        )
        declared_parquet = Path(
            str(manifest.get("output_data_path") or "")
        ).expanduser()
        if declared_parquet != parquet_path:
            _fail(context, f"{split} manifest output_data_path mismatch")
        report_row = report_splits[split]
        if not isinstance(report_row, Mapping):
            _fail(context, f"prediction report {split} manifest binding mismatch")
        for kind, expected_path in (
            ("manifest", manifest_path),
            ("parquet", parquet_path),
        ):
            if (
                Path(str(report_row.get(f"{kind}_path") or "")).expanduser()
                != expected_path
                or str(report_row.get(f"{kind}_sha256") or "").lower()
                != bound[f"{kind}_sha256"]
            ):
                _fail(
                    context,
                    f"prediction report {split} {kind} binding mismatch",
                )
        if sha256_file(manifest_path) != bound["manifest_sha256"]:
            _fail(context, f"{split} manifest changed during validation")
        dataset_times = pd.to_datetime(
            pd.read_parquet(parquet_path, columns=["time"])["time"],
            utc=True,
            errors="coerce",
        )
        prediction_times = pd.to_datetime(
            prediction_frame.loc[
                (prediction_frame["split"].astype(str) == split)
                & (prediction_frame["model"].astype(str) == "candidate"),
                "time",
            ],
            utc=True,
            errors="coerce",
        )
        if (
            dataset_times.isna().any()
            or prediction_times.isna().any()
            or dataset_times.duplicated().any()
            or prediction_times.duplicated().any()
            or not dataset_times.reset_index(drop=True).equals(
                prediction_times.reset_index(drop=True)
            )
        ):
            _fail(context, f"prediction rows do not exactly cover {split} dataset rows")
        if expected_stage == "pre_calibration":
            if "y_position_size_target" not in prediction_frame.columns:
                _fail(context, "fit prediction evidence lacks y_position_size_target")
            dataset_targets = pd.to_numeric(
                pd.read_parquet(
                    parquet_path, columns=["y_position_size_target"]
                )["y_position_size_target"],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            prediction_targets = pd.to_numeric(
                prediction_frame.loc[
                    (prediction_frame["split"].astype(str) == split)
                    & (prediction_frame["model"].astype(str) == "candidate"),
                    "y_position_size_target",
                ],
                errors="coerce",
            ).to_numpy(dtype=np.float64)
            if (
                not np.isfinite(dataset_targets).all()
                or not np.isfinite(prediction_targets).all()
                or not np.array_equal(dataset_targets, prediction_targets)
            ):
                _fail(context, f"fit target differs from exact {split} dataset target")
    return observed


def require_sizing_evaluation_bundle(
    value: Mapping[str, Any] | Any,
    *,
    calibration: Mapping[str, Any],
    context: str,
    verify_files: bool,
) -> dict[str, Any]:
    """Bind proof/adoption to one exact bundle metadata, lock, and checkpoint."""

    observed = _exact_keys(value, _EVALUATION_BUNDLE_KEYS, context=context)
    bundle_dir = Path(str(observed["bundle_dir"] or "")).expanduser()
    if not bundle_dir.is_absolute():
        _fail(context, "evaluation bundle_dir must be absolute")
    bundle_dir = bundle_dir.resolve()
    paths = {
        "bundle_metadata": Path(str(observed["bundle_metadata_path"] or "")).expanduser(),
        "master_transformer_lock": Path(
            str(observed["master_transformer_lock_path"] or "")
        ).expanduser(),
        "model_state_dict": Path(str(observed["model_state_dict_path"] or "")).expanduser(),
    }
    expected_paths = {
        "bundle_metadata": bundle_dir / "bundle_metadata.json",
        "master_transformer_lock": bundle_dir / "MASTER_TRANSFORMER_LOCK.json",
        "model_state_dict": bundle_dir / "model_state_dict.pt",
    }
    for name, path in paths.items():
        if not path.is_absolute() or path.resolve() != expected_paths[name]:
            _fail(context, f"{name} path is not the exact evaluation bundle file")
    hashes = {
        "bundle_metadata": _sha(
            observed["bundle_metadata_sha256"], context=f"{context}.bundle_metadata_sha256"
        ),
        "master_transformer_lock": _sha(
            observed["master_transformer_lock_sha256"],
            context=f"{context}.master_transformer_lock_sha256",
        ),
        "model_state_dict": _sha(
            observed["model_state_dict_sha256"],
            context=f"{context}.model_state_dict_sha256",
        ),
    }
    if hashes["model_state_dict"] != _sha(
        calibration["lineage"]["model_checkpoint_sha256"],
        context=f"{context}.calibration_checkpoint_sha256",
    ):
        _fail(context, "evaluation checkpoint differs from calibration checkpoint")
    canonical = {
        "bundle_dir": str(bundle_dir),
        "bundle_metadata_path": str(expected_paths["bundle_metadata"]),
        "bundle_metadata_sha256": hashes["bundle_metadata"],
        "master_transformer_lock_path": str(expected_paths["master_transformer_lock"]),
        "master_transformer_lock_sha256": hashes["master_transformer_lock"],
        "model_state_dict_path": str(expected_paths["model_state_dict"]),
        "model_state_dict_sha256": hashes["model_state_dict"],
    }
    if canonical != observed:
        _fail(context, "evaluation bundle path/hash canonicalization mismatch")
    if not verify_files:
        return observed
    for name, path in expected_paths.items():
        if not path.is_file() or sha256_file(path) != hashes[name]:
            _fail(context, f"evaluation bundle {name} missing or hash mismatch")
    metadata = _json_file(expected_paths["bundle_metadata"], context=f"{context}.metadata")
    lock = _json_file(
        expected_paths["master_transformer_lock"], context=f"{context}.lock"
    )
    if str(metadata.get("state_dict_sha256") or "").lower() != hashes["model_state_dict"]:
        _fail(context, "bundle metadata state_dict_sha256 mismatch")
    if str(lock.get("model_sha256") or "").lower() != hashes["model_state_dict"]:
        _fail(context, "transformer lock model_sha256 mismatch")
    if lock.get("model_path_relative") != "model_state_dict.pt":
        _fail(context, "transformer lock model_path_relative mismatch")
    return observed


def require_sizing_calibration_artifact(
    payload: Mapping[str, Any] | Any,
    *,
    context: str,
    verify_lineage_files: bool,
) -> dict[str, Any]:
    observed = _exact_keys(payload, _CALIBRATION_KEYS, context=context)
    if observed["schema_version"] != MODEL_NATIVE_SIZING_CALIBRATION_SCHEMA_VERSION:
        _fail(context, "schema_version mismatch")
    _utc(observed["created_utc"], context=f"{context}.created_utc")
    json_path = Path(str(observed["json_path"] or "")).expanduser()
    if not json_path.is_absolute():
        _fail(context, "json_path must be an absolute immutable self-reference")
    expected_static = {
        "fit_scope": MODEL_NATIVE_SIZING_FIT_SCOPE,
        "fit_splits": list(MODEL_NATIVE_SIZING_FIT_SPLITS),
        "holdout_split": MODEL_NATIVE_SIZING_HOLDOUT_SPLIT,
        "source_head": "position_size_logit",
        "transform_version": MODEL_NATIVE_SIZING_TRANSFORM_VERSION,
        "monotonic_direction": "non_decreasing",
    }
    for key, expected in expected_static.items():
        if observed[key] != expected:
            _fail(context, f"{key}={observed[key]!r} expected={expected!r}")
    if observed["fit_contract"] != sizing_fit_contract_metadata():
        _fail(context, "fit_contract differs from code-owned deterministic fit")

    instrument = _exact_keys(
        observed["instrument_constraints"],
        _INSTRUMENT_KEYS,
        context=f"{context}.instrument_constraints",
    )
    if instrument != sizing_offline_instrument_constraints_metadata():
        _fail(context, "instrument constraints differ from code-owned offline policy")
    step = _strict_int(instrument["unit_step"], context=f"{context}.unit_step", minimum=1)
    minimum = _strict_int(
        instrument["minimum_order_units"], context=f"{context}.minimum_units", minimum=1
    )
    maximum = _strict_int(
        instrument["maximum_gross_xau_units"],
        context=f"{context}.maximum_gross_xau_units",
        minimum=minimum,
    )
    if minimum % step or maximum % step:
        _fail(context, "instrument unit constraints must align to unit_step")
    if maximum != MODEL_NATIVE_SIZING_MAX_GROSS_XAU_UNITS:
        _fail(context, "maximum_gross_xau_units differs from immutable risk policy")
    margin_rate = _finite(instrument["margin_rate"], context=f"{context}.margin_rate")
    if not 0.0 < margin_rate <= 1.0:
        _fail(context, "margin_rate must be in (0,1]")

    parameters = _exact_keys(
        observed["parameters"], _PARAMETER_KEYS, context=f"{context}.parameters"
    )
    _finite(parameters["logit_center"], context=f"{context}.logit_center")
    temperature = _finite(
        parameters["logit_temperature"], context=f"{context}.logit_temperature"
    )
    low = _finite(
        parameters["minimum_capacity_fraction"], context=f"{context}.minimum_fraction"
    )
    reference = _finite(
        parameters["reference_capacity_fraction"], context=f"{context}.reference_fraction"
    )
    high = _finite(
        parameters["maximum_capacity_fraction"], context=f"{context}.maximum_fraction"
    )
    if temperature <= 0.0 or not 0.0 <= low < reference < high <= 1.0:
        _fail(context, "invalid monotone logistic parameter bounds")
    if not math.isclose(reference, (low + high) / 2.0, rel_tol=0.0, abs_tol=1e-15):
        _fail(context, "reference_capacity_fraction must equal transform value at logit_center")
    if high != MODEL_NATIVE_SIZING_MAX_CAPACITY_FRACTION:
        _fail(context, "maximum_capacity_fraction must equal immutable risk-policy cap")

    lineage = _exact_keys(observed["lineage"], _LINEAGE_KEYS, context=f"{context}.lineage")
    for stem in (
        "dataset_manifest",
        "fit_predictions",
        "model_checkpoint",
    ):
        _lineage_file(
            lineage[f"{stem}_path"],
            lineage[f"{stem}_sha256"],
            context=f"{context}.lineage.{stem}",
            verify_file=verify_lineage_files,
        )
    fit_binding = {
        "path": str(Path(str(lineage["fit_predictions_path"])).expanduser().resolve()),
        "sha256": str(lineage["fit_predictions_sha256"]),
    }
    provenance = require_sizing_prediction_provenance(
        observed["fit_prediction_provenance"],
        predictions_binding=fit_binding,
        expected_splits=MODEL_NATIVE_SIZING_FIT_SPLITS,
        expected_stage="pre_calibration",
        context=f"{context}.fit_prediction_provenance",
        verify_files=verify_lineage_files,
    )
    if verify_lineage_files:
        fit_bundle_checkpoint = (
            Path(provenance["bundle_dir"]).resolve() / "model_state_dict.pt"
        )
        lineage_checkpoint = Path(
            str(lineage["model_checkpoint_path"])
        ).expanduser().resolve()
        if fit_bundle_checkpoint != lineage_checkpoint:
            _fail(context, "fit provenance checkpoint is not exact calibration checkpoint")
        manifest_path = Path(str(lineage["dataset_manifest_path"])).resolve()
        dataset_dir = Path(provenance["dataset_dir"]).resolve()
        proven_val_manifest = Path(
            provenance["dataset_split_bindings"]["val"]["manifest_path"]
        ).resolve()
        if manifest_path.parent != dataset_dir or manifest_path != proven_val_manifest:
            _fail(context, "lineage dataset manifest is not exact proven VAL manifest")
        manifest = _json_file(manifest_path, context=f"{context}.dataset_manifest")
        coverage = manifest.get("ts_min_max_by_split")
        if not isinstance(coverage, Mapping):
            _fail(context, "dataset manifest lacks ts_min_max_by_split")
        fit_frame = pd.read_parquet(
            Path(lineage["fit_predictions_path"]),
            columns=[
                "time",
                "split",
                "model",
                "position_size_logit",
                "y_position_size_target",
            ],
        )
        if fit_frame.empty or set(fit_frame["split"].astype(str)) != set(
            MODEL_NATIVE_SIZING_FIT_SPLITS
        ) or set(fit_frame["model"].astype(str)) != {"candidate"}:
            _fail(context, "fit predictions are not exact candidate VAL")
        fit_times = pd.to_datetime(fit_frame["time"], utc=True, errors="coerce")
        if (
            fit_times.isna().any()
            or fit_times.duplicated().any()
            or not fit_times.is_monotonic_increasing
        ):
            _fail(context, "fit prediction time must be unique UTC and increasing")
        counts = fit_frame.groupby(fit_frame["split"].astype(str)).size().to_dict()
        if any(
            int(counts.get(split, 0)) < MODEL_NATIVE_SIZING_MIN_FIT_ROWS_PER_SPLIT
            for split in MODEL_NATIVE_SIZING_FIT_SPLITS
        ):
            _fail(context, "fit prediction support is below code-owned minimum")
        for split in MODEL_NATIVE_SIZING_FIT_SPLITS:
            rows = fit_times.loc[fit_frame["split"].astype(str) == split]
            declared = coverage.get(split)
            if not isinstance(declared, Mapping):
                _fail(context, f"dataset manifest lacks {split} coverage")
            try:
                declared_min = pd.Timestamp(declared["ts_min"])
                declared_max = pd.Timestamp(declared["ts_max"])
                declared_min = (
                    declared_min.tz_localize("UTC")
                    if declared_min.tzinfo is None
                    else declared_min.tz_convert("UTC")
                )
                declared_max = (
                    declared_max.tz_localize("UTC")
                    if declared_max.tzinfo is None
                    else declared_max.tz_convert("UTC")
                )
            except Exception as exc:
                _fail(context, f"dataset manifest {split} coverage invalid: {exc}")
            if rows.empty or rows.iloc[0] != declared_min or rows.iloc[-1] != declared_max:
                _fail(context, f"fit predictions do not cover full declared {split} range")
        logits = pd.to_numeric(
            fit_frame["position_size_logit"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        raw_targets = pd.to_numeric(
            fit_frame["y_position_size_target"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        if not np.isfinite(raw_targets).all() or np.any(
            (raw_targets < 0.0) | (raw_targets > 1.0)
        ):
            _fail(context, "fit target must be finite in [0,1]")
        recomputed_parameters = fit_monotone_sizing_parameters(
            logits,
            raw_targets * MODEL_NATIVE_SIZING_MAX_CAPACITY_FRACTION,
        )
        tolerance = float(
            sizing_fit_contract_metadata()["parameter_recompute_abs_tolerance"]
        )
        mismatched_parameters = [
            key
            for key, value in recomputed_parameters.items()
            if not math.isclose(
                float(parameters[key]), float(value), rel_tol=0.0, abs_tol=tolerance
            )
        ]
        if mismatched_parameters:
            _fail(
                context,
                f"calibration parameters differ from VAL refit: {mismatched_parameters}",
            )
    return observed


def require_runtime_sizing_constraints(
    value: Mapping[str, Any] | Any,
    *,
    calibration: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    """Validate code-owned offline TEST account facts only."""

    observed = _exact_keys(value, _RUNTIME_CONSTRAINT_KEYS, context=context)
    instrument = calibration["instrument_constraints"]
    for key in (
        "instrument",
        "account_currency",
        "margin_rate",
        "unit_step",
        "minimum_order_units",
        "maximum_gross_xau_units",
    ):
        if observed[key] != instrument[key]:
            _fail(context, f"runtime {key} differs from immutable instrument contract")
    numeric_positive = ("account_equity", "account_balance", "mark_price")
    for key in numeric_positive:
        if _finite(observed[key], context=f"{context}.{key}") <= 0.0:
            _fail(context, f"{key} must be positive")
    for key in (
        "margin_available",
        "margin_used",
        "current_xau_abs_units",
        "account_floating_drawdown_bps",
    ):
        if _finite(observed[key], context=f"{context}.{key}") < 0.0:
            _fail(context, f"{key} must be non-negative")
    expected_drawdown = max(
        0.0,
        (float(observed["account_balance"]) - float(observed["account_equity"]))
        / float(observed["account_balance"])
        * 10_000.0,
    )
    if not math.isclose(
        float(observed["account_floating_drawdown_bps"]),
        expected_drawdown,
        rel_tol=1e-9,
        abs_tol=1e-7,
    ):
        _fail(context, "account_floating_drawdown_bps does not match balance/equity")
    if observed["fact_provenance_mode"] != "canonical_oos_reference":
        _fail(context, "only canonical_oos_reference facts are allowed")
    return observed


def _floor_step(value: float, step: int) -> int:
    return int(math.floor(value / float(step))) * step


def calibrated_sizing_transform(
    *,
    calibration: Mapping[str, Any],
    position_size_logit: Any,
    model_direction_index: Any,
    runtime_constraints: Mapping[str, Any],
    context: str,
) -> dict[str, Any]:
    """Pure transform used only by frozen offline TEST evidence."""

    validated_calibration = require_sizing_calibration_artifact(
        calibration, context=f"{context}.calibration", verify_lineage_files=False
    )
    constraints = require_runtime_sizing_constraints(
        runtime_constraints, calibration=validated_calibration, context=f"{context}.runtime"
    )
    logit = _finite(position_size_logit, context=f"{context}.position_size_logit")
    direction = _strict_int(model_direction_index, context=f"{context}.direction", minimum=0)
    if direction not in (0, 1, 2):
        _fail(context, "model_direction_index must be LONG=0/SHORT=1/FLAT=2")
    parameters = validated_calibration["parameters"]
    scaled = (logit - float(parameters["logit_center"])) / float(
        parameters["logit_temperature"]
    )
    probability = 1.0 / (1.0 + math.exp(-max(-80.0, min(80.0, scaled))))
    low = float(parameters["minimum_capacity_fraction"])
    high = float(parameters["maximum_capacity_fraction"])
    reference_fraction = float(parameters["reference_capacity_fraction"])
    fraction = low + (high - low) * probability
    multiplier = fraction / reference_fraction

    equity = float(constraints["account_equity"])
    margin_used = float(constraints["margin_used"])
    margin_available = float(constraints["margin_available"])
    policy_remaining_margin = max(
        0.0,
        equity * MODEL_NATIVE_SIZING_MAX_ACCOUNT_MARGIN_FRACTION - margin_used,
    )
    admissible_margin = min(margin_available, policy_remaining_margin)
    unit_margin = float(constraints["mark_price"]) * float(constraints["margin_rate"])
    margin_capacity = admissible_margin / unit_margin
    gross_capacity = max(
        0.0,
        float(constraints["maximum_gross_xau_units"])
        - float(constraints["current_xau_abs_units"]),
    )
    step = int(constraints["unit_step"])
    capacity_units = _floor_step(min(margin_capacity, gross_capacity), step)
    reference_pre_round_units = float(capacity_units) * reference_fraction
    pre_round_units = reference_pre_round_units * multiplier
    units = _floor_step(pre_round_units, step)
    minimum_units = int(constraints["minimum_order_units"])
    if direction == 2:
        units = 0
        reason = "MODEL_DIRECTION_FLAT"
    elif (
        float(constraints["account_floating_drawdown_bps"])
        > MODEL_NATIVE_SIZING_MAX_ACCOUNT_FLOATING_DRAWDOWN_BPS
    ):
        units = 0
        reason = "IMMUTABLE_ACCOUNT_FLOATING_DRAWDOWN_CAP"
    elif capacity_units < minimum_units:
        units = 0
        reason = "INSUFFICIENT_ADMISSIBLE_CAPACITY"
    elif units < minimum_units:
        units = 0
        reason = "CALIBRATED_SIZE_BELOW_MINIMUM"
    else:
        reason = None
    return {
        "position_size_logit": logit,
        "model_direction_index": direction,
        "calibrated_size_fraction": fraction,
        "applied_size_multiplier": multiplier,
        "capacity_units": capacity_units,
        "reference_pre_round_units": reference_pre_round_units,
        "pre_round_units": pre_round_units,
        "units": units,
        "authorized_order": reason is None,
        "no_order_reason": reason,
        "runtime_constraints": constraints,
    }


def _read_table(binding: Mapping[str, Any], *, context: str) -> pd.DataFrame:
    validated = _source_binding(binding, context=context, verify_file=True)
    path = Path(validated["path"])
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    elif path.suffix.lower() in (".parquet", ".pq"):
        frame = pd.read_parquet(path)
    else:
        _fail(context, "source table must be CSV or parquet")
    if frame.empty:
        _fail(context, "source table is empty")
    return frame


def _require_source_tape_binding(
    value: Mapping[str, Any] | Any,
    *,
    test_prediction_provenance: Mapping[str, Any],
    context: str,
    verify_file: bool,
) -> dict[str, str]:
    observed = _exact_keys(value, _SOURCE_TAPE_KEYS, context=context)
    tape_path = Path(str(observed["path"] or "")).expanduser()
    manifest_path = Path(str(observed["dataset_test_manifest_path"] or "")).expanduser()
    if not tape_path.is_absolute() or not manifest_path.is_absolute():
        _fail(context, "source tape and TEST manifest paths must be absolute")
    canonical = {
        "path": str(tape_path.resolve()),
        "sha256": _sha(observed["sha256"], context=f"{context}.sha256"),
        "dataset_test_manifest_path": str(manifest_path.resolve()),
        "dataset_test_manifest_sha256": _sha(
            observed["dataset_test_manifest_sha256"],
            context=f"{context}.dataset_test_manifest_sha256",
        ),
    }
    if canonical != observed:
        _fail(context, "source tape binding canonicalization mismatch")
    proven_manifest = test_prediction_provenance["dataset_split_bindings"]["test"]
    if (
        canonical["dataset_test_manifest_path"] != proven_manifest["manifest_path"]
        or canonical["dataset_test_manifest_sha256"]
        != proven_manifest["manifest_sha256"]
    ):
        _fail(context, "source tape is not bound to exact proven TEST manifest")
    if not verify_file:
        return canonical
    if (
        not tape_path.is_file()
        or sha256_file(tape_path) != canonical["sha256"]
        or not manifest_path.is_file()
        or sha256_file(manifest_path) != canonical["dataset_test_manifest_sha256"]
    ):
        _fail(context, "source tape/TEST manifest missing or hash mismatch")
    manifest = _json_file(manifest_path, context=f"{context}.manifest")
    extra = manifest.get("extra") if isinstance(manifest.get("extra"), Mapping) else {}
    source_frame = (
        extra.get("source_frame")
        if isinstance(extra.get("source_frame"), Mapping)
        else {}
    )
    declared_path = Path(
        str(source_frame.get("parquet_path") or "")
    ).expanduser()
    if (
        not declared_path.is_absolute()
        or declared_path.resolve() != tape_path.resolve()
        or str(source_frame.get("parquet_sha256") or "").lower()
        != canonical["sha256"]
    ):
        _fail(context, "TEST manifest source frame does not bind exact source tape")
    return canonical


def derive_canonical_sizing_oos_rows(
    *,
    calibration: Mapping[str, Any],
    test_predictions_binding: Mapping[str, Any],
    test_prediction_provenance: Mapping[str, Any],
    evaluation_bundle: Mapping[str, Any],
    source_tape: Mapping[str, Any],
    context: str,
) -> pd.DataFrame:
    """Derive one full TEST sizing table from proven predictions and SourceTape."""

    calibration = require_sizing_calibration_artifact(
        calibration, context=f"{context}.calibration", verify_lineage_files=False
    )
    prediction_binding = _source_binding(
        test_predictions_binding,
        context=f"{context}.test_predictions",
        verify_file=True,
    )
    provenance = require_sizing_prediction_provenance(
        test_prediction_provenance,
        predictions_binding=prediction_binding,
        expected_splits=("test",),
        expected_stage="runtime_authoritative",
        context=f"{context}.prediction_provenance",
        verify_files=True,
    )
    evaluation = require_sizing_evaluation_bundle(
        evaluation_bundle,
        calibration=calibration,
        context=f"{context}.evaluation_bundle",
        verify_files=True,
    )
    tape_binding = _require_source_tape_binding(
        source_tape,
        test_prediction_provenance=provenance,
        context=f"{context}.source_tape",
        verify_file=True,
    )
    predictions = pd.read_parquet(Path(prediction_binding["path"]))
    predictions = predictions.loc[
        (predictions["split"].astype(str) == "test")
        & (predictions["model"].astype(str) == "candidate")
    ].copy()
    predictions["time"] = pd.to_datetime(predictions["time"], utc=True, errors="coerce")
    predictions = predictions.sort_values("time", kind="mergesort").reset_index(drop=True)
    dataset_path = Path(provenance["dataset_split_bindings"]["test"]["parquet_path"])
    horizons = pd.read_parquet(dataset_path, columns=["time", "label_horizon_bars"])
    horizons["time"] = pd.to_datetime(horizons["time"], utc=True, errors="coerce")
    joined = predictions.merge(horizons, on="time", how="left", validate="one_to_one")
    if len(joined) != len(horizons) or joined["label_horizon_bars"].isna().any():
        _fail(context, "TEST horizon coverage is not exact")
    times = pd.to_datetime(joined["time"], utc=True, errors="coerce")
    directions_raw = pd.to_numeric(joined["pred_direction"], errors="coerce").to_numpy()
    targets_raw = pd.to_numeric(joined["y_direction"], errors="coerce").to_numpy()
    logits = pd.to_numeric(joined["position_size_logit"], errors="coerce").to_numpy(float)
    if (
        times.isna().any()
        or times.duplicated().any()
        or not times.is_monotonic_increasing
        or not np.isfinite(logits).all()
        or not np.isfinite(directions_raw).all()
        or not np.array_equal(directions_raw, directions_raw.astype(np.int64))
        or not bool(np.isin(directions_raw.astype(np.int64), [0, 1, 2]).all())
        or not np.isfinite(targets_raw).all()
        or not np.array_equal(targets_raw, targets_raw.astype(np.int64))
        or not bool(np.isin(targets_raw.astype(np.int64), [0, 1, 2]).all())
    ):
        _fail(context, "canonical TEST prediction/target direction rows are invalid")
    directions = directions_raw.astype(np.int64)
    horizon_values = pd.to_numeric(joined["label_horizon_bars"], errors="coerce").to_numpy(float)
    if (
        not np.isfinite(horizon_values).all()
        or np.any(horizon_values <= 0.0)
        or not np.array_equal(horizon_values, np.floor(horizon_values))
    ):
        _fail(context, "canonical TEST label horizons are invalid")
    horizons_int = horizon_values.astype(np.int64)

    tape = SourceTape.load(Path(tape_binding["path"]))
    decision_indices = tape.indices_for_times(times)
    resolved_horizons = [
        tape.label_horizon_indices(
            decision_time=pd.Timestamp(timestamp),
            horizon_m5_bars=int(horizon),
        )
        for timestamp, horizon in zip(times, horizons_int, strict=True)
    ]
    fill_indices = np.asarray([row[0] for row in resolved_horizons], dtype=np.int64)
    exit_indices = np.asarray([row[1] for row in resolved_horizons], dtype=np.int64)
    policy = sizing_oos_reference_account_policy_metadata()
    row_scenario = policy["scenarios"][policy["canonical_row_scenario"]]
    instrument = calibration["instrument_constraints"]
    outcomes: list[dict[str, Any]] = []
    replay: list[dict[str, Any]] = []
    calculated: list[dict[str, Any]] = []
    for index, timestamp in enumerate(times):
        direction = int(directions[index])
        decision_idx = int(decision_indices[index])
        fill_idx = int(fill_indices[index])
        exit_idx = int(exit_indices[index])
        decision_utc = pd.Timestamp(timestamp).isoformat()
        mark_price = float(
            (tape.bid_close[decision_idx] + tape.ask_close[decision_idx]) / 2.0
        )
        constraints = {
            "instrument": instrument["instrument"],
            "account_currency": policy["account_currency"],
            "account_equity": row_scenario["account_equity"],
            "account_balance": row_scenario["account_balance"],
            "account_floating_drawdown_bps": row_scenario[
                "account_floating_drawdown_bps"
            ],
            "margin_available": row_scenario["margin_available"],
            "margin_used": row_scenario["margin_used"],
            "mark_price": mark_price,
            "margin_rate": instrument["margin_rate"],
            "unit_step": instrument["unit_step"],
            "minimum_order_units": instrument["minimum_order_units"],
            "maximum_gross_xau_units": instrument["maximum_gross_xau_units"],
            "current_xau_abs_units": row_scenario["current_xau_abs_units"],
            "fact_provenance_mode": "canonical_oos_reference",
        }
        transformed = calibrated_sizing_transform(
            calibration=calibration,
            position_size_logit=float(logits[index]),
            model_direction_index=direction,
            runtime_constraints=constraints,
            context=f"{context}.row[{index}]",
        )
        calculated.append(transformed)
        outcomes.append(
            {
                "time": decision_utc,
                "account_equity": row_scenario["account_equity"],
                "account_balance": row_scenario["account_balance"],
                "account_floating_drawdown_bps": 0.0,
                "margin_available": row_scenario["margin_available"],
                "margin_used": row_scenario["margin_used"],
                "current_xau_abs_units": row_scenario["current_xau_abs_units"],
                # Sizing is decided from the decision-bar mid, before the
                # subsequent fill.  Persist that exact canonical input so the
                # proof recomputation cannot silently substitute a fill price.
                "mark_price": mark_price,
                "entry_bid": float(tape.bid_open[fill_idx]),
                "entry_ask": float(tape.ask_open[fill_idx]),
                "exit_bid": float(tape.bid_close[exit_idx]),
                "exit_ask": float(tape.ask_close[exit_idx]),
                "fact_provenance_mode": "canonical_oos_reference",
            }
        )
        replay.append(
            {
                "time": decision_utc,
                **{
                    key: transformed[key]
                    for key in (
                        "position_size_logit",
                        "model_direction_index",
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
            }
        )
    required_prediction_fields = [
        "time",
        "split",
        "model",
        "position_size_logit",
        "pred_direction",
        "y_direction",
        "session",
        "vol_regime",
    ]
    missing_prediction_fields = sorted(
        set(required_prediction_fields) - set(predictions.columns)
    )
    if missing_prediction_fields:
        _fail(
            context,
            f"TEST predictions lack sizing slice fields: {missing_prediction_fields}",
        )
    prediction_rows = predictions.loc[:, required_prediction_fields].copy()
    prediction_rows["time"] = pd.to_datetime(
        prediction_rows["time"], utc=True
    ).map(lambda value: value.isoformat())
    prediction_rows = prediction_rows.rename(
        columns={
            "pred_direction": "model_direction_index",
            "y_direction": "target_direction_index",
        }
    )
    outcome_frame = pd.DataFrame(outcomes)
    replay_frame = pd.DataFrame(replay).drop(
        columns=["position_size_logit", "model_direction_index"]
    )
    combined = prediction_rows.merge(
        outcome_frame, on="time", how="inner", validate="one_to_one"
    ).merge(replay_frame, on="time", how="inner", validate="one_to_one")
    if len(combined) != len(predictions):
        _fail(context, "canonical OOS row combination lost TEST rows")
    combined["bundle_metadata_sha256"] = evaluation["bundle_metadata_sha256"]
    combined["model_state_dict_sha256"] = evaluation["model_state_dict_sha256"]
    combined["test_predictions_sha256"] = prediction_binding["sha256"]
    combined["source_tape_sha256"] = tape_binding["sha256"]
    combined["reference_row_id"] = [
        f"canonical-oos-row-{index:09d}" for index in range(len(combined))
    ]
    return combined


def load_bound_sizing_oos_source(
    binding: Mapping[str, Any] | Any,
    *,
    calibration: Mapping[str, Any],
    calibration_artifact_sha256: str,
    context: str,
    verify_source_files: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Load and independently re-derive the one canonical OOS source event."""

    canonical_binding = require_immutable_json_binding(
        binding,
        event_prefix=_OOS_SOURCE_EVENT_PREFIX,
        context=f"{context}.binding",
        verify_file=True,
    )
    payload = _exact_keys(
        _json_file(Path(canonical_binding["json_path"]), context=context),
        _OOS_SOURCE_KEYS,
        context=context,
    )
    if payload["schema_version"] != MODEL_NATIVE_SIZING_OOS_SOURCE_SCHEMA_VERSION:
        _fail(context, "OOS source schema_version mismatch")
    _utc(payload["created_utc"], context=f"{context}.created_utc")
    if Path(str(payload["json_path"] or "")).expanduser().resolve() != Path(
        canonical_binding["json_path"]
    ):
        _fail(context, "OOS source json_path self-reference mismatch")
    if payload["decision"] != "PASS" or payload["failures"] != []:
        _fail(context, "OOS source must be zero-failure PASS")
    if _sha(
        payload["calibration_artifact_sha256"],
        context=f"{context}.calibration_artifact_sha256",
    ) != _sha(calibration_artifact_sha256, context=f"{context}.expected_calibration"):
        _fail(context, "OOS source calibration hash mismatch")
    prediction_binding = _source_binding(
        payload["test_predictions"],
        context=f"{context}.test_predictions",
        verify_file=verify_source_files,
    )
    provenance = require_sizing_prediction_provenance(
        payload["test_prediction_provenance"],
        predictions_binding=prediction_binding,
        expected_splits=("test",),
        expected_stage="runtime_authoritative",
        context=f"{context}.test_prediction_provenance",
        verify_files=verify_source_files,
    )
    evaluation = require_sizing_evaluation_bundle(
        payload["evaluation_bundle"],
        calibration=calibration,
        context=f"{context}.evaluation_bundle",
        verify_files=verify_source_files,
    )
    tape = _require_source_tape_binding(
        payload["source_tape"],
        test_prediction_provenance=provenance,
        context=f"{context}.source_tape",
        verify_file=verify_source_files,
    )
    if payload["reference_account_policy"] != sizing_oos_reference_account_policy_metadata():
        _fail(context, "OOS source reference-account policy mismatch")
    raw_bindings = _exact_keys(
        payload["source_bindings"],
        frozenset(_SOURCE_BINDING_NAMES),
        context=f"{context}.source_bindings",
    )
    source_bindings = {
        name: _source_binding(
            raw_bindings[name],
            context=f"{context}.source_bindings.{name}",
            verify_file=verify_source_files,
        )
        for name in _SOURCE_BINDING_NAMES
    }
    if source_bindings != raw_bindings:
        _fail(context, "OOS source output binding canonicalization mismatch")
    if verify_source_files:
        expected = derive_canonical_sizing_oos_rows(
            calibration=calibration,
            test_predictions_binding=prediction_binding,
            test_prediction_provenance=provenance,
            evaluation_bundle=evaluation,
            source_tape=tape,
            context=f"{context}.rederive",
        )
        observed_rows = _read_table(
            source_bindings["oos_rows"], context=f"{context}.oos_rows"
        )
        try:
            pd.testing.assert_frame_equal(
                observed_rows.reset_index(drop=True),
                expected.reset_index(drop=True),
                check_dtype=False,
                check_exact=True,
            )
        except AssertionError as exc:
            _fail(context, f"published OOS rows differ from independent re-derivation: {exc}")
    return payload, canonical_binding


def _require_columns(frame: pd.DataFrame, columns: set[str], *, context: str) -> None:
    missing = sorted(columns - set(frame.columns))
    unexpected = sorted(set(frame.columns) - columns)
    if missing or unexpected:
        _fail(context, f"exact columns mismatch: missing={missing} unexpected={unexpected}")


def _utc_series(frame: pd.DataFrame, *, context: str) -> pd.Series:
    parsed = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    if parsed.isna().any() or parsed.duplicated().any() or not parsed.is_monotonic_increasing:
        _fail(context, "time must be unique, finite UTC, and increasing")
    return parsed


def _bool_series(series: pd.Series, *, context: str) -> np.ndarray:
    mapped = series.map(
        lambda value: value
        if isinstance(value, (bool, np.bool_))
        else {"True": True, "False": False}.get(str(value))
    )
    if mapped.isna().any():
        _fail(context, "boolean column contains non-boolean values")
    return mapped.to_numpy(dtype=bool)


def _strict_table_int(
    value: Any,
    *,
    context: str,
    minimum: int = 0,
    allowed: frozenset[int] | None = None,
) -> int:
    """Accept an integer-typed table cell without coercing a float/string."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        _fail(context, f"exact integer table cell required: {value!r}")
    parsed = int(value)
    if parsed < minimum:
        _fail(context, f"integer={parsed} must be >= {minimum}")
    if allowed is not None and parsed not in allowed:
        _fail(context, f"integer={parsed} not in exact allowed set {sorted(allowed)}")
    return parsed


def _finite_numeric_column(
    frame: pd.DataFrame,
    column: str,
    *,
    context: str,
) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        _fail(context, f"{column} contains non-finite/non-numeric values")
    return values


def _max_drawdown(values: np.ndarray) -> float:
    cumulative = np.cumsum(values, dtype=np.float64)
    peaks = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))[:-1]
    return float(np.max(peaks - cumulative, initial=0.0))


def _block_lower_95(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        return {
            "blocks": int(len(values)),
            "mean_bps": 0.0,
            "sample_std_bps": 0.0,
            "student_t_critical_two_sided_95": None,
            "lower_95_bps": -1e308,
        }
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    critical = float(student_t.ppf(0.975, df=len(values) - 1)) if len(values) > 1 else None
    lower = (
        mean - float(critical) * std / math.sqrt(len(values))
        if critical is not None
        else -1e308
    )
    return {
        "blocks": int(len(values)),
        "mean_bps": mean,
        "sample_std_bps": std,
        "student_t_critical_two_sided_95": critical,
        "lower_95_bps": float(lower),
    }


def _capacity_capped_equal_total_continuous(
    capacities: np.ndarray, total_units: float
) -> np.ndarray:
    """Deterministic equal-allocation control with the same continuous exposure."""

    caps = np.asarray(capacities, dtype=np.float64)
    total = float(total_units)
    if (
        caps.ndim != 1
        or not np.isfinite(caps).all()
        or np.any(caps < 0.0)
        or not math.isfinite(total)
        or total < -1e-12
        or total > float(np.sum(caps)) + 1e-9
    ):
        raise ModelNativeSizingContractError(
            "equal-total continuous allocation inputs are infeasible"
        )
    if len(caps) == 0 or total <= 0.0:
        return np.zeros_like(caps)
    low = 0.0
    high = float(np.max(caps, initial=0.0))
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if float(np.sum(np.minimum(caps, midpoint))) < total:
            low = midpoint
        else:
            high = midpoint
    allocated = np.minimum(caps, high)
    residual = total - float(np.sum(allocated))
    if abs(residual) > 1e-9:
        eligible = np.flatnonzero(allocated < caps - 1e-12)
        if len(eligible):
            allocated[int(eligible[0])] += residual
    if not math.isclose(float(np.sum(allocated)), total, rel_tol=0.0, abs_tol=1e-8):
        raise ModelNativeSizingContractError(
            "equal-total continuous allocation failed exposure conservation"
        )
    return allocated


def _deterministic_equal_total_integer(
    capacities: np.ndarray, total_units: int
) -> np.ndarray:
    """Chronology-stable near-equal integer allocation used as diagnostics only."""

    caps = np.asarray(capacities, dtype=np.int64)
    total = int(total_units)
    if caps.ndim != 1 or np.any(caps < 0) or total < 0 or total > int(caps.sum()):
        raise ModelNativeSizingContractError(
            "equal-total integer allocation inputs are infeasible"
        )
    allocated = np.zeros_like(caps)
    remaining = total
    while remaining:
        eligible = np.flatnonzero(allocated < caps)
        if len(eligible) == 0:
            raise ModelNativeSizingContractError(
                "equal-total integer allocation exhausted capacity"
            )
        take = min(remaining, len(eligible))
        allocated[eligible[:take]] += 1
        remaining -= take
    return allocated


def _paired_block_admission(
    delta: np.ndarray,
    *,
    times: pd.DatetimeIndex,
    sessions: np.ndarray,
    regimes: np.ndarray,
) -> dict[str, Any]:
    """Student-t admission on independent calendar blocks and required slices."""

    values = np.asarray(delta, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) != len(times)
        or len(values) != len(sessions)
        or len(values) != len(regimes)
        or not np.isfinite(values).all()
    ):
        raise ModelNativeSizingContractError("paired block evidence is not aligned")
    iso = times.isocalendar()
    frame = pd.DataFrame(
        {
            "delta": values,
            "week": [
                f"{int(year):04d}-W{int(week):02d}"
                for year, week in zip(iso.year, iso.week, strict=True)
            ],
            "month": times.strftime("%Y-%m"),
            "session": np.asarray(sessions, dtype=str),
            "vol_regime": np.asarray(regimes, dtype=str),
        }
    )
    weekly = _block_lower_95(
        frame.groupby("week", sort=True)["delta"].sum().to_numpy(float)
    )
    monthly = _block_lower_95(
        frame.groupby("month", sort=True)["delta"].sum().to_numpy(float)
    )

    def sliced(column: str) -> dict[str, dict[str, Any]]:
        output: dict[str, dict[str, Any]] = {}
        for value, group in frame.groupby(column, sort=True):
            stats = _block_lower_95(
                group.groupby("week", sort=True)["delta"].sum().to_numpy(float)
            )
            stats["evaluated_orders"] = int(len(group))
            output[str(value)] = stats
        return output

    session = sliced("session")
    regime = sliced("vol_regime")
    slice_values = [*session.values(), *regime.values()]
    admitted = (
        int(weekly["blocks"]) >= MODEL_NATIVE_SIZING_MIN_OOS_WEEK_BLOCKS
        and int(monthly["blocks"]) >= MODEL_NATIVE_SIZING_MIN_OOS_MONTH_BLOCKS
        and float(weekly["lower_95_bps"]) > 0.0
        and float(monthly["lower_95_bps"]) > 0.0
        and bool(slice_values)
        and all(
            int(row["blocks"]) >= MODEL_NATIVE_SIZING_MIN_SLICE_WEEK_BLOCKS
            and float(row["lower_95_bps"]) > 0.0
            for row in slice_values
        )
    )
    return {
        "decision": "PASS" if admitted else "FAIL",
        "evaluated_orders": int(len(values)),
        "row_delta_mean_bps_diagnostic_only": float(np.mean(values)),
        "row_delta_sample_std_bps_diagnostic_only": (
            float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ),
        "admission_statistic": (
            "paired_iso_week_and_month_student_t_lower95_with_"
            "session_and_vol_regime_week_student_t_lower95"
        ),
        "minimum_week_blocks": MODEL_NATIVE_SIZING_MIN_OOS_WEEK_BLOCKS,
        "minimum_month_blocks": MODEL_NATIVE_SIZING_MIN_OOS_MONTH_BLOCKS,
        "minimum_slice_week_blocks": MODEL_NATIVE_SIZING_MIN_SLICE_WEEK_BLOCKS,
        "iso_week_blocks": weekly,
        "month_blocks": monthly,
        "session_week_blocks": session,
        "vol_regime_week_blocks": regime,
    }


def sizing_direction_edge_policy_metadata() -> dict[str, Any]:
    """Bind the exact immutable core precision policy enforced on TEST."""

    policy_path = Path(__file__).with_name("entry_foundation_audit_policy_v1.py")
    if policy_path.is_symlink() or not policy_path.is_file():
        _fail("SIZING_DIRECTION_EDGE_POLICY", "foundation policy source is absent")
    try:
        source = policy_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(policy_path))
    except Exception as exc:
        _fail("SIZING_DIRECTION_EDGE_POLICY", f"foundation policy unreadable: {exc}")
    schema_node: ast.AST | None = None
    policy_node: ast.Dict | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            target = node.target
            value_node = node.value
        else:
            continue
        if not isinstance(target, ast.Name):
            continue
        if target.id == "FOUNDATION_AUDIT_POLICY_SCHEMA_VERSION":
            schema_node = value_node
        elif target.id == "_FOUNDATION_AUDIT_POLICY" and isinstance(
            value_node, ast.Dict
        ):
            policy_node = value_node
    try:
        schema = ast.literal_eval(schema_node) if schema_node is not None else None
    except Exception as exc:
        _fail("SIZING_DIRECTION_EDGE_POLICY", f"policy schema is not literal: {exc}")
    if (
        not isinstance(schema, str)
        or _FOUNDATION_AUDIT_POLICY_SCHEMA_RE.fullmatch(schema) is None
        or policy_node is None
    ):
        _fail("SIZING_DIRECTION_EDGE_POLICY", "foundation policy schema/payload changed")

    def dict_value(node: ast.Dict, key: str) -> ast.AST | None:
        for raw_key, raw_value in zip(node.keys, node.values, strict=True):
            try:
                parsed_key = ast.literal_eval(raw_key) if raw_key is not None else None
            except Exception:
                continue
            if parsed_key == key:
                return raw_value
        return None

    edge_node = dict_value(policy_node, "smoke_edge_pockets")
    if not isinstance(edge_node, ast.Dict):
        _fail("SIZING_DIRECTION_EDGE_POLICY", "smoke_edge_pockets is absent")
    enforced_core: dict[str, Any] = {}
    for name in _DIRECTION_EDGE_POLICY_FIELDS:
        value_node = dict_value(edge_node, name)
        try:
            value = ast.literal_eval(value_node) if value_node is not None else None
        except Exception as exc:
            _fail("SIZING_DIRECTION_EDGE_POLICY", f"{name} is not literal: {exc}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _fail("SIZING_DIRECTION_EDGE_POLICY", f"{name} is not numeric")
        floor = _DIRECTION_EDGE_POLICY_FLOORS[name]
        if float(value) < float(floor):
            _fail(
                "SIZING_DIRECTION_EDGE_POLICY",
                f"{name}={value} weakens immutable floor={floor}",
            )
        enforced_core[name] = value
    core_sha = hashlib.sha256(
        json.dumps(
            enforced_core,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "foundation_audit_policy_binding": {
            "foundation_audit_policy_schema_version": schema,
            "foundation_audit_policy_source": (
                "gx1/contracts/entry_foundation_audit_policy_v1.py"
            ),
            "foundation_audit_policy_source_sha256": hashlib.sha256(
                source.encode("utf-8")
            ).hexdigest(),
            "enforced_core_sha256": core_sha,
        },
        "policy_section": "smoke_edge_pockets",
        "enforced_core": enforced_core,
        "direction_order": ["LONG", "SHORT", "FLAT"],
        "evaluation_split": "test",
        "evidence_stage": "runtime_authoritative",
    }


def _wilson_lower(successes: int, trials: int, *, z_score: float) -> float:
    if trials <= 0 or successes < 0 or successes > trials:
        return 0.0
    proportion = successes / trials
    denominator = 1.0 + (z_score * z_score) / trials
    center = proportion + (z_score * z_score) / (2.0 * trials)
    radius = z_score * math.sqrt(
        proportion * (1.0 - proportion) / trials
        + (z_score * z_score) / (4.0 * trials * trials)
    )
    return float((center - radius) / denominator)


def recompute_direction_edge_admission(
    *,
    predicted: np.ndarray,
    target: np.ndarray,
    context: str,
) -> dict[str, Any]:
    """Recompute hard TEST precision/Wilson/class admission from exact rows."""

    predicted = np.asarray(predicted, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    if (
        predicted.ndim != 1
        or target.shape != predicted.shape
        or len(predicted) == 0
        or not np.isin(predicted, [0, 1, 2]).all()
        or not np.isin(target, [0, 1, 2]).all()
    ):
        _fail(context, "direction edge rows are invalid")
    policy = sizing_direction_edge_policy_metadata()["enforced_core"]
    z_score = float(policy["wilson_z_score"])
    names = ("LONG", "SHORT", "FLAT")
    correct = predicted == target
    class_rows: dict[str, dict[str, Any]] = {}
    recalls: list[float] = []
    failures: list[str] = []
    for class_id, name in enumerate(names):
        predicted_mask = predicted == class_id
        target_mask = target == class_id
        predicted_rows = int(np.count_nonzero(predicted_mask))
        target_rows = int(np.count_nonzero(target_mask))
        successes = int(np.count_nonzero(predicted_mask & correct))
        precision = successes / predicted_rows if predicted_rows else 0.0
        recall = successes / target_rows if target_rows else 0.0
        wilson = _wilson_lower(successes, predicted_rows, z_score=z_score)
        recalls.append(recall)
        class_rows[name] = {
            "predicted_rows": predicted_rows,
            "target_rows": target_rows,
            "correct_rows": successes,
            "precision": float(precision),
            "precision_wilson_lower": wilson,
            "recall": float(recall),
        }
        if predicted_rows < int(policy["min_prediction_rows_per_class"]):
            failures.append(f"{name}:prediction_support")
        if precision < float(policy["min_class_precision"]):
            failures.append(f"{name}:precision")
        if wilson < float(policy["min_class_precision_wilson_lower"]):
            failures.append(f"{name}:precision_wilson_lower")
    trade_mask = predicted != 2
    trade_rows = int(np.count_nonzero(trade_mask))
    trade_successes = int(np.count_nonzero(trade_mask & correct))
    trade_precision = trade_successes / trade_rows if trade_rows else 0.0
    trade_wilson = _wilson_lower(trade_successes, trade_rows, z_score=z_score)
    direction_accuracy = float(np.mean(correct))
    balanced_accuracy = float(np.mean(recalls))
    if trade_rows < int(policy["min_trade_rows"]):
        failures.append("trade_prediction_support")
    if trade_precision < float(policy["min_trade_direction_precision"]):
        failures.append("trade_direction_precision")
    if trade_wilson < float(policy["min_trade_precision_wilson_lower"]):
        failures.append("trade_precision_wilson_lower")
    if direction_accuracy < float(policy["min_direction_accuracy"]):
        failures.append("direction_accuracy")
    if balanced_accuracy < float(policy["min_balanced_accuracy"]):
        failures.append("balanced_accuracy")
    return {
        "decision": "PASS" if not failures else "FAIL",
        "failures": failures,
        "rows": int(len(predicted)),
        "direction_accuracy": direction_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "trade_rows": trade_rows,
        "trade_correct_rows": trade_successes,
        "trade_direction_precision": float(trade_precision),
        "trade_direction_precision_wilson_lower": trade_wilson,
        "classes": class_rows,
    }


def recompute_sizing_oos_evidence(
    *,
    calibration: Mapping[str, Any],
    source_bindings: Mapping[str, Any],
    evaluation_bundle: Mapping[str, Any],
    context: str,
    fact_provenance_mode: str = "canonical_oos_reference",
    extra_row_columns: frozenset[str] = frozenset(),
    outcome_price_mode: str = "label_horizon",
) -> dict[str, Any]:
    """Recompute every admission metric from immutable row-level sources."""

    if outcome_price_mode not in {"label_horizon", "model_exit_fill"}:
        _fail(
            context,
            f"unsupported outcome_price_mode={outcome_price_mode!r}",
        )
    bindings = _exact_keys(
        source_bindings, frozenset(_SOURCE_BINDING_NAMES), context=f"{context}.bindings"
    )
    combined = _read_table(bindings["oos_rows"], context=f"{context}.oos_rows")
    _require_columns(
        combined,
        MODEL_NATIVE_SIZING_OOS_ROW_COLUMNS | extra_row_columns,
        context=f"{context}.oos_rows",
    )
    predictions = combined.loc[
        :, sorted(MODEL_NATIVE_SIZING_OOS_PREDICTION_COLUMNS)
    ].copy().rename(
        columns={"model_direction_index": "pred_direction"}
    )
    outcomes = combined.loc[:, sorted(MODEL_NATIVE_SIZING_OOS_OUTCOME_COLUMNS)].copy()
    replay = combined.loc[:, sorted(MODEL_NATIVE_SIZING_OOS_REPLAY_COLUMNS)].copy()
    times = _utc_series(predictions, context=f"{context}.predictions")
    outcome_times = _utc_series(outcomes, context=f"{context}.outcomes")
    replay_times = _utc_series(replay, context=f"{context}.replay")
    if not times.equals(outcome_times) or not times.equals(replay_times):
        _fail(context, "prediction/outcome/replay TEST rows are not exactly time-aligned")
    if set(predictions["split"].astype(str)) != {"test"} or set(
        predictions["model"].astype(str)
    ) != {"candidate"}:
        _fail(context, "prediction source must contain only candidate TEST rows")
    rows = len(predictions)
    if rows < 256:
        _fail(context, "full TEST proof requires at least 256 rows")
    evaluation = require_sizing_evaluation_bundle(
        evaluation_bundle,
        calibration=calibration,
        context=f"{context}.evaluation_bundle",
        verify_files=True,
    )
    for field, expected in (
        ("bundle_metadata_sha256", evaluation["bundle_metadata_sha256"]),
        ("model_state_dict_sha256", evaluation["model_state_dict_sha256"]),
    ):
        if {str(value).strip().lower() for value in combined[field].array} != {
            expected
        }:
            _fail(context, f"OOS rows {field} differs from evaluation bundle")
    expected_row_ids = [f"canonical-oos-row-{index:09d}" for index in range(rows)]
    if list(combined["reference_row_id"].astype(str)) != expected_row_ids:
        _fail(context, "OOS reference_row_id sequence mismatch")
    if fact_provenance_mode != "canonical_oos_reference":
        _fail(context, f"unsupported fact_provenance_mode={fact_provenance_mode!r}")
    if set(outcomes["fact_provenance_mode"].astype(str)) != {
        fact_provenance_mode
    }:
        _fail(context, f"OOS rows must use {fact_provenance_mode} facts")
    logits = _finite_numeric_column(
        predictions, "position_size_logit", context=f"{context}.predictions"
    )
    directions = np.asarray(
        [
            _strict_table_int(
                value,
                context=f"{context}.predictions.pred_direction[{index}]",
                allowed=frozenset({0, 1, 2}),
            )
            for index, value in enumerate(predictions["pred_direction"].array)
        ],
        dtype=np.int64,
    )
    targets = np.asarray(
        [
            _strict_table_int(
                value,
                context=f"{context}.predictions.target_direction_index[{index}]",
                allowed=frozenset({0, 1, 2}),
            )
            for index, value in enumerate(
                predictions["target_direction_index"].array
            )
        ],
        dtype=np.int64,
    )
    replay_directions = np.asarray(
        [
            _strict_table_int(
                value,
                context=f"{context}.replay.model_direction_index[{index}]",
                allowed=frozenset({0, 1, 2}),
            )
            for index, value in enumerate(replay["model_direction_index"].array)
        ],
        dtype=np.int64,
    )
    replay_integer_fields = {
        field: np.asarray(
            [
                _strict_table_int(
                    value,
                    context=f"{context}.replay.{field}[{index}]",
                    minimum=0,
                )
                for index, value in enumerate(replay[field].array)
            ],
            dtype=np.int64,
        )
        for field in ("capacity_units", "units")
    }
    replay_float_fields = {
        field: _finite_numeric_column(replay, field, context=f"{context}.replay")
        for field in (
            "position_size_logit",
            "calibrated_size_fraction",
            "applied_size_multiplier",
            "reference_pre_round_units",
            "pre_round_units",
        )
    }
    replay_authorized = _bool_series(
        replay["authorized_order"], context=f"{context}.replay.authorized_order"
    )
    outcome_numeric = {
        field: _finite_numeric_column(outcomes, field, context=f"{context}.outcomes")
        for field in (
            "account_equity",
            "account_balance",
            "account_floating_drawdown_bps",
            "margin_available",
            "margin_used",
            "current_xau_abs_units",
            "mark_price",
            "entry_bid",
            "entry_ask",
            "exit_bid",
            "exit_ask",
        )
    }
    for field in (
        "account_equity",
        "account_balance",
        "mark_price",
        "entry_bid",
        "entry_ask",
        "exit_bid",
        "exit_ask",
    ):
        if np.any(outcome_numeric[field] <= 0.0):
            _fail(context, f"outcomes.{field} must be finite and positive")
    for field in (
        "account_floating_drawdown_bps",
        "margin_available",
        "margin_used",
        "current_xau_abs_units",
    ):
        if np.any(outcome_numeric[field] < 0.0):
            _fail(context, f"outcomes.{field} must be finite and non-negative")
    if np.any(outcome_numeric["entry_bid"] > outcome_numeric["entry_ask"]):
        _fail(context, "outcomes entry_bid must be <= entry_ask")
    if np.any(outcome_numeric["exit_bid"] > outcome_numeric["exit_ask"]):
        _fail(context, "outcomes exit_bid must be <= exit_ask")
    realized_exit_bid = outcome_numeric["exit_bid"]
    realized_exit_ask = outcome_numeric["exit_ask"]
    if outcome_price_mode == "model_exit_fill":
        realized_exit_bid = pd.to_numeric(
            combined["model_exit_fill_bid"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        realized_exit_ask = pd.to_numeric(
            combined["model_exit_fill_ask"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        trade_mask = np.isin(directions, [0, 1])
        flat_mask = directions == 2
        if (
            not np.isfinite(realized_exit_bid[trade_mask]).all()
            or not np.isfinite(realized_exit_ask[trade_mask]).all()
            or np.any(realized_exit_bid[trade_mask] <= 0.0)
            or np.any(
                realized_exit_ask[trade_mask]
                < realized_exit_bid[trade_mask]
            )
            or not np.isnan(realized_exit_bid[flat_mask]).all()
            or not np.isnan(realized_exit_ask[flat_mask]).all()
        ):
            _fail(context, "model Exit fill outcome prices are invalid")
    instrument = calibration["instrument_constraints"]
    if int(instrument["unit_step"]) != 1 or int(instrument["minimum_order_units"]) != 1:
        _fail(context, "historical control requires exact executable XAU 1-unit sizing")
    calculated: list[dict[str, Any]] = []
    price_pnl_per_unit: list[float] = []
    for index in range(rows):
        outcome = outcomes.iloc[index]
        # This is the hash-bound decision-time price written by the canonical
        # SourceTape producer, not a price reconstructed from the later fill.
        mark_price = outcome_numeric["mark_price"][index]
        constraints = {
            "instrument": instrument["instrument"],
            "account_currency": instrument["account_currency"],
            "account_equity": outcome_numeric["account_equity"][index],
            "account_balance": outcome_numeric["account_balance"][index],
            "account_floating_drawdown_bps": outcome_numeric[
                "account_floating_drawdown_bps"
            ][index],
            "margin_available": outcome_numeric["margin_available"][index],
            "margin_used": outcome_numeric["margin_used"][index],
            "mark_price": mark_price,
            "margin_rate": instrument["margin_rate"],
            "unit_step": instrument["unit_step"],
            "minimum_order_units": instrument["minimum_order_units"],
            "maximum_gross_xau_units": instrument["maximum_gross_xau_units"],
            "current_xau_abs_units": outcome_numeric["current_xau_abs_units"][index],
            "fact_provenance_mode": str(outcome["fact_provenance_mode"]),
        }
        transformed = calibrated_sizing_transform(
            calibration=calibration,
            position_size_logit=logits[index],
            model_direction_index=int(directions[index]),
            runtime_constraints=constraints,
            context=f"{context}.row[{index}]",
        )
        calculated.append(transformed)
        row = replay.iloc[index]
        for field in replay_float_fields:
            if not math.isclose(
                float(replay_float_fields[field][index]),
                float(transformed[field]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                _fail(context, f"replay row {index} {field} differs from recomputation")
        if replay_directions[index] != transformed["model_direction_index"]:
            _fail(context, f"replay row {index} model_direction_index differs from recomputation")
        for field, values in replay_integer_fields.items():
            if int(values[index]) != transformed[field]:
                _fail(context, f"replay row {index} {field} differs from recomputation")
        if bool(replay_authorized[index]) != bool(transformed["authorized_order"]):
            _fail(context, f"replay row {index} authorized_order mismatch")
        observed_reason = None if pd.isna(row["no_order_reason"]) else str(row["no_order_reason"])
        if observed_reason != transformed["no_order_reason"]:
            _fail(context, f"replay row {index} no_order_reason mismatch")
        direction = transformed["model_direction_index"]
        if direction == 0:
            per_unit_pnl = (
                realized_exit_bid[index]
                - outcome_numeric["entry_ask"][index]
            )
        elif direction == 1:
            per_unit_pnl = (
                outcome_numeric["entry_bid"][index]
                - realized_exit_ask[index]
            )
        else:
            per_unit_pnl = 0.0
        price_pnl_per_unit.append(per_unit_pnl)

    order = np.argsort(logits, kind="mergesort")
    sorted_fractions = np.asarray(
        [calculated[index]["calibrated_size_fraction"] for index in order], dtype=float
    )
    fraction_delta = np.diff(sorted_fractions)
    violation_count = int(np.count_nonzero(fraction_delta < -1e-12))
    reference_policy = sizing_oos_reference_account_policy_metadata()
    canonical_scenario = reference_policy["scenarios"][
        reference_policy["canonical_row_scenario"]
    ]
    for field in (
        "account_equity",
        "account_balance",
        "account_floating_drawdown_bps",
        "margin_available",
        "margin_used",
        "current_xau_abs_units",
    ):
        if not np.array_equal(
            outcome_numeric[field],
            np.full(rows, float(canonical_scenario[field]), dtype=np.float64),
        ):
            _fail(context, f"canonical OOS rows do not match medium grid field {field}")

    price_pnl = np.asarray(price_pnl_per_unit, dtype=np.float64)
    sessions = predictions["session"].astype(str).to_numpy()
    regimes = predictions["vol_regime"].astype(str).to_numpy()
    utility_scenarios: dict[str, dict[str, Any]] = {}
    exposure_scenarios: dict[str, dict[str, Any]] = {}
    drawdown_scenarios: dict[str, dict[str, Any]] = {}
    grid_scenarios: dict[str, dict[str, Any]] = {}
    for scenario_name in reference_policy["scenario_order"]:
        scenario = reference_policy["scenarios"][scenario_name]
        scenario_transforms: list[dict[str, Any]] = []
        admitted: list[bool] = []
        for index in range(rows):
            constraints = {
                "instrument": instrument["instrument"],
                "account_currency": reference_policy["account_currency"],
                **scenario,
                "mark_price": outcome_numeric["mark_price"][index],
                "margin_rate": instrument["margin_rate"],
                "unit_step": instrument["unit_step"],
                "minimum_order_units": instrument["minimum_order_units"],
                "maximum_gross_xau_units": instrument["maximum_gross_xau_units"],
                "fact_provenance_mode": "canonical_oos_reference",
            }
            transformed = calibrated_sizing_transform(
                calibration=calibration,
                position_size_logit=logits[index],
                model_direction_index=int(directions[index]),
                runtime_constraints=constraints,
                context=f"{context}.grid.{scenario_name}.row[{index}]",
            )
            scenario_transforms.append(transformed)
            admitted.append(
                int(directions[index]) in (0, 1)
                and float(scenario["account_floating_drawdown_bps"])
                <= MODEL_NATIVE_SIZING_MAX_ACCOUNT_FLOATING_DRAWDOWN_BPS
                and int(transformed["capacity_units"])
                >= int(instrument["minimum_order_units"])
            )
        evaluated_mask = np.asarray(admitted, dtype=bool)
        evaluated = int(np.count_nonzero(evaluated_mask))
        if evaluated < MODEL_NATIVE_SIZING_MIN_OOS_ORDERS:
            _fail(
                context,
                f"scenario {scenario_name} has insufficient OOS non-FLAT orders",
            )
        actual_units = np.asarray(
            [
                int(row["units"]) if evaluated_mask[index] else 0
                for index, row in enumerate(scenario_transforms)
            ],
            dtype=np.int64,
        )
        continuous_units = np.asarray(
            [
                float(row["pre_round_units"]) if evaluated_mask[index] else 0.0
                for index, row in enumerate(scenario_transforms)
            ],
            dtype=np.float64,
        )
        capacities = np.asarray(
            [int(row["capacity_units"]) for row in scenario_transforms],
            dtype=np.int64,
        )
        account_bps_per_unit = (
            price_pnl / float(scenario["account_equity"]) * 10_000.0
        )
        learned_pnl = account_bps_per_unit * actual_units
        historical_units = evaluated_mask.astype(np.int64)
        historical_pnl = account_bps_per_unit * historical_units
        evaluated_times = pd.DatetimeIndex(times.loc[evaluated_mask])
        evaluated_sessions = sessions[evaluated_mask]
        evaluated_regimes = regimes[evaluated_mask]
        historical_control = _paired_block_admission(
            (learned_pnl - historical_pnl)[evaluated_mask],
            times=evaluated_times,
            sessions=evaluated_sessions,
            regimes=evaluated_regimes,
        )
        historical_control.update(
            {
                "control": "actual_historical_fixed_1_xau_unit",
                "control_units_per_safety_admitted_order": 1,
                "learned_mean_account_net_pnl_bps": float(
                    np.mean(learned_pnl[evaluated_mask])
                ),
                "control_mean_account_net_pnl_bps": float(
                    np.mean(historical_pnl[evaluated_mask])
                ),
            }
        )

        evaluated_continuous = continuous_units[evaluated_mask]
        continuous_caps = (
            capacities[evaluated_mask].astype(np.float64)
            * MODEL_NATIVE_SIZING_MAX_CAPACITY_FRACTION
        )
        equal_continuous = _capacity_capped_equal_total_continuous(
            continuous_caps, float(np.sum(evaluated_continuous))
        )
        allocation_control = _paired_block_admission(
            account_bps_per_unit[evaluated_mask]
            * (evaluated_continuous - equal_continuous),
            times=evaluated_times,
            sessions=evaluated_sessions,
            regimes=evaluated_regimes,
        )
        allocation_control.update(
            {
                "control": "capacity_capped_equal_total_continuous_units",
                "learned_total_continuous_units": float(
                    np.sum(evaluated_continuous)
                ),
                "control_total_continuous_units": float(np.sum(equal_continuous)),
                "absolute_total_units_difference": float(
                    abs(np.sum(evaluated_continuous) - np.sum(equal_continuous))
                ),
            }
        )

        integer_caps = np.floor(continuous_caps).astype(np.int64)
        rounded_equal = _deterministic_equal_total_integer(
            integer_caps, int(np.sum(actual_units[evaluated_mask]))
        )
        rounded_delta = account_bps_per_unit[evaluated_mask] * (
            actual_units[evaluated_mask] - rounded_equal
        )
        rounded_diagnostic = {
            "role": "diagnostic_only_not_admission",
            "allocation": "chronology_stable_capacity_capped_near_equal_integer_units",
            "learned_total_integer_units": int(np.sum(actual_units[evaluated_mask])),
            "control_total_integer_units": int(np.sum(rounded_equal)),
            "row_delta_mean_bps": float(np.mean(rounded_delta)),
            "row_delta_sample_std_bps": (
                float(np.std(rounded_delta, ddof=1))
                if len(rounded_delta) > 1
                else 0.0
            ),
        }

        exposure_fractions = np.divide(
            actual_units,
            capacities,
            out=np.zeros(rows, dtype=np.float64),
            where=capacities > 0,
        )
        margin_fractions = (
            actual_units
            * outcome_numeric["mark_price"]
            * float(instrument["margin_rate"])
            / float(scenario["account_equity"])
        )
        exposure_pass = bool(
            np.max(exposure_fractions, initial=0.0)
            <= MODEL_NATIVE_SIZING_MAX_CAPACITY_FRACTION + 1e-12
            and np.max(margin_fractions, initial=0.0)
            <= MODEL_NATIVE_SIZING_MAX_ACCOUNT_MARGIN_FRACTION + 1e-12
        )
        observed_drawdown = _max_drawdown(learned_pnl)
        drawdown_pass = (
            observed_drawdown
            <= MODEL_NATIVE_SIZING_MAX_OOS_CUMULATIVE_DRAWDOWN_BPS
        )
        utility_pass = (
            historical_control["decision"] == "PASS"
            and allocation_control["decision"] == "PASS"
        )
        scenario_pass = exposure_pass and drawdown_pass and utility_pass
        utility_scenarios[scenario_name] = {
            "decision": "PASS" if utility_pass else "FAIL",
            "evaluated_orders": evaluated,
            "historical_1_unit_control": historical_control,
            "equal_total_continuous_allocation_control": allocation_control,
            "rounded_equal_total_allocation_diagnostic": rounded_diagnostic,
        }
        exposure_scenarios[scenario_name] = {
            "decision": "PASS" if exposure_pass else "FAIL",
            "evaluated_orders": evaluated,
            "capacity_fraction_breach_count": int(
                np.count_nonzero(
                    exposure_fractions
                    > MODEL_NATIVE_SIZING_MAX_CAPACITY_FRACTION + 1e-12
                )
            ),
            "margin_fraction_breach_count": int(
                np.count_nonzero(
                    margin_fractions
                    > MODEL_NATIVE_SIZING_MAX_ACCOUNT_MARGIN_FRACTION + 1e-12
                )
            ),
            "max_observed_capacity_fraction": float(
                np.max(exposure_fractions, initial=0.0)
            ),
            "max_observed_margin_fraction": float(
                np.max(margin_fractions, initial=0.0)
            ),
        }
        drawdown_scenarios[scenario_name] = {
            "decision": "PASS" if drawdown_pass else "FAIL",
            "observed_max_drawdown_bps": observed_drawdown,
            "policy_max_drawdown_bps": (
                MODEL_NATIVE_SIZING_MAX_OOS_CUMULATIVE_DRAWDOWN_BPS
            ),
        }
        grid_scenarios[scenario_name] = {
            "decision": "PASS" if scenario_pass else "FAIL",
            "account_facts": dict(scenario),
            "evaluated_orders": evaluated,
            "minimum_capacity_units": int(np.min(capacities[evaluated_mask])),
            "maximum_capacity_units": int(np.max(capacities[evaluated_mask])),
            "learned_total_integer_units": int(np.sum(actual_units[evaluated_mask])),
        }

    grid_admitted = all(
        row["decision"] == "PASS" for row in grid_scenarios.values()
    )

    direction_mismatch = int(np.count_nonzero(replay_directions != directions))
    direction_edge_policy = sizing_direction_edge_policy_metadata()
    direction_edge_admission = recompute_direction_edge_admission(
        predicted=directions,
        target=targets,
        context=f"{context}.direction_edge",
    )
    utc_ns = times.astype("int64").to_numpy(np.int64)
    return {
        "full_test_coverage": {
            "rows": rows,
            "first_utc": times.iloc[0].isoformat(),
            "last_utc": times.iloc[-1].isoformat(),
            "utc_ns_sha256": hashlib.sha256(utc_ns.tobytes()).hexdigest(),
        },
        "position_size_head_liveness": {
            "decision": "PASS" if float(np.std(logits)) > MODEL_NATIVE_SIZING_HEAD_VARIATION_EPSILON else "FAIL",
            "rows": rows,
            "finite": True,
            "std": float(np.std(logits)),
        },
        "monotonicity": {
            "decision": "PASS" if violation_count == 0 else "FAIL",
            "checked_pairs": max(rows - 1, 0),
            "violation_count": violation_count,
            "min_fraction_delta": float(np.min(fraction_delta, initial=0.0)),
        },
        "exposure_bounds": {
            "decision": (
                "PASS"
                if all(
                    row["decision"] == "PASS"
                    for row in exposure_scenarios.values()
                )
                else "FAIL"
            ),
            "scenarios": exposure_scenarios,
        },
        "drawdown_bounds": {
            "decision": (
                "PASS"
                if all(
                    row["decision"] == "PASS"
                    for row in drawdown_scenarios.values()
                )
                else "FAIL"
            ),
            "scenarios": drawdown_scenarios,
        },
        "paired_oos_utility": {
            "decision": (
                "PASS"
                if all(
                    row["decision"] == "PASS"
                    for row in utility_scenarios.values()
                )
                else "FAIL"
            ),
            "required_controls": [
                "actual_historical_fixed_1_xau_unit",
                "capacity_capped_equal_total_continuous_units",
            ],
            "scenarios": utility_scenarios,
        },
        "account_capacity_grid": {
            "decision": "PASS" if grid_admitted else "FAIL",
            "policy": reference_policy,
            "required_scenarios": list(reference_policy["scenario_order"]),
            "scenarios": grid_scenarios,
        },
        "direction_edge_policy": direction_edge_policy,
        "direction_edge_admission": direction_edge_admission,
        "direction_invariance": {
            "decision": "PASS" if direction_mismatch == 0 else "FAIL",
            "compared_rows": rows,
            "direction_mismatch_count": direction_mismatch,
        },
    }


def require_sizing_oos_proof_artifact(
    payload: Mapping[str, Any] | Any,
    *,
    calibration: Mapping[str, Any],
    calibration_artifact_sha256: str,
    context: str,
    verify_source_files: bool,
) -> dict[str, Any]:
    observed = _exact_keys(payload, _PROOF_KEYS, context=context)
    if observed["schema_version"] != MODEL_NATIVE_SIZING_OOS_PROOF_SCHEMA_VERSION:
        _fail(context, "schema_version mismatch")
    _utc(observed["created_utc"], context=f"{context}.created_utc")
    json_path = Path(str(observed["json_path"] or "")).expanduser()
    if not json_path.is_absolute():
        _fail(context, "json_path must be an absolute immutable self-reference")
    if observed["decision"] != "PASS" or observed["failures"] != []:
        _fail(context, "proof must be zero-failure PASS")
    if observed["evaluation_scope"] != MODEL_NATIVE_SIZING_OOS_SCOPE or observed[
        "evaluation_split"
    ] != MODEL_NATIVE_SIZING_HOLDOUT_SPLIT:
        _fail(context, "proof is not exact full TEST OOS")
    if _sha(observed["calibration_artifact_sha256"], context=context) != calibration_artifact_sha256:
        _fail(context, "proof calibration hash mismatch")
    if observed["risk_policy"] != sizing_risk_policy_metadata():
        _fail(context, "proof risk policy differs from immutable admission policy")
    bindings = _exact_keys(
        observed["source_bindings"], frozenset(_SOURCE_BINDING_NAMES), context=f"{context}.sources"
    )
    canonical_bindings = {
        name: _source_binding(
            bindings[name], context=f"{context}.sources.{name}", verify_file=verify_source_files
        )
        for name in _SOURCE_BINDING_NAMES
    }
    if canonical_bindings != bindings:
        _fail(context, "source binding canonicalization mismatch")
    source_payload, source_artifact = load_bound_sizing_oos_source(
        observed["oos_source_artifact"],
        calibration=calibration,
        calibration_artifact_sha256=calibration_artifact_sha256,
        context=f"{context}.oos_source",
        verify_source_files=verify_source_files,
    )
    provenance = source_payload["test_prediction_provenance"]
    evaluation = source_payload["evaluation_bundle"]
    if (
        source_payload["source_bindings"] != bindings
        or observed["test_prediction_provenance"] != provenance
        or observed["evaluation_bundle"] != evaluation
    ):
        _fail(context, "proof inputs differ from canonical OOS source event")
    if _utc(
        source_payload["created_utc"], context=f"{context}.oos_source.created_utc"
    ) >= _utc(observed["created_utc"], context=f"{context}.created_utc"):
        _fail(context, "OOS source event must predate proof event")
    section_names = (
        "full_test_coverage", "position_size_head_liveness", "monotonicity",
        "exposure_bounds", "drawdown_bounds", "paired_oos_utility",
        "account_capacity_grid",
        "direction_edge_policy", "direction_edge_admission",
        "direction_invariance",
    )
    if verify_source_files:
        recomputed = recompute_sizing_oos_evidence(
            calibration=calibration,
            source_bindings=bindings,
            evaluation_bundle=evaluation,
            context=f"{context}.recompute",
        )
        mismatched = [name for name in section_names if observed[name] != recomputed[name]]
        if mismatched:
            _fail(context, f"reported proof differs from row-level recomputation: {mismatched}")
    if observed["direction_edge_policy"] != sizing_direction_edge_policy_metadata():
        _fail(context, "direction edge policy binding mismatch")
    for name in (
        "position_size_head_liveness",
        "monotonicity",
        "exposure_bounds",
        "drawdown_bounds",
        "paired_oos_utility",
        "account_capacity_grid",
        "direction_edge_admission",
        "direction_invariance",
    ):
        section = observed[name]
        if not isinstance(section, Mapping) or section.get("decision") != "PASS":
            _fail(context, f"{name} must be recomputed PASS")
    utility = observed["paired_oos_utility"]
    scenarios = utility.get("scenarios") if isinstance(utility, Mapping) else None
    policy = sizing_oos_reference_account_policy_metadata()
    if not isinstance(scenarios, Mapping) or set(scenarios) != set(
        policy["scenario_order"]
    ):
        _fail(context, "paired OOS utility lacks exact account grid")
    for scenario_name in policy["scenario_order"]:
        scenario = scenarios[scenario_name]
        if not isinstance(scenario, Mapping) or scenario.get("decision") != "PASS":
            _fail(context, f"paired OOS scenario {scenario_name} must PASS")
        for control_name in (
            "historical_1_unit_control",
            "equal_total_continuous_allocation_control",
        ):
            control = scenario.get(control_name)
            if not isinstance(control, Mapping) or control.get("decision") != "PASS":
                _fail(context, f"{scenario_name}.{control_name} must PASS")
            for block_name in ("iso_week_blocks", "month_blocks"):
                block = control.get(block_name)
                if not isinstance(block, Mapping) or _finite(
                    block.get("lower_95_bps"),
                    context=(
                        f"{context}.utility.{scenario_name}.{control_name}."
                        f"{block_name}.lower"
                    ),
                ) <= 0.0:
                    _fail(
                        context,
                        f"{scenario_name}.{control_name}.{block_name} lower95 must be > 0",
                    )
    return observed


def load_bound_sizing_calibration(
    binding: Mapping[str, Any] | Any,
    *,
    context: str,
    verify_lineage_files: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    validated = require_immutable_json_binding(
        binding,
        event_prefix=_CALIBRATION_EVENT_PREFIX,
        context=f"{context}.binding",
        verify_file=True,
    )
    try:
        payload = json.loads(Path(validated["json_path"]).read_text(encoding="utf-8"))
    except Exception as exc:
        _fail(context, f"calibration artifact unreadable: {exc}")
    artifact = require_sizing_calibration_artifact(
        payload, context=context, verify_lineage_files=verify_lineage_files
    )
    if Path(str(artifact["json_path"])).expanduser().resolve() != Path(
        validated["json_path"]
    ):
        _fail(context, "calibration json_path differs from bound event")
    return artifact, validated


def load_bound_sizing_oos_proof(
    binding: Mapping[str, Any] | Any,
    *,
    calibration: Mapping[str, Any],
    calibration_artifact_sha256: str,
    context: str,
    verify_source_files: bool,
) -> tuple[dict[str, Any], dict[str, str]]:
    validated = require_immutable_json_binding(
        binding,
        event_prefix=_OOS_PROOF_EVENT_PREFIX,
        context=f"{context}.binding",
        verify_file=True,
    )
    try:
        payload = json.loads(Path(validated["json_path"]).read_text(encoding="utf-8"))
    except Exception as exc:
        _fail(context, f"OOS proof unreadable: {exc}")
    artifact = require_sizing_oos_proof_artifact(
        payload,
        calibration=calibration,
        calibration_artifact_sha256=calibration_artifact_sha256,
        context=context,
        verify_source_files=verify_source_files,
    )
    if Path(str(artifact["json_path"])).expanduser().resolve() != Path(
        validated["json_path"]
    ):
        _fail(context, "OOS proof json_path differs from bound event")
    return artifact, validated


__all__ = [
    "MODEL_NATIVE_SIZING_BUNDLE_CALIBRATION_SCHEMA_VERSION",
    "MODEL_NATIVE_SIZING_CALIBRATION_SCHEMA_VERSION",
    "MODEL_NATIVE_SIZING_FIT_SCOPE",
    "MODEL_NATIVE_SIZING_FIT_SPLITS",
    "MODEL_NATIVE_SIZING_HEAD_VARIATION_EPSILON",
    "MODEL_NATIVE_SIZING_HOLDOUT_SPLIT",
    "MODEL_NATIVE_SIZING_MAX_ACCOUNT_FLOATING_DRAWDOWN_BPS",
    "MODEL_NATIVE_SIZING_MAX_OOS_CUMULATIVE_DRAWDOWN_BPS",
    "MODEL_NATIVE_SIZING_MAX_ACCOUNT_MARGIN_FRACTION",
    "MODEL_NATIVE_SIZING_MAX_CAPACITY_FRACTION",
    "MODEL_NATIVE_SIZING_MAX_GROSS_XAU_UNITS",
    "MODEL_NATIVE_SIZING_MIN_OOS_ORDERS",
    "MODEL_NATIVE_SIZING_OOS_PROOF_SCHEMA_VERSION",
    "MODEL_NATIVE_SIZING_OOS_SCOPE",
    "MODEL_NATIVE_SIZING_RISK_POLICY_SCHEMA_VERSION",
    "MODEL_NATIVE_SIZING_RUNTIME_CONSTRAINT_KEYS",
    "MODEL_NATIVE_SIZING_TRANSFORM_VERSION",
    "ModelNativeSizingContractError",
    "calibrated_sizing_transform",
    "load_bound_sizing_calibration",
    "load_bound_sizing_oos_proof",
    "model_native_sizing_bundle_calibration_metadata",
    "recompute_direction_edge_admission",
    "recompute_sizing_oos_evidence",
    "require_immutable_json_binding",
    "require_model_native_sizing_bundle_calibration",
    "require_runtime_sizing_constraints",
    "require_sizing_evaluation_bundle",
    "require_sizing_calibration_artifact",
    "require_sizing_oos_proof_artifact",
    "require_sizing_prediction_provenance",
    "sha256_file",
    "sizing_direction_edge_policy_metadata",
    "sizing_offline_instrument_constraints_metadata",
    "sizing_risk_policy_metadata",
]
