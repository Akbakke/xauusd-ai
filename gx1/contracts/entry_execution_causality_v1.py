"""Fail-closed audit contract for Entry decision-to-fill causality.

The Entry fitted-Q teacher already has a dedicated M1 lifecycle.  Older
diagnostic and sizing labels, however, were derived from M5 same-close quotes.
Those labels are still consumed by active auxiliary heads, so they must be
bound to the same decision/fill timeline before a trainer is allowed to run.
This module is deliberately an audit/launch boundary only: it cannot make an
old label causal by declaration.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


ENTRY_EXECUTION_CAUSALITY_AUDIT_SCHEMA_VERSION = (
    "entry_execution_causality_audit_v1"
)
ENTRY_EXECUTION_CAUSALITY_AUDIT_DECISION_PASS = "PASS"
ENTRY_EXECUTION_CAUSALITY_AUDIT_DECISION_BLOCK = "BLOCK"
ENTRY_EXECUTION_CAUSALITY_REQUIRED_SPLITS = ("train", "val")

_SHA256_HEX = frozenset("0123456789abcdef")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "decision",
        "training_authorized",
        "dataset_dir",
        "entry_run_id",
        "signal_manifest_path",
        "signal_manifest_sha256",
        "legacy_m5_same_close_label_present",
        "entry_fitted_q_m1_fill_lifecycle_bound",
        "active_auxiliary_targets_m1_fill_bound",
        "future_causal_rebuild_required",
        "splits",
        "failures",
        "remediation",
    }
)


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value)
    )


def legacy_same_close_target_contract_failures(value: Any) -> list[str]:
    """Return exact reasons why a ranking target lacks a causal fill proof."""

    if not isinstance(value, Mapping):
        return ["ENTRY_EXECUTION_CAUSALITY_RANKING_TARGET_CONTRACT_MISSING"]
    failures: list[str] = []
    if value.get("long_entry_price") == "ask_close_t0":
        failures.append("ENTRY_EXECUTION_CAUSALITY_LONG_SAME_CLOSE_ENTRY")
    if value.get("short_entry_price") == "bid_close_t0":
        failures.append("ENTRY_EXECUTION_CAUSALITY_SHORT_SAME_CLOSE_ENTRY")
    required = {
        "entry_decision_time": "authoritative_m5_bar_close_available_at",
        "long_entry_price": (
            "ask_open_first_authoritative_m1_at_or_after_entry_decision"
        ),
        "short_entry_price": (
            "bid_open_first_authoritative_m1_at_or_after_entry_decision"
        ),
        "long_exit_price": (
            "bid_open_first_authoritative_m1_at_or_after_fitted_exit_decision"
        ),
        "short_exit_price": (
            "ask_open_first_authoritative_m1_at_or_after_fitted_exit_decision"
        ),
        "entry_fill_binding": "exact_m1_quote_time_and_bid_ask",
        "target_affects_feature_availability": False,
    }
    missing_or_wrong = [
        key for key, expected in required.items() if value.get(key) != expected
    ]
    if missing_or_wrong:
        failures.append(
            "ENTRY_EXECUTION_CAUSALITY_M1_FILL_CONTRACT_UNBOUND:"
            + ",".join(missing_or_wrong)
        )
    return failures


def _require_bool(value: Any, *, label: str) -> bool:
    if type(value) is not bool:
        raise RuntimeError(f"ENTRY_EXECUTION_CAUSALITY_{label}_INVALID")
    return value


def require_entry_execution_causality_audit(
    value: Any,
    *,
    expected_dataset_dir: str | None = None,
    expected_entry_run_id: str | None = None,
    require_training_authorized: bool,
) -> dict[str, Any]:
    """Validate a content-addressed causality report and optionally require PASS."""

    if not isinstance(value, Mapping) or set(value) != _TOP_LEVEL_KEYS:
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_AUDIT_SCHEMA_INVALID")
    report = json.loads(json.dumps(value, sort_keys=True, allow_nan=False))
    if report["schema_version"] != ENTRY_EXECUTION_CAUSALITY_AUDIT_SCHEMA_VERSION:
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_AUDIT_VERSION_INVALID")
    decision = report["decision"]
    if decision not in {
        ENTRY_EXECUTION_CAUSALITY_AUDIT_DECISION_PASS,
        ENTRY_EXECUTION_CAUSALITY_AUDIT_DECISION_BLOCK,
    }:
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_AUDIT_DECISION_INVALID")
    training_authorized = _require_bool(
        report["training_authorized"], label="TRAINING_AUTHORIZATION"
    )
    legacy_same_close = _require_bool(
        report["legacy_m5_same_close_label_present"],
        label="LEGACY_SAME_CLOSE",
    )
    fitted_q_bound = _require_bool(
        report["entry_fitted_q_m1_fill_lifecycle_bound"],
        label="FITTED_Q_FILL_BINDING",
    )
    auxiliary_bound = _require_bool(
        report["active_auxiliary_targets_m1_fill_bound"],
        label="AUXILIARY_FILL_BINDING",
    )
    future_rebuild = _require_bool(
        report["future_causal_rebuild_required"], label="REBUILD_REQUIRED"
    )
    if not isinstance(report["dataset_dir"], str) or not report["dataset_dir"]:
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_AUDIT_DATASET_INVALID")
    if expected_dataset_dir is not None and report["dataset_dir"] != expected_dataset_dir:
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_AUDIT_DATASET_MISMATCH")
    if not isinstance(report["entry_run_id"], str) or not report["entry_run_id"]:
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_AUDIT_RUN_ID_INVALID")
    if expected_entry_run_id is not None and report["entry_run_id"] != expected_entry_run_id:
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_AUDIT_RUN_ID_MISMATCH")
    if not isinstance(report["signal_manifest_path"], str) or not report[
        "signal_manifest_path"
    ]:
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_AUDIT_SIGNAL_PATH_INVALID")
    if not is_sha256(report["signal_manifest_sha256"]):
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_AUDIT_SIGNAL_HASH_INVALID")
    failures = report["failures"]
    remediation = report["remediation"]
    if (
        not isinstance(failures, list)
        or not all(isinstance(item, str) and item for item in failures)
        or not isinstance(remediation, list)
        or not all(isinstance(item, str) and item for item in remediation)
    ):
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_AUDIT_DETAILS_INVALID")
    rows = report["splits"]
    if not isinstance(rows, list) or len(rows) != len(
        ENTRY_EXECUTION_CAUSALITY_REQUIRED_SPLITS
    ):
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_AUDIT_SPLITS_INVALID")
    by_split = {
        str(row.get("split") or ""): row for row in rows if isinstance(row, Mapping)
    }
    if set(by_split) != set(ENTRY_EXECUTION_CAUSALITY_REQUIRED_SPLITS):
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_AUDIT_SPLIT_SET_INVALID")
    for split in ENTRY_EXECUTION_CAUSALITY_REQUIRED_SPLITS:
        row = by_split[split]
        required_keys = {
            "split",
            "dataset_manifest_path",
            "dataset_manifest_sha256",
            "lifecycle_manifest_path",
            "lifecycle_manifest_sha256",
            "entry_fitted_q_m1_fill_lifecycle_bound",
            "active_auxiliary_targets_m1_fill_bound",
        }
        if set(row) != required_keys:
            raise RuntimeError(
                f"ENTRY_EXECUTION_CAUSALITY_AUDIT_SPLIT_SCHEMA_INVALID:{split}"
            )
        if (
            not isinstance(row["dataset_manifest_path"], str)
            or not is_sha256(row["dataset_manifest_sha256"])
            or not isinstance(row["lifecycle_manifest_path"], str)
            or not is_sha256(row["lifecycle_manifest_sha256"])
            or type(row["entry_fitted_q_m1_fill_lifecycle_bound"]) is not bool
            or type(row["active_auxiliary_targets_m1_fill_bound"]) is not bool
        ):
            raise RuntimeError(
                f"ENTRY_EXECUTION_CAUSALITY_AUDIT_SPLIT_VALUES_INVALID:{split}"
            )
    pass_expected = fitted_q_bound and auxiliary_bound and not legacy_same_close
    if (
        training_authorized != pass_expected
        or (decision == ENTRY_EXECUTION_CAUSALITY_AUDIT_DECISION_PASS)
        != pass_expected
        or future_rebuild != (not pass_expected)
        or bool(failures) != (not pass_expected)
    ):
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_AUDIT_OUTCOME_INCONSISTENT")
    if any(
        row["entry_fitted_q_m1_fill_lifecycle_bound"] != fitted_q_bound
        or row["active_auxiliary_targets_m1_fill_bound"] != auxiliary_bound
        for row in by_split.values()
    ):
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_AUDIT_SPLIT_OUTCOME_MISMATCH")
    if require_training_authorized and not training_authorized:
        raise RuntimeError("ENTRY_EXECUTION_CAUSALITY_AUDIT_TRAINING_BLOCKED")
    return report


def build_entry_execution_causality_audit(
    *,
    dataset_dir: str,
    entry_run_id: str,
    signal_manifest_path: str,
    signal_manifest_sha256: str,
    ranking_target_contract: Any,
    split_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create a report from verified manifests; never infer quote bindings."""

    target_failures = legacy_same_close_target_contract_failures(
        ranking_target_contract
    )
    auxiliary_bound = not target_failures
    fitted_q_bound = bool(split_rows) and all(
        bool(row.get("entry_fitted_q_m1_fill_lifecycle_bound"))
        for row in split_rows
    )
    training_authorized = fitted_q_bound and auxiliary_bound
    failures = list(target_failures)
    if not fitted_q_bound:
        failures.append("ENTRY_EXECUTION_CAUSALITY_FITTED_Q_M1_LIFECYCLE_UNBOUND")
    remediation = []
    if not training_authorized:
        remediation = [
            "rebuild_all_active_auxiliary_outcomes_from_exact_m1_fill_quotes",
            "bind_each_entry_decision_and_exit_quote_time_in_split_evidence",
            "rerun_entry_execution_causality_audit_before_training",
        ]
    report = {
        "schema_version": ENTRY_EXECUTION_CAUSALITY_AUDIT_SCHEMA_VERSION,
        "decision": (
            ENTRY_EXECUTION_CAUSALITY_AUDIT_DECISION_PASS
            if training_authorized
            else ENTRY_EXECUTION_CAUSALITY_AUDIT_DECISION_BLOCK
        ),
        "training_authorized": training_authorized,
        "dataset_dir": dataset_dir,
        "entry_run_id": entry_run_id,
        "signal_manifest_path": signal_manifest_path,
        "signal_manifest_sha256": signal_manifest_sha256,
        "legacy_m5_same_close_label_present": bool(target_failures),
        "entry_fitted_q_m1_fill_lifecycle_bound": fitted_q_bound,
        "active_auxiliary_targets_m1_fill_bound": auxiliary_bound,
        "future_causal_rebuild_required": not training_authorized,
        "splits": [dict(row) for row in split_rows],
        "failures": failures,
        "remediation": remediation,
    }
    return require_entry_execution_causality_audit(
        report, require_training_authorized=False
    )
