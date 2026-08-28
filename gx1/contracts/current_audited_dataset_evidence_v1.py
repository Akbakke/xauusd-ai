"""Fail-closed binding for a reviewed, but deliberately unadmitted, dataset.

The launch-state file may name one explicit dataset review.  This module makes
that status useful without turning it into an admission path: every named
report is content-addressed, no ``latest`` discovery is permitted, and the
only accepted V1 status still says that production economics blocks use of the
research fitted-Q target. The selected V46 review repaired the older same-close
auxiliary-label defect; that causal PASS must not be confused with admission.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gx1.contracts.entry_fitted_q_v1 import (
    require_entry_fitted_q_production_economics_readiness,
)
from gx1.contracts.entry_execution_causality_v1 import (
    ENTRY_EXECUTION_CAUSALITY_AUDIT_SCHEMA_VERSION,
    require_entry_execution_causality_audit,
)
from gx1.contracts.entry_model_native_readiness_v1 import sha256_file


CURRENT_AUDITED_DATASET_EVIDENCE_SCHEMA_VERSION = (
    "gx1_current_audited_dataset_evidence_v1"
)
CURRENT_AUDITED_DATASET_STATUS = (
    "AUDITED_REPORT_ONLY_PRODUCTION_ECONOMICS_BLOCKED"
)
CURRENT_AUDITED_DATASET_BLOCKER = (
    "ENTRY_FITTED_Q_PRODUCTION_ECONOMICS_NOT_BOUND"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# These are report kinds, not a mutable lifecycle.  A new dataset review must
# declare all of them explicitly; this contract never selects a newer file.
_REQUIRED_REPORTS: dict[str, tuple[str, str]] = {
    "rebuild_terminal": (
        "entry_model_native_seq513_dataset_rebuild_terminal_v1",
        "COMPLETED_MODEL_NATIVE_SEQ513_DATASET_REBUILD",
    ),
    "post_rebuild_readiness": (
        "entry_model_native_seq513_post_rebuild_readiness_v2",
        "READY_FOR_MODEL_NATIVE_SEQ513_POST_REBUILD_REVIEW",
    ),
    "full_input_liveness": (
        "entry_full_input_liveness_contract_v9",
        "PASS",
    ),
    "feature_audit": ("entry_feature_foundation_audit_v1", "PASS"),
    "target_audit": ("entry_target_foundation_audit_v4", "PASS"),
    "specialist_audit": (
        "entry_specialist_feature_group_audit_v1",
        "PASS",
    ),
    "smoke_manifest": (
        "entry_model_native_seq513_smoke_manifest_v3",
        "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_REVIEW",
    ),
    "smoke_readiness": (
        "entry_model_native_seq513_smoke_readiness_v3",
        "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_READINESS_REVIEW",
    ),
    "trainability_readiness": (
        "entry_model_native_seq513_trainability_readiness_v1",
        "READY_FOR_MODEL_NATIVE_SEQ513_TRAINABILITY_REVIEW",
    ),
    "train_recipe": ("entry_model_native_seq513_train_recipe_audit_v9", "PASS"),
    "execution_causality": (
        ENTRY_EXECUTION_CAUSALITY_AUDIT_SCHEMA_VERSION,
        "PASS",
    ),
    "adoption_candidate": (
        "entry_model_native_adoption_candidate_v1",
        "BLOCKED_MODEL_NATIVE_ADOPTION_REVIEW",
    ),
}
_ENTRY_FITTED_Q_BLOCKED_CHECK = (
    "target audit proves canonical aux targets and production-ready Entry-Q"
)


def _require_regular_path(value: Any, *, label: str, root: Path | None = None) -> Path:
    raw = Path(str(value or "")).expanduser()
    if not raw.is_absolute() or any("latest" in part.lower() for part in raw.parts):
        raise RuntimeError(f"[{label}_PATH_INVALID]")
    path = raw.absolute()
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise RuntimeError(f"[{label}_PATH_NOT_REGULAR]")
    if root is not None:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(f"[{label}_OUTSIDE_ROOT]") from exc
        current = path.parent
        while current != root:
            if current.is_symlink():
                raise RuntimeError(f"[{label}_SYMLINK_ANCESTOR]")
            current = current.parent
    return path


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"[{label}_JSON_INVALID] {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"[{label}_JSON_ROOT_INVALID]")
    return payload


def _require_empty_failures(payload: Mapping[str, Any], *, label: str) -> None:
    failures = payload.get("failures")
    if failures not in (None, []):
        raise RuntimeError(f"[{label}_FAILURES_PRESENT]")


def _require_dataset_binding(
    payload: Mapping[str, Any], *, dataset_dir: Path, dataset_run_id: str, label: str
) -> None:
    # A few readiness reports carry their identity under the embedded smoke
    # manifest instead of at the top level.  Those are handled explicitly.
    observed_dir = payload.get("dataset_dir")
    observed_run = payload.get("dataset_run_id")
    smoke = payload.get("smoke_manifest")
    if isinstance(smoke, Mapping):
        observed_dir = smoke.get("dataset_dir", observed_dir)
        observed_run = smoke.get("dataset_run_id", observed_run)
    if observed_dir is not None and Path(str(observed_dir)).expanduser().absolute() != dataset_dir:
        raise RuntimeError(f"[{label}_DATASET_PATH_MISMATCH]")
    if observed_run is not None and observed_run != dataset_run_id:
        raise RuntimeError(f"[{label}_DATASET_RUN_ID_MISMATCH]")


def _require_adoption_block(payload: Mapping[str, Any]) -> None:
    failures = payload.get("failures")
    if not isinstance(failures, list) or len(failures) != 1:
        raise RuntimeError("[AUDITED_ADOPTION_FAILURE_SHAPE_INVALID]")
    failure = failures[0]
    if not isinstance(failure, Mapping):
        raise RuntimeError("[AUDITED_ADOPTION_FAILURE_INVALID]")
    details = failure.get("details")
    if (
        failure.get("gate") != "target_audit"
        or failure.get("check") != _ENTRY_FITTED_Q_BLOCKED_CHECK
        or not isinstance(details, Mapping)
        or details.get("aux_contract_valid") is not True
        or details.get("aux_contract_error") is not None
        or details.get("entry_fitted_q_target_valid") is not False
        or tuple(details.get("expected_extra_active_target_heads") or ())
        != ("side_mae", "trendline_event")
        or tuple(details.get("observed_extra_active_target_heads") or ())
        != ("side_mae", "trendline_event")
        or details.get("extra_active_target_head_liveness")
        != {"side_mae": True, "trendline_event": True}
    ):
        raise RuntimeError("[AUDITED_ADOPTION_BLOCKER_NOT_PRODUCTION_ECONOMICS]")
    readiness = payload.get("model_native_readiness_contract")
    economics = (
        readiness.get("entry_fitted_q_production_economics")
        if isinstance(readiness, Mapping)
        else None
    )
    require_entry_fitted_q_production_economics_readiness(
        economics, context="AUDITED_ADOPTION", require_ready=False
    )
    for key in (
        "adoption_evidence_ready",
        "candidate_ready_for_activation",
        "direction_selection_authority",
        "training_allowed",
        "replay_allowed",
        "shadow_live_allowed",
    ):
        if payload.get(key) is not False:
            raise RuntimeError(f"[AUDITED_ADOPTION_{key.upper()}_OPEN]")


def _require_execution_causality_pass(
    payload: Mapping[str, Any], *, dataset_dir: Path, dataset_run_id: str
) -> None:
    """Require the repaired M1 decision-to-fill evidence without admission."""

    report = require_entry_execution_causality_audit(
        payload,
        expected_dataset_dir=str(dataset_dir),
        expected_entry_run_id=dataset_run_id,
        require_training_authorized=True,
    )
    if (
        report["training_authorized"] is not True
        or report["legacy_m5_same_close_label_present"] is not False
        or report["entry_fitted_q_m1_fill_lifecycle_bound"] is not True
        or report["active_auxiliary_targets_m1_fill_bound"] is not True
        or report["future_causal_rebuild_required"] is not False
    ):
        raise RuntimeError("[AUDITED_EXECUTION_CAUSALITY_PASS_INVALID]")


def require_current_audited_dataset_evidence(
    value: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Validate one explicit reviewed dataset without granting it admission."""

    if not isinstance(value, Mapping):
        raise RuntimeError("[CURRENT_AUDITED_DATASET_EVIDENCE_MISSING]")
    evidence = dict(value)
    expected_keys = {
        "schema_version",
        "status",
        "blocker",
        "dataset_run_id",
        "root_dir",
        "dataset_dir",
        "admission_allowed",
        "activation_allowed",
        "reports",
    }
    if set(evidence) != expected_keys:
        raise RuntimeError("[CURRENT_AUDITED_DATASET_EVIDENCE_SCHEMA_INVALID]")
    if (
        evidence["schema_version"] != CURRENT_AUDITED_DATASET_EVIDENCE_SCHEMA_VERSION
        or evidence["status"] != CURRENT_AUDITED_DATASET_STATUS
        or evidence["blocker"] != CURRENT_AUDITED_DATASET_BLOCKER
        or evidence["admission_allowed"] is not False
        or evidence["activation_allowed"] is not False
        or not isinstance(evidence["dataset_run_id"], str)
        or not evidence["dataset_run_id"]
    ):
        raise RuntimeError("[CURRENT_AUDITED_DATASET_EVIDENCE_STATUS_INVALID]")

    root = Path(str(evidence["root_dir"] or "")).expanduser().absolute()
    dataset_dir = Path(str(evidence["dataset_dir"] or "")).expanduser().absolute()
    if (
        not root.is_absolute()
        or root.is_symlink()
        or not root.is_dir()
        or dataset_dir != root / "dataset"
        or dataset_dir.is_symlink()
        or not dataset_dir.is_dir()
    ):
        raise RuntimeError("[CURRENT_AUDITED_DATASET_ROOT_INVALID]")
    reports = evidence["reports"]
    if not isinstance(reports, Mapping) or set(reports) != set(_REQUIRED_REPORTS):
        raise RuntimeError("[CURRENT_AUDITED_DATASET_REPORT_SET_INVALID]")

    validated: dict[str, dict[str, Any]] = {}
    for name, (expected_schema, expected_decision) in _REQUIRED_REPORTS.items():
        row = reports[name]
        if not isinstance(row, Mapping) or set(row) != {
            "path",
            "sha256",
            "schema_version",
            "decision",
        }:
            raise RuntimeError(f"[CURRENT_AUDITED_DATASET_{name.upper()}_ROW_INVALID]")
        if row["schema_version"] != expected_schema or row["decision"] != expected_decision:
            raise RuntimeError(f"[CURRENT_AUDITED_DATASET_{name.upper()}_EXPECTATION_INVALID]")
        digest = str(row["sha256"] or "")
        if _SHA256_RE.fullmatch(digest) is None:
            raise RuntimeError(f"[CURRENT_AUDITED_DATASET_{name.upper()}_HASH_INVALID]")
        path = _require_regular_path(row["path"], label=f"CURRENT_AUDITED_{name.upper()}", root=root)
        observed_sha = sha256_file(path)
        if observed_sha != digest:
            raise RuntimeError(
                f"[CURRENT_AUDITED_DATASET_{name.upper()}_HASH_MISMATCH]"
            )
        payload = _read_json(path, label=f"CURRENT_AUDITED_{name.upper()}")
        if (
            payload.get("schema_version") != expected_schema
            or payload.get("decision") != expected_decision
        ):
            raise RuntimeError(f"[CURRENT_AUDITED_DATASET_{name.upper()}_REPORT_INVALID]")
        _require_dataset_binding(
            payload,
            dataset_dir=dataset_dir,
            dataset_run_id=evidence["dataset_run_id"],
            label=f"CURRENT_AUDITED_{name.upper()}",
        )
        if name == "adoption_candidate":
            _require_adoption_block(payload)
        elif name == "execution_causality":
            _require_execution_causality_pass(
                payload,
                dataset_dir=dataset_dir,
                dataset_run_id=evidence["dataset_run_id"],
            )
        else:
            _require_empty_failures(payload, label=f"CURRENT_AUDITED_{name.upper()}")
        validated[name] = {"path": str(path), "sha256": observed_sha}

    recipe = _read_json(Path(validated["train_recipe"]["path"]), label="CURRENT_AUDITED_RECIPE")
    if recipe.get("execution_allowed") is not True or recipe.get("activation_authority") is not False:
        raise RuntimeError("[CURRENT_AUDITED_DATASET_RECIPE_AUTHORITY_INVALID]")
    return {
        "status": CURRENT_AUDITED_DATASET_STATUS,
        "blocker": CURRENT_AUDITED_DATASET_BLOCKER,
        "dataset_run_id": evidence["dataset_run_id"],
        "dataset_dir": str(dataset_dir),
        "report_count": len(validated),
    }


def require_blocked_launch_state_with_current_audited_dataset(
    value: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Require that reviewed evidence never changes the launch state's BLOCK."""

    if not isinstance(value, Mapping):
        raise RuntimeError("[CURRENT_AUDITED_LAUNCH_STATE_MISSING]")
    state = dict(value)
    if (
        state.get("decision") != "BLOCK"
        or state.get("latest_terminal_event_id") != "NO_CURRENT_ADMITTED_EVENT"
        or state.get("latest_terminal_event_decision") != "BLOCK"
        or state.get("dataset_event_id") is not None
        or state.get("dataset_admission_stage") != "NO_ADMITTED_UNIFIED_DATASET"
        or state.get("accepted_dataset_dir") is not None
        or state.get("accepted_dataset_terminal_evidence") is not None
        or state.get("accepted_bundle_dir") is not None
        or state.get("bundle_metadata_sha256") is not None
        or state.get("current_smoke_launch_evidence") is not None
        or state.get("accepted_via_vedtak") is not None
    ):
        raise RuntimeError("[CURRENT_AUDITED_LAUNCH_STATE_NOT_FAIL_CLOSED]")
    return require_current_audited_dataset_evidence(
        state.get("current_audited_dataset_evidence")
    )
