#!/usr/bin/env python3
"""Verify immutable model-native seq513 evidence before adoption review.

This report-only gate never activates a dataset, starts training or replay,
selects a trading direction, or touches shadow/live state.  It accepts only
explicit newest immutable evidence and fails closed on any partial contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gx1.contracts.entry_foundation_audit_policy_v1 import (
    FOUNDATION_AUDIT_DATA_SPLITS,
    FOUNDATION_TARGET_AUDIT_SCHEMA_VERSION,
    foundation_audit_policy_binding,
    require_foundation_audit_report_policy,
)
from gx1.contracts.entry_model_native_aux_targets_v3 import (
    MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS,
    require_model_native_aux_target_contract,
)
from gx1.contracts.entry_fitted_q_v1 import (
    require_entry_fitted_q_contract,
    require_entry_fitted_q_production_economics_readiness,
)
from gx1.contracts.entry_dataset_split_artifacts_v1 import (
    ENTRY_DATASET_SPLIT_ARTIFACTS_SCHEMA_VERSION,
)
from gx1.contracts.entry_model_native_readiness_v1 import (
    MODEL_NATIVE_BASE_ACTIVE_HEADS,
    MODEL_NATIVE_BLOCKED_HEADS,
    MODEL_NATIVE_REQUIRED_SPECIALISTS,
    artifact_fingerprint_checks,
    artifact_fingerprints,
    model_native_readiness_contract_metadata,
    readiness_check as _check,
    require_model_native_readiness_contract,
    sha256_file,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_DIRECTION_LOGIT_MODE,
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT,
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION,
    require_model_native_signal_contract,
)
from gx1.contracts.immutable_event_authority_v1 import (
    require_newest_immutable_event,
    write_immutable_json_event,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT,
)


SPLITS = FOUNDATION_AUDIT_DATA_SPLITS
REPORT_SCHEMA_VERSION = "entry_model_native_adoption_candidate_v1"
EVENT_PREFIX = "ENTRY_MODEL_NATIVE_ADOPTION_CANDIDATE"
SMOKE_REPORT_SCHEMA = "entry_model_native_seq513_smoke_manifest_v3"
SMOKE_REPORT_DECISION = "READY_FOR_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST_REVIEW"
SMOKE_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SEQ513_SMOKE_MANIFEST"
SMOKE_DATASET_SCHEMA = "entry_model_native_seq513_smoke_dataset_v3"
SMOKE_SPLIT_SCHEMA = MODEL_NATIVE_SPLIT_MANIFEST_SCHEMA_VERSION
AUDIT_EVENT_PREFIXES = {
    "feature_audit": "ENTRY_FEATURE_FOUNDATION_AUDIT",
    "target_audit": "ENTRY_TARGET_FOUNDATION_AUDIT",
    "specialist_audit": "ENTRY_SPECIALIST_FEATURE_GROUP_AUDIT",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON evidence {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON evidence root is not an object: {path}")
    return payload


def _same_path(value: Any, expected: Path) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return Path(value).expanduser().resolve() == expected.resolve()
    except (OSError, RuntimeError):
        return False


def _sha256_json(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_equal(left: Any, right: Any) -> bool:
    try:
        return json.dumps(
            left,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) == json.dumps(
            right,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return False


def _declared_split_artifact(
    value: Any,
    expected_sha256: Any,
    *,
    dataset_dir: Path,
    label: str,
) -> Path:
    raw = Path(str(value or "")).expanduser()
    if (
        not raw.is_absolute()
        or raw.is_symlink()
        or not raw.is_file()
        or raw.resolve() != raw
        or raw.parent != dataset_dir
        or any("latest" in part.lower() for part in raw.parts)
    ):
        raise RuntimeError(
            f"[ADOPTION_DATASET_ARTIFACT_IDENTITY_INVALID] {label}={raw}"
        )
    expected = str(expected_sha256 or "").strip().lower()
    if len(expected) != 64 or any(ch not in "0123456789abcdef" for ch in expected):
        raise RuntimeError(f"[ADOPTION_DATASET_ARTIFACT_HASH_INVALID] {label}")
    observed = sha256_file(raw)
    if observed != expected:
        raise RuntimeError(
            f"[ADOPTION_DATASET_ARTIFACT_HASH_MISMATCH] {label} "
            f"expected={expected} observed={observed}"
        )
    return raw


def _dataset_contract(
    dataset_dir: Path,
    smoke_report: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Path]]:
    checks = [
        _check(
            "model-native candidate dataset directory exists",
            dataset_dir.is_dir(),
            {"dataset_dir": str(dataset_dir)},
        )
    ]
    split_rows: dict[str, dict[str, Any]] = {}
    artifacts: dict[str, Path] = {}
    contracts: list[dict[str, Any]] = []
    embedded = smoke_report.get("smoke_manifest")
    embedded_splits = (
        embedded.get("splits") if isinstance(embedded, dict) else None
    )
    if not isinstance(embedded_splits, dict) or set(embedded_splits) != set(SPLITS):
        raise RuntimeError("[ADOPTION_DATASET_SPLIT_IDENTITY_MISSING]")
    for split in SPLITS:
        declaration = embedded_splits[split]
        if not isinstance(declaration, dict):
            raise RuntimeError(
                f"[ADOPTION_DATASET_SPLIT_IDENTITY_INVALID] split={split}"
            )
        parquet = _declared_split_artifact(
            declaration.get("out_parquet"),
            declaration.get("out_parquet_sha256"),
            dataset_dir=dataset_dir,
            label=f"{split}_parquet",
        )
        manifest_path = _declared_split_artifact(
            declaration.get("out_manifest"),
            declaration.get("out_manifest_sha256"),
            dataset_dir=dataset_dir,
            label=f"{split}_manifest",
        )
        artifacts[f"dataset_{split}_parquet"] = parquet
        artifacts[f"dataset_{split}_manifest"] = manifest_path
        manifest = _read_json(manifest_path)
        extra = manifest.get("extra") if isinstance(manifest.get("extra"), dict) else {}
        bridge = (
            extra.get("signal_bridge")
            if isinstance(extra.get("signal_bridge"), dict)
            else {}
        )
        contract = (
            extra.get("model_native_signal_contract")
            if isinstance(extra.get("model_native_signal_contract"), dict)
            else {}
        )
        contract_error = ""
        try:
            require_model_native_signal_contract(
                contract,
                context=f"ADOPTION_DATASET_{split.upper()}",
            )
        except RuntimeError as exc:
            contract_error = str(exc)
        fields = [str(value) for value in bridge.get("fields", [])]
        output_path = str(manifest.get("output_data_path") or "")
        row = {
            "parquet_path": str(parquet),
            "manifest_path": str(manifest_path),
            "parquet_sha256": sha256_file(parquet),
            "manifest_sha256": sha256_file(manifest_path),
            "schema_version": manifest.get("schema_version"),
            "manifest_variant": manifest.get("manifest_variant"),
            "expected_seq_snap_width": manifest.get("expected_seq_snap_width"),
            "output_data_path": output_path,
            "contract_mode": extra.get("contract_mode"),
            "direction_logit_mode": extra.get("direction_logit_mode"),
            "seq_input_dim": bridge.get("seq_input_dim"),
            "snap_input_dim": bridge.get("snap_input_dim"),
            "field_count": len(fields),
            "contract_error": contract_error,
        }
        split_rows[split] = row
        if not contract_error:
            contracts.append(contract)
        checks.append(
            _check(
                f"{split} carries the exact model-native seq513 signal contract",
                not contract_error
                and manifest.get("schema_version") == SMOKE_SPLIT_SCHEMA
                and manifest.get("manifest_variant") == MODEL_NATIVE_CONTRACT_MODE
                and int(manifest.get("expected_seq_snap_width") or 0)
                == MODEL_NATIVE_SIGNAL_DIM
                and _same_path(output_path, parquet)
                and extra.get("contract_mode") == MODEL_NATIVE_CONTRACT_MODE
                and extra.get("direction_logit_mode")
                == MODEL_NATIVE_DIRECTION_LOGIT_MODE
                and int(bridge.get("seq_input_dim") or 0) == MODEL_NATIVE_SIGNAL_DIM
                and int(bridge.get("snap_input_dim") or 0) == MODEL_NATIVE_SIGNAL_DIM
                and fields == contract.get("fields"),
                row,
            )
        )
        if sha256_file(manifest_path) != str(
            declaration.get("out_manifest_sha256") or ""
        ).strip().lower():
            raise RuntimeError(
                f"[ADOPTION_DATASET_MANIFEST_CHANGED_DURING_VALIDATION] split={split}"
            )
    checks.append(
        _check(
            "train val signal contracts are identical",
            len(contracts) == len(SPLITS)
            and all(contract == contracts[0] for contract in contracts[1:]),
            {"validated_contract_count": len(contracts)},
        )
    )
    return checks, split_rows, artifacts


def _base_evidence_checks(
    report: dict[str, Any],
    *,
    schema_version: str,
    audit_kind: str,
    dataset_dir: Path,
    split_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    policy_error = ""
    try:
        require_foundation_audit_report_policy(
            report,
            audit_kind=audit_kind,
            context="ADOPTION_FOUNDATION_AUDIT",
        )
    except RuntimeError as exc:
        policy_error = str(exc)
    expected_split_artifacts = {
        split: {
            "manifest_path": row["manifest_path"],
            "manifest_sha256": row["manifest_sha256"],
            "parquet_path": row["parquet_path"],
            "parquet_sha256": row["parquet_sha256"],
        }
        for split, row in split_rows.items()
    }
    return [
        _check(
            "evidence schema is exact",
            report.get("schema_version") == schema_version,
            {"schema_version": report.get("schema_version")},
        ),
        _check(
            "evidence decision is PASS with zero failures",
            report.get("decision") == "PASS" and not report.get("failures"),
            {
                "decision": report.get("decision"),
                "failures": report.get("failures"),
            },
        ),
        _check(
            "evidence dataset matches the explicit seq513 candidate",
            _same_path(report.get("dataset_dir"), dataset_dir),
            {
                "expected_dataset_dir": str(dataset_dir),
                "reported_dataset_dir": report.get("dataset_dir"),
            },
        ),
        _check(
            "foundation audit policy identity and full payload are exact",
            not policy_error,
            {"error": policy_error},
        ),
        _check(
            "foundation audit covers exact train val split order",
            tuple(report.get("data_splits") or ())
            == FOUNDATION_AUDIT_DATA_SPLITS,
            {
                "expected": list(FOUNDATION_AUDIT_DATA_SPLITS),
                "observed": report.get("data_splits"),
            },
        ),
        _check(
            "foundation audit is content-bound to exact candidate split artifacts",
            report.get("split_artifacts_schema_version")
            == ENTRY_DATASET_SPLIT_ARTIFACTS_SCHEMA_VERSION
            and _canonical_equal(
                report.get("split_artifacts"),
                expected_split_artifacts,
            ),
            {
                "expected": expected_split_artifacts,
                "observed": report.get("split_artifacts"),
            },
        ),
    ]


def _feature_checks(
    report: dict[str, Any],
    dataset_dir: Path,
    split_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    signal_contract = (
        report.get("model_native_signal_contract")
        if isinstance(report.get("model_native_signal_contract"), dict)
        else {}
    )
    contract_error = ""
    try:
        require_model_native_signal_contract(
            signal_contract,
            context="ADOPTION_FEATURE_AUDIT",
        )
    except RuntimeError as exc:
        contract_error = str(exc)
    base_fields = tuple(str(value) for value in signal_contract.get("base_fields", []))
    selected_fields = tuple(
        str(value) for value in signal_contract.get("selected_fields", [])
    )
    mandatory_prefix = selected_fields[:MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT]
    available_candidates = selected_fields[
        MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT:
    ]
    ranking_sha256 = str(report.get("feature_ranking_sha256") or "")
    partition_exact = (
        not contract_error
        and int(report.get("model_native_signal_dim") or 0)
        == MODEL_NATIVE_SIGNAL_DIM
        and int(report.get("base_signal_dim") or 0)
        == MODEL_NATIVE_BASE_SIGNAL_DIM
        and tuple(str(value) for value in report.get("base_signal_fields", []))
        == MODEL_NATIVE_BASE_FIELDS
        and base_fields == MODEL_NATIVE_BASE_FIELDS
        and int(report.get("selected_feature_count") or 0)
        == MODEL_NATIVE_SELECTED_FEATURE_COUNT
        and int(report.get("manifest_selected_feature_count") or 0)
        == MODEL_NATIVE_SELECTED_FEATURE_COUNT
        and int(report.get("mandatory_selected_feature_count") or 0)
        == MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
        and int(report.get("manifest_mandatory_selected_feature_count") or 0)
        == MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
        and mandatory_prefix == MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
        and int(report.get("available_candidate_feature_count") or 0)
        == MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT
        and int(report.get("manifest_available_candidate_feature_count") or 0)
        == MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT
        and tuple(available_candidates) == MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS
        and report.get("available_candidate_fields_sha256")
        == _sha256_json(list(available_candidates))
        and report.get("feature_ranking_fit_scope") == "train_only"
        and len(ranking_sha256) == 64
        and all(character in "0123456789abcdef" for character in ranking_sha256)
    )
    return [
        *_base_evidence_checks(
            report,
            schema_version="entry_feature_foundation_audit_v1",
            audit_kind="feature",
            dataset_dir=dataset_dir,
            split_rows=split_rows,
        ),
        _check(
            (
                "feature audit proves exact model-native "
                f"{MODEL_NATIVE_BASE_SIGNAL_DIM} plus "
                f"{MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT} plus "
                f"{MODEL_NATIVE_AVAILABLE_CANDIDATE_FEATURE_COUNT} full-pool partition"
            ),
            partition_exact,
            {
                "contract_error": contract_error,
                "model_native_signal_dim": report.get("model_native_signal_dim"),
                "base_signal_dim": report.get("base_signal_dim"),
                "base_field_count": len(base_fields),
                "selected_feature_count": report.get("selected_feature_count"),
                "manifest_selected_feature_count": report.get(
                    "manifest_selected_feature_count"
                ),
                "mandatory_selected_feature_count": report.get(
                    "mandatory_selected_feature_count"
                ),
                "manifest_mandatory_selected_feature_count": report.get(
                    "manifest_mandatory_selected_feature_count"
                ),
                "mandatory_prefix_exact": (
                    mandatory_prefix == MODEL_NATIVE_MANDATORY_SELECTED_FIELDS
                ),
                "available_candidate_feature_count": report.get(
                    "available_candidate_feature_count"
                ),
                "manifest_available_candidate_feature_count": report.get(
                    "manifest_available_candidate_feature_count"
                ),
                "available_candidate_observed_count": len(available_candidates),
                "available_candidate_hash_exact": (
                    report.get("available_candidate_fields_sha256")
                    == _sha256_json(list(available_candidates))
                ),
                "feature_ranking_fit_scope": report.get(
                    "feature_ranking_fit_scope"
                ),
                "feature_ranking_sha256": ranking_sha256,
            },
        ),
        _check(
            "feature objective and source liveness are fully live",
            report.get("foundation_objective_coverage_all_present") is True
            and report.get("foundation_objective_liveness_all_live") is True
            and report.get("foundation_source_field_liveness_all_live") is True,
            {
                "objective_coverage": report.get(
                    "foundation_objective_coverage_all_present"
                ),
                "objective_liveness": report.get(
                    "foundation_objective_liveness_all_live"
                ),
                "source_liveness": report.get(
                    "foundation_source_field_liveness_all_live"
                ),
            },
        ),
    ]


def _target_checks(
    report: dict[str, Any],
    dataset_dir: Path,
    split_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    contract = (
        report.get("target_head_contract")
        if isinstance(report.get("target_head_contract"), dict)
        else {}
    )
    active = tuple(str(value) for value in contract.get("active_training_heads", []))
    blocked = tuple(str(value) for value in contract.get("blocked_heads", []))
    extra = tuple(
        str(value) for value in contract.get("extra_active_target_heads", [])
    )
    extra_liveness = contract.get("extra_active_target_head_liveness")
    aux_contract_valid = True
    aux_contract_error = None
    try:
        require_model_native_aux_target_contract(
            report.get("model_native_aux_target_contract"),
            context="ADOPTION_TARGET_AUDIT",
        )
    except RuntimeError as exc:
        aux_contract_valid = False
        aux_contract_error = str(exc)
    entry_q_target = report.get("entry_fitted_q_target_contract")
    entry_q_contract_valid = isinstance(entry_q_target, dict)
    if entry_q_contract_valid:
        try:
            require_entry_fitted_q_contract(
                entry_q_target.get("entry_fitted_q_contract"),
                context="ADOPTION_TARGET_AUDIT",
            )
            require_entry_fitted_q_production_economics_readiness(
                entry_q_target.get("production_economics"),
                context="ADOPTION_TARGET_AUDIT",
                require_ready=True,
            )
        except RuntimeError:
            entry_q_contract_valid = False
    return [
        *_base_evidence_checks(
            report,
            schema_version=FOUNDATION_TARGET_AUDIT_SCHEMA_VERSION,
            audit_kind="target",
            dataset_dir=dataset_dir,
            split_rows=split_rows,
        ),
        _check(
            "target audit head contract is exact model-native base",
            active == MODEL_NATIVE_BASE_ACTIVE_HEADS
            and blocked == MODEL_NATIVE_BLOCKED_HEADS,
            {
                "expected_active_heads": list(MODEL_NATIVE_BASE_ACTIVE_HEADS),
                "observed_active_heads": list(active),
                "expected_blocked_heads": list(MODEL_NATIVE_BLOCKED_HEADS),
                "observed_blocked_heads": list(blocked),
            },
        ),
        _check(
            "target audit proves canonical aux targets and production-ready Entry-Q",
            aux_contract_valid
            and entry_q_contract_valid
            and extra == MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS
            and isinstance(extra_liveness, dict)
            and all(
                extra_liveness.get(head) is True
                for head in MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS
            ),
            {
                "aux_contract_valid": aux_contract_valid,
                "aux_contract_error": aux_contract_error,
                "entry_fitted_q_target_valid": entry_q_contract_valid,
                "expected_extra_active_target_heads": list(
                    MODEL_NATIVE_EXTRA_ACTIVE_TARGET_HEADS
                ),
                "observed_extra_active_target_heads": list(extra),
                "extra_active_target_head_liveness": extra_liveness,
            },
        ),
    ]


def _specialist_checks(
    report: dict[str, Any],
    dataset_dir: Path,
    split_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    required = tuple(
        str(value) for value in report.get("required_training_specialists", [])
    )
    architecture = (
        report.get("architecture_contract")
        if isinstance(report.get("architecture_contract"), dict)
        else {}
    )
    fusion = (
        architecture.get("recommended_fusion")
        if isinstance(architecture.get("recommended_fusion"), dict)
        else {}
    )
    return [
        *_base_evidence_checks(
            report,
            schema_version="entry_specialist_feature_group_audit_v1",
            audit_kind="specialist",
            dataset_dir=dataset_dir,
            split_rows=split_rows,
        ),
        _check(
            "specialist audit uses exact model-native seq513 dimensions",
            report.get("contract_mode") == MODEL_NATIVE_CONTRACT_MODE
            and int(report.get("signal_field_count") or 0)
            == MODEL_NATIVE_SIGNAL_DIM
            and int(report.get("selected_feature_count") or 0)
            == MODEL_NATIVE_SELECTED_FEATURE_COUNT
            and int(architecture.get("input_dim") or 0)
            == MODEL_NATIVE_SIGNAL_DIM,
            {
                "contract_mode": report.get("contract_mode"),
                "signal_field_count": report.get("signal_field_count"),
                "selected_feature_count": report.get("selected_feature_count"),
                "architecture_input_dim": architecture.get("input_dim"),
            },
        ),
        _check(
            "specialist set and model-role contract are exact",
            required == MODEL_NATIVE_REQUIRED_SPECIALISTS
            and _canonical_equal(
                report.get("specialist_model_contract"),
                MODEL_NATIVE_SPECIALIST_MODEL_CONTRACT,
            )
            and report.get("specialist_model_contract_valid") is True,
            {
                "expected_specialists": list(MODEL_NATIVE_REQUIRED_SPECIALISTS),
                "observed_specialists": list(required),
                "specialist_model_contract_valid": report.get(
                    "specialist_model_contract_valid"
                ),
            },
        ),
        _check(
            "specialist fusion heads and liveness are exact",
            tuple(fusion.get("active_heads", [])) == MODEL_NATIVE_BASE_ACTIVE_HEADS
            and tuple(fusion.get("blocked_heads", []))
            == MODEL_NATIVE_BLOCKED_HEADS
            and report.get("specialist_input_liveness_all_live") is True
            and report.get(
                "foundation_objective_routing_all_present_and_expected"
            )
            is True,
            {
                "fusion_active_heads": fusion.get("active_heads"),
                "fusion_blocked_heads": fusion.get("blocked_heads"),
                "specialist_input_liveness_all_live": report.get(
                    "specialist_input_liveness_all_live"
                ),
                "objective_routing_all_present": report.get(
                    "foundation_objective_routing_all_present_and_expected"
                ),
            },
        ),
    ]


def _smoke_checks(
    report: dict[str, Any],
    *,
    dataset_dir: Path,
    split_rows: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    split_artifacts = (
        report.get("split_artifacts")
        if isinstance(report.get("split_artifacts"), dict)
        else {}
    )
    embedded = (
        report.get("smoke_manifest")
        if isinstance(report.get("smoke_manifest"), dict)
        else {}
    )
    embedded_splits = (
        embedded.get("splits") if isinstance(embedded.get("splits"), dict) else {}
    )
    contract_error = ""
    try:
        require_model_native_readiness_contract(
            report.get("model_native_readiness_contract"),
            context="ADOPTION_SMOKE_MANIFEST",
        )
    except RuntimeError as exc:
        contract_error = str(exc)

    def split_matches(split: str) -> bool:
        expected = split_rows.get(split) or {}
        observed = split_artifacts.get(split) or {}
        compact = embedded_splits.get(split) or {}
        return (
            int(observed.get("rows") or 0) > 0
            and _same_path(observed.get("output_data_path"), Path(expected.get("parquet_path") or "/"))
            and _same_path(observed.get("manifest_path"), Path(expected.get("manifest_path") or "/"))
            and observed.get("parquet_sha256") == expected.get("parquet_sha256")
            and observed.get("manifest_sha256") == expected.get("manifest_sha256")
            and int(observed.get("seq_input_dim") or 0) == MODEL_NATIVE_SIGNAL_DIM
            and int(observed.get("snap_input_dim") or 0) == MODEL_NATIVE_SIGNAL_DIM
            and int(observed.get("field_count") or 0) == MODEL_NATIVE_SIGNAL_DIM
            and compact.get("out_parquet_sha256") == expected.get("parquet_sha256")
            and compact.get("out_manifest_sha256") == expected.get("manifest_sha256")
            and int(compact.get("seq_input_dim") or 0) == MODEL_NATIVE_SIGNAL_DIM
            and int(compact.get("snap_input_dim") or 0) == MODEL_NATIVE_SIGNAL_DIM
            and int(compact.get("field_count") or 0) == MODEL_NATIVE_SIGNAL_DIM
        )

    side_effects = (
        report.get("side_effects_started")
        if isinstance(report.get("side_effects_started"), dict)
        else {}
    )
    return [
        _check(
            "smoke manifest report is exact and green",
            report.get("schema_version") == SMOKE_REPORT_SCHEMA
            and report.get("decision") == SMOKE_REPORT_DECISION
            and report.get("report_only") is True
            and report.get("manifest_embedded") is True
            and not report.get("failures"),
            {
                "schema_version": report.get("schema_version"),
                "decision": report.get("decision"),
                "failures": report.get("failures"),
            },
        ),
        _check(
            "smoke manifest carries exact model-native readiness contract",
            not contract_error,
            {"error": contract_error},
        ),
        _check(
            "smoke manifest points at the explicit seq513 candidate",
            report.get("manifest_variant") == MODEL_NATIVE_CONTRACT_MODE
            and int(report.get("expected_seq_snap_width") or 0)
            == MODEL_NATIVE_SIGNAL_DIM
            and _same_path(report.get("smart_smoke_dataset_dir"), dataset_dir)
            and embedded.get("schema_version") == SMOKE_DATASET_SCHEMA
            and embedded.get("manifest_variant") == MODEL_NATIVE_CONTRACT_MODE
            and int(embedded.get("expected_seq_snap_width") or 0)
            == MODEL_NATIVE_SIGNAL_DIM
            and _same_path(embedded.get("dataset_dir"), dataset_dir),
            {
                "smart_smoke_dataset_dir": report.get("smart_smoke_dataset_dir"),
                "embedded_dataset_dir": embedded.get("dataset_dir"),
            },
        ),
        _check(
            "smoke manifest is content-bound to all split artifacts",
            set(split_artifacts) == set(SPLITS)
            and set(embedded_splits) == set(SPLITS)
            and set(split_rows) == set(SPLITS)
            and all(split_matches(split) for split in SPLITS)
            and report.get("manifest_sha256") == _sha256_json(embedded),
            {
                "manifest_sha256": report.get("manifest_sha256"),
                "computed_manifest_sha256": _sha256_json(embedded),
            },
        ),
        _check(
            "smoke evidence starts no training replay or live side effect",
            bool(side_effects)
            and all(value is False for value in side_effects.values())
            and report.get("training_allowed") is False
            and report.get("replay_allowed") is False
            and report.get("shadow_live_allowed") is False,
            {"side_effects_started": side_effects},
        ),
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    evidence_paths = {
        "feature_audit": Path(args.feature_audit_json).expanduser().resolve(),
        "target_audit": Path(args.target_audit_json).expanduser().resolve(),
        "specialist_audit": Path(args.specialist_audit_json).expanduser().resolve(),
        "smoke_manifest": Path(args.smoke_manifest_json).expanduser().resolve(),
    }
    for name, prefix in AUDIT_EVENT_PREFIXES.items():
        require_newest_immutable_event(evidence_paths[name], prefix)
    require_newest_immutable_event(evidence_paths["smoke_manifest"], SMOKE_EVENT_PREFIX)

    reports = {name: _read_json(path) for name, path in evidence_paths.items()}
    dataset_checks, split_rows, dataset_artifacts = _dataset_contract(
        dataset_dir,
        reports["smoke_manifest"],
    )
    artifacts = {**evidence_paths, **dataset_artifacts}
    fingerprints = artifact_fingerprints(artifacts)
    gate_checks = {
        "candidate_dataset": dataset_checks,
        "feature_audit": _feature_checks(
            reports["feature_audit"],
            dataset_dir,
            split_rows,
        ),
        "target_audit": _target_checks(
            reports["target_audit"],
            dataset_dir,
            split_rows,
        ),
        "specialist_audit": _specialist_checks(
            reports["specialist_audit"],
            dataset_dir,
            split_rows,
        ),
        "smoke_manifest": _smoke_checks(
            reports["smoke_manifest"],
            dataset_dir=dataset_dir,
            split_rows=split_rows,
        ),
        "artifact_provenance": artifact_fingerprint_checks(fingerprints),
    }
    gates: list[dict[str, Any]] = []
    for name, checks in gate_checks.items():
        passed = sum(bool(check.get("ok")) for check in checks)
        gates.append(
            {
                "name": name,
                "decision": "PASS" if passed == len(checks) else "FAIL",
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
        if not check.get("ok")
    ]
    ready = not failures
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "decision": (
            "READY_FOR_MODEL_NATIVE_ADOPTION_REVIEW"
            if ready
            else "BLOCKED_MODEL_NATIVE_ADOPTION_REVIEW"
        ),
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        **foundation_audit_policy_binding(),
        "expected_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "model_native_readiness_contract": model_native_readiness_contract_metadata(),
        "report_only": True,
        "adoption_evidence_ready": bool(ready),
        "candidate_ready_for_activation": False,
        "training_allowed": False,
        "replay_allowed": False,
        "shadow_live_allowed": False,
        "direction_selection_authority": False,
        "dataset_dir": str(dataset_dir),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
        "artifact_fingerprints": fingerprints,
        "split_contracts": split_rows,
        "gates": gates,
        "failures": failures,
        "next_required_gate": (
            "immutable lifecycle admission with all activation evidence bound; this report grants no activation authority"
            if ready
            else "repair exact seq513/head/specialist evidence and publish new immutable events"
        ),
    }
    _, report = write_immutable_json_event(out_dir, EVENT_PREFIX, report)
    if not args.quiet:
        print(json.dumps(report, indent=2, sort_keys=True))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--feature-audit-json", required=True)
    parser.add_argument("--target-audit-json", required=True)
    parser.add_argument("--specialist-audit-json", required=True)
    parser.add_argument("--smoke-manifest-json", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = run(args)
    if not report["adoption_evidence_ready"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
