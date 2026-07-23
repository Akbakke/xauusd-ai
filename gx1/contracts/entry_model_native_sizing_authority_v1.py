"""Fail-closed learned execution sizing authority for model-native Entry.

The final model bundle binds only the TRAIN/VAL-fitted calibration.  A separate
immutable adoption event is admissible only after the sizing-head diagnostic
and a full-TEST joint Entry + ACTIVE Exit-V3/Exit-IQL/Strategy-F execution
replay are both proven.  Paper/live launch additionally requires a fresh
post-adoption broker-runtime sizing parity event.

Only ``learned_calibrated`` is executable.  ``historical_fixed_1x`` exists only
as an explicitly named negative-control description and has no application
path.  Missing, older, malformed, red, or hash-mismatched evidence raises; the
caller must place no order.  Direction is an input fact and is never changed.
"""

from __future__ import annotations

import json
import math
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from gx1.contracts.entry_model_native_sizing_calibration_v1 import (
    MODEL_NATIVE_SIZING_FIT_SCOPE,
    MODEL_NATIVE_SIZING_RUNTIME_CONSTRAINT_KEYS,
    MODEL_NATIVE_SIZING_TRANSFORM_VERSION,
    ModelNativeSizingContractError,
    calibrated_sizing_transform,
    load_bound_sizing_calibration,
    load_bound_sizing_oos_proof,
    require_immutable_json_binding,
    sha256_file,
    sizing_risk_policy_metadata,
)
from gx1.contracts.immutable_event_authority_v1 import (
    ImmutableEventAuthorityError,
    require_newest_immutable_event,
)
from gx1.contracts.entry_model_native_sizing_execution_v1 import (
    ModelNativeSizingExecutionContractError,
    load_bound_joint_exit_sizing_proof,
)
from gx1.contracts.entry_run_lineage_v1 import EntryRunLineageError, require_entry_run_id


MODEL_NATIVE_SIZING_AUTHORITY_SCHEMA_VERSION = (
    "entry_model_native_sizing_authority_v4"
)
MODEL_NATIVE_SIZING_ADOPTION_SCHEMA_VERSION = (
    "entry_model_native_sizing_adoption_v4"
)
MODEL_NATIVE_SIZING_BUNDLE_CALIBRATION_SCHEMA_VERSION = (
    "entry_model_native_sizing_bundle_calibration_v1"
)
MODEL_NATIVE_SIZING_APPLICATION_SCHEMA_VERSION = (
    "entry_model_native_sizing_application_v2"
)
MODEL_NATIVE_SIZING_MODE_HISTORICAL_FIXED = "historical_fixed_1x"
MODEL_NATIVE_SIZING_MODE_LEARNED = "learned_calibrated"
MODEL_NATIVE_SIZING_MODES = (MODEL_NATIVE_SIZING_MODE_LEARNED,)

_ADOPTION_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_ADOPTION"
_CALIBRATION_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_CALIBRATION"
_OOS_PROOF_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_OOS_PROOF"
_OOS_SOURCE_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_OOS_SOURCE"
_INSTRUMENT_EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SIZING_INSTRUMENT_EVIDENCE"
_PREDICTION_REPORT_EVENT_PREFIX = "ENTRY_CANDIDATE_SELECTIVE_EDGE"
_MODEL_HEAD_SERVE_PARITY_EVENT_PREFIX = "MODEL_NATIVE_SERVE_PARITY"
_AUTHORITY_KEYS = frozenset(
    {
        "schema_version",
        "adoption_mode",
        "adoption_artifact",
        "position_size_head_required",
        "position_size_source",
        "position_size_head_role",
        "execution_size_mode",
        "transform_version",
        "legacy_post_model_sizing_overlays_allowed",
        "caller_or_environment_dynamic_sizing_tuners_allowed",
        "fixed_1x_fallback_allowed",
        "direction_authority",
        "flat_units",
        "required_runtime_constraints",
        "rounding_mode",
    }
)
_ADOPTION_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "failures",
        "adoption_mode",
        "authority_root",
        "bundle_dir",
        "bundle_metadata_path",
        "bundle_metadata_sha256",
        "master_transformer_lock_path",
        "master_transformer_lock_sha256",
        "model_state_dict_path",
        "model_state_dict_sha256",
        "calibration_artifact",
        "oos_proof_artifact",
        "joint_exit_sizing_proof_artifact",
        "risk_policy",
        "runtime_constraint_authority",
        "direction_authority",
        "fixed_1x_fallback_allowed",
        "entry_run_id",
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
_APPLICATION_KEYS = frozenset(
    {
        "schema_version",
        "sizing_mode",
        "model_direction",
        "model_direction_index",
        "position_size_logit",
        "calibrated_size_fraction",
        "applied_size_multiplier",
        "capacity_units",
        "reference_pre_round_units",
        "pre_round_units",
        "units",
        "rounding_mode",
        "authorized_order",
        "no_order_reason",
        "runtime_constraints",
        "calibration_artifact_sha256",
        "oos_proof_artifact_sha256",
        "adoption_artifact_sha256",
        "sizing_authority_contract",
    }
)
_AUTHORITY_STATIC = {
    "schema_version": MODEL_NATIVE_SIZING_AUTHORITY_SCHEMA_VERSION,
    "adoption_mode": MODEL_NATIVE_SIZING_MODE_LEARNED,
    "position_size_head_required": True,
    "position_size_source": "position_size_logit",
    "position_size_head_role": "sole_learned_execution_sizing_authority",
    "execution_size_mode": "learned_admissible_capacity_fraction",
    "transform_version": MODEL_NATIVE_SIZING_TRANSFORM_VERSION,
    "legacy_post_model_sizing_overlays_allowed": False,
    "caller_or_environment_dynamic_sizing_tuners_allowed": False,
    "fixed_1x_fallback_allowed": False,
    "direction_authority": "none",
    "flat_units": 0,
    "required_runtime_constraints": list(
        MODEL_NATIVE_SIZING_RUNTIME_CONSTRAINT_KEYS
    ),
    "rounding_mode": "floor_to_unit_step",
}
_ADOPTION_STATIC = {
    "decision": "PASS",
    "failures": [],
    "adoption_mode": MODEL_NATIVE_SIZING_MODE_LEARNED,
    "risk_policy": sizing_risk_policy_metadata(),
    "runtime_constraint_authority": (
        "exact_broker_account_instrument_exposure_facts_with_transaction_ids"
    ),
    "direction_authority": "none",
    "fixed_1x_fallback_allowed": False,
}


class ModelNativeSizingUnavailable(RuntimeError):
    """No exact learned sizing authority exists; callers must place no order."""


@dataclass(frozen=True)
class ValidatedLearnedSizingAuthority:
    """Process-local immutable snapshot of one fully verified adoption chain."""

    authority_json: str
    adoption_json: str
    calibration_json: str
    proof_json: str
    joint_proof_json: str
    content_hash_key: tuple[tuple[str, str, str], ...]
    file_stats: tuple[tuple[str, int, int, int, int], ...]

    @property
    def authority(self) -> dict[str, Any]:
        return json.loads(self.authority_json)

    @property
    def adoption(self) -> dict[str, Any]:
        return json.loads(self.adoption_json)

    @property
    def calibration(self) -> dict[str, Any]:
        return json.loads(self.calibration_json)

    @property
    def proof(self) -> dict[str, Any]:
        return json.loads(self.proof_json)

    @property
    def joint_proof(self) -> dict[str, Any]:
        return json.loads(self.joint_proof_json)


_VALIDATED_CACHE: dict[tuple[str, str], ValidatedLearnedSizingAuthority] = {}
_TAINTED_CACHE_KEYS: set[tuple[str, str]] = set()
_CACHE_LOCK = threading.RLock()


def _fail(context: str, detail: str) -> None:
    raise ModelNativeSizingUnavailable(
        f"[{str(context).strip() or 'MODEL_NATIVE_SIZING'}_UNAVAILABLE] {detail}"
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


def _exact_sha(value: Any, *, context: str) -> str:
    parsed = str(value or "").strip().lower()
    if len(parsed) != 64 or any(ch not in "0123456789abcdef" for ch in parsed):
        _fail(context, "not an exact SHA-256")
    return parsed


def _absolute_file(value: Any, *, context: str, verify: bool) -> Path:
    path = Path(str(value or "")).expanduser()
    if not path.is_absolute():
        _fail(context, f"path must be absolute: {path}")
    resolved = path.resolve()
    if verify and not resolved.is_file():
        _fail(context, f"bound file is missing: {resolved}")
    return resolved


def _utc(value: Any, *, context: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        _fail(context, f"invalid UTC timestamp: {value!r}")
    offset = parsed.utcoffset() if parsed.tzinfo is not None else None
    if offset is None or offset.total_seconds() != 0.0:
        _fail(context, "timestamp must be timezone-aware UTC")
    return parsed


def _json_object(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _fail(context, f"JSON unreadable: {path}: {exc}")
    if not isinstance(payload, dict):
        _fail(context, f"JSON root is not an object: {path}")
    return payload


def _canonical_json(value: Mapping[str, Any], *, context: str) -> str:
    try:
        return json.dumps(
            dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        _fail(context, f"validated evidence is not strict JSON: {exc}")


def _cache_lookup_key(authority: Mapping[str, Any]) -> tuple[str, str]:
    binding = authority["adoption_artifact"]
    return str(binding["json_path"]), str(binding["sha256"])


def _snapshot_file_specs(
    *,
    authority: Mapping[str, Any],
    adoption: Mapping[str, Any],
    calibration: Mapping[str, Any],
    proof: Mapping[str, Any],
    joint_proof: Mapping[str, Any],
    context: str,
) -> tuple[tuple[str, str, str], ...]:
    rows: list[tuple[str, str, str]] = [
        (
            "adoption",
            str(authority["adoption_artifact"]["json_path"]),
            str(authority["adoption_artifact"]["sha256"]),
        ),
        (
            "bundle_metadata",
            str(adoption["bundle_metadata_path"]),
            str(adoption["bundle_metadata_sha256"]),
        ),
        (
            "master_transformer_lock",
            str(adoption["master_transformer_lock_path"]),
            str(adoption["master_transformer_lock_sha256"]),
        ),
        (
            "bundle_model_state_dict",
            str(adoption["model_state_dict_path"]),
            str(adoption["model_state_dict_sha256"]),
        ),
        (
            "calibration",
            str(adoption["calibration_artifact"]["json_path"]),
            str(adoption["calibration_artifact"]["sha256"]),
        ),
        (
            "oos_proof",
            str(adoption["oos_proof_artifact"]["json_path"]),
            str(adoption["oos_proof_artifact"]["sha256"]),
        ),
        (
            "joint_exit_sizing_proof",
            str(adoption["joint_exit_sizing_proof_artifact"]["json_path"]),
            str(adoption["joint_exit_sizing_proof_artifact"]["sha256"]),
        ),
        (
            "joint_exit_replay_rows",
            str(joint_proof["replay_rows"]["path"]),
            str(joint_proof["replay_rows"]["sha256"]),
        ),
        (
            "joint_exit_trace_rows",
            str(joint_proof["exit_trace_rows"]["path"]),
            str(joint_proof["exit_trace_rows"]["sha256"]),
        ),
        (
            "joint_exit_artifact_registry",
            str(joint_proof["artifact_registry"]["path"]),
            str(joint_proof["artifact_registry"]["sha256"]),
        ),
    ]
    for role, manifest in joint_proof[
        "active_exit_artifact_manifests"
    ].items():
        root = Path(str(manifest["root_path"]))
        for file_binding in manifest["files"]:
            relative = str(file_binding["relative_path"])
            path = root if relative == "." else root / relative
            rows.append(
                (
                    f"active_exit.{role}.{relative}",
                    str(path),
                    str(file_binding["sha256"]),
                )
            )
    lineage = calibration["lineage"]
    for stem in (
        "dataset_manifest",
        "fit_predictions",
        "model_checkpoint",
        "instrument_evidence",
    ):
        rows.append(
            (
                f"lineage.{stem}",
                str(lineage[f"{stem}_path"]),
                str(lineage[f"{stem}_sha256"]),
            )
        )
    fit_provenance = calibration["fit_prediction_provenance"]
    fit_report = fit_provenance["prediction_report_artifact"]
    rows.append(
        (
            "fit_prediction_report",
            str(fit_report["json_path"]),
            str(fit_report["sha256"]),
        )
    )
    fit_report_payload = _json_object(
        Path(fit_report["json_path"]), context=f"{context}.fit_prediction_report"
    )
    fit_prediction_evidence = fit_report_payload.get("prediction_evidence")
    if not isinstance(fit_prediction_evidence, Mapping):
        _fail(context, "fit prediction report lacks prediction_evidence")
    rows.append(
        (
            "fit_source_bundle_metadata",
            str(fit_prediction_evidence["bundle_metadata_path"]),
            str(fit_prediction_evidence["bundle_metadata_sha256"]),
        )
    )
    for split, binding in fit_provenance["dataset_split_bindings"].items():
        rows.extend(
            (
                (
                    f"fit_dataset.{split}.manifest",
                    str(binding["manifest_path"]),
                    str(binding["manifest_sha256"]),
                ),
                (
                    f"fit_dataset.{split}.parquet",
                    str(binding["parquet_path"]),
                    str(binding["parquet_sha256"]),
                ),
            )
        )
    oos_source_binding = proof["oos_source_artifact"]
    rows.append(
        (
            "oos_source",
            str(oos_source_binding["json_path"]),
            str(oos_source_binding["sha256"]),
        )
    )
    oos_source = _json_object(
        Path(oos_source_binding["json_path"]), context=f"{context}.oos_source"
    )
    rows.extend(
        (
            (
                "test_predictions",
                str(oos_source["test_predictions"]["path"]),
                str(oos_source["test_predictions"]["sha256"]),
            ),
            (
                "test_prediction_report",
                str(
                    oos_source["test_prediction_provenance"][
                        "prediction_report_artifact"
                    ]["json_path"]
                ),
                str(
                    oos_source["test_prediction_provenance"][
                        "prediction_report_artifact"
                    ]["sha256"]
                ),
            ),
            (
                "source_tape",
                str(oos_source["source_tape"]["path"]),
                str(oos_source["source_tape"]["sha256"]),
            ),
            (
                "model_head_serve_parity",
                str(oos_source["model_head_serve_parity_artifact"]["json_path"]),
                str(oos_source["model_head_serve_parity_artifact"]["sha256"]),
            ),
        )
    )
    for split, binding in oos_source["test_prediction_provenance"][
        "dataset_split_bindings"
    ].items():
        rows.extend(
            (
                (
                    f"test_dataset.{split}.manifest",
                    str(binding["manifest_path"]),
                    str(binding["manifest_sha256"]),
                ),
                (
                    f"test_dataset.{split}.parquet",
                    str(binding["parquet_path"]),
                    str(binding["parquet_sha256"]),
                ),
            )
        )
    for name, binding in proof["source_bindings"].items():
        rows.append((f"source.{name}", str(binding["path"]), str(binding["sha256"])))
    canonical: list[tuple[str, str, str]] = []
    seen_paths: dict[str, str] = {}
    for label, path_raw, sha_raw in rows:
        path = _absolute_file(path_raw, context=f"{context}.{label}", verify=True)
        sha = _exact_sha(sha_raw, context=f"{context}.{label}.sha256")
        prior = seen_paths.get(str(path))
        if prior is not None and prior != sha:
            _fail(context, f"same file has conflicting hashes: {path}")
        seen_paths[str(path)] = sha
        canonical.append((label, str(path), sha))
    return tuple(sorted(canonical))


def _capture_file_stats(
    specs: tuple[tuple[str, str, str], ...], *, context: str
) -> tuple[tuple[str, int, int, int, int], ...]:
    rows: list[tuple[str, int, int, int, int]] = []
    for _label, path_raw, _sha in specs:
        path = Path(path_raw)
        try:
            stat = path.stat()
        except OSError as exc:
            _fail(context, f"validated sizing file disappeared: {path}: {exc}")
        rows.append(
            (str(path), int(stat.st_dev), int(stat.st_ino), int(stat.st_size), int(stat.st_mtime_ns))
        )
    return tuple(sorted(set(rows)))


def _require_snapshot_unchanged(
    snapshot: ValidatedLearnedSizingAuthority, *, context: str
) -> None:
    for path_raw, device, inode, size, mtime_ns in snapshot.file_stats:
        path = Path(path_raw)
        try:
            stat = path.stat()
        except OSError as exc:
            _fail(context, f"validated sizing file missing after startup: {path}: {exc}")
        observed = (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )
        if observed != (device, inode, size, mtime_ns):
            _fail(context, f"validated sizing file changed after startup: {path}")
    specs = {label: Path(path) for label, path, _sha in snapshot.content_hash_key}
    for label, prefix in (
        ("adoption", _ADOPTION_EVENT_PREFIX),
        ("calibration", _CALIBRATION_EVENT_PREFIX),
        ("oos_proof", _OOS_PROOF_EVENT_PREFIX),
        ("oos_source", _OOS_SOURCE_EVENT_PREFIX),
        ("lineage.instrument_evidence", _INSTRUMENT_EVENT_PREFIX),
        ("fit_prediction_report", _PREDICTION_REPORT_EVENT_PREFIX),
        ("test_prediction_report", _PREDICTION_REPORT_EVENT_PREFIX),
        ("model_head_serve_parity", _MODEL_HEAD_SERVE_PARITY_EVENT_PREFIX),
    ):
        path = specs[label]
        try:
            require_newest_immutable_event(path, prefix)
        except ImmutableEventAuthorityError as exc:
            _fail(
                context,
                f"cached {label} is no longer newest immutable family authority: {exc}",
            )


def _require_or_taint_snapshot_unchanged(
    snapshot: ValidatedLearnedSizingAuthority, *, context: str
) -> None:
    key = _cache_lookup_key(snapshot.authority)
    try:
        _require_snapshot_unchanged(snapshot, context=context)
    except ModelNativeSizingUnavailable:
        with _CACHE_LOCK:
            _VALIDATED_CACHE.pop(key, None)
            _TAINTED_CACHE_KEYS.add(key)
        raise


def historical_fixed_1x_negative_control_metadata() -> dict[str, Any]:
    """Describe the historical 1x benchmark; never executable authority."""

    return {
        "name": MODEL_NATIVE_SIZING_MODE_HISTORICAL_FIXED,
        "role": "historical_negative_control_only",
        "executable_order_authority": False,
        "current_launch_authority": False,
        "fallback_allowed": False,
    }


def model_native_sizing_bundle_calibration_metadata(
    *,
    calibration_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the only sizing declaration allowed inside the final bundle."""

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
    observed = _exact_keys(value, _BUNDLE_CALIBRATION_KEYS, context=context)
    expected = model_native_sizing_bundle_calibration_metadata(
        calibration_artifact=observed["calibration_artifact"]
    )
    if observed != expected:
        mismatched = sorted(
            key for key, expected_value in expected.items()
            if observed.get(key) != expected_value
        )
        _fail(context, f"bundle calibration mismatch: {mismatched}")
    return observed


def learned_sizing_authority_contract_metadata(
    *,
    adoption_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Build runtime authority from one immutable adoption-event binding."""

    adoption_binding = require_immutable_json_binding(
        adoption_artifact,
        event_prefix=_ADOPTION_EVENT_PREFIX,
        context="LEARNED_SIZING_AUTHORITY.adoption_artifact",
        verify_file=False,
    )
    return {**_AUTHORITY_STATIC, "adoption_artifact": adoption_binding}


def model_native_sizing_authority_contract_metadata(
    *,
    adoption_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Compatibility spelling with no default/fixed fallback."""

    return learned_sizing_authority_contract_metadata(
        adoption_artifact=adoption_artifact
    )


def require_model_native_sizing_authority_contract(
    contract: Mapping[str, Any] | Any,
    *,
    context: str,
    required_mode: str | None = None,
    verify_bound_artifacts: bool = False,
    verify_lineage_files: bool = False,
) -> dict[str, Any]:
    """Require exact learned authority; historical fixed mode is never accepted."""

    observed = _exact_keys(contract, _AUTHORITY_KEYS, context=context)
    if required_mode not in (None, MODEL_NATIVE_SIZING_MODE_LEARNED):
        _fail(context, f"non-executable required_mode={required_mode!r}")
    expected = learned_sizing_authority_contract_metadata(
        adoption_artifact=observed["adoption_artifact"]
    )
    if observed != expected:
        mismatched = sorted(
            key for key, expected_value in expected.items()
            if observed.get(key) != expected_value
        )
        _fail(context, f"learned authority mismatch: {mismatched}")
    if verify_bound_artifacts:
        prepare_model_native_sizing_authority(
            observed,
            context=context,
        )
    return observed


def require_model_native_sizing_adoption_artifact(
    value: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Validate adoption structure; file/hash cross-binding is done by the loader."""

    observed = _exact_keys(value, _ADOPTION_KEYS, context=context)
    if observed["schema_version"] != MODEL_NATIVE_SIZING_ADOPTION_SCHEMA_VERSION:
        _fail(context, "adoption schema_version mismatch")
    _utc(observed["created_utc"], context=f"{context}.created_utc")
    _absolute_file(observed["json_path"], context=f"{context}.json_path", verify=False)
    for key, expected in _ADOPTION_STATIC.items():
        if observed[key] != expected:
            _fail(context, f"{key}={observed[key]!r} expected={expected!r}")
    try:
        require_entry_run_id(observed["entry_run_id"])
    except EntryRunLineageError as exc:
        _fail(context, str(exc))
    bundle = Path(str(observed["bundle_dir"] or "")).expanduser()
    if not bundle.is_absolute():
        _fail(context, "bundle_dir must be absolute")
    authority_root = Path(str(observed["authority_root"] or "")).expanduser()
    if not authority_root.is_absolute():
        _fail(context, "authority_root must be absolute")
    metadata_path = _absolute_file(
        observed["bundle_metadata_path"],
        context=f"{context}.bundle_metadata_path",
        verify=False,
    )
    lock_path = _absolute_file(
        observed["master_transformer_lock_path"],
        context=f"{context}.master_transformer_lock_path",
        verify=False,
    )
    state_path = _absolute_file(
        observed["model_state_dict_path"],
        context=f"{context}.model_state_dict_path",
        verify=False,
    )
    if metadata_path != (bundle.resolve() / "bundle_metadata.json"):
        _fail(context, "bundle_metadata_path is not exact bundle metadata")
    if lock_path != (bundle.resolve() / "MASTER_TRANSFORMER_LOCK.json"):
        _fail(context, "master_transformer_lock_path is not exact bundle lock")
    if state_path != (bundle.resolve() / "model_state_dict.pt"):
        _fail(context, "model_state_dict_path is not exact bundle checkpoint")
    _exact_sha(observed["bundle_metadata_sha256"], context=f"{context}.metadata_sha")
    _exact_sha(
        observed["master_transformer_lock_sha256"], context=f"{context}.lock_sha"
    )
    _exact_sha(observed["model_state_dict_sha256"], context=f"{context}.state_sha")
    require_immutable_json_binding(
        observed["calibration_artifact"],
        event_prefix=_CALIBRATION_EVENT_PREFIX,
        context=f"{context}.calibration_artifact",
        verify_file=False,
    )
    require_immutable_json_binding(
        observed["oos_proof_artifact"],
        event_prefix=_OOS_PROOF_EVENT_PREFIX,
        context=f"{context}.oos_proof_artifact",
        verify_file=False,
    )
    require_immutable_json_binding(
        observed["joint_exit_sizing_proof_artifact"],
        event_prefix="ENTRY_MODEL_NATIVE_JOINT_EXIT_SIZING_PROOF",
        context=f"{context}.joint_exit_sizing_proof_artifact",
        verify_file=False,
    )
    return observed


def _validate_learned_sizing_authority_snapshot(
    contract: Mapping[str, Any] | Any,
    *,
    context: str,
) -> ValidatedLearnedSizingAuthority:
    """Do the one expensive full-chain hash/load/recompute validation."""

    authority = require_model_native_sizing_authority_contract(
        contract,
        context=context,
        required_mode=MODEL_NATIVE_SIZING_MODE_LEARNED,
        verify_bound_artifacts=False,
    )
    try:
        adoption_binding = require_immutable_json_binding(
            authority["adoption_artifact"],
            event_prefix=_ADOPTION_EVENT_PREFIX,
            context=f"{context}.adoption.binding",
            verify_file=True,
        )
        adoption_path = Path(adoption_binding["json_path"])
        adoption = require_model_native_sizing_adoption_artifact(
            _json_object(adoption_path, context=f"{context}.adoption"),
            context=f"{context}.adoption",
        )
        if Path(adoption["json_path"]).resolve() != adoption_path:
            _fail(context, "adoption json_path differs from bound event")

        metadata_path = _absolute_file(
            adoption["bundle_metadata_path"],
            context=f"{context}.bundle_metadata_path",
            verify=True,
        )
        lock_path = _absolute_file(
            adoption["master_transformer_lock_path"],
            context=f"{context}.master_transformer_lock_path",
            verify=True,
        )
        state_path = _absolute_file(
            adoption["model_state_dict_path"],
            context=f"{context}.model_state_dict_path",
            verify=True,
        )
        for path, hash_key in (
            (metadata_path, "bundle_metadata_sha256"),
            (lock_path, "master_transformer_lock_sha256"),
            (state_path, "model_state_dict_sha256"),
        ):
            expected_sha = _exact_sha(
                adoption[hash_key], context=f"{context}.{hash_key}"
            )
            actual_sha = sha256_file(path)
            if actual_sha != expected_sha:
                _fail(
                    context,
                    f"{hash_key} mismatch: declared={expected_sha} actual={actual_sha}",
                )

        calibration, calibration_binding = load_bound_sizing_calibration(
            adoption["calibration_artifact"],
            context=f"{context}.calibration",
            verify_lineage_files=True,
        )
        proof, proof_binding = load_bound_sizing_oos_proof(
            adoption["oos_proof_artifact"],
            calibration=calibration,
            calibration_artifact_sha256=calibration_binding["sha256"],
            context=f"{context}.oos_proof",
            verify_source_files=True,
        )
        joint_proof, joint_proof_binding = load_bound_joint_exit_sizing_proof(
            adoption["joint_exit_sizing_proof_artifact"],
            context=f"{context}.joint_exit_sizing_proof",
            verify_source_files=True,
        )
        if (
            joint_proof["calibration_artifact"] != calibration_binding
            or joint_proof["oos_proof_artifact"] != proof_binding
        ):
            _fail(context, "joint Exit sizing proof differs from adopted sizing chain")
        expected_declaration = model_native_sizing_bundle_calibration_metadata(
            calibration_artifact=calibration_binding
        )
        metadata = _json_object(metadata_path, context=f"{context}.bundle_metadata")
        lock = _json_object(lock_path, context=f"{context}.transformer_lock")
        for label, payload in (("bundle_metadata", metadata), ("transformer_lock", lock)):
            declaration = require_model_native_sizing_bundle_calibration(
                payload.get("model_native_sizing_calibration"),
                context=f"{context}.{label}.model_native_sizing_calibration",
            )
            if declaration != expected_declaration:
                _fail(context, f"{label} calibration declaration differs from adoption")
        state_sha = _exact_sha(
            adoption["model_state_dict_sha256"],
            context=f"{context}.model_state_dict_sha256",
        )
        if (
            state_sha != calibration["lineage"]["model_checkpoint_sha256"]
            or str(metadata.get("state_dict_sha256") or "").lower() != state_sha
            or str(lock.get("model_sha256") or "").lower() != state_sha
            or lock.get("model_path_relative") != "model_state_dict.pt"
        ):
            _fail(context, "adoption bundle checkpoint identity mismatch")
        evaluation = proof["evaluation_bundle"]
        expected_evaluation = {
            "bundle_dir": adoption["bundle_dir"],
            "bundle_metadata_path": adoption["bundle_metadata_path"],
            "bundle_metadata_sha256": adoption["bundle_metadata_sha256"],
            "master_transformer_lock_path": adoption[
                "master_transformer_lock_path"
            ],
            "master_transformer_lock_sha256": adoption[
                "master_transformer_lock_sha256"
            ],
            "model_state_dict_path": adoption["model_state_dict_path"],
            "model_state_dict_sha256": adoption["model_state_dict_sha256"],
        }
        if evaluation != expected_evaluation:
            _fail(context, "proof evaluation bundle differs from adopted bundle")
        authority_root = Path(adoption["authority_root"]).resolve()
        stage_paths = {
            "adoption": adoption_path,
            "calibration": Path(calibration_binding["json_path"]),
            "oos": Path(proof["oos_source_artifact"]["json_path"]),
            "proof": Path(proof_binding["json_path"]),
            "joint_replay": Path(joint_proof_binding["json_path"]),
            "instrument": Path(calibration["lineage"]["instrument_evidence_path"]),
        }
        expected_stage_dirs = {
            "adoption": "adoption",
            "calibration": "calibration",
            "oos": "oos",
            "proof": "proof",
            "joint_replay": "joint_replay",
            "instrument": "instrument",
        }
        for stage, path in stage_paths.items():
            if path.resolve().parent != authority_root / expected_stage_dirs[stage]:
                _fail(context, f"{stage} event escapes adopted authority_root")
        calibration_time = _utc(
            calibration["created_utc"], context=f"{context}.calibration.created_utc"
        )
        proof_time = _utc(proof["created_utc"], context=f"{context}.proof.created_utc")
        joint_proof_time = _utc(
            joint_proof["created_utc"],
            context=f"{context}.joint_exit_sizing_proof.created_utc",
        )
        oos_source = _json_object(
            Path(proof["oos_source_artifact"]["json_path"]),
            context=f"{context}.oos_source",
        )
        oos_source_time = _utc(
            oos_source["created_utc"], context=f"{context}.oos_source.created_utc"
        )
        adoption_time = _utc(
            adoption["created_utc"], context=f"{context}.adoption.created_utc"
        )
        if not (
            calibration_time
            < oos_source_time
            < proof_time
            < joint_proof_time
            < adoption_time
        ):
            _fail(
                context,
                "required chronology is calibration < OOS source < proof < "
                "joint Exit replay < adoption",
            )
        if calibration_binding != adoption["calibration_artifact"]:
            _fail(context, "calibration binding canonicalization mismatch")
        if proof_binding != adoption["oos_proof_artifact"]:
            _fail(context, "OOS proof binding canonicalization mismatch")
        if joint_proof_binding != adoption["joint_exit_sizing_proof_artifact"]:
            _fail(context, "joint Exit proof binding canonicalization mismatch")
    except (ModelNativeSizingContractError, ModelNativeSizingExecutionContractError) as exc:
        raise ModelNativeSizingUnavailable(str(exc)) from exc
    specs = _snapshot_file_specs(
        authority=authority,
        adoption=adoption,
        calibration=calibration,
        proof=proof,
        joint_proof=joint_proof,
        context=f"{context}.snapshot",
    )
    return ValidatedLearnedSizingAuthority(
        authority_json=_canonical_json(authority, context=f"{context}.authority"),
        adoption_json=_canonical_json(adoption, context=f"{context}.adoption"),
        calibration_json=_canonical_json(calibration, context=f"{context}.calibration"),
        proof_json=_canonical_json(proof, context=f"{context}.proof"),
        joint_proof_json=_canonical_json(
            joint_proof, context=f"{context}.joint_exit_sizing_proof"
        ),
        content_hash_key=specs,
        file_stats=_capture_file_stats(specs, context=f"{context}.snapshot"),
    )


def prepare_model_native_sizing_authority(
    contract: Mapping[str, Any] | Any,
    *,
    context: str,
) -> ValidatedLearnedSizingAuthority:
    """Verify once per unique adoption hash and then perform O(files) stat checks.

    A file-stat change taints that adoption for the rest of the process.  It is
    never silently reloaded or re-authorized after mutation.
    """

    authority = require_model_native_sizing_authority_contract(
        contract,
        context=context,
        required_mode=MODEL_NATIVE_SIZING_MODE_LEARNED,
        verify_bound_artifacts=False,
    )
    key = _cache_lookup_key(authority)
    with _CACHE_LOCK:
        if key in _TAINTED_CACHE_KEYS:
            _fail(context, "sizing adoption was invalidated by a changed/missing file")
        cached = _VALIDATED_CACHE.get(key)
        if cached is not None:
            _require_or_taint_snapshot_unchanged(cached, context=context)
            return cached
        snapshot = _validate_learned_sizing_authority_snapshot(
            authority,
            context=context,
        )
        _require_or_taint_snapshot_unchanged(snapshot, context=context)
        _VALIDATED_CACHE[key] = snapshot
        return snapshot


def load_learned_sizing_authority_evidence(
    contract: Mapping[str, Any] | Any,
    *,
    context: str,
    verify_lineage_files: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compatibility loader backed by the immutable validated snapshot cache."""

    if verify_lineage_files is not True:
        _fail(context, "full learned authority requires lineage-file verification")
    snapshot = prepare_model_native_sizing_authority(contract, context=context)
    return (
        snapshot.authority,
        snapshot.adoption,
        snapshot.calibration,
        snapshot.proof,
    )


def _finite(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        _fail(context, "boolean is not numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        _fail(context, f"numeric value required: {value!r}")
    if not math.isfinite(parsed):
        _fail(context, f"non-finite value: {value!r}")
    return parsed


def _apply_validated_model_native_sizing(
    *,
    validated_authority: ValidatedLearnedSizingAuthority,
    position_size_logit: Any,
    model_direction: str,
    runtime_constraints: Mapping[str, Any] | Any,
    context: str,
    check_file_stats: bool,
) -> dict[str, Any]:
    """Pure/O(1) application against a startup-validated snapshot."""

    if not isinstance(validated_authority, ValidatedLearnedSizingAuthority):
        _fail(context, "startup-validated sizing snapshot is required")
    if check_file_stats:
        _require_or_taint_snapshot_unchanged(validated_authority, context=context)
    direction = str(model_direction).strip().upper()
    direction_index_by_name = {"LONG": 0, "SHORT": 1, "FLAT": 2}
    if direction not in direction_index_by_name:
        _fail(context, f"invalid model_direction={model_direction!r}")
    logit = _finite(position_size_logit, context=f"{context}.position_size_logit")
    authority = validated_authority.authority
    adoption = validated_authority.adoption
    calibration = validated_authority.calibration
    transformed = calibrated_sizing_transform(
        calibration=calibration,
        position_size_logit=logit,
        model_direction_index=direction_index_by_name[direction],
        runtime_constraints=runtime_constraints,
        context=f"{context}.transform",
    )
    return {
        "schema_version": MODEL_NATIVE_SIZING_APPLICATION_SCHEMA_VERSION,
        "sizing_mode": MODEL_NATIVE_SIZING_MODE_LEARNED,
        "model_direction": direction,
        "model_direction_index": transformed["model_direction_index"],
        "position_size_logit": transformed["position_size_logit"],
        "calibrated_size_fraction": transformed["calibrated_size_fraction"],
        "applied_size_multiplier": transformed["applied_size_multiplier"],
        "capacity_units": transformed["capacity_units"],
        "reference_pre_round_units": transformed["reference_pre_round_units"],
        "pre_round_units": transformed["pre_round_units"],
        "units": transformed["units"],
        "rounding_mode": "floor_to_unit_step",
        "authorized_order": transformed["authorized_order"],
        "no_order_reason": transformed["no_order_reason"],
        "runtime_constraints": transformed["runtime_constraints"],
        "calibration_artifact_sha256": adoption["calibration_artifact"]["sha256"],
        "oos_proof_artifact_sha256": adoption["oos_proof_artifact"]["sha256"],
        "adoption_artifact_sha256": authority["adoption_artifact"]["sha256"],
        "sizing_authority_contract": authority,
    }


def apply_model_native_sizing(
    *,
    validated_authority: ValidatedLearnedSizingAuthority,
    position_size_logit: Any,
    model_direction: str,
    runtime_constraints: Mapping[str, Any] | Any,
    context: str,
) -> dict[str, Any]:
    """Apply after startup verification with only cheap stat/newest-event checks."""

    return _apply_validated_model_native_sizing(
        validated_authority=validated_authority,
        position_size_logit=position_size_logit,
        model_direction=model_direction,
        runtime_constraints=runtime_constraints,
        context=context,
        check_file_stats=True,
    )


def require_model_native_sizing_application_record(
    application: Mapping[str, Any] | Any,
    *,
    context: str,
) -> dict[str, Any]:
    """Reload the adoption chain and recompute one recorded application exactly."""

    observed = _exact_keys(application, _APPLICATION_KEYS, context=context)
    if observed["schema_version"] != MODEL_NATIVE_SIZING_APPLICATION_SCHEMA_VERSION:
        _fail(context, "application schema_version mismatch")
    snapshot = prepare_model_native_sizing_authority(
        observed["sizing_authority_contract"], context=context
    )
    expected = _apply_validated_model_native_sizing(
        validated_authority=snapshot,
        position_size_logit=observed["position_size_logit"],
        model_direction=observed["model_direction"],
        runtime_constraints=observed["runtime_constraints"],
        context=context,
        check_file_stats=False,
    )
    if observed != expected:
        mismatched = sorted(
            key for key, expected_value in expected.items()
            if observed.get(key) != expected_value
        )
        _fail(context, f"application recomputation mismatch: {mismatched}")
    if observed["model_direction"] == "FLAT" and observed["units"] != 0:
        _fail(context, "FLAT must have zero units")
    if observed["authorized_order"] is not (observed["units"] > 0):
        _fail(context, "authorized_order/units mismatch")
    return observed


__all__ = [
    "MODEL_NATIVE_SIZING_ADOPTION_SCHEMA_VERSION",
    "MODEL_NATIVE_SIZING_APPLICATION_SCHEMA_VERSION",
    "MODEL_NATIVE_SIZING_AUTHORITY_SCHEMA_VERSION",
    "MODEL_NATIVE_SIZING_BUNDLE_CALIBRATION_SCHEMA_VERSION",
    "MODEL_NATIVE_SIZING_MODE_HISTORICAL_FIXED",
    "MODEL_NATIVE_SIZING_MODE_LEARNED",
    "MODEL_NATIVE_SIZING_MODES",
    "ModelNativeSizingUnavailable",
    "ValidatedLearnedSizingAuthority",
    "apply_model_native_sizing",
    "historical_fixed_1x_negative_control_metadata",
    "learned_sizing_authority_contract_metadata",
    "load_learned_sizing_authority_evidence",
    "model_native_sizing_authority_contract_metadata",
    "model_native_sizing_bundle_calibration_metadata",
    "prepare_model_native_sizing_authority",
    "require_model_native_sizing_adoption_artifact",
    "require_model_native_sizing_application_record",
    "require_model_native_sizing_authority_contract",
    "require_model_native_sizing_bundle_calibration",
]
