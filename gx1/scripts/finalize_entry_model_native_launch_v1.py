"""Transactionally publish one complete model-native Entry launch authority.

This is the only producer allowed to add ``active.v10_entry`` and change the
XAU direction launch state to ``ALLOW``.  It validates the complete learned
serve/sizing/Exit/lifecycle chain against staged bytes before either authority
file is replaced.  An immutable commit event binds both final file hashes; a
partial replacement is therefore never launch-valid and is rolled back with a
newer terminal failure event.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import timezone
from functools import wraps
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.entry_model_native_bundle_commit_v1 import (
    require_bundle_commit_manifest,
)
from gx1.contracts.entry_model_native_launch_approval_v1 import (
    EVENT_PREFIX as APPROVAL_EVENT_PREFIX,
    PROJECT,
    SCHEMA_VERSION as APPROVAL_SCHEMA_VERSION,
    launch_vedtak_request,
    launch_state_approval_payload_sha256,
    require_preexisting_launch_vedtak,
)
from gx1.contracts.entry_model_native_launch_transaction_v1 import (
    EVENT_PREFIX,
    SCHEMA_VERSION,
    EntryLaunchTransactionError,
    entry_launch_authority_lock,
    launch_transaction_declaration,
    require_launch_transaction_commit_event,
    sha256_file,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_SIGNAL_DIM,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DIM,
    MODEL_NATIVE_CTX_CONT_DIM,
    MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT,
    MODEL_NATIVE_SELECTED_FEATURE_COUNT,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_model_native_sizing_authority_v1 import (
    MODEL_NATIVE_SIZING_MODE_LEARNED,
    learned_sizing_authority_contract_metadata,
)
from gx1.contracts.entry_model_native_sizing_execution_v1 import (
    ModelNativeSizingExecutionContractError,
    require_canonical_unified_replay_launch_authority,
)
from gx1.contracts.entry_run_lineage_v1 import (
    EntryRunLineageError,
    require_entry_run_id,
)
from gx1.contracts.immutable_event_authority_v1 import (
    next_immutable_event_created_utc,
    select_latest_immutable_event,
    write_immutable_json_event,
)
from gx1.contracts.live_tail_publication_v1 import (
    LiveTailAuthorityError,
    live_tail_launch_authority,
    require_newest_live_tail_runtime_authority,
)
from gx1.features.entry_model_native_feature_layers_v1 import (
    MODEL_NATIVE_MANDATORY_FAMILY_COUNT,
)
from gx1.models.entry_v10.direction_decision_contract import (
    MODEL_DIRECTION_CLASS_ORDER,
    MODEL_DIRECTION_SELECTION_MODE,
    UNIFIED_ENTRY_EXIT_CONTRACT_SCHEMA_VERSION,
    UNIFIED_EXIT_ACTION_ORDER,
    require_model_direction_operating_point,
)
from gx1_guards import artifacts as artifact_guard


FAILURE_SCHEMA_VERSION = "entry_model_native_launch_transaction_terminal_failure_v1"
_BINDING_KEYS = frozenset({"json_path", "sha256"})
CANONICAL_LAUNCH_AUTHORITY_ROOT = Path(
    "/home/andre2/GX1_DATA/reports/entry_model_native_launch_authority"
)
CANONICAL_LAUNCH_VEDTAK_DIR = CANONICAL_LAUNCH_AUTHORITY_ROOT / "vedtak"
CANONICAL_LAUNCH_APPROVAL_DIR = CANONICAL_LAUNCH_AUTHORITY_ROOT / "approval"
CANONICAL_LAUNCH_TRANSACTION_DIR = (
    CANONICAL_LAUNCH_AUTHORITY_ROOT / "transaction"
)
_FAILURE_EVENT_KEYS = frozenset(
    {
        "schema_version",
        "created_utc",
        "json_path",
        "decision",
        "failures",
        "project",
        "transaction_id",
        "commit_event",
        "rollback_complete",
    }
)


class EntryLaunchFinalizationError(RuntimeError):
    """The complete launch authority could not be committed."""


def _serialized_launch_finalization(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if args:
            raise TypeError("launch finalization accepts keyword arguments only")
        with entry_launch_authority_lock(
            registry_path=Path(kwargs["artifact_registry_path"]),
            launch_state_path=Path(kwargs["launch_state_path"]),
        ):
            return function(**kwargs)

    return wrapped


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(value),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EntryLaunchFinalizationError(
            "launch payload is not strict JSON"
        ) from exc


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _binding(path: Path, *, label: str) -> dict[str, str]:
    raw = Path(path).expanduser()
    if (
        not raw.is_absolute()
        or raw.is_symlink()
        or not raw.is_file()
        or "latest" in raw.name.lower()
    ):
        raise EntryLaunchFinalizationError(
            f"{label} must be an explicit immutable absolute file: {raw}"
        )
    path = raw.resolve()
    return {"json_path": str(path), "sha256": sha256_file(path)}


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    raw = Path(path).expanduser()
    if not raw.is_absolute() or raw.is_symlink() or not raw.is_file():
        raise EntryLaunchFinalizationError(
            f"{label} must be an existing absolute regular file: {raw}"
        )
    try:
        value = json.loads(raw.read_text(encoding="utf-8"))
    except Exception as exc:
        raise EntryLaunchFinalizationError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise EntryLaunchFinalizationError(f"{label} root must be an object")
    return value


def _object_from_bytes(
    encoded: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    try:
        value = json.loads(encoded.decode("utf-8"))
    except Exception as exc:
        raise EntryLaunchFinalizationError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise EntryLaunchFinalizationError(f"{label} root must be an object")
    return value


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _stage_bytes(target: Path, encoded: bytes) -> Path:
    target = target.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{target.name}.launch-staging.",
        dir=str(target.parent),
    )
    stage = Path(name)
    try:
        os.fchmod(fd, 0o644)
        view = memoryview(encoded)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError(f"short write while staging {target}")
            view = view[written:]
        os.fsync(fd)
    except Exception:
        os.close(fd)
        stage.unlink(missing_ok=True)
        raise
    os.close(fd)
    return stage


def _publish_backup(path: Path, encoded: bytes) -> dict[str, str]:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise EntryLaunchFinalizationError(
            f"immutable transaction backup already exists: {path}"
        )
    stage = _stage_bytes(path, encoded)
    try:
        os.link(stage, path)
        _fsync_directory(path.parent)
    finally:
        stage.unlink(missing_ok=True)
    return {"path": str(path), "sha256": _sha_bytes(encoded)}


def _replace_target(stage: Path, target: Path) -> None:
    os.replace(stage, target)
    _fsync_directory(target.parent)


def _restore_target(backup: Mapping[str, Any], target: Path) -> None:
    if set(backup) != {"path", "sha256"}:
        raise EntryLaunchFinalizationError("transaction backup binding is malformed")
    backup_path = Path(str(backup["path"])).expanduser()
    if (
        not backup_path.is_absolute()
        or backup_path.is_symlink()
        or not backup_path.is_file()
    ):
        raise EntryLaunchFinalizationError("transaction backup bytes are invalid")
    encoded = backup_path.read_bytes()
    if _sha_bytes(encoded) != backup["sha256"]:
        raise EntryLaunchFinalizationError("transaction backup bytes are invalid")
    stage = _stage_bytes(target, encoded)
    _replace_target(stage, target)


def _failure_event(
    *,
    transaction_dir: Path,
    transaction_id: str,
    error: BaseException,
    commit_event: Mapping[str, str] | None,
    rollback_complete: bool,
) -> tuple[Path, dict[str, Any]]:
    return write_immutable_json_event(
        transaction_dir,
        EVENT_PREFIX,
        {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "created_utc": next_immutable_event_created_utc(
                transaction_dir,
                EVENT_PREFIX,
            ).isoformat(),
            "decision": "FAIL",
            "failures": [f"{type(error).__name__}: {error}"],
            "project": PROJECT,
            "transaction_id": str(transaction_id),
            "commit_event": (
                dict(commit_event) if commit_event is not None else None
            ),
            "rollback_complete": bool(rollback_complete),
        },
    )


def _require_vedtak_unused(
    *,
    vedtak_id: str,
    approval_dir: Path,
    transaction_dir: Path,
) -> None:
    for root, prefix in (
        (approval_dir, APPROVAL_EVENT_PREFIX),
        (transaction_dir, EVENT_PREFIX),
    ):
        for path in sorted(root.glob(f"{prefix}_*.json")):
            if path.is_symlink() or not path.is_file():
                raise EntryLaunchFinalizationError(
                    f"launch authority inventory contains an invalid path: {path}"
                )
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise EntryLaunchFinalizationError(
                    f"launch authority inventory is unreadable: {path}"
                ) from exc
            if (
                isinstance(payload, Mapping)
                and payload.get("vedtak_id") == vedtak_id
            ):
                raise EntryLaunchFinalizationError(
                    f"launch vedtak has already been consumed: {vedtak_id}"
                )


def _load_commit_from_event(
    event: Mapping[str, Any],
    *,
    transaction_dir: Path,
    registry_path: Path,
    launch_state_path: Path,
) -> tuple[Path, dict[str, Any]] | None:
    if event.get("schema_version") == SCHEMA_VERSION and event.get("decision") == "COMMIT":
        path = Path(str(event.get("json_path") or "")).resolve()
        payload = require_launch_transaction_commit_event(
            path,
            transaction_dir=transaction_dir,
            target_launch_state_path=launch_state_path,
            target_registry_path=registry_path,
        )
        return path, payload
    binding = event.get("commit_event")
    if not isinstance(binding, Mapping) or set(binding) != set(_BINDING_KEYS):
        return None
    path = Path(str(binding.get("json_path") or "")).expanduser()
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve().parent != transaction_dir.resolve()
        or sha256_file(path.resolve()) != binding.get("sha256")
    ):
        raise EntryLaunchFinalizationError(
            "failed transaction commit binding is invalid"
        )
    payload = require_launch_transaction_commit_event(
        path.resolve(),
        transaction_dir=transaction_dir,
        target_launch_state_path=launch_state_path,
        target_registry_path=registry_path,
        expected_event_sha256=str(binding["sha256"]),
    )
    return path.resolve(), payload


def _require_terminal_failure_event(
    event: Mapping[str, Any],
    *,
    event_path: Path,
) -> dict[str, Any]:
    observed = dict(event)
    if (
        set(observed) != set(_FAILURE_EVENT_KEYS)
        or observed.get("schema_version") != FAILURE_SCHEMA_VERSION
        or observed.get("decision") != "FAIL"
        or observed.get("project") != PROJECT
        or not isinstance(observed.get("failures"), list)
        or not observed["failures"]
        or not all(isinstance(item, str) and item for item in observed["failures"])
        or not isinstance(observed.get("rollback_complete"), bool)
        or Path(str(observed.get("json_path") or "")).expanduser().resolve()
        != event_path.resolve()
    ):
        raise EntryLaunchFinalizationError(
            "newest launch transaction failure event is malformed"
        )
    launch_transaction_declaration(
        transaction_id=str(observed.get("transaction_id") or ""),
        event_dir=event_path.parent,
    )
    binding = observed.get("commit_event")
    if binding is not None and (
        not isinstance(binding, Mapping) or set(binding) != set(_BINDING_KEYS)
    ):
        raise EntryLaunchFinalizationError(
            "newest launch transaction failure commit binding is malformed"
        )
    return observed


def _recover_interrupted_transaction(
    *,
    transaction_dir: Path,
    registry_path: Path,
    launch_state_path: Path,
) -> tuple[Path, dict[str, Any]] | None:
    newest_path = select_latest_immutable_event(transaction_dir, EVENT_PREFIX)
    if newest_path is None:
        return
    newest = _read_object(newest_path, label="newest launch transaction")
    if newest.get("decision") == "FAIL":
        newest = _require_terminal_failure_event(
            newest,
            event_path=newest_path,
        )
    commit = _load_commit_from_event(
        newest,
        transaction_dir=transaction_dir,
        registry_path=registry_path,
        launch_state_path=launch_state_path,
    )
    if commit is None:
        if newest.get("decision") == "FAIL" and newest.get("rollback_complete") is True:
            return
        raise EntryLaunchFinalizationError(
            "newest transaction is neither recoverable nor committed"
        )
    commit_path, payload = commit
    rollback_complete = (
        newest.get("decision") == "FAIL"
        and newest.get("rollback_complete") is True
        and registry_path.is_file()
        and launch_state_path.is_file()
        and sha256_file(registry_path)
        == payload.get("registry_before_artifact", {}).get("sha256")
        and sha256_file(launch_state_path)
        == payload.get("launch_state_before_artifact", {}).get("sha256")
    )
    if rollback_complete:
        return
    target_complete = (
        registry_path.is_file()
        and launch_state_path.is_file()
        and sha256_file(registry_path) == payload.get("registry_sha256")
        and sha256_file(launch_state_path) == payload.get("launch_state_sha256")
    )
    if newest.get("decision") == "COMMIT" and target_complete:
        return commit_path, payload
    rollback_complete = False
    recovery_error = EntryLaunchFinalizationError(
        "recovered an interrupted prior launch transaction; explicit rerun required"
    )
    try:
        _restore_target(payload["registry_before_artifact"], registry_path)
        _restore_target(payload["launch_state_before_artifact"], launch_state_path)
        rollback_complete = True
    finally:
        _failure_event(
            transaction_dir=transaction_dir,
            transaction_id=str(payload.get("transaction_id") or "UNKNOWN"),
            error=recovery_error,
            commit_event={
                "json_path": str(commit_path),
                "sha256": sha256_file(commit_path),
            },
            rollback_complete=rollback_complete,
        )
    raise recovery_error


def _require_idempotent_retry_matches(
    *,
    commit: Mapping[str, Any],
    launch_state_path: Path,
    transaction_id: str,
    accepted_bundle_dir: Path,
    sizing_adoption_path: Path,
    joint_exit_proof_path: Path,
    sizing_runtime_parity_path: Path,
    serve_parity_path: Path,
    direction_pocket_path: Path,
    adaptation_lifecycle_path: Path,
    live_tail_admission_path: Path,
    launch_vedtak_path: Path,
    max_trades: int,
) -> None:
    """Allow a completed retry only for the exact already-committed request."""

    state = _read_object(launch_state_path, label="committed launch state")
    if str(commit.get("transaction_id") or "") != str(transaction_id):
        raise EntryLaunchFinalizationError(
            "completed launch transaction differs from retry transaction_id"
        )
    try:
        observed_bundle = Path(
            str(state.get("accepted_bundle_dir") or "")
        ).expanduser().resolve()
        requested_bundle = Path(accepted_bundle_dir).expanduser().resolve()
        observed_max_trades = state["operating_point"]["max_trades"]
        observed_transaction = state["launch_transaction"]["transaction_id"]
        vedtak_binding = state["accepted_via_vedtak"]["vedtak_authority"]
        expected_paths = {
            "sizing adoption": state["sizing_authority_contract"][
                "adoption_artifact"
            ]["json_path"],
            "joint Exit proof": state[
                "joint_exit_execution_proof_evidence"
            ]["json_path"],
            "sizing runtime parity": state[
                "sizing_runtime_parity_evidence"
            ]["json_path"],
            "serve parity": state["serve_gate_evidence"][
                "model_native_serve_parity"
            ]["json_path"],
            "direction pocket": state["serve_gate_evidence"][
                "model_native_direction_pocket_audit"
            ]["json_path"],
            "adaptation lifecycle": state[
                "adaptation_lifecycle_evidence"
            ]["json_path"],
            "live-tail admission": state[
                "new_entry_live_tail_authority"
            ]["launch_admission"]["json_path"],
        }
    except (KeyError, TypeError, AttributeError) as exc:
        raise EntryLaunchFinalizationError(
            "completed launch state cannot authorize idempotent retry"
        ) from exc
    if (
        observed_bundle != requested_bundle
        or observed_max_trades != max_trades
        or observed_transaction != str(transaction_id)
    ):
        raise EntryLaunchFinalizationError(
            "completed launch transaction differs from retry request"
        )
    requested_paths = {
        "sizing adoption": sizing_adoption_path,
        "joint Exit proof": joint_exit_proof_path,
        "sizing runtime parity": sizing_runtime_parity_path,
        "serve parity": serve_parity_path,
        "direction pocket": direction_pocket_path,
        "adaptation lifecycle": adaptation_lifecycle_path,
        "live-tail admission": live_tail_admission_path,
    }
    for label, requested in requested_paths.items():
        if Path(requested).expanduser().resolve() != Path(
            str(expected_paths[label])
        ).expanduser().resolve():
            raise EntryLaunchFinalizationError(
                f"completed launch transaction differs from retry {label}"
            )
    vedtak = Path(launch_vedtak_path).expanduser().resolve()
    if (
        vedtak != Path(str(vedtak_binding.get("json_path") or "")).resolve()
        or not vedtak.is_file()
        or sha256_file(vedtak) != vedtak_binding.get("sha256")
    ):
        raise EntryLaunchFinalizationError(
            "completed launch transaction differs from retry vedtak"
        )


def _build_registry(
    current: Mapping[str, Any],
    *,
    bundle_dir: Path,
    bundle_commit_sha256: str,
    transaction_id: str,
    vedtak_id: str,
    operating_point: Mapping[str, Any],
    updated_utc: str,
) -> dict[str, Any]:
    registry = json.loads(json.dumps(dict(current), allow_nan=False))
    if (
        registry.get("schema_version") != "gx1_artifact_selection_v2"
        or registry.get("project") != PROJECT
        or not isinstance(registry.get("active"), dict)
        or not isinstance(registry.get("history"), list)
    ):
        raise EntryLaunchFinalizationError(
            "artifact registry is not the exact XAUUSD v2 authority"
        )
    active = registry["active"]
    previous = active.get("v10_entry")
    if previous is not None:
        if not isinstance(previous, dict):
            raise EntryLaunchFinalizationError(
                "existing active.v10_entry is malformed"
            )
        if Path(str(previous.get("path") or "")).expanduser().resolve() == bundle_dir:
            raise EntryLaunchFinalizationError(
                "refusing no-op launch transaction for the already active bundle"
            )
        registry["history"].append(
            {
                "role": "v10_entry",
                "path": previous.get("path"),
                "status": "SUPERSEDED_BY_MODEL_NATIVE_LAUNCH_TRANSACTION",
                "reason": f"superseded by {transaction_id}",
            }
        )
    active["v10_entry"] = {
        "path": str(bundle_dir),
        "status": "ACTIVE",
        "in_sample_only": False,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "operating_point": dict(operating_point),
        "launch_transaction_id": transaction_id,
        "bundle_commit_sha256": bundle_commit_sha256,
        "vedtak": vedtak_id,
    }
    registry["updated_utc"] = updated_utc
    registry["note"] = (
        "Only exact transaction-bound XAUUSD artifacts under active are "
        "decision-valid. Entry activation additionally requires the matching "
        "launch-state, approval and newest launch-transaction commit."
    )
    return registry


def _build_launch_state(
    *,
    bundle_dir: Path,
    bundle_metadata_sha256: str,
    dataset_run_id: str,
    training_run_id: str,
    sizing_adoption_binding: Mapping[str, Any],
    joint_exit_binding: Mapping[str, Any],
    runtime_parity_binding: Mapping[str, Any],
    serve_parity_binding: Mapping[str, Any],
    direction_pocket_binding: Mapping[str, Any],
    lifecycle_binding: Mapping[str, Any],
    live_tail_authority: Mapping[str, Any],
    operating_point: Mapping[str, Any],
    transaction_id: str,
    transaction_dir: Path,
    updated_utc: str,
) -> dict[str, Any]:
    return {
        "schema_version": "gx1_xau_direction_launch_state_v1",
        "project": PROJECT,
        "updated_utc": updated_utc,
        "decision": "ALLOW",
        "latest_terminal_event_id": training_run_id,
        "latest_terminal_event_decision": "PASS",
        "decision_surface": "model_direction_argmax",
        "public_trade_flat_surface": "public_trade_flat_decision_logits",
        "required_contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "required_unified_entry_exit_contract": (
            UNIFIED_ENTRY_EXIT_CONTRACT_SCHEMA_VERSION
        ),
        "required_entry_action_order": list(MODEL_DIRECTION_CLASS_ORDER),
        "required_exit_action_order": list(UNIFIED_EXIT_ACTION_ORDER),
        "required_same_bundle_shared_encoder": True,
        "required_exact_closed_m1_exit_path_envelope": True,
        "external_decision_models_allowed": False,
        "required_signal_dim": MODEL_NATIVE_SIGNAL_DIM,
        "required_base_signal_dim": MODEL_NATIVE_BASE_SIGNAL_DIM,
        "required_selected_feature_count": MODEL_NATIVE_SELECTED_FEATURE_COUNT,
        "required_mandatory_causal_layer_feature_count": (
            MODEL_NATIVE_MANDATORY_SELECTED_FEATURE_COUNT
        ),
        "required_train_ranked_remainder_feature_count": (
            MODEL_NATIVE_RANKED_REMAINDER_FEATURE_COUNT
        ),
        "required_mandatory_causal_layer_count": (
            MODEL_NATIVE_MANDATORY_FAMILY_COUNT
        ),
        "required_ctx_cont_dim": MODEL_NATIVE_CTX_CONT_DIM,
        "required_ctx_cat_dim": MODEL_NATIVE_CTX_CAT_DIM,
        "dataset_event_id": dataset_run_id,
        "accepted_bundle_dir": str(bundle_dir),
        "bundle_metadata_sha256": bundle_metadata_sha256,
        "sizing_adoption_mode": MODEL_NATIVE_SIZING_MODE_LEARNED,
        "sizing_authority_contract": (
            learned_sizing_authority_contract_metadata(
                adoption_artifact=sizing_adoption_binding,
            )
        ),
        "joint_exit_execution_proof_evidence": dict(joint_exit_binding),
        "sizing_runtime_parity_evidence": dict(runtime_parity_binding),
        "serve_gate_evidence": {
            "model_native_serve_parity": dict(serve_parity_binding),
            "model_native_direction_pocket_audit": dict(
                direction_pocket_binding
            ),
        },
        "adaptation_lifecycle_evidence": dict(lifecycle_binding),
        "new_entry_live_tail_authority": dict(live_tail_authority),
        "operating_point": dict(operating_point),
        "launch_transaction": launch_transaction_declaration(
            transaction_id=transaction_id,
            event_dir=transaction_dir,
        ),
        "accepted_via_vedtak": None,
        "blockers": [],
    }


def _require_current_launch_live_tail(
    *,
    authority: Mapping[str, Any],
    admission_binding: Mapping[str, str],
) -> dict[str, Any]:
    """Prove that the exact launch admission is still newest and fresh."""

    try:
        runtime = require_newest_live_tail_runtime_authority(authority)
    except LiveTailAuthorityError as exc:
        raise EntryLaunchFinalizationError(
            f"live-tail launch admission invalid: {exc}"
        ) from exc
    current = runtime.get("current_admission")
    anchor = authority.get("launch_anchor")
    if not isinstance(current, Mapping) or not isinstance(anchor, Mapping):
        raise EntryLaunchFinalizationError(
            "live-tail launch authority response is malformed"
        )
    expected = {
        "path": admission_binding["json_path"],
        "sha256": admission_binding["sha256"],
        "pair_generation_id": anchor["pair_generation_id"],
        "generation_manifest_sha256": anchor[
            "generation_manifest_sha256"
        ],
    }
    if any(current.get(field) != value for field, value in expected.items()):
        raise EntryLaunchFinalizationError(
            "live-tail launch admission is not the newest current authority"
        )
    return runtime


@_serialized_launch_finalization
def finalize_entry_model_native_launch(
    *,
    accepted_bundle_dir: Path,
    sizing_adoption_path: Path,
    joint_exit_proof_path: Path,
    sizing_runtime_parity_path: Path,
    serve_parity_path: Path,
    direction_pocket_path: Path,
    adaptation_lifecycle_path: Path,
    live_tail_admission_path: Path,
    launch_vedtak_path: Path,
    transaction_id: str,
    max_trades: int,
    artifact_registry_path: Path,
    launch_state_path: Path,
    approval_event_dir: Path,
    transaction_event_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    """Validate, stage and commit one complete launch transaction."""

    registry_path = Path(artifact_registry_path).expanduser()
    state_path = Path(launch_state_path).expanduser()
    approval_dir = Path(approval_event_dir).expanduser()
    transaction_dir = Path(transaction_event_dir).expanduser()
    for path, label in (
        (registry_path, "artifact registry"),
        (state_path, "launch state"),
    ):
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            raise EntryLaunchFinalizationError(
                f"{label} must be an explicit absolute regular file"
            )
    for path, label in (
        (approval_dir, "approval event dir"),
        (transaction_dir, "transaction event dir"),
    ):
        if not path.is_absolute() or path.is_symlink():
            raise EntryLaunchFinalizationError(f"{label} must be absolute")
        path.mkdir(parents=True, exist_ok=True)
    registry_path = registry_path.resolve()
    state_path = state_path.resolve()
    approval_dir = approval_dir.resolve()
    transaction_dir = transaction_dir.resolve()
    if approval_dir == transaction_dir:
        raise EntryLaunchFinalizationError(
            "approval and transaction events require separate directories"
        )
    try:
        recovered_commit = _recover_interrupted_transaction(
            transaction_dir=transaction_dir,
            registry_path=registry_path,
            launch_state_path=state_path,
        )
    except EntryLaunchTransactionError as exc:
        raise EntryLaunchFinalizationError(str(exc)) from exc
    if recovered_commit is not None:
        recovered_path, recovered_event = recovered_commit
        _require_idempotent_retry_matches(
            commit=recovered_event,
            launch_state_path=state_path,
            transaction_id=str(transaction_id),
            accepted_bundle_dir=accepted_bundle_dir,
            sizing_adoption_path=sizing_adoption_path,
            joint_exit_proof_path=joint_exit_proof_path,
            sizing_runtime_parity_path=sizing_runtime_parity_path,
            serve_parity_path=serve_parity_path,
            direction_pocket_path=direction_pocket_path,
            adaptation_lifecycle_path=adaptation_lifecycle_path,
            live_tail_admission_path=live_tail_admission_path,
            launch_vedtak_path=launch_vedtak_path,
            max_trades=max_trades,
        )
        return recovered_path, recovered_event

    operating_point = require_model_direction_operating_point(
        {
            "selection_score": MODEL_DIRECTION_SELECTION_MODE,
            "max_trades": max_trades,
        },
        context="entry launch finalizer",
    )
    bundle = Path(accepted_bundle_dir).expanduser()
    if (
        not bundle.is_absolute()
        or bundle.is_symlink()
        or not bundle.is_dir()
    ):
        raise EntryLaunchFinalizationError(
            "accepted bundle must be an explicit immutable absolute directory"
        )
    bundle = bundle.resolve()
    bundle_commit = require_bundle_commit_manifest(bundle)
    metadata_path = bundle / "bundle_metadata.json"
    metadata = _read_object(metadata_path, label="bundle metadata")
    lineage = metadata.get("run_lineage")
    if (
        not isinstance(lineage, Mapping)
        or set(lineage)
        != {
            "schema_version",
            "training_run_id",
            "dataset_run_id",
            "training_profile",
            "requested_subsample_rows",
            "physical_train_rows",
            "effective_train_rows",
        }
        or lineage.get("schema_version")
        != "entry_model_native_training_run_lineage_v2"
        or lineage.get("training_profile") != "candidate"
        or lineage.get("requested_subsample_rows") != 0
        or lineage.get("physical_train_rows")
        != lineage.get("effective_train_rows")
    ):
        raise EntryLaunchFinalizationError(
            "bundle run_lineage schema is absent or noncanonical"
        )
    try:
        dataset_run_id = require_entry_run_id(lineage.get("dataset_run_id"))
        training_run_id = require_entry_run_id(lineage.get("training_run_id"))
    except EntryRunLineageError as exc:
        raise EntryLaunchFinalizationError(str(exc)) from exc
    if dataset_run_id == training_run_id:
        raise EntryLaunchFinalizationError(
            "dataset and training run IDs must remain distinct"
        )

    sizing_adoption = _binding(
        sizing_adoption_path,
        label="sizing adoption",
    )
    joint_exit = _binding(joint_exit_proof_path, label="joint Exit proof")
    try:
        require_canonical_unified_replay_launch_authority(
            _read_object(joint_exit_proof_path, label="joint Exit proof"),
            context="ENTRY_LAUNCH_ACTIVE_EXIT_REPLAY_PRODUCER",
        )
    except ModelNativeSizingExecutionContractError as exc:
        raise EntryLaunchFinalizationError(str(exc)) from exc
    runtime_parity = _binding(
        sizing_runtime_parity_path,
        label="sizing runtime parity",
    )
    serve_parity = _binding(serve_parity_path, label="serve parity")
    direction_pocket = _binding(
        direction_pocket_path,
        label="direction pocket audit",
    )
    lifecycle = _binding(
        adaptation_lifecycle_path,
        label="adaptation lifecycle",
    )
    lifecycle_payload = _read_object(
        adaptation_lifecycle_path,
        label="adaptation lifecycle",
    )
    if lifecycle_payload.get("entry_run_id") != training_run_id:
        raise EntryLaunchFinalizationError(
            "lifecycle entry_run_id differs from bundle training lineage"
        )
    live_tail_admission = _binding(
        live_tail_admission_path,
        label="live-tail admission",
    )
    try:
        new_entry_live_tail_authority = live_tail_launch_authority(
            Path(live_tail_admission["json_path"]),
            expected_sha256=live_tail_admission["sha256"],
        )
    except LiveTailAuthorityError as exc:
        raise EntryLaunchFinalizationError(
            f"live-tail launch admission invalid: {exc}"
        ) from exc
    _require_current_launch_live_tail(
        authority=new_entry_live_tail_authority,
        admission_binding=live_tail_admission,
    )
    evidence = {
        "sizing_adoption": sizing_adoption,
        "joint_exit_sizing_proof": joint_exit,
        "sizing_runtime_parity": runtime_parity,
        "model_native_serve_parity": serve_parity,
        "model_native_direction_pocket_audit": direction_pocket,
        "adaptation_lifecycle": lifecycle,
        "live_tail_admission": live_tail_admission,
    }
    request = launch_vedtak_request(
        transaction_id=str(transaction_id),
        accepted_bundle_dir=bundle,
        bundle_commit_sha256=bundle_commit["commit_sha256"],
        target_registry_path=registry_path,
        target_launch_state_path=state_path,
        operating_point=operating_point,
        evidence=evidence,
    )
    vedtak_authority = require_preexisting_launch_vedtak(
        launch_vedtak_path,
        expected_request=request,
    )
    vedtak_id = str(vedtak_authority["vedtak_id"])
    vedtak_binding = {
        "json_path": str(vedtak_authority["event_path"]),
        "sha256": str(vedtak_authority["event_sha256"]),
    }
    _require_vedtak_unused(
        vedtak_id=vedtak_id,
        approval_dir=approval_dir,
        transaction_dir=transaction_dir,
    )

    current_registry_bytes = registry_path.read_bytes()
    current_registry = _object_from_bytes(
        current_registry_bytes,
        label="artifact registry",
    )
    current_state_bytes = state_path.read_bytes()
    updated = next_immutable_event_created_utc(
        transaction_dir,
        EVENT_PREFIX,
    ).astimezone(timezone.utc)
    updated_utc = updated.isoformat()
    state = _build_launch_state(
        bundle_dir=bundle,
        bundle_metadata_sha256=sha256_file(metadata_path),
        dataset_run_id=dataset_run_id,
        training_run_id=training_run_id,
        sizing_adoption_binding=sizing_adoption,
        joint_exit_binding=joint_exit,
        runtime_parity_binding=runtime_parity,
        serve_parity_binding=serve_parity,
        direction_pocket_binding=direction_pocket,
        lifecycle_binding=lifecycle,
        live_tail_authority=new_entry_live_tail_authority,
        operating_point=operating_point,
        transaction_id=transaction_id,
        transaction_dir=transaction_dir,
        updated_utc=updated_utc,
    )
    registry = _build_registry(
        current_registry,
        bundle_dir=bundle,
        bundle_commit_sha256=bundle_commit["commit_sha256"],
        transaction_id=str(transaction_id),
        vedtak_id=str(vedtak_id),
        operating_point=operating_point,
        updated_utc=updated_utc,
    )
    registry_bytes = _json_bytes(registry)
    commit_path: Path | None = None
    commit_binding: dict[str, str] | None = None
    backup_registry: dict[str, str] | None = None
    backup_state: dict[str, str] | None = None
    registry_stage: Path | None = None
    state_stage: Path | None = None
    replaced_registry = False
    replaced_state = False
    try:
        # Close the validation-to-publication gap: a newer admission, BLOCK,
        # pointer advance, or expiry after request construction invalidates
        # this launch before any approval/commit event is emitted.
        _require_current_launch_live_tail(
            authority=new_entry_live_tail_authority,
            admission_binding=live_tail_admission,
        )
        approval_path, _ = write_immutable_json_event(
            approval_dir,
            APPROVAL_EVENT_PREFIX,
            {
                "schema_version": APPROVAL_SCHEMA_VERSION,
                "created_utc": next_immutable_event_created_utc(
                    approval_dir,
                    APPROVAL_EVENT_PREFIX,
                ).isoformat(),
                "decision": "ALLOW",
                "project": PROJECT,
                "vedtak_id": str(vedtak_id),
                "accepted_bundle_dir": str(bundle),
                "bundle_commit_sha256": bundle_commit["commit_sha256"],
                "vedtak_authority": vedtak_binding,
                "launch_state_payload_sha256": (
                    launch_state_approval_payload_sha256(state)
                ),
            },
        )
        state["accepted_via_vedtak"] = {
            "schema_version": APPROVAL_SCHEMA_VERSION,
            "vedtak_id": str(vedtak_id),
            "event_path": str(approval_path),
            "event_sha256": sha256_file(approval_path),
            "vedtak_authority": vedtak_binding,
        }
        state_bytes = _json_bytes(state)
        backup_registry = _publish_backup(
            transaction_dir
            / f"ENTRY_MODEL_NATIVE_LAUNCH_BACKUP_{transaction_id}_registry.json",
            current_registry_bytes,
        )
        backup_state = _publish_backup(
            transaction_dir
            / f"ENTRY_MODEL_NATIVE_LAUNCH_BACKUP_{transaction_id}_state.json",
            current_state_bytes,
        )
        registry_stage = _stage_bytes(registry_path, registry_bytes)
        state_stage = _stage_bytes(state_path, state_bytes)
        _require_current_launch_live_tail(
            authority=new_entry_live_tail_authority,
            admission_binding=live_tail_admission,
        )
        commit_path, commit_event = write_immutable_json_event(
            transaction_dir,
            EVENT_PREFIX,
            {
                "schema_version": SCHEMA_VERSION,
                "created_utc": next_immutable_event_created_utc(
                    transaction_dir,
                    EVENT_PREFIX,
                    updated,
                ).isoformat(),
                "decision": "COMMIT",
                "failures": [],
                "project": PROJECT,
                "transaction_id": str(transaction_id),
                "vedtak_id": str(vedtak_id),
                "target_registry_path": str(registry_path),
                "target_launch_state_path": str(state_path),
                "registry_sha256": _sha_bytes(registry_bytes),
                "launch_state_sha256": _sha_bytes(state_bytes),
                "registry_before_artifact": backup_registry,
                "launch_state_before_artifact": backup_state,
                "accepted_bundle_dir": str(bundle),
                "bundle_commit_sha256": bundle_commit["commit_sha256"],
                "vedtak_authority": vedtak_binding,
            },
        )
        commit_binding = {
            "json_path": str(commit_path),
            "sha256": sha256_file(commit_path),
        }
        artifact_guard._check_v10_entry_launch_contract(
            bundle,
            launch_contract_path=state_stage,
            selection_contract_path=registry_stage,
            target_launch_contract_path=state_path,
            target_selection_contract_path=registry_path,
        )
        if (
            sha256_file(registry_path) != backup_registry["sha256"]
            or sha256_file(state_path) != backup_state["sha256"]
        ):
            raise EntryLaunchFinalizationError(
                "authority files changed concurrently before commit"
            )
        _replace_target(registry_stage, registry_path)
        replaced_registry = True
        _replace_target(state_stage, state_path)
        replaced_state = True
        artifact_guard._check_v10_entry_launch_contract(
            bundle,
            launch_contract_path=state_path,
            selection_contract_path=registry_path,
            target_launch_contract_path=state_path,
            target_selection_contract_path=registry_path,
        )
        return commit_path, commit_event
    except Exception as exc:
        rollback_complete = False
        rollback_error: Exception | None = None
        try:
            if replaced_registry or replaced_state:
                if backup_registry is None or backup_state is None:
                    raise EntryLaunchFinalizationError(
                        "authority replacement lacks rollback artifacts"
                    )
                _restore_target(backup_registry, registry_path)
                _restore_target(backup_state, state_path)
            rollback_complete = True
        except Exception as observed:
            rollback_error = observed
        try:
            _failure_event(
                transaction_dir=transaction_dir,
                transaction_id=str(transaction_id),
                error=exc,
                commit_event=commit_binding,
                rollback_complete=rollback_complete,
            )
        finally:
            if registry_stage is not None:
                registry_stage.unlink(missing_ok=True)
            if state_stage is not None:
                state_stage.unlink(missing_ok=True)
        if rollback_error is not None:
            exc.add_note(f"launch rollback also failed: {rollback_error}")
        raise EntryLaunchFinalizationError(str(exc)) from exc
    finally:
        if registry_stage is not None:
            registry_stage.unlink(missing_ok=True)
        if state_stage is not None:
            state_stage.unlink(missing_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--accepted-bundle-dir", type=Path, required=True)
    parser.add_argument("--sizing-adoption-json", type=Path, required=True)
    parser.add_argument("--joint-exit-proof-json", type=Path, required=True)
    parser.add_argument("--sizing-runtime-parity-json", type=Path, required=True)
    parser.add_argument("--serve-parity-json", type=Path, required=True)
    parser.add_argument("--direction-pocket-json", type=Path, required=True)
    parser.add_argument("--adaptation-lifecycle-json", type=Path, required=True)
    parser.add_argument("--live-tail-admission-json", type=Path, required=True)
    parser.add_argument("--launch-vedtak-json", type=Path, required=True)
    parser.add_argument("--transaction-id", required=True)
    parser.add_argument("--max-trades", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    vedtak_path = args.launch_vedtak_json.expanduser()
    if (
        not vedtak_path.is_absolute()
        or vedtak_path.is_symlink()
        or not vedtak_path.is_file()
        or vedtak_path.resolve().parent
        != CANONICAL_LAUNCH_VEDTAK_DIR.resolve()
    ):
        raise EntryLaunchFinalizationError(
            "launch vedtak must be an immutable event in the canonical "
            f"authority directory: {CANONICAL_LAUNCH_VEDTAK_DIR}"
        )
    event_path, event = finalize_entry_model_native_launch(
        accepted_bundle_dir=args.accepted_bundle_dir,
        sizing_adoption_path=args.sizing_adoption_json,
        joint_exit_proof_path=args.joint_exit_proof_json,
        sizing_runtime_parity_path=args.sizing_runtime_parity_json,
        serve_parity_path=args.serve_parity_json,
        direction_pocket_path=args.direction_pocket_json,
        adaptation_lifecycle_path=args.adaptation_lifecycle_json,
        live_tail_admission_path=args.live_tail_admission_json,
        launch_vedtak_path=vedtak_path.resolve(),
        transaction_id=args.transaction_id,
        max_trades=args.max_trades,
        artifact_registry_path=artifact_guard.SELECTION_CONTRACT.resolve(),
        launch_state_path=artifact_guard.XAU_DIRECTION_LAUNCH_CONTRACT.resolve(),
        approval_event_dir=CANONICAL_LAUNCH_APPROVAL_DIR,
        transaction_event_dir=CANONICAL_LAUNCH_TRANSACTION_DIR,
    )
    print(
        json.dumps(
            {
                "json_path": str(event_path),
                "sha256": sha256_file(event_path),
                "decision": event["decision"],
                "transaction_id": event["transaction_id"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
