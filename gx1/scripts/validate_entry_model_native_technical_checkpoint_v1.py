"""Run a sealed, TRAIN/VAL-only ten-task validation from a technical pair.

This is deliberately a preflight instrument, not a training, selection,
promotion, paper, live, or TEST route.  It consumes a *pair* of frozen model
and target-model states because the exact fitted-Q/Exit validation cannot be
truthfully reconstructed from an inference-only bundle by copying the online
weights into the target model.

The accepted checkpoint source is an existing attended technical session.  Its
authority remains non-candidate and non-promotable; this module merely proves
that the full VAL loader and all ten existing objective paths can execute from
that independently stored model/target pair.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from gx1.contracts.entry_model_native_post_rebuild_v1 import (
    PrefreezeTestSealLineageError,
    require_pretest_or_prefreeze_test_guard_lineage,
)
from gx1.contracts.gx1_capped_execution_v1 import (
    require_guarded_cuda_producer_execution,
)
from gx1.contracts.unified_exit_lifecycle_v1 import UnifiedExitLifecycleCorpus
from gx1.models.entry_v10.entry_v10_bundle import load_entry_v10_ctx_bundle
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    EntryV10CtxDataset,
    JOINT_TASK_NAMES,
    _ExactIndexSampler,
    _AttendedResearchSession,
    _model_forward_fp32,
    _multi_tf_kwargs_from_batch,
    _restore_candidate_validation_snapshot,
    _sha256_file,
    validate,
)


SCHEMA_VERSION = "gx1_technical_checkpoint_full_val_v1"
TEST_BOUNDARY_UTC = pd.Timestamp("2026-07-01T00:00:00Z")
_SHA256_LENGTH = 64
_REFERENCE_ROWS = 8
_SESSION_SCHEMA_VERSION = "gx1_technical_checkpoint_full_val_session_v1"
_SESSION_CONTRACT_FILENAME = "TECHNICAL_FULL_VAL_SESSION_CONTRACT.json"
_SESSION_ACTIVE_FILENAME = "TECHNICAL_FULL_VAL_SESSION_ACTIVE.json"
_SESSION_STATE_FILENAMES = (
    "technical_full_val_state_slot_0.pt",
    "technical_full_val_state_slot_1.pt",
)


class TechnicalValidationError(RuntimeError):
    """The sealed technical validation route cannot prove its inputs."""


def _regular_absolute(path: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or any(parent.is_symlink() for parent in candidate.parents)
        or not candidate.is_file()
    ):
        raise TechnicalValidationError(f"[{label}_PATH_INVALID]")
    return candidate


def _directory_absolute(path: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if (
        not candidate.is_absolute()
        or candidate.is_symlink()
        or any(parent.is_symlink() for parent in candidate.parents)
        or not candidate.is_dir()
    ):
        raise TechnicalValidationError(f"[{label}_PATH_INVALID]")
    return candidate


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    path = _regular_absolute(path, label=label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise TechnicalValidationError(f"[{label}_JSON_INVALID]") from exc
    if not isinstance(payload, dict):
        raise TechnicalValidationError(f"[{label}_JSON_INVALID]")
    return payload


def _sha256(value: Path) -> str:
    return _sha256_file(value)


def _sha256_is_valid(value: object) -> bool:
    return isinstance(value, str) and len(value) == _SHA256_LENGTH and all(
        character in "0123456789abcdef" for character in value
    )


def _exact_bound_path(
    binding: object,
    *,
    actual: Path,
    label: str,
) -> dict[str, str]:
    if not isinstance(binding, Mapping):
        raise TechnicalValidationError(f"[{label}_BINDING_INVALID]")
    path_text = binding.get("path")
    digest = binding.get("sha256")
    if (
        not isinstance(path_text, str)
        or Path(path_text) != actual
        or not _sha256_is_valid(digest)
        or _sha256(actual) != digest
    ):
        raise TechnicalValidationError(f"[{label}_BINDING_INVALID]")
    return {"path": path_text, "sha256": str(digest)}


def _attended_checkpoint_pair(
    *,
    session_dir: Path,
    bundle_metadata: Mapping[str, Any],
    train_parquet: Path,
    val_parquet: Path,
    m5_prebuilt: Path,
    lifecycle_manifest: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load the hash-bound model/target pair without granting promotion authority."""

    session_dir = _directory_absolute(session_dir, label="TECHNICAL_SESSION")
    contract_path = session_dir / "ATTENDED_RESEARCH_SESSION_CONTRACT.json"
    contract = _read_json(contract_path, label="TECHNICAL_SESSION_CONTRACT")
    authority = contract.get("authority")
    required_authority = {
        "bundle": False,
        "candidate": False,
        "live": False,
        "paper": False,
        "promotion": False,
        "research_trainability_only": True,
        "test": False,
        "validation": False,
    }
    if authority != required_authority:
        raise TechnicalValidationError("[TECHNICAL_SESSION_AUTHORITY_INVALID]")
    if contract.get("schema_version") != "gx1_attended_research_session_v1":
        raise TechnicalValidationError("[TECHNICAL_SESSION_SCHEMA_INVALID]")
    if contract.get("profile") != "smoke" or contract.get("execution_tier") not in {
        "attended_only",
        "attended_cpu_only",
    }:
        raise TechnicalValidationError("[TECHNICAL_SESSION_PROFILE_INVALID]")
    artifacts = contract.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise TechnicalValidationError("[TECHNICAL_SESSION_ARTIFACTS_INVALID]")
    for name, actual in (
        ("train_parquet", train_parquet),
        ("val_parquet", val_parquet),
        ("m5_prebuilt_path", m5_prebuilt),
        ("unified_exit_lifecycle_manifest", lifecycle_manifest),
    ):
        _exact_bound_path(artifacts.get(name), actual=actual, label=f"TECHNICAL_SESSION_{name.upper()}")
    normalization = bundle_metadata.get("input_normalization")
    if (
        not isinstance(normalization, Mapping)
        or contract.get("input_normalization_sha256")
        != str(normalization.get("contract_sha256") or "")
    ):
        raise TechnicalValidationError("[TECHNICAL_SESSION_NORMALIZATION_MISMATCH]")
    # This class is the source owner for pointer, slot, SHA and weights-only
    # state validation.  Existing session directories are opened read-only by
    # this code path; no state is written or advanced.
    try:
        session = _AttendedResearchSession(
            out_bundle_dir=Path(str(contract["out_bundle_dir"])),
            contract=contract,
        )
        state = session.load_checkpoint()
    except (KeyError, OSError, RuntimeError, ValueError) as exc:
        raise TechnicalValidationError("[TECHNICAL_SESSION_CHECKPOINT_INVALID]") from exc
    if state is None:
        raise TechnicalValidationError("[TECHNICAL_SESSION_CHECKPOINT_MISSING]")
    model_state = state.get("model_state")
    target_state = state.get("target_model_state")
    if not isinstance(model_state, Mapping) or not isinstance(target_state, Mapping):
        raise TechnicalValidationError("[TECHNICAL_SESSION_MODEL_PAIR_MISSING]")
    return contract, dict(model_state), dict(target_state)


def _require_no_test_rows(val_parquet: Path) -> dict[str, Any]:
    """Read only the explicitly supplied VAL clock, and reject TEST timestamps."""

    try:
        series = pd.read_parquet(val_parquet, columns=["time"])["time"]
    except Exception as exc:
        raise TechnicalValidationError("[TECHNICAL_VAL_CLOCK_READ_FAILED]") from exc
    timestamps = pd.to_datetime(series, utc=True)
    if len(timestamps) == 0:
        raise TechnicalValidationError("[TECHNICAL_VAL_EMPTY]")
    if bool((timestamps >= TEST_BOUNDARY_UTC).any()):
        raise TechnicalValidationError("[TECHNICAL_VAL_TEST_BOUNDARY_VIOLATION]")
    if bool(timestamps.duplicated().any()) or not bool(timestamps.is_monotonic_increasing):
        raise TechnicalValidationError("[TECHNICAL_VAL_CLOCK_INTEGRITY_INVALID]")
    return {
        "rows": int(len(timestamps)),
        "start_utc": str(timestamps.iloc[0]),
        "end_utc_inclusive": str(timestamps.iloc[-1]),
        "duplicate_timestamps": 0,
        "timestamps_at_or_after_test_boundary": 0,
    }


def _require_pretest_guard(
    *,
    guard_json: Path,
    guard_sha256: str,
    dataset_run_id: str,
    dataset_dir: Path,
) -> dict[str, Any]:
    """Validate the unopened-TEST seal with its mandatory dataset lineage."""

    try:
        observed = require_pretest_or_prefreeze_test_guard_lineage(
            guard_json,
            guard_sha256,
            expected_dataset_run_id=dataset_run_id,
            expected_dataset_dir=dataset_dir,
        )
    except (PrefreezeTestSealLineageError, OSError, ValueError) as exc:
        raise TechnicalValidationError("[TECHNICAL_TEST_GUARD_INVALID]") from exc
    if observed.get("test_accessed", None) is True or observed.get("access_proof", {}).get("test_dataset_bytes_read") is True:
        raise TechnicalValidationError("[TECHNICAL_TEST_GUARD_NOT_SEALED]")
    return observed


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _safe_json(value.item())
        return _safe_json(value.detach().cpu().tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() and path.is_symlink():
        raise TechnicalValidationError("[TECHNICAL_SESSION_JSON_PATH_INVALID]")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(_canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _model_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash a state dict unambiguously, including names, shapes and dtypes."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TechnicalValidationError("[TECHNICAL_MODEL_STATE_HASH_INVALID]")
        value = tensor.detach().cpu().contiguous()
        encoded_name = name.encode("utf-8")
        descriptor = json.dumps(
            {"dtype": str(value.dtype), "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(len(descriptor).to_bytes(4, "big"))
        digest.update(descriptor)
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


class _TechnicalValidationSession:
    """Two-slot, hash-bound progress for a bounded technical VAL run only."""

    def __init__(self, *, directory: Path, contract: Mapping[str, Any]) -> None:
        candidate = Path(directory).expanduser()
        if (
            not candidate.is_absolute()
            or candidate.is_symlink()
            or any(parent.is_symlink() for parent in candidate.parents)
            or not candidate.parent.is_dir()
        ):
            raise TechnicalValidationError("[TECHNICAL_SESSION_DIRECTORY_INVALID]")
        self.directory = candidate
        self.contract = dict(contract)
        self.contract_sha256 = hashlib.sha256(
            _canonical_json_bytes(self.contract)
        ).hexdigest()
        self.contract_path = self.directory / _SESSION_CONTRACT_FILENAME
        self.active_path = self.directory / _SESSION_ACTIVE_FILENAME
        if self.directory.exists():
            state = os.stat(self.directory, follow_symlinks=False)
            if (
                not self.directory.is_dir()
                or state.st_uid != os.getuid()
                or state.st_mode & 0o077
            ):
                raise TechnicalValidationError("[TECHNICAL_SESSION_DIRECTORY_INVALID]")
            on_disk = _read_json(self.contract_path, label="TECHNICAL_SESSION_CONTRACT")
            if on_disk != self.contract:
                raise TechnicalValidationError("[TECHNICAL_SESSION_CONTRACT_MISMATCH]")
        else:
            self.directory.mkdir(mode=0o700)
            fd = os.open(
                self.contract_path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(fd, "wb") as handle:
                handle.write(_canonical_json_bytes(self.contract))
                handle.flush()
                os.fsync(handle.fileno())

    def _slot_path(self, slot: int) -> Path:
        if slot not in (0, 1):
            raise TechnicalValidationError("[TECHNICAL_SESSION_SLOT_INVALID]")
        return self.directory / _SESSION_STATE_FILENAMES[slot]

    def load(self, *, expected_batches: int) -> dict[str, Any] | None:
        if self.active_path.is_symlink():
            raise TechnicalValidationError("[TECHNICAL_SESSION_ACTIVE_PATH_INVALID]")
        if not self.active_path.exists():
            if any(self._slot_path(slot).exists() for slot in (0, 1)):
                raise TechnicalValidationError("[TECHNICAL_SESSION_ACTIVE_MISSING]")
            return None
        active = _read_json(self.active_path, label="TECHNICAL_SESSION_ACTIVE")
        expected_active = {
            "schema_version",
            "session_contract_sha256",
            "slot",
            "checkpoint_index",
            "state_sha256",
            "next_batch_offset",
            "complete",
        }
        if (
            set(active) != expected_active
            or active.get("schema_version") != _SESSION_SCHEMA_VERSION
            or active.get("session_contract_sha256") != self.contract_sha256
            or active.get("slot") not in (0, 1)
            or not isinstance(active.get("checkpoint_index"), int)
            or int(active["checkpoint_index"]) < 1
            or not isinstance(active.get("next_batch_offset"), int)
            or not 0 <= int(active["next_batch_offset"]) <= int(expected_batches)
            or not _sha256_is_valid(active.get("state_sha256"))
            or not isinstance(active.get("complete"), bool)
        ):
            raise TechnicalValidationError("[TECHNICAL_SESSION_ACTIVE_INVALID]")
        state_path = self._slot_path(int(active["slot"]))
        if state_path.is_symlink() or not state_path.is_file() or _sha256(state_path) != active["state_sha256"]:
            raise TechnicalValidationError("[TECHNICAL_SESSION_STATE_SHA256_INVALID]")
        try:
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, ValueError) as exc:
            raise TechnicalValidationError("[TECHNICAL_SESSION_STATE_LOAD_INVALID]") from exc
        expected_state = {
            "schema_version",
            "session_contract_sha256",
            "checkpoint_index",
            "next_batch_offset",
            "validation_snapshot",
            "elapsed_seconds_completed",
            "complete",
        }
        if (
            not isinstance(state, Mapping)
            or set(state) != expected_state
            or state.get("schema_version") != _SESSION_SCHEMA_VERSION
            or state.get("session_contract_sha256") != self.contract_sha256
            or state.get("checkpoint_index") != active["checkpoint_index"]
            or state.get("next_batch_offset") != active["next_batch_offset"]
            or state.get("complete") != active["complete"]
            or not isinstance(state.get("elapsed_seconds_completed"), float)
            or not math.isfinite(float(state["elapsed_seconds_completed"]))
            or float(state["elapsed_seconds_completed"]) < 0.0
        ):
            raise TechnicalValidationError("[TECHNICAL_SESSION_STATE_INVALID]")
        snapshot = state.get("validation_snapshot")
        if bool(state["complete"]):
            if snapshot is not None or int(state["next_batch_offset"]) != int(expected_batches):
                raise TechnicalValidationError("[TECHNICAL_SESSION_COMPLETE_STATE_INVALID]")
        else:
            if not isinstance(snapshot, Mapping) or int(state["next_batch_offset"]) < 1:
                raise TechnicalValidationError("[TECHNICAL_SESSION_PROGRESS_STATE_INVALID]")
            try:
                _restore_candidate_validation_snapshot(snapshot)
            except RuntimeError as exc:
                raise TechnicalValidationError("[TECHNICAL_SESSION_PROGRESS_STATE_INVALID]") from exc
        return dict(state)

    def save(
        self,
        *,
        expected_batches: int,
        next_batch_offset: int,
        validation_snapshot: Mapping[str, Any] | None,
        elapsed_seconds_completed: float,
        complete: bool,
    ) -> None:
        if (
            not 1 <= int(next_batch_offset) <= int(expected_batches)
            or bool(complete) != (int(next_batch_offset) == int(expected_batches))
            or not math.isfinite(float(elapsed_seconds_completed))
            or float(elapsed_seconds_completed) < 0.0
        ):
            raise TechnicalValidationError("[TECHNICAL_SESSION_SAVE_ARGUMENT_INVALID]")
        if complete:
            if validation_snapshot is not None:
                raise TechnicalValidationError("[TECHNICAL_SESSION_COMPLETE_SNAPSHOT_INVALID]")
        else:
            if not isinstance(validation_snapshot, Mapping):
                raise TechnicalValidationError("[TECHNICAL_SESSION_SNAPSHOT_INVALID]")
            try:
                _restore_candidate_validation_snapshot(validation_snapshot)
            except RuntimeError as exc:
                raise TechnicalValidationError("[TECHNICAL_SESSION_SNAPSHOT_INVALID]") from exc
        prior = self.load(expected_batches=expected_batches)
        if prior is not None and bool(prior["complete"]):
            raise TechnicalValidationError("[TECHNICAL_SESSION_ALREADY_COMPLETE]")
        if prior is not None and int(next_batch_offset) <= int(prior["next_batch_offset"]):
            raise TechnicalValidationError("[TECHNICAL_SESSION_NONMONOTONIC_PROGRESS]")
        checkpoint_index = 1 if prior is None else int(prior["checkpoint_index"]) + 1
        prior_slot = -1 if prior is None else int(
            _read_json(self.active_path, label="TECHNICAL_SESSION_ACTIVE")["slot"]
        )
        slot = 0 if prior_slot != 0 else 1
        state_path = self._slot_path(slot)
        if state_path.is_symlink():
            raise TechnicalValidationError("[TECHNICAL_SESSION_STATE_PATH_INVALID]")
        state = {
            "schema_version": _SESSION_SCHEMA_VERSION,
            "session_contract_sha256": self.contract_sha256,
            "checkpoint_index": checkpoint_index,
            "next_batch_offset": int(next_batch_offset),
            "validation_snapshot": dict(validation_snapshot) if validation_snapshot is not None else None,
            "elapsed_seconds_completed": float(elapsed_seconds_completed),
            "complete": bool(complete),
        }
        fd, temporary = tempfile.mkstemp(prefix=f".{state_path.name}.", dir=str(self.directory))
        try:
            os.close(fd)
            torch.save(state, temporary)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, state_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        _atomic_json(
            self.active_path,
            {
                "schema_version": _SESSION_SCHEMA_VERSION,
                "session_contract_sha256": self.contract_sha256,
                "slot": slot,
                "checkpoint_index": checkpoint_index,
                "state_sha256": _sha256(state_path),
                "next_batch_offset": int(next_batch_offset),
                "complete": bool(complete),
            },
        )


def _task_summary(stats: Mapping[str, Any], model: torch.nn.Module) -> dict[str, Any]:
    loss_keys = {
        task: f"joint_task_raw_loss_mean_{task}" for task in JOINT_TASK_NAMES
    }
    diagnostics = stats.get("active_head_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, Mapping) else {}
    alias = {
        "position_size": "position_size",
        "dip_bps": "dip",
        "forecast_return_bps": "forecast",
        "dip_timing_fraction": "timing",
        "tail_risk_bps": "tail_risk",
        "forward_volatility_bps": "vol_forecast",
    }
    result: dict[str, Any] = {}
    task_variances = getattr(model, "task_log_variances", {})
    for task in JOINT_TASK_NAMES:
        variance = task_variances.get(task) if hasattr(task_variances, "get") else None
        log_variance = float(variance.detach().cpu().item()) if isinstance(variance, torch.Tensor) else None
        row: dict[str, Any] = {
            "task_name": task,
            "learned_log_variance": log_variance,
            "effective_weight": (math.exp(-log_variance) if log_variance is not None else None),
            "validation_loss": _safe_json(stats.get(loss_keys[task])),
            "supervised_cells": _safe_json(
                stats.get(f"joint_task_supervised_cells_{task}")
            ),
            "diagnostics": _safe_json(diagnostics.get(alias.get(task, task))),
        }
        if task == "unified_exit_action":
            row["sample_count"] = _safe_json(stats.get("unified_exit_q_valid_cells"))
            row["population_rows"] = _safe_json(stats.get("unified_exit_population_rows"))
            row["unique_target_action_agreement"] = _safe_json(
                stats.get("unified_exit_unique_target_action_agreement")
            )
        elif task == "entry_action_q":
            row["sample_count"] = _safe_json(stats.get("entry_unique_target_rows"))
            row["target_equivalent_rows"] = _safe_json(stats.get("entry_target_equivalent_rows"))
            row["unique_target_action_agreement"] = _safe_json(
                stats.get("entry_unique_target_action_agreement")
            )
        result[task] = row
    return result


def _reference_predictions(
    *,
    model: torch.nn.Module,
    dataset: EntryV10CtxDataset,
    device: torch.device,
    out_path: Path,
) -> dict[str, Any]:
    """Persist a small, deterministic full-VAL Entry reference batch."""

    positions = np.linspace(0, len(dataset) - 1, num=_REFERENCE_ROWS, dtype=np.int64)
    if len(set(int(item) for item in positions.tolist())) != _REFERENCE_ROWS:
        raise TechnicalValidationError("[TECHNICAL_REFERENCE_POSITIONS_INVALID]")
    from torch.utils.data._utils.collate import default_collate

    batch = default_collate([dataset[int(position)] for position in positions])
    with torch.no_grad():
        out = _model_forward_fp32(
            model,
            batch["seq_x"].to(device),
            batch["snap_x"].to(device),
            ctx_cat=batch["ctx_cat"].to(device),
            ctx_cont=batch["ctx_cont"].to(device),
            **_multi_tf_kwargs_from_batch(batch, device),
        )
    required = {
        "entry_action_q_bps",
        "side_mae_bps",
        "trendline_event_logits",
        "position_size_logit",
        "dip_pred",
        "forecast_pred",
        "timing_pred",
        "tail_risk_pred",
        "vol_forecast_pred",
    }
    missing = sorted(name for name in required if not isinstance(out.get(name), torch.Tensor))
    if missing:
        raise TechnicalValidationError(f"[TECHNICAL_REFERENCE_OUTPUT_MISSING] {missing}")
    values: dict[str, Any] = {
        "time": pd.to_datetime(
            [dataset.df.iloc[int(dataset.indices[position])]["time"] for position in positions],
            utc=True,
        ),
        "entry_row_index": [int(dataset.indices[position]) for position in positions],
    }
    for name in sorted(required):
        tensor = out[name].detach().cpu().float()
        if not bool(torch.isfinite(tensor).all().item()):
            raise TechnicalValidationError("[TECHNICAL_REFERENCE_OUTPUT_NONFINITE]")
        if tensor.ndim == 1:
            values[name] = tensor.numpy()
        elif tensor.ndim == 2:
            for column in range(tensor.shape[1]):
                values[f"{name}_{column}"] = tensor[:, column].numpy()
        else:
            raise TechnicalValidationError("[TECHNICAL_REFERENCE_OUTPUT_SHAPE_INVALID]")
    frame = pd.DataFrame(values)
    if out_path.exists() or out_path.is_symlink():
        raise TechnicalValidationError("[TECHNICAL_REFERENCE_OUTPUT_EXISTS]")
    frame.to_parquet(out_path, index=False)
    return {
        "path": str(out_path),
        "sha256": _sha256(out_path),
        "rows": int(len(frame)),
        "columns": [str(column) for column in frame.columns],
    }


def _git_identity(repo: Path) -> dict[str, Any]:
    def command(*parts: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *parts], text=True).strip()

    return {
        "commit": command("rev-parse", "HEAD"),
        "branch": command("branch", "--show-current"),
        "dirty_paths": [
            row for row in command("status", "--short").splitlines() if row
        ],
    }


def run(
    *,
    bundle_dir: Path,
    session_dir: Path,
    train_parquet: Path,
    val_parquet: Path,
    m5_prebuilt: Path,
    multi_tf_cache_dir: Path,
    lifecycle_manifest: Path,
    val_sequence_source_audit: Path,
    test_guard_json: Path,
    test_guard_sha256: str,
    device: str,
    batch_size: int,
    out_dir: Path,
    max_validation_batches: int | None,
    validation_session_dir: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    if device not in {"cpu", "cuda"} or int(batch_size) != 8:
        raise TechnicalValidationError("[TECHNICAL_VALIDATION_GEOMETRY_INVALID]")
    if max_validation_batches is not None and int(max_validation_batches) < 1:
        raise TechnicalValidationError("[TECHNICAL_VALIDATION_MAX_BATCHES_INVALID]")
    if validation_session_dir is not None and max_validation_batches is None:
        raise TechnicalValidationError("[TECHNICAL_SESSION_BOUNDED_RUN_REQUIRED]")
    repo = Path(__file__).resolve().parents[2]
    bundle_dir = _directory_absolute(bundle_dir, label="TECHNICAL_BUNDLE")
    train_parquet = _regular_absolute(train_parquet, label="TECHNICAL_TRAIN")
    val_parquet = _regular_absolute(val_parquet, label="TECHNICAL_VAL")
    m5_prebuilt = _regular_absolute(m5_prebuilt, label="TECHNICAL_M5")
    lifecycle_manifest = _regular_absolute(lifecycle_manifest, label="TECHNICAL_LIFECYCLE")
    val_sequence_source_audit = _regular_absolute(val_sequence_source_audit, label="TECHNICAL_VAL_SEQUENCE")
    test_guard_json = _regular_absolute(test_guard_json, label="TECHNICAL_TEST_GUARD")
    cache_dir = _directory_absolute(multi_tf_cache_dir, label="TECHNICAL_MTF_CACHE")
    cache_manifest = _regular_absolute(cache_dir / "manifest.json", label="TECHNICAL_MTF_MANIFEST")
    # The guard validator requires the immutable dataset lineage before it
    # permits even the supplied VAL route.  Reading this tiny session contract
    # is control-plane only; its complete cryptographic validation happens in
    # `_attended_checkpoint_pair` before any checkpoint state is loaded.
    session_preview = _read_json(
        _directory_absolute(session_dir, label="TECHNICAL_SESSION")
        / "ATTENDED_RESEARCH_SESSION_CONTRACT.json",
        label="TECHNICAL_SESSION_CONTRACT",
    )
    dataset_run_id_preview = session_preview.get("dataset_run_id")
    if not isinstance(dataset_run_id_preview, str) or not dataset_run_id_preview:
        raise TechnicalValidationError("[TECHNICAL_SESSION_DATASET_RUN_ID_INVALID]")
    val_clock = _require_no_test_rows(val_parquet)
    guard = _require_pretest_guard(
        guard_json=test_guard_json,
        guard_sha256=test_guard_sha256,
        dataset_run_id=dataset_run_id_preview,
        dataset_dir=val_parquet.parent,
    )
    if dry_run:
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "PASS",
            "mode": "dry_run",
            "test_accessed": False,
            "val_clock": val_clock,
            "input_paths": {
                "bundle_dir": str(bundle_dir),
                "session_dir": str(session_dir),
                "train_parquet": str(train_parquet),
                "val_parquet": str(val_parquet),
                "m5_prebuilt": str(m5_prebuilt),
                "multi_tf_cache_manifest": str(cache_manifest),
                "lifecycle_manifest": str(lifecycle_manifest),
                "val_sequence_source_audit": str(val_sequence_source_audit),
            },
        }
    if device == "cuda":
        require_guarded_cuda_producer_execution()
    if out_dir.exists() or out_dir.is_symlink() or not out_dir.is_absolute() or not out_dir.parent.is_dir():
        raise TechnicalValidationError("[TECHNICAL_OUTPUT_DIR_INVALID]")
    bundle = load_entry_v10_ctx_bundle(bundle_dir=bundle_dir, device="cpu")
    metadata = bundle.metadata
    contract, model_state, target_state = _attended_checkpoint_pair(
        session_dir=session_dir,
        bundle_metadata=metadata,
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        m5_prebuilt=m5_prebuilt,
        lifecycle_manifest=lifecycle_manifest,
    )
    session_commit = str(contract.get("source_commit") or "")
    bundle_commit = str(metadata.get("git_commit") or "")
    if len(session_commit) != 40 or session_commit != bundle_commit:
        raise TechnicalValidationError("[TECHNICAL_CHECKPOINT_SOURCE_COMMIT_MISMATCH]")
    model = bundle.transformer_model
    try:
        model.load_state_dict(model_state, strict=True)
        target_model = copy.deepcopy(model)
        target_model.load_state_dict(target_state, strict=True)
    except RuntimeError as exc:
        raise TechnicalValidationError("[TECHNICAL_CHECKPOINT_STATE_MODEL_MISMATCH]") from exc
    target_model.requires_grad_(False)
    model.eval()
    target_model.eval()
    runtime_device = torch.device(device)
    if runtime_device.type == "cuda":
        if not torch.cuda.is_available():
            raise TechnicalValidationError("[TECHNICAL_CUDA_UNAVAILABLE]")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(runtime_device)
    model = model.to(runtime_device)
    target_model = target_model.to(runtime_device)
    os.environ["GX1_V10_MULTI_TF_V4_CACHE_DIR"] = str(cache_dir)
    mtf = metadata.get("multi_tf")
    if not isinstance(mtf, Mapping):
        raise TechnicalValidationError("[TECHNICAL_BUNDLE_MTF_INVALID]")
    per_tf = {
        "M5": int(mtf["m5_seq_len"]),
        "M15": int(mtf["m15_seq_len"]),
        "H1": int(mtf["h1_seq_len"]),
        "H4": int(mtf["h4_seq_len"]),
        "D1": int(mtf["d1_seq_len"]),
    }
    dataset_run_id = str(contract.get("dataset_run_id") or "")
    corpus = UnifiedExitLifecycleCorpus(
        root_manifest_path=lifecycle_manifest,
        entry_parquets={"val": val_parquet},
        dataset_run_id=dataset_run_id,
        splits=("val",),
    )
    val_dataset = EntryV10CtxDataset(
        val_parquet,
        seq_len=int(metadata["seq_len"]),
        m5_prebuilt_path=m5_prebuilt,
        per_tf_seq_lens=per_tf,
        multi_tf_closed_bar=True,
        sequence_source_audit_json=val_sequence_source_audit,
    )
    val_dataset.bind_unified_exit_lifecycle(corpus.splits["val"])
    if int(len(val_dataset)) != int(val_clock["rows"]):
        raise TechnicalValidationError("[TECHNICAL_VAL_SAMPLE_COUNT_MISMATCH]")
    expected_batches = -(-len(val_dataset) // int(batch_size))
    resumed_state: Mapping[str, Any] | None = None
    next_batch_offset = 0
    elapsed_before = 0.0
    validation_session: _TechnicalValidationSession | None = None
    if validation_session_dir is not None:
        identity = _git_identity(repo)
        if identity["dirty_paths"]:
            raise TechnicalValidationError("[TECHNICAL_SESSION_SOURCE_DIRTY]")
        validation_session = _TechnicalValidationSession(
            directory=validation_session_dir,
            contract={
                "schema_version": _SESSION_SCHEMA_VERSION,
                "authority": {
                    "technical_preflight": True,
                    "candidate": False,
                    "test": False,
                    "promotion": False,
                    "paper": False,
                    "live": False,
                },
                "source_commit": identity["commit"],
                "checkpoint": {
                    "source_commit": session_commit,
                    "session_contract_sha256": _sha256(
                        session_dir / "ATTENDED_RESEARCH_SESSION_CONTRACT.json"
                    ),
                    "online_model_state_sha256": _model_state_sha256(model_state),
                    "target_model_state_sha256": _model_state_sha256(target_state),
                    "bundle_metadata_sha256": _sha256(bundle_dir / "bundle_metadata.json"),
                },
                "inputs": {
                    "train_parquet": {"path": str(train_parquet), "sha256": _sha256(train_parquet)},
                    "val_parquet": {"path": str(val_parquet), "sha256": _sha256(val_parquet)},
                    "m5_prebuilt": {"path": str(m5_prebuilt), "sha256": _sha256(m5_prebuilt)},
                    "multi_tf_cache_manifest": {"path": str(cache_manifest), "sha256": _sha256(cache_manifest)},
                    "lifecycle_manifest": {"path": str(lifecycle_manifest), "sha256": _sha256(lifecycle_manifest)},
                    "val_sequence_source_audit": {"path": str(val_sequence_source_audit), "sha256": _sha256(val_sequence_source_audit)},
                    "test_guard": {"path": str(test_guard_json), "sha256": str(test_guard_sha256)},
                },
                "val_clock": val_clock,
                "geometry": {
                    "batch_size": int(batch_size),
                    "expected_batches": int(expected_batches),
                },
            },
        )
        existing = validation_session.load(expected_batches=expected_batches)
        if existing is not None:
            if bool(existing["complete"]):
                raise TechnicalValidationError("[TECHNICAL_SESSION_ALREADY_COMPLETE]")
            next_batch_offset = int(existing["next_batch_offset"])
            elapsed_before = float(existing["elapsed_seconds_completed"])
            snapshot_value = existing["validation_snapshot"]
            if not isinstance(snapshot_value, Mapping):  # guarded by `load`
                raise TechnicalValidationError("[TECHNICAL_SESSION_PROGRESS_STATE_INVALID]")
            resumed_state = dict(snapshot_value)
        loader = DataLoader(
            val_dataset,
            batch_size=int(batch_size),
            sampler=_ExactIndexSampler(
                torch.arange(len(val_dataset), dtype=torch.int64),
                batch_offset=next_batch_offset,
                batch_size=int(batch_size),
            ),
            num_workers=0,
            pin_memory=False,
        )
        if len(loader) != int(expected_batches) - int(next_batch_offset):
            raise TechnicalValidationError("[TECHNICAL_SESSION_REMAINING_LOADER_INVALID]")
    else:
        loader = DataLoader(
            val_dataset,
            batch_size=int(batch_size),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
    started = time.monotonic()
    snapshot: dict[str, Any] | None = None

    def on_partial(*, next_batch_offset: int, validation_snapshot: Mapping[str, Any]) -> None:
        nonlocal snapshot
        if validation_session is not None:
            validation_session.save(
                expected_batches=expected_batches,
                next_batch_offset=int(next_batch_offset),
                validation_snapshot=validation_snapshot,
                elapsed_seconds_completed=elapsed_before + (time.monotonic() - started),
                complete=False,
            )
        snapshot = {
            "next_batch_offset": int(next_batch_offset),
            "validation_snapshot_keys": sorted(str(key) for key in validation_snapshot),
        }

    validation_loss, _auc, accuracy, _short_to_long, stats = validate(
        model,
        target_model,
        loader,
        runtime_device,
        collect_full_exit_trajectory=True,
        resume_validation_state=resumed_state,
        validation_batch_offset=next_batch_offset,
        max_validation_batches=max_validation_batches,
        validation_checkpoint_hook=(on_partial if max_validation_batches is not None else None),
        validation_session_log_label="TECHNICAL_PRELIGHT",
    )
    elapsed = time.monotonic() - started
    partial = bool(stats.get("candidate_session_partial", False))
    if partial:
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "PARTIAL",
            "test_accessed": False,
            "val_clock": val_clock,
            "completed_batches": snapshot,
            "validation_session": (
                {
                    "path": str(validation_session.directory),
                    "contract_sha256": validation_session.contract_sha256,
                }
                if validation_session is not None
                else None
            ),
            "elapsed_seconds": elapsed,
            "elapsed_seconds_cumulative": elapsed_before + elapsed,
            "authority": {
                "technical_preflight": True,
                "candidate": False,
                "test": False,
                "promotion": False,
                "paper": False,
                "live": False,
            },
        }
    if not math.isfinite(float(validation_loss)) or not math.isfinite(float(accuracy)):
        raise TechnicalValidationError("[TECHNICAL_VALIDATION_NONFINITE]")
    out_dir.mkdir(mode=0o700)
    reference = _reference_predictions(
        model=model,
        dataset=val_dataset,
        device=runtime_device,
        out_path=out_dir / "full_val_predictions_reference.parquet",
    )
    peak_gpu: dict[str, int] | None = None
    if runtime_device.type == "cuda":
        peak_gpu = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(runtime_device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(runtime_device)),
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "decision": "PASS",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_accessed": False,
        "test_accessed_confirmation": "NO",
        "authority": {
            "technical_preflight": True,
            "technical_full_val": True,
            "candidate": False,
            "test": False,
            "promotion": False,
            "paper": False,
            "live": False,
        },
        "git": _git_identity(repo),
        "checkpoint": {
            "source": "attended_technical_model_target_pair_only",
            "source_commit": session_commit,
            "session_contract_sha256": _sha256(session_dir / "ATTENDED_RESEARCH_SESSION_CONTRACT.json"),
            "online_model_state_sha256": _model_state_sha256(model_state),
            "target_model_state_sha256": _model_state_sha256(target_state),
            "bundle_metadata_sha256": _sha256(bundle_dir / "bundle_metadata.json"),
        },
        "inputs": {
            "train_parquet": {"path": str(train_parquet), "sha256": _sha256(train_parquet)},
            "val_parquet": {"path": str(val_parquet), "sha256": _sha256(val_parquet)},
            "m5_prebuilt": {"path": str(m5_prebuilt), "sha256": _sha256(m5_prebuilt)},
            "multi_tf_cache_manifest": {"path": str(cache_manifest), "sha256": _sha256(cache_manifest)},
            "lifecycle_manifest": {"path": str(lifecycle_manifest), "sha256": _sha256(lifecycle_manifest)},
            "val_sequence_source_audit": {"path": str(val_sequence_source_audit), "sha256": _sha256(val_sequence_source_audit)},
            "test_guard": {"path": str(test_guard_json), "sha256": str(test_guard_sha256)},
        },
        "val_clock": val_clock,
        "validation": {
            "rows": int(len(val_dataset)),
            "batches": int(expected_batches),
            "batches_executed_this_invocation": int(len(loader)),
            "batch_size": int(batch_size),
            "loss": float(validation_loss),
            "entry_unique_target_action_agreement": float(accuracy),
            "all_ten_tasks": _task_summary(stats, model),
            "stats": _safe_json(stats),
        },
        "reference_predictions": reference,
        "resources": {
            "elapsed_seconds": elapsed,
            "elapsed_seconds_cumulative": elapsed_before + elapsed,
            "samples_per_second": (
                float(len(val_dataset) / (elapsed_before + elapsed))
                if elapsed_before + elapsed > 0.0
                else None
            ),
            "batches_per_second": (
                float(expected_batches / (elapsed_before + elapsed))
                if elapsed_before + elapsed > 0.0
                else None
            ),
            "batches_executed_this_invocation": int(len(loader)),
            "peak_gpu": peak_gpu,
            "peak_process_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
    }
    payload = json.dumps(_safe_json(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    report_path = out_dir / "full_val_metrics.json"
    descriptor = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if validation_session is not None:
        validation_session.save(
            expected_batches=expected_batches,
            next_batch_offset=expected_batches,
            validation_snapshot=None,
            elapsed_seconds_completed=elapsed_before + elapsed,
            complete=True,
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-session-dir", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--val-parquet", type=Path, required=True)
    parser.add_argument("--m5-prebuilt", type=Path, required=True)
    parser.add_argument("--multi-tf-cache-dir", type=Path, required=True)
    parser.add_argument("--lifecycle-manifest", type=Path, required=True)
    parser.add_argument("--val-sequence-source-audit", type=Path, required=True)
    parser.add_argument("--test-guard-json", type=Path, required=True)
    parser.add_argument("--test-guard-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--validation-session-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    report = run(
        bundle_dir=args.bundle_dir,
        session_dir=args.checkpoint_session_dir,
        train_parquet=args.train_parquet,
        val_parquet=args.val_parquet,
        m5_prebuilt=args.m5_prebuilt,
        multi_tf_cache_dir=args.multi_tf_cache_dir,
        lifecycle_manifest=args.lifecycle_manifest,
        val_sequence_source_audit=args.val_sequence_source_audit,
        test_guard_json=args.test_guard_json,
        test_guard_sha256=str(args.test_guard_sha256),
        device=str(args.device),
        batch_size=int(args.batch_size),
        out_dir=args.out_dir,
        max_validation_batches=args.max_validation_batches,
        validation_session_dir=args.validation_session_dir,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(_safe_json(report), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI route
    raise SystemExit(main())
