"""Canonical fixed-step TRAIN-only random-access Exit v2 smoke executor."""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.unified_exit_pilot_final_bindings_v1 import (
    require_composite_normalization_binding,
)
from gx1.contracts.unified_exit_random_access_checkpoint_v1 import (
    build_checkpoint,
    load_checkpoint_strict,
    save_checkpoint_atomic,
)
from gx1.contracts.unified_exit_random_access_cuda_smoke_v1 import (
    require_bootstrap_composite_normalization,
    require_smoke_manifest,
)
from gx1.contracts.unified_exit_random_access_model_v1 import (
    bind_preserved_v7_input_normalization,
    bootstrap_random_access_v2_from_pretrained,
)
from gx1.contracts.unified_exit_random_access_train_factory_v1 import (
    build_random_access_train_adapter_factory_v1,
)
from gx1.contracts.unified_exit_selected_sampler_v1 import (
    file_sha256,
    require_selected_sampler_artifact,
)
from gx1.contracts.unified_exit_gpu_batch_selection_v1 import require_selection
from gx1.contracts.unified_exit_train_session_manifest_v1 import (
    require_train_session_manifest,
)
from gx1.contracts.unified_exit_lifecycle_v1 import UnifiedExitLifecycleCorpus
from gx1.features.entry_specialist_feature_groups_v1 import (
    require_multi_tf_specialist_routing_v4,
)
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EntryV10CtxHybridTransformer,
)
from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256 as campaign_sha256
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    EntryV10CtxDataset,
    JOINT_TASK_NAMES,
    _announce_attended_preflight_ready,
    _set_deterministic,
    train_epoch,
)

LAUNCH_SCHEMA = "gx1_unified_exit_random_access_cuda_smoke_launch_v2"


def _read(path: Path) -> dict[str, Any]:
    if (
        not path.is_absolute()
        or path.resolve() != path
        or not path.is_file()
        or path.is_symlink()
    ):
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_PATH_INVALID")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_JSON_INVALID")
    return value


def _bind_multi_tf_cache_from_source_bundle_metadata(
    metadata: Mapping[str, Any],
) -> Path:
    """Bind the exact source-owned V4 MTF cache before dataset construction."""

    multi_tf = metadata.get("multi_tf")
    if not isinstance(multi_tf, Mapping):
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_MULTI_TF_CACHE_BINDING_MISSING")
    cache_dir_raw = multi_tf.get("shared_cache_dir")
    manifest_path_raw = multi_tf.get("shared_cache_manifest_path")
    manifest_sha256 = multi_tf.get("shared_cache_manifest_sha256")
    cache_identity_sha256 = multi_tf.get("shared_cache_identity_sha256")
    if not all(
        isinstance(value, str) and value
        for value in (
            cache_dir_raw,
            manifest_path_raw,
            manifest_sha256,
            cache_identity_sha256,
        )
    ):
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_MULTI_TF_CACHE_BINDING_MISSING")
    cache_dir = Path(cache_dir_raw)
    manifest_path = Path(manifest_path_raw)
    if (
        not cache_dir.is_absolute()
        or cache_dir.resolve() != cache_dir
        or not cache_dir.is_dir()
        or cache_dir.is_symlink()
        or not manifest_path.is_absolute()
        or manifest_path.resolve() != manifest_path
        or not manifest_path.is_file()
        or manifest_path.is_symlink()
        or manifest_path != cache_dir / "manifest.json"
    ):
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_MULTI_TF_CACHE_PATH_INVALID")
    if (
        len(manifest_sha256) != 64
        or file_sha256(manifest_path) != manifest_sha256
    ):
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_MULTI_TF_CACHE_MANIFEST_MISMATCH")
    manifest = _read(manifest_path)
    if manifest.get("cache_identity_sha256") != cache_identity_sha256:
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_MULTI_TF_CACHE_IDENTITY_MISMATCH")
    os.environ["GX1_V10_MULTI_TF_V4_CACHE_DIR"] = str(cache_dir)
    return cache_dir


def _canonical(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def require_launch_manifest(
    value: Mapping[str, Any], *, verify_files: bool = True
) -> dict[str, Any]:
    data = dict(value)
    claimed = data.pop("manifest_sha256", None)
    if (
        value.get("schema_version") != LAUNCH_SCHEMA
        or value.get("decision") != "PASS_GPU_SMOKE_MATRIX_ELIGIBLE"
        or claimed != _canonical(data)
        or value.get("precision_policy") != "deterministic_fp32"
        or value.get("batch_sizes") != [4, 8, 16]
        or value.get("selected_batch_size") is not None
        or value.get("gpu_batch_selection_status") != "PENDING_MEASURED_CUDA_MATRIX"
        or value.get("gpu_batch_selection_metric")
        != "highest_guarded_measured_optimizer_transitions_per_second_then_lower_batch_tie_break"
        or value.get("warmup_optimizer_steps_per_arm") != 1
        or value.get("measured_optimizer_steps_per_arm") != 2
        or value.get("bootstrap_optimizer_steps") != 3
        or value.get("resume_probe_optimizer_steps") != 1
        or value.get("optimizer_state_origin") != "fresh_random_access_v2"
        or value.get("scheduler_state_origin") != "fresh_cosine_epoch_boundary_tmax30"
        or value.get("weight_ema_state_origin") != "fresh_train_only_epoch_horizon_v1"
        or value.get("target_update_policy")
        != "frozen_compatible_v1_target_for_epoch_zero"
        or value.get("weight_ema_decay") != 1.0 - (1.0 / 1024.0)
        or value.get("test_data_used") is not False
        or value.get("safety")
        != {
            "physical_power_limit_w": 160,
            "maximum_actual_draw_w": 170,
            "maximum_core_temperature_c": 65,
            "maximum_memory_junction_temperature_c": 80,
            "maximum_vram_mib": 12288,
            "telemetry_interval_seconds": 1,
        }
    ):
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_LAUNCH_INVALID")
    required = {
        "blocked_smoke_manifest",
        "selected_sampler",
        "random_access_root",
        "train_random_access_index",
        "child_composite_normalization",
        "bootstrap_composite_normalization",
        "economics_readiness",
        "train_cost_authority",
        "source_bundle_metadata",
        "checkpoint_pointer",
        "feature_lifecycle_root",
        "entry_train_parquet",
        "entry_train_manifest",
        "entry_val_parquet",
        "entry_val_manifest",
        "m5_prebuilt",
        "sequence_source_audit",
        "state_view_source",
        "fixed_step_executor_source",
        "capped_runner",
    }
    if set(value.get("files", {})) != required:
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_FILE_SET_INVALID")
    if verify_files:
        for binding in value["files"].values():
            p = Path(str(binding.get("path", "")))
            if (
                not p.is_absolute()
                or not p.is_file()
                or p.is_symlink()
                or file_sha256(p) != binding.get("sha256")
            ):
                raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_FILE_INVALID")
    selected = require_selected_sampler_artifact(
        _read(Path(value["files"]["selected_sampler"]["path"])),
        verify_files=verify_files,
    )
    blocked = require_smoke_manifest(
        _read(Path(value["files"]["blocked_smoke_manifest"]["path"])),
        verify_files=verify_files,
    )
    expected_root = selected["random_access_root"]
    if (
        value["files"]["random_access_root"]
        != {
            "path": expected_root["path"],
            "sha256": expected_root["file_sha256"],
        }
        or blocked["benchmark_matrix"]["batch_sizes"] != value["batch_sizes"]
    ):
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_SELECTED_ROOT_INVALID")
    source_repo = Path(str(value["source_repo"]))
    launch_path = (
        Path(str(value["files"]["fixed_step_launch_manifest"]["path"]))
        if "fixed_step_launch_manifest" in value["files"]
        else None
    )
    # The launch manifest cannot bind its own bytes. Its path is reconstructed
    # from every emitted command and all command tokens must agree exactly.
    commands = value.get("launcher_commands")
    if not isinstance(commands, Mapping) or launch_path is not None:
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_COMMANDS_INVALID")
    arm_commands = commands.get("arms")
    if not isinstance(arm_commands, Mapping):
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_COMMANDS_INVALID")
    observed_launch_paths = set()
    for batch_size in value["batch_sizes"]:
        command = arm_commands.get(str(batch_size))
        if not isinstance(command, list):
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_COMMANDS_INVALID")
        try:
            observed_launch_paths.add(command[command.index("--launch-manifest") + 1])
        except (ValueError, IndexError):
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_COMMANDS_INVALID") from None
    if len(observed_launch_paths) != 1:
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_COMMANDS_INVALID")
    launch_manifest_path = Path(observed_launch_paths.pop())
    checkpoint_dir = Path(str(value["checkpoint_dir"]))
    expected_commands = {
        "arms": {
            str(batch_size): _launch_command(
                source_repo=source_repo,
                launch_manifest_path=launch_manifest_path,
                checkpoint_dir=checkpoint_dir,
                batch_size=batch_size,
            )
            for batch_size in value["batch_sizes"]
        },
    }
    if commands != expected_commands:
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_COMMANDS_INVALID")
    return dict(value)


def _launch_command(
    *,
    source_repo: Path,
    launch_manifest_path: Path,
    checkpoint_dir: Path,
    batch_size: int,
) -> list[str]:
    return [
        str(source_repo / "scripts/gx1_capped_run.sh"),
        "--class",
        "trainer",
        "--attended-smoke",
        "--",
        str(source_repo / ".venv/bin/python"),
        "-m",
        "gx1.scripts.run_unified_exit_random_access_fixed_step_v1",
        "--launch-manifest",
        str(launch_manifest_path),
        "--stage",
        "smoke-arm",
        "--checkpoint-dir",
        str(checkpoint_dir / f"batch_{batch_size}"),
        "--arm-batch-size",
        str(batch_size),
        "--progress-path",
        str(checkpoint_dir / f"batch_{batch_size}" / "PROGRESS.json"),
        "--device",
        "cuda",
    ]


def build_launch_manifest(
    *,
    source_repo: Path,
    source_commit: str,
    files: Mapping[str, Mapping[str, str]],
    launch_manifest_path: Path,
    checkpoint_dir: Path,
    seed: int = 20260911,
) -> dict[str, Any]:
    blocked = _read(Path(files["blocked_smoke_manifest"]["path"]))
    selected = require_selected_sampler_artifact(
        _read(Path(files["selected_sampler"]["path"])), verify_files=True
    )
    root_binding = selected["random_access_root"]
    if files["random_access_root"] != {
        "path": root_binding["path"],
        "sha256": root_binding["file_sha256"],
    } or blocked["benchmark_matrix"]["batch_sizes"] != [4, 8, 16]:
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_SELECTED_ROOT_INVALID")
    batch_sizes = [4, 8, 16]
    value = {
        "schema_version": LAUNCH_SCHEMA,
        "decision": "PASS_GPU_SMOKE_MATRIX_ELIGIBLE",
        "source_repo": str(source_repo),
        "source_commit": source_commit,
        "files": {name: dict(binding) for name, binding in files.items()},
        "seed": int(seed),
        "dataset_run_id": "PRETEST_V3_20260829T173000Z",
        "precision_policy": "deterministic_fp32",
        "batch_sizes": batch_sizes,
        "selected_batch_size": None,
        "gpu_batch_selection_status": "PENDING_MEASURED_CUDA_MATRIX",
        "gpu_batch_selection_metric": "highest_guarded_measured_optimizer_transitions_per_second_then_lower_batch_tie_break",
        "warmup_optimizer_steps_per_arm": 1,
        "measured_optimizer_steps_per_arm": 2,
        "bootstrap_optimizer_steps": 3,
        "resume_probe_optimizer_steps": 1,
        "optimizer_state_origin": "fresh_random_access_v2",
        "scheduler_state_origin": "fresh_cosine_epoch_boundary_tmax30",
        "weight_ema_state_origin": "fresh_train_only_epoch_horizon_v1",
        "target_update_policy": "frozen_compatible_v1_target_for_epoch_zero",
        "weight_ema_decay": 1.0 - (1.0 / 1024.0),
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "safety": {
            "physical_power_limit_w": 160,
            "maximum_actual_draw_w": 170,
            "maximum_core_temperature_c": 65,
            "maximum_memory_junction_temperature_c": 80,
            "maximum_vram_mib": 12288,
            "telemetry_interval_seconds": 1,
        },
        "telemetry_metrics": [
            "loader_materialize_seconds",
            "h2d_seconds",
            "target_forward_seconds",
            "online_forward_seconds",
            "backward_seconds",
            "optimizer_seconds",
            "vram_mib",
            "gpu_utilization_percent",
            "power_draw_w",
            "core_temperature_c",
            "memory_junction_temperature_c",
        ],
        "checkpoint_dir": str(checkpoint_dir),
        "launcher_commands": {
            "arms": {
                str(batch_size): _launch_command(
                    source_repo=source_repo,
                    launch_manifest_path=launch_manifest_path,
                    checkpoint_dir=checkpoint_dir,
                    batch_size=batch_size,
                )
                for batch_size in batch_sizes
            },
        },
        "bootstrap_source_receipt_sha256": _read(
            Path(files["blocked_smoke_manifest"]["path"]).parent
            / "BOOTSTRAP_SOURCE_RECEIPT.json"
        )["receipt_sha256"],
        "test_data_used": False,
    }
    value["manifest_sha256"] = _canonical(value)
    require_launch_manifest(value)
    return value


class _FreshWeightEma:
    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        if not 0.0 < self.decay < 1.0:
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_EMA_DECAY_INVALID")
        self.parameter_names = frozenset(name for name, _ in model.named_parameters())
        self.shadow = {
            name: tensor.detach().clone() for name, tensor in model.state_dict().items()
        }
        self.steps = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        current = model.state_dict()
        if set(current) != set(self.shadow):
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_EMA_KEYS_INVALID")
        for name, tensor in current.items():
            if name in self.parameter_names:
                self.shadow[name].mul_(self.decay).add_(
                    tensor.detach(), alpha=1.0 - self.decay
                )
            else:
                self.shadow[name].copy_(tensor.detach())
        self.steps += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "steps": self.steps,
            "parameter_names": sorted(self.parameter_names),
            "shadow": {
                name: tensor.detach().clone() for name, tensor in self.shadow.items()
            },
        }

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if (
            float(value.get("decay", -1.0)) != self.decay
            or set(value.get("parameter_names", ())) != self.parameter_names
            or set(value.get("shadow", {})) != set(self.shadow)
            or isinstance(value.get("steps"), bool)
            or int(value.get("steps", -1)) < 0
        ):
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_EMA_STATE_INVALID")
        for name, tensor in value["shadow"].items():
            if (
                tensor.shape != self.shadow[name].shape
                or tensor.dtype != self.shadow[name].dtype
            ):
                raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_EMA_STATE_INVALID")
            self.shadow[name].copy_(tensor)
        self.steps = int(value["steps"])


def _schedule_witness(
    *,
    child_order: Sequence[int],
    parent_order: Sequence[int],
    batch_size: int,
    epoch_schedule_sha256: str,
    selected_sampler_artifact_sha256: str,
    full_population_schedule: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_count = (
        16384 if full_population_schedule is None
        else int(full_population_schedule["entry_pair_count"])
    )
    if len(child_order) != expected_count or len(parent_order) != len(child_order):
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_SCHEDULE_SIZE_INVALID")
    next_start = 4 * batch_size
    next_stop = next_start + batch_size
    next_child = [int(value) for value in child_order[next_start:next_stop]]
    next_parent = [int(value) for value in parent_order[next_start:next_stop]]
    if len(next_child) != batch_size or len(next_parent) != batch_size:
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_NEXT_BATCH_INVALID")
    next_identity = {
        "batch_offset": 4,
        "batch_size": batch_size,
        "child_entry_row_indices": next_child,
        "parent_entry_row_indices": next_parent,
    }
    next_identity["identity_sha256"] = _canonical(next_identity)
    value = {
        "schema_version": "gx1_unified_exit_epoch_schedule_witness_v1",
        "epoch_index": 0,
        "batch_size": batch_size,
        "entry_pair_count": len(child_order),
        "transition_count": 4 * len(child_order),
        "selected_sampler_artifact_sha256": selected_sampler_artifact_sha256,
        "epoch_schedule_sha256": epoch_schedule_sha256,
        "child_order_sha256": _canonical([int(value) for value in child_order]),
        "parent_order_sha256": _canonical([int(value) for value in parent_order]),
        "next_batch_after_optimizer_step_4": next_identity,
        "test_data_used": False,
    }
    if full_population_schedule is not None:
        if (
            full_population_schedule["entry_order_sha256"]
            != _canonical([int(item) for item in child_order])
            or full_population_schedule["every_entry_pair_exactly_once"] is not True
        ):
            raise RuntimeError("UNIFIED_EXIT_FULL_YEAR_SCHEDULE_INVALID")
        value["schema_version"] = "gx1_unified_exit_full_population_schedule_witness_v1"
        value["full_population_schedule"] = dict(full_population_schedule)
    value["witness_sha256"] = _canonical(value)
    return value


def _write_or_require_schedule_witness(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if path.is_symlink() or _read(path) != dict(value):
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_SCHEDULE_WITNESS_MISMATCH")
        return
    _write_progress_atomic(path, value)


def _write_progress_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _write_campaign_progress(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    """Publish with the receiving campaign owner's canonical identity."""
    progress = dict(value)
    progress["progress_sha256"] = campaign_sha256(progress)
    _write_progress_atomic(path, progress)
    return progress


def _absolute_optimizer_step(
    *, initial_global_step: int, start_batch_offset: int, next_batch_offset: int
) -> int:
    completed = next_batch_offset - start_batch_offset
    if completed < 1 or initial_global_step < 0:
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_CHECKPOINT_CURSOR_INVALID")
    return initial_global_step + completed


class _ParentSampler(Sampler[int]):
    def __init__(self, rows: Sequence[int], batch_offset: int, batch_size: int):
        self.rows = tuple(int(x) for x in rows)
        self.batch_offset = int(batch_offset)
        self.batch_size = int(batch_size)
        if self.batch_offset < 0 or self.batch_size not in (4, 8, 16):
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_SAMPLER_CURSOR_INVALID")
        self.row_offset = self.batch_offset * self.batch_size
        if self.row_offset > len(self.rows):
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_SAMPLER_CURSOR_INVALID")

    def __iter__(self):
        return iter(self.rows[self.row_offset :])

    def __len__(self):
        return len(self.rows) - self.row_offset


def _model(
    meta: Mapping[str, Any], normalization: Mapping[str, Any], device: torch.device
) -> EntryV10CtxHybridTransformer:
    mtf = meta["multi_tf"]
    specialist = meta["specialist_fusion"]
    indices = specialist["input_indices"]
    return EntryV10CtxHybridTransformer(
        seq_input_dim=int(meta["seq_input_dim"]),
        snap_input_dim=int(meta["snap_input_dim"]),
        seq_len=int(meta["seq_len"]),
        dropout=float(meta["dropout"]),
        ctx_cont_dim=int(meta["ctx_cont_dim"]),
        ctx_cat_dim=int(meta["ctx_cat_dim"]),
        m15_seq_dim=int(mtf["m15_seq_dim"]),
        h1_seq_dim=int(mtf["h1_seq_dim"]),
        h4_seq_dim=int(mtf["h4_seq_dim"]),
        d1_seq_dim=int(mtf["d1_seq_dim"]),
        m15_seq_len=int(mtf["m15_seq_len"]),
        h1_seq_len=int(mtf["h1_seq_len"]),
        h4_seq_len=int(mtf["h4_seq_len"]),
        d1_seq_len=int(mtf["d1_seq_len"]),
        m5_seq_dim=int(mtf["m5_seq_dim"]),
        m5_seq_len=int(mtf["m5_seq_len"]),
        multi_tf_num_layers=int(mtf["multi_tf_num_layers"]),
        multi_tf_scale=float(mtf["multi_tf_scale"]),
        specialist_input_indices={str(k): list(v) for k, v in indices.items()},
        specialist_ctx_cont_indices={
            str(k): list(v)
            for k, v in specialist["context_routing"]["ctx_cont_indices"].items()
        },
        specialist_ctx_cont_nominal_indices={
            str(k): list(v)
            for k, v in specialist["context_routing"][
                "ctx_cont_nominal_indices"
            ].items()
        },
        specialist_ctx_cat_indices={
            str(k): list(v)
            for k, v in specialist["context_routing"]["ctx_cat_indices"].items()
        },
        multi_tf_specialist_input_indices={
            str(k): list(v)
            for k, v in require_multi_tf_specialist_routing_v4(
                mtf["feature_names"]
            ).items()
        },
        temporal_alias_signal_indices=list(
            specialist["context_routing"]["temporal_alias_policy"]["signal_indices"]
        ),
        temporal_alias_ctx_cont_indices=list(
            specialist["context_routing"]["temporal_alias_policy"]["ctx_cont_indices"]
        ),
        specialist_num_layers=int(specialist["num_layers"]),
        specialist_fusion_scale=float(specialist["fusion_scale"]),
        cross_family_fusion_scale=float(specialist["cross_family_fusion_scale"]),
        input_normalization=normalization,
    ).to(device)


def run(
    *,
    manifest_path: Path,
    stage: str,
    checkpoint_dir: Path,
    device: torch.device,
    arm_batch_size: int,
    progress_path: Path,
    train_session_manifest_path: Path | None,
    max_optimizer_steps: int | None,
    plan_sha256: str,
    invocation_sha256: str,
) -> dict[str, Any]:
    launch = require_launch_manifest(_read(manifest_path))
    if any(
        len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
        for value in (plan_sha256, invocation_sha256)
    ):
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_CAMPAIGN_BINDING_INVALID")
    files = {k: Path(v["path"]) for k, v in launch["files"].items()}
    selected = require_selected_sampler_artifact(_read(files["selected_sampler"]))
    if arm_batch_size not in launch["batch_sizes"]:
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_BATCH_ARM_INVALID")
    root_checkpoint_dir = Path(launch["checkpoint_dir"])
    gpu_selection = None
    full_session = None
    if stage == "smoke-arm":
        expected_checkpoint_dir = root_checkpoint_dir / f"batch_{arm_batch_size}"
        expected_progress_path = expected_checkpoint_dir / "PROGRESS.json"
        if train_session_manifest_path is not None or max_optimizer_steps is not None:
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_STAGE_ARGUMENT_INVALID")
    else:
        if train_session_manifest_path is None:
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_TRAIN_SESSION_REQUIRED")
        session_phase = "epoch1" if stage == "epoch1-window" else "resume_proof"
        train_session = require_train_session_manifest(
            _read(train_session_manifest_path), expected_phase=session_phase
        )
        gpu_selection = require_selection(
            _read(Path(train_session["gpu_batch_selection"]["path"]))
        )
        from gx1.contracts.unified_exit_full_population_train_session_v1 import (
            SCHEMA_VERSION as FULL_POPULATION_SCHEMA,
        )
        if train_session["schema_version"] == FULL_POPULATION_SCHEMA:
            if stage != "epoch1-window":
                raise RuntimeError("UNIFIED_EXIT_FULL_YEAR_STAGE_INVALID")
            full_session = train_session
            current_repo = str(Path(__file__).resolve().parents[2])
            current_head = subprocess.check_output(
                ["git", "-C", current_repo, "rev-parse", "HEAD"], text=True
            ).strip()
            current_dirty = subprocess.check_output(
                ["git", "-C", current_repo, "status", "--porcelain", "--untracked-files=all"],
                text=True,
            )
            if (
                current_repo != full_session["source_repo"]
                or current_head != full_session["source_commit"] or current_dirty
                or full_session["predecessor_source_commit"] != launch["source_commit"]
            ):
                raise RuntimeError("UNIFIED_EXIT_FULL_YEAR_EXECUTION_SOURCE_INVALID")
        if (
            train_session["prelaunch_manifest_sha256"] != launch["manifest_sha256"]
            or (full_session is None and train_session["source_commit"] != launch["source_commit"])
            or train_session["gpu_batch_selection_artifact_sha256"]
            != gpu_selection["artifact_sha256"]
            or gpu_selection["launch_manifest_sha256"] != launch["manifest_sha256"]
            or gpu_selection["selected_batch_size"] != arm_batch_size
        ):
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_TRAIN_SESSION_INVALID")
        if stage == "reference-4":
            expected_checkpoint_dir = (
                root_checkpoint_dir
                / "comparison"
                / f"reference_4_batch_{arm_batch_size}"
            )
            expected_progress_path = expected_checkpoint_dir / "PROGRESS.json"
        elif stage in {"resume-proof-first", "resume-proof-second"}:
            expected_checkpoint_dir = (
                root_checkpoint_dir / "comparison" / f"split_batch_{arm_batch_size}"
            )
            suffix = "FIRST" if stage == "resume-proof-first" else "SECOND"
            expected_progress_path = expected_checkpoint_dir / f"PROGRESS_{suffix}.json"
        elif stage == "epoch1-window":
            expected_checkpoint_dir = (
                root_checkpoint_dir / f"epoch1_batch_{arm_batch_size}"
                if full_session is None else Path(full_session["checkpoint_dir"])
            )
            expected_progress_path = expected_checkpoint_dir / "PROGRESS.json"
            if (
                isinstance(max_optimizer_steps, bool)
                or not isinstance(max_optimizer_steps, int)
                or max_optimizer_steps < 1
            ):
                raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_EPOCH_WINDOW_INVALID")
        else:
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_STAGE_INVALID")
        if stage != "epoch1-window" and max_optimizer_steps is not None:
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_STAGE_ARGUMENT_INVALID")
    if (
        checkpoint_dir != expected_checkpoint_dir
        or progress_path != expected_progress_path
    ):
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_CHECKPOINT_DIR_INVALID")
    selected_root = selected["random_access_root"]
    if files["random_access_root"] != Path(selected_root["path"]):
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_RANDOM_ACCESS_ROOT_INVALID")
    head = subprocess.run(
        ["git", "-C", launch["source_repo"], "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if head != launch["source_commit"]:
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_SOURCE_COMMIT_INVALID")
    status = subprocess.run(
        [
            "git",
            "-C",
            launch["source_repo"],
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    if status:
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_SOURCE_NOT_CLEAN")
    _set_deterministic(int(launch["seed"]), device, "deterministic_fp32")
    meta = _read(files["source_bundle_metadata"])
    _bind_multi_tf_cache_from_source_bundle_metadata(meta)
    child = require_composite_normalization_binding(
        _read(files["child_composite_normalization"])
    )
    bootstrap = require_bootstrap_composite_normalization(
        _read(files["bootstrap_composite_normalization"])
    )
    child_norm = child["base_feature_normalization"]["artifact"]["contract"]
    old_norm = bootstrap["base_feature_normalization"]["artifact"]["base_artifact"][
        "contract"
    ]
    per_tf = {
        name.upper(): int(meta["multi_tf"][f"{name}_seq_len"])
        for name in ("m5", "m15", "h1", "h4", "d1")
    }
    dataset = EntryV10CtxDataset(
        files["entry_train_parquet"],
        seq_len=int(meta["seq_len"]),
        m5_prebuilt_path=files["m5_prebuilt"],
        per_tf_seq_lens=per_tf,
        multi_tf_closed_bar=True,
        sequence_source_audit_json=files["sequence_source_audit"],
    )
    corpus = UnifiedExitLifecycleCorpus(
        root_manifest_path=files["feature_lifecycle_root"],
        entry_parquets={
            "train": files["entry_train_parquet"],
            "val": files["entry_val_parquet"],
        },
        entry_manifest_bindings={
            "train": {
                "path": str(files["entry_train_manifest"]),
                "sha256": file_sha256(files["entry_train_manifest"]),
            },
            "val": {
                "path": str(files["entry_val_manifest"]),
                "sha256": file_sha256(files["entry_val_manifest"]),
            },
        },
        dataset_run_id=str(launch["dataset_run_id"]),
        splits=("train", "val"),
    )
    factory = build_random_access_train_adapter_factory_v1(
        root_manifest_path=files["random_access_root"],
        candidate_set_path=Path(selected["candidate_set"]["path"]),
        composite_normalization_path=files["child_composite_normalization"],
        economics_readiness_path=files["economics_readiness"],
        train_cost_authority_path=files["train_cost_authority"],
        train_dataset=dataset,
        train_feature_source_owner=corpus.splits["train"],
    )
    adapter = factory(65536)
    full_population_schedule = None
    if full_session is None:
        adapter.set_epoch_index(0)
    else:
        full_population_schedule = adapter.set_full_population_epoch_index(0)
        if full_population_schedule != full_session["full_population_schedule"]:
            raise RuntimeError("UNIFIED_EXIT_FULL_YEAR_REBUILT_SCHEDULE_DIFFERS")
    dataset.bind_unified_exit_lifecycle_v2(adapter)
    index = pd.read_parquet(
        files["train_random_access_index"],
        columns=["entry_row_index", "parent_entry_row_index"],
    )
    children = index["entry_row_index"].astype("int64").tolist()
    parents = index["parent_entry_row_index"].astype("int64").tolist()
    dataset.bind_random_access_entry_coordinate_mapping_v1(
        parent_entry_row_indices=parents, child_entry_row_indices=children
    )
    child_to_parent = dict(zip(children, parents))
    child_order = [int(x) for x in adapter.random_access_selected_entry_rows_v1()]
    parent_order = [child_to_parent[x] for x in child_order]
    schedule_identity = {
        "epoch_index": 0,
        "selected_sampler_contract_sha256": selected["selected_sampler_contract_sha256"],
        "child_entry_order": child_order,
    }
    if full_population_schedule is not None:
        schedule_identity["full_population_schedule_sha256"] = full_population_schedule["schedule_sha256"]
    epoch_schedule_sha256 = _canonical(schedule_identity)
    schedule_witness = _schedule_witness(
        child_order=child_order,
        parent_order=parent_order,
        batch_size=arm_batch_size,
        epoch_schedule_sha256=epoch_schedule_sha256,
        selected_sampler_artifact_sha256=selected["artifact_sha256"],
        full_population_schedule=full_population_schedule,
    )
    source_pointer = _read(files["checkpoint_pointer"])
    source_state_path = (
        files["checkpoint_pointer"].parent
        / f"candidate_training_state_slot_{source_pointer['slot']}.pt"
    )
    if file_sha256(source_state_path) != source_pointer["state_sha256"]:
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_SOURCE_STATE_INVALID")
    source = torch.load(source_state_path, map_location="cpu", weights_only=True)
    expected_bootstrap_sources = {
        "online": canonical_model_state_sha256(source["model_state"]),
        "target": canonical_model_state_sha256(source["target_model_state"]),
    }
    if device.type == "cuda" and stage in {
        "smoke-arm", "reference-4", "resume-proof-first", "resume-proof-second"
    }:
        _announce_attended_preflight_ready(execution_tier="attended_only")
    model = _model(meta, child_norm, device)
    target = copy.deepcopy(model).to(device)
    joint_task_parameters = list(model.task_log_variances.parameters())
    joint_task_parameter_ids = {id(parameter) for parameter in joint_task_parameters}
    representation_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in joint_task_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": representation_parameters,
                "weight_decay": float(launch["weight_decay"]),
            },
            {"params": joint_task_parameters, "weight_decay": 0.0},
        ],
        lr=float(launch["learning_rate"]),
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=30, eta_min=0.0
    )
    weight_ema = _FreshWeightEma(model, float(launch["weight_ema_decay"]))
    pointer_path = checkpoint_dir / "RESUME_POINTER.json"
    prefix_resume = full_session is not None and not pointer_path.exists()
    bootstrap_stage = full_session is None and (
        stage in {"smoke-arm", "reference-4", "resume-proof-first"}
        or (stage == "epoch1-window" and not pointer_path.exists())
    )
    if bootstrap_stage:
        if checkpoint_dir.exists():
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_CHECKPOINT_DIR_EXISTS")
        online_receipt = bootstrap_random_access_v2_from_pretrained(
            model, source["model_state"]
        )
        target_receipt = bootstrap_random_access_v2_from_pretrained(
            target, source["target_model_state"]
        )
        if (
            online_receipt["initialized_v2_state_sha256"]
            != target_receipt["initialized_v2_state_sha256"]
            or online_receipt["source_pretrained_state_sha256"]
            != expected_bootstrap_sources["online"]
            or target_receipt["source_pretrained_state_sha256"]
            != expected_bootstrap_sources["target"]
        ):
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_TARGET_INIT_MISMATCH")
        bind_preserved_v7_input_normalization(model, old_norm)
        bind_preserved_v7_input_normalization(target, old_norm)
        weight_ema = _FreshWeightEma(model, float(launch["weight_ema_decay"]))
        start = 0
        receipts = {"online": online_receipt, "target": target_receipt}
        steps = {"smoke-arm": 3, "reference-4": 4, "resume-proof-first": 3}.get(
            stage, 0
        )
        global_optimizer_steps = 0
        resume_probe = False
    else:
        resume_pointer_path = (
            Path(full_session["prefix_checkpoint"]["path"])
            if prefix_resume else pointer_path
        )
        resume_schedule_sha256 = (
            full_session["prefix_epoch_schedule_sha256"]
            if prefix_resume else epoch_schedule_sha256
        )
        progress = load_checkpoint_strict(
            pointer_path=resume_pointer_path,
            model=model,
            target_model=target,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            expected_launch_manifest_sha256=launch["manifest_sha256"],
            expected_selected_sampler_artifact_sha256=selected["artifact_sha256"],
            expected_bootstrap_source_receipt_sha256=launch[
                "bootstrap_source_receipt_sha256"
            ],
            expected_base_normalization_sha256=old_norm["contract_sha256"],
            expected_summary_normalization_sha256=child[
                "lifetime_summary_normalization"
            ]["normalization_sha256"],
            expected_batch_size=arm_batch_size,
            expected_epoch_schedule_sha256=resume_schedule_sha256,
        )
        if prefix_resume and (
            progress["next_batch_offset"] != full_session["initial_batch_offset"]
            or progress["global_step"] != full_session["initial_global_step"]
            or progress["epoch_index"] != 0
        ):
            raise RuntimeError("UNIFIED_EXIT_FULL_YEAR_PREFIX_CURSOR_INVALID")
        bind_preserved_v7_input_normalization(model, old_norm)
        bind_preserved_v7_input_normalization(target, old_norm)
        start = int(progress["next_batch_offset"])
        global_optimizer_steps = int(progress["global_step"])
        receipts = progress["bootstrap_model_receipts"]
        if (
            set(receipts) != {"online", "target"}
            or any(
                receipts[name].get("source_pretrained_state_sha256")
                != expected_bootstrap_sources[name]
                for name in expected_bootstrap_sources
            )
            or receipts["online"].get("initialized_v2_state_sha256")
            != receipts["target"].get("initialized_v2_state_sha256")
        ):
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_BOOTSTRAP_RECEIPT_INVALID")
        weight_ema.load_state_dict(progress["weight_ema_state"])
        if stage == "resume-proof-second":
            steps = 1
            resume_probe = True
        elif stage == "epoch1-window":
            steps = 0
            resume_probe = False
        else:
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_RESUME_STAGE_INVALID")
    _write_or_require_schedule_witness(
        checkpoint_dir / "EPOCH_SCHEDULE.json", schedule_witness
    )
    total_batches = (
        int(full_session["total_batches_per_epoch"]) if full_session is not None
        else int(gpu_selection["total_batches_per_epoch"])
        if gpu_selection is not None else -(-16384 // arm_batch_size)
    )
    if stage == "epoch1-window":
        remaining = total_batches - start
        if remaining < 1:
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_EPOCH_ALREADY_COMPLETE")
        steps = min(int(max_optimizer_steps), remaining)
    checkpoint_interval = (
        int(gpu_selection["checkpoint_interval_optimizer_steps"])
        if stage == "epoch1-window"
        else None
    )
    checkpoint_times: list[float] = []
    target.requires_grad_(False)
    target.eval()
    target_state_before = canonical_model_state_sha256(target.state_dict())
    loader = DataLoader(
        dataset,
        batch_size=arm_batch_size,
        sampler=_ParentSampler(parent_order, start, arm_batch_size),
        num_workers=0,
        generator=torch.Generator().manual_seed(int(launch["seed"]) + arm_batch_size),
    )

    initial_global_optimizer_steps = global_optimizer_steps

    def checkpoint(*, next_batch_offset: int, complete_epoch: bool) -> None:
        nonlocal global_optimizer_steps
        global_optimizer_steps = _absolute_optimizer_step(
            initial_global_step=initial_global_optimizer_steps,
            start_batch_offset=start,
            next_batch_offset=next_batch_offset,
        )
        if weight_ema.steps != global_optimizer_steps:
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_EMA_CURSOR_MISMATCH")
        checkpoint_times.append(time.perf_counter())
        state = build_checkpoint(
            model=model,
            target_model=target,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            weight_ema_state=weight_ema.state_dict(),
            global_step=global_optimizer_steps,
            next_batch_offset=next_batch_offset,
            epoch_index=0,
            batch_size=arm_batch_size,
            epoch_schedule_sha256=epoch_schedule_sha256,
            launch_manifest_sha256=launch["manifest_sha256"],
            selected_sampler_artifact_sha256=selected["artifact_sha256"],
            bootstrap_source_receipt_sha256=launch["bootstrap_source_receipt_sha256"],
            bootstrap_model_receipts=receipts,
            base_normalization_sha256=old_norm["contract_sha256"],
            summary_normalization_sha256=child["lifetime_summary_normalization"][
                "normalization_sha256"
            ],
        )
        save_checkpoint_atomic(state, directory=checkpoint_dir)

    supervision = {name: False for name in JOINT_TASK_NAMES}
    gradients = {name: False for name in JOINT_TASK_NAMES}
    _, stats, _ = train_epoch(
        model,
        target,
        loader,
        optimizer,
        device,
        grad_accum_steps=1,
        task_supervision_observed=supervision,
        task_gradient_observed=gradients,
        weight_ema=weight_ema,
        session_batch_offset=start,
        session_max_optimizer_steps=steps,
        session_checkpoint_hook=checkpoint,
        session_exit_action_forward_chunk_rows=None,
        session_checkpoint_every_optimizer_step=stage != "epoch1-window",
        session_checkpoint_interval_optimizer_steps=checkpoint_interval,
        session_log_label="RANDOM_ACCESS_FIXED_STEP",
        performance_warmup_optimizer_steps=0,
        session_resume_probe=resume_probe,
    )
    if canonical_model_state_sha256(target.state_dict()) != target_state_before:
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_TARGET_CHANGED_WITHIN_EPOCH")
    if (
        global_optimizer_steps != initial_global_optimizer_steps + steps
        or weight_ema.steps != global_optimizer_steps
    ):
        raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_OPTIMIZER_CURSOR_MISMATCH")
    if stage == "smoke-arm":
        if len(checkpoint_times) != 3 or checkpoint_times[-1] <= checkpoint_times[0]:
            raise RuntimeError("UNIFIED_EXIT_FIXED_STEP_MEASUREMENT_MISSING")
        measurement = {
            "schema_version": "gx1_unified_exit_cuda_smoke_measurement_v1",
            "launch_manifest_sha256": launch["manifest_sha256"],
            "batch_size": arm_batch_size,
            "warmup_optimizer_steps": 1,
            "measured_optimizer_steps": 2,
            "measured_entry_rows": 2 * arm_batch_size,
            "transitions_per_entry": 4,
            "measured_transition_count": 8 * arm_batch_size,
            "measured_train_seconds": checkpoint_times[-1] - checkpoint_times[0],
            "test_data_used": False,
        }
        measurement["measurement_sha256"] = _canonical(measurement)
        _write_progress_atomic(
            progress_path.with_name("SMOKE_MEASUREMENT.json"), measurement
        )
    completed = start + steps
    progress_outcome = (
        "COMPLETE"
        if stage == "epoch1-window" and completed == total_batches
        else "RESUMABLE"
        if stage == "epoch1-window"
        else "COMPLETE"
    )
    progress = {
        "schema_version": "gx1_local_random_access_progress_v2",
        "plan_sha256": plan_sha256,
        "invocation_sha256": invocation_sha256,
        "phase": {
            "reference-4": "reference_run",
            "smoke-arm": "smoke_arm",
            "resume-proof-first": "resume_proof_first",
            "resume-proof-second": "resume_proof_second",
            "epoch1-window": "epoch1_window",
        }[stage],
        "epoch_index": 0,
        "global_optimizer_steps": global_optimizer_steps,
        "next_batch_offset": completed,
        "total_batches": total_batches,
        "completed_units": completed,
        "total_units": total_batches,
        "epoch_schedule_sha256": epoch_schedule_sha256,
        "selection_receipt_sha256": gpu_selection["artifact_sha256"]
        if gpu_selection is not None
        else None,
        "checkpoint_pointer": {
            "path": str(pointer_path),
            "sha256": file_sha256(pointer_path),
        },
        "terminal": True,
        "outcome": progress_outcome,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
    }
    progress = _write_campaign_progress(progress_path, progress)
    return {**progress, "progress_path": str(progress_path), "stats": stats}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--launch-manifest", type=Path, required=True)
    p.add_argument(
        "--stage",
        choices=(
            "smoke-arm",
            "reference-4",
            "resume-proof-first",
            "resume-proof-second",
            "epoch1-window",
        ),
        required=True,
    )
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--arm-batch-size", type=int, choices=(4, 8, 16), required=True)
    p.add_argument("--progress-path", type=Path, required=True)
    p.add_argument("--train-session-manifest", type=Path)
    p.add_argument("--max-optimizer-steps", type=int)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    a = p.parse_args(argv)
    print(
        json.dumps(
            run(
                manifest_path=a.launch_manifest.resolve(),
                stage=a.stage,
                checkpoint_dir=a.checkpoint_dir.resolve(),
                device=torch.device(a.device),
                arm_batch_size=a.arm_batch_size,
                progress_path=a.progress_path.resolve(),
                train_session_manifest_path=(
                    a.train_session_manifest.resolve()
                    if a.train_session_manifest
                    else None
                ),
                max_optimizer_steps=a.max_optimizer_steps,
                plan_sha256=os.environ.get("GX1_CAMPAIGN_PLAN_SHA256", ""),
                invocation_sha256=os.environ.get(
                    "GX1_CAMPAIGN_INVOCATION_SHA256", ""
                ),
            ),
            sort_keys=True,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
