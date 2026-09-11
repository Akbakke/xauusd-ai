#!/usr/bin/env python3
"""VAL-only paired permutation audit for learned Entry/Exit usefulness.

The executable core accepts a frozen candidate predictor and immutable VAL
state tensors.  It swaps complete feature trajectories between genuine rows
according to a label-independent whole-block donor plan.  It does not retrain,
select a checkpoint, change model outputs or inspect TEST.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from gx1.contracts.entry_exit_feature_usefulness_v1 import (
    DECISION,
    DONOR_PLAN_SCHEMA_VERSION,
    PERTURBATION_POLICY,
    POLICY,
    SCHEMA_VERSION,
    SIDE_PAIR_PLAN_SCHEMA_VERSION,
    SPLIT,
    StreamingArrayDigest,
    StreamingPairedSummary,
    TASKS,
    TASK_CLASS_ORDER,
    canonical_json_sha256,
    feature_usefulness_layout,
    require_feature_usefulness_identity,
    require_feature_usefulness_report,
    require_feature_usefulness_teacher_identity,
)
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS,
    MTF_SEMANTIC_CATEGORICAL_DOMAINS,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DOMAINS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
)
from gx1.contracts.entry_fitted_q_v1 import (
    ENTRY_FITTED_Q_SCHEMA_VERSION,
    entry_fitted_q_contract,
    require_entry_fitted_q_iteration_state,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    UNIFIED_EXIT_FITTED_Q_SCHEMA_VERSION,
    require_unified_exit_fitted_q_iteration_state,
    unified_exit_fitted_q_contract,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
from gx1.scripts.entry_candidate_prediction_evidence_v1 import atomic_write_text

if TYPE_CHECKING:
    from gx1.contracts.unified_exit_lifecycle_v1 import UnifiedExitLifecycleCorpus
    from gx1.models.entry_v10.entry_v10_ctx_train_v3 import EntryV10CtxDataset
    from gx1.scripts.entry_exit_feature_usefulness_native_v1 import SelectedNativeExitPair


Predictor = Callable[[Mapping[str, np.ndarray]], np.ndarray]


@dataclass(frozen=True)
class NativeVALInputs:
    """Opening-time provenance and owner-backed inputs, valid inside the context.

    This is not a report identity, population prediction or F2 completion.
    Entry clocks are complete immutable row-order arrays, not Exit-state tapes.
    The consumer must not retain the dataset/corpus past context exit and owns
    post-consumption artifact stability and actual prediction evidence.
    """

    dataset: EntryV10CtxDataset
    corpus: UnifiedExitLifecycleCorpus
    entry_bar_open_time_ns: np.ndarray
    entry_decision_time_ns: np.ndarray
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class NativeUsefulnessBaseline:
    """Frozen predictions/targets, not a rebuilt dataset or published report.

    Only O(Entry rows) tokens/seals and O(Exit states) small Q/mask/clock arrays
    are retained. Native feature histories and intervention loss vectors are
    never cached for the full population. The explicit byte budget covers
    retained ndarray payloads, not model, input-owner or process peak memory.
    """

    arrays: Mapping[str, np.ndarray]
    episode_pack_sha256: tuple[str | None, ...]
    fill_binding_sha256: tuple[str | None, ...]
    entry_input_sha256: tuple[str, ...]
    native_mtf_geometry_by_block: Mapping[str, str]
    identity: Mapping[str, Any]
    donor_indices: np.ndarray
    donor_plan: Mapping[str, Any]
    side_pair_plan: Mapping[str, Any]
    retained_array_bytes: int


def _native_entry_batch(
    dataset: Any, rows: np.ndarray, *, task_layout: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    import torch
    from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_MTF_CONTEXT_TIMEFRAMES

    keys = {
        "seq_signal": "seq_x", "snap_signal": "snap_x",
        "ctx_cont": "ctx_cont", "ctx_cat": "ctx_cat",
        **{f"seq_{timeframe.lower()}": f"seq_{timeframe.lower()}" for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES},
    }
    collected = {key: [] for key in keys}
    for index in rows:
        sample = dataset[int(index)]
        entry_index = sample.get("entry_row_index")
        if (
            not isinstance(entry_index, torch.Tensor) or entry_index.dtype != torch.int64
            or entry_index.ndim != 0 or int(entry_index.item()) != int(index)
        ):
            raise RuntimeError("FEATURE_USEFULNESS_NATIVE_ENTRY_BATCH_INDEX_MISMATCH")
        for surface, key in keys.items():
            value = sample.get(key)
            dtype = torch.int64 if key == "ctx_cat" else torch.float32
            if (
                not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != dtype or value.requires_grad
                or not bool(torch.isfinite(value).all().item())
            ):
                raise RuntimeError("FEATURE_USEFULNESS_NATIVE_ENTRY_BATCH_INPUT_INVALID")
            collected[surface].append(value.detach().numpy())
    result = {key: np.stack(values) for key, values in collected.items()}
    _require_state_surfaces(
        result, task="entry", task_layout=task_layout, row_count=len(rows),
        timeframes=task_layout["timeframes"],
    )
    if result["seq_signal"].shape[1] != dataset.seq_len or any(
        result[f"seq_{timeframe.lower()}"].shape[1] != dataset.per_tf_seq_lens[timeframe]
        for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES
    ):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_ENTRY_BATCH_SEQUENCE_LENGTH_MISMATCH")
    _require_alias_manifold(result, task_layout["physical_field_perturbations"])
    return result


def _native_entry_input_digests(batch: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    return tuple(
        canonical_json_sha256({
            key: _array_sha256(value[index:index + 1], domain=f"native_usefulness_entry_input:{key}\0".encode())
            for key, value in batch.items()
        })
        for index in range(len(batch["seq_signal"]))
    )


def _native_entry_forward(
    *, model: Any, batch: Mapping[str, np.ndarray], device: Any,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from gx1.contracts.entry_decision_token_v1 import ENTRY_DECISION_TOKEN_DIM, ENTRY_DECISION_TOKEN_KEY
    from gx1.models.entry_v10.entry_v10_ctx_train_v3 import _model_forward_fp32

    tensors = {key: torch.from_numpy(value.copy()).to(device) for key, value in batch.items()}
    with torch.no_grad():
        outputs = _model_forward_fp32(
            model, tensors["seq_signal"], tensors["snap_signal"],
            ctx_cont=tensors["ctx_cont"], ctx_cat=tensors["ctx_cat"],
            **{key: value for key, value in tensors.items() if key.startswith("seq_") and key != "seq_signal"},
        )
    checked = []
    for key, width in (("entry_action_q_bps", len(TASK_CLASS_ORDER["entry"])), (ENTRY_DECISION_TOKEN_KEY, ENTRY_DECISION_TOKEN_DIM)):
        value = outputs.get(key)
        if (
            not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or tuple(value.shape) != (len(batch["seq_signal"]), width)
            or not bool(torch.isfinite(value).all().item())
        ):
            raise RuntimeError("FEATURE_USEFULNESS_NATIVE_ENTRY_FORWARD_INVALID")
        checked.append(value.detach().cpu().numpy().copy())
    return checked[0], checked[1]


def _native_baseline_shapes(entry_rows: int, pair_count: int) -> dict[str, tuple[tuple[int, ...], str]]:
    from gx1.contracts.entry_decision_token_v1 import ENTRY_DECISION_TOKEN_DIM
    from gx1.contracts.unified_exit_episode_pack_v1 import (
        UNIFIED_EXIT_EPISODE_ACTION_COUNT, UNIFIED_EXIT_EPISODE_SIDE_COUNT,
        UNIFIED_EXIT_EPISODE_STATE_COUNT,
    )

    entry_shape = (entry_rows, len(TASK_CLASS_ORDER["entry"]))
    exit_shape = (pair_count, UNIFIED_EXIT_EPISODE_SIDE_COUNT, UNIFIED_EXIT_EPISODE_STATE_COUNT, UNIFIED_EXIT_EPISODE_ACTION_COUNT)
    return {
        **{f"entry_{name}": (entry_shape, dtype) for name, dtype in (
            ("q_bps", "<f4"), ("targets_bps", "<f4"), ("valid", "?"), ("equivalent", "?"),
        )},
        **{f"exit_{name}": (exit_shape, dtype) for name, dtype in (
            ("q_bps", "<f4"), ("targets_bps", "<f4"), ("valid", "?"), ("equivalent", "?"),
        )},
        "exit_terminal": (exit_shape[:-1], "?"),
        "exit_decision_time_ns": ((pair_count, UNIFIED_EXIT_EPISODE_STATE_COUNT), "<i8"),
        "eligible_entry_indices": ((pair_count,), "<i8"),
        "online_tokens": ((entry_rows, ENTRY_DECISION_TOKEN_DIM), "<f4"),
        "target_tokens": ((entry_rows, ENTRY_DECISION_TOKEN_DIM), "<f4"),
    }


def _require_native_val_population_unchanged(inputs: NativeVALInputs) -> None:
    for value, key, domain in (
        (inputs.dataset.indices, "entry_indices_sha256", b"native_val_entry_indices"),
        (inputs.entry_bar_open_time_ns, "entry_bar_open_time_ns_sha256", b"native_val_entry_bar_open"),
        (inputs.entry_decision_time_ns, "entry_decision_time_ns_sha256", b"native_val_entry_decision"),
    ):
        if _array_sha256(value, domain=domain) != inputs.provenance[key]:
            raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_POPULATION_CHANGED")


def require_native_val_inputs_unchanged(inputs: NativeVALInputs) -> None:
    """Re-enter the original file authorities, without rebuilding input data.

    Direct selected files are pinned before and after admission. Lifecycle
    re-admission verifies only the selected split and its original transitive
    sources without allocating another M1 corpus. The cold cache loader,
    rather than the trainer's in-memory fast path, rechecks every NPY byte,
    cache inventory and six-clock parameter binding. It releases clean mapped
    pages before returning. This is bounded by the caller's process cap, not
    a claim of zero IO or measured integrated memory headroom.
    """

    from gx1.features.htf_features import load_multi_tf_v4_cache
    from gx1.models.entry_v10.entry_v10_input_normalization import (
        require_manifest_bound_multi_tf_v4_cache,
    )
    from gx1.scripts import entry_exit_feature_usefulness_native_v1 as native

    if not isinstance(inputs, NativeVALInputs):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_INPUTS_REQUIRED")
    _require_native_val_population_unchanged(inputs)
    files = inputs.provenance["files"]
    for path, digest in files.items():
        native._selected_file(Path(path), digest)
    inputs.corpus.require_files_unchanged()
    if inputs.corpus.evidence != inputs.provenance["lifecycle"]:
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_LIFECYCLE_CHANGED")
    binding = inputs.provenance["mtf_source_provenance"]["cache_binding"]
    val = inputs.provenance["dataset_contract"]["splits"]["val"]
    if (
        files.get(binding["manifest_path"]) != binding["manifest_sha256"]
        or files.get(val["manifest_path"]) != val["manifest_sha256"]
    ):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_CACHE_BINDING_CHANGED")
    checked_cache = load_multi_tf_v4_cache(Path(binding["cache_dir"]))
    observed = require_manifest_bound_multi_tf_v4_cache(
        Path(val["manifest_path"]),
        dataset_run_id=inputs.provenance["source_feature_surface"]["dataset_run_id"],
        cache_dir=Path(binding["cache_dir"]), cache=checked_cache,
        context="FEATURE_USEFULNESS_NATIVE_VAL_MTF_RECHECK",
    )
    if observed != binding:
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_CACHE_BINDING_CHANGED")
    for path, digest in files.items():
        native._selected_file(Path(path), digest)
    _require_native_val_population_unchanged(inputs)


def collect_native_usefulness_baseline(
    *, selected_pair: SelectedNativeExitPair, inputs: NativeVALInputs,
    repo: Path, device: Any, batch_rows: int, max_baseline_bytes: int,
    max_episode_bytes: int,
) -> NativeUsefulnessBaseline:
    """Collect every VAL baseline through the existing native model/Q owners.

    The public producer owns execution authority, caps and CUDA guarding. This
    function neither starts a guard nor grants GPU permission. It reads a
    selected completed-VAL pair and already admitted VAL input context only.
    Every Entry row receives separate online/teacher tokens, including FLAT
    and lifecycle-ineligible rows. Genuine ineligible rows get the existing
    Entry owner's FLAT-only mask, never an invented runtime trade snapshot.
    Each eligible pair has one online Exit and one frozen teacher forward;
    raw teacher first-state values, not Bellman target[0], supervise Entry.
    """

    import torch
    from gx1.contracts.entry_decision_token_v1 import entry_decision_token_projection_metadata
    from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_DECISION_BAR_SECONDS, EXIT_MTF_CONTEXT_TIMEFRAMES
    from gx1.contracts.entry_fitted_q_v1 import build_entry_fitted_q_targets, entry_fill_binding_sha256
    from gx1.contracts import unified_exit_episode_pack_v1 as episode_owner
    from gx1.scripts import entry_exit_feature_usefulness_native_v1 as native

    for value in (batch_rows, max_baseline_bytes, max_episode_bytes):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RuntimeError("FEATURE_USEFULNESS_NATIVE_BASELINE_BUDGET_INVALID")
    if not isinstance(inputs, NativeVALInputs):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_INPUTS_REQUIRED")
    _require_native_val_population_unchanged(inputs)
    native.require_selected_native_pair_unchanged(selected_pair=selected_pair, repo=repo)
    dataset = inputs.dataset
    lifecycle = inputs.corpus.splits["val"]
    entry_rows = len(dataset)
    if (
        dataset._unified_exit_lifecycle is not lifecycle
        or lifecycle.split != "val" or lifecycle.entry_row_count != entry_rows
        or inputs.provenance["entry_row_count"] != entry_rows
        or not np.array_equal(dataset.indices, np.arange(entry_rows, dtype=np.int64))
        or inputs.entry_decision_time_ns.shape != (entry_rows,)
    ):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_BASELINE_POPULATION_MISMATCH")
    eligible = sorted({int(index) for index, _side in lifecycle._episode_pointers})
    if (
        len(eligible) < 2 or eligible[0] < 0 or eligible[-1] >= entry_rows
        or set(lifecycle._episode_pointers) != {(index, side) for index in eligible for side in range(episode_owner.UNIFIED_EXIT_EPISODE_SIDE_COUNT)}
        or lifecycle.state_population_rows != len(eligible) * episode_owner.UNIFIED_EXIT_EPISODE_SIDE_COUNT * episode_owner.UNIFIED_EXIT_EPISODE_STATE_COUNT
    ):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_BASELINE_POPULATION_MISMATCH")
    shapes = _native_baseline_shapes(entry_rows, len(eligible))
    required_bytes = sum(math.prod(shape) * np.dtype(dtype).itemsize for shape, dtype in shapes.values())
    required_bytes += len(eligible) * np.dtype("<i8").itemsize
    if required_bytes > max_baseline_bytes:
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_BASELINE_BYTE_BUDGET_EXCEEDED")
    require_native_val_inputs_unchanged(inputs)
    arrays = {key: np.empty(shape, dtype=dtype) for key, (shape, dtype) in shapes.items()}
    arrays["eligible_entry_indices"][:] = eligible
    pair_indices = {entry: index for index, entry in enumerate(eligible)}
    seals: list[str | None] = []
    fills: list[str | None] = []
    entry_inputs: list[str] = []
    geometry: dict[str, str] = {}
    adapter = native.CompactNativeExitUsefulnessAdapter(
        model=selected_pair.model, ordered_signal_names=dataset.signal_names,
        device=device, max_episode_bytes=max_episode_bytes,
    )
    window = inputs.provenance["entry_split_window"]
    end_ns = pd.Timestamp(window["end"]).value
    entry_layout = feature_usefulness_layout(dataset.signal_names)["tasks"]["entry"]
    for start in range(0, entry_rows, batch_rows):
        stop = min(start + batch_rows, entry_rows)
        batch = _native_entry_batch(
            dataset, np.arange(start, stop, dtype=np.int64), task_layout=entry_layout,
        )
        entry_inputs.extend(_native_entry_input_digests(batch))
        online_q, online_tokens = _native_entry_forward(model=selected_pair.model, batch=batch, device=device)
        _teacher_entry_q, teacher_tokens = _native_entry_forward(model=selected_pair.target_model, batch=batch, device=device)
        arrays["entry_q_bps"][start:stop] = online_q
        arrays["online_tokens"][start:stop] = online_tokens
        arrays["target_tokens"][start:stop] = teacher_tokens
        first_values = torch.zeros((stop - start, episode_owner.UNIFIED_EXIT_EPISODE_SIDE_COUNT), dtype=torch.float32)
        side_valid = torch.zeros_like(first_values, dtype=torch.bool)
        for entry in range(start, stop):
            episode = dataset.materialize_full_exit_episode(entry)
            if (episode is None) != (entry not in pair_indices):
                raise RuntimeError("FEATURE_USEFULNESS_NATIVE_BASELINE_EPISODE_OMITTED")
            if episode is None:
                seals.append(None)
                fills.append(None)
                continue
            readiness = episode["unbounded_exit_training_readiness"]
            episode = episode_owner.require_unified_exit_episode_pack(
                {key: value for key, value in episode.items()
                 if key != "unbounded_exit_training_readiness"},
                per_tf_seq_lens=dataset.per_tf_seq_lens,
                expected_mtf_cache_identity_sha256=dataset._multi_tf_cache_identity_sha256,
                context="FEATURE_USEFULNESS_NATIVE_BASELINE",
            )
            episode["unbounded_exit_training_readiness"] = readiness
            if (
                episode["entry_row_index"] != entry
                or episode["lifecycle_state_population_sha256"] != lifecycle.state_population_sha256
                or int(episode["exit_state_row_time_ns"][0]) != int(inputs.entry_decision_time_ns[entry])
                or int(episode["exit_decision_time_ns"][-1]) > end_ns
            ):
                raise RuntimeError("FEATURE_USEFULNESS_NATIVE_BASELINE_EPISODE_IDENTITY_MISMATCH")
            pair_index = pair_indices[entry]
            seal = episode["episode_pack_sha256"]
            seals.append(seal)
            fills.append(entry_fill_binding_sha256(
                entry_row_index=entry, episode_pack_sha256=seal,
                first_exit_state_time_ns=int(episode["exit_state_row_time_ns"][0]),
                exit_entry_bid_ask=episode["exit_entry_bid_ask"],
            ))
            digest = hashlib.sha256(b"gx1_native_mtf_episode_geometry_census_v1\0")
            for timeframe in EXIT_MTF_CONTEXT_TIMEFRAMES:
                lower = timeframe.lower()
                digest.update(timeframe.encode("ascii") + b"\0")
                digest.update(np.asarray(episode[f"exit_mtf_history_{lower}"].shape, dtype="<i8").tobytes())
                digest.update(np.asarray(episode[f"exit_mtf_gather_{lower}"], dtype="<i8").tobytes())
            geometry[str(entry)] = digest.hexdigest()
            offset = entry - start
            online_token = torch.from_numpy(online_tokens[offset:offset + 1].copy()).to(device)
            target_token = torch.from_numpy(teacher_tokens[offset:offset + 1].copy()).to(device)
            exit_q = adapter.predict_spec(episode=episode, online_entry_token=online_token)
            supervision = adapter.baseline_supervision(
                episode=episode, target_model=selected_pair.target_model, target_entry_token=target_token,
            )
            _fitted_q_loss_and_unique_target_margin(
                exit_q.reshape(-1, episode_owner.UNIFIED_EXIT_EPISODE_ACTION_COUNT),
                q_targets_bps=supervision.q_targets_bps.reshape(-1, episode_owner.UNIFIED_EXIT_EPISODE_ACTION_COUNT),
                action_valid_mask=supervision.action_valid_mask.reshape(-1, episode_owner.UNIFIED_EXIT_EPISODE_ACTION_COUNT),
                action_equivalence_mask=supervision.action_equivalence_mask.reshape(-1, episode_owner.UNIFIED_EXIT_EPISODE_ACTION_COUNT),
            )
            for key, value in (
                ("exit_q_bps", exit_q), ("exit_targets_bps", supervision.q_targets_bps),
                ("exit_valid", supervision.action_valid_mask), ("exit_equivalent", supervision.action_equivalence_mask),
                ("exit_terminal", supervision.terminal_mask), ("exit_decision_time_ns", episode["exit_decision_time_ns"]),
            ):
                if value.shape != arrays[key][pair_index].shape or value.dtype != arrays[key].dtype:
                    raise RuntimeError("FEATURE_USEFULNESS_NATIVE_BASELINE_OUTPUT_DTYPE_OR_SHAPE_INVALID")
                arrays[key][pair_index] = value
            # The final HOLD remains legal but has no observed Bellman successor.
            expected_target_mask = episode["exit_action_valid_mask"].copy()
            expected_target_mask[..., -1, 0] = False
            if (
                not np.array_equal(supervision.action_valid_mask, expected_target_mask)
                or not np.array_equal(supervision.terminal_mask, episode["exit_terminal_mask"])
            ):
                raise RuntimeError("FEATURE_USEFULNESS_NATIVE_BASELINE_MASK_MISMATCH")
            if (
                supervision.entry_first_side_values_bps.shape != (episode_owner.UNIFIED_EXIT_EPISODE_SIDE_COUNT,)
                or supervision.entry_first_side_values_bps.dtype != np.dtype("<f4")
                or not np.isfinite(supervision.entry_first_side_values_bps).all()
                or supervision.entry_side_valid_mask.dtype != np.dtype(np.bool_)
                or not np.array_equal(supervision.entry_side_valid_mask, episode["exit_action_valid_mask"][:, 0].any(axis=-1))
            ):
                raise RuntimeError("FEATURE_USEFULNESS_NATIVE_BASELINE_ENTRY_BRIDGE_INVALID")
            first_values[offset] = torch.from_numpy(supervision.entry_first_side_values_bps)
            side_valid[offset] = torch.from_numpy(supervision.entry_side_valid_mask)
        targets, valid, binding = build_entry_fitted_q_targets(
            frozen_exit_first_state_values_bps=first_values, exit_side_valid_mask=side_valid,
            episode_pack_sha256=seals[start:stop], fill_binding_sha256=fills[start:stop],
        )
        arrays["entry_targets_bps"][start:stop] = targets.numpy()
        arrays["entry_valid"][start:stop] = valid.numpy()
        arrays["entry_equivalent"][start:stop] = np.asarray(binding["action_equivalence_mask"], dtype=np.bool_)
    donor, donor_plan, side_plan = build_native_exit_structure_plans(
        entry_row_indices=arrays["eligible_entry_indices"], native_mtf_geometry_by_block=geometry,
    )
    if sum(value.nbytes for value in arrays.values()) + donor.nbytes != required_bytes:
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_BASELINE_BYTE_ACCOUNTING_MISMATCH")
    _require_native_val_population_unchanged(inputs)
    native.require_selected_native_pair_unchanged(selected_pair=selected_pair, repo=repo)
    require_native_val_inputs_unchanged(inputs)
    metadata = selected_pair.metadata
    pair = selected_pair.bindings
    signal = inputs.provenance["dataset_contract"]["contract"]
    val = inputs.provenance["dataset_contract"]["splits"]["val"]
    identity = require_feature_usefulness_identity({
        "bundle_dir": str(Path(pair["bundle_metadata_path"]).parent),
        "bundle_metadata_sha256": pair["bundle_metadata_sha256"],
        "model_state_sha256": pair["online_model_state_sha256"],
        "target_model_state_sha256": pair["target_model_state_sha256"],
        "dataset_dir": str(Path(val["parquet_path"]).parent),
        "dataset_run_id": metadata["run_lineage"]["dataset_run_id"],
        "val_manifest_path": val["manifest_path"], "val_manifest_sha256": val["manifest_sha256"],
        "val_data_path": val["parquet_path"], "val_data_sha256": val["parquet_sha256"],
        "val_start_utc": (pd.Timestamp(window["emission_start"]) + pd.Timedelta(seconds=ENTRY_DECISION_BAR_SECONDS)).isoformat(),
        "val_end_utc": (pd.Timestamp(window["end"]) + pd.Timedelta(seconds=ENTRY_DECISION_BAR_SECONDS)).isoformat(),
        "entry_val_population_row_count": entry_rows,
        "exit_val_population_row_count": lifecycle.state_population_rows,
        "normalization_path": pair["bundle_metadata_path"],
        "normalization_file_sha256": pair["bundle_metadata_sha256"],
        "normalization_contract_sha256": pair["input_normalization_sha256"],
        "online_entry_token_stream_sha256": _array_sha256(arrays["online_tokens"], domain=b"native_usefulness_online_entry_tokens_v1\0"),
        "target_entry_token_stream_sha256": _array_sha256(arrays["target_tokens"], domain=b"native_usefulness_target_entry_tokens_v1\0"),
        "entry_row_indices_sha256": _array_sha256(dataset.indices, domain=b"native_val_entry_indices"),
        "native_episode_pack_set_sha256": canonical_json_sha256(list(enumerate(seals))),
        "native_episode_entry_indices_sha256": _array_sha256(arrays["eligible_entry_indices"], domain=b"native_usefulness_episode_entry_indices_v1\0"),
        "native_episode_pair_count": len(eligible),
        "native_mtf_geometry_sha256": donor_plan["native_mtf_geometry_sha256"],
        "native_episode_pack_contract": episode_owner.unified_exit_episode_pack_contract(),
        "selected_epoch": pair["selected_epoch"], "last_epoch": pair["last_epoch"],
        "train_split_sha256": pair["dataset_artifact_declarations"]["train_parquet"]["sha256"],
        "lifecycle_manifest_path": inputs.provenance["lifecycle"]["root_manifest_path"],
        "lifecycle_manifest_sha256": inputs.provenance["lifecycle"]["root_manifest_sha256"],
        "selection_artifacts": pair["selection_artifacts"], "recipe_source_provenance": pair["recipe_source_provenance"],
        "contract_mode": signal["contract_mode"], "signal_schema_version": signal["schema_version"],
        "signal_static_contract_sha256": signal["static_contract_sha256"],
        "entry_decision_token_projection": entry_decision_token_projection_metadata(),
    })
    require_feature_usefulness_teacher_identity(identity, metadata["unified_exit_training_evidence"]["selected_fitted_q_iteration_state"])
    for array in (*arrays.values(), donor):
        array.setflags(write=False)
    return NativeUsefulnessBaseline(
        arrays=MappingProxyType(arrays), episode_pack_sha256=tuple(seals),
        fill_binding_sha256=tuple(fills), entry_input_sha256=tuple(entry_inputs),
        native_mtf_geometry_by_block=MappingProxyType(geometry), identity=MappingProxyType(identity),
        donor_indices=donor, donor_plan=MappingProxyType(donor_plan), side_pair_plan=MappingProxyType(side_plan),
        retained_array_bytes=required_bytes,
    )


def _require_native_val_population(
    *, dataset: EntryV10CtxDataset, corpus: UnifiedExitLifecycleCorpus,
    window: Mapping[str, Any], reconstruction: Mapping[str, Any],
    signal_contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    from gx1.contracts.entry_exit_feature_base_v1 import ENTRY_DECISION_BAR_SECONDS

    if set(corpus.splits) != {"val"}:
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_SPLIT_INVALID")
    lifecycle = corpus.splits["val"]
    indices = dataset.indices
    rows = window["rows"]
    if (
        not isinstance(indices, np.ndarray) or indices.dtype != np.dtype(np.int64)
        or indices.shape != (rows,)
        or not np.array_equal(indices, np.arange(rows, dtype=np.int64))
        or len(dataset) != rows or len(dataset.df) != rows
        or lifecycle.split != "val" or lifecycle.entry_row_count != rows
        or dataset._compact_row_indices is not None
    ):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_FULL_INDEX_MISMATCH")
    if (
        dataset._sequence_source_reconstructed is not True
        or dataset._sequence_roll_reconstructed is not False
        or dataset._sequence_source_audit != reconstruction
        or dataset._np_seq is not None
        or dataset.model_native_signal_contract != signal_contract
        or list(dataset.signal_names) != list(signal_contract["fields"])
    ):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_RECONSTRUCTION_MISMATCH")
    try:
        times = pd.DatetimeIndex(dataset.df["time"]).as_unit("ns")
        valid_clock = (
            not times.hasnans and times.tz is not None
            and times.is_unique and times.is_monotonic_increasing
            and times[0].utcoffset() == pd.Timedelta(0)
            and times.floor(f"{ENTRY_DECISION_BAR_SECONDS}s").equals(times)
            and times.equals(lifecycle._entry_times)
            and times[0] == window["observed_start"]
            and times[-1] == window["observed_end"]
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_CLOCK_MISMATCH") from exc
    if not valid_clock:
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_CLOCK_MISMATCH")
    positions = dataset._sequence_source_positions
    source_times = dataset._sequence_source_times_ns
    if (
        not isinstance(positions, np.ndarray) or positions.dtype != np.dtype(np.int64)
        or positions.shape != (rows,)
        or not isinstance(source_times, np.ndarray)
        or source_times.dtype != np.dtype(np.int64) or source_times.ndim != 1
        or np.any(positions < dataset.seq_len - 1)
        or np.any(positions >= len(source_times))
        or not np.array_equal(source_times[positions], times.asi8)
    ):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_SOURCE_CLOCK_MISMATCH")
    bar_open = times.asi8.copy()
    decision = (times + pd.Timedelta(seconds=ENTRY_DECISION_BAR_SECONDS)).asi8.copy()
    indices.setflags(write=False)
    bar_open.setflags(write=False)
    decision.setflags(write=False)
    return bar_open, decision


@contextmanager
def open_native_val_inputs(
    *, selected_pair: SelectedNativeExitPair,
    recipe_audit_path: Path, recipe_audit_sha256: str,
) -> Iterator[NativeVALInputs]:
    """Open one full, file-bound pre-TEST candidate VAL population, without forward.

    The pair must come from the strict selected-pair loader. Reuse its exact
    recipe/source closure: no legacy recipe, source exception or live session
    substitution. Only VAL lifecycle files are selected; TRAIN/TEST references
    in the recipe remain declarations, never dataset reads here.

    Existing owners allocate the source-backed Entry dataset and temporary
    mmap M1 corpus once. No normalization fit, feature rebuild, episode tape,
    subsampling or second M1 representation is created. Memory remains subject
    to the caller's process cap, not an inferred RAM guarantee. This context
    temporarily owns the cache environment and is for serial consumption;
    both the prior environment and owner-created scratch are released on exit.
    """

    from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
        require_pretest_technical_recipe_metadata,
    )
    from gx1.contracts.entry_sequence_source_reconstruction_v1 import (
        feature_surface_binding_from_split_manifest,
        require_sequence_source_reconstruction_audit,
    )
    from gx1.contracts.unified_exit_lifecycle_v1 import (
        UnifiedExitLifecycleCorpus,
        _require_entry_split_window_binding,
    )
    from gx1.models.entry_v10.entry_v10_input_normalization import (
        require_manifest_bound_multi_tf_v4_cache,
    )
    from gx1.scripts import entry_exit_feature_usefulness_native_v1 as native
    from gx1.scripts.audit_entry_foundation_smoke_bundle_v1 import _bundle_dataset_kwargs
    from gx1.scripts.evaluate_entry_candidate_selective_edge_v1 import (
        _dataset_model_native_contract,
        _require_evaluation_mtf_source_provenance,
    )

    if not isinstance(selected_pair, native.SelectedNativeExitPair):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_SELECTED_PAIR_REQUIRED")
    metadata = dict(selected_pair.metadata)
    pair_bindings = selected_pair.bindings
    provenance = native.require_training_recipe_source_provenance_metadata(
        pair_bindings["recipe_source_provenance"], context="FEATURE_USEFULNESS_NATIVE_VAL"
    )
    if (
        provenance["recipe_audit_path"] != str(recipe_audit_path)
        or provenance["recipe_audit_sha256"] != recipe_audit_sha256
        or metadata["recipe_source_provenance"] != provenance
        or pair_bindings["files"].get(str(recipe_audit_path)) != recipe_audit_sha256
        or pair_bindings["input_normalization_sha256"]
        != selected_pair.input_normalization["contract_sha256"]
    ):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_PAIR_RECIPE_MISMATCH")
    metadata_path = Path(pair_bindings["bundle_metadata_path"])
    if native._selected_json(metadata_path, pair_bindings["bundle_metadata_sha256"]) != metadata:
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_PAIR_METADATA_MISMATCH")
    recipe = require_pretest_technical_recipe_metadata(
        native._selected_json(recipe_audit_path, recipe_audit_sha256),
        expected_profile="candidate",
        expected_run_id=metadata["run_lineage"]["training_run_id"],
        expected_dataset_run_id=metadata["run_lineage"]["dataset_run_id"],
    )
    observed_source = native.require_training_recipe_source_provenance(
        recipe_audit_path=recipe_audit_path, recipe_audit_sha256=recipe_audit_sha256,
        repo=Path(__file__).resolve().parents[2], profile="candidate",
        run_id=recipe["run_id"], dataset_run_id=recipe["dataset_run_id"],
        dataset_dir=Path(recipe["dataset_dir"]), out_bundle_dir=Path(recipe["out_bundle_dir"]),
    )
    if observed_source != provenance:
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_RECIPE_SOURCE_MISMATCH")
    artifacts = recipe["artifact_bindings"]
    declared = pair_bindings["dataset_artifact_declarations"]
    if declared != {
        "train_parquet": artifacts["train_parquet"],
        "val_parquet": artifacts["val_parquet"],
        "m5_prebuilt_path": artifacts["m5_prebuilt"],
        "unified_exit_lifecycle_manifest": artifacts["unified_exit_lifecycle_manifest"],
    }:
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_RECIPE_ARTIFACT_MISMATCH")
    cli = recipe["trainer_cli"]
    if (
        cli["execution_tier"] != "canonical" or cli["subsample_rows"] != 0
        or cli["num_workers"] != 0 or cli["train_time_window"] is not None
        or cli["seq_len"] != metadata["seq_len"]
    ):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_FULL_RECIPE_REQUIRED")
    files = {
        str(metadata_path): pair_bindings["bundle_metadata_sha256"],
        str(recipe_audit_path): recipe_audit_sha256,
    }
    for name in (
        "val_manifest", "val_parquet", "m5_prebuilt", "multi_tf_cache_manifest",
        "unified_exit_lifecycle_manifest", "val_sequence_source_reconstruction",
    ):
        binding = artifacts[name]
        path, digest = native._selected_file(Path(binding["path"]), binding["sha256"])
        files[str(path)] = digest
    val_path = Path(artifacts["val_parquet"]["path"])
    manifest_path = Path(artifacts["val_manifest"]["path"])
    m5_path = Path(artifacts["m5_prebuilt"]["path"])
    cache_manifest = Path(artifacts["multi_tf_cache_manifest"]["path"])
    if cache_manifest.name != "manifest.json":
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_CACHE_PATH_INVALID")
    dataset_contract = _dataset_model_native_contract(
        Path(recipe["dataset_dir"]), ["val"], {"val": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": artifacts["val_manifest"]["sha256"],
            "parquet_path": str(val_path),
            "parquet_sha256": artifacts["val_parquet"]["sha256"],
        }},
    )
    signal = dataset_contract["contract"]
    if (
        dataset_contract["splits"]["val"]["dataset_run_id"] != recipe["dataset_run_id"]
        or signal != metadata["model_native_signal_contract"]
        or signal["fields"] != list(metadata["ordered_signal_names"])
    ):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_SIGNAL_LINEAGE_MISMATCH")
    mtf = _require_evaluation_mtf_source_provenance(
        dataset_contract=dataset_contract, bundle_metadata=metadata,
        m5_prebuilt=m5_path, mtf_cache_dir=cache_manifest.parent,
    )
    window = _require_entry_split_window_binding(
        binding=artifacts["val_manifest"], entry_path=val_path,
        dataset_run_id=recipe["dataset_run_id"], split="val",
    )
    manifest = native._selected_json(manifest_path, artifacts["val_manifest"]["sha256"])
    surface = feature_surface_binding_from_split_manifest(manifest)
    if surface["dataset_run_id"] != recipe["dataset_run_id"]:
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_SOURCE_LINEAGE_MISMATCH")
    for path_key, digest_key in (("path", "sha256"), ("manifest_path", "manifest_sha256")):
        path, digest = native._selected_file(Path(surface[path_key]), surface[digest_key])
        files[str(path)] = digest
    reconstruction_path = Path(artifacts["val_sequence_source_reconstruction"]["path"])
    reconstruction = require_sequence_source_reconstruction_audit(
        native._selected_json(reconstruction_path, files[str(reconstruction_path)]),
        expected_parquet_path=val_path, expected_manifest_path=manifest_path,
        expected_parquet_sha256=artifacts["val_parquet"]["sha256"],
        expected_manifest_sha256=artifacts["val_manifest"]["sha256"],
        expected_feature_surface=manifest, expected_rows=window["rows"],
        expected_seq_len=metadata["seq_len"], expected_signal_dim=MODEL_NATIVE_SIGNAL_DIM,
    )
    if metadata["sequence_source_reconstruction"]["splits"]["val"] != reconstruction:
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_BUNDLE_RECONSTRUCTION_MISMATCH")
    lifecycle_path = Path(artifacts["unified_exit_lifecycle_manifest"]["path"])
    lifecycle_root = native._selected_json(lifecycle_path, files[str(lifecycle_path)])
    if (
        lifecycle_root["m1_authority"].get("authority_mode") != "pretest_quote_complete_native_v1"
        or set(lifecycle_root["splits"]) != {"train", "val"}
    ):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_PRETEST_LIFECYCLE_REQUIRED")
    dataset_kwargs = _bundle_dataset_kwargs(metadata, m5_path)
    if any(
        cli[f"per_tf_seq_len_{timeframe.lower()}"] != length
        for timeframe, length in dataset_kwargs["per_tf_seq_lens"].items()
    ):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_RECIPE_MTF_GEOMETRY_MISMATCH")
    cache_env = native.trainer._TRAIN_MULTI_TF_CACHE_ENV
    previous_cache = os.environ.get(cache_env)
    try:
        os.environ[cache_env] = str(cache_manifest.parent)
        with ExitStack() as scratch:
            cache_features = native.trainer._prebuild_multi_tf_features_once(m5_path)
            cache = require_manifest_bound_multi_tf_v4_cache(
                manifest_path, dataset_run_id=recipe["dataset_run_id"],
                cache_dir=cache_manifest.parent, cache=cache_features,
                context="FEATURE_USEFULNESS_NATIVE_VAL_MTF",
            )
            if cache != mtf["cache_binding"]:
                raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_LOADED_CACHE_MISMATCH")
            corpus = UnifiedExitLifecycleCorpus(
                root_manifest_path=lifecycle_path, entry_parquets={"val": val_path},
                entry_manifest_bindings={"val": artifacts["val_manifest"]},
                dataset_run_id=recipe["dataset_run_id"], splits=("val",),
            )
            scratch.callback(corpus._m1_feature_tempdir.cleanup)
            selected_lifecycle = metadata["unified_exit_training_evidence"]["lifecycle"]
            if corpus.evidence != {
                **selected_lifecycle,
                "splits": {"val": selected_lifecycle["splits"]["val"]},
            } or corpus.evidence["root_manifest_sha256"] != files[str(lifecycle_path)]:
                raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_LIFECYCLE_BINDING_MISMATCH")
            dataset = native.trainer.EntryV10CtxDataset(
                parquet_path=val_path, seq_len=metadata["seq_len"],
                sequence_source_audit_json=reconstruction_path, **dataset_kwargs,
            )
            if dataset._memmap_tmpdir is not None:
                scratch.callback(dataset._memmap_tmpdir.cleanup)
            if dataset._multi_tf_feats is not cache_features or any(
                getattr(dataset, attribute) != cache[key]
                for attribute, key in (
                    ("_multi_tf_cache_identity_sha256", "cache_identity_sha256"),
                    ("_multi_tf_cache_manifest_sha256", "manifest_sha256"),
                    ("_multi_tf_cache_dir", "cache_dir"),
                    ("_multi_tf_cache_manifest_path", "manifest_path"),
                    ("_multi_tf_cache_m5_source", "m5_prebuilt_source"),
                    ("_multi_tf_cache_m5_source_sha256", "m5_prebuilt_source_sha256"),
                )
            ):
                raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_LOADED_CACHE_MISMATCH")
            bar_open, decision = _require_native_val_population(
                dataset=dataset, corpus=corpus, window=window,
                reconstruction=reconstruction, signal_contract=signal,
            )
            dataset.bind_unified_exit_lifecycle(corpus.splits["val"])
            for path, digest in files.items():
                native._selected_file(Path(path), digest)
            yield NativeVALInputs(
                dataset=dataset, corpus=corpus,
                entry_bar_open_time_ns=bar_open, entry_decision_time_ns=decision,
                provenance=MappingProxyType({
                    "files": MappingProxyType(files), "recipe_source_provenance": observed_source,
                    "bundle_metadata_path": str(metadata_path),
                    "bundle_metadata_sha256": pair_bindings["bundle_metadata_sha256"],
                    "input_normalization_contract": selected_pair.input_normalization,
                    "dataset_contract": dataset_contract, "mtf_source_provenance": mtf,
                    "source_reconstruction": reconstruction,
                    "source_feature_surface": surface, "entry_split_window": window,
                    "lifecycle": copy.deepcopy(corpus.evidence),
                    "entry_row_count": len(dataset),
                    "entry_indices_sha256": _array_sha256(dataset.indices, domain=b"native_val_entry_indices"),
                    "entry_bar_open_time_ns_sha256": _array_sha256(bar_open, domain=b"native_val_entry_bar_open"),
                    "entry_decision_time_ns_sha256": _array_sha256(decision, domain=b"native_val_entry_decision"),
                }),
            )
    finally:
        if previous_cache is None:
            os.environ.pop(cache_env, None)
        else:
            os.environ[cache_env] = previous_cache


def _array_sha256(value: np.ndarray, *, domain: bytes) -> str:
    array = np.asarray(value)
    digest = StreamingArrayDigest(array.shape, array.dtype, domain)
    digest.update(array)
    return digest.finalize()


def build_structure_preserving_donor_plan(
    *,
    block_ids: Sequence[Any],
    within_block_positions: Sequence[int],
    native_mtf_geometry_by_block: Mapping[str, str] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Rotate whole equal-geometry blocks; never shuffle individual rows."""

    raw_blocks = np.asarray(block_ids)
    positions = np.asarray(within_block_positions)
    if (
        raw_blocks.ndim != 1
        or positions.ndim != 1
        or raw_blocks.shape != positions.shape
        or raw_blocks.size < 2
        or positions.dtype.kind not in "iu"
    ):
        raise RuntimeError("FEATURE_USEFULNESS_DONOR_INPUT_INVALID")
    block_tokens = [str(value) for value in raw_blocks.tolist()]
    if any(not value or "\x00" in value for value in block_tokens):
        raise RuntimeError("FEATURE_USEFULNESS_DONOR_BLOCK_ID_INVALID")
    indices_by_block: dict[str, list[int]] = {}
    order: list[str] = []
    previous: str | None = None
    closed: set[str] = set()
    for index, token in enumerate(block_tokens):
        if token != previous:
            if token in closed:
                raise RuntimeError("FEATURE_USEFULNESS_DONOR_BLOCK_NOT_CONTIGUOUS")
            if previous is not None:
                closed.add(previous)
            order.append(token)
            indices_by_block[token] = []
            previous = token
        indices_by_block[token].append(index)
    if native_mtf_geometry_by_block is not None and (
        not isinstance(native_mtf_geometry_by_block, Mapping)
        or set(native_mtf_geometry_by_block) != set(order)
        or any(
            not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in native_mtf_geometry_by_block.values()
        )
    ):
        raise RuntimeError("FEATURE_USEFULNESS_DONOR_NATIVE_MTF_GEOMETRY_INVALID")
    signatures: dict[tuple[str | None, tuple[int, ...]], list[str]] = defaultdict(list)
    for token in order:
        block_indices = indices_by_block[token]
        block_positions = tuple(int(positions[index]) for index in block_indices)
        if len(block_positions) != len(set(block_positions)):
            raise RuntimeError("FEATURE_USEFULNESS_DONOR_WITHIN_POSITION_DUPLICATE")
        signature = (
            None if native_mtf_geometry_by_block is None
            else native_mtf_geometry_by_block[token],
            block_positions,
        )
        signatures[signature].append(token)
    singleton_signatures = [signature for signature, rows in signatures.items() if len(rows) < 2]
    if singleton_signatures:
        raise RuntimeError(
            "FEATURE_USEFULNESS_DONOR_STRUCTURE_HAS_NO_PEER: "
            f"signature_count={len(singleton_signatures)}"
        )
    donor = np.full(raw_blocks.shape[0], -1, dtype=np.int64)
    block_mapping: dict[str, str] = {}
    for signature, tokens in signatures.items():
        for source_position, source_token in enumerate(tokens):
            donor_token = tokens[(source_position + 1) % len(tokens)]
            block_mapping[source_token] = donor_token
            source_rows = indices_by_block[source_token]
            donor_rows = indices_by_block[donor_token]
            if len(source_rows) != len(signature[1]) or len(donor_rows) != len(signature[1]):
                raise RuntimeError("FEATURE_USEFULNESS_DONOR_BLOCK_GEOMETRY_INVALID")
            donor[np.asarray(source_rows, dtype=np.int64)] = np.asarray(
                donor_rows, dtype=np.int64
            )
    if (
        (donor < 0).any()
        or np.array_equal(donor, np.arange(len(donor), dtype=np.int64))
        or (donor == np.arange(len(donor), dtype=np.int64)).any()
        or sorted(donor.tolist()) != list(range(len(donor)))
    ):
        raise RuntimeError("FEATURE_USEFULNESS_DONOR_NOT_DERANGED_PERMUTATION")
    block_id_sha = canonical_json_sha256(block_tokens)
    positions_sha = _array_sha256(
        positions.astype("<i8", copy=False),
        domain=b"feature_usefulness_within_block_positions_v1\0",
    )
    donor_sha = _array_sha256(
        donor.astype("<i8", copy=False),
        domain=b"feature_usefulness_donor_indices_v1\0",
    )
    plan: dict[str, Any] = {
        "schema_version": DONOR_PLAN_SCHEMA_VERSION,
        "row_count": int(len(donor)),
        "block_count": len(order),
        "signature_group_count": len(signatures),
        "label_independent": True,
        "source_fields": ["structure_block_id", "within_block_position"] + (
            [] if native_mtf_geometry_by_block is None else ["native_mtf_geometry"]
        ),
        "native_mtf_geometry_sha256": (
            None if native_mtf_geometry_by_block is None
            else canonical_json_sha256(dict(native_mtf_geometry_by_block))
        ),
        "block_ids_sha256": block_id_sha,
        "within_block_positions_sha256": positions_sha,
        "donor_indices_sha256": donor_sha,
        "all_rows_deranged": True,
        "whole_equal_geometry_blocks_preserved": True,
        "block_mapping_sha256": canonical_json_sha256(block_mapping),
    }
    plan["plan_sha256"] = canonical_json_sha256(plan)
    return donor, plan


def _require_state_surfaces(
    states: Mapping[str, Any],
    *,
    task: str,
    task_layout: Mapping[str, Any],
    row_count: int,
    timeframes: Sequence[str],
) -> dict[str, np.ndarray]:
    required = {
        "seq_signal",
        "snap_signal",
        "ctx_cont",
        "ctx_cat",
        *(f"seq_{timeframe.lower()}" for timeframe in timeframes),
    }
    if task == "exit":
        required.update(
            {
                "entry_decision_representation",
                "exit_path",
                "exit_path_lengths",
                "exit_side_index",
                "exit_episode_index",
                "exit_state_index",
            }
        )
    if not isinstance(states, Mapping) or not required.issubset(states):
        raise RuntimeError("FEATURE_USEFULNESS_STATE_SURFACES_MISSING")
    arrays = {str(name): np.asarray(value) for name, value in states.items()}
    if any(array.ndim < 1 or array.shape[0] != row_count for array in arrays.values()):
        raise RuntimeError("FEATURE_USEFULNESS_STATE_ROW_COUNT_INVALID")
    exact_shapes = {
        "seq_signal": (3, MODEL_NATIVE_SIGNAL_DIM),
        "snap_signal": (2, MODEL_NATIVE_SIGNAL_DIM),
        "ctx_cont": (2, len(MODEL_NATIVE_CTX_CONT_FIELDS)),
        "ctx_cat": (2, len(MODEL_NATIVE_CTX_CAT_FIELDS)),
    }
    for surface, (ndim, width) in exact_shapes.items():
        array = arrays[surface]
        if array.ndim != ndim or array.shape[-1] != width:
            raise RuntimeError(f"FEATURE_USEFULNESS_{surface.upper()}_SHAPE_INVALID")
    for timeframe in timeframes:
        surface = f"seq_{timeframe.lower()}"
        array = arrays[surface]
        if array.ndim != 3 or array.shape[-1] != len(MULTI_TF_PER_BAR_FEATURES_V4):
            raise RuntimeError(f"FEATURE_USEFULNESS_{surface.upper()}_SHAPE_INVALID")
    if task == "exit":
        episode_layout = task_layout["exit_episode_effects"]
        token_width = len(
            episode_layout[0]["targets"][0]["source_indices"]
        )
        path_width = len(episode_layout[1]["targets"][0]["source_indices"])
        if (
            arrays["entry_decision_representation"].ndim != 2
            or arrays["entry_decision_representation"].shape[1] != token_width
            or arrays["exit_path"].ndim != 3
            or arrays["exit_path"].shape[2] != path_width
            or arrays["exit_path_lengths"].shape != (row_count,)
            or arrays["exit_side_index"].shape != (row_count,)
            or arrays["exit_episode_index"].shape != (row_count,)
            or arrays["exit_state_index"].shape != (row_count,)
        ):
            raise RuntimeError("FEATURE_USEFULNESS_EXIT_EPISODE_SHAPE_INVALID")
        for name in (
            "exit_path_lengths", "exit_side_index", "exit_episode_index",
            "exit_state_index",
        ):
            if arrays[name].dtype.kind not in "iu":
                raise RuntimeError(
                    f"FEATURE_USEFULNESS_{name.upper()}_NOT_INTEGER"
                )
        if (
            (arrays["exit_path_lengths"] < 1).any()
            or (arrays["exit_path_lengths"] > arrays["exit_path"].shape[1]).any()
            or not np.isin(arrays["exit_side_index"], (0, 1)).all()
            or (arrays["exit_episode_index"] < 0).any()
            or (arrays["exit_state_index"] < 0).any()
        ):
            raise RuntimeError("FEATURE_USEFULNESS_EXIT_EPISODE_DOMAIN_INVALID")
    for name in required:
        if not np.isfinite(arrays[name]).all():
            raise RuntimeError(f"FEATURE_USEFULNESS_{name.upper()}_NONFINITE")
    ctx_cat = arrays["ctx_cat"]
    if not np.equal(ctx_cat, np.rint(ctx_cat)).all():
        raise RuntimeError("FEATURE_USEFULNESS_CTX_CAT_NOT_INTEGER")
    for index, field in enumerate(MODEL_NATIVE_CTX_CAT_FIELDS):
        if not np.isin(ctx_cat[:, index], MODEL_NATIVE_CTX_CAT_DOMAINS[field]).all():
            raise RuntimeError(f"FEATURE_USEFULNESS_CTX_CAT_DOMAIN_INVALID: {field}")
    for field, domain in CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS.items():
        index = list(MODEL_NATIVE_CTX_CONT_FIELDS).index(field)
        if not np.isin(arrays["ctx_cont"][:, index], domain).all():
            raise RuntimeError(f"FEATURE_USEFULNESS_CTX_CONT_DOMAIN_INVALID: {field}")
    mtf_index = {name: index for index, name in enumerate(MULTI_TF_PER_BAR_FEATURES_V4)}
    for timeframe in timeframes:
        values = arrays[f"seq_{timeframe.lower()}"]
        for field, domain in MTF_SEMANTIC_CATEGORICAL_DOMAINS.items():
            if not np.isin(values[..., mtf_index[field]], domain).all():
                raise RuntimeError(
                    f"FEATURE_USEFULNESS_MTF_DOMAIN_INVALID: {timeframe}:{field}"
                )
    return arrays


def _build_exit_side_pair_plan(
    states: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    episode = np.asarray(states["exit_episode_index"], dtype=np.int64)
    state = np.asarray(states["exit_state_index"], dtype=np.int64)
    side = np.asarray(states["exit_side_index"], dtype=np.int64)
    lookup: dict[tuple[int, int, int], int] = {}
    for index, key in enumerate(zip(episode, state, side, strict=True)):
        normalized = tuple(int(value) for value in key)
        if normalized in lookup:
            raise RuntimeError("FEATURE_USEFULNESS_EXIT_SIDE_PAIR_DUPLICATE")
        lookup[normalized] = index
    pair = np.empty(len(episode), dtype=np.int64)
    for index, (episode_id, state_id, side_id) in enumerate(
        zip(episode, state, side, strict=True)
    ):
        peer = lookup.get((int(episode_id), int(state_id), 1 - int(side_id)))
        if peer is None:
            raise RuntimeError("FEATURE_USEFULNESS_EXIT_SIDE_PAIR_MISSING")
        pair[index] = peer
    if (
        not np.array_equal(pair[pair], np.arange(len(pair), dtype=np.int64))
        or not np.array_equal(episode[pair], episode)
        or not np.array_equal(state[pair], state)
        or not np.array_equal(side[pair], 1 - side)
    ):
        raise RuntimeError("FEATURE_USEFULNESS_EXIT_SIDE_PAIR_INVALID")
    token = states["entry_decision_representation"]
    for episode_id in np.unique(episode):
        rows = np.flatnonzero(episode == episode_id)
        if not np.array_equal(token[rows], np.broadcast_to(token[rows[0]], token[rows].shape)):
            raise RuntimeError("FEATURE_USEFULNESS_EXIT_FROZEN_TOKEN_NOT_EPISODE_IMMUTABLE")
    plan: dict[str, Any] = {
        "schema_version": SIDE_PAIR_PLAN_SCHEMA_VERSION,
        "row_count": len(pair),
        "source_fields": [
            "exit_episode_index", "exit_state_index", "exit_side_index"
        ],
        "pair_indices_sha256": _array_sha256(
            pair.astype("<i8", copy=False), domain=b"exit_side_pair_indices_v1\0"
        ),
        "episode_indices_sha256": _array_sha256(
            episode.astype("<i8", copy=False), domain=b"exit_episode_indices_v1\0"
        ),
        "state_indices_sha256": _array_sha256(
            state.astype("<i8", copy=False), domain=b"exit_state_indices_v1\0"
        ),
        "side_indices_sha256": _array_sha256(
            side.astype("<i8", copy=False), domain=b"exit_side_indices_v1\0"
        ),
        "involutive": True,
        "same_episode_state": True,
        "opposite_side": True,
    }
    plan["plan_sha256"] = canonical_json_sha256(plan)
    return pair, plan


def build_native_exit_structure_plans(
    *, entry_row_indices: Sequence[int],
    native_mtf_geometry_by_block: Mapping[str, str],
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Bind full [entry,side,state] plans using O(entry count) retained memory.

    Each native paired pack is one structure block named by its immutable Entry
    row index. That same index is the shared episode identity on both sides.
    The existing permutation owner selects donors; only its repeated state-row
    representation is hashed here. Episode/source/token validation belongs to
    the caller, not to these structural plans.
    """

    from gx1.contracts.unified_exit_episode_pack_v1 import (
        UNIFIED_EXIT_EPISODE_STATE_COUNT,
    )
    from gx1.models.entry_v10.direction_decision_contract import UNIFIED_EXIT_SIDE_ORDER

    entries = np.asarray(entry_row_indices)
    if (
        entries.ndim != 1 or entries.size < 2 or entries.dtype.kind not in "iu"
        or (entries < 0).any() or (entries > np.iinfo(np.int64).max).any()
        or (entries[1:] <= entries[:-1]).any()
    ):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_ENTRY_ROWS_INVALID")
    blocks = [str(int(entry)) for entry in entries]
    donor, compact_plan = build_structure_preserving_donor_plan(
        block_ids=blocks,
        within_block_positions=np.zeros(len(entries), dtype=np.int64),
        native_mtf_geometry_by_block=native_mtf_geometry_by_block,
    )
    state_count = UNIFIED_EXIT_EPISODE_STATE_COUNT
    side_count = len(UNIFIED_EXIT_SIDE_ORDER)
    block_rows = state_count * side_count
    row_count = len(entries) * block_rows
    domains = {
        "within_block_positions_sha256": b"feature_usefulness_within_block_positions_v1\0",
        "donor_indices_sha256": b"feature_usefulness_donor_indices_v1\0",
        "pair_indices_sha256": b"exit_side_pair_indices_v1\0",
        "episode_indices_sha256": b"exit_episode_indices_v1\0",
        "state_indices_sha256": b"exit_state_indices_v1\0",
        "side_indices_sha256": b"exit_side_indices_v1\0",
    }
    digests = {
        name: StreamingArrayDigest((row_count,), np.dtype("<i8"), domain)
        for name, domain in domains.items()
    }
    positions = np.arange(block_rows, dtype="<i8")
    state_indices = np.tile(np.arange(state_count, dtype="<i8"), side_count)
    side_indices = np.repeat(np.arange(side_count, dtype="<i8"), state_count)
    opposite_indices = (1 - side_indices) * state_count + state_indices
    block_digest = hashlib.sha256(b"[")
    for index, entry in enumerate(entries):
        if index:
            block_digest.update(b",")
        block_digest.update(
            json.dumps([blocks[index]] * block_rows, separators=(",", ":"), ensure_ascii=True)[1:-1].encode("utf-8")
        )
        values = {
            "within_block_positions_sha256": positions,
            "donor_indices_sha256": int(donor[index]) * block_rows + positions,
            "pair_indices_sha256": index * block_rows + opposite_indices,
            "episode_indices_sha256": np.full(block_rows, int(entry), dtype="<i8"),
            "state_indices_sha256": state_indices,
            "side_indices_sha256": side_indices,
        }
        for name, array in values.items():
            digests[name].update(array)
    block_digest.update(b"]")
    hashes = {name: digest.finalize() for name, digest in digests.items()}
    donor_plan = {
        **compact_plan,
        "row_count": row_count,
        "block_ids_sha256": block_digest.hexdigest(),
        "within_block_positions_sha256": hashes["within_block_positions_sha256"],
        "donor_indices_sha256": hashes["donor_indices_sha256"],
    }
    donor_plan.pop("plan_sha256")
    donor_plan["plan_sha256"] = canonical_json_sha256(donor_plan)
    side_plan = {
        "schema_version": SIDE_PAIR_PLAN_SCHEMA_VERSION,
        "row_count": row_count,
        "source_fields": ["exit_episode_index", "exit_state_index", "exit_side_index"],
        **{name: hashes[name] for name in (
            "pair_indices_sha256", "episode_indices_sha256",
            "state_indices_sha256", "side_indices_sha256",
        )},
        "involutive": True,
        "same_episode_state": True,
        "opposite_side": True,
    }
    side_plan["plan_sha256"] = canonical_json_sha256(side_plan)
    return donor, donor_plan, side_plan


def _require_alias_manifold(
    states: Mapping[str, np.ndarray],
    perturbations: Sequence[Mapping[str, Any]],
) -> None:
    for spec in perturbations:
        signal_index = spec.get("alias_signal_index")
        ctx_index = spec.get("alias_ctx_cont_index")
        if signal_index is None and ctx_index is None:
            continue
        if signal_index is None or ctx_index is None:
            raise RuntimeError("FEATURE_USEFULNESS_ALIAS_OWNER_INCOMPLETE")
        seq = states["seq_signal"][:, -1, int(signal_index)]
        snap = states["snap_signal"][:, int(signal_index)]
        ctx = states["ctx_cont"][:, int(ctx_index)]
        if not np.array_equal(seq, snap) or not np.array_equal(snap, ctx):
            raise RuntimeError("FEATURE_USEFULNESS_ALIAS_SOURCE_OFF_MANIFOLD")


def _slice_states(states: Mapping[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {name: array[indices] for name, array in states.items()}


def _apply_perturbation(
    baseline: Mapping[str, np.ndarray],
    donor: Mapping[str, np.ndarray],
    spec: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    output = dict(baseline)
    cloned: dict[str, np.ndarray] = {}
    for target in spec["targets"]:
        surface = str(target["surface"])
        whole_surface = target.get("whole_surface") is True
        indices = (
            []
            if whole_surface
            else [int(index) for index in target["source_indices"]]
        )
        if surface not in output or (not whole_surface and not indices):
            raise RuntimeError("FEATURE_USEFULNESS_PERTURBATION_TARGET_INVALID")
        if surface not in cloned:
            cloned[surface] = np.array(output[surface], copy=True)
            output[surface] = cloned[surface]
        if whole_surface:
            output[surface][...] = donor[surface]
        else:
            output[surface][..., indices] = donor[surface][..., indices]
    signal_index = spec.get("alias_signal_index")
    ctx_index = spec.get("alias_ctx_cont_index")
    if signal_index is not None:
        if not np.array_equal(
            output["seq_signal"][:, -1, int(signal_index)],
            output["snap_signal"][:, int(signal_index)],
        ) or not np.array_equal(
            output["snap_signal"][:, int(signal_index)],
            output["ctx_cont"][:, int(ctx_index)],
        ):
            raise RuntimeError("FEATURE_USEFULNESS_ALIAS_PERTURBATION_OFF_MANIFOLD")
    return output


def _predict(
    *,
    states: Mapping[str, np.ndarray],
    predictor: Predictor,
    class_count: int,
    batch_rows: int,
    donor_indices: np.ndarray | None = None,
    perturbation: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, int]:
    row_count = next(iter(states.values())).shape[0]
    if batch_rows < 1:
        raise RuntimeError("FEATURE_USEFULNESS_BATCH_ROWS_INVALID")
    parts: list[np.ndarray] = []
    calls = 0
    for start in range(0, row_count, batch_rows):
        stop = min(row_count, start + batch_rows)
        rows = np.arange(start, stop, dtype=np.int64)
        batch = _slice_states(states, rows)
        if perturbation is not None:
            if donor_indices is None:
                raise RuntimeError("FEATURE_USEFULNESS_DONOR_REQUIRED")
            donor = _slice_states(states, donor_indices[rows])
            batch = _apply_perturbation(batch, donor, perturbation)
        outputs = np.asarray(predictor(batch), dtype=np.float64)
        calls += 1
        if outputs.shape != (len(rows), class_count) or not np.isfinite(outputs).all():
            raise RuntimeError("FEATURE_USEFULNESS_PREDICTOR_OUTPUTS_INVALID")
        parts.append(outputs)
    return np.concatenate(parts, axis=0), calls


def _fitted_q_loss_and_unique_target_margin(
    predicted_q_bps: np.ndarray,
    *,
    q_targets_bps: np.ndarray,
    action_valid_mask: np.ndarray,
    action_equivalence_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted_q_bps, dtype=np.float64)
    if predicted.ndim != 2:
        raise RuntimeError("FEATURE_USEFULNESS_FITTED_Q_PREDICTION_INVALID")
    row_count, action_count = predicted.shape
    valid = np.asarray(action_valid_mask, dtype=np.bool_)
    equivalent = np.asarray(action_equivalence_mask, dtype=np.bool_)
    q_targets = np.asarray(q_targets_bps, dtype=np.float64)
    if (
        action_count < 2
        or q_targets.shape != predicted.shape
        or valid.shape != predicted.shape
        or equivalent.shape != predicted.shape
        or not np.isfinite(predicted).all()
        or not np.isfinite(q_targets).all()
        or not valid.any(axis=1).all()
        or not equivalent.any(axis=1).all()
        or (equivalent & ~valid).any()
    ):
        raise RuntimeError("FEATURE_USEFULNESS_FITTED_Q_SUPERVISION_INVALID")
    masked_q = np.where(valid, q_targets, -np.inf)
    expected_equivalence = valid & np.equal(
        q_targets, np.max(masked_q, axis=1, keepdims=True)
    )
    if not np.array_equal(equivalent, expected_equivalence):
        raise RuntimeError("FEATURE_USEFULNESS_FITTED_Q_EQUIVALENCE_INVALID")
    squared_error = np.square(predicted - q_targets)
    loss = np.where(valid, squared_error, 0.0).sum(axis=1) / valid.sum(axis=1)
    alternative = valid & ~equivalent
    margin_mask = (equivalent.sum(axis=1) == 1) & alternative.any(axis=1)
    unique_target_score = np.max(
        np.where(equivalent[margin_mask], predicted[margin_mask], -np.inf),
        axis=1,
    )
    alternative_score = np.max(
        np.where(
            alternative[margin_mask], predicted[margin_mask], -np.inf
        ),
        axis=1,
    )
    margin = unique_target_score - alternative_score
    if not np.isfinite(loss).all() or not np.isfinite(margin).all():
        raise RuntimeError("FEATURE_USEFULNESS_FITTED_Q_METRIC_NONFINITE")
    return loss, margin, margin_mask


def _require_exit_fitted_q_state_binding(
    *,
    states: Mapping[str, np.ndarray],
    action_valid_mask: np.ndarray,
    terminal_mask: np.ndarray,
    row_times: pd.DatetimeIndex,
) -> None:
    """Bind terminal action masks to the exact full-state episode surface."""

    valid = np.asarray(action_valid_mask, dtype=np.bool_)
    terminal = np.asarray(terminal_mask, dtype=np.bool_)
    episode = np.asarray(states["exit_episode_index"], dtype=np.int64)
    side = np.asarray(states["exit_side_index"], dtype=np.int64)
    state_index = np.asarray(states["exit_state_index"], dtype=np.int64)
    row_count = len(state_index)
    if (
        valid.shape != (row_count, 2)
        or terminal.shape != (row_count,)
        or episode.shape != (row_count,)
        or side.shape != (row_count,)
        or len(row_times) != row_count
        or row_times.hasnans
        or np.any(state_index < 0)
        or not np.array_equal(terminal, valid[:, 1] & ~valid[:, 0])
        or np.any(~valid[:, 1])
    ):
        raise RuntimeError(
            "FEATURE_USEFULNESS_EXIT_FITTED_Q_STATE_BINDING_INVALID"
        )
    pairs = np.column_stack([episode, side])
    for pair in np.unique(pairs, axis=0):
        rows = np.flatnonzero((episode == pair[0]) & (side == pair[1]))
        terminal_rows = rows[terminal[rows]]
        if (
            terminal_rows.size != 1
            or state_index[terminal_rows[0]] != np.max(state_index[rows])
            or not np.array_equal(
                state_index[rows], np.arange(rows.size, dtype=np.int64)
            )
            or not row_times[rows].is_monotonic_increasing
            or not row_times[rows].is_unique
        ):
            raise RuntimeError(
                "FEATURE_USEFULNESS_EXIT_FITTED_Q_STATE_BINDING_INVALID"
            )


def _paired_summary(values: np.ndarray, *, domain: bytes) -> dict[str, Any]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size < 1:
        raise RuntimeError("FEATURE_USEFULNESS_PAIRED_VECTOR_INVALID")
    summary = StreamingPairedSummary(int(raw.size), domain)
    summary.update(raw)
    return summary.finalize()


def _effect_metric(
    *,
    physical_id: str,
    loss_summary: Mapping[str, Any],
    margin_summary: Mapping[str, Any],
) -> dict[str, Any]:
    interpretation = (
        "non_positive_mean_on_both_raw_paired_metrics"
        if loss_summary["mean"] <= 0.0 and margin_summary["mean"] <= 0.0
        else "mixed_or_positive_raw_paired_evidence"
    )
    return {
        "physical_id": physical_id,
        "paired_loss_delta": dict(loss_summary),
        "paired_margin_delta": dict(margin_summary),
        "interpretation": interpretation,
    }


class _FeatureUsefulnessMetrics:
    """Accumulate every effect in row order, retaining only one batch's vectors."""

    def __init__(
        self, *, task: str, task_layout: Mapping[str, Any],
        row_count: int, margin_row_count: int,
    ):
        self.task_layout = task_layout
        self.specs = tuple(
            spec
            for section in (
                "physical_field_perturbations", "family_tf_routes",
                "local_family_effects", "joint_effects", "exit_episode_effects",
            )
            for spec in task_layout[section]
        )
        self.component_ids = {
            str(spec["physical_id"])
            for section in ("family_tf_routes", "local_family_effects", "joint_effects")
            for spec in task_layout[section]
        }
        self.effects = {
            str(spec["physical_id"]): (
                StreamingPairedSummary(
                    row_count,
                    f"feature_usefulness_loss:{spec['physical_id']}\0".encode("utf-8"),
                ),
                StreamingPairedSummary(
                    margin_row_count,
                    f"feature_usefulness_margin:{spec['physical_id']}\0".encode("utf-8"),
                ),
            )
            for spec in self.specs
        }
        if len(self.effects) != len(self.specs):
            raise RuntimeError("FEATURE_USEFULNESS_EFFECT_ID_DUPLICATE")
        self.synergies = {
            str(row["token"]): (
                StreamingPairedSummary(
                    row_count,
                    f"feature_usefulness_synergy_loss:{task}:{row['token']}\0".encode("utf-8"),
                ),
                StreamingPairedSummary(
                    margin_row_count,
                    f"feature_usefulness_synergy_margin:{task}:{row['token']}\0".encode("utf-8"),
                ),
            )
            for row in task_layout["interaction_synergy"]
        }
        self._failed = False

    def update(
        self, *, baseline_outputs: np.ndarray,
        q_targets_bps: np.ndarray, action_valid_mask: np.ndarray,
        action_equivalence_mask: np.ndarray,
        predict_spec: Callable[[Mapping[str, Any]], np.ndarray],
    ) -> None:
        if self._failed:
            raise RuntimeError("FEATURE_USEFULNESS_METRIC_STREAM_FAILED")
        self._failed = True
        baseline_loss, baseline_margin, margin_mask = (
            _fitted_q_loss_and_unique_target_margin(
                baseline_outputs, q_targets_bps=q_targets_bps,
                action_valid_mask=action_valid_mask,
                action_equivalence_mask=action_equivalence_mask,
            )
        )
        component_vectors: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for spec in self.specs:
            physical_id = str(spec["physical_id"])
            loss, margin, observed_margin_mask = (
                _fitted_q_loss_and_unique_target_margin(
                    predict_spec(spec), q_targets_bps=q_targets_bps,
                    action_valid_mask=action_valid_mask,
                    action_equivalence_mask=action_equivalence_mask,
                )
            )
            if not np.array_equal(observed_margin_mask, margin_mask):
                raise RuntimeError("FEATURE_USEFULNESS_FITTED_Q_MARGIN_MASK_CHANGED")
            loss_delta = loss - baseline_loss
            margin_delta = baseline_margin - margin
            loss_summary, margin_summary = self.effects[physical_id]
            loss_summary.update(loss_delta)
            if margin_delta.size:
                margin_summary.update(margin_delta)
            if physical_id in self.component_ids:
                component_vectors[physical_id] = (loss_delta, margin_delta)
        for row in self.task_layout["interaction_synergy"]:
            left_loss, left_margin = component_vectors[str(row["left_effect_id"])]
            right_loss, right_margin = component_vectors[str(row["right_effect_id"])]
            joint_loss, joint_margin = component_vectors[str(row["joint_effect_id"])]
            loss_summary, margin_summary = self.synergies[str(row["token"])]
            loss_summary.update(joint_loss - left_loss - right_loss)
            if joint_margin.size:
                margin_summary.update(joint_margin - left_margin - right_margin)
        self._failed = False

    def finalize(self) -> dict[str, Any]:
        if self._failed:
            raise RuntimeError("FEATURE_USEFULNESS_METRIC_STREAM_FAILED")
        metrics = {
            physical_id: _effect_metric(
                physical_id=physical_id,
                loss_summary=summaries[0].finalize(),
                margin_summary=summaries[1].finalize(),
            )
            for physical_id, summaries in self.effects.items()
        }
        synergy_metrics = {}
        for row in self.task_layout["interaction_synergy"]:
            token = str(row["token"])
            loss_summary, margin_summary = self.synergies[token]
            synergy_metrics[token] = {
                "kind": row["kind"], "formula": row["formula"],
                "left_effect_id": row["left_effect_id"],
                "right_effect_id": row["right_effect_id"],
                "joint_effect_id": row["joint_effect_id"],
                "left_effect": metrics[str(row["left_effect_id"])],
                "right_effect": metrics[str(row["right_effect_id"])],
                "joint_effect": metrics[str(row["joint_effect_id"])],
                "paired_loss_delta": loss_summary.finalize(),
                "paired_margin_delta": margin_summary.finalize(),
            }
        return {
            "logical_field_metrics": {
                group: {
                    str(row["token"]): dict(metrics[str(row["physical_id"])])
                    for row in rows
                }
                for group, rows in self.task_layout["logical_fields"].items()
            },
            "family_tf_route_metrics": {
                str(spec["token"]): metrics[str(spec["physical_id"])]
                for spec in self.task_layout["family_tf_routes"]
            },
            "exit_episode_effect_metrics": {
                str(spec["token"]): metrics[str(spec["physical_id"])]
                for spec in self.task_layout["exit_episode_effects"]
            },
            "interaction_synergy_metrics": synergy_metrics,
        }


def audit_task_feature_usefulness(
    *,
    task: str,
    ordered_signal_names: Sequence[str],
    identity: Mapping[str, Any],
    states: Mapping[str, Any],
    row_times: Sequence[Any],
    row_splits: Sequence[str],
    block_ids: Sequence[Any],
    within_block_positions: Sequence[int],
    predictor: Predictor,
    entry_action_q_target_bps: Any | None = None,
    entry_action_valid_mask: Any | None = None,
    entry_action_equivalence_mask: Any | None = None,
    entry_fitted_q_iteration_state: Mapping[str, Any] | None = None,
    exit_fitted_q_iteration_state: Mapping[str, Any] | None = None,
    exit_action_q_target_bps: Any | None = None,
    exit_action_valid_mask: Any | None = None,
    exit_action_equivalence_mask: Any | None = None,
    exit_terminal_mask: Any | None = None,
    batch_rows: int = 256,
    native_mtf_geometry_by_block: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Measure one task on complete immutable VAL rows without selection."""

    if task not in TASKS:
        raise RuntimeError("FEATURE_USEFULNESS_TASK_INVALID")
    if native_mtf_geometry_by_block is not None and task != "exit":
        raise RuntimeError("FEATURE_USEFULNESS_DONOR_NATIVE_MTF_GEOMETRY_INVALID")
    checked_identity = require_feature_usefulness_identity(identity)
    if not isinstance(exit_fitted_q_iteration_state, Mapping) or (
        task == "entry" and not isinstance(entry_fitted_q_iteration_state, Mapping)
    ):
        raise RuntimeError(
            f"FEATURE_USEFULNESS_{task.upper()}_FITTED_Q_ITERATION_REQUIRED"
        )
    exit_iteration = require_unified_exit_fitted_q_iteration_state(
        exit_fitted_q_iteration_state,
        context=(
            "FEATURE_USEFULNESS_ENTRY_EXIT_TEACHER"
            if task == "entry" else "FEATURE_USEFULNESS_EXIT"
        ),
    )
    entry_iteration = (
        require_entry_fitted_q_iteration_state(
            entry_fitted_q_iteration_state,
            exit_fitted_q_iteration_state=exit_iteration,
            context="FEATURE_USEFULNESS_ENTRY",
        )
        if task == "entry" else None
    )
    require_feature_usefulness_teacher_identity(checked_identity, exit_iteration)
    layout = feature_usefulness_layout(ordered_signal_names)
    task_layout = layout["tasks"][task]
    times = pd.DatetimeIndex(pd.to_datetime(row_times, utc=True, errors="raise"))
    row_count = len(times)
    if (
        row_count < 2
        or times.hasnans
        or row_count
        != checked_identity[f"{task}_val_population_row_count"]
    ):
        raise RuntimeError("FEATURE_USEFULNESS_ROW_CLOCK_INVALID")
    if task == "entry" and (
        not times.is_monotonic_increasing or not times.is_unique
    ):
        raise RuntimeError("FEATURE_USEFULNESS_ENTRY_ROW_CLOCK_ORDER_INVALID")
    if len(row_splits) != row_count or any(str(value) != SPLIT for value in row_splits):
        raise RuntimeError("FEATURE_USEFULNESS_NON_VAL_ROW_FORBIDDEN")
    val_start = pd.Timestamp(checked_identity["val_start_utc"]).tz_convert("UTC")
    val_end = pd.Timestamp(checked_identity["val_end_utc"]).tz_convert("UTC")
    if (times < val_start).any() or (times > val_end).any():
        raise RuntimeError("FEATURE_USEFULNESS_FUTURE_OR_OUTSIDE_VAL_ROW_FORBIDDEN")
    class_count = len(TASK_CLASS_ORDER[task])
    arrays = _require_state_surfaces(
        states,
        task=task,
        task_layout=task_layout,
        row_count=row_count,
        timeframes=task_layout["timeframes"],
    )
    _require_alias_manifold(
        arrays,
        task_layout["physical_field_perturbations"],
    )
    donor, donor_plan = build_structure_preserving_donor_plan(
        block_ids=block_ids,
        within_block_positions=within_block_positions,
        native_mtf_geometry_by_block=native_mtf_geometry_by_block,
    )
    side_pair_indices: np.ndarray | None = None
    side_pair_plan: dict[str, Any] | None = None
    if task == "exit":
        side_pair_indices, side_pair_plan = _build_exit_side_pair_plan(arrays)
        if not np.array_equal(times.asi8, times.asi8[side_pair_indices]):
            raise RuntimeError("FEATURE_USEFULNESS_EXIT_SIDE_PAIR_CLOCK_MISMATCH")
    baseline_outputs, forward_calls = _predict(
        states=arrays,
        predictor=predictor,
        class_count=class_count,
        batch_rows=batch_rows,
    )
    if task == "entry":
        if any(
            value is not None
            for value in (
                exit_action_q_target_bps, exit_action_valid_mask,
                exit_action_equivalence_mask, exit_terminal_mask,
            )
        ):
            raise RuntimeError("FEATURE_USEFULNESS_ENTRY_EXIT_TARGETS_FORBIDDEN")
        q_targets = np.asarray(entry_action_q_target_bps, dtype=np.float64)
        action_valid = np.asarray(entry_action_valid_mask, dtype=np.bool_)
        action_equivalent = np.asarray(
            entry_action_equivalence_mask, dtype=np.bool_
        )
        baseline_loss, baseline_margin, margin_mask = (
            _fitted_q_loss_and_unique_target_margin(
                baseline_outputs,
                q_targets_bps=q_targets,
                action_valid_mask=action_valid,
                action_equivalence_mask=action_equivalent,
            )
        )
        if not action_valid[:, 2].all():
            raise RuntimeError("FEATURE_USEFULNESS_ENTRY_FLAT_MASK_INVALID")
        supervision: dict[str, Any] = {
            "schema_version": ENTRY_FITTED_Q_SCHEMA_VERSION,
            "fitted_q_contract": entry_fitted_q_contract(),
            "fitted_q_iteration_state": entry_iteration,
            "fitted_q_iteration_state_sha256": canonical_json_sha256(
                entry_iteration
            ),
            "exit_fitted_q_iteration_state": exit_iteration,
            "exit_fitted_q_iteration_state_sha256": canonical_json_sha256(
                exit_iteration
            ),
            "q_targets_bps_sha256": _array_sha256(
                np.ascontiguousarray(q_targets, dtype="<f8"),
                domain=b"feature_usefulness_entry_fitted_q_target_v1\0",
            ),
            "action_valid_mask_sha256": _array_sha256(
                np.ascontiguousarray(action_valid, dtype="u1"),
                domain=b"feature_usefulness_entry_action_valid_v1\0",
            ),
            "action_equivalence_mask_sha256": _array_sha256(
                np.ascontiguousarray(action_equivalent, dtype="u1"),
                domain=b"feature_usefulness_entry_action_equivalence_v1\0",
            ),
            "loss_valid_row_count": row_count,
            "margin_valid_row_count": int(margin_mask.sum()),
            "action_valid_cell_count": int(action_valid.sum()),
            "target_tied_row_count": int(
                (action_equivalent.sum(axis=1) > 1).sum()
            ),
            "single_valid_action_row_count": int(
                (action_valid.sum(axis=1) == 1).sum()
            ),
        }
    else:
        if any(
            value is not None
            for value in (
                entry_action_q_target_bps,
                entry_action_valid_mask,
                entry_action_equivalence_mask,
                entry_fitted_q_iteration_state,
            )
        ):
            raise RuntimeError("FEATURE_USEFULNESS_EXIT_ENTRY_TARGETS_FORBIDDEN")
        q_targets = np.asarray(exit_action_q_target_bps, dtype=np.float64)
        action_valid = np.asarray(exit_action_valid_mask, dtype=np.bool_)
        action_equivalent = np.asarray(
            exit_action_equivalence_mask, dtype=np.bool_
        )
        terminal = np.asarray(exit_terminal_mask, dtype=np.bool_)
        _require_exit_fitted_q_state_binding(
            states=arrays,
            action_valid_mask=action_valid,
            terminal_mask=terminal,
            row_times=times,
        )
        baseline_loss, baseline_margin, margin_mask = (
            _fitted_q_loss_and_unique_target_margin(
                baseline_outputs,
                q_targets_bps=q_targets,
                action_valid_mask=action_valid,
                action_equivalence_mask=action_equivalent,
            )
        )
        supervision = {
            "schema_version": UNIFIED_EXIT_FITTED_Q_SCHEMA_VERSION,
            "fitted_q_contract": unified_exit_fitted_q_contract(),
            "fitted_q_iteration_state": exit_iteration,
            "fitted_q_iteration_state_sha256": canonical_json_sha256(
                exit_iteration
            ),
            "q_targets_bps_sha256": _array_sha256(
                np.ascontiguousarray(q_targets, dtype="<f8"),
                domain=b"feature_usefulness_exit_fitted_q_bellman_target_v1\0",
            ),
            "action_valid_mask_sha256": _array_sha256(
                np.ascontiguousarray(action_valid, dtype="u1"),
                domain=b"feature_usefulness_exit_action_valid_v1\0",
            ),
            "action_equivalence_mask_sha256": _array_sha256(
                np.ascontiguousarray(action_equivalent, dtype="u1"),
                domain=b"feature_usefulness_exit_action_equivalence_v1\0",
            ),
            "terminal_mask_sha256": _array_sha256(
                np.ascontiguousarray(terminal, dtype="u1"),
                domain=b"feature_usefulness_exit_terminal_v1\0",
            ),
            "loss_valid_row_count": row_count,
            "margin_valid_row_count": int(margin_mask.sum()),
            "action_valid_cell_count": int(action_valid.sum()),
            "target_tied_row_count": int(
                (action_equivalent.sum(axis=1) > 1).sum()
            ),
            "single_valid_action_row_count": int(
                (action_valid.sum(axis=1) == 1).sum()
            ),
            "terminal_row_count": int(terminal.sum()),
        }

    metrics = _FeatureUsefulnessMetrics(
        task=task, task_layout=task_layout, row_count=row_count,
        margin_row_count=int(margin_mask.sum()),
    )
    for start in range(0, row_count, batch_rows):
        stop = min(row_count, start + batch_rows)
        rows = np.arange(start, stop, dtype=np.int64)
        batch = _slice_states(arrays, rows)
        donor_batch = _slice_states(arrays, donor[rows])
        side_batch = (
            None if side_pair_indices is None
            else _slice_states(arrays, side_pair_indices[rows])
        )

        def predict_spec(spec: Mapping[str, Any]) -> np.ndarray:
            nonlocal forward_calls
            donor_kind = str(spec.get("donor_kind", "structure_block"))
            selected_donor = donor_batch
            if donor_kind == "same_state_opposite_side":
                if side_batch is None:
                    raise RuntimeError("FEATURE_USEFULNESS_EXIT_SIDE_PAIR_REQUIRED")
                selected_donor = side_batch
            elif donor_kind != "structure_block":
                raise RuntimeError("FEATURE_USEFULNESS_DONOR_KIND_INVALID")
            outputs = np.asarray(
                predictor(_apply_perturbation(batch, selected_donor, spec)),
                dtype=np.float64,
            )
            forward_calls += 1
            if outputs.shape != (len(rows), class_count) or not np.isfinite(outputs).all():
                raise RuntimeError("FEATURE_USEFULNESS_PREDICTOR_OUTPUTS_INVALID")
            return outputs

        metrics.update(
            baseline_outputs=baseline_outputs[start:stop],
            q_targets_bps=q_targets[start:stop],
            action_valid_mask=action_valid[start:stop],
            action_equivalence_mask=action_equivalent[start:stop],
            predict_spec=predict_spec,
        )
    metric_payload = metrics.finalize()

    return _usefulness_task_payload(
        task=task, ordered_signal_names=ordered_signal_names, task_layout=task_layout,
        row_count=row_count, row_times_sha256=_array_sha256(
            np.ascontiguousarray(times.asi8, dtype="<i8"),
            domain=f"feature_usefulness_times:{task}\0".encode("utf-8"),
        ),
        supervision=supervision, baseline_outputs_sha256=_array_sha256(
            np.ascontiguousarray(baseline_outputs, dtype="<f8"),
            domain=f"feature_usefulness_baseline_outputs:{task}\0".encode("utf-8"),
        ),
        donor_plan=donor_plan, side_pair_plan=side_pair_plan,
        metric_payload=metric_payload, forward_calls=forward_calls,
    )


def _native_task_supervision(
    *, task: str, baseline: NativeUsefulnessBaseline,
    selected_pair: SelectedNativeExitPair, chunk_rows: int,
) -> tuple[dict[str, Any], str]:
    """Stream report hashes/counts through the shared fitted-Q metric owner."""

    arrays = baseline.arrays
    width = len(TASK_CLASS_ORDER[task])
    outputs = arrays[f"{task}_q_bps"].reshape(-1, width)
    targets = arrays[f"{task}_targets_bps"].reshape(-1, width)
    valid = arrays[f"{task}_valid"].reshape(-1, width)
    equivalent = arrays[f"{task}_equivalent"].reshape(-1, width)
    row_count = len(outputs)
    exit_iteration = require_unified_exit_fitted_q_iteration_state(
        selected_pair.metadata["unified_exit_training_evidence"]["selected_fitted_q_iteration_state"],
        context="FEATURE_USEFULNESS_NATIVE",
    )
    require_feature_usefulness_teacher_identity(baseline.identity, exit_iteration)
    iteration = (
        require_entry_fitted_q_iteration_state(
            selected_pair.metadata["selected_entry_fitted_q_iteration_state"],
            exit_fitted_q_iteration_state=exit_iteration,
            context="FEATURE_USEFULNESS_NATIVE_ENTRY",
        ) if task == "entry" else exit_iteration
    )
    supervision = {
        "schema_version": ENTRY_FITTED_Q_SCHEMA_VERSION if task == "entry" else UNIFIED_EXIT_FITTED_Q_SCHEMA_VERSION,
        "fitted_q_contract": entry_fitted_q_contract() if task == "entry" else unified_exit_fitted_q_contract(),
        "fitted_q_iteration_state": iteration,
        "fitted_q_iteration_state_sha256": canonical_json_sha256(iteration),
        "loss_valid_row_count": row_count, "margin_valid_row_count": 0,
        "action_valid_cell_count": 0, "target_tied_row_count": 0,
        "single_valid_action_row_count": 0,
    }
    target_domain = (
        b"feature_usefulness_entry_fitted_q_target_v1\0" if task == "entry"
        else b"feature_usefulness_exit_fitted_q_bellman_target_v1\0"
    )
    streams = {
        "q_targets_bps_sha256": (targets, "<f8", target_domain),
        "action_valid_mask_sha256": (valid, "u1", f"feature_usefulness_{task}_action_valid_v1\0".encode()),
        "action_equivalence_mask_sha256": (equivalent, "u1", f"feature_usefulness_{task}_action_equivalence_v1\0".encode()),
        "baseline_outputs_sha256": (outputs, "<f8", f"feature_usefulness_baseline_outputs:{task}\0".encode()),
    }
    if task == "entry":
        if not valid[:, 2].all():
            raise RuntimeError("FEATURE_USEFULNESS_ENTRY_FLAT_MASK_INVALID")
        supervision.update({
            "exit_fitted_q_iteration_state": exit_iteration,
            "exit_fitted_q_iteration_state_sha256": canonical_json_sha256(exit_iteration),
        })
    else:
        terminal = arrays["exit_terminal"].reshape(-1)
        streams["terminal_mask_sha256"] = (terminal, "u1", b"feature_usefulness_exit_terminal_v1\0")
        supervision["terminal_row_count"] = int(terminal.sum())
    digests = {
        name: StreamingArrayDigest(value.shape, np.dtype(dtype), domain)
        for name, (value, dtype, domain) in streams.items()
    }
    for start in range(0, row_count, chunk_rows):
        stop = min(row_count, start + chunk_rows)
        _loss, _margin, margin_mask = _fitted_q_loss_and_unique_target_margin(
            outputs[start:stop], q_targets_bps=targets[start:stop],
            action_valid_mask=valid[start:stop], action_equivalence_mask=equivalent[start:stop],
        )
        supervision["margin_valid_row_count"] += int(margin_mask.sum())
        supervision["action_valid_cell_count"] += int(valid[start:stop].sum())
        supervision["target_tied_row_count"] += int((equivalent[start:stop].sum(axis=1) > 1).sum())
        supervision["single_valid_action_row_count"] += int((valid[start:stop].sum(axis=1) == 1).sum())
        for name, (value, dtype, _domain) in streams.items():
            digests[name].update(np.ascontiguousarray(value[start:stop], dtype=dtype))
    hashes = {name: digest.finalize() for name, digest in digests.items()}
    output_sha256 = hashes.pop("baseline_outputs_sha256")
    return {**supervision, **hashes}, output_sha256


def _usefulness_task_payload(
    *, task: str, ordered_signal_names: Sequence[str], task_layout: Mapping[str, Any],
    row_count: int, row_times_sha256: str, baseline_outputs_sha256: str,
    supervision: Mapping[str, Any], donor_plan: Mapping[str, Any],
    side_pair_plan: Mapping[str, Any] | None, metric_payload: Mapping[str, Any],
    forward_calls: int,
) -> dict[str, Any]:
    return {
        "ordered_signal_names": list(ordered_signal_names), "row_count": row_count,
        "class_order": list(TASK_CLASS_ORDER[task]),
        "comparison_surface": (
            "raw_entry_action_q_bps_valid_action_masked_mse_and_unique_target_q_margin"
            if task == "entry" else
            "raw_exit_action_q_bps_frozen_fitted_q_bellman_target_masked_mse_and_unique_target_q_margin"
        ),
        "row_times_sha256": row_times_sha256, "supervision": dict(supervision),
        "baseline_outputs_sha256": baseline_outputs_sha256,
        "donor_plan": dict(donor_plan),
        "side_pair_plan": None if side_pair_plan is None else dict(side_pair_plan),
        "forward_variant_count": 1 + sum(len(task_layout[section]) for section in (
            "physical_field_perturbations", "family_tf_routes", "local_family_effects",
            "joint_effects", "exit_episode_effects",
        )),
        **metric_payload,
        "coverage": {
            **task_layout["coverage_counts"],
            "reported_logical_fields": sum(len(rows) for rows in task_layout["logical_fields"].values()),
            "reported_family_tf_routes": len(metric_payload["family_tf_route_metrics"]),
            "reported_exit_episode_effects": len(metric_payload["exit_episode_effect_metrics"]),
            "reported_interaction_synergy": len(metric_payload["interaction_synergy_metrics"]),
            "omitted_tokens": [], "complete": True,
        },
        "_forward_batch_calls": forward_calls,
    }


def audit_native_feature_usefulness(
    *, selected_pair: SelectedNativeExitPair, inputs: NativeVALInputs,
    repo: Path, device: Any, batch_rows: int, max_baseline_bytes: int,
    max_episode_bytes: int, max_forward_calls: int,
    created: datetime | None = None,
) -> dict[str, Any]:
    """Measure the complete layout without expanding native feature histories.

    This internal serial consumer grants no execution or publication authority.
    An explicit call budget must cover the entire declared layout before any
    baseline forward. A smaller budget rejects; it never samples or publishes
    a partial report. Real runtime and process memory remain unmeasured here.
    """

    import torch
    from gx1.contracts import unified_exit_episode_pack_v1 as episode_owner
    from gx1.scripts import entry_exit_feature_usefulness_native_v1 as native

    for value in (batch_rows, max_baseline_bytes, max_episode_bytes, max_forward_calls):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise RuntimeError("FEATURE_USEFULNESS_NATIVE_AUDIT_BUDGET_INVALID")
    if not isinstance(inputs, NativeVALInputs):
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_VAL_INPUTS_REQUIRED")
    dataset = inputs.dataset
    lifecycle = inputs.corpus.splits["val"]
    layout = feature_usefulness_layout(dataset.signal_names)
    entry_rows = len(dataset)
    pair_count = len({entry for entry, _side in lifecycle._episode_pointers})
    entry_batches = (entry_rows + batch_rows - 1) // batch_rows
    spec_counts = {
        task: sum(len(layout["tasks"][task][section]) for section in (
            "physical_field_perturbations", "family_tf_routes", "local_family_effects",
            "joint_effects", "exit_episode_effects",
        )) for task in TASKS
    }
    required_calls = entry_batches * (2 + spec_counts["entry"]) + pair_count * (2 + spec_counts["exit"])
    if required_calls > max_forward_calls:
        raise RuntimeError(f"FEATURE_USEFULNESS_NATIVE_FORWARD_BUDGET_EXCEEDED: required={required_calls}")
    baseline = collect_native_usefulness_baseline(
        selected_pair=selected_pair, inputs=inputs, repo=repo, device=device,
        batch_rows=batch_rows, max_baseline_bytes=max_baseline_bytes,
        max_episode_bytes=max_episode_bytes,
    )
    arrays = baseline.arrays
    rows_per_pair = episode_owner.UNIFIED_EXIT_EPISODE_SIDE_COUNT * episode_owner.UNIFIED_EXIT_EPISODE_STATE_COUNT
    supervision = {}
    output_hashes = {}
    metrics = {}
    for task in TASKS:
        supervision[task], output_hashes[task] = _native_task_supervision(
            task=task, baseline=baseline, selected_pair=selected_pair,
            chunk_rows=batch_rows if task == "entry" else rows_per_pair,
        )
        metrics[task] = _FeatureUsefulnessMetrics(
            task=task, task_layout=layout["tasks"][task],
            row_count=baseline.identity[f"{task}_val_population_row_count"],
            margin_row_count=supervision[task]["margin_valid_row_count"],
        )
    entry_donor, entry_plan = build_structure_preserving_donor_plan(
        block_ids=dataset.indices, within_block_positions=np.zeros(entry_rows, dtype=np.int64),
    )
    calls = {"entry": entry_batches, "exit": pair_count}
    for start in range(0, entry_rows, batch_rows):
        stop = min(entry_rows, start + batch_rows)
        rows = np.arange(start, stop, dtype=np.int64)
        donors = entry_donor[rows]
        batch = _native_entry_batch(dataset, rows, task_layout=layout["tasks"]["entry"])
        donor_batch = _native_entry_batch(dataset, donors, task_layout=layout["tasks"]["entry"])
        for indices, values in ((rows, batch), (donors, donor_batch)):
            if _native_entry_input_digests(values) != tuple(baseline.entry_input_sha256[int(index)] for index in indices):
                raise RuntimeError("FEATURE_USEFULNESS_NATIVE_ENTRY_INPUT_CHANGED")

        def predict_entry(spec: Mapping[str, Any]) -> np.ndarray:
            outputs, _tokens = _native_entry_forward(
                model=selected_pair.model, batch=_apply_perturbation(batch, donor_batch, spec), device=device,
            )
            calls["entry"] += 1
            return outputs

        metrics["entry"].update(
            baseline_outputs=arrays["entry_q_bps"][start:stop],
            q_targets_bps=arrays["entry_targets_bps"][start:stop],
            action_valid_mask=arrays["entry_valid"][start:stop],
            action_equivalence_mask=arrays["entry_equivalent"][start:stop], predict_spec=predict_entry,
        )
    adapter = native.CompactNativeExitUsefulnessAdapter(
        model=selected_pair.model, ordered_signal_names=dataset.signal_names,
        device=device, max_episode_bytes=max_episode_bytes,
    )

    def read_episode(entry: int) -> Mapping[str, Any]:
        episode = dataset.materialize_full_exit_episode(entry)
        if episode is None:
            raise RuntimeError("FEATURE_USEFULNESS_NATIVE_EXIT_EPISODE_CHANGED")
        readiness = episode["unbounded_exit_training_readiness"]
        episode = episode_owner.require_unified_exit_episode_pack(
            {key: value for key, value in episode.items()
             if key != "unbounded_exit_training_readiness"},
            per_tf_seq_lens=dataset.per_tf_seq_lens,
            expected_mtf_cache_identity_sha256=dataset._multi_tf_cache_identity_sha256,
            context="FEATURE_USEFULNESS_NATIVE_INTERVENTION",
        )
        episode["unbounded_exit_training_readiness"] = readiness
        if episode["entry_row_index"] != entry or episode["episode_pack_sha256"] != baseline.episode_pack_sha256[entry]:
            raise RuntimeError("FEATURE_USEFULNESS_NATIVE_EXIT_EPISODE_CHANGED")
        return episode

    for pair_index, raw_entry in enumerate(arrays["eligible_entry_indices"]):
        entry = int(raw_entry)
        donor_entry = int(arrays["eligible_entry_indices"][baseline.donor_indices[pair_index]])
        episode, donor_episode = read_episode(entry), read_episode(donor_entry)
        token = torch.from_numpy(arrays["online_tokens"][entry:entry + 1].copy()).to(device)
        donor_token = torch.from_numpy(arrays["online_tokens"][donor_entry:donor_entry + 1].copy()).to(device)

        def predict_exit(spec: Mapping[str, Any]) -> np.ndarray:
            outputs = adapter.predict_spec(
                episode=episode, online_entry_token=token, spec=spec,
                donor_episode=donor_episode, donor_online_entry_token=donor_token,
            )
            calls["exit"] += 1
            return outputs.reshape(rows_per_pair, len(TASK_CLASS_ORDER["exit"]))

        metrics["exit"].update(
            baseline_outputs=arrays["exit_q_bps"][pair_index].reshape(rows_per_pair, -1),
            q_targets_bps=arrays["exit_targets_bps"][pair_index].reshape(rows_per_pair, -1),
            action_valid_mask=arrays["exit_valid"][pair_index].reshape(rows_per_pair, -1),
            action_equivalence_mask=arrays["exit_equivalent"][pair_index].reshape(rows_per_pair, -1),
            predict_spec=predict_exit,
        )
    if sum(calls.values()) + entry_batches + pair_count != required_calls:
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_FORWARD_COUNT_MISMATCH")
    if set(lifecycle._episode_pointers) != {
        (int(entry), side) for entry in arrays["eligible_entry_indices"]
        for side in range(episode_owner.UNIFIED_EXIT_EPISODE_SIDE_COUNT)
    }:
        raise RuntimeError("FEATURE_USEFULNESS_NATIVE_EXIT_POPULATION_CHANGED")
    _require_native_val_population_unchanged(inputs)
    native.require_selected_native_pair_unchanged(selected_pair=selected_pair, repo=repo)
    require_native_val_inputs_unchanged(inputs)
    exit_times = StreamingArrayDigest(
        (pair_count * rows_per_pair,), np.dtype("<i8"), b"feature_usefulness_times:exit\0",
    )
    for times in arrays["exit_decision_time_ns"]:
        for _side in range(episode_owner.UNIFIED_EXIT_EPISODE_SIDE_COUNT):
            exit_times.update(times)
    clock_hashes = {
        "entry": _array_sha256(inputs.entry_decision_time_ns, domain=b"feature_usefulness_times:entry\0"),
        "exit": exit_times.finalize(),
    }
    tasks = {
        task: _usefulness_task_payload(
            task=task, ordered_signal_names=dataset.signal_names, task_layout=layout["tasks"][task],
            row_count=baseline.identity[f"{task}_val_population_row_count"],
            row_times_sha256=clock_hashes[task], baseline_outputs_sha256=output_hashes[task],
            supervision=supervision[task], donor_plan=entry_plan if task == "entry" else baseline.donor_plan,
            side_pair_plan=None if task == "entry" else baseline.side_pair_plan,
            metric_payload=metrics[task].finalize(), forward_calls=calls[task],
        ) for task in TASKS
    }
    return build_feature_usefulness_report(
        identity=baseline.identity, ordered_signal_names=dataset.signal_names,
        entry_task=tasks["entry"], exit_task=tasks["exit"], created=created,
    )


def build_feature_usefulness_report(
    *,
    identity: Mapping[str, Any],
    ordered_signal_names: Sequence[str],
    entry_task: Mapping[str, Any],
    exit_task: Mapping[str, Any],
    created: datetime | None = None,
) -> dict[str, Any]:
    checked_identity = require_feature_usefulness_identity(identity)
    layout = feature_usefulness_layout(ordered_signal_names)
    tasks: dict[str, Any] = {}
    for task, raw in (("entry", entry_task), ("exit", exit_task)):
        row = dict(raw)
        row.pop("_forward_batch_calls", None)
        tasks[task] = row
    created = created or datetime.now(timezone.utc)
    if created.tzinfo is None or created.utcoffset() is None:
        raise RuntimeError("FEATURE_USEFULNESS_CREATED_UTC_INVALID")
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created.astimezone(timezone.utc).isoformat(),
        "decision": DECISION,
        "split": SPLIT,
        "policy": dict(POLICY),
        "perturbation_policy": dict(PERTURBATION_POLICY),
        "identity": checked_identity,
        "identity_sha256": canonical_json_sha256(checked_identity),
        "layout_sha256": layout["layout_sha256"],
        "layout_counts": {
            task: layout["tasks"][task]["coverage_counts"] for task in TASKS
        },
        "tasks": tasks,
        "test_rows_read": False,
        "test_artifacts_read": [],
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return require_feature_usefulness_report(report)


def write_immutable_feature_usefulness_report(
    path: Path,
    report: Mapping[str, Any],
) -> Path:
    checked = require_feature_usefulness_report(report)
    out = path.expanduser()
    if out.is_symlink():
        raise RuntimeError(f"FEATURE_USEFULNESS_OUTPUT_EXISTS: {out}")
    out = out.resolve()
    if out.exists():
        raise RuntimeError(f"FEATURE_USEFULNESS_OUTPUT_EXISTS: {out}")
    atomic_write_text(
        out,
        json.dumps(checked, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return out


def _native_execution_paths(args: argparse.Namespace, *, repo: Path) -> None:
    for name in ("bundle_dir", "recipe_audit_json", "out_json"):
        path = getattr(args, name)
        if (
            not isinstance(path, Path) or not path.is_absolute()
            or ".." in path.parts
            or any(part.is_symlink() for part in (path, *path.parents))
        ):
            raise RuntimeError(f"FEATURE_USEFULNESS_EXECUTION_PATH_INVALID:{name}")
    if not args.bundle_dir.is_dir() or not args.recipe_audit_json.is_file():
        raise RuntimeError("FEATURE_USEFULNESS_EXECUTION_INPUT_PATH_INVALID")
    if (
        args.out_json.exists() or not args.out_json.parent.is_dir()
        or any(args.out_json.is_relative_to(root) for root in (repo, args.bundle_dir))
        or next(args.out_json.parent.iterdir(), None) is not None
    ):
        raise RuntimeError("FEATURE_USEFULNESS_EXECUTION_OUTPUT_REQUIRES_EMPTY_EXTERNAL_DIRECTORY")


def _set_native_cpu_numerics(seed: int) -> None:
    import torch

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise RuntimeError("FEATURE_USEFULNESS_EXECUTION_SEED_INVALID")
    if torch.get_default_dtype() != torch.float32 or torch.is_autocast_enabled("cpu"):
        raise RuntimeError("FEATURE_USEFULNESS_EXECUTION_FP32_REQUIRED")
    torch.set_num_threads(1)
    torch.default_generator.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    _require_native_cpu_numerics()


def _require_native_cpu_numerics() -> None:
    import torch

    if (
        torch.get_num_threads() != 1 or torch.get_default_dtype() != torch.float32
        or torch.is_autocast_enabled("cpu")
        or not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
    ):
        raise RuntimeError("FEATURE_USEFULNESS_EXECUTION_NUMERICS_CHANGED")


def _execute_native_feature_usefulness(args: argparse.Namespace, *, repo: Path) -> Path:
    from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution
    from gx1.contracts.entry_model_native_train_launch_v1 import (
        _require_training_review_hold_cleared,
    )

    _native_execution_paths(args, repo=repo)
    require_capped_cpu_audit_execution()
    _require_training_review_hold_cleared(repo)

    import torch
    from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
        require_pretest_technical_recipe_metadata,
    )
    from gx1.scripts import entry_exit_feature_usefulness_native_v1 as native
    from gx1.scripts.evaluate_entry_candidate_selective_edge_v1 import (
        _bundle_core_integrity_snapshot,
    )

    session_dir = args.bundle_dir.parent / (
        native.trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + args.bundle_dir.name
    )
    if args.out_json.is_relative_to(session_dir):
        raise RuntimeError("FEATURE_USEFULNESS_EXECUTION_OUTPUT_INSIDE_SESSION")
    recipe = require_pretest_technical_recipe_metadata(
        native._selected_json(args.recipe_audit_json, args.recipe_audit_sha256),
        expected_profile="candidate", expected_out_bundle_dir=args.bundle_dir,
    )
    native.require_training_recipe_source_provenance(
        recipe_audit_path=args.recipe_audit_json, recipe_audit_sha256=args.recipe_audit_sha256,
        repo=repo, profile="candidate", run_id=recipe["run_id"],
        dataset_run_id=recipe["dataset_run_id"], dataset_dir=Path(recipe["dataset_dir"]),
        out_bundle_dir=args.bundle_dir,
    )
    metadata = native._selected_json(
        args.bundle_dir / "bundle_metadata.json", args.bundle_metadata_sha256,
    )
    bundle_snapshot = _bundle_core_integrity_snapshot(bundle_dir=args.bundle_dir, bundle_metadata=metadata)
    _set_native_cpu_numerics(recipe["trainer_cli"]["seed"])
    selected = native.load_selected_native_exit_pair(
        bundle_dir=args.bundle_dir, bundle_metadata_sha256=args.bundle_metadata_sha256,
        session_out_bundle_dir=args.bundle_dir, session_contract_sha256=args.session_contract_sha256,
        active_pointer_sha256=args.active_pointer_sha256,
        selected_checkpoint_sha256=args.selected_checkpoint_sha256,
        recipe_audit_path=args.recipe_audit_json, recipe_audit_sha256=args.recipe_audit_sha256,
        repo=repo,
    )
    if selected.bindings["bundle_commit_sha256"] != bundle_snapshot["bundle_commit_declared_sha256"]:
        raise RuntimeError("FEATURE_USEFULNESS_EXECUTION_SELECTED_BUNDLE_CHANGED")
    with open_native_val_inputs(
        selected_pair=selected, recipe_audit_path=args.recipe_audit_json,
        recipe_audit_sha256=args.recipe_audit_sha256,
    ) as inputs:
        with torch.autocast("cpu", enabled=False):
            report = audit_native_feature_usefulness(
                selected_pair=selected, inputs=inputs, repo=repo, device=torch.device("cpu"),
                batch_rows=args.batch_size, max_baseline_bytes=args.max_baseline_bytes,
                max_episode_bytes=args.max_episode_bytes, max_forward_calls=args.max_forward_calls,
            )
        require_native_val_inputs_unchanged(inputs)
    native.require_selected_native_pair_unchanged(selected_pair=selected, repo=repo)
    if _bundle_core_integrity_snapshot(bundle_dir=args.bundle_dir, bundle_metadata=metadata) != bundle_snapshot:
        raise RuntimeError("FEATURE_USEFULNESS_EXECUTION_BUNDLE_CHANGED")
    require_capped_cpu_audit_execution()
    _require_training_review_hold_cleared(repo)
    _require_native_cpu_numerics()
    _native_execution_paths(args, repo=repo)
    return write_immutable_feature_usefulness_report(args.out_json, report)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate or explicitly produce one complete CPU VAL-only usefulness report",
        allow_abbrev=False,
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-json", type=Path)
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--device", choices=("cpu",))
    for name in ("bundle-dir", "recipe-audit-json", "out-json"):
        parser.add_argument("--" + name, type=Path)
    for name in (
        "bundle-metadata-sha256", "session-contract-sha256", "active-pointer-sha256",
        "selected-checkpoint-sha256", "recipe-audit-sha256",
    ):
        parser.add_argument("--" + name)
    for name in ("batch-size", "max-baseline-bytes", "max-episode-bytes", "max-forward-calls"):
        parser.add_argument("--" + name, type=int)
    arguments = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in arguments if value.startswith("--")]
    if len(flags) != len(set(flags)):
        parser.error("duplicate arguments are forbidden")
    args = parser.parse_args(arguments)
    production = {key: value for key, value in vars(args).items() if key not in {"execute", "validate_json"}}
    if args.execute:
        for name, value in production.items():
            if value is None or value == "":
                parser.error(f"--execute requires explicit --{name.replace('_', '-')}")
            if name.endswith("sha256") and (len(value) != 64 or any(char not in "0123456789abcdef" for char in value)):
                parser.error(f"invalid SHA-256: --{name.replace('_', '-')}")
            if isinstance(value, int) and value < 1:
                parser.error(f"positive budget required: --{name.replace('_', '-')}")
        path = _execute_native_feature_usefulness(args, repo=Path(__file__).resolve().parents[2])
        print(json.dumps({"decision": DECISION, "path": str(path), "device": "cpu"}, sort_keys=True))
        return 0
    if any(value is not None for value in production.values()):
        parser.error("--validate-json does not accept execution arguments")
    path = args.validate_json.expanduser().resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    require_feature_usefulness_report(value)
    print(json.dumps({"decision": DECISION, "path": str(path)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "NativeVALInputs",
    "NativeUsefulnessBaseline",
    "collect_native_usefulness_baseline",
    "audit_native_feature_usefulness",
    "open_native_val_inputs",
    "require_native_val_inputs_unchanged",
    "audit_task_feature_usefulness",
    "build_native_exit_structure_plans",
    "build_feature_usefulness_report",
    "build_structure_preserving_donor_plan",
    "write_immutable_feature_usefulness_report",
]
