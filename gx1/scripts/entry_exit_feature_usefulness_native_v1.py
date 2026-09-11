"""Bounded compact-native Exit interventions, not an artifact admission owner.

The caller validates VAL population, episode seals, model/normalization identity,
online frozen Entry tokens, teacher identity and the label-independent donor
plan before calling this adapter. Its perturbation API reads no files and selects
no donor. The separate selected-pair loader below binds persisted models only.
One call scans a complete two-side episode from origin through the existing
trainer wrapper. Local and MTF GRUs see their entire unique histories, never a
reconstructed rolling-window tape. Outputs are ordered [side, state, action].

The explicit byte budget bounds each recipient/donor compact payload before any
copy or forward. Scratch is O(one compact episode), plus native model inference
activations; this is not a process RAM/VRAM guard. The caller must stream episodes
and effects and apply the existing execution guard. There is no population cache.

Interventions are unsealed model-input mappings. Original seals are retained only
as source evidence, never asserted to hash perturbed bytes. Observed clocks,
rewards, supervision and masks are not transplanted or regenerated. Current
native packs have equal full-capacity sides: observed path-prefix lengths are
implicit in the state index. Variable-length or geometry-changing swaps are not
supported. Side binding is the exact opposite-side native baseline Q, obtained
by swapping the side axis after a genuine forward, not a fictitious side input.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from gx1.contracts.entry_decision_token_v1 import ENTRY_DECISION_TOKEN_DIM
from gx1.contracts.entry_exit_feature_base_v1 import (
    EXIT_FEATURE_SEQUENCE_BARS,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_exit_feature_usefulness_v1 import feature_usefulness_layout
from gx1.contracts import unified_exit_episode_pack_v1 as episode_owner
from gx1.contracts.entry_fitted_q_v1 import require_entry_fitted_q_iteration_state
from gx1.contracts.entry_model_native_train_launch_v1 import (
    canonical_json_sha256 as recipe_json_sha256,
    require_training_recipe_source_provenance,
    require_training_recipe_source_provenance_metadata,
)
from gx1.contracts.entry_model_native_pretest_technical_recipe_v1 import (
    SCHEMA_VERSION as PRETEST_RECIPE_SCHEMA_VERSION,
)
from gx1.contracts.model_state_digest_v1 import canonical_model_state_sha256
from gx1.contracts.unified_exit_fitted_q_v1 import (
    require_unified_exit_fitted_q_iteration_state,
)
from gx1.features.htf_features import MULTI_TF_FEATURE_COUNT_V4
from gx1.models.entry_v10 import entry_v10_bundle as bundle_owner
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    _fitted_q_targets_for_episode_batch,
    _forward_unified_exit_episode_pack,
)


_TIMEFRAMES = tuple(timeframe.lower() for timeframe in EXIT_MTF_CONTEXT_TIMEFRAMES)
_INPUT_KEYS = (
    "exit_local_history_x",
    "exit_state_ctx_cont",
    "exit_state_ctx_cat",
    "exit_path_x",
    *(f"exit_mtf_history_{timeframe}" for timeframe in _TIMEFRAMES),
    *(f"exit_mtf_gather_{timeframe}" for timeframe in _TIMEFRAMES),
)
_MASK_KEYS = (
    "exit_action_valid_mask",
    "exit_state_valid_mask",
    "exit_terminal_mask",
    "exit_terminal_reason_index",
    "exit_episode_lengths",
)
_SURFACE_KEYS = {
    "seq_signal": "exit_local_history_x",
    "snap_signal": "exit_local_history_x",
    "ctx_cont": "exit_state_ctx_cont",
    "ctx_cat": "exit_state_ctx_cat",
    "exit_path": "exit_path_x",
    **{
        f"seq_{timeframe}": f"exit_mtf_history_{timeframe}"
        for timeframe in _TIMEFRAMES
    },
}


@dataclass(frozen=True)
class NativeExitIntervention:
    """Unsealed forward inputs; source seals describe originals only.

    Arrays and token are independent copies. ``swap_output_sides`` must be
    applied after the native wrapper, as ``predict_spec`` does. This object is not
    a validated episode and must not be passed to an episode sealer/validator.
    """

    model_inputs: Mapping[str, np.ndarray]
    online_entry_token: torch.Tensor
    source_episode_pack_sha256: str
    donor_episode_pack_sha256: str | None
    physical_id: str | None
    swap_output_sides: bool


@dataclass(frozen=True)
class NativeExitSupervision:
    """Baseline Exit targets [side,state,action] and raw teacher Entry bridge [side]."""

    q_targets_bps: np.ndarray
    action_valid_mask: np.ndarray
    action_equivalence_mask: np.ndarray
    terminal_mask: np.ndarray
    entry_first_side_values_bps: np.ndarray
    entry_side_valid_mask: np.ndarray


class CompactNativeExitUsefulnessAdapter:
    """Prepare/predict exact usefulness-layout interventions on native packs.

    ``predict_spec(episode=..., online_entry_token=..., spec=None)`` is baseline.
    A structure-block spec requires exactly the caller's ``donor_episode``;
    token effects additionally require its ``donor_online_entry_token``.
    An optional donor token on other structure effects is validated but not used.
    Same-state opposite-side specs forbid external donors. Specs must equal
    the corresponding Exit spec from ``feature_usefulness_layout`` exactly.

    Tokens are [1, ENTRY_DECISION_TOKEN_DIM] float32 tensors from the online
    Entry snapshot, detached and copied without projection or normalization.
    A teacher token is accepted only by ``baseline_supervision``. Model mode,
    parameters, normalization buffers and caller arrays are never changed.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        ordered_signal_names: Sequence[str],
        device: torch.device,
        max_episode_bytes: int,
    ) -> None:
        if (
            isinstance(max_episode_bytes, bool)
            or not isinstance(max_episode_bytes, int)
            or max_episode_bytes < 1
        ):
            raise RuntimeError("NATIVE_EXIT_USEFULNESS_BYTE_BUDGET_INVALID")
        self.model = model
        self.device = torch.device(device)
        self.max_episode_bytes = max_episode_bytes
        layout = feature_usefulness_layout(ordered_signal_names)["tasks"]["exit"]
        self._specs = {
            spec["physical_id"]: spec
            for section in (
                "physical_field_perturbations",
                "family_tf_routes",
                "local_family_effects",
                "joint_effects",
                "exit_episode_effects",
            )
            for spec in layout[section]
        }
        self._aliases = tuple(
            (int(spec["alias_signal_index"]), int(spec["alias_ctx_cont_index"]))
            for spec in layout["physical_field_perturbations"]
            if spec.get("alias_signal_index") is not None
        )

    def _require_aliases(self, episode: Mapping[str, Any]) -> None:
        current = episode["exit_local_history_x"][EXIT_FEATURE_SEQUENCE_BARS - 1 :]
        context = episode["exit_state_ctx_cont"]
        for signal_index, context_index in self._aliases:
            signal_bytes = np.ascontiguousarray(current[:, signal_index]).view(np.uint32)
            context_bytes = np.ascontiguousarray(context[:, context_index]).view(np.uint32)
            if not np.array_equal(signal_bytes, context_bytes):
                raise RuntimeError("NATIVE_EXIT_USEFULNESS_ALIAS_OFF_MANIFOLD")

    def _require_episode(self, episode: Mapping[str, Any]) -> None:
        """Check compact transport, not provenance or a new content seal."""

        if not isinstance(episode, Mapping):
            raise RuntimeError("NATIVE_EXIT_USEFULNESS_COMPACT_EPISODE_REQUIRED")
        seal = episode.get("episode_pack_sha256")
        if (
            episode.get("schema_version")
            != episode_owner.UNIFIED_EXIT_EPISODE_PACK_SCHEMA_VERSION
            or not isinstance(seal, str)
            or len(seal) != 64
            or any(character not in "0123456789abcdef" for character in seal)
        ):
            raise RuntimeError("NATIVE_EXIT_USEFULNESS_SOURCE_SEAL_REQUIRED")
        byte_count = ENTRY_DECISION_TOKEN_DIM * np.dtype(np.float32).itemsize
        for value in episode.values():
            if isinstance(value, np.ndarray):
                byte_count += value.nbytes
        if byte_count > self.max_episode_bytes:
            raise RuntimeError("NATIVE_EXIT_USEFULNESS_EPISODE_BYTE_BUDGET_EXCEEDED")
        for name, shape in episode_owner._FIXED_ARRAY_SHAPES.items():
            value = episode.get(name)
            if not isinstance(value, np.ndarray) or value.shape != shape:
                raise RuntimeError(f"NATIVE_EXIT_USEFULNESS_COMPACT_SHAPE_INVALID:{name}")
        state_count = episode_owner.UNIFIED_EXIT_EPISODE_STATE_COUNT
        for timeframe in _TIMEFRAMES:
            history = episode.get(f"exit_mtf_history_{timeframe}")
            gather = episode.get(f"exit_mtf_gather_{timeframe}")
            times = episode.get(f"exit_mtf_history_time_ns_{timeframe}")
            if (
                not isinstance(history, np.ndarray)
                or history.ndim != 2
                or history.shape[0] < 1
                or history.shape[1] != MULTI_TF_FEATURE_COUNT_V4
                or not isinstance(gather, np.ndarray)
                or gather.shape != (state_count,)
                or gather.dtype != np.int64
                or not isinstance(times, np.ndarray)
                or times.shape != (history.shape[0],)
                or np.any(gather < 0)
                or np.any(gather >= history.shape[0])
                or np.any(np.diff(gather) < 0)
                or int(gather[-1]) != history.shape[0] - 1
            ):
                raise RuntimeError(f"NATIVE_EXIT_USEFULNESS_MTF_GEOMETRY_INVALID:{timeframe}")
        for name in (*_INPUT_KEYS, *_MASK_KEYS):
            value = episode[name]
            if name in ("exit_action_valid_mask", "exit_state_valid_mask", "exit_terminal_mask"):
                expected_dtype = np.dtype(np.bool_)
            elif name in (
                "exit_state_ctx_cat", "exit_terminal_reason_index", "exit_episode_lengths"
            ) or name.startswith("exit_mtf_gather_"):
                expected_dtype = np.dtype(np.int64)
            else:
                expected_dtype = np.dtype(np.float32)
            if value.dtype != expected_dtype or not np.isfinite(value).all():
                raise RuntimeError(f"NATIVE_EXIT_USEFULNESS_INPUT_DTYPE_OR_VALUE_INVALID:{name}")
        terminal = episode["exit_terminal_mask"]
        reason = episode["exit_terminal_reason_index"]
        valid = episode["exit_action_valid_mask"]
        if (
            not episode["exit_state_valid_mask"].all()
            or not np.all(episode["exit_episode_lengths"] == state_count)
            or terminal.any()
            or np.any(reason != 0)
            or not np.array_equal(valid[..., 1], episode["exit_state_valid_mask"])
            or not np.array_equal(valid[..., 0], episode["exit_state_valid_mask"])
        ):
            raise RuntimeError("NATIVE_EXIT_USEFULNESS_COMPLETE_EPISODE_REQUIRED")
        self._require_aliases(episode)

    def _require_token(self, token: torch.Tensor) -> None:
        if (
            not isinstance(token, torch.Tensor)
            or tuple(token.shape) != (1, ENTRY_DECISION_TOKEN_DIM)
            or token.dtype != torch.float32
            or not bool(torch.isfinite(token).all().item())
        ):
            raise RuntimeError("NATIVE_EXIT_USEFULNESS_FROZEN_TOKEN_INVALID")

    def prepare(
        self,
        *,
        episode: Mapping[str, Any],
        online_entry_token: torch.Tensor,
        spec: Mapping[str, Any] | None = None,
        donor_episode: Mapping[str, Any] | None = None,
        donor_online_entry_token: torch.Tensor | None = None,
    ) -> NativeExitIntervention:
        """Copy only compact forward surfaces; never reseal an intervention."""

        self._require_episode(episode)
        self._require_token(online_entry_token)
        checked_spec = None
        if spec is not None:
            if not isinstance(spec, Mapping) or not isinstance(spec.get("physical_id"), str):
                raise RuntimeError("NATIVE_EXIT_USEFULNESS_LAYOUT_SPEC_INVALID")
            checked_spec = self._specs.get(spec["physical_id"])
            if checked_spec is None or dict(spec) != checked_spec:
                raise RuntimeError("NATIVE_EXIT_USEFULNESS_LAYOUT_SPEC_INVALID")
        swap_sides = (
            checked_spec is not None
            and checked_spec.get("donor_kind") == "same_state_opposite_side"
        )
        targets = [] if checked_spec is None else checked_spec["targets"]
        token_effect = any(
            target["surface"] == "entry_decision_representation" for target in targets
        ) and not swap_sides
        if checked_spec is None or swap_sides:
            if donor_episode is not None or donor_online_entry_token is not None:
                raise RuntimeError("NATIVE_EXIT_USEFULNESS_EXTERNAL_DONOR_FORBIDDEN")
        else:
            if donor_episode is None:
                raise RuntimeError("NATIVE_EXIT_USEFULNESS_EXPLICIT_DONOR_REQUIRED")
            self._require_episode(donor_episode)
            if donor_episode["episode_pack_sha256"] == episode["episode_pack_sha256"]:
                raise RuntimeError("NATIVE_EXIT_USEFULNESS_SELF_DONOR_FORBIDDEN")
            for name in _MASK_KEYS:
                if not np.array_equal(episode[name], donor_episode[name]):
                    raise RuntimeError("NATIVE_EXIT_USEFULNESS_DONOR_STATE_GEOMETRY_MISMATCH")
            if token_effect:
                self._require_token(donor_online_entry_token)
            elif donor_online_entry_token is not None:
                self._require_token(donor_online_entry_token)
            for target in targets:
                surface = target["surface"]
                if surface in {f"seq_{timeframe}" for timeframe in _TIMEFRAMES}:
                    timeframe = surface.removeprefix("seq_")
                    history_key = f"exit_mtf_history_{timeframe}"
                    gather_key = f"exit_mtf_gather_{timeframe}"
                    if (
                        episode[history_key].shape != donor_episode[history_key].shape
                        or not np.array_equal(episode[gather_key], donor_episode[gather_key])
                    ):
                        raise RuntimeError(
                            f"NATIVE_EXIT_USEFULNESS_DONOR_MTF_GEOMETRY_MISMATCH:{timeframe}"
                        )
        inputs = {
            name: np.array(episode[name], copy=True, order="C")
            for name in (*_INPUT_KEYS, *_MASK_KEYS)
        }
        selected_token = donor_online_entry_token if token_effect else online_entry_token
        frozen_token = selected_token.detach().to(self.device).clone()
        if checked_spec is not None and not swap_sides:
            fields_by_key: dict[str, set[int]] = {}
            for target in targets:
                surface = target["surface"]
                if surface in ("entry_decision_representation", "exit_path_lengths"):
                    continue
                key = _SURFACE_KEYS[surface]
                fields_by_key.setdefault(key, set()).update(target["source_indices"])
            for key, fields in fields_by_key.items():
                indices = sorted(fields)
                inputs[key][..., indices] = donor_episode[key][..., indices]
        self._require_aliases(inputs)
        return NativeExitIntervention(
            model_inputs=MappingProxyType(inputs),
            online_entry_token=frozen_token,
            source_episode_pack_sha256=episode["episode_pack_sha256"],
            donor_episode_pack_sha256=(
                None if donor_episode is None else donor_episode["episode_pack_sha256"]
            ),
            physical_id=None if checked_spec is None else checked_spec["physical_id"],
            swap_output_sides=swap_sides,
        )

    def predict_spec(
        self,
        *,
        episode: Mapping[str, Any],
        online_entry_token: torch.Tensor,
        spec: Mapping[str, Any] | None = None,
        donor_episode: Mapping[str, Any] | None = None,
        donor_online_entry_token: torch.Tensor | None = None,
    ) -> np.ndarray:
        """Return complete raw native Q [side,state,action], with no row filtering."""

        if self.model.training:
            raise RuntimeError("NATIVE_EXIT_USEFULNESS_MODEL_MUST_BE_EVAL")
        intervention = self.prepare(
            episode=episode,
            online_entry_token=online_entry_token,
            spec=spec,
            donor_episode=donor_episode,
            donor_online_entry_token=donor_online_entry_token,
        )
        with torch.no_grad():
            q_values, _valid, _state_valid, _terminal, _lengths = (
                _forward_unified_exit_episode_pack(
                    model=self.model,
                    entry_decision_representation=intervention.online_entry_token,
                    episode=intervention.model_inputs,
                    device=self.device,
                )
            )
        result = q_values[0].detach().cpu().numpy()
        if intervention.swap_output_sides:
            result = result[::-1]
        return np.array(result, copy=True, order="C")

    def baseline_supervision(
        self,
        *,
        episode: Mapping[str, Any],
        target_model: torch.nn.Module,
        target_entry_token: torch.Tensor,
    ) -> NativeExitSupervision:
        """Call the existing teacher/Bellman owner on the original episode only.

        One teacher forward supplies Exit targets/masks and raw first-state
        side values for Entry, not Bellman targets at state zero. Compute once
        per recipient and retain every returned field unchanged for all effects.
        Binding the teacher to its frozen fitted-Q iteration remains the caller's
        responsibility. No effect/donor argument or online-token substitution is
        accepted here; no target is computed from perturbed features or rewards.
        """

        if target_model.training:
            raise RuntimeError("NATIVE_EXIT_USEFULNESS_TARGET_MODEL_MUST_BE_EVAL")
        original = self.prepare(episode=episode, online_entry_token=target_entry_token)
        teacher_episode = {
            **original.model_inputs,
            "exit_now_reward_bps": np.array(episode["exit_now_reward_bps"], copy=True),
            "unbounded_exit_training_readiness": episode["unbounded_exit_training_readiness"],
        }
        with torch.no_grad():
            (
                targets, valid, terminal, first_side_values, first_side_valid
            ) = _fitted_q_targets_for_episode_batch(
                target_model=target_model,
                target_entry_decision_representations=original.online_entry_token,
                episodes=[teacher_episode],
                device=self.device,
            )
            maximum = targets.masked_fill(~valid, -torch.inf).amax(dim=-1, keepdim=True)
            equivalent = valid & (targets == maximum)
        return NativeExitSupervision(
            q_targets_bps=targets[0].detach().cpu().numpy().copy(),
            action_valid_mask=valid[0].detach().cpu().numpy().copy(),
            action_equivalence_mask=equivalent[0].detach().cpu().numpy().copy(),
            terminal_mask=terminal[0].detach().cpu().numpy().copy(),
            entry_first_side_values_bps=first_side_values[0].detach().cpu().numpy().copy(),
            entry_side_valid_mask=first_side_valid[0].detach().cpu().numpy().copy(),
        )


@dataclass(frozen=True)
class SelectedNativeExitPair:
    """Strict CPU model pair, not population validation or execution authority."""

    model: torch.nn.Module
    target_model: torch.nn.Module
    metadata: Mapping[str, Any]
    bindings: Mapping[str, Any]

    @property
    def input_normalization(self) -> Mapping[str, Any]:
        return self.metadata["input_normalization"]


def _selected_file(path: Path, expected_sha256: str) -> tuple[Path, str]:
    supplied = Path(path)
    if (
        not supplied.is_absolute()
        or ".." in supplied.parts
        or any(component.is_symlink() for component in (supplied, *supplied.parents))
        or not supplied.is_file()
    ):
        raise RuntimeError("NATIVE_EXIT_SELECTED_FILE_PATH_INVALID")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise RuntimeError("NATIVE_EXIT_SELECTED_FILE_SHA_INVALID")
    digest = hashlib.sha256()
    with supplied.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise RuntimeError(f"NATIVE_EXIT_SELECTED_FILE_SHA_MISMATCH:{supplied.name}")
    return supplied, observed


def _selected_json(path: Path, expected_sha256: str) -> dict[str, Any]:
    checked, _digest = _selected_file(path, expected_sha256)
    value = json.loads(checked.read_text(encoding="utf-8"))
    _selected_file(checked, expected_sha256)
    if not isinstance(value, dict):
        raise RuntimeError("NATIVE_EXIT_SELECTED_JSON_INVALID")
    return value


def _require_selected_recipe_bindings(
    *, recipe: Mapping[str, Any], contract: Mapping[str, Any], metadata: Mapping[str, Any]
) -> None:
    """Compare declarations only; dataset byte/population validation is external."""

    lineage = metadata["run_lineage"]
    if (
        contract.get("run_id") != lineage["training_run_id"]
        or contract.get("dataset_run_id") != lineage["dataset_run_id"]
        or recipe.get("run_id") != contract.get("run_id")
        or recipe.get("dataset_run_id") != contract.get("dataset_run_id")
        or recipe.get("out_bundle_dir") != contract.get("out_bundle_dir")
    ):
        raise RuntimeError("NATIVE_EXIT_SELECTED_RECIPE_SESSION_LINEAGE_MISMATCH")
    artifact_names = {
        "train_parquet": "train_parquet",
        "val_parquet": "val_parquet",
        "m5_prebuilt_path": (
            "m5_prebuilt" if recipe["schema_version"] == PRETEST_RECIPE_SCHEMA_VERSION
            else "m5_prebuilt_path"
        ),
        "unified_exit_lifecycle_manifest": (
            "unified_exit_lifecycle_manifest"
            if recipe["schema_version"] == PRETEST_RECIPE_SCHEMA_VERSION
            else "unified_exit_lifecycle_manifest_json"
        ),
    }
    artifacts = contract.get("artifacts")
    declared = recipe.get("artifact_bindings")
    if not isinstance(artifacts, Mapping) or set(artifacts) != set(artifact_names) or not isinstance(declared, Mapping):
        raise RuntimeError("NATIVE_EXIT_SELECTED_ARTIFACT_DECLARATIONS_INVALID")
    if recipe.get("artifact_bindings_sha256") != recipe_json_sha256(declared):
        raise RuntimeError("NATIVE_EXIT_SELECTED_ARTIFACT_DECLARATIONS_INVALID")
    for name, recipe_name in artifact_names.items():
        binding = declared.get(recipe_name)
        if (
            not isinstance(binding, Mapping)
            or artifacts[name] != {key: binding.get(key) for key in ("path", "sha256")}
        ):
            raise RuntimeError(f"NATIVE_EXIT_SELECTED_ARTIFACT_DECLARATION_MISMATCH:{name}")
    for split in ("train", "val"):
        binding = artifacts[f"{split}_parquet"]
        if metadata.get(f"{split}_data") != binding["path"] or metadata.get(f"{split}_data_sha256") != binding["sha256"]:
            raise RuntimeError("NATIVE_EXIT_SELECTED_BUNDLE_ARTIFACT_DECLARATION_MISMATCH")


def _load_selected_candidate_state(
    *,
    session_out_bundle_dir: Path,
    session_contract_sha256: str,
    active_pointer_sha256: str,
    selected_checkpoint_sha256: str,
    metadata: Mapping[str, Any],
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, str], dict[str, dict[str, str]]
]:
    """Reuse the read-only session owner; never restore training/live state."""

    output = Path(session_out_bundle_dir)
    directory = output.parent / (trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + output.name)
    contract_path = directory / trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME
    active_path = directory / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME
    contract = _selected_json(contract_path, session_contract_sha256)
    authority = contract.get("authority")
    if (
        contract.get("schema_version") != trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION
        or contract.get("out_bundle_dir") != str(output)
        or contract.get("profile") != "candidate"
        or contract.get("execution_tier") != "canonical"
        or not isinstance(authority, Mapping)
        or authority.get("candidate_training") is not True
        or authority.get("bundle") is not False
    ):
        raise RuntimeError("NATIVE_EXIT_SELECTED_SESSION_CONTRACT_INVALID")
    session = trainer._CandidateTrainingSession(
        out_bundle_dir=output, contract=contract, read_only=True
    )
    if session.contract_sha256 != session_contract_sha256:
        raise RuntimeError("NATIVE_EXIT_SELECTED_SESSION_CONTRACT_HASH_MISMATCH")
    active = _selected_json(active_path, active_pointer_sha256)
    state = session.load_checkpoint()
    if state is None or state["complete"] is not True or state["phase"] != "validation":
        raise RuntimeError("NATIVE_EXIT_SELECTED_SESSION_NOT_COMPLETE")
    progress = state["training_progress"]
    selection = progress["checkpoint_selection"]
    best_epoch = selection["best_epoch"]
    last_epoch = selection["last_epoch"]
    if (
        isinstance(best_epoch, bool) or isinstance(last_epoch, bool)
        or not 1 <= best_epoch <= last_epoch
        or state["epoch_index"] + 1 != last_epoch
        or progress["validation_snapshot"] is not None
        or metadata.get("best_epoch") != best_epoch
        or metadata.get("last_epoch") != last_epoch
        or metadata.get("early_stopped") is not selection["early_stopped"]
        or contract.get("training", {}).get("checkpoint_policy") != selection["checkpoint_policy"]
    ):
        raise RuntimeError("NATIVE_EXIT_SELECTED_EPOCH_BINDING_INVALID")
    checkpoint = selection["best_checkpoint"]
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint["epoch"] != best_epoch
        or checkpoint not in selection["top_k_checkpoints"]
        or checkpoint["sha256"] != selected_checkpoint_sha256
        or checkpoint["metric"] != selection["best_policy_pnl"]
        or checkpoint["metric"] != metadata.get("best_entry_policy_realized_gross_spread_inclusive_pnl_bps")
    ):
        raise RuntimeError("NATIVE_EXIT_SELECTED_CHECKPOINT_BINDING_INVALID")
    checkpoint_path, _digest = _selected_file(directory / checkpoint["path"], selected_checkpoint_sha256)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    _selected_file(checkpoint_path, selected_checkpoint_sha256)
    online_digest = canonical_model_state_sha256(selection["best_state"])
    target_digest = canonical_model_state_sha256(selection["best_fitted_q_target_state"])
    if (
        payload["session_contract_sha256"] != session.contract_sha256
        or payload["epoch"] != best_epoch
        or payload["metric"] != checkpoint["metric"]
        or canonical_model_state_sha256(payload["model_state"]) != online_digest
        or canonical_model_state_sha256(payload["target_model_state"]) != target_digest
        or metadata.get("selected_online_model_state_sha256") != online_digest
    ):
        raise RuntimeError("NATIVE_EXIT_SELECTED_CHECKPOINT_STATE_MISMATCH")
    exit_iteration = require_unified_exit_fitted_q_iteration_state(
        selection["best_unified_exit_fitted_q_state"], context="NATIVE_EXIT_SELECTED"
    )
    entry_iteration = require_entry_fitted_q_iteration_state(
        selection["best_entry_fitted_q_state"],
        exit_fitted_q_iteration_state=exit_iteration, context="NATIVE_EXIT_SELECTED",
    )
    evidence = metadata["unified_exit_training_evidence"]
    full_trajectory = selection["best_unified_exit_full_trajectory_validation"]
    if (
        exit_iteration["target_model_state_sha256"] != target_digest
        or entry_iteration["entry_target_model_state_sha256"] != target_digest
        or exit_iteration["iteration_index"] != best_epoch - 1
        or exit_iteration["train_split_sha256"] != contract["artifacts"]["train_parquet"]["sha256"]
        or exit_iteration["source_lineage_sha256"] != contract["artifacts"]["unified_exit_lifecycle_manifest"]["sha256"]
        or exit_iteration != evidence["selected_fitted_q_iteration_state"]
        or entry_iteration != metadata["selected_entry_fitted_q_iteration_state"]
        or full_trajectory != evidence["full_trajectory_validation"]
        or exit_iteration["normalization_sha256"] != metadata["input_normalization"]["contract_sha256"]
        or contract.get("input_normalization_sha256") != exit_iteration["normalization_sha256"]
    ):
        raise RuntimeError("NATIVE_EXIT_SELECTED_TEACHER_EVIDENCE_MISMATCH")
    bundle_owner._require_candidate_full_trajectory_bindings(
        full_trajectory, selected_online_model_state_sha256=online_digest,
        target_model_state_sha256=target_digest,
    )
    _selected_file(contract_path, session_contract_sha256)
    _selected_file(active_path, active_pointer_sha256)
    state_path = session._slot_path(active["slot"])
    _selected_file(state_path, active["state_sha256"])
    selection_artifacts = {
        "session_contract": {"path": str(contract_path), "sha256": session_contract_sha256},
        "active_pointer": {"path": str(active_path), "sha256": active_pointer_sha256},
        "active_state": {"path": str(state_path), "sha256": active["state_sha256"]},
        "selected_checkpoint": {"path": str(checkpoint_path), "sha256": selected_checkpoint_sha256},
    }
    files = {binding["path"]: binding["sha256"] for binding in selection_artifacts.values()}
    return contract, selection, files, selection_artifacts


def load_selected_native_exit_pair(
    *,
    bundle_dir: Path,
    bundle_metadata_sha256: str,
    session_out_bundle_dir: Path,
    session_contract_sha256: str,
    active_pointer_sha256: str,
    selected_checkpoint_sha256: str,
    recipe_audit_path: Path,
    recipe_audit_sha256: str,
    repo: Path,
) -> SelectedNativeExitPair:
    """Load a selected CPU pair without inference, training or artifact writes.

    All paths/hashes are explicit; the active pointer is a pinned file, not a
    request for the latest checkpoint. The retained best online state may be
    EMA and may precede the final epoch. Only its retained target is loaded.
    Source provenance is checked by the existing read-only recipe owner, with
    its clean-checkout and exact source-byte requirements, not an exception for
    this staging patch, and rechecked after the strict model/teacher load.
    ``bindings.selection_artifacts`` preserves six explicit artifact roles even
    when different role paths have identical hashes. Dataset artifact
    declarations are cross-bound but no dataset tensors/population are loaded
    here; the parent owns those proofs.

    The strict bundle owner validates publication, online state, metadata and
    normalization. The selected teacher strict-loads into an independent copy
    and revalidates those exact normalization buffers. Both models are CPU,
    eval and gradient-frozen. Session/checkpoint file loading is governed by
    the existing owner and the caller's process memory cap, not the episode
    adapter's compact-input byte budget. This grants no resume/TEST authority.
    """

    metadata_path = Path(bundle_dir) / "bundle_metadata.json"
    metadata = _selected_json(metadata_path, bundle_metadata_sha256)
    recipe = _selected_json(recipe_audit_path, recipe_audit_sha256)
    for module in (trainer, bundle_owner):
        expected_source = Path(repo) / Path(*module.__name__.split(".")).with_suffix(".py")
        if Path(module.__file__).resolve(strict=True) != expected_source.resolve(strict=True):
            raise RuntimeError("NATIVE_EXIT_SELECTED_IMPORTED_SOURCE_ROOT_MISMATCH")
    if metadata.get("run_lineage", {}).get("training_profile") != "candidate":
        raise RuntimeError("NATIVE_EXIT_SELECTED_CANDIDATE_BUNDLE_REQUIRED")
    output = Path(session_out_bundle_dir)
    session_directory = output.parent / (trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + output.name)
    contract = _selected_json(
        session_directory / trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME,
        session_contract_sha256,
    )
    source_arguments = dict(
        recipe_audit_path=recipe_audit_path, recipe_audit_sha256=recipe_audit_sha256,
        repo=Path(repo), profile="candidate", run_id=contract["run_id"],
        dataset_run_id=contract["dataset_run_id"], dataset_dir=Path(recipe["dataset_dir"]),
        out_bundle_dir=Path(session_out_bundle_dir),
    )
    provenance = require_training_recipe_source_provenance(**source_arguments)
    for value in (contract.get("recipe_source_provenance"), metadata.get("recipe_source_provenance")):
        if require_training_recipe_source_provenance_metadata(value, context="NATIVE_EXIT_SELECTED") != provenance:
            raise RuntimeError("NATIVE_EXIT_SELECTED_RECIPE_SOURCE_MISMATCH")
    _require_selected_recipe_bindings(recipe=recipe, contract=contract, metadata=metadata)
    contract, selection, files, selection_artifacts = _load_selected_candidate_state(
        session_out_bundle_dir=session_out_bundle_dir,
        session_contract_sha256=session_contract_sha256,
        active_pointer_sha256=active_pointer_sha256,
        selected_checkpoint_sha256=selected_checkpoint_sha256,
        metadata=metadata,
    )
    bundle = bundle_owner.load_entry_v10_ctx_bundle(bundle_dir=bundle_dir, device="cpu")
    if any(bundle.metadata.get(key) != value for key, value in metadata.items()):
        raise RuntimeError("NATIVE_EXIT_SELECTED_LOADED_METADATA_MISMATCH")
    model = bundle.transformer_model
    if canonical_model_state_sha256(model.state_dict()) != canonical_model_state_sha256(selection["best_state"]):
        raise RuntimeError("NATIVE_EXIT_SELECTED_EXPORTED_MODEL_MISMATCH")
    model.require_input_normalization_state()
    target_model = deepcopy(model)
    target_model.load_state_dict(selection["best_fitted_q_target_state"], strict=True)
    target_model.require_input_normalization_state()
    if canonical_model_state_sha256(target_model.state_dict()) != canonical_model_state_sha256(selection["best_fitted_q_target_state"]):
        raise RuntimeError("NATIVE_EXIT_SELECTED_LOADED_TEACHER_MISMATCH")
    model.eval().requires_grad_(False)
    target_model.eval().requires_grad_(False)
    selection_artifacts.update({
        "bundle_metadata": {"path": str(metadata_path), "sha256": bundle_metadata_sha256},
        "recipe_audit": {"path": str(recipe_audit_path), "sha256": recipe_audit_sha256},
    })
    files.update({str(metadata_path): bundle_metadata_sha256, str(recipe_audit_path): recipe_audit_sha256})
    for path, digest in files.items():
        _selected_file(Path(path), digest)
    if require_training_recipe_source_provenance(**source_arguments) != provenance:
        raise RuntimeError("NATIVE_EXIT_SELECTED_RECIPE_SOURCE_CHANGED_DURING_LOAD")
    return SelectedNativeExitPair(
        model=model, target_model=target_model, metadata=MappingProxyType(metadata),
        bindings=MappingProxyType({
            "files": files,
            "selection_artifacts": selection_artifacts,
            "bundle_metadata_path": str(metadata_path),
            "bundle_metadata_sha256": bundle_metadata_sha256,
            "bundle_commit_sha256": bundle.bundle_sha256,
            "selected_epoch": selection["best_epoch"],
            "last_epoch": selection["last_epoch"],
            "online_model_state_sha256": canonical_model_state_sha256(model.state_dict()),
            "target_model_state_sha256": canonical_model_state_sha256(target_model.state_dict()),
            "input_normalization_sha256": contract["input_normalization_sha256"],
            "recipe_source_provenance": provenance,
            "dataset_artifact_declarations": contract["artifacts"],
            "dataset_bytes_validated_by_this_loader": False,
        }),
    )


def require_selected_native_pair_unchanged(
    *, selected_pair: SelectedNativeExitPair, repo: Path
) -> None:
    """Recheck a strict-loaded pair before/after an audit, without loading state.

    This is preservation, not selection or fresh candidate admission. Explicit
    role/file closure, metadata, separate online/teacher states and the existing
    clean/exact recipe-source owner must still agree. Every submodule must stay
    eval, every parameter frozen and every floating parameter/buffer FP32;
    integer/bool buffers retain their digest-bound dtype and bytes. No module
    conversion, movement, forward, checkpoint deserialization or dataset read
    occurs. The canonical digest/normalization owners may copy tensor values to
    CPU for comparison without modifying their source tensors.

    Callers must serialize model/artifact changes with the audit; these boundary
    checks are not an atomic lock. They confer no population, resume or TEST
    authority and do not substitute for the strict initial pair loader.
    """

    if (
        not isinstance(selected_pair, SelectedNativeExitPair)
        or not isinstance(selected_pair.model, torch.nn.Module)
        or not isinstance(selected_pair.target_model, torch.nn.Module)
        or selected_pair.model is selected_pair.target_model
    ):
        raise RuntimeError("NATIVE_EXIT_SELECTED_SEPARATE_PAIR_REQUIRED")
    bindings = selected_pair.bindings
    if not isinstance(bindings, Mapping):
        raise RuntimeError("NATIVE_EXIT_SELECTED_ROLE_FILE_CLOSURE_MISMATCH")
    roles = bindings.get("selection_artifacts")
    files = bindings.get("files")
    expected_roles = {
        "session_contract", "active_pointer", "active_state",
        "selected_checkpoint", "bundle_metadata", "recipe_audit",
    }
    if not isinstance(roles, Mapping) or set(roles) != expected_roles or not isinstance(files, Mapping):
        raise RuntimeError("NATIVE_EXIT_SELECTED_ROLE_FILE_CLOSURE_MISMATCH")
    role_files = {}
    for binding in roles.values():
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or not isinstance(binding["path"], str)
            or binding["path"] in role_files
        ):
            raise RuntimeError("NATIVE_EXIT_SELECTED_ROLE_FILE_CLOSURE_MISMATCH")
        role_files[binding["path"]] = binding["sha256"]
    if dict(files) != role_files:
        raise RuntimeError("NATIVE_EXIT_SELECTED_ROLE_FILE_CLOSURE_MISMATCH")
    for path, digest in role_files.items():
        _selected_file(Path(path), digest)
    if roles["bundle_metadata"] != {
        "path": bindings.get("bundle_metadata_path"),
        "sha256": bindings.get("bundle_metadata_sha256"),
    }:
        raise RuntimeError("NATIVE_EXIT_SELECTED_METADATA_ROLE_MISMATCH")
    documents = {
        role: _selected_json(Path(roles[role]["path"]), roles[role]["sha256"])
        for role in ("bundle_metadata", "recipe_audit", "session_contract", "active_pointer")
    }
    metadata = documents["bundle_metadata"]
    if metadata != selected_pair.metadata:
        raise RuntimeError("NATIVE_EXIT_SELECTED_METADATA_CHANGED")
    recipe = documents["recipe_audit"]
    contract = documents["session_contract"]
    active = documents["active_pointer"]
    output = Path(contract["out_bundle_dir"])
    directory = output.parent / (trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + output.name)
    slot = active.get("slot")
    if (
        roles["session_contract"]["path"] != str(directory / trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME)
        or roles["active_pointer"]["path"] != str(directory / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME)
        or type(slot) is not int
        or slot not in range(len(trainer._CANDIDATE_TRAINING_STATE_FILENAMES))
        or roles["active_state"]["path"] != str(directory / trainer._CANDIDATE_TRAINING_STATE_FILENAMES[slot])
        or roles["active_state"]["sha256"] != active.get("state_sha256")
        or roles["session_contract"]["sha256"] != active.get("session_contract_sha256")
    ):
        raise RuntimeError("NATIVE_EXIT_SELECTED_SESSION_ROLE_MISMATCH")
    if (
        bindings.get("selected_epoch") != metadata["best_epoch"]
        or bindings.get("last_epoch") != metadata["last_epoch"]
        or bindings.get("online_model_state_sha256") != metadata["selected_online_model_state_sha256"]
        or bindings.get("target_model_state_sha256") != metadata["unified_exit_training_evidence"]["selected_fitted_q_iteration_state"]["target_model_state_sha256"]
        or bindings.get("input_normalization_sha256") != metadata["input_normalization"]["contract_sha256"]
        or bindings.get("input_normalization_sha256") != contract["input_normalization_sha256"]
        or bindings.get("dataset_artifact_declarations") != contract["artifacts"]
    ):
        raise RuntimeError("NATIVE_EXIT_SELECTED_METADATA_BINDINGS_MISMATCH")
    _require_selected_recipe_bindings(recipe=recipe, contract=contract, metadata=metadata)
    provenance = require_training_recipe_source_provenance_metadata(
        bindings.get("recipe_source_provenance"), context="NATIVE_EXIT_SELECTED"
    )
    if roles["recipe_audit"] != {
        "path": provenance["recipe_audit_path"], "sha256": provenance["recipe_audit_sha256"]
    }:
        raise RuntimeError("NATIVE_EXIT_SELECTED_RECIPE_ROLE_MISMATCH")
    for value in (contract.get("recipe_source_provenance"), metadata.get("recipe_source_provenance")):
        if require_training_recipe_source_provenance_metadata(value, context="NATIVE_EXIT_SELECTED") != provenance:
            raise RuntimeError("NATIVE_EXIT_SELECTED_RECIPE_SOURCE_MISMATCH")
    for module in (trainer, bundle_owner):
        expected_source = Path(repo) / Path(*module.__name__.split(".")).with_suffix(".py")
        if Path(module.__file__).resolve(strict=True) != expected_source.resolve(strict=True):
            raise RuntimeError("NATIVE_EXIT_SELECTED_IMPORTED_SOURCE_ROOT_MISMATCH")
    for role, model, digest_key in (
        ("online", selected_pair.model, "online_model_state_sha256"),
        ("teacher", selected_pair.target_model, "target_model_state_sha256"),
    ):
        if any(module.training for module in model.modules()):
            raise RuntimeError(f"NATIVE_EXIT_SELECTED_MODEL_MUST_REMAIN_EVAL:{role}")
        if any(parameter.requires_grad for parameter in model.parameters()):
            raise RuntimeError(f"NATIVE_EXIT_SELECTED_PARAMETERS_MUST_REMAIN_FROZEN:{role}")
        for tensors in (model.named_parameters(), model.named_buffers()):
            for name, tensor in tensors:
                if tensor.is_floating_point() and tensor.dtype != torch.float32:
                    raise RuntimeError(f"NATIVE_EXIT_SELECTED_FLOATING_STATE_MUST_BE_FP32:{role}:{name}")
        model.require_input_normalization_state()
        if canonical_model_state_sha256(model.state_dict()) != bindings[digest_key]:
            raise RuntimeError(f"NATIVE_EXIT_SELECTED_MODEL_STATE_CHANGED:{role}")
    observed_provenance = require_training_recipe_source_provenance(
        recipe_audit_path=Path(roles["recipe_audit"]["path"]),
        recipe_audit_sha256=roles["recipe_audit"]["sha256"],
        repo=Path(repo), profile="candidate", run_id=contract["run_id"],
        dataset_run_id=contract["dataset_run_id"], dataset_dir=Path(recipe["dataset_dir"]),
        out_bundle_dir=output,
    )
    if observed_provenance != provenance:
        raise RuntimeError("NATIVE_EXIT_SELECTED_RECIPE_SOURCE_MISMATCH")
    for path, digest in role_files.items():
        _selected_file(Path(path), digest)
