"""Immutable VAL-only learned feature-usefulness diagnostics.

This contract is deliberately separate from input liveness and candidate
admission.  It measures paired counterfactual loss/margin deltas after a model
has been trained, but it never removes an input, changes a model output, selects a
checkpoint or reads TEST.  Zero and negative measurements are valid evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from itertools import combinations
from collections.abc import Mapping, Sequence
from typing import Any

from gx1.contracts.entry_decision_token_v1 import (
    entry_decision_token_projection_metadata,
)
from gx1.contracts.entry_fitted_q_v1 import (
    ENTRY_FITTED_Q_ACTION_ORDER,
    ENTRY_FITTED_Q_SCHEMA_VERSION,
    entry_fitted_q_contract,
    require_entry_fitted_q_iteration_state,
)
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_MTF_CONTEXT_TIMEFRAMES,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_BASE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_SIGNAL_DIM,
    MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
    MODEL_NATIVE_STATIC_CONTRACT_SHA256,
    ordered_model_native_signal_fields,
)
from gx1.contracts.model_native_serve_gate_v1 import (
    individual_input_influence_layout,
)
from gx1.contracts.unified_exit_input_influence_v1 import (
    unified_exit_input_influence_layout,
)
from gx1.contracts.unified_exit_input_v1 import (
    UNIFIED_EXIT_INPUT_ENVELOPE_SCHEMA_VERSION,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    UNIFIED_EXIT_FITTED_Q_SCHEMA_VERSION,
    require_unified_exit_fitted_q_iteration_state,
    unified_exit_fitted_q_contract,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    classify_entry_specialist_feature,
    require_multi_tf_specialist_routing_v4,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4


SCHEMA_VERSION = "gx1_entry_exit_feature_usefulness_v5"
LAYOUT_SCHEMA_VERSION = "gx1_entry_exit_feature_usefulness_layout_v4"
DONOR_PLAN_SCHEMA_VERSION = "gx1_structure_preserving_val_block_swap_v1"
SIDE_PAIR_PLAN_SCHEMA_VERSION = "gx1_exit_same_state_opposite_side_pair_v1"
SPLIT = "val"
DECISION = "VAL_DIAGNOSTIC_COMPLETE_NO_SELECTION_AUTHORITY"
TASKS = ("entry", "exit")
TASK_CLASS_ORDER = {
    "entry": ENTRY_FITTED_Q_ACTION_ORDER,
    "exit": tuple(unified_exit_fitted_q_contract()["action_order"]),
}
POLICY = {
    "fit_or_selection_authority": False,
    "checkpoint_admission_authority": False,
    "model_output_or_gradient_authority": False,
    "retirement_authority": False,
    "diagnostic_split": SPLIT,
    "diagnostic_population": "complete_immutable_val_population_no_sampling",
    "test_rows_read": False,
    "test_rows_tune_or_select": False,
    "zero_or_negative_usefulness_is_valid_evidence": True,
    "automatic_importance_threshold": None,
    "automatic_top_k": None,
    "automatic_family_quota": None,
    "retirement_requires_explicit_later_code_contract_change": True,
}
PERTURBATION_POLICY = {
    "source": "genuine_immutable_val_rows",
    "mapping": "label_independent_whole_structure_block_cyclic_swap",
    "local_sequence": "swap_complete_field_trajectory_and_snap_together",
    "temporal_alias": "swap_seq_field_snap_and_ctx_copy_together",
    "categorical": "swap_observed_valid_category_never_synthesize",
    "mtf": "swap_complete_closed_tf_field_trajectory",
    "family_tf": "swap_exact_owner_indices_as_one_block",
    "interaction": "inclusion_exclusion_over_two_disjoint_owner_blocks",
    "exit_entry_token": "swap_complete_frozen_token_between_genuine_val_blocks",
    "exit_path": "swap_complete_path_and_observed_length_together",
    "exit_side_entry_binding": (
        "swap_complete_token_path_length_and_side_from_exact_same_state_"
        "opposite_side_pair"
    ),
    "arbitrary_row_shuffle": False,
    "static_replacement_value": None,
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTITY_KEYS = {
    "bundle_dir",
    "bundle_metadata_sha256",
    "model_state_sha256",
    "dataset_dir",
    "dataset_run_id",
    "val_manifest_path",
    "val_manifest_sha256",
    "val_data_path",
    "val_data_sha256",
    "val_start_utc",
    "val_end_utc",
    "entry_val_population_row_count",
    "exit_val_population_row_count",
    "normalization_path",
    "normalization_file_sha256",
    "normalization_contract_sha256",
    "entry_decision_token_snapshot_set_sha256",
    "unified_exit_input_envelope_set_sha256",
    "contract_mode",
    "signal_schema_version",
    "signal_static_contract_sha256",
    "entry_decision_token_projection",
    "unified_exit_input_envelope_schema_version",
}


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


def _targets(*rows: tuple[str, Sequence[int]]) -> list[dict[str, Any]]:
    return [
        {"surface": surface, "source_indices": [int(index) for index in indices]}
        for surface, indices in rows
    ]


def _whole_target(surface: str) -> dict[str, Any]:
    return {"surface": surface, "whole_surface": True}


def _merge_disjoint_targets(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Union two exact perturbation blocks and reject overlapping ownership."""

    occupied: dict[str, set[int] | None] = {}
    merged: dict[str, set[int] | None] = {}
    for side_name, targets in (("left", left), ("right", right)):
        for target in targets:
            surface = str(target["surface"])
            whole = target.get("whole_surface") is True
            indices = None if whole else {int(v) for v in target["source_indices"]}
            present = surface in occupied
            previous = occupied.get(surface)
            if side_name == "right" and present:
                if indices is None or previous is None or previous.intersection(indices):
                    raise RuntimeError(
                        "FEATURE_USEFULNESS_INTERACTION_BLOCKS_NOT_DISJOINT"
                    )
            occupied[surface] = (
                indices
                if not present
                else None
                if previous is None or indices is None
                else previous.union(indices)
            )
            if surface not in merged:
                merged[surface] = None if whole else set(indices or ())
            elif merged[surface] is None or whole:
                merged[surface] = None
            else:
                merged[surface].update(indices or ())
    return [
        _whole_target(surface)
        if indices is None
        else {
            "surface": surface,
            "source_indices": sorted(indices),
        }
        for surface, indices in merged.items()
    ]


def _task_layout(
    signal_names: tuple[str, ...],
    *,
    task: str,
    timeframes: tuple[str, ...],
) -> dict[str, Any]:
    individual_owner_layout = individual_input_influence_layout(
        signal_names,
        mtf_timeframes=timeframes,
    )
    owner_layout = (
        individual_owner_layout
        if task == "entry"
        else unified_exit_input_influence_layout(signal_names)
    )
    alias_rows = [
        *individual_owner_layout.get("continuous_manifold", []),
        *[
            row
            for row in individual_owner_layout["categorical"]
            if row.get("surface") == "temporal_alias"
        ],
    ]
    alias_by_signal = {
        str(row["signal_field"]): dict(row) for row in alias_rows
    }
    alias_by_ctx = {
        str(row["ctx_cont_field"]): dict(row) for row in alias_rows
    }

    physical: dict[str, dict[str, Any]] = {}
    local_rows: list[dict[str, Any]] = []
    for index, field in enumerate(signal_names):
        alias = alias_by_signal.get(field)
        physical_id = (
            f"temporal_alias.{alias['ctx_cont_field']}"
            if alias is not None
            else f"local_signal.{field}"
        )
        targets = _targets(("seq_signal", (index,)), ("snap_signal", (index,)))
        manifold = "genuine_val_local_sequence_and_snap_block_swap"
        if alias is not None:
            targets.append(
                {
                    "surface": "ctx_cont",
                    "source_indices": [int(alias["ctx_cont_index"])],
                }
            )
            manifold = "genuine_val_joint_seq_snap_ctx_temporal_alias_block_swap"
        physical.setdefault(
            physical_id,
            {
                "physical_id": physical_id,
                "manifold": manifold,
                "targets": targets,
                "alias_signal_index": index if alias is not None else None,
                "alias_ctx_cont_index": (
                    int(alias["ctx_cont_index"]) if alias is not None else None
                ),
            },
        )
        local_rows.append(
            {
                "token": f"local_signal.{field}",
                "field": field,
                "source_index": index,
                "physical_id": physical_id,
            }
        )

    ctx_cont_rows: list[dict[str, Any]] = []
    for index, field in enumerate(MODEL_NATIVE_CTX_CONT_FIELDS):
        alias = alias_by_ctx.get(field)
        physical_id = (
            f"temporal_alias.{field}" if alias is not None else f"ctx_cont.{field}"
        )
        if alias is None:
            physical[physical_id] = {
                "physical_id": physical_id,
                "manifold": "genuine_val_ctx_cont_row_block_swap",
                "targets": _targets(("ctx_cont", (index,))),
                "alias_signal_index": None,
                "alias_ctx_cont_index": None,
            }
        ctx_cont_rows.append(
            {
                "token": f"ctx_cont.{field}",
                "field": field,
                "source_index": index,
                "physical_id": physical_id,
            }
        )

    ctx_cat_rows: list[dict[str, Any]] = []
    for index, field in enumerate(MODEL_NATIVE_CTX_CAT_FIELDS):
        physical_id = f"ctx_cat.{field}"
        physical[physical_id] = {
            "physical_id": physical_id,
            "manifold": "genuine_val_observed_ctx_category_block_swap",
            "targets": _targets(("ctx_cat", (index,))),
            "alias_signal_index": None,
            "alias_ctx_cont_index": None,
        }
        ctx_cat_rows.append(
            {
                "token": physical_id,
                "field": field,
                "source_index": index,
                "physical_id": physical_id,
            }
        )

    mtf_rows: list[dict[str, Any]] = []
    for timeframe in timeframes:
        surface = f"seq_{timeframe.lower()}"
        for index, field in enumerate(MULTI_TF_PER_BAR_FEATURES_V4):
            physical_id = f"mtf.{timeframe.lower()}.{field}"
            physical[physical_id] = {
                "physical_id": physical_id,
                "manifold": "genuine_val_complete_closed_tf_field_trajectory_swap",
                "targets": _targets((surface, (index,))),
                "alias_signal_index": None,
                "alias_ctx_cont_index": None,
            }
            mtf_rows.append(
                {
                    "token": physical_id,
                    "field": field,
                    "timeframe": timeframe,
                    "source_index": index,
                    "physical_id": physical_id,
                }
            )

    routing = require_multi_tf_specialist_routing_v4(
        MULTI_TF_PER_BAR_FEATURES_V4
    )
    family_order = tuple(routing)
    route_rows: list[dict[str, Any]] = []
    route_by_family_tf: dict[tuple[str, str], dict[str, Any]] = {}
    local_family_effects: dict[str, dict[str, Any]] = {}
    joint_effects: dict[str, dict[str, Any]] = {}
    interactions: list[dict[str, Any]] = []

    local_indices_by_family: dict[str, list[int]] = {
        family: [] for family in family_order
    }
    for index, field in enumerate(signal_names):
        family = classify_entry_specialist_feature(field)
        if family not in local_indices_by_family:
            raise RuntimeError(
                f"FEATURE_USEFULNESS_LOCAL_SPECIALIST_OWNER_INVALID: {field}:{family}"
            )
        local_indices_by_family[family].append(index)
    if sum(len(indices) for indices in local_indices_by_family.values()) != len(
        signal_names
    ):
        raise RuntimeError("FEATURE_USEFULNESS_LOCAL_SPECIALIST_PARTITION_INVALID")
    for family, signal_indices in local_indices_by_family.items():
        physical_id = f"local_family.{family}"
        ctx_indices = sorted(
            int(alias_by_signal[signal_names[index]]["ctx_cont_index"])
            for index in signal_indices
            if signal_names[index] in alias_by_signal
        )
        targets = _targets(
            ("seq_signal", tuple(signal_indices)),
            ("snap_signal", tuple(signal_indices)),
        )
        if ctx_indices:
            targets.extend(_targets(("ctx_cont", tuple(ctx_indices))))
        local_family_effects[physical_id] = {
            "physical_id": physical_id,
            "manifold": "genuine_val_complete_local_specialist_owner_block_swap",
            "targets": targets,
        }

    for timeframe in timeframes:
        for family, indices in routing.items():
            route_id = f"family_tf.{timeframe.lower()}.{family}"
            route = {
                "token": route_id,
                "family": family,
                "timeframe": timeframe,
                "source_indices": list(indices),
                "physical_id": route_id,
                "manifold": "genuine_val_exact_family_tf_trajectory_block_swap",
                "targets": _targets(
                    (f"seq_{timeframe.lower()}", tuple(indices))
                ),
            }
            route_rows.append(route)
            route_by_family_tf[(family, timeframe)] = route

    def add_interaction(
        *,
        token: str,
        kind: str,
        left: Mapping[str, Any],
        right: Mapping[str, Any],
    ) -> None:
        left_id = str(left["physical_id"])
        right_id = str(right["physical_id"])
        joint_id = f"joint.{token}"
        joint_effects[joint_id] = {
            "physical_id": joint_id,
            "manifold": "genuine_val_union_of_two_disjoint_owner_blocks",
            "targets": _merge_disjoint_targets(left["targets"], right["targets"]),
        }
        interactions.append(
            {
                "token": token,
                "kind": kind,
                "left_effect_id": left_id,
                "right_effect_id": right_id,
                "joint_effect_id": joint_id,
                "formula": "joint_delta-left_delta-right_delta",
            }
        )

    for left_family, right_family in combinations(family_order, 2):
        add_interaction(
            token=f"interaction.local.{left_family}__x__{right_family}",
            kind="local_cross_family",
            left=local_family_effects[f"local_family.{left_family}"],
            right=local_family_effects[f"local_family.{right_family}"],
        )
    for timeframe in timeframes:
        for left_family, right_family in combinations(family_order, 2):
            add_interaction(
                token=(
                    f"interaction.{timeframe.lower()}.{left_family}__x__"
                    f"{right_family}"
                ),
                kind="per_tf_cross_family",
                left=route_by_family_tf[(left_family, timeframe)],
                right=route_by_family_tf[(right_family, timeframe)],
            )
    for family in family_order:
        for left_tf, right_tf in combinations(timeframes, 2):
            add_interaction(
                token=(
                    f"interaction.{family}.{left_tf.lower()}__x__"
                    f"{right_tf.lower()}"
                ),
                kind="cross_tf_same_family",
                left=route_by_family_tf[(family, left_tf)],
                right=route_by_family_tf[(family, right_tf)],
            )

    episode_effects: list[dict[str, Any]] = []
    if task == "exit":
        path_indices = owner_layout["numeric"]["exit_path"]["source_indices"]
        token_indices = owner_layout["numeric"][
            "entry_decision_representation"
        ]["source_indices"]
        episode_effects = [
            {
                "token": "exit_episode.frozen_entry_decision_token",
                "physical_id": "exit_episode.frozen_entry_decision_token",
                "donor_kind": "structure_block",
                "manifold": "genuine_val_complete_frozen_entry_token_block_swap",
                "targets": _targets(
                    ("entry_decision_representation", tuple(token_indices))
                ),
            },
            {
                "token": "exit_episode.path_and_observed_length",
                "physical_id": "exit_episode.path_and_observed_length",
                "donor_kind": "structure_block",
                "manifold": "genuine_val_complete_path_and_length_block_swap",
                "targets": [
                    *_targets(("exit_path", tuple(path_indices))),
                    _whole_target("exit_path_lengths"),
                ],
            },
            {
                "token": "exit_episode.side_entry_binding",
                "physical_id": "exit_episode.side_entry_binding",
                "donor_kind": "same_state_opposite_side",
                "manifold": (
                    "genuine_val_exact_same_episode_state_opposite_side_"
                    "complete_entry_envelope_swap"
                ),
                "targets": [
                    *_targets(
                        ("entry_decision_representation", tuple(token_indices)),
                        ("exit_path", tuple(path_indices)),
                    ),
                    _whole_target("exit_path_lengths"),
                    _whole_target("exit_side_index"),
                ],
            },
        ]

    logical = {
        "local_signal": local_rows,
        "ctx_cont": ctx_cont_rows,
        "ctx_cat": ctx_cat_rows,
        "mtf_fields": mtf_rows,
    }
    return {
        "task": task,
        "class_order": list(TASK_CLASS_ORDER[task]),
        "timeframes": list(timeframes),
        "source_influence_ownership_sha256": canonical_json_sha256(
            {
                "individual": individual_owner_layout,
                "unified_exit": owner_layout if task == "exit" else None,
            }
        ),
        "logical_fields": logical,
        "physical_field_perturbations": list(physical.values()),
        "family_tf_routes": route_rows,
        "local_family_effects": list(local_family_effects.values()),
        "joint_effects": list(joint_effects.values()),
        "interaction_synergy": interactions,
        "exit_episode_effects": episode_effects,
        "coverage_counts": {
            "local_signal": len(local_rows),
            "ctx_cont": len(ctx_cont_rows),
            "ctx_cat": len(ctx_cat_rows),
            "mtf_fields": len(mtf_rows),
            "physical_field_perturbations": len(physical),
            "family_tf_routes": len(route_rows),
            "local_family_effects": len(local_family_effects),
            "joint_interaction_effects": len(joint_effects),
            "interaction_synergy": len(interactions),
            "exit_episode_effects": len(episode_effects),
        },
    }


def feature_usefulness_layout(
    ordered_signal_names: Sequence[str],
) -> dict[str, Any]:
    signal_names = tuple(str(name) for name in ordered_signal_names)
    selected = signal_names[len(MODEL_NATIVE_BASE_FIELDS) :]
    if (
        len(signal_names) != MODEL_NATIVE_SIGNAL_DIM
        or signal_names != ordered_model_native_signal_fields(selected)
    ):
        raise RuntimeError("FEATURE_USEFULNESS_SIGNAL_ORDER_INVALID")
    payload = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "ordered_signal_names": list(signal_names),
        "ordered_signal_names_sha256": canonical_json_sha256(signal_names),
        "ctx_cont_names": list(MODEL_NATIVE_CTX_CONT_FIELDS),
        "ctx_cat_names": list(MODEL_NATIVE_CTX_CAT_FIELDS),
        "mtf_field_names": list(MULTI_TF_PER_BAR_FEATURES_V4),
        "tasks": {
            "entry": _task_layout(
                signal_names,
                task="entry",
                timeframes=ENTRY_MTF_CONTEXT_TIMEFRAMES,
            ),
            "exit": _task_layout(
                signal_names,
                task="exit",
                timeframes=EXIT_MTF_CONTEXT_TIMEFRAMES,
            ),
        },
    }
    payload["layout_sha256"] = canonical_json_sha256(payload)
    return payload


def require_feature_usefulness_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _IDENTITY_KEYS:
        raise RuntimeError("FEATURE_USEFULNESS_IDENTITY_SURFACE_INVALID")
    result = dict(value)
    for field in (
        "bundle_dir",
        "dataset_dir",
        "val_manifest_path",
        "val_data_path",
        "normalization_path",
    ):
        raw = result.get(field)
        if not isinstance(raw, str) or not raw.startswith("/") or "\x00" in raw:
            raise RuntimeError(f"FEATURE_USEFULNESS_IDENTITY_{field.upper()}_INVALID")
    for field in (
        "bundle_metadata_sha256",
        "model_state_sha256",
        "val_manifest_sha256",
        "val_data_sha256",
        "normalization_file_sha256",
        "normalization_contract_sha256",
        "entry_decision_token_snapshot_set_sha256",
        "unified_exit_input_envelope_set_sha256",
        "signal_static_contract_sha256",
    ):
        if not isinstance(result.get(field), str) or not _SHA256_RE.fullmatch(
            result[field]
        ):
            raise RuntimeError(f"FEATURE_USEFULNESS_IDENTITY_{field.upper()}_INVALID")
    if (
        result.get("contract_mode") != MODEL_NATIVE_CONTRACT_MODE
        or result.get("signal_schema_version") != MODEL_NATIVE_SIGNAL_SCHEMA_VERSION
        or result.get("signal_static_contract_sha256")
        != MODEL_NATIVE_STATIC_CONTRACT_SHA256
        or result.get("entry_decision_token_projection")
        != entry_decision_token_projection_metadata()
        or result.get("unified_exit_input_envelope_schema_version")
        != UNIFIED_EXIT_INPUT_ENVELOPE_SCHEMA_VERSION
    ):
        raise RuntimeError("FEATURE_USEFULNESS_IDENTITY_CONTRACT_INVALID")
    if not isinstance(result.get("dataset_run_id"), str) or not result["dataset_run_id"]:
        raise RuntimeError("FEATURE_USEFULNESS_IDENTITY_DATASET_RUN_ID_INVALID")
    for field in (
        "entry_val_population_row_count", "exit_val_population_row_count"
    ):
        count = result.get(field)
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise RuntimeError(
                f"FEATURE_USEFULNESS_IDENTITY_{field.upper()}_INVALID"
            )
    try:
        import pandas as pd

        start = pd.Timestamp(result["val_start_utc"])
        end = pd.Timestamp(result["val_end_utc"])
    except Exception as exc:
        raise RuntimeError("FEATURE_USEFULNESS_IDENTITY_VAL_WINDOW_INVALID") from exc
    if (
        start.tzinfo is None
        or end.tzinfo is None
        or start.utcoffset() != pd.Timedelta(0)
        or end.utcoffset() != pd.Timedelta(0)
        or start >= end
    ):
        raise RuntimeError("FEATURE_USEFULNESS_IDENTITY_VAL_WINDOW_INVALID")
    return result


def _require_summary(value: Any, *, count: int, label: str) -> None:
    keys = {
        "count",
        "sum",
        "mean",
        "sample_variance",
        "standard_error",
        "minimum",
        "maximum",
        "positive_count",
        "zero_count",
        "negative_count",
        "paired_vector_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != keys:
        raise RuntimeError(f"FEATURE_USEFULNESS_{label}_SUMMARY_SURFACE_INVALID")
    if value.get("count") != count or count < 1:
        raise RuntimeError(f"FEATURE_USEFULNESS_{label}_SUMMARY_COUNT_INVALID")
    for field in (
        "sum",
        "mean",
        "sample_variance",
        "standard_error",
        "minimum",
        "maximum",
    ):
        raw = value.get(field)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise RuntimeError(f"FEATURE_USEFULNESS_{label}_{field.upper()}_INVALID")
    counts = [value.get(field) for field in ("positive_count", "zero_count", "negative_count")]
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in counts) or sum(counts) != count:
        raise RuntimeError(f"FEATURE_USEFULNESS_{label}_SIGN_COUNTS_INVALID")
    if not isinstance(value.get("paired_vector_sha256"), str) or not _SHA256_RE.fullmatch(value["paired_vector_sha256"]):
        raise RuntimeError(f"FEATURE_USEFULNESS_{label}_VECTOR_HASH_INVALID")


def _require_metric(
    value: Any,
    *,
    loss_count: int,
    margin_count: int,
    physical_id: str,
    label: str,
) -> None:
    if not isinstance(value, Mapping) or set(value) != {
        "physical_id",
        "paired_loss_delta",
        "paired_margin_delta",
        "interpretation",
    }:
        raise RuntimeError(f"FEATURE_USEFULNESS_{label}_METRIC_SURFACE_INVALID")
    if value.get("physical_id") != physical_id:
        raise RuntimeError(f"FEATURE_USEFULNESS_{label}_PHYSICAL_ID_INVALID")
    _require_summary(
        value["paired_loss_delta"], count=loss_count, label=f"{label}_LOSS"
    )
    _require_summary(
        value["paired_margin_delta"], count=margin_count, label=f"{label}_MARGIN"
    )
    expected = (
        "non_positive_mean_on_both_raw_paired_metrics"
        if float(value["paired_loss_delta"]["mean"]) <= 0.0
        and float(value["paired_margin_delta"]["mean"]) <= 0.0
        else "mixed_or_positive_raw_paired_evidence"
    )
    if value.get("interpretation") != expected:
        raise RuntimeError(f"FEATURE_USEFULNESS_{label}_INTERPRETATION_INVALID")


def require_feature_usefulness_report(value: Mapping[str, Any]) -> dict[str, Any]:
    exact_keys = {
        "schema_version",
        "created_utc",
        "decision",
        "split",
        "policy",
        "perturbation_policy",
        "identity",
        "identity_sha256",
        "layout_sha256",
        "layout_counts",
        "tasks",
        "test_rows_read",
        "test_artifacts_read",
        "report_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != exact_keys:
        raise RuntimeError("FEATURE_USEFULNESS_REPORT_SURFACE_INVALID")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("decision") != DECISION
        or value.get("split") != SPLIT
        or value.get("policy") != POLICY
        or value.get("perturbation_policy") != PERTURBATION_POLICY
        or value.get("test_rows_read") is not False
        or value.get("test_artifacts_read") != []
    ):
        raise RuntimeError("FEATURE_USEFULNESS_REPORT_POLICY_INVALID")
    identity = require_feature_usefulness_identity(value["identity"])
    if value.get("identity_sha256") != canonical_json_sha256(identity):
        raise RuntimeError("FEATURE_USEFULNESS_IDENTITY_HASH_INVALID")
    try:
        import pandas as pd

        created = pd.Timestamp(value["created_utc"])
    except Exception as exc:
        raise RuntimeError("FEATURE_USEFULNESS_CREATED_UTC_INVALID") from exc
    if created.tzinfo is None or created.utcoffset() != pd.Timedelta(0):
        raise RuntimeError("FEATURE_USEFULNESS_CREATED_UTC_INVALID")
    tasks = value.get("tasks")
    if not isinstance(tasks, Mapping) or set(tasks) != set(TASKS):
        raise RuntimeError("FEATURE_USEFULNESS_TASK_SET_INVALID")
    entry_task = tasks.get("entry")
    if not isinstance(entry_task, Mapping):
        raise RuntimeError("FEATURE_USEFULNESS_ENTRY_SURFACE_INVALID")
    layout = feature_usefulness_layout(entry_task.get("ordered_signal_names", ()))
    if value.get("layout_sha256") != layout["layout_sha256"]:
        raise RuntimeError("FEATURE_USEFULNESS_LAYOUT_HASH_INVALID")
    expected_layout_counts = {
        task: layout["tasks"][task]["coverage_counts"] for task in TASKS
    }
    if value.get("layout_counts") != expected_layout_counts:
        raise RuntimeError("FEATURE_USEFULNESS_LAYOUT_COUNTS_INVALID")
    for task in TASKS:
        row = tasks[task]
        task_layout = layout["tasks"][task]
        expected_keys = {
            "ordered_signal_names",
            "row_count",
            "class_order",
            "comparison_surface",
            "row_times_sha256",
            "supervision",
            "baseline_outputs_sha256",
            "frozen_entry_decision_token_sha256",
            "donor_plan",
            "side_pair_plan",
            "forward_variant_count",
            "logical_field_metrics",
            "family_tf_route_metrics",
            "exit_episode_effect_metrics",
            "interaction_synergy_metrics",
            "coverage",
        }
        if not isinstance(row, Mapping) or set(row) != expected_keys:
            raise RuntimeError(f"FEATURE_USEFULNESS_{task.upper()}_SURFACE_INVALID")
        row_count = row.get("row_count")
        expected_surface = (
            "raw_entry_action_q_bps_valid_action_masked_mse_and_unique_target_q_margin"
            if task == "entry"
            else "raw_exit_action_q_bps_frozen_fitted_q_bellman_target_masked_mse_and_unique_target_q_margin"
        )
        if (
            row.get("ordered_signal_names") != layout["ordered_signal_names"]
            or isinstance(row_count, bool)
            or not isinstance(row_count, int)
            or row_count < 2
            or row_count
            != identity[f"{task}_val_population_row_count"]
            or row.get("class_order") != list(TASK_CLASS_ORDER[task])
            or row.get("comparison_surface") != expected_surface
        ):
            raise RuntimeError(f"FEATURE_USEFULNESS_{task.upper()}_ROWS_INVALID")
        for field in ("row_times_sha256", "baseline_outputs_sha256"):
            if not isinstance(row.get(field), str) or not _SHA256_RE.fullmatch(row[field]):
                raise RuntimeError(
                    f"FEATURE_USEFULNESS_{task.upper()}_{field.upper()}_INVALID"
                )
        supervision = row.get("supervision")
        common_supervision = {
            "fitted_q_contract",
            "fitted_q_iteration_state",
            "fitted_q_iteration_state_sha256",
            "q_targets_bps_sha256",
            "action_valid_mask_sha256",
            "action_equivalence_mask_sha256",
            "loss_valid_row_count",
            "margin_valid_row_count",
            "action_valid_cell_count",
            "target_tied_row_count",
            "single_valid_action_row_count",
        }
        if task == "entry":
            expected_supervision_keys = {
                "schema_version",
                "exit_fitted_q_iteration_state",
                "exit_fitted_q_iteration_state_sha256",
                *common_supervision,
            }
            hashes = (
                "fitted_q_iteration_state_sha256",
                "exit_fitted_q_iteration_state_sha256",
                "q_targets_bps_sha256",
                "action_valid_mask_sha256",
                "action_equivalence_mask_sha256",
            )
            if (
                not isinstance(supervision, Mapping)
                or set(supervision) != expected_supervision_keys
                or supervision.get("schema_version")
                != ENTRY_FITTED_Q_SCHEMA_VERSION
                or supervision.get("fitted_q_contract")
                != entry_fitted_q_contract()
                or row.get("frozen_entry_decision_token_sha256") is not None
                or row.get("side_pair_plan") is not None
            ):
                raise RuntimeError("FEATURE_USEFULNESS_ENTRY_SUPERVISION_INVALID")
            exit_iteration = require_unified_exit_fitted_q_iteration_state(
                supervision["exit_fitted_q_iteration_state"],
                context="FEATURE_USEFULNESS_ENTRY_EXIT_TEACHER",
            )
            entry_iteration = require_entry_fitted_q_iteration_state(
                supervision["fitted_q_iteration_state"],
                exit_fitted_q_iteration_state=exit_iteration,
                context="FEATURE_USEFULNESS_ENTRY",
            )
            if (
                supervision["fitted_q_iteration_state_sha256"]
                != canonical_json_sha256(entry_iteration)
                or supervision["exit_fitted_q_iteration_state_sha256"]
                != canonical_json_sha256(exit_iteration)
            ):
                raise RuntimeError(
                    "FEATURE_USEFULNESS_ENTRY_ITERATION_BINDING_INVALID"
                )
        else:
            expected_supervision_keys = {
                "schema_version",
                "terminal_mask_sha256",
                "terminal_row_count",
                *common_supervision,
            }
            hashes = (
                "q_targets_bps_sha256",
                "action_valid_mask_sha256",
                "action_equivalence_mask_sha256",
                "terminal_mask_sha256",
            )
            if (
                not isinstance(supervision, Mapping)
                or set(supervision) != expected_supervision_keys
                or supervision.get("schema_version")
                != UNIFIED_EXIT_FITTED_Q_SCHEMA_VERSION
                or supervision.get("fitted_q_contract")
                != unified_exit_fitted_q_contract()
                or not isinstance(row.get("frozen_entry_decision_token_sha256"), str)
                or not _SHA256_RE.fullmatch(row["frozen_entry_decision_token_sha256"])
            ):
                raise RuntimeError("FEATURE_USEFULNESS_EXIT_SUPERVISION_INVALID")
            exit_iteration = require_unified_exit_fitted_q_iteration_state(
                supervision["fitted_q_iteration_state"],
                context="FEATURE_USEFULNESS_EXIT",
            )
            if (
                supervision["fitted_q_iteration_state_sha256"]
                != canonical_json_sha256(exit_iteration)
            ):
                raise RuntimeError(
                    "FEATURE_USEFULNESS_EXIT_ITERATION_BINDING_INVALID"
                )
        for field in hashes:
            if not isinstance(supervision.get(field), str) or not _SHA256_RE.fullmatch(
                supervision[field]
            ):
                raise RuntimeError(
                    f"FEATURE_USEFULNESS_{task.upper()}_SUPERVISION_HASH_INVALID"
                )
        loss_count = supervision.get("loss_valid_row_count")
        margin_count = supervision.get("margin_valid_row_count")
        valid_cell_count = supervision.get("action_valid_cell_count")
        tied_count = supervision.get("target_tied_row_count")
        single_valid_count = supervision.get("single_valid_action_row_count")
        if (
            isinstance(loss_count, bool)
            or not isinstance(loss_count, int)
            or not 1 <= loss_count <= row_count
            or isinstance(margin_count, bool)
            or not isinstance(margin_count, int)
            or not 1 <= margin_count <= loss_count
            or isinstance(valid_cell_count, bool)
            or not isinstance(valid_cell_count, int)
            or not row_count <= valid_cell_count <= row_count * len(TASK_CLASS_ORDER[task])
            or isinstance(tied_count, bool)
            or not isinstance(tied_count, int)
            or not 0 <= tied_count <= row_count
            or isinstance(single_valid_count, bool)
            or not isinstance(single_valid_count, int)
            or not 0 <= single_valid_count <= row_count
        ):
            raise RuntimeError(
                f"FEATURE_USEFULNESS_{task.upper()}_SUPERVISION_COUNTS_INVALID"
            )
        if task == "exit":
            terminal_count = supervision.get("terminal_row_count")
            if (
                isinstance(terminal_count, bool)
                or not isinstance(terminal_count, int)
                or not 1 <= terminal_count <= row_count
                or terminal_count != single_valid_count
            ):
                raise RuntimeError(
                    "FEATURE_USEFULNESS_EXIT_SUPERVISION_COUNTS_INVALID"
                )

        plan = row.get("donor_plan")
        plan_keys = {
            "schema_version", "row_count", "block_count",
            "signature_group_count", "label_independent", "source_fields",
            "block_ids_sha256", "within_block_positions_sha256",
            "donor_indices_sha256", "all_rows_deranged",
            "whole_equal_geometry_blocks_preserved", "block_mapping_sha256",
            "plan_sha256",
        }
        if (
            not isinstance(plan, Mapping)
            or set(plan) != plan_keys
            or plan.get("schema_version") != DONOR_PLAN_SCHEMA_VERSION
            or plan.get("row_count") != row_count
            or not isinstance(plan.get("block_count"), int)
            or plan["block_count"] < 2
            or not isinstance(plan.get("signature_group_count"), int)
            or plan["signature_group_count"] < 1
            or plan.get("label_independent") is not True
            or plan.get("source_fields")
            != ["structure_block_id", "within_block_position"]
            or plan.get("all_rows_deranged") is not True
            or plan.get("whole_equal_geometry_blocks_preserved") is not True
        ):
            raise RuntimeError(f"FEATURE_USEFULNESS_{task.upper()}_DONOR_PLAN_INVALID")
        for field in (
            "block_ids_sha256", "within_block_positions_sha256",
            "donor_indices_sha256", "block_mapping_sha256",
        ):
            if not isinstance(plan.get(field), str) or not _SHA256_RE.fullmatch(plan[field]):
                raise RuntimeError(
                    f"FEATURE_USEFULNESS_{task.upper()}_DONOR_PLAN_HASH_INVALID"
                )
        unsigned_plan = dict(plan)
        plan_sha = unsigned_plan.pop("plan_sha256")
        if plan_sha != canonical_json_sha256(unsigned_plan):
            raise RuntimeError(
                f"FEATURE_USEFULNESS_{task.upper()}_DONOR_PLAN_BINDING_INVALID"
            )

        side_plan = row.get("side_pair_plan")
        if task == "exit":
            side_keys = {
                "schema_version", "row_count", "source_fields",
                "pair_indices_sha256", "episode_indices_sha256",
                "state_indices_sha256", "side_indices_sha256",
                "involutive", "same_episode_state", "opposite_side",
                "plan_sha256",
            }
            if (
                not isinstance(side_plan, Mapping)
                or set(side_plan) != side_keys
                or side_plan.get("schema_version") != SIDE_PAIR_PLAN_SCHEMA_VERSION
                or side_plan.get("row_count") != row_count
                or side_plan.get("source_fields")
                != ["exit_episode_index", "exit_state_index", "exit_side_index"]
                or side_plan.get("involutive") is not True
                or side_plan.get("same_episode_state") is not True
                or side_plan.get("opposite_side") is not True
            ):
                raise RuntimeError("FEATURE_USEFULNESS_EXIT_SIDE_PAIR_PLAN_INVALID")
            for field in (
                "pair_indices_sha256", "episode_indices_sha256",
                "state_indices_sha256", "side_indices_sha256",
            ):
                if not isinstance(side_plan.get(field), str) or not _SHA256_RE.fullmatch(
                    side_plan[field]
                ):
                    raise RuntimeError(
                        "FEATURE_USEFULNESS_EXIT_SIDE_PAIR_PLAN_HASH_INVALID"
                    )
            unsigned_side = dict(side_plan)
            side_sha = unsigned_side.pop("plan_sha256")
            if side_sha != canonical_json_sha256(unsigned_side):
                raise RuntimeError(
                    "FEATURE_USEFULNESS_EXIT_SIDE_PAIR_PLAN_BINDING_INVALID"
                )

        expected_forward_variants = (
            1
            + len(task_layout["physical_field_perturbations"])
            + len(task_layout["family_tf_routes"])
            + len(task_layout["local_family_effects"])
            + len(task_layout["joint_effects"])
            + len(task_layout["exit_episode_effects"])
        )
        if row.get("forward_variant_count") != expected_forward_variants:
            raise RuntimeError(
                f"FEATURE_USEFULNESS_{task.upper()}_FORWARD_VARIANTS_INVALID"
            )
        logical = row.get("logical_field_metrics")
        expected_logical = task_layout["logical_fields"]
        if not isinstance(logical, Mapping) or set(logical) != set(expected_logical):
            raise RuntimeError(
                f"FEATURE_USEFULNESS_{task.upper()}_LOGICAL_GROUPS_INVALID"
            )
        for group, expected_rows in expected_logical.items():
            expected_by_token = {item["token"]: item for item in expected_rows}
            observed = logical[group]
            if not isinstance(observed, Mapping) or set(observed) != set(expected_by_token):
                raise RuntimeError(
                    f"FEATURE_USEFULNESS_{task.upper()}_{group.upper()}_COVERAGE_INVALID"
                )
            for token, expected in expected_by_token.items():
                _require_metric(
                    observed[token], loss_count=loss_count,
                    margin_count=margin_count, physical_id=expected["physical_id"],
                    label=f"{task}_{group}_{token}",
                )
        routes = row.get("family_tf_route_metrics")
        expected_routes = {item["token"]: item for item in task_layout["family_tf_routes"]}
        if not isinstance(routes, Mapping) or set(routes) != set(expected_routes):
            raise RuntimeError(
                f"FEATURE_USEFULNESS_{task.upper()}_ROUTE_COVERAGE_INVALID"
            )
        for token, expected in expected_routes.items():
            _require_metric(
                routes[token], loss_count=loss_count, margin_count=margin_count,
                physical_id=expected["physical_id"], label=f"{task}_route_{token}",
            )
        episode = row.get("exit_episode_effect_metrics")
        expected_episode = {
            item["token"]: item for item in task_layout["exit_episode_effects"]
        }
        if not isinstance(episode, Mapping) or set(episode) != set(expected_episode):
            raise RuntimeError(
                f"FEATURE_USEFULNESS_{task.upper()}_EPISODE_COVERAGE_INVALID"
            )
        for token, expected in expected_episode.items():
            _require_metric(
                episode[token], loss_count=loss_count, margin_count=margin_count,
                physical_id=expected["physical_id"],
                label=f"{task}_episode_{token}",
            )
        synergy = row.get("interaction_synergy_metrics")
        expected_synergy = {
            item["token"]: item for item in task_layout["interaction_synergy"]
        }
        if not isinstance(synergy, Mapping) or set(synergy) != set(expected_synergy):
            raise RuntimeError(
                f"FEATURE_USEFULNESS_{task.upper()}_SYNERGY_COVERAGE_INVALID"
            )
        for token, expected in expected_synergy.items():
            metric = synergy[token]
            exact = {
                "kind", "formula", "left_effect_id", "right_effect_id",
                "joint_effect_id", "left_effect", "right_effect",
                "joint_effect", "paired_loss_delta", "paired_margin_delta",
            }
            identity_fields = (
                "kind", "formula", "left_effect_id", "right_effect_id",
                "joint_effect_id",
            )
            if (
                not isinstance(metric, Mapping)
                or set(metric) != exact
                or any(metric.get(field) != expected[field] for field in identity_fields)
            ):
                raise RuntimeError(
                    f"FEATURE_USEFULNESS_{task.upper()}_{token}_SYNERGY_INVALID"
                )
            for name in ("left", "right", "joint"):
                _require_metric(
                    metric[f"{name}_effect"], loss_count=loss_count,
                    margin_count=margin_count,
                    physical_id=expected[f"{name}_effect_id"],
                    label=f"{task}_{token}_{name.upper()}_EFFECT",
                )
            _require_summary(
                metric["paired_loss_delta"], count=loss_count,
                label=f"{task}_{token}_LOSS",
            )
            _require_summary(
                metric["paired_margin_delta"], count=margin_count,
                label=f"{task}_{token}_MARGIN",
            )
        expected_coverage = {
            **task_layout["coverage_counts"],
            "reported_logical_fields": sum(
                len(items) for items in expected_logical.values()
            ),
            "reported_family_tf_routes": len(expected_routes),
            "reported_exit_episode_effects": len(expected_episode),
            "reported_interaction_synergy": len(expected_synergy),
            "omitted_tokens": [],
            "complete": True,
        }
        if row.get("coverage") != expected_coverage:
            raise RuntimeError(f"FEATURE_USEFULNESS_{task.upper()}_COVERAGE_INVALID")
    entry_exit_teacher_sha = tasks["entry"]["supervision"][
        "exit_fitted_q_iteration_state_sha256"
    ]
    audited_exit_teacher_sha = tasks["exit"]["supervision"][
        "fitted_q_iteration_state_sha256"
    ]
    if entry_exit_teacher_sha != audited_exit_teacher_sha:
        raise RuntimeError(
            "FEATURE_USEFULNESS_ENTRY_EXIT_ITERATION_SPLIT_BRAIN"
        )
    unsigned = dict(value)
    report_sha = unsigned.pop("report_sha256")
    if report_sha != canonical_json_sha256(unsigned):
        raise RuntimeError("FEATURE_USEFULNESS_REPORT_HASH_INVALID")
    return dict(value)


__all__ = [
    "DECISION",
    "DONOR_PLAN_SCHEMA_VERSION",
    "LAYOUT_SCHEMA_VERSION",
    "PERTURBATION_POLICY",
    "POLICY",
    "SCHEMA_VERSION",
    "SPLIT",
    "TASKS",
    "TASK_CLASS_ORDER",
    "canonical_json_sha256",
    "feature_usefulness_layout",
    "require_feature_usefulness_identity",
    "require_feature_usefulness_report",
]
