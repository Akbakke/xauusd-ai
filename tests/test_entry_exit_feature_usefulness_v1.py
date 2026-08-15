from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gx1.contracts.entry_decision_token_v1 import (
    entry_decision_token_projection_metadata,
)
from gx1.contracts.entry_exit_feature_usefulness_v1 import (
    DECISION,
    POLICY,
    canonical_json_sha256,
    feature_usefulness_layout,
    require_feature_usefulness_report,
)
from gx1.contracts.entry_fitted_q_v1 import (
    ENTRY_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
    ENTRY_FITTED_Q_SCHEMA_VERSION,
    entry_fitted_q_contract,
)
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS,
    MTF_SEMANTIC_CATEGORICAL_DOMAINS,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    MODEL_NATIVE_CONTRACT_MODE,
    MODEL_NATIVE_CTX_CAT_DOMAINS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
    MODEL_NATIVE_STATIC_CONTRACT_SHA256,
    ordered_model_native_signal_fields,
)
from gx1.contracts.unified_exit_input_v1 import (
    UNIFIED_EXIT_INPUT_ENVELOPE_SCHEMA_VERSION,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    UNIFIED_EXIT_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
    UNIFIED_EXIT_FITTED_Q_SCHEMA_VERSION,
    unified_exit_fitted_q_contract,
)
from gx1.models.entry_v10.direction_decision_contract import (
    UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM,
    UNIFIED_EXIT_PATH_FEATURE_ORDER,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    require_multi_tf_specialist_routing_v4,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
from gx1.scripts.audit_entry_exit_feature_usefulness_v1 import (
    _build_exit_side_pair_plan,
    _fitted_q_loss_and_unique_target_margin,
    audit_task_feature_usefulness,
    build_feature_usefulness_report,
    build_structure_preserving_donor_plan,
)


def _exit_fitted_q_iteration() -> dict[str, object]:
    return {
        "schema_version": UNIFIED_EXIT_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
        "iteration_index": 4,
        "target_model_state_sha256": "a" * 64,
        "train_split_sha256": "b" * 64,
        "train_fold_sha256": "c" * 64,
        "source_lineage_sha256": "d" * 64,
        "normalization_sha256": "e" * 64,
        "fitted_q_contract": unified_exit_fitted_q_contract(),
        "target_updated_from_val_or_test": False,
    }


def _entry_fitted_q_iteration() -> dict[str, object]:
    exit_iteration = _exit_fitted_q_iteration()
    return {
        "schema_version": ENTRY_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
        "iteration_index": 4,
        "entry_target_model_state_sha256": "a" * 64,
        "exit_target_model_state_sha256": "a" * 64,
        "exit_fitted_q_iteration_state_sha256": canonical_json_sha256(
            exit_iteration
        ),
        "train_split_sha256": "b" * 64,
        "train_fold_sha256": "c" * 64,
        "source_lineage_sha256": "d" * 64,
        "normalization_sha256": "e" * 64,
        "entry_fitted_q_contract": entry_fitted_q_contract(),
        "exit_fitted_q_contract": unified_exit_fitted_q_contract(),
        "target_updated_from_val_or_test": False,
    }


def _signal_names() -> tuple[str, ...]:
    return ordered_model_native_signal_fields(
        [
            *MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
            *MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
        ]
    )


def _identity(tmp_path: Path) -> dict[str, object]:
    return {
        "bundle_dir": str((tmp_path / "bundle").resolve()),
        "bundle_metadata_sha256": "1" * 64,
        "model_state_sha256": "2" * 64,
        "dataset_dir": str((tmp_path / "dataset").resolve()),
        "dataset_run_id": "FEATURE_USEFULNESS_SYNTHETIC_V1",
        "val_manifest_path": str((tmp_path / "dataset/val.manifest.json").resolve()),
        "val_manifest_sha256": "3" * 64,
        "val_data_path": str((tmp_path / "dataset/val.parquet").resolve()),
        "val_data_sha256": "4" * 64,
        "val_start_utc": "2026-01-01T00:00:00+00:00",
        "val_end_utc": "2026-01-02T00:00:00+00:00",
        "entry_val_population_row_count": 8,
        "exit_val_population_row_count": 8,
        "normalization_path": str((tmp_path / "bundle/input_normalization.json").resolve()),
        "normalization_file_sha256": "5" * 64,
        "normalization_contract_sha256": "6" * 64,
        "entry_decision_token_snapshot_set_sha256": "7" * 64,
        "unified_exit_input_envelope_set_sha256": "8" * 64,
        "contract_mode": MODEL_NATIVE_CONTRACT_MODE,
        "signal_schema_version": MODEL_NATIVE_SIGNAL_SCHEMA_VERSION,
        "signal_static_contract_sha256": MODEL_NATIVE_STATIC_CONTRACT_SHA256,
        "entry_decision_token_projection": entry_decision_token_projection_metadata(),
        "unified_exit_input_envelope_schema_version": (
            UNIFIED_EXIT_INPUT_ENVELOPE_SCHEMA_VERSION
        ),
    }


def _states(timeframes: tuple[str, ...]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    names = _signal_names()
    rows = 8
    block_sign = np.repeat(np.array([1.0, -1.0, 1.0, -1.0]), 2)
    row = np.arange(rows, dtype=np.float32)
    seq = np.empty((rows, 3, len(names)), dtype=np.float32)
    for index in range(len(names)):
        seq[:, :, index] = (
            row[:, None] * np.float32(0.01 + index / 10000.0)
            + np.arange(3, dtype=np.float32)[None, :] * 0.001
            + np.float32(index / 100.0)
        )
    useful_index = names.index("_v1_atr14")
    noise_index = names.index("atr_z")
    seq[:, :, useful_index] = block_sign[:, None] * np.array(
        [1.6, 1.8, 2.0], dtype=np.float32
    )[None, :]
    snap = seq[:, -1, :].copy()

    ctx = np.empty((rows, len(MODEL_NATIVE_CTX_CONT_FIELDS)), dtype=np.float32)
    for index, field in enumerate(MODEL_NATIVE_CTX_CONT_FIELDS):
        if field in CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS:
            domain = CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS[field]
            ctx[:, index] = np.asarray(
                [domain[position % len(domain)] for position in range(rows)],
                dtype=np.float32,
            )
        else:
            ctx[:, index] = row * np.float32(0.03 + index / 1000.0)
        signal_index = names.index(f"ctx_cont.{field}")
        seq[:, -1, signal_index] = ctx[:, index]
        snap[:, signal_index] = ctx[:, index]
    ctx_cat = np.column_stack(
        [
            np.asarray(
                [domain[position % len(domain)] for position in range(rows)],
                dtype=np.int64,
            )
            for domain in MODEL_NATIVE_CTX_CAT_DOMAINS.values()
        ]
    )

    routing = require_multi_tf_specialist_routing_v4(MULTI_TF_PER_BAR_FEATURES_V4)
    trend_index = routing["trend_ema_encoder"][0]
    momentum_index = routing["momentum_flow_encoder"][0]
    mtf_index = {name: index for index, name in enumerate(MULTI_TF_PER_BAR_FEATURES_V4)}
    states: dict[str, np.ndarray] = {
        "seq_signal": seq,
        "snap_signal": snap,
        "ctx_cont": ctx,
        "ctx_cat": ctx_cat,
    }
    for timeframe in timeframes:
        values = np.empty(
            (rows, 2, len(MULTI_TF_PER_BAR_FEATURES_V4)), dtype=np.float32
        )
        for index in range(len(MULTI_TF_PER_BAR_FEATURES_V4)):
            values[:, :, index] = (
                row[:, None] * np.float32(0.02 + index / 10000.0)
                + np.arange(2, dtype=np.float32)[None, :] * 0.002
            )
        for field, domain in MTF_SEMANTIC_CATEGORICAL_DOMAINS.items():
            index = mtf_index[field]
            values[:, :, index] = np.asarray(
                [domain[position % len(domain)] for position in range(rows)],
                dtype=np.float32,
            )[:, None]
        if timeframe == "H1":
            values[:, :, trend_index] = block_sign[:, None]
            values[:, :, momentum_index] = block_sign[:, None]
        elif timeframe == "M15":
            values[:, :, momentum_index] = block_sign[:, None]
        states[f"seq_{timeframe.lower()}"] = values
    if "M5" in timeframes:
        episode = np.repeat(np.array([0, 1], dtype=np.int64), 4)
        state_index = np.tile(np.array([0, 7, 0, 7], dtype=np.int64), 2)
        side_index = np.tile(np.array([0, 0, 1, 1], dtype=np.int64), 2)
        token = np.zeros(
            (rows, UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM), dtype=np.float32
        )
        token[episode == 0, :] = 0.4
        token[episode == 1, :] = -0.3
        path = np.zeros(
            (rows, 4, len(UNIFIED_EXIT_PATH_FEATURE_ORDER)), dtype=np.float32
        )
        path[:, :, :] = row[:, None, None] * 0.01
        path[:, :, 0] += np.arange(4, dtype=np.float32)[None, :] * 0.1
        states.update(
            {
                "entry_decision_representation": token,
                "exit_path": path,
                "exit_path_lengths": np.tile(
                    np.array([2, 3, 2, 3], dtype=np.int64), 2
                ),
                "exit_side_index": side_index,
                "exit_episode_index": episode,
                "exit_state_index": state_index,
            }
        )
    return states, {
        "useful_index": useful_index,
        "noise_index": noise_index,
        "trend_index": trend_index,
        "momentum_index": momentum_index,
    }


class _SyntheticPredictor:
    def __init__(self, *, task: str, indices: dict[str, int], names: tuple[str, ...]):
        self.task = task
        self.indices = indices
        self.names = names
        self.calls = 0
        self.alias_signal_indices = np.asarray(
            [names.index(f"ctx_cont.{field}") for field in MODEL_NATIVE_CTX_CONT_FIELDS],
            dtype=np.int64,
        )
        self.alias_ctx_indices = np.arange(
            len(MODEL_NATIVE_CTX_CONT_FIELDS), dtype=np.int64
        )

    def __call__(self, states: dict[str, np.ndarray]) -> np.ndarray:
        self.calls += 1
        np.testing.assert_array_equal(
            states["seq_signal"][:, -1, self.alias_signal_indices],
            states["snap_signal"][:, self.alias_signal_indices],
        )
        np.testing.assert_array_equal(
            states["snap_signal"][:, self.alias_signal_indices],
            states["ctx_cont"][:, self.alias_ctx_indices],
        )
        for index, field in enumerate(MODEL_NATIVE_CTX_CAT_FIELDS):
            assert np.isin(states["ctx_cat"][:, index], MODEL_NATIVE_CTX_CAT_DOMAINS[field]).all()
        local = states["snap_signal"][:, self.indices["useful_index"]]
        interaction = (
            states["seq_h1"][:, -1, self.indices["trend_index"]]
            * states["seq_h1"][:, -1, self.indices["momentum_index"]]
        )
        score = 2.5 * local + 0.75 * interaction
        if self.task == "entry":
            return np.column_stack([score, -score, np.zeros_like(score)])
        score = (
            score
            + 0.3 * states["entry_decision_representation"][:, 0]
            + 0.2 * states["exit_path"][:, 0, 0]
            + 0.4 * (0.5 - states["exit_side_index"])
        )
        return np.column_stack([score, -score])


def _audit_task(tmp_path: Path, task: str) -> dict[str, object]:
    names = _signal_names()
    layout = feature_usefulness_layout(names)
    timeframes = tuple(layout["tasks"][task]["timeframes"])
    states, indices = _states(timeframes)
    predictor = _SyntheticPredictor(task=task, indices=indices, names=names)
    useful = states["snap_signal"][:, indices["useful_index"]]
    common = dict(
        task=task,
        ordered_signal_names=names,
        identity=_identity(tmp_path),
        states=states,
        row_times=pd.date_range("2026-01-01", periods=len(useful), freq="h", tz="UTC"),
        row_splits=["val"] * len(useful),
        block_ids=np.repeat(["day0", "day1", "day2", "day3"], 2),
        within_block_positions=np.tile([0, 1], 4),
        predictor=predictor,
        batch_rows=len(useful),
    )
    baseline_q_bps = predictor(states)
    if task == "entry":
        q_targets = np.asarray(baseline_q_bps, dtype=np.float64).copy()
        action_valid = np.ones((len(useful), 3), dtype=np.bool_)
        q_targets[0] = [0.0, 0.0, 0.0]
        equivalence = action_valid & np.equal(
            q_targets,
            np.max(
                np.where(action_valid, q_targets, -np.inf),
                axis=1,
                keepdims=True,
            ),
        )
        return audit_task_feature_usefulness(
            **common,
            entry_action_q_target_bps=q_targets,
            entry_action_valid_mask=action_valid,
            entry_action_equivalence_mask=equivalence,
            entry_fitted_q_iteration_state=_entry_fitted_q_iteration(),
            exit_fitted_q_iteration_state=_exit_fitted_q_iteration(),
        )
    q_targets = np.asarray(baseline_q_bps, dtype=np.float64).copy()
    action_valid = np.ones((len(useful), 2), dtype=np.bool_)
    terminal = states["exit_state_index"] == 7
    action_valid[terminal, 0] = False
    q_targets[0] = [0.0, 0.0]
    equivalence = action_valid & np.equal(
        q_targets,
        np.max(np.where(action_valid, q_targets, -np.inf), axis=1, keepdims=True),
    )
    return audit_task_feature_usefulness(
        **common,
        exit_action_q_target_bps=q_targets,
        exit_action_valid_mask=action_valid,
        exit_action_equivalence_mask=equivalence,
        exit_terminal_mask=terminal,
        exit_fitted_q_iteration_state=_exit_fitted_q_iteration(),
    )


@pytest.fixture(scope="module")
def usefulness_report(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    tmp_path = tmp_path_factory.mktemp("feature_usefulness")
    entry = _audit_task(tmp_path, "entry")
    exit_task = _audit_task(tmp_path, "exit")
    return build_feature_usefulness_report(
        identity=_identity(tmp_path),
        ordered_signal_names=_signal_names(),
        entry_task=entry,
        exit_task=exit_task,
    )


def test_layout_covers_every_logical_field_and_route_without_threshold() -> None:
    layout = feature_usefulness_layout(_signal_names())
    local_count = len(_signal_names())
    ctx_cont_count = len(MODEL_NATIVE_CTX_CONT_FIELDS)
    ctx_cat_count = len(MODEL_NATIVE_CTX_CAT_FIELDS)
    mtf_width = len(MULTI_TF_PER_BAR_FEATURES_V4)
    family_count = 8
    family_pairs = family_count * (family_count - 1) // 2
    for task, timeframe_count, exit_effect_count in (
        ("entry", 4, 0),
        ("exit", 5, 3),
    ):
        mtf_count = mtf_width * timeframe_count
        interaction_count = (
            family_pairs
            + timeframe_count * family_pairs
            + family_count * timeframe_count * (timeframe_count - 1) // 2
        )
        assert layout["tasks"][task]["coverage_counts"] == {
            "local_signal": local_count,
            "ctx_cont": ctx_cont_count,
            "ctx_cat": ctx_cat_count,
            "mtf_fields": mtf_count,
            "physical_field_perturbations": local_count + ctx_cat_count + mtf_count,
            "family_tf_routes": family_count * timeframe_count,
            "local_family_effects": family_count,
            "joint_interaction_effects": interaction_count,
            "interaction_synergy": interaction_count,
            "exit_episode_effects": exit_effect_count,
        }
    assert POLICY["automatic_importance_threshold"] is None
    assert POLICY["automatic_top_k"] is None
    assert POLICY["retirement_authority"] is False


def test_interactions_are_exhaustive_disjoint_owner_pairs() -> None:
    layout = feature_usefulness_layout(_signal_names())
    expected_kinds = {
        "entry": {
            "local_cross_family": 28,
            "per_tf_cross_family": 112,
            "cross_tf_same_family": 48,
        },
        "exit": {
            "local_cross_family": 28,
            "per_tf_cross_family": 140,
            "cross_tf_same_family": 80,
        },
    }
    for task in ("entry", "exit"):
        task_layout = layout["tasks"][task]
        effects = {
            row["physical_id"]: row
            for section in ("local_family_effects", "family_tf_routes")
            for row in task_layout[section]
        }
        observed = {kind: 0 for kind in expected_kinds[task]}
        for row in task_layout["interaction_synergy"]:
            assert row["formula"] == "joint_delta-left_delta-right_delta"
            observed[row["kind"]] += 1
            left = effects[row["left_effect_id"]]
            right = effects[row["right_effect_id"]]
            left_tokens = {
                (target["surface"], index)
                for target in left["targets"]
                for index in target["source_indices"]
            }
            right_tokens = {
                (target["surface"], index)
                for target in right["targets"]
                for index in target["source_indices"]
            }
            assert left_tokens.isdisjoint(right_tokens)
        assert observed == expected_kinds[task]


def test_synthetic_useful_noise_and_interaction_are_measured(
    usefulness_report: dict[str, object],
) -> None:
    checked = require_feature_usefulness_report(usefulness_report)
    assert checked["decision"] == DECISION
    for task in ("entry", "exit"):
        metrics = checked["tasks"][task]["logical_field_metrics"]["local_signal"]
        useful = metrics["local_signal._v1_atr14"]
        noise = metrics["local_signal.atr_z"]
        assert useful["paired_loss_delta"]["mean"] > 0.0
        assert useful["paired_margin_delta"]["mean"] > 0.0
        assert noise["paired_loss_delta"]["mean"] == 0.0
        assert noise["paired_margin_delta"]["mean"] == 0.0
        synergy = checked["tasks"][task]["interaction_synergy_metrics"][
            "interaction.h1.trend_ema_encoder__x__momentum_flow_encoder"
        ]
        assert synergy["formula"] == "joint_delta-left_delta-right_delta"
        assert synergy["paired_loss_delta"]["sample_variance"] > 0.0
        assert synergy["paired_margin_delta"]["sample_variance"] > 0.0
        assert (
            synergy["paired_loss_delta"]["positive_count"]
            + synergy["paired_loss_delta"]["negative_count"]
        ) > 0
    exit_episode = checked["tasks"]["exit"]["exit_episode_effect_metrics"]
    assert set(exit_episode) == {
        "exit_episode.frozen_entry_decision_token",
        "exit_episode.path_and_observed_length",
        "exit_episode.side_entry_binding",
    }
    assert checked["tasks"]["exit"]["side_pair_plan"]["opposite_side"] is True
    for metric in exit_episode.values():
        assert (
            metric["paired_loss_delta"]["positive_count"]
            + metric["paired_loss_delta"]["negative_count"]
        ) > 0


def test_supervision_binds_entry_and_exit_frozen_fitted_q_iterations(
    usefulness_report: dict[str, object],
) -> None:
    entry = usefulness_report["tasks"]["entry"]
    exit_task = usefulness_report["tasks"]["exit"]
    assert entry["comparison_surface"] == (
        "raw_entry_action_q_bps_valid_action_masked_mse_and_unique_target_q_margin"
    )
    assert exit_task["comparison_surface"] == (
        "raw_exit_action_q_bps_frozen_fitted_q_bellman_target_masked_mse_and_unique_target_q_margin"
    )
    entry_supervision = entry["supervision"]
    exit_supervision = exit_task["supervision"]
    assert entry_supervision["schema_version"] == ENTRY_FITTED_Q_SCHEMA_VERSION
    assert exit_supervision["schema_version"] == UNIFIED_EXIT_FITTED_Q_SCHEMA_VERSION
    assert entry_supervision["target_tied_row_count"] == 1
    assert exit_supervision["target_tied_row_count"] == 1
    assert entry_supervision["margin_valid_row_count"] < entry_supervision[
        "loss_valid_row_count"
    ]
    assert exit_supervision["margin_valid_row_count"] < exit_supervision[
        "loss_valid_row_count"
    ]
    assert entry_supervision["exit_fitted_q_iteration_state_sha256"] == (
        exit_supervision["fitted_q_iteration_state_sha256"]
    )


def test_masked_q_mse_and_unique_target_margin_exclude_ties() -> None:
    predicted_q = np.array([[1.0, 1.0], [0.4, -0.1], [7.0, 8.0]])
    q_targets = np.array([[1.0, 1.0], [2.0, 1.0], [7.0, 8.0]])
    valid = np.array([[True, True], [True, True], [False, True]])
    equivalent = np.array([[True, True], [True, False], [False, True]])
    loss, margin, margin_mask = _fitted_q_loss_and_unique_target_margin(
        predicted_q,
        q_targets_bps=q_targets,
        action_valid_mask=valid,
        action_equivalence_mask=equivalent,
    )
    assert loss[0] == 0.0
    assert loss[1] == pytest.approx(((0.4 - 2.0) ** 2 + (-0.1 - 1.0) ** 2) / 2.0)
    assert loss[2] == 0.0
    assert margin_mask.tolist() == [False, True, False]
    assert margin.tolist() == [pytest.approx(0.5)]
    invalid = equivalent.copy()
    invalid[1] = [False, True]
    with pytest.raises(RuntimeError, match="FITTED_Q_EQUIVALENCE_INVALID"):
        _fitted_q_loss_and_unique_target_margin(
            predicted_q,
            q_targets_bps=q_targets,
            action_valid_mask=valid,
            action_equivalence_mask=invalid,
        )

    entry_predicted = np.array([[1.5, 0.25, -0.5], [0.0, 0.0, 0.0]])
    entry_targets = np.array([[2.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    entry_valid = np.ones((2, 3), dtype=np.bool_)
    entry_equivalent = np.array([[True, False, False], [True, True, True]])
    entry_loss, entry_margin, entry_margin_mask = (
        _fitted_q_loss_and_unique_target_margin(
            entry_predicted,
            q_targets_bps=entry_targets,
            action_valid_mask=entry_valid,
            action_equivalence_mask=entry_equivalent,
        )
    )
    assert entry_loss.tolist() == pytest.approx(
        [((1.5 - 2.0) ** 2 + (0.25 - 1.0) ** 2 + (-0.5) ** 2) / 3.0, 0.0]
    )
    assert entry_margin_mask.tolist() == [True, False]
    assert entry_margin.tolist() == pytest.approx([1.25])


def test_iteration_state_or_val_test_target_update_mutation_fails_closed(
    usefulness_report: dict[str, object],
) -> None:
    entry_mutation = copy.deepcopy(usefulness_report)
    entry_mutation["tasks"]["entry"]["supervision"][
        "fitted_q_iteration_state"
    ]["target_updated_from_val_or_test"] = True
    with pytest.raises(RuntimeError, match="ENTRY_FITTED_Q"):
        require_feature_usefulness_report(entry_mutation)

    exit_mutation = copy.deepcopy(usefulness_report)
    exit_mutation["tasks"]["exit"]["supervision"][
        "fitted_q_iteration_state"
    ]["target_updated_from_val_or_test"] = True
    with pytest.raises(RuntimeError, match="UNIFIED_EXIT_FITTED_Q"):
        require_feature_usefulness_report(exit_mutation)

    split_brain = copy.deepcopy(usefulness_report)
    entry_supervision = split_brain["tasks"]["entry"]["supervision"]
    alternate_exit = entry_supervision["exit_fitted_q_iteration_state"]
    alternate_exit["iteration_index"] = 5
    alternate_exit["target_model_state_sha256"] = "f" * 64
    alternate_exit_sha = canonical_json_sha256(alternate_exit)
    alternate_entry = entry_supervision["fitted_q_iteration_state"]
    alternate_entry["iteration_index"] = 5
    alternate_entry["entry_target_model_state_sha256"] = "f" * 64
    alternate_entry["exit_target_model_state_sha256"] = "f" * 64
    alternate_entry["exit_fitted_q_iteration_state_sha256"] = alternate_exit_sha
    entry_supervision["exit_fitted_q_iteration_state_sha256"] = alternate_exit_sha
    entry_supervision["fitted_q_iteration_state_sha256"] = canonical_json_sha256(
        alternate_entry
    )
    with pytest.raises(RuntimeError, match="ITERATION_SPLIT_BRAIN"):
        require_feature_usefulness_report(split_brain)


def test_exit_frozen_token_and_same_state_side_pair_fail_closed() -> None:
    timeframes = tuple(
        feature_usefulness_layout(_signal_names())["tasks"]["exit"]["timeframes"]
    )
    states, _indices = _states(timeframes)
    pair, plan = _build_exit_side_pair_plan(states)
    assert np.array_equal(pair[pair], np.arange(len(pair)))
    assert plan["same_episode_state"] is True

    changed_token = {name: value.copy() for name, value in states.items()}
    changed_token["entry_decision_representation"][1, 0] += 1.0
    with pytest.raises(RuntimeError, match="FROZEN_TOKEN_NOT_EPISODE_IMMUTABLE"):
        _build_exit_side_pair_plan(changed_token)

    missing_side = {name: value.copy() for name, value in states.items()}
    missing_side["exit_side_index"][2] = 0
    with pytest.raises(RuntimeError, match="SIDE_PAIR_DUPLICATE"):
        _build_exit_side_pair_plan(missing_side)


def test_alias_and_categorical_perturbations_use_genuine_rows(
    usefulness_report: dict[str, object],
) -> None:
    layout = feature_usefulness_layout(_signal_names())
    alias = next(
        row
        for row in layout["tasks"]["entry"]["physical_field_perturbations"]
        if row["physical_id"].startswith("temporal_alias.")
    )
    assert alias["manifold"] == (
        "genuine_val_joint_seq_snap_ctx_temporal_alias_block_swap"
    )
    assert {target["surface"] for target in alias["targets"]} == {
        "seq_signal",
        "snap_signal",
        "ctx_cont",
    }
    assert usefulness_report["perturbation_policy"]["categorical"] == (
        "swap_observed_valid_category_never_synthesize"
    )
    episode = {
        row["token"]: row
        for row in layout["tasks"]["exit"]["exit_episode_effects"]
    }
    assert episode["exit_episode.frozen_entry_decision_token"]["targets"] == [
        {
            "surface": "entry_decision_representation",
            "source_indices": list(range(UNIFIED_EXIT_ENTRY_REPRESENTATION_DIM)),
        }
    ]
    assert episode["exit_episode.path_and_observed_length"]["targets"][-1] == {
        "surface": "exit_path_lengths",
        "whole_surface": True,
    }
    side_binding = episode["exit_episode.side_entry_binding"]
    assert side_binding["donor_kind"] == "same_state_opposite_side"
    assert {target["surface"] for target in side_binding["targets"]} == {
        "entry_decision_representation",
        "exit_path",
        "exit_path_lengths",
        "exit_side_index",
    }


def test_donor_plan_preserves_whole_equal_geometry_blocks() -> None:
    donor, plan = build_structure_preserving_donor_plan(
        block_ids=["a", "a", "b", "b", "c", "c", "d", "d"],
        within_block_positions=[0, 1, 0, 1, 0, 1, 0, 1],
    )
    assert donor.tolist() == [2, 3, 4, 5, 6, 7, 0, 1]
    assert plan["all_rows_deranged"] is True
    assert plan["whole_equal_geometry_blocks_preserved"] is True
    with pytest.raises(RuntimeError, match="STRUCTURE_HAS_NO_PEER"):
        build_structure_preserving_donor_plan(
            block_ids=["a", "a", "b"],
            within_block_positions=[0, 1, 0],
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("test_split", "NON_VAL_ROW_FORBIDDEN"),
        ("future_time", "FUTURE_OR_OUTSIDE_VAL_ROW_FORBIDDEN"),
        ("alias", "ALIAS_SOURCE_OFF_MANIFOLD"),
    ],
)
def test_test_future_and_off_manifold_alias_rows_fail_closed(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    names = _signal_names()
    layout = feature_usefulness_layout(names)
    timeframes = tuple(layout["tasks"]["entry"]["timeframes"])
    states, indices = _states(timeframes)
    predictor = _SyntheticPredictor(task="entry", indices=indices, names=names)
    q_targets = predictor(states)
    action_valid = np.ones_like(q_targets, dtype=np.bool_)
    action_equivalent = action_valid & np.equal(
        q_targets,
        np.max(np.where(action_valid, q_targets, -np.inf), axis=1, keepdims=True),
    )
    times = pd.date_range("2026-01-01", periods=len(q_targets), freq="h", tz="UTC")
    splits = ["val"] * len(q_targets)
    if mutation == "test_split":
        splits[-1] = "test"
    elif mutation == "future_time":
        times = times[:-1].append(pd.DatetimeIndex([pd.Timestamp("2026-01-03", tz="UTC")]))
    else:
        alias = layout["tasks"]["entry"]["physical_field_perturbations"][0]
        while alias["alias_ctx_cont_index"] is None:
            alias = layout["tasks"]["entry"]["physical_field_perturbations"][
                layout["tasks"]["entry"]["physical_field_perturbations"].index(alias) + 1
            ]
        states["ctx_cont"][0, alias["alias_ctx_cont_index"]] += 1.0
    with pytest.raises(RuntimeError, match=message):
        audit_task_feature_usefulness(
            task="entry",
            ordered_signal_names=names,
            identity=_identity(tmp_path),
            states=states,
            entry_action_q_target_bps=q_targets,
            entry_action_valid_mask=action_valid,
            entry_action_equivalence_mask=action_equivalent,
            entry_fitted_q_iteration_state=_entry_fitted_q_iteration(),
            exit_fitted_q_iteration_state=_exit_fitted_q_iteration(),
            row_times=times,
            row_splits=splits,
            block_ids=np.repeat(["day0", "day1", "day2", "day3"], 2),
            within_block_positions=np.tile([0, 1], 4),
            predictor=predictor,
            batch_rows=len(q_targets),
        )


def test_identity_metric_and_coverage_mutations_fail_closed(
    usefulness_report: dict[str, object],
) -> None:
    identity = copy.deepcopy(usefulness_report)
    identity["identity"]["bundle_metadata_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="IDENTITY_HASH_INVALID"):
        require_feature_usefulness_report(identity)

    coverage = copy.deepcopy(usefulness_report)
    coverage["tasks"]["entry"]["logical_field_metrics"]["local_signal"].pop(
        "local_signal.atr_z"
    )
    with pytest.raises(RuntimeError, match="LOCAL_SIGNAL_COVERAGE_INVALID"):
        require_feature_usefulness_report(coverage)

    zero_is_valid = copy.deepcopy(usefulness_report)
    require_feature_usefulness_report(zero_is_valid)
    assert zero_is_valid["tasks"]["entry"]["logical_field_metrics"][
        "local_signal"
    ]["local_signal.atr_z"]["interpretation"] == (
        "non_positive_mean_on_both_raw_paired_metrics"
    )


def test_source_has_no_selection_cutoff_or_hindsight_or_classification_authority() -> None:
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "gx1/contracts/entry_exit_feature_usefulness_v1.py",
            "gx1/scripts/audit_entry_exit_feature_usefulness_v1.py",
        )
    )
    for forbidden in (
        "exit_action_target",
        "exit_baseline_action_target",
        "unified_exit_optimal_stopping",
        "UNIFIED_EXIT_OPTIMAL_STOPPING",
        "finite_horizon_optimal",
        "pathwise_hindsight",
        "direction_logits",
        "exact_label_ce",
        "labels_sha256",
        "joint_delta-family_delta-timeframe_delta",
        "automatic_importance_threshold\": 0",
        "automatic_top_k\": 133",
    ):
        assert forbidden not in source
