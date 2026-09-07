"""Mechanical synthetic fixtures and real native-model arithmetic parity.

These tests do not load a trained bundle or VAL artifacts and cannot establish
production usefulness, provenance, admission or edge. Native parity uses the
actual model at owner-declared feature dimensions, not a fake Q predictor.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from gx1.contracts.entry_decision_token_v1 import ENTRY_DECISION_TOKEN_DIM
from gx1.contracts.entry_exit_feature_base_v1 import (
    ENTRY_MTF_CONTEXT_TIMEFRAMES,
    EXIT_FEATURE_SEQUENCE_BARS,
    EXIT_MTF_CONTEXT_TIMEFRAMES,
)
from gx1.contracts.entry_exit_feature_usefulness_v1 import feature_usefulness_layout
from gx1.contracts.entry_fitted_q_v1 import (
    ENTRY_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
    entry_fitted_q_contract,
)
from gx1.contracts.entry_model_native_train_launch_v1 import (
    RECIPE_AUDIT_SCHEMA,
    TRAINING_RECIPE_SOURCE_PROVENANCE_SCHEMA,
    artifact_binding,
    canonical_json_sha256,
)
from gx1.contracts.unified_exit_fitted_q_v1 import (
    UNIFIED_EXIT_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
    unified_exit_fitted_q_contract,
)
from gx1.contracts.entry_exit_production_architecture_v1 import (
    PRODUCTION_MTF_PER_TF_WINDOW_BARS,
)
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS,
    MODEL_NATIVE_CTX_CAT_DOMAINS,
    MODEL_NATIVE_CTX_CAT_FIELDS,
    MODEL_NATIVE_CTX_CONT_FIELDS,
    MODEL_NATIVE_MANDATORY_SELECTED_FIELDS,
    MODEL_NATIVE_SEQ_LEN,
    ordered_model_native_signal_fields,
)
from gx1.contracts import unified_exit_episode_pack_v1 as episode_owner
from gx1.features.entry_specialist_feature_groups_v1 import (
    MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT,
    MODEL_NATIVE_TRAINING_SPECIALISTS,
    classify_entry_specialist_feature,
    model_native_context_temporal_alias_policy,
    require_multi_tf_specialist_routing_v4,
)
from gx1.features.htf_features import MULTI_TF_PER_BAR_FEATURES_V4
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
    EntryV10CtxHybridTransformer,
)
from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
    _fitted_q_targets_for_episode,
    _fitted_q_targets_for_episode_batch,
    _forward_unified_exit_episode_pack,
)
from gx1.scripts.entry_exit_feature_usefulness_native_v1 import (
    CompactNativeExitUsefulnessAdapter,
)
from gx1.scripts import entry_exit_feature_usefulness_native_v1 as native_usefulness
from tests.model_native_input_normalization_support import input_normalization_fixture


_SIGNAL_NAMES = ordered_model_native_signal_fields(
    [*MODEL_NATIVE_MANDATORY_SELECTED_FIELDS, *MODEL_NATIVE_AVAILABLE_CANDIDATE_FIELDS]
)
_TF_LENGTHS = dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS)
_TF_NAMES = tuple(timeframe.lower() for timeframe in EXIT_MTF_CONTEXT_TIMEFRAMES)
_STATE_COUNT = episode_owner.UNIFIED_EXIT_EPISODE_STATE_COUNT
_WARM_ROWS = EXIT_FEATURE_SEQUENCE_BARS - 1


@pytest.fixture(scope="module")
def layout():
    return feature_usefulness_layout(_SIGNAL_NAMES)["tasks"]["exit"]


@pytest.fixture(scope="module")
def normalization():
    return input_normalization_fixture(
        signal_names=list(_SIGNAL_NAMES),
        mtf_names=list(MULTI_TF_PER_BAR_FEATURES_V4),
        per_tf_seq_lens=_TF_LENGTHS,
    )


def _surface(normalization, surface, rows, rng):
    contract = normalization["surfaces"][surface]
    names = contract["field_names"]
    values = rng.normal(size=(rows, len(names))).astype(np.float32)
    for index, is_binary in enumerate(contract["binary_mask"]):
        if is_binary:
            values[:, index] = rng.integers(0, 2, size=rows)
    for field, domain in contract["categorical_domains"].items():
        values[:, names.index(field)] = rng.choice(domain, size=rows)
    if surface.startswith("mtf_") and "ema_stack_aligned_v2" in names:
        values[:, names.index("ema_stack_aligned_v2")] = rng.choice((-1, 0, 1), size=rows)
    return values


def _episode(normalization, seed):
    rng = np.random.default_rng(seed)
    pack = {
        name: rng.normal(size=shape).astype(np.float32)
        for name, shape in episode_owner._FIXED_ARRAY_SHAPES.items()
    }
    local_rows = episode_owner.UNIFIED_EXIT_EPISODE_LOCAL_HISTORY_ROWS
    pack["exit_local_history_x"] = _surface(normalization, "signal", local_rows, rng)
    pack["exit_state_ctx_cont"] = _surface(normalization, "ctx_cont", _STATE_COUNT, rng)
    for alias in model_native_context_temporal_alias_policy(_SIGNAL_NAMES)["aliases"]:
        pack["exit_state_ctx_cont"][:, alias["ctx_cont_index"]] = (
            pack["exit_local_history_x"][_WARM_ROWS:, alias["signal_index"]]
        )
    pack["exit_state_ctx_cat"] = np.column_stack(
        [rng.choice(MODEL_NATIVE_CTX_CAT_DOMAINS[field], size=_STATE_COUNT)
         for field in MODEL_NATIVE_CTX_CAT_FIELDS]
    ).astype(np.int64)
    minute_ns = 60_000_000_000
    origin_ns = 1_700_000_000_000_000_000 + seed * 86400 * 1_000_000_000
    pack["exit_local_history_time_ns"] = (
        origin_ns + (np.arange(local_rows, dtype=np.int64) - _WARM_ROWS) * minute_ns
    )
    pack["exit_state_row_time_ns"] = pack["exit_local_history_time_ns"][_WARM_ROWS:].copy()
    pack["exit_decision_time_ns"] = pack["exit_state_row_time_ns"] + minute_ns
    pack["exit_state_valid_mask"] = np.ones((2, _STATE_COUNT), dtype=np.bool_)
    pack["exit_terminal_mask"] = np.zeros((2, _STATE_COUNT), dtype=np.bool_)
    pack["exit_terminal_mask"][:, -1] = True
    pack["exit_terminal_reason_index"] = pack["exit_terminal_mask"].astype(np.int64)
    pack["exit_action_valid_mask"] = np.stack(
        (~pack["exit_terminal_mask"], pack["exit_state_valid_mask"]), axis=-1
    )
    pack["exit_episode_lengths"] = np.full(2, _STATE_COUNT, dtype=np.int64)
    for timeframe, stride in zip(EXIT_MTF_CONTEXT_TIMEFRAMES, (5, 15, 60, 240, 1440)):
        lower = timeframe.lower()
        warm = _TF_LENGTHS[timeframe]
        gather = warm - 1 + np.arange(_STATE_COUNT, dtype=np.int64) // stride
        history_rows = int(gather[-1]) + 1
        pack[f"exit_mtf_history_{lower}"] = _surface(
            normalization, f"mtf_{lower}", history_rows, rng
        )
        pack[f"exit_mtf_gather_{lower}"] = gather
        pack[f"exit_mtf_history_time_ns_{lower}"] = (
            origin_ns + (np.arange(history_rows, dtype=np.int64) - warm + 1) * stride * minute_ns
        )
    pack.update(
        schema_version=episode_owner.UNIFIED_EXIT_EPISODE_PACK_SCHEMA_VERSION,
        entry_row_index=seed,
        episode_index_by_side=[2 * seed, 2 * seed + 1],
        m1_start_row=_WARM_ROWS + seed,
        lifecycle_state_population_sha256="a" * 64,
        multi_tf_cache_identity_sha256="b" * 64,
    )
    return _seal_fixture(pack)


def _seal_fixture(pack):
    unsealed = {name: value for name, value in pack.items() if name != "episode_pack_sha256"}
    return episode_owner.require_unified_exit_episode_pack(
        episode_owner.seal_unified_exit_episode_pack(unsealed),
        per_tf_seq_lens=_TF_LENGTHS,
        expected_mtf_cache_identity_sha256="b" * 64,
        context="SYNTHETIC_NATIVE_USEFULNESS_TEST",
    )


@pytest.fixture(scope="module")
def episodes(normalization):
    return _episode(normalization, 101), _episode(normalization, 102)


def _budget(episodes):
    return max(
        sum(value.nbytes for value in episode.values() if isinstance(value, np.ndarray))
        for episode in episodes
    ) + ENTRY_DECISION_TOKEN_DIM * np.dtype(np.float32).itemsize


def _adapter(episodes, model=None, budget=None):
    return CompactNativeExitUsefulnessAdapter(
        model=torch.nn.Module().eval() if model is None else model,
        ordered_signal_names=_SIGNAL_NAMES,
        device=torch.device("cpu"),
        max_episode_bytes=_budget(episodes) if budget is None else budget,
    )


def _token(value=0.25):
    return torch.full((1, ENTRY_DECISION_TOKEN_DIM), value, dtype=torch.float32)


def _spec(layout, kind):
    if kind == "alias":
        return next(spec for spec in layout["physical_field_perturbations"]
                    if spec.get("alias_signal_index") is not None)
    if kind in ("token", "path", "side"):
        return layout["exit_episode_effects"][("token", "path", "side").index(kind)]
    if kind in ("local_family", "joint"):
        return layout[{"local_family": "local_family_effects", "joint": "joint_effects"}[kind]][0]
    if kind == "family_tf":
        return layout["family_tf_routes"][0]
    if kind == "ctx_cont":
        physical_id = layout["logical_fields"]["ctx_cont"][0]["physical_id"]
        return next(spec for spec in layout["physical_field_perturbations"]
                    if spec["physical_id"] == physical_id)
    return next(spec for spec in layout["physical_field_perturbations"]
                if spec["physical_id"].startswith(kind + "."))


def _kwargs(episodes, layout, kind, token=None, donor_token=None):
    arguments = {
        "episode": episodes[0],
        "online_entry_token": _token() if token is None else token,
        "spec": _spec(layout, kind),
    }
    if kind != "side":
        arguments["donor_episode"] = episodes[1]
    if kind == "token":
        arguments["donor_online_entry_token"] = _token(-0.5) if donor_token is None else donor_token
    return arguments


@pytest.mark.parametrize(
    "kind", ("local_signal", "alias", "ctx_cont", "ctx_cat", "mtf",
             "local_family", "family_tf", "joint", "token", "path", "side")
)
def test_compact_exact_target_columns_and_source_preservation(episodes, layout, kind):
    recipient, donor = episodes
    before = [deepcopy(episode) for episode in episodes]
    arguments = _kwargs(episodes, layout, kind)
    arguments["online_entry_token"].requires_grad_()
    prepared = _adapter(episodes).prepare(**arguments)
    target_fields = {}
    surface_keys = {
        "seq_signal": "exit_local_history_x", "snap_signal": "exit_local_history_x",
        "ctx_cont": "exit_state_ctx_cont", "ctx_cat": "exit_state_ctx_cat",
        "exit_path": "exit_path_x",
        **{f"seq_{timeframe}": f"exit_mtf_history_{timeframe}" for timeframe in _TF_NAMES},
    }
    if kind != "side":
        for target in arguments["spec"]["targets"]:
            if target["surface"] in surface_keys:
                key = surface_keys[target["surface"]]
                target_fields.setdefault(key, set()).update(target["source_indices"])
    for key, actual in prepared.model_inputs.items():
        expected = recipient[key].copy()
        if key in target_fields:
            indices = sorted(target_fields[key])
            expected[..., indices] = donor[key][..., indices]
        np.testing.assert_array_equal(actual, expected)
        assert not np.shares_memory(actual, recipient[key])
    assert "seq_signal" not in prepared.model_inputs
    assert "snap_signal" not in prepared.model_inputs
    assert "exit_side_index" not in prepared.model_inputs
    assert "episode_pack_sha256" not in prepared.model_inputs
    assert "schema_version" not in prepared.model_inputs
    assert "exit_now_reward_bps" not in prepared.model_inputs
    assert prepared.source_episode_pack_sha256 == recipient["episode_pack_sha256"]
    assert prepared.swap_output_sides == (kind == "side")
    expected_token = arguments.get("donor_online_entry_token", arguments["online_entry_token"])
    assert torch.equal(prepared.online_entry_token, expected_token)
    assert prepared.online_entry_token.data_ptr() != expected_token.data_ptr()
    assert not prepared.online_entry_token.requires_grad
    assert arguments["online_entry_token"].grad is None
    for original, snapshot in zip(episodes, before):
        for key, value in original.items():
            if isinstance(value, np.ndarray):
                assert value.tobytes() == snapshot[key].tobytes()
            else:
                assert value == snapshot[key]


@pytest.mark.parametrize(
    "timeframe,mismatch",
    [(timeframe, "history_shape") for timeframe in _TF_NAMES]
    + [(timeframe, "gather") for timeframe in _TF_NAMES if timeframe != "d1"],
)
def test_partial_mtf_swap_rejects_geometry_without_changing_plan(episodes, layout, timeframe, mismatch):
    recipient, original_donor = episodes
    donor = deepcopy(original_donor)
    history_key = f"exit_mtf_history_{timeframe}"
    gather_key = f"exit_mtf_gather_{timeframe}"
    times_key = f"exit_mtf_history_time_ns_{timeframe}"
    if mismatch == "history_shape":
        donor[history_key] = np.concatenate((donor[history_key], donor[history_key][-1:]))
        donor[times_key] = np.append(donor[times_key], donor[times_key][-1] + 60_000_000_000)
        donor[gather_key][-1] += 1
    else:
        change = int(np.flatnonzero(np.diff(donor[gather_key]))[0])
        donor[gather_key][change] += 1
    donor = _seal_fixture(donor)
    spec = next(spec for spec in layout["family_tf_routes"]
                if spec["timeframe"].lower() == timeframe)
    plan = deepcopy(spec)
    with pytest.raises(RuntimeError, match="DONOR_MTF_GEOMETRY_MISMATCH"):
        _adapter((recipient, donor)).prepare(
            episode=recipient, online_entry_token=_token(), spec=spec, donor_episode=donor
        )
    assert spec == plan


def test_all_family_tf_routes_swap_whole_unique_history(episodes, layout):
    adapter = _adapter(episodes)
    recipient, donor = episodes
    for spec in layout["family_tf_routes"]:
        prepared = adapter.prepare(
            episode=recipient, online_entry_token=_token(), spec=spec, donor_episode=donor
        )
        key = f"exit_mtf_history_{spec['timeframe'].lower()}"
        indices = spec["source_indices"]
        np.testing.assert_array_equal(prepared.model_inputs[key][:, indices], donor[key][:, indices])
        assert prepared.model_inputs[key].shape == recipient[key].shape


@pytest.mark.parametrize("failure", ("budget", "prefix", "alias", "signed_zero_alias", "bad_spec", "token_dtype"))
def test_transport_fails_closed_before_forward(episodes, layout, failure):
    episode = deepcopy(episodes[0])
    token = _token().requires_grad_()
    adapter = _adapter(episodes)
    arguments = {"episode": episode, "online_entry_token": token}
    if failure == "budget":
        adapter = _adapter(episodes, budget=1)
        expected = "BYTE_BUDGET_EXCEEDED"
    elif failure == "prefix":
        episode["exit_local_history_x"] = episode["exit_local_history_x"][1:]
        expected = "COMPACT_SHAPE_INVALID"
    elif failure in ("alias", "signed_zero_alias"):
        alias = _spec(layout, "alias")
        signal_index, context_index = alias["alias_signal_index"], alias["alias_ctx_cont_index"]
        episode["exit_local_history_x"][_WARM_ROWS, signal_index] = np.float32(0.0)
        episode["exit_state_ctx_cont"][0, context_index] = np.float32(-0.0 if failure == "signed_zero_alias" else 1.0)
        expected = "ALIAS_OFF_MANIFOLD"
    elif failure == "bad_spec":
        spec = deepcopy(_spec(layout, "alias"))
        spec["targets"] = spec["targets"][:1]
        arguments.update(spec=spec, donor_episode=episodes[1])
        expected = "LAYOUT_SPEC_INVALID"
    else:
        arguments["online_entry_token"] = token.double()
        expected = "FROZEN_TOKEN_INVALID"
    with pytest.raises(RuntimeError, match=expected):
        adapter.prepare(**arguments)
    assert token.grad is None


def test_explicit_donor_and_side_plan_are_never_replaced(episodes, layout):
    adapter = _adapter(episodes)
    with pytest.raises(RuntimeError, match="EXPLICIT_DONOR_REQUIRED"):
        adapter.prepare(episode=episodes[0], online_entry_token=_token(), spec=_spec(layout, "path"))
    with pytest.raises(RuntimeError, match="EXTERNAL_DONOR_FORBIDDEN"):
        adapter.prepare(**{**_kwargs(episodes, layout, "side"), "donor_episode": episodes[1]})
    with pytest.raises(RuntimeError, match="SELF_DONOR_FORBIDDEN"):
        adapter.prepare(**{**_kwargs(episodes, layout, "path"), "donor_episode": episodes[0]})
    with pytest.raises(RuntimeError, match="FROZEN_TOKEN_INVALID"):
        arguments = _kwargs(episodes, layout, "token")
        arguments.pop("donor_online_entry_token")
        adapter.prepare(**arguments)


def test_intervention_has_no_new_seal_or_source_admission(episodes, layout):
    adapter = _adapter(episodes)
    prepared = adapter.prepare(**_kwargs(episodes, layout, "alias"))
    with pytest.raises(RuntimeError, match="SOURCE_SEAL_REQUIRED"):
        adapter.prepare(episode=prepared.model_inputs, online_entry_token=_token())
    with pytest.raises(RuntimeError, match="EPISODE_PACK_HASH_INVALID"):
        episode_owner.require_unified_exit_episode_pack(
            {**episodes[0], **prepared.model_inputs},
            per_tf_seq_lens=_TF_LENGTHS,
            expected_mtf_cache_identity_sha256="b" * 64,
            context="INTERVENTION_IS_NOT_A_REAL_SOURCE",
        )


def test_training_mode_is_rejected_without_changing_model(episodes):
    model = torch.nn.Module()
    adapter = _adapter(episodes, model)
    with pytest.raises(RuntimeError, match="MODEL_MUST_BE_EVAL"):
        adapter.predict_spec(episode=episodes[0], online_entry_token=_token())
    with pytest.raises(RuntimeError, match="TARGET_MODEL_MUST_BE_EVAL"):
        adapter.baseline_supervision(
            episode=episodes[0], target_model=model, target_entry_token=_token()
        )
    assert model.training


@pytest.fixture(scope="module")
def native_model(normalization):
    routing = {name: [] for name in MODEL_NATIVE_TRAINING_SPECIALISTS}
    for index, field in enumerate(_SIGNAL_NAMES):
        routing[classify_entry_specialist_feature(field)].append(index)
    context = MODEL_NATIVE_CONTEXT_SPECIALIST_ROUTING_CONTRACT
    aliases = model_native_context_temporal_alias_policy(_SIGNAL_NAMES)
    tf_arguments = {
        **{f"{timeframe}_seq_dim": len(MULTI_TF_PER_BAR_FEATURES_V4) for timeframe in _TF_NAMES},
        **{f"{timeframe.lower()}_seq_len": length for timeframe, length in _TF_LENGTHS.items()},
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260906)
        model = EntryV10CtxHybridTransformer(
            seq_input_dim=len(_SIGNAL_NAMES), snap_input_dim=len(_SIGNAL_NAMES),
            seq_len=MODEL_NATIVE_SEQ_LEN, dropout=0.0,
            ctx_cont_dim=len(MODEL_NATIVE_CTX_CONT_FIELDS),
            ctx_cat_dim=len(MODEL_NATIVE_CTX_CAT_FIELDS),
            multi_tf_num_layers=1, multi_tf_scale=0.5,
            specialist_num_layers=1, specialist_fusion_scale=0.25,
            cross_family_fusion_scale=0.25,
            specialist_input_indices=routing,
            multi_tf_specialist_input_indices={
                name: list(indices)
                for name, indices in require_multi_tf_specialist_routing_v4(MULTI_TF_PER_BAR_FEATURES_V4).items()
            },
            specialist_ctx_cont_indices={name: list(indices) for name, indices in context["ctx_cont_indices"].items()},
            specialist_ctx_cont_nominal_indices={name: list(indices) for name, indices in context["ctx_cont_nominal_indices"].items()},
            specialist_ctx_cat_indices={name: list(indices) for name, indices in context["ctx_cat_indices"].items()},
            temporal_alias_signal_indices=aliases["signal_indices"],
            temporal_alias_ctx_cont_indices=aliases["ctx_cont_indices"],
            input_normalization=normalization,
            **tf_arguments,
        ).eval()
    return model


@pytest.fixture(scope="module")
def online_tokens(native_model, normalization):
    tokens = []
    for seed in (201, 202):
        rng = np.random.default_rng(seed)
        sequence = _surface(normalization, "signal", MODEL_NATIVE_SEQ_LEN, rng)
        context = _surface(normalization, "ctx_cont", 1, rng)
        for alias in model_native_context_temporal_alias_policy(_SIGNAL_NAMES)["aliases"]:
            context[0, alias["ctx_cont_index"]] = sequence[-1, alias["signal_index"]]
        mtf = {
            f"seq_{timeframe.lower()}": torch.from_numpy(
                _surface(normalization, f"mtf_{timeframe.lower()}", _TF_LENGTHS[timeframe], rng)
            ).unsqueeze(0)
            for timeframe in ENTRY_MTF_CONTEXT_TIMEFRAMES
        }
        sequence_tensor = torch.from_numpy(sequence).unsqueeze(0)
        with torch.no_grad():
            outputs = native_model(
                sequence_tensor, sequence_tensor[:, -1],
                ctx_cat=torch.zeros((1, len(MODEL_NATIVE_CTX_CAT_FIELDS)), dtype=torch.long),
                ctx_cont=torch.from_numpy(context), **mtf,
            )
        tokens.append(outputs["entry_decision_representation"].detach().clone())
    return tokens


def _native_inputs(prepared):
    episode = prepared.model_inputs
    return {
        "entry_decision_representation": prepared.online_entry_token,
        **{key: torch.from_numpy(episode[key]).unsqueeze(0) for key in (
            "exit_local_history_x", "exit_state_ctx_cat", "exit_state_ctx_cont", "exit_path_x"
        )},
        "exit_mtf_histories": {
            timeframe: torch.from_numpy(episode[f"exit_mtf_history_{timeframe}"]).unsqueeze(0)
            for timeframe in _TF_NAMES
        },
        "exit_mtf_gathers": {
            timeframe: torch.from_numpy(episode[f"exit_mtf_gather_{timeframe}"]).unsqueeze(0)
            for timeframe in _TF_NAMES
        },
        "exit_mtf_history_lengths": {
            timeframe: torch.tensor([len(episode[f"exit_mtf_history_{timeframe}"])], dtype=torch.long)
            for timeframe in _TF_NAMES
        },
    }


def test_actual_native_baseline_matches_existing_wrapper_and_full_gru_scans(
    episodes, native_model, online_tokens
):
    adapter = _adapter(episodes, native_model)
    history_lengths = []
    family = MODEL_NATIVE_TRAINING_SPECIALISTS[0]
    handle = native_model.exit_episode_mtf_family_gru[family].register_forward_pre_hook(
        lambda module, arguments: history_lengths.append(arguments[0].shape[1])
    )
    try:
        actual = adapter.predict_spec(episode=episodes[0], online_entry_token=online_tokens[0])
    finally:
        handle.remove()
    assert history_lengths == [len(episodes[0][f"exit_mtf_history_{timeframe}"]) for timeframe in _TF_NAMES]
    with torch.no_grad():
        expected, valid, state_valid, terminal, lengths = _forward_unified_exit_episode_pack(
            model=native_model, entry_decision_representation=online_tokens[0],
            episode=episodes[0], device=torch.device("cpu"),
        )
    np.testing.assert_array_equal(actual, expected[0].numpy())
    assert actual.shape == (2, _STATE_COUNT, 2)
    assert valid[0, :, -1].tolist() == [[False, True], [False, True]]
    assert state_valid.all() and terminal[:, :, -1].all()
    assert lengths.tolist() == [[_STATE_COUNT, _STATE_COUNT]]
    assert not native_model.training
    assert all(parameter.grad is None for parameter in native_model.parameters())
    native_model.require_input_normalization_state()


@pytest.mark.parametrize("kind", ("alias", "family_tf", "joint", "token", "path", "side"))
def test_actual_native_perturbation_and_side_axis_parity(
    episodes, layout, native_model, online_tokens, kind
):
    adapter = _adapter(episodes, native_model)
    arguments = _kwargs(episodes, layout, kind, online_tokens[0], online_tokens[1])
    prepared = adapter.prepare(**arguments)
    with torch.no_grad():
        reference = native_model.forward_exit_episode(**_native_inputs(prepared))["exit_action_q_bps"][0].numpy()
    if kind == "side":
        baseline = adapter.predict_spec(episode=episodes[0], online_entry_token=online_tokens[0])
        assert not np.array_equal(baseline[0], baseline[1])
        np.testing.assert_array_equal(reference, baseline)
        reference = reference[::-1]
    actual = adapter.predict_spec(**arguments)
    np.testing.assert_array_equal(actual, reference)
    assert actual.flags.c_contiguous


def test_actual_perturbed_native_scan_matches_unbroken_incremental_carry_all_states(
    episodes, layout, native_model, online_tokens
):
    adapter = _adapter(episodes, native_model)
    arguments = _kwargs(episodes, layout, "family_tf", online_tokens[0])
    prepared = adapter.prepare(**arguments)
    inputs = _native_inputs(prepared)
    offline = adapter.predict_spec(**arguments)
    carry = None
    prior_gather = {timeframe: -1 for timeframe in _TF_NAMES}
    pieces = []
    with torch.no_grad():
        for state in range(_STATE_COUNT):
            new_rows = {}
            for timeframe in _TF_NAMES:
                current = int(inputs["exit_mtf_gathers"][timeframe][0, state])
                new_rows[timeframe] = inputs["exit_mtf_histories"][timeframe][
                    :, prior_gather[timeframe] + 1 : current + 1
                ]
                prior_gather[timeframe] = current
            local_start = 0 if state == 0 else _WARM_ROWS + state
            outputs, carry = native_model.forward_exit_incremental_step(
                entry_decision_representation=prepared.online_entry_token,
                exit_local_rows_x=inputs["exit_local_history_x"][:, local_start : _WARM_ROWS + state + 1],
                exit_state_ctx_cat=inputs["exit_state_ctx_cat"][:, state],
                exit_state_ctx_cont=inputs["exit_state_ctx_cont"][:, state],
                exit_path_row_x=inputs["exit_path_x"][:, :, state],
                exit_mtf_new_rows=new_rows, carry=carry,
            )
            assert carry.step_count == state + 1
            pieces.append(outputs["exit_action_q_bps"])
    np.testing.assert_allclose(torch.cat(pieces, dim=2)[0].numpy(), offline, rtol=1e-5, atol=1e-5)


def test_baseline_teacher_targets_and_masks_do_not_follow_interventions(
    episodes, layout, native_model, online_tokens
):
    adapter = _adapter(episodes, native_model)
    teacher_token = online_tokens[1]
    supervision = adapter.baseline_supervision(
        episode=episodes[0], target_model=native_model, target_entry_token=teacher_token
    )
    expected, valid, terminal = _fitted_q_targets_for_episode(
        target_model=native_model, target_entry_decision_representation=teacher_token,
        episode=episodes[0], device=torch.device("cpu"),
    )
    np.testing.assert_array_equal(supervision.q_targets_bps, expected[0].numpy())
    np.testing.assert_array_equal(supervision.action_valid_mask, valid[0].numpy())
    np.testing.assert_array_equal(supervision.terminal_mask, terminal[0].numpy())
    maximum = expected.masked_fill(~valid, -torch.inf).amax(dim=-1, keepdim=True)
    np.testing.assert_array_equal(supervision.action_equivalence_mask, ((expected == maximum) & valid)[0].numpy())
    original_targets = supervision.q_targets_bps.copy()
    original_first_values = supervision.entry_first_side_values_bps.copy()
    original_side_valid = supervision.entry_side_valid_mask.copy()
    for kind in ("token", "path", "side"):
        adapter.predict_spec(**_kwargs(episodes, layout, kind, online_tokens[0], online_tokens[1]))
        np.testing.assert_array_equal(supervision.q_targets_bps, original_targets)
        np.testing.assert_array_equal(supervision.action_valid_mask, episodes[0]["exit_action_valid_mask"])
        np.testing.assert_array_equal(supervision.entry_first_side_values_bps, original_first_values)
        np.testing.assert_array_equal(supervision.entry_side_valid_mask, original_side_valid)
    with pytest.raises(TypeError):
        adapter.baseline_supervision(
            episode=episodes[0], target_model=native_model, target_entry_token=teacher_token,
            spec=_spec(layout, "path"),
        )


def test_native_supervision_entry_bridge_matches_batch_owner_in_one_teacher_forward(
    episodes, native_model, online_tokens, monkeypatch
):
    calls = []
    forward = native_model.forward_exit_episode
    def observed_forward(**inputs):
        calls.append(inputs["entry_decision_representation"].detach().clone())
        return forward(**inputs)
    with monkeypatch.context() as patch:
        patch.setattr(native_model, "forward_exit_episode", observed_forward)
        supervision = _adapter(episodes, native_model).baseline_supervision(
            episode=episodes[0], target_model=native_model, target_entry_token=online_tokens[1]
        )
    assert len(calls) == 1
    assert torch.equal(calls[0], online_tokens[1])
    targets, valid, terminal, first_values, first_valid = _fitted_q_targets_for_episode_batch(
        target_model=native_model, target_entry_decision_representations=online_tokens[1],
        episodes=[episodes[0]], device=torch.device("cpu"),
    )
    np.testing.assert_array_equal(supervision.q_targets_bps, targets[0].numpy())
    np.testing.assert_array_equal(supervision.action_valid_mask, valid[0].numpy())
    np.testing.assert_array_equal(supervision.terminal_mask, terminal[0].numpy())
    np.testing.assert_array_equal(supervision.entry_first_side_values_bps, first_values[0].numpy())
    np.testing.assert_array_equal(supervision.entry_side_valid_mask, first_valid[0].numpy())
    assert supervision.entry_first_side_values_bps.shape == (episode_owner.UNIFIED_EXIT_EPISODE_SIDE_COUNT,)
    assert supervision.entry_side_valid_mask.dtype == np.bool_


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _selected_files_fixture(tmp_path, *, online_state=None, target_state=None, normalization=None):
    """Synthetic file bindings, not a production recipe/bundle or trained pair."""

    trainer = native_usefulness.trainer
    root = tmp_path.resolve()
    output = root / "SELECTED_BUNDLE"
    source = root / "fixture_source.py"
    source.write_text('"""Mechanical source-binding fixture."""\n')
    source_bindings = {"fixture_source": artifact_binding(source)}
    artifacts = {
        "train_parquet": {"path": "/fixture/TRAIN.parquet", "sha256": "a" * 64},
        "val_parquet": {"path": "/fixture/VAL.parquet", "sha256": "b" * 64},
        "m5_prebuilt_path": {"path": "/fixture/M5.parquet", "sha256": "c" * 64},
        "unified_exit_lifecycle_manifest": {"path": "/fixture/lifecycle.json", "sha256": "d" * 64},
    }
    recipe_artifacts = {
        ("unified_exit_lifecycle_manifest_json" if name == "unified_exit_lifecycle_manifest" else name): value
        for name, value in artifacts.items()
    }
    recipe_path = root / "recipe.json"
    recipe = {
        "schema_version": RECIPE_AUDIT_SCHEMA, "decision": "PASS", "failures": [],
        "profile": "candidate", "run_id": "SELECTED_FIXTURE", "dataset_run_id": "DATA_FIXTURE",
        "out_bundle_dir": str(output), "dataset_dir": "/fixture",
        "source_commit": "1" * 40, "source_bindings": source_bindings,
        "source_bindings_sha256": canonical_json_sha256(source_bindings),
        "artifact_bindings": recipe_artifacts,
        "artifact_bindings_sha256": canonical_json_sha256(recipe_artifacts),
    }
    recipe_path.write_text(json.dumps(recipe))
    provenance = {
        "schema_version": TRAINING_RECIPE_SOURCE_PROVENANCE_SCHEMA,
        "recipe_audit_path": str(recipe_path), "recipe_audit_sha256": _file_sha(recipe_path),
        **{key: recipe[key] for key in ("source_commit", "source_bindings", "source_bindings_sha256")},
    }
    normalizer = {"contract_sha256": "e" * 64} if normalization is None else normalization
    contract = {
        "schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
        "out_bundle_dir": str(output), "profile": "candidate", "execution_tier": "canonical",
        "authority": {"candidate_training": True, "bundle": False},
        "run_id": recipe["run_id"], "dataset_run_id": recipe["dataset_run_id"],
        "source_commit": recipe["source_commit"], "recipe_source_provenance": provenance,
        "artifacts": artifacts, "input_normalization_sha256": normalizer["contract_sha256"],
        "training": {"checkpoint_policy": trainer.checkpoint_policy_metadata()},
    }
    session = trainer._CandidateTrainingSession(out_bundle_dir=output, contract=contract)
    online = {"weight": torch.tensor([1.0])} if online_state is None else online_state
    target = {"weight": torch.tensor([2.0])} if target_state is None else target_state
    online_sha = native_usefulness.canonical_model_state_sha256(online)
    target_sha = native_usefulness.canonical_model_state_sha256(target)
    exit_iteration = {
        "schema_version": UNIFIED_EXIT_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
        "iteration_index": 2, "target_model_state_sha256": target_sha,
        "train_split_sha256": artifacts["train_parquet"]["sha256"],
        "train_fold_sha256": "f" * 64, "source_lineage_sha256": artifacts["unified_exit_lifecycle_manifest"]["sha256"],
        "normalization_sha256": normalizer["contract_sha256"],
        "fitted_q_contract": unified_exit_fitted_q_contract(), "target_updated_from_val_or_test": False,
    }
    entry_iteration = {
        "schema_version": ENTRY_FITTED_Q_ITERATION_STATE_SCHEMA_VERSION,
        "iteration_index": exit_iteration["iteration_index"],
        "entry_target_model_state_sha256": target_sha, "exit_target_model_state_sha256": target_sha,
        "exit_fitted_q_iteration_state_sha256": canonical_json_sha256(exit_iteration),
        **{key: exit_iteration[key] for key in (
            "train_split_sha256", "train_fold_sha256", "source_lineage_sha256", "normalization_sha256"
        )},
        "entry_fitted_q_contract": entry_fitted_q_contract(),
        "exit_fitted_q_contract": unified_exit_fitted_q_contract(), "target_updated_from_val_or_test": False,
    }
    full_trajectory = {
        "online_model_state_sha256": online_sha, "target_model_state_sha256": target_sha,
        "state_prediction_stream_sha256": "5" * 64,
    }
    checkpoint = session.save_top_k_checkpoint(
        epoch=3, metric=1.0, model_state=online, target_model_state=target
    )
    progress = trainer._new_candidate_training_progress()
    progress["checkpoint_selection"].update(
        best_checkpoint=checkpoint, top_k_checkpoints=[checkpoint],
        best_epoch=3, last_epoch=8, epochs_since_improve=5, early_stopped=True,
        best_state=online, best_fitted_q_target_state=target, best_policy_pnl=1.0,
        best_unified_exit_fitted_q_state=exit_iteration,
        best_entry_fitted_q_state=entry_iteration,
        best_unified_exit_full_trajectory_validation=full_trajectory,
    )
    state = {
        "schema_version": trainer._CANDIDATE_TRAINING_SESSION_SCHEMA_VERSION,
        "session_contract_sha256": session.contract_sha256, "checkpoint_index": 1,
        "phase": "validation", "epoch_index": 7, "next_batch_offset": 1,
        "global_optimizer_steps": 8, "epoch_order": torch.arange(1, dtype=torch.int64),
        "model_state": {"weight": torch.tensor([99.0])},
        "target_model_state": {"weight": torch.tensor([88.0])},
        "optimizer_state": {}, "weight_ema_state": None, "lr_scheduler_state": None,
        "rng_state": {}, "training_progress": progress, "complete": True,
    }
    session.save_checkpoint(state)
    output.mkdir()
    metadata = {
        "run_lineage": {"training_profile": "candidate", "training_run_id": recipe["run_id"], "dataset_run_id": recipe["dataset_run_id"]},
        "recipe_source_provenance": provenance, "input_normalization": normalizer,
        "best_epoch": 3, "last_epoch": 8, "early_stopped": True,
        "best_entry_policy_realized_gross_spread_inclusive_pnl_bps": 1.0,
        "selected_online_model_state_sha256": online_sha,
        "selected_entry_fitted_q_iteration_state": entry_iteration,
        "unified_exit_training_evidence": {
            "selected_fitted_q_iteration_state": exit_iteration,
            "full_trajectory_validation": full_trajectory,
        },
        **{f"{split}_data": artifacts[f"{split}_parquet"]["path"] for split in ("train", "val")},
        **{f"{split}_data_sha256": artifacts[f"{split}_parquet"]["sha256"] for split in ("train", "val")},
    }
    metadata_path = output / "bundle_metadata.json"
    metadata_path.write_text(json.dumps(metadata))
    arguments = {
        "bundle_dir": output, "bundle_metadata_sha256": _file_sha(metadata_path),
        "session_out_bundle_dir": output, "session_contract_sha256": session.contract_sha256,
        "active_pointer_sha256": _file_sha(session.directory / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME),
        "selected_checkpoint_sha256": checkpoint["sha256"],
        "recipe_audit_path": recipe_path, "recipe_audit_sha256": _file_sha(recipe_path),
        "repo": Path(native_usefulness.__file__).resolve().parents[2],
    }
    return SimpleNamespace(
        arguments=arguments, metadata=metadata, contract=contract, session=session,
        state=state, provenance=provenance, recipe=recipe,
    )


def _selected_state_arguments(fixture):
    return {
        key: fixture.arguments[key] for key in (
            "session_out_bundle_dir", "session_contract_sha256", "active_pointer_sha256", "selected_checkpoint_sha256"
        )
    } | {"metadata": fixture.metadata}


def _expected_selection_artifacts(fixture):
    trainer = native_usefulness.trainer
    active_path = fixture.session.directory / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME
    active = json.loads(active_path.read_text())
    selected = fixture.state["training_progress"]["checkpoint_selection"]["best_checkpoint"]
    paths = {
        "session_contract": fixture.session.directory / trainer._CANDIDATE_TRAINING_CONTRACT_FILENAME,
        "active_pointer": active_path,
        "active_state": fixture.session._slot_path(active["slot"]),
        "selected_checkpoint": fixture.session.directory / selected["path"],
        "bundle_metadata": fixture.arguments["bundle_dir"] / "bundle_metadata.json",
        "recipe_audit": fixture.arguments["recipe_audit_path"],
    }
    return {role: {"path": str(path), "sha256": _file_sha(path)} for role, path in paths.items()}


def test_file_selected_earlier_epoch_uses_retained_pair_not_live_target(tmp_path):
    fixture = _selected_files_fixture(tmp_path)
    before = {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    contract, selection, files, artifacts = native_usefulness._load_selected_candidate_state(**_selected_state_arguments(fixture))
    assert selection["best_epoch"] == 3 and selection["last_epoch"] == 8
    assert torch.equal(selection["best_fitted_q_target_state"]["weight"], torch.tensor([2.0]))
    assert not torch.equal(selection["best_fitted_q_target_state"]["weight"], fixture.state["target_model_state"]["weight"])
    assert contract == fixture.contract
    assert all(_file_sha(path) == digest for path, digest in files.items())
    assert artifacts == {
        role: binding for role, binding in _expected_selection_artifacts(fixture).items()
        if role in {"session_contract", "active_pointer", "active_state", "selected_checkpoint"}
    }
    assert files == {binding["path"]: binding["sha256"] for binding in artifacts.values()}
    assert {str(path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_file_selected_policy_admits_only_the_already_bound_best_checkpoint(tmp_path):
    fixture = _selected_files_fixture(tmp_path)
    trainer = native_usefulness.trainer
    selection = fixture.state["training_progress"]["checkpoint_selection"]
    assert selection["checkpoint_policy"]["save_top_k"] == 1
    assert selection["top_k_checkpoints"] == [selection["best_checkpoint"]]
    extra = fixture.session.save_top_k_checkpoint(
        epoch=2, metric=0.5, model_state={"weight": torch.tensor([3.0])},
        target_model_state={"weight": torch.tensor([4.0])},
    )
    selection["top_k_checkpoints"].append(extra)
    active_path = fixture.session.directory / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME
    active = json.loads(active_path.read_text())
    state_path = fixture.session._slot_path(active["slot"])
    torch.save(fixture.state, state_path)
    active["state_sha256"] = _file_sha(state_path)
    active_path.write_text(json.dumps(active))
    fixture.arguments["active_pointer_sha256"] = _file_sha(active_path)
    with pytest.raises(RuntimeError, match="CANDIDATE_TRAINING_SELECTION_STATE_INVALID"):
        native_usefulness._load_selected_candidate_state(**_selected_state_arguments(fixture))


@pytest.mark.parametrize("fault", ("active_hash", "checkpoint_hash", "contract_hash", "top_k_bytes", "epoch", "full_trajectory", "selected_target", "iteration", "incomplete"))
def test_file_selected_pair_refuses_mixed_or_tampered_evidence(tmp_path, fault):
    fixture = _selected_files_fixture(tmp_path)
    selection = fixture.state["training_progress"]["checkpoint_selection"]
    if fault in ("active_hash", "checkpoint_hash", "contract_hash"):
        key = {"active_hash": "active_pointer_sha256", "checkpoint_hash": "selected_checkpoint_sha256", "contract_hash": "session_contract_sha256"}[fault]
        fixture.arguments[key] = "0" * 64
    elif fault == "top_k_bytes":
        path = fixture.session.directory / selection["best_checkpoint"]["path"]
        path.write_bytes(path.read_bytes() + b"tamper")
    elif fault == "epoch":
        fixture.metadata["best_epoch"] = 8
    elif fault == "full_trajectory":
        fixture.metadata = deepcopy(fixture.metadata)
        fixture.metadata["unified_exit_training_evidence"]["full_trajectory_validation"]["target_model_state_sha256"] = "0" * 64
    else:
        if fault == "selected_target":
            selection["best_fitted_q_target_state"] = fixture.state["target_model_state"]
        elif fault == "iteration":
            selection["best_unified_exit_fitted_q_state"]["iteration_index"] += 1
        else:
            fixture.state["complete"] = False
        fixture.state["checkpoint_index"] += 1
        fixture.session.save_checkpoint(fixture.state)
        fixture.arguments["active_pointer_sha256"] = _file_sha(fixture.session.directory / native_usefulness.trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME)
    with pytest.raises(RuntimeError):
        native_usefulness._load_selected_candidate_state(**_selected_state_arguments(fixture))


def _stub_external_pair_boundaries(monkeypatch, fixture, native_model):
    """Only file-binding tests stub strict recipe/bundle admission, never parity."""

    calls = []
    def source_owner(**arguments):
        calls.append("source")
        assert arguments["recipe_audit_sha256"] == _file_sha(arguments["recipe_audit_path"])
        for binding in fixture.provenance["source_bindings"].values():
            if artifact_binding(Path(binding["path"])) != binding:
                raise RuntimeError("STRICT_SOURCE_OWNER_REJECTED_CHANGED_BYTES")
        return fixture.provenance
    def bundle_loader(**arguments):
        calls.append("bundle")
        assert arguments == {"bundle_dir": fixture.arguments["bundle_dir"], "device": "cpu"}
        return SimpleNamespace(
            transformer_model=deepcopy(native_model), metadata=fixture.metadata,
            bundle_sha256="6" * 64,
        )
    monkeypatch.setattr(native_usefulness, "require_training_recipe_source_provenance", source_owner)
    monkeypatch.setattr(native_usefulness.bundle_owner, "load_entry_v10_ctx_bundle", bundle_loader)
    return calls


@pytest.mark.parametrize("bad_normalizer", (False, True))
def test_file_pair_loader_strict_teacher_state_and_embedded_normalization(
    tmp_path, monkeypatch, native_model, normalization, bad_normalizer
):
    online_state = deepcopy(native_model.state_dict())
    target_state = deepcopy(online_state)
    key = "input_norm_signal_center" if bad_normalizer else next(iter(dict(native_model.named_parameters())))
    target_state[key] = target_state[key] + 0.125
    fixture = _selected_files_fixture(
        tmp_path, online_state=online_state, target_state=target_state, normalization=normalization
    )
    calls = _stub_external_pair_boundaries(monkeypatch, fixture, native_model)
    if bad_normalizer:
        with pytest.raises(RuntimeError, match="NORMALIZATION_STATE"):
            native_usefulness.load_selected_native_exit_pair(**fixture.arguments)
    else:
        result = native_usefulness.load_selected_native_exit_pair(**fixture.arguments)
        assert calls == ["source", "bundle", "source"]
        assert result.input_normalization == normalization
        assert result.bindings["bundle_metadata_path"] == str(fixture.arguments["bundle_dir"] / "bundle_metadata.json")
        assert result.bindings["bundle_metadata_sha256"] == fixture.arguments["bundle_metadata_sha256"]
        assert result.bindings["dataset_bytes_validated_by_this_loader"] is False
        assert not (fixture.arguments["bundle_dir"] / "input_normalization.json").exists()
        assert not result.model.training and not result.target_model.training
        assert all(not parameter.requires_grad for model in (result.model, result.target_model) for parameter in model.parameters())
        assert torch.equal(result.target_model.state_dict()[key], target_state[key])
        assert not torch.equal(result.target_model.state_dict()[key], result.model.state_dict()[key])


@pytest.mark.parametrize("shared_digest", (False, True))
def test_file_pair_selection_roles_are_exact_with_rotated_slot_and_equal_hashes(
    tmp_path, monkeypatch, native_model, normalization, shared_digest
):
    """Role plumbing, not trained-pair admission: external admission is mocked.

    The equal-byte case additionally stubs session state admission because a
    top-k payload is not a session-state payload. Actual file SHA checks remain
    active; this only proves equal digests cannot substitute artifact roles.
    """

    weights = deepcopy(native_model.state_dict())
    fixture = _selected_files_fixture(
        tmp_path, online_state=weights, target_state=deepcopy(weights), normalization=normalization
    )
    fixture.state["checkpoint_index"] += 1
    fixture.session.save_checkpoint(fixture.state)
    trainer = native_usefulness.trainer
    active_path = fixture.session.directory / trainer._CANDIDATE_TRAINING_ACTIVE_FILENAME
    active = json.loads(active_path.read_text())
    assert active["slot"] == 1
    if shared_digest:
        checkpoint = fixture.state["training_progress"]["checkpoint_selection"]["best_checkpoint"]
        selected_path = fixture.session.directory / checkpoint["path"]
        fixture.session._slot_path(active["slot"]).write_bytes(selected_path.read_bytes())
        active["state_sha256"] = checkpoint["sha256"]
        active_path.write_text(json.dumps(active))
        def admitted_state(session):
            assert session.directory == fixture.session.directory
            assert session._read_only is True
            return fixture.state
        monkeypatch.setattr(trainer._CandidateTrainingSession, "load_checkpoint", admitted_state)
    fixture.arguments["active_pointer_sha256"] = _file_sha(active_path)
    calls = _stub_external_pair_boundaries(monkeypatch, fixture, native_model)
    result = native_usefulness.load_selected_native_exit_pair(**fixture.arguments)
    artifacts = result.bindings["selection_artifacts"]
    assert artifacts == _expected_selection_artifacts(fixture)
    assert set(artifacts) == {
        "session_contract", "active_pointer", "active_state", "selected_checkpoint",
        "bundle_metadata", "recipe_audit",
    }
    assert all(set(binding) == {"path", "sha256"} and Path(binding["path"]).is_absolute() for binding in artifacts.values())
    assert result.bindings["files"] == {
        binding["path"]: binding["sha256"] for binding in artifacts.values()
    }
    assert len(result.bindings["files"]) == 6
    assert artifacts["active_state"]["path"] != artifacts["selected_checkpoint"]["path"]
    assert (artifacts["active_state"]["sha256"] == artifacts["selected_checkpoint"]["sha256"]) is shared_digest
    assert result.bindings["selected_epoch"] == 3 and result.bindings["last_epoch"] == 8
    assert result.metadata["early_stopped"] is True
    assert calls == ["source", "bundle", "source"]
    native_usefulness.require_selected_native_pair_unchanged(
        selected_pair=result, repo=fixture.arguments["repo"]
    )
    assert calls == ["source", "bundle", "source", "source"]


@pytest.mark.parametrize("drift_phase", ("bundle", "teacher", "provenance"))
def test_file_pair_source_drift_during_load_fails_closed(
    tmp_path, monkeypatch, native_model, normalization, drift_phase
):
    """Mocked admission boundaries verify late source checks, not real Git admission."""

    weights = deepcopy(native_model.state_dict())
    fixture = _selected_files_fixture(
        tmp_path, online_state=weights, target_state=deepcopy(weights), normalization=normalization
    )
    calls = _stub_external_pair_boundaries(monkeypatch, fixture, native_model)
    source_path = Path(fixture.provenance["source_bindings"]["fixture_source"]["path"])
    original_source = native_usefulness.require_training_recipe_source_provenance
    original_bundle = native_usefulness.bundle_owner.load_entry_v10_ctx_bundle
    original_load_state = type(native_model).load_state_dict
    source_arguments = []
    def checked_source(**arguments):
        source_arguments.append(dict(arguments))
        observed = original_source(**arguments)
        if drift_phase == "provenance" and len(source_arguments) == 2:
            return {**observed, "source_commit": "f" * 40}
        return observed
    def loading_bundle(**arguments):
        bundle = original_bundle(**arguments)
        if drift_phase == "bundle":
            source_path.write_bytes(source_path.read_bytes() + b"\n")
        return bundle
    def loading_teacher(model, *arguments, **keywords):
        result = original_load_state(model, *arguments, **keywords)
        calls.append("teacher_load")
        if drift_phase == "teacher":
            source_path.write_bytes(source_path.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(native_usefulness, "require_training_recipe_source_provenance", checked_source)
    monkeypatch.setattr(native_usefulness.bundle_owner, "load_entry_v10_ctx_bundle", loading_bundle)
    monkeypatch.setattr(type(native_model), "load_state_dict", loading_teacher)
    expected = (
        "NATIVE_EXIT_SELECTED_RECIPE_SOURCE_CHANGED_DURING_LOAD"
        if drift_phase == "provenance" else "STRICT_SOURCE_OWNER_REJECTED_CHANGED_BYTES"
    )
    with pytest.raises(RuntimeError, match=expected):
        native_usefulness.load_selected_native_exit_pair(**fixture.arguments)
    assert calls == ["source", "bundle", "teacher_load", "source"]
    assert len(source_arguments) == 2 and source_arguments[0] == source_arguments[1]


@pytest.mark.parametrize("fault", ("recipe_bytes", "source_rejected", "artifact_declaration"))
def test_file_pair_recipe_failures_precede_checkpoint_or_bundle_load(tmp_path, monkeypatch, fault):
    fixture = _selected_files_fixture(tmp_path)
    def forbidden_load(*arguments, **keywords):
        pytest.fail("Invalid recipe must fail before checkpoint/model deserialization")
    monkeypatch.setattr(native_usefulness.torch, "load", forbidden_load)
    if fault == "recipe_bytes":
        fixture.arguments["recipe_audit_path"].write_text("{}")
    elif fault == "source_rejected":
        def rejected_source(**arguments):
            raise RuntimeError("STRICT_SOURCE_OWNER_REJECTED")
        monkeypatch.setattr(native_usefulness, "require_training_recipe_source_provenance", rejected_source)
    else:
        fixture.recipe["artifact_bindings"]["train_parquet"] = {"path": "/other/train", "sha256": "0" * 64}
        fixture.recipe["artifact_bindings_sha256"] = canonical_json_sha256(fixture.recipe["artifact_bindings"])
        fixture.arguments["recipe_audit_path"].write_text(json.dumps(fixture.recipe))
        fixture.arguments["recipe_audit_sha256"] = _file_sha(fixture.arguments["recipe_audit_path"])
        monkeypatch.setattr(native_usefulness, "require_training_recipe_source_provenance", lambda **arguments: fixture.provenance)
    with pytest.raises(RuntimeError):
        native_usefulness.load_selected_native_exit_pair(**fixture.arguments)


@pytest.mark.parametrize("field", ("bundle_metadata_sha256", "session_contract_sha256", "recipe_audit_sha256"))
def test_file_pair_missing_explicit_hash_cannot_select_current_bytes(tmp_path, field):
    fixture = _selected_files_fixture(tmp_path)
    fixture.arguments[field] = None
    with pytest.raises(RuntimeError, match="FILE_SHA_INVALID"):
        native_usefulness.load_selected_native_exit_pair(**fixture.arguments)


@pytest.fixture
def preservation_pair(tmp_path, monkeypatch, native_model, normalization):
    """Mechanical admission mocks around real model state/normalization owners.

    This fixture does not admit a trained candidate or execute a model forward.
    The separate teacher differs from online at the retained earlier epoch.
    """

    weights = deepcopy(native_model.state_dict())
    target_weights = deepcopy(weights)
    parameter_name = next(iter(dict(native_model.named_parameters())))
    target_weights[parameter_name] = target_weights[parameter_name] + 0.125
    fixture = _selected_files_fixture(
        tmp_path, online_state=weights, target_state=target_weights, normalization=normalization
    )
    calls = _stub_external_pair_boundaries(monkeypatch, fixture, native_model)
    pair = native_usefulness.load_selected_native_exit_pair(**fixture.arguments)
    source_owner = native_usefulness.require_training_recipe_source_provenance
    source_arguments = []
    def checked_source(**arguments):
        source_arguments.append(dict(arguments))
        return source_owner(**arguments)
    def forbidden_load(*arguments, **keywords):
        pytest.fail("Preservation must not deserialize, reselect, allocate data or forward")
    monkeypatch.setattr(native_usefulness, "require_training_recipe_source_provenance", checked_source)
    monkeypatch.setattr(native_usefulness.torch, "load", forbidden_load)
    monkeypatch.setattr(native_usefulness, "_load_selected_candidate_state", forbidden_load)
    monkeypatch.setattr(native_usefulness.bundle_owner, "load_entry_v10_ctx_bundle", forbidden_load)
    monkeypatch.setattr(native_usefulness.trainer, "_CandidateTrainingSession", forbidden_load)
    monkeypatch.setattr(native_usefulness.trainer, "EntryV10CtxDataset", forbidden_load)
    for model in (pair.model, pair.target_model):
        monkeypatch.setattr(model, "forward", forbidden_load)
        monkeypatch.setattr(model, "forward_exit_episode", forbidden_load)
    return SimpleNamespace(
        pair=pair, fixture=fixture, repo=fixture.arguments["repo"], calls=calls,
        source_arguments=source_arguments, forbidden_load=forbidden_load,
    )


def _pair_model_observation(pair):
    return tuple({
        "state": native_usefulness.canonical_model_state_sha256(model.state_dict()),
        "buffers": native_usefulness.canonical_model_state_sha256(dict(model.named_buffers())),
        "tensor_properties": tuple(
            (name, id(tensor), tensor.data_ptr(), tensor.dtype, str(tensor.device),
             tensor._version, tensor.requires_grad)
            for name, tensor in (*model.named_parameters(), *model.named_buffers())
        ),
        "training": tuple((name, module.training) for name, module in model.named_modules()),
    } for model in (pair.model, pair.target_model))


def _pair_file_observation(fixture):
    return {
        str(path): (_file_sha(path), path.stat().st_mode, path.stat().st_mtime_ns)
        for path in fixture.arguments["recipe_audit_path"].parent.rglob("*") if path.is_file()
    }


def test_pair_preservation_before_after_keeps_distinct_earlier_selected_models(
    preservation_pair, monkeypatch
):
    """Mocked recipe/bundle admission; genuine native tensor identity checks."""

    fixture = preservation_pair
    pair = fixture.pair
    normalization_calls = []
    for role, model in (("online", pair.model), ("teacher", pair.target_model)):
        owner = model.require_input_normalization_state
        def check_normalization(owner=owner, role=role):
            normalization_calls.append(role)
            return owner()
        monkeypatch.setattr(model, "require_input_normalization_state", check_normalization)
        for method in ("to", "cpu", "float", "double", "eval", "train", "requires_grad_", "_apply"):
            monkeypatch.setattr(model, method, fixture.forbidden_load)
    model_before = _pair_model_observation(pair)
    files_before = _pair_file_observation(fixture.fixture)
    assert pair.model is not pair.target_model
    assert pair.bindings["online_model_state_sha256"] != pair.bindings["target_model_state_sha256"]
    assert pair.bindings["selected_epoch"] == 3 < pair.bindings["last_epoch"] == 8
    for _boundary in ("before", "after"):
        assert native_usefulness.require_selected_native_pair_unchanged(
            selected_pair=pair, repo=fixture.repo
        ) is None
    assert normalization_calls == ["online", "teacher", "online", "teacher"]
    assert fixture.source_arguments == [dict(
        recipe_audit_path=fixture.fixture.arguments["recipe_audit_path"],
        recipe_audit_sha256=fixture.fixture.arguments["recipe_audit_sha256"],
        repo=fixture.repo, profile="candidate", run_id=fixture.fixture.contract["run_id"],
        dataset_run_id=fixture.fixture.contract["dataset_run_id"],
        dataset_dir=Path(fixture.fixture.recipe["dataset_dir"]),
        out_bundle_dir=fixture.fixture.arguments["session_out_bundle_dir"],
    )] * 2
    assert _pair_model_observation(pair) == model_before
    assert _pair_file_observation(fixture.fixture) == files_before
    assert any(buffer.dtype == torch.bool for buffer in pair.model.buffers())
    assert any(buffer.dtype == torch.uint8 for buffer in pair.target_model.buffers())


@pytest.mark.parametrize("role", ("model", "target_model"))
@pytest.mark.parametrize("fault", ("parameter", "normalization", "integer_dtype", "eval", "child_eval", "grad"))
def test_pair_preservation_rejects_model_drift_without_repair(preservation_pair, role, fault):
    fixture = preservation_pair
    pair = fixture.pair
    native_usefulness.require_selected_native_pair_unchanged(selected_pair=pair, repo=fixture.repo)
    model = getattr(pair, role)
    if fault == "parameter":
        with torch.no_grad():
            next(model.parameters()).add_(0.125)
        expected = "MODEL_STATE_CHANGED"
    elif fault == "normalization":
        with torch.no_grad():
            model.input_norm_signal_center.add_(0.125)
        expected = "NORMALIZATION_STATE"
    elif fault == "integer_dtype":
        model.input_norm_contract_sha256 = model.input_norm_contract_sha256.to(dtype=torch.int64)
        expected = "MODEL_STATE_CHANGED"
    elif fault == "eval":
        model.train()
        expected = "MODEL_MUST_REMAIN_EVAL"
    elif fault == "child_eval":
        next(model.children()).train()
        assert not model.training
        expected = "MODEL_MUST_REMAIN_EVAL"
    else:
        next(model.parameters()).requires_grad_(True)
        expected = "PARAMETERS_MUST_REMAIN_FROZEN"
    before = _pair_model_observation(pair)
    files_before = _pair_file_observation(fixture.fixture)
    with pytest.raises(RuntimeError, match=expected):
        native_usefulness.require_selected_native_pair_unchanged(selected_pair=pair, repo=fixture.repo)
    assert _pair_model_observation(pair) == before
    assert _pair_file_observation(fixture.fixture) == files_before


@pytest.mark.parametrize("role", ("model", "target_model"))
@pytest.mark.parametrize("surface", ("parameter", "buffer", "nonpersistent_buffer"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.float64))
def test_pair_preservation_requires_fp32_without_casting(preservation_pair, role, surface, dtype):
    pair = preservation_pair.pair
    model = getattr(pair, role)
    if surface == "parameter":
        parameter = next(model.parameters())
        parameter.data = parameter.detach().to(dtype=dtype)
    elif surface == "buffer":
        model.input_norm_signal_center = model.input_norm_signal_center.to(dtype=dtype)
    else:
        model.pos_enc = model.pos_enc.to(dtype=dtype)
    before = _pair_model_observation(pair)
    with pytest.raises(RuntimeError, match="FLOATING_STATE_MUST_BE_FP32"):
        native_usefulness.require_selected_native_pair_unchanged(
            selected_pair=pair, repo=preservation_pair.repo
        )
    assert _pair_model_observation(pair) == before


def test_pair_preservation_requires_separate_teacher(preservation_pair):
    fixture = preservation_pair
    pair = replace(fixture.pair, target_model=fixture.pair.model)
    with pytest.raises(RuntimeError, match="SEPARATE_PAIR_REQUIRED"):
        native_usefulness.require_selected_native_pair_unchanged(selected_pair=pair, repo=fixture.repo)


@pytest.mark.parametrize("fault", ("normalization", "teacher_evidence", "extra_metadata"))
def test_pair_preservation_detects_nested_metadata_drift(preservation_pair, fault):
    fixture = preservation_pair
    metadata = deepcopy(dict(fixture.pair.metadata))
    if fault == "normalization":
        metadata["input_normalization"]["surfaces"]["signal"]["center"][0] += 0.125
    elif fault == "teacher_evidence":
        metadata["unified_exit_training_evidence"]["full_trajectory_validation"]["target_model_state_sha256"] = "0" * 64
    else:
        metadata["undeclared"] = {"nested": True}
    pair = replace(fixture.pair, metadata=metadata)
    before = _pair_model_observation(pair)
    with pytest.raises(RuntimeError, match="METADATA_CHANGED"):
        native_usefulness.require_selected_native_pair_unchanged(selected_pair=pair, repo=fixture.repo)
    assert _pair_model_observation(pair) == before


@pytest.mark.parametrize("fault,expected", (
    ("missing_role", "ROLE_FILE_CLOSURE_MISMATCH"),
    ("extra_role", "ROLE_FILE_CLOSURE_MISMATCH"),
    ("role_keys", "ROLE_FILE_CLOSURE_MISMATCH"),
    ("role_hash", "ROLE_FILE_CLOSURE_MISMATCH"),
    ("matching_role_file_hash", "FILE_SHA_MISMATCH"),
    ("missing_file", "ROLE_FILE_CLOSURE_MISMATCH"),
    ("extra_file", "ROLE_FILE_CLOSURE_MISMATCH"),
    ("duplicate_role_path", "ROLE_FILE_CLOSURE_MISMATCH"),
    ("swapped_session_roles", "SESSION_ROLE_MISMATCH"),
    ("metadata_path", "METADATA_ROLE_MISMATCH"),
    ("online_digest", "METADATA_BINDINGS_MISMATCH"),
    ("teacher_digest", "METADATA_BINDINGS_MISMATCH"),
    ("normalization_digest", "METADATA_BINDINGS_MISMATCH"),
    ("epoch", "METADATA_BINDINGS_MISMATCH"),
    ("artifact_declarations", "METADATA_BINDINGS_MISMATCH"),
    ("recipe_provenance", "RECIPE_SOURCE_MISMATCH"),
))
def test_pair_preservation_role_file_and_metadata_binding_closure(preservation_pair, fault, expected):
    fixture = preservation_pair
    bindings = deepcopy(dict(fixture.pair.bindings))
    roles = bindings["selection_artifacts"]
    if fault == "missing_role":
        del roles["active_state"]
    elif fault == "extra_role":
        roles["invented"] = dict(roles["active_state"])
    elif fault == "role_keys":
        roles["active_state"]["inferred"] = True
    elif fault in ("role_hash", "matching_role_file_hash"):
        roles["selected_checkpoint"]["sha256"] = "0" * 64
        if fault == "matching_role_file_hash":
            bindings["files"][roles["selected_checkpoint"]["path"]] = "0" * 64
    elif fault == "missing_file":
        del bindings["files"][roles["active_state"]["path"]]
    elif fault == "extra_file":
        bindings["files"][str(fixture.fixture.arguments["recipe_audit_path"].parent / "extra")] = "0" * 64
    elif fault == "duplicate_role_path":
        roles["selected_checkpoint"] = dict(roles["active_state"])
        bindings["files"] = {binding["path"]: binding["sha256"] for binding in roles.values()}
    elif fault == "swapped_session_roles":
        roles["selected_checkpoint"], roles["active_state"] = roles["active_state"], roles["selected_checkpoint"]
    elif fault == "metadata_path":
        bindings["bundle_metadata_path"] = roles["recipe_audit"]["path"]
    elif fault in ("online_digest", "teacher_digest", "normalization_digest"):
        binding_key = {
            "online_digest": "online_model_state_sha256",
            "teacher_digest": "target_model_state_sha256",
            "normalization_digest": "input_normalization_sha256",
        }[fault]
        bindings[binding_key] = "0" * 64
    elif fault == "epoch":
        bindings["selected_epoch"] = bindings["last_epoch"]
    elif fault == "artifact_declarations":
        bindings["dataset_artifact_declarations"]["val_parquet"]["sha256"] = "0" * 64
    else:
        bindings["recipe_source_provenance"]["source_commit"] = "f" * 40
    pair = replace(fixture.pair, bindings=bindings)
    files_before = _pair_file_observation(fixture.fixture)
    with pytest.raises(RuntimeError, match=expected):
        native_usefulness.require_selected_native_pair_unchanged(selected_pair=pair, repo=fixture.repo)
    assert _pair_file_observation(fixture.fixture) == files_before


@pytest.mark.parametrize("role", (
    "session_contract", "active_pointer", "active_state", "selected_checkpoint",
    "bundle_metadata", "recipe_audit",
))
def test_pair_preservation_rehashes_every_explicit_role(preservation_pair, role):
    fixture = preservation_pair
    pair = fixture.pair
    native_usefulness.require_selected_native_pair_unchanged(selected_pair=pair, repo=fixture.repo)
    path = Path(pair.bindings["selection_artifacts"][role]["path"])
    path.write_bytes(path.read_bytes() + b"\n")
    files_before = _pair_file_observation(fixture.fixture)
    with pytest.raises(RuntimeError, match="FILE_SHA_MISMATCH"):
        native_usefulness.require_selected_native_pair_unchanged(selected_pair=pair, repo=fixture.repo)
    assert _pair_file_observation(fixture.fixture) == files_before


@pytest.mark.parametrize("fault", ("source_bytes", "owner_rejects", "owner_changed_provenance"))
def test_pair_preservation_rechecks_exact_source_after_audit(preservation_pair, monkeypatch, fault):
    """Mechanical source-owner boundary stubs, not real Git/candidate admission."""

    fixture = preservation_pair
    pair = fixture.pair
    native_usefulness.require_selected_native_pair_unchanged(selected_pair=pair, repo=fixture.repo)
    if fault == "source_bytes":
        source = Path(fixture.fixture.provenance["source_bindings"]["fixture_source"]["path"])
        source.write_bytes(source.read_bytes() + b"\n")
        expected = "STRICT_SOURCE_OWNER_REJECTED_CHANGED_BYTES"
    else:
        original_owner = native_usefulness.require_training_recipe_source_provenance
        def changed_owner(**arguments):
            provenance = original_owner(**arguments)
            if fault == "owner_rejects":
                raise RuntimeError("STRICT_SOURCE_OWNER_REQUIRES_CLEAN_EXACT_SOURCE")
            return {**provenance, "source_commit": "f" * 40}
        monkeypatch.setattr(native_usefulness, "require_training_recipe_source_provenance", changed_owner)
        expected = "STRICT_SOURCE_OWNER_REQUIRES_CLEAN_EXACT_SOURCE" if fault == "owner_rejects" else "RECIPE_SOURCE_MISMATCH"
    before = _pair_model_observation(pair)
    files_before = _pair_file_observation(fixture.fixture)
    with pytest.raises(RuntimeError, match=expected):
        native_usefulness.require_selected_native_pair_unchanged(selected_pair=pair, repo=fixture.repo)
    assert len(fixture.source_arguments) == 2
    assert fixture.source_arguments[0] == fixture.source_arguments[1]
    assert _pair_model_observation(pair) == before
    assert _pair_file_observation(fixture.fixture) == files_before


@pytest.mark.parametrize("module_name", ("trainer", "bundle_owner"))
def test_pair_preservation_rechecks_import_root(preservation_pair, monkeypatch, module_name):
    fixture = preservation_pair
    module = getattr(native_usefulness, module_name)
    monkeypatch.setattr(module, "__file__", native_usefulness.__file__)
    with pytest.raises(RuntimeError, match="IMPORTED_SOURCE_ROOT_MISMATCH"):
        native_usefulness.require_selected_native_pair_unchanged(selected_pair=fixture.pair, repo=fixture.repo)
