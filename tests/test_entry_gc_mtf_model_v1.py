"""Small CPU graph/domain tests for fixed218; synthetic consistency, never edge."""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest
import torch

from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts.entry_model_native_signal_v1 import (
    MODEL_NATIVE_CTX_CAT_DIM, MODEL_NATIVE_CTX_CONT_DIM,
)
from gx1.contracts.entry_model_native_training_objective_v1 import (
    ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
)
from gx1.features.entry_specialist_feature_groups_v1 import (
    compile_compact_entry_execution_profile, compile_gc_mtf_entry_execution_profile,
    group_features_by_specialist, model_native_context_temporal_alias_policy,
    require_compact_entry_execution_profile_reference, require_multi_tf_specialist_routing_v4,
)
from gx1.models.entry_v10 import entry_v10_ctx_train_v3 as trainer
from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import EXACT_SPECIALIST_NAMES
from tests.model_native_input_normalization_support import input_normalization_fixture
from tests.test_entry_v10_ctx_model_shapes import (
    SEQ_LEN, _make_dynamic_m5_model, _make_inputs, _valid_range_observation_input,
)
from tests.test_entry_gc_mtf_history_v1 import MINUTE, gc_history_fixture
from tests.test_entry_v10_input_normalization_fit import _mtf_frame, _mtf_values


@pytest.fixture(scope="module")
def bound_history():
    return gc_history_fixture()


def _reference(profile):
    return {key: profile[key] for key in ("name", "contract_sha256")}


@pytest.fixture(scope="module")
def candidate_case(bound_history):
    bound, surfaces, selected, _start, _end = bound_history
    layout = bound["layout"]
    base = compile_compact_entry_execution_profile()
    profile = compile_gc_mtf_entry_execution_profile(bound)
    physical = base["physical_field_names"]
    signal_names, mtf_names = list(physical["signal"]), list(physical["mtf"])
    aliases = model_native_context_temporal_alias_policy(signal_names)["aliases"]
    grouped = group_features_by_specialist(signal_names)
    normalization = input_normalization_fixture(
        signal_names=signal_names, mtf_names=mtf_names,
        per_tf_seq_lens=layout["window_rows"],
        entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE,
    )
    original_normalization = deepcopy(normalization)
    torch.manual_seed(7218)
    model = _make_dynamic_m5_model(
        seq_input_dim=len(signal_names), snap_input_dim=len(signal_names),
        specialist_input_indices={family: [i for i, name in enumerate(signal_names) if name in set(grouped[family])]
                                  for family in EXACT_SPECIALIST_NAMES},
        **{tf.lower() + "_seq_dim": len(mtf_names) for tf in layout["timeframes"]},
        **{tf.lower() + "_seq_len": layout["window_rows"][tf] for tf in layout["timeframes"]},
        multi_tf_specialist_input_indices={family: list(indices) for family, indices in require_multi_tf_specialist_routing_v4(mtf_names).items()},
        temporal_alias_signal_indices=[int(alias["signal_index"]) for alias in aliases],
        temporal_alias_ctx_cont_indices=[int(alias["ctx_cont_index"]) for alias in aliases],
        input_normalization=normalization, entry_execution_profile=_reference(profile),
        gc_mtf_history=bound, dropout=0.0,
    ).eval()
    assert normalization == original_normalization

    def valid_values(shape, surface):
        stats = normalization["surfaces"][surface]
        names = stats["field_names"]
        raw = torch.linspace(-0.2, 0.2, int(np.prod(shape))).reshape(shape)
        for index, binary in enumerate(stats["binary_mask"]):
            if binary:
                raw[..., index] = 1.0
        for name, domain in stats["categorical_domains"].items():
            raw[..., names.index(name)] = min(domain)
        if "ema_stack_aligned_v2" in names:
            raw[..., names.index("ema_stack_aligned_v2")] = 0.0
        return _valid_range_observation_input(raw, names)

    seq = valid_values((1, SEQ_LEN, len(signal_names)), "signal")
    cont = valid_values((1, MODEL_NATIVE_CTX_CONT_DIM), "ctx_cont")
    for alias in aliases:
        seq[..., int(alias["signal_index"])] = cont[:, int(alias["ctx_cont_index"])].unsqueeze(1)
    inputs = {
        "seq_x": seq, "snap_x": seq[:, -1, :].clone(),
        "ctx_cat": torch.zeros(1, MODEL_NATIVE_CTX_CAT_DIM, dtype=torch.long),
        "ctx_cont": cont,
    }
    validated = history.require_gc_mtf_history_surfaces(surfaces, layout=layout)
    for tf in layout["timeframes"]:
        length = layout["window_rows"][tf]
        inputs["seq_" + tf.lower()] = valid_values((1, length, len(mtf_names)), "mtf_" + tf.lower())
        ids = selected[tf][-length:]
        closes = surfaces[tf]["available_ns"][ids]
        gc_values = history.gather_gc_mtf_history_rows(validated[tf], tf=tf, native_row_indices=ids, available_ns=closes, layout=layout)
        inputs["gc_seq_" + tf.lower()] = torch.tensor(gc_values).unsqueeze(0)
    return profile, model, inputs, normalization


@pytest.mark.parametrize("bad", [False, {}, {"layout": {}}, "gc", 1])
def test_false_empty_partial_or_wrong_type_history_fails_before_module_allocation(monkeypatch, bad):
    def forbidden(*_args, **_kwargs):
        pytest.fail("malformed GC metadata reached torch module allocation")
    monkeypatch.setattr(torch.nn.Module, "__init__", forbidden)
    with pytest.raises(RuntimeError):
        _make_dynamic_m5_model(gc_mtf_history=bad)


def test_candidate_profile_requires_exact_base_layout_and_train_normalization_binding(bound_history):
    bound, _surfaces, _rows, _start, _end = bound_history
    base = compile_compact_entry_execution_profile()
    before = deepcopy(base)
    profile = compile_gc_mtf_entry_execution_profile(bound)
    assert profile["base_entry_execution_profile"] == _reference(base)
    assert profile["physical_field_names"] == base["physical_field_names"]
    for key in ("signal_indices", "ctx_cont_indices", "mtf_indices", "local_specialists", "macro_context_fields"):
        assert profile[key] == base[key]
    assert compile_compact_entry_execution_profile() == before
    assert require_compact_entry_execution_profile_reference(_reference(profile), gc_mtf_history=bound) == _reference(profile)
    for reference, metadata in ((None, bound), (_reference(base), bound), (_reference(profile), None), ({**_reference(profile), "contract_sha256": "0" * 64}, bound)):
        with pytest.raises(RuntimeError):
            require_compact_entry_execution_profile_reference(reference, gc_mtf_history=metadata)
    changed = deepcopy(bound)
    changed["normalization"]["contract_sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        compile_gc_mtf_entry_execution_profile(changed)


def test_source_owned_widths_and_specialist_indices_reach_separate_gc_projections(candidate_case):
    profile, model, _inputs, normalization = candidate_case
    layout = model.gc_mtf_history["layout"]
    assert profile["base_mtf_specialists"] == compile_compact_entry_execution_profile()["mtf_specialists"]
    assert tuple(model.mtf_family_encoder) == tuple(profile["mtf_specialists"])
    assert {"momentum_flow_encoder", "price_action_candle_encoder"}.issubset(model.mtf_family_encoder)
    for tf in layout["timeframes"]:
        names = layout["field_names"][tf]
        routed = []
        for family, indices in layout["family_indices"][tf].items():
            if not indices:
                continue
            key = tf.lower() + "__" + family
            projection = model.gc_mtf_family_proj[key]
            assert projection.in_features == len(indices) and projection.bias is None
            assert getattr(model, "gc_mtf_specialist_idx_" + key).tolist() == indices
            routed.extend(indices)
        assert sorted(routed) == list(range(len(names)))
        assert model._input_normalization_field_names["gc_mtf_" + tf.lower()] == names
    # The separate candidate surfaces must preserve every immutable base055
    # statistic. This synthetic contract check does not claim physical055 I/O.
    for surface, stats in normalization["surfaces"].items():
        assert model._input_normalization_field_names[surface] == stats["field_names"]
        for field in ("center", "scale"):
            np.testing.assert_array_equal(getattr(model, f"input_norm_{surface}_{field}").detach().numpy(), np.asarray(stats[field], dtype=np.float32))
    model.require_input_normalization_state()


def test_missing_or_extra_candidate_tensors_fail_before_market_encoder(candidate_case, monkeypatch):
    _profile, original, inputs, _normalization = candidate_case
    model = deepcopy(original)
    monkeypatch.setattr(model, "_encode_shared_feature_base", lambda **_kw: pytest.fail("missing GC input reached market encoder"))
    for tf in model.gc_mtf_history["layout"]["timeframes"]:
        bad = dict(inputs)
        del bad["gc_seq_" + tf.lower()]
        with pytest.raises(RuntimeError, match="GC_MTF_MODEL_INPUTS_REQUIRED"):
            model(**bad)
    with pytest.raises(TypeError):
        model(**inputs, gc_seq_m30=inputs["gc_seq_m5"])


def test_batch_helper_preserves_old_route_and_requires_the_complete_candidate_set():
    device = torch.device("cpu")
    base = {"seq_" + tf.lower(): torch.ones(1, 2, 3) for tf in history.TIMEFRAMES}
    actual = trainer._multi_tf_kwargs_from_batch(base, device)
    assert actual.keys() == base.keys() and all(actual[key] is base[key] for key in base)
    gc_inputs = {"gc_seq_" + tf.lower(): torch.ones(1, 2, i + 1) for i, tf in enumerate(history.TIMEFRAMES)}
    complete = {**base, **gc_inputs}
    assert trainer._multi_tf_kwargs_from_batch(complete, device).keys() == complete.keys()
    for tf in history.TIMEFRAMES:
        partial = dict(complete); del partial["gc_seq_" + tf.lower()]
        with pytest.raises(RuntimeError):
            trainer._multi_tf_kwargs_from_batch(partial, device)
    with pytest.raises(RuntimeError):
        trainer._multi_tf_kwargs_from_batch({**complete, "gc_seq_m30": torch.ones(1)}, device)
    with pytest.raises(RuntimeError):
        trainer._multi_tf_kwargs_from_batch({**base, "gc_seq_m30": torch.ones(1)}, device)
    with pytest.raises(RuntimeError):
        trainer._multi_tf_kwargs_from_batch({**complete, "gc_seq_h1": None}, device)


def test_dataset_exact_windows_collate_and_batch_helper_preserve_base_and_gc_row_bindings(bound_history):
    from torch.utils.data import default_collate
    from gx1.features.htf_features import MULTI_TF_SHIFT, get_model_native_multi_tf_route_windows
    bound, surfaces, selected, _start, end = bound_history
    layout = bound["layout"]
    frames = {tf: _mtf_frame(surface["available_ns"] - int(MULTI_TF_SHIFT[tf].value),
                             _mtf_values(len(surface["available_ns"])))
              for tf, surface in surfaces.items()}
    old = trainer.EntryV10CtxDataset.__new__(trainer.EntryV10CtxDataset)
    old._multi_tf_feats = frames
    old.per_tf_seq_lens = layout["window_rows"]
    old.gc_mtf_history = None
    candidate = trainer.EntryV10CtxDataset.__new__(trainer.EntryV10CtxDataset)
    candidate._multi_tf_feats = frames
    candidate.per_tf_seq_lens = layout["window_rows"]
    candidate.gc_mtf_history = bound
    candidate._gc_mtf_surfaces = history.require_gc_mtf_history_surfaces(surfaces, layout=layout)
    target = pd.Timestamp(end - MINUTE - 5 * MINUTE, tz="UTC")
    options = dict(route_timeframes=tuple(layout["timeframes"]), base_bar_seconds=300, key_prefix="seq_")
    old_sample = old._get_multi_tf_window(target, **options)
    sample = candidate._get_multi_tf_window(target, **options)
    exact = get_model_native_multi_tf_route_windows(
        frames, decision_bar_start=target, per_tf_seq_lens=layout["window_rows"],
        route_timeframes=tuple(layout["timeframes"]), base_bar_duration=pd.Timedelta(minutes=5),
        return_row_indices=True,
    )
    for tf, (window, ids) in exact.items():
        base_key, gc_key = "seq_" + tf.lower(), "gc_seq_" + tf.lower()
        np.testing.assert_array_equal(sample[base_key], old_sample[base_key])
        assert sample[base_key].tobytes() == window.tobytes()
        np.testing.assert_array_equal(sample[gc_key], surfaces[tf]["features"][ids])
        np.testing.assert_array_equal(surfaces[tf]["available_ns"][ids], frames[tf].attrs["ts_int64"][ids] + int(MULTI_TF_SHIFT[tf].value))
    collated = default_collate([sample, sample])
    kwargs = trainer._multi_tf_kwargs_from_batch(collated, torch.device("cpu"))
    assert set(kwargs) == {"seq_" + tf.lower() for tf in layout["timeframes"]} | {"gc_seq_" + tf.lower() for tf in layout["timeframes"]}
    for key, value in sample.items():
        assert kwargs[key].shape == (2, *value.shape)
        np.testing.assert_array_equal(kwargs[key][0].numpy(), value)
    row = int(selected["M5"][-1])
    for fault in ("closed_clock", "missing_native_id"):
        bad = deepcopy(surfaces)
        if fault == "closed_clock":
            bad["M5"]["available_ns"][row] += MINUTE
        else:
            for key in ("native_row_indices", "available_ns", "features"):
                bad["M5"][key] = np.delete(bad["M5"][key], row, axis=0)
        candidate._gc_mtf_surfaces = history.require_gc_mtf_history_surfaces(bad, layout=layout)
        with pytest.raises(RuntimeError):
            candidate._get_multi_tf_window(target, **options)


def test_no_gc_argument_preserves_initial_rng_state_parameters_and_exact_outputs():
    torch.manual_seed(7219)
    implicit = _make_dynamic_m5_model(dropout=0.0).eval()
    after = torch.get_rng_state().clone()
    torch.manual_seed(7219)
    explicit = _make_dynamic_m5_model(dropout=0.0, gc_mtf_history=None).eval()
    assert torch.equal(torch.get_rng_state(), after)
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    assert all(torch.equal(value, explicit.state_dict()[key]) for key, value in implicit.state_dict().items())
    assert not any(key.startswith(("gc_mtf_", "input_norm_gc_mtf_")) for key in implicit.state_dict())
    seq, snap, cat, cont, mtf = _make_inputs(1)
    kwargs = dict(seq_x=seq, snap_x=snap, ctx_cat=cat, ctx_cont=cont, **mtf)
    with torch.no_grad():
        left = implicit(**kwargs)
        right = explicit(**kwargs, **{"gc_seq_" + tf.lower(): None for tf in history.TIMEFRAMES})
    assert left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)
    with pytest.raises(RuntimeError):
        implicit(**kwargs, gc_seq_m5=torch.ones(1, 4, 3))


def test_new_gap_reuses_own_bbo_mask_after_asinh_normalization(candidate_case):
    _profile, model, inputs, _normalization = candidate_case
    for tf in history.TIMEFRAMES:
        names = model.gc_mtf_history["layout"]["field_names"][tf]
        vi = names.index(f"gc_{tf.lower()}_observed_book_trailing_support_gap_minutes")
        mi = names.index(f"gc_{tf.lower()}_observed_book_valid")
        raw = inputs["gc_seq_" + tf.lower()][:, :1].repeat(2, 1, 1)
        raw[..., vi] = 0.0
        raw[0, :, mi] = 0.0
        normalized = model._normalize_input_surface(raw, surface="gc_mtf_" + tf.lower())
        assert normalized[0, 0, vi] == 0.0
        assert normalized[1, 0, vi] != 0.0
        bad = raw.clone(); bad[0, 0, vi] = 1.0
        with pytest.raises(RuntimeError, match="MASKED_STORAGE"):
            model._normalize_input_surface(bad, surface="gc_mtf_" + tf.lower())


def test_every_tf_and_legitimate_gc_family_has_component_gradient(candidate_case):
    from gx1.contracts.entry_observed_market_v1 import is_exit_owned_parameter
    _profile, original, inputs, _normalization = candidate_case
    model = deepcopy(original).train()
    leaves = {key: value.detach().clone().requires_grad_(key.startswith("gc_seq_")) for key, value in inputs.items()}
    with torch.no_grad():
        model.specialist_out.weight.copy_(torch.eye(model.cfg.d_model))
    output = model(**leaves)
    loss = sum(output[key].square().mean() for key in (
        "dynamic_terminal_net_markout_bps", "dynamic_mfe_bps", "dynamic_pre_mfe_mae_upper_bps",
        "dynamic_closed_mfe_bps", "dynamic_closed_pre_mfe_mae_upper_bps", "forecast_pred",
    ))
    loss.backward()
    for key, tensor in leaves.items():
        if key.startswith("gc_seq_"):
            assert tensor.grad is not None and bool(torch.isfinite(tensor.grad).all()), key
            assert bool((tensor.grad != 0.0).any()), key
            assert bool((tensor.grad[:, :-1] != 0.0).any()), key  # Historical path, not just final row.
    for modules in (model.gc_mtf_family_proj, model.gc_mtf_feature_context_gate):
        for key, module in modules.items():
            gradients = [parameter.grad for parameter in module.parameters()]
            assert all(gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients), key
            assert any(bool((gradient != 0.0).any()) for gradient in gradients), key
    assert all(parameter.grad is None for name, parameter in model.named_parameters() if is_exit_owned_parameter(name))


def test_isolated_historical_gc_change_on_each_tf_reaches_forecasts_in_one_batch(candidate_case):
    _profile, model, inputs, _normalization = candidate_case
    count = 1 + len(history.TIMEFRAMES)
    batched = {key: value.repeat(count, *([1] * (value.ndim - 1))) for key, value in inputs.items()}
    for row, tf in enumerate(history.TIMEFRAMES, start=1):
        names = model.gc_mtf_history["layout"]["field_names"][tf]
        index = names.index(f"gc_{tf.lower()}_observed_book_trailing_support_gap_minutes")
        batched["gc_seq_" + tf.lower()][row, 0, index] += 1.0
    with torch.no_grad():
        output = model(**batched)
    for row, tf in enumerate(history.TIMEFRAMES, start=1):
        assert any(not torch.equal(output[key][0], output[key][row]) for key in (
            "dynamic_terminal_net_markout_bps", "dynamic_mfe_bps", "dynamic_pre_mfe_mae_upper_bps",
        )), tf
    # These are synthetic response checks, not feature usefulness/generalization.
    gates, feature_gates = trainer._dynamic_m5_gate_execution_kwargs(model)
    assert gates["gate_widths"]["family_tf_cooperation_gate"] == 5 * len(model._mtf_specialist_names)
    assert feature_gates["gate_shape"] == (5, len(compile_compact_entry_execution_profile()["mtf_indices"]))


@pytest.mark.parametrize("fault", ["state_hash", "owner_indices", "mask_buffer"])
def test_candidate_state_rejects_changed_bound_metadata_or_runtime_indices(candidate_case, fault):
    _profile, original, _inputs, _normalization = candidate_case
    model = deepcopy(original)
    if fault == "state_hash":
        model.gc_mtf_layout_sha256[0] ^= 1
    elif fault == "owner_indices":
        name, indices = next((key, value) for key, value in model._gc_mtf_specialist_index_tuples.items() if len(value) > 1)
        model._gc_mtf_specialist_index_tuples[name] = tuple(reversed(indices))
    else:
        model.input_norm_gc_mtf_m5_binary_mask[0] ^= True
    with pytest.raises(RuntimeError):
        model.require_input_normalization_state()
