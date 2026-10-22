"""Audit-only compact dynamic Entry input sensitivity; no load, fit or admission.

The caller binds ONLINE state, mode/RNG preservation, normalization, TRAIN witnesses,
case-plan SHA and hardware. This owner only handles source-owned input coordinates,
legal encoder-domain interventions and sensitivity arithmetic. A derivative is local
sensitivity, not a genuine market counterfactual, learned utility or trading edge.
"""
from __future__ import annotations

import math
from itertools import islice
from collections.abc import Mapping

SCHEMA_VERSION = "gx1_dynamic_compact_input_influence_v1"
BATCH_ROWS = 16
REVERSE_CALLS = 13
MAX_CASES = 3446
MAX_FORWARD_CALLS = 218
_COMPONENTS = (
    "dynamic_terminal_net_markout_bps",
    "dynamic_mfe_bps",
    "dynamic_pre_mfe_mae_upper_bps",
    "dynamic_closed_mfe_bps",
    "dynamic_closed_pre_mfe_mae_upper_bps",
)
_CANDIDATE = "dynamic_entry_action_quality_bps_candidate"
_CURVE = "dynamic_entry_quality_bps_candidate"
_TIE = "dynamic_entry_quality_best_horizon_mask"
_HORIZON = "dynamic_query_horizon_minutes"
_MTF = ("seq_m5", "seq_m15", "seq_h1", "seq_h4", "seq_d1")
_FLOAT_INPUTS = ("seq_x", "snap_x", "ctx_cont", *_MTF)


def _fail(reason):
    raise RuntimeError("DYNAMIC_COMPACT_INFLUENCE_" + reason)


def compile_dynamic_compact_influence_layout():
    """Adapt existing FULL-layout owners to the approved compact physical indices."""
    from gx1.features.entry_specialist_feature_groups_v1 import (
        compile_compact_entry_execution_profile,
        model_native_context_specialist_routing_contract,
    )
    from gx1.contracts.model_native_serve_gate_v1 import individual_input_influence_layout
    from gx1.features import gc_native_context_v1 as gc
    from gx1.features import micro_structure_v1 as micro
    from gx1.features.basic_v1 import session_native_binary_event_fields
    from gx1.features.htf_features import (
        MULTI_TF_STRUCTURAL_BINARY_FEATURES_V4,
        MULTI_TF_SOURCE_CONSTANT_BINARY_EVENTS_V4,
    )

    profile = compile_compact_entry_execution_profile()
    names = profile["physical_field_names"]
    full = individual_input_influence_layout(names["signal"])
    selected = {
        "seq_signal": set(profile["signal_indices"]),
        "snap_signal": set(profile["generic_snap_signal_indices"]),
        "ctx_cont": set(profile["ctx_cont_indices"]),
        **{lane: set(profile["mtf_indices"]) for lane in _MTF},
    }
    numeric = []
    for surface, spec in full["numeric"].items():
        for token, index in zip(spec["tokens"], spec["source_indices"]):
            if index in selected[surface]:
                physical = {"seq_signal": "seq_x", "snap_signal": "snap_x"}.get(surface, surface)
                field = names["signal" if physical in ("seq_x", "snap_x") else
                              "ctx_cont" if physical == "ctx_cont" else "mtf"][index]
                numeric.append({"token": surface + "." + token,
                                "surface": physical, "source_index": index, "field": field})
    aliases = [dict(item) for item in full["continuous_manifold"]
               if item["signal_index"] in selected["seq_signal"]
               and item["ctx_cont_index"] in selected["ctx_cont"]]
    categorical = []
    for item in full["categorical"]:
        surface = item["surface"]
        index = item.get("source_index")
        keep = (surface == "signal" and index in selected["seq_signal"] or
                surface == "ctx_cat" or
                surface == "ctx_cont" and index in selected["ctx_cont"] or
                surface in _MTF and index in selected[surface])
        if keep:
            categorical.append(dict(item))
    domains = {}
    for surface in ("seq_x", "snap_x", "ctx_cont", "ctx_cat", *_MTF):
        field_names = names["signal" if surface in ("seq_x", "snap_x") else
                            "ctx_cont" if surface == "ctx_cont" else
                            "ctx_cat" if surface == "ctx_cat" else "mtf"]
        binary = gc.binary_fields(field_names) | micro.range_observation_binary_fields(field_names)
        binary |= session_native_binary_event_fields(field_names)
        if surface in _MTF:
            binary |= set(MULTI_TF_STRUCTURAL_BINARY_FEATURES_V4).intersection(field_names)
        pairs = (*gc.masking_pairs(field_names), *micro.range_observation_masking_pairs(field_names))
        enum_domains = {}
        for item in categorical:
            if (item["surface"] == surface or
                    item["surface"] == "signal" and surface in ("seq_x", "snap_x")):
                enum_domains[item["source_index"]] = tuple(item["domain"])
        if surface in _MTF and "ema_stack_aligned_v2" in field_names:
            enum_domains[field_names.index("ema_stack_aligned_v2")] = (-1, 0, 1)
        constants = {}
        if surface in _MTF:
            for field, value in MULTI_TF_SOURCE_CONSTANT_BINARY_EVENTS_V4.get(surface[4:].upper(), {}).items():
                constants[field_names.index(field)] = value
        domains[surface] = {"field_names": tuple(field_names),
                            "binary_indices": tuple(i for i, name in enumerate(field_names) if name in binary),
                            "enum_domains": enum_domains, "masking_pairs": tuple(pairs),
                            "source_constants": constants}
    # Domain metadata is declaration-derived, never inferred from witness values.
    for group in (numeric, aliases):
        for item in group:
            surface = "ctx_cont" if "ctx_cont_index" in item else item["surface"]
            index = item["ctx_cont_index"] if "ctx_cont_index" in item else item["source_index"]
            spec = domains[surface]
            own = [mask for value, mask in spec["masking_pairs"] if value == index]
            dependent = tuple(value for value, mask in spec["masking_pairs"] if mask == index)
            item["own_observation_index"] = own[0] if own else None
            item["is_observation_mask"] = bool(dependent)
            item["dependent_value_indices"] = dependent
            item["domain_kind"] = ("observation_mask" if dependent else "source_enum" if index in spec["enum_domains"]
                                   else "source_binary" if index in spec["binary_indices"] else "finite_asinh_input")
            item["source_constant"] = spec["source_constants"].get(index)
            item["finite_intervention_required"] = item["domain_kind"] != "finite_asinh_input"
    context_routes = model_native_context_specialist_routing_contract()
    families = []
    for family in profile["local_specialists"]:
        families.append({"token": "local." + family, "surface": "local",
                         "signal_indices": tuple(profile["local_signal_indices_by_specialist"][family]),
                         "ctx_cont_indices": tuple(profile["ctx_cont_indices_by_specialist"][family]),
                         "ctx_cat_indices": tuple(context_routes["ctx_cat_indices"][family])})
    for lane in _MTF:
        for family in profile["mtf_specialists"]:
            families.append({"token": lane + "." + family, "surface": lane,
                             "source_indices": tuple(profile["mtf_indices_by_specialist"][family])})
    counts = {"numeric": len(numeric), "aliases": len(aliases),
              "categorical": len(categorical), "families": len(families)}
    if counts != {"numeric": 882, "aliases": 2525, "categorical": 7, "families": 32}:
        _fail("SOURCE_LAYOUT_DRIFT")
    tokens = [item["token"] for group in (numeric, aliases, categorical, families) for item in group]
    if len(tokens) != len(set(tokens)):
        _fail("TOKEN_DUPLICATE")
    return {"schema_version": SCHEMA_VERSION, "profile": profile,
            "physical_field_names": names, "numeric": numeric, "aliases": aliases,
            "categorical": categorical, "families": families, "domains": domains, "counts": counts,
            "physical_aliases": tuple(full["continuous_manifold"])}


def _flat_inputs(inputs):
    if not isinstance(inputs, Mapping) or set(inputs) != {"seq_x", "snap_x", "ctx_cont", "ctx_cat", "mtf"}:
        _fail("INPUT_KEYS")
    if not isinstance(inputs["mtf"], Mapping) or set(inputs["mtf"]) != set(_MTF):
        _fail("MTF_KEYS")
    return {name: inputs["mtf"][name] if name in _MTF else inputs[name]
            for name in ("seq_x", "snap_x", "ctx_cont", "ctx_cat", *_MTF)}


def _nest_inputs(flat):
    return {**{name: flat[name] for name in ("seq_x", "snap_x", "ctx_cont", "ctx_cat")},
            "mtf": {name: flat[name] for name in _MTF}}


def _same_bits(left, right):
    import torch
    return left.shape == right.shape and left.dtype == right.dtype and torch.equal(
        left.contiguous().view(torch.int32) if left.dtype == torch.float32 else left,
        right.contiguous().view(torch.int32) if right.dtype == torch.float32 else right)


def require_dynamic_compact_inputs(inputs, *, layout):
    """Validate full physical domains before compact routing, including stored zeros."""
    import torch
    flat = _flat_inputs(inputs)
    if (not isinstance(layout, Mapping) or layout.get("schema_version") != SCHEMA_VERSION
            or any(not isinstance(value, torch.Tensor) for value in flat.values())):
        _fail("LAYOUT_OR_INPUT_TENSOR")
    if flat["seq_x"].ndim != 3:
        _fail("INPUT_SHAPE_DTYPE_FINITE:seq_x")
    batch = flat["seq_x"].shape[0]
    if type(batch) is not int or not 1 <= batch <= BATCH_ROWS:
        _fail("BATCH_ROWS")
    device = flat["seq_x"].device
    for surface, values in flat.items():
        spec = layout["domains"][surface]
        rank = 3 if surface == "seq_x" or surface in _MTF else 2
        if (not isinstance(values, torch.Tensor) or values.ndim != rank or values.shape[0] != batch
                or values.shape[-1] != len(spec["field_names"]) or values.device != device
                or (surface != "ctx_cat" and values.dtype != torch.float32)
                or (surface == "ctx_cat" and values.dtype != torch.int64)
                or not bool(torch.isfinite(values).all())):
            _fail("INPUT_SHAPE_DTYPE_FINITE:" + surface)
        if rank == 3 and values.shape[1] < 1:
            _fail("EMPTY_HISTORY:" + surface)
        if surface == "seq_x" and values.shape[1] != 96:
            _fail("LOCAL_HISTORY_MUST_BE_96")
        if spec["binary_indices"]:
            binary = values[..., list(spec["binary_indices"])]
            if bool(((binary != 0) & (binary != 1)).any()):
                _fail("BINARY_DOMAIN:" + surface)
        for index, domain in spec["enum_domains"].items():
            column = values[..., index]
            allowed = torch.zeros_like(column, dtype=torch.bool)
            for value in domain:
                allowed |= column == value
            if not bool(allowed.all()):
                _fail("ENUM_DOMAIN:" + surface)
        if spec["masking_pairs"]:
            value_indices, mask_indices = zip(*spec["masking_pairs"])
            observed = values[..., list(mask_indices)]
            if bool((((observed != 0) & (observed != 1)) |
                     ((observed == 0) & (values[..., list(value_indices)] != 0))).any()):
                _fail("MASKED_STORAGE:" + surface)
        for index, value in spec["source_constants"].items():
            if bool((values[..., index] != value).any()):
                _fail("SOURCE_CONSTANT_DOMAIN:" + surface)
    if not _same_bits(flat["seq_x"][:, -1], flat["snap_x"]):
        _fail("SEQ_LAST_SNAP_NOT_EXACT")
    alias_s = [item["signal_index"] for item in layout["physical_aliases"]]
    alias_c = [item["ctx_cont_index"] for item in layout["physical_aliases"]]
    if not _same_bits(flat["snap_x"][:, alias_s], flat["ctx_cont"][:, alias_c]):
        _fail("CURRENT_ALIAS_NOT_EXACT")
    return flat


def _require_outputs(outputs, batch):
    import torch
    from gx1.contracts.entry_causal_m1_outcomes_v1 import DYNAMIC_M1_PREFIX_HORIZON_MINUTES
    horizons = len(DYNAMIC_M1_PREFIX_HORIZON_MINUTES)
    if not isinstance(outputs, Mapping):
        _fail("OUTPUT_MAPPING")
    for name in (*_COMPONENTS, _CANDIDATE, _CURVE):
        value = outputs.get(name)
        shape = (batch, 3) if name == _CANDIDATE else (batch, horizons, 3 if name in (_COMPONENTS[0], _CURVE) else 2)
        if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.float32
                or not bool(torch.isfinite(value).all())):
            _fail("OUTPUT_SHAPE_DTYPE_FINITE:" + name)
    if any(bool((outputs[name][..., 2] != 0).any()) for name in (_COMPONENTS[0], _CANDIDATE, _CURVE)):
        _fail("ANALYTIC_FLAT_NOT_ZERO")
    for name in (_COMPONENTS[2], _COMPONENTS[4]):
        if bool((outputs[name] < 0).any()):
            _fail("NEGATIVE_MAE:" + name)
    horizon = outputs.get(_HORIZON)
    if (not isinstance(horizon, torch.Tensor) or horizon.dtype != torch.int64 or horizon.ndim != 1
            or horizon.tolist() != list(DYNAMIC_M1_PREFIX_HORIZON_MINUTES)):
        _fail("HORIZON_IDENTITY")
    tie = outputs.get(_TIE)
    if not isinstance(tie, torch.Tensor) or tie.dtype != torch.bool or tie.shape != (batch, horizons, 2):
        _fail("TIE_SHAPE")
    curve = outputs[_CURVE]
    expected_curve = outputs[_COMPONENTS[0]][..., :2] - outputs[_COMPONENTS[2]]
    expected_max = expected_curve.max(dim=1).values
    if (not _same_bits(curve[..., :2], expected_curve) or
            not _same_bits(outputs[_CANDIDATE][:, :2], expected_max) or
            not torch.equal(tie, expected_curve == expected_max[:, None, :])):
        _fail("CANDIDATE_COMPONENT_IDENTITY")
    return outputs


def _observed(values, spec):
    import torch
    result = torch.ones_like(values, dtype=torch.bool)
    if spec["masking_pairs"]:
        value_indices, mask_indices = zip(*spec["masking_pairs"])
        result[..., list(value_indices)] = values[..., list(mask_indices)] == 1
    return result


def dynamic_compact_input_vjps(forward, inputs, *, layout):
    """One forward and thirteen input-only VJPs, reduced before the next reverse.

    Current seq+snapshot derivatives are summed signed before abs (all fields).
    Current context aliases additionally sum their ctx derivative. Physical partials
    remain diagnostics; neither they nor historical sensitivity prove current legal
    influence. Binary/enum slopes still require a valid finite intervention.
    """
    import torch
    require_dynamic_compact_inputs(inputs, layout=layout)
    flat = {name: value.detach().clone().requires_grad_(name != "ctx_cat")
            for name, value in _flat_inputs(inputs).items()}
    ordered = tuple(flat[name] for name in _FLOAT_INPUTS)
    observed = {name: _observed(value, layout["domains"][name]) for name, value in flat.items()
                if name != "ctx_cat"}
    routes = {item["token"]: {} for group in (layout["numeric"], layout["aliases"]) for item in group}
    alias_s = [item["signal_index"] for item in layout["aliases"]]
    alias_c = [item["ctx_cont_index"] for item in layout["aliases"]]
    gradients = {}
    with torch.enable_grad():
        outputs = _require_outputs(forward(_nest_inputs(flat)), flat["seq_x"].shape[0])
        q = outputs[_CANDIDATE]
        projections = {
            "candidate.long_short": (q[:, 0] - q[:, 1]).sum(),
            "candidate.long_flat": (q[:, 0] - q[:, 2]).sum(),
            "candidate.short_flat": (q[:, 1] - q[:, 2]).sum(),
        }
        for name in _COMPONENTS:
            for side, label in enumerate(("long", "short")):
                projections[name + "." + label] = outputs[name][..., side].sum()
        if len(projections) != REVERSE_CALLS:
            _fail("PROJECTION_COUNT")
        for i, (projection, scalar) in enumerate(projections.items()):
            if not scalar.requires_grad:
                scalar = scalar + ordered[0].reshape(-1)[0] * 0.0
            raw = torch.autograd.grad(scalar, ordered, retain_graph=i < REVERSE_CALLS - 1,
                                      create_graph=False, allow_unused=True)
            grads = {surface: torch.zeros_like(flat[surface]) if grad is None else grad.detach()
                     for surface, grad in zip(_FLOAT_INPUTS, raw)}
            summaries = {}
            partials = {}
            for surface, grad in grads.items():
                if not bool(torch.isfinite(grad).all()):
                    _fail("NONFINITE_INPUT_GRADIENT:" + projection + ":" + surface)
                obs = observed[surface]
                masked = torch.where(obs, grad, torch.zeros_like(grad))
                current = grad[:, -1] if grad.ndim == 3 else grad
                partials[surface] = {"current": current.clone()}
                summaries[surface] = {"current_abs": masked[:, -1].abs() if grad.ndim == 3 else masked.abs(),
                                      "observed_current": obs[:, -1] if grad.ndim == 3 else obs}
                if grad.ndim == 3:
                    history = masked[:, :-1].abs().amax(dim=1) if grad.shape[1] > 1 else torch.zeros_like(masked[:, 0])
                    summaries[surface]["history_abs"] = history
                    summaries[surface]["observed_history"] = obs[:, :-1].any(dim=1) if grad.shape[1] > 1 else torch.zeros_like(obs[:, 0])
                    partials[surface]["history_abs"] = history
            local_joint = grads["seq_x"][:, -1] + grads["snap_x"]
            local_joint_abs = torch.where(observed["snap_x"], local_joint, torch.zeros_like(local_joint)).abs()
            for item in layout["numeric"]:
                surface, index = item["surface"], item["source_index"]
                entry = {key: value[:, index] for key, value in summaries[surface].items()}
                if surface in ("seq_x", "snap_x"):
                    entry["current_partial_abs"] = entry["current_abs"]
                    entry["current_abs"] = local_joint_abs[:, index]
                    entry["current_signed"] = local_joint[:, index]
                routes[item["token"]][projection] = entry
            joint = local_joint[:, alias_s] + grads["ctx_cont"][:, alias_c]
            alias_observed = observed["ctx_cont"][:, alias_c]
            joint_abs = torch.where(alias_observed, joint, torch.zeros_like(joint)).abs()
            history = summaries["seq_x"]
            for j, item in enumerate(layout["aliases"]):
                routes[item["token"]][projection] = {
                    "current_abs": joint_abs[:, j], "current_signed": joint[:, j],
                    "observed_current": alias_observed[:, j],
                    "history_abs": history["history_abs"][:, item["signal_index"]],
                    "observed_history": history["observed_history"][:, item["signal_index"]],
                }
            gradients[projection] = partials
            del raw, grads, summaries
    return {"outputs": {name: outputs[name].detach() for name in (*_COMPONENTS, _CANDIDATE, _CURVE, _TIE, _HORIZON)},
            "projections": tuple(projections), "gradients": gradients, "routes": routes,
            "forward_calls": 1, "reverse_calls": REVERSE_CALLS,
            "interpretation": "local_input_sensitivity_only_no_learning_or_admission"}


def _normalization_surface(normalization, surface, layout):
    import torch
    key = "signal" if surface in ("seq_x", "snap_x") else "mtf_" + surface[4:] if surface in _MTF else surface
    surfaces = normalization.get("surfaces", {}) if isinstance(normalization, Mapping) else {}
    spec = surfaces.get(key)
    if not isinstance(spec, Mapping) or tuple(spec.get("field_names", ())) != tuple(layout["domains"][surface]["field_names"]):
        _fail("NORMALIZATION_FIELD_ORDER:" + key)
    center, scale = torch.as_tensor(spec.get("center")), torch.as_tensor(spec.get("scale"))
    if center.ndim != 1 or scale.shape != center.shape or center.numel() != len(spec["field_names"]) or not bool(
            torch.isfinite(center).all() & torch.isfinite(scale).all() & (scale > 0).all()):
        _fail("NORMALIZATION_STATS:" + key)
    return center, scale


def make_dynamic_compact_counterfactual(inputs, *, layout, token, witness_row, kind, normalization, time_index=None):
    """Build one B1 encoder-domain case. No absent observation is invented.

    kind=value|mask_drop|category|family. Numeric/alias cases default to the current
    bar; a supplied negative history index is explicit and never current evidence.
    Family zeroing is a coupled ablation, not attribution to each constituent.
    """
    import torch
    original = _flat_inputs(inputs)
    if type(witness_row) is not int or not 0 <= witness_row < original["seq_x"].shape[0]:
        _fail("WITNESS_ROW")
    groups = (layout["numeric"], layout["aliases"], layout["categorical"], layout["families"])
    matches = [item for group in groups for item in group if item["token"] == token]
    if len(matches) != 1:
        _fail("OWNER_TOKEN")
    owner = matches[0]
    flat = {name: value[witness_row:witness_row + 1].detach().clone() for name, value in original.items()}
    require_dynamic_compact_inputs(_nest_inputs(flat), layout=layout)
    changed = []
    result = {"token": token, "kind": kind, "witness_row": witness_row,
              "route": "current" if time_index is None or time_index == -1 else "history",
              "time_index": time_index, "changed_coordinates": changed}

    def reject(status, reason):
        return {**result, "status": status, "reason": reason}

    def put(surface, index, value, row=None):
        array = flat[surface]
        location = (0, row, index) if array.ndim == 3 else (0, index)
        before = array[location].clone()
        array[location] = value
        if not _same_bits(before.reshape(1), array[location].reshape(1)):
            changed.append({"surface": surface, "source_index": index, "time_index": row})

    def zero_closure(surface, indices, rows):
        # Mask dependents are closed transitively, with compact change descriptors.
        closure = set(indices)
        while True:
            expanded = closure | {value for value, mask in layout["domains"][surface]["masking_pairs"] if mask in closure}
            if expanded == closure:
                break
            closure = expanded
        columns, rows = sorted(closure), tuple(rows)
        if not columns:
            return
        array = flat[surface]
        if array.ndim == 3:
            selected_rows = list(rows)
            before = array[0, selected_rows][:, columns]
            array[0, selected_rows] = array[0, selected_rows].index_fill(-1, torch.tensor(columns, device=array.device), 0)
        else:
            before = array[:, columns]
            array[:, columns] = 0
        for index, constant in layout["domains"][surface]["source_constants"].items():
            if index in closure:
                if array.ndim == 3:
                    array[0, list(rows), index] = constant
                else:
                    array[:, index] = constant
        after = array[0, list(rows)][:, columns] if array.ndim == 3 else array[:, columns]
        if not _same_bits(before, after):
            changed.append({"surface": surface, "source_indices": tuple(columns), "time_indices": rows})

    is_family = owner in layout["families"]
    if is_family:
        if kind != "family" or time_index is not None:
            _fail("FAMILY_KIND")
        if owner["surface"] == "local":
            zero_closure("seq_x", owner["signal_indices"], range(flat["seq_x"].shape[1]))
            zero_closure("snap_x", owner["signal_indices"], (None,))
            zero_closure("ctx_cont", owner["ctx_cont_indices"], (None,))
            zero_closure("ctx_cat", owner["ctx_cat_indices"], (None,))
        else:
            lane = owner["surface"]
            zero_closure(lane, owner["source_indices"], range(flat[lane].shape[1]))
        result["route"] = "family_all_consumed_bars"
    else:
        alias = owner in layout["aliases"]
        surface = "seq_x" if alias or owner["surface"] == "signal" else owner["surface"]
        index = owner["signal_index"] if alias else owner["source_index"]
        spec = layout["domains"][surface]
        rank = flat[surface].ndim
        if rank == 3:
            row = -1 if time_index is None else time_index
            if type(row) is not int or not -flat[surface].shape[1] <= row < 0:
                _fail("TIME_INDEX_MUST_BE_CAUSAL_NEGATIVE")
        else:
            if time_index is not None:
                _fail("SCALAR_TIME_INDEX")
            row = None
        result["time_index"] = row
        locations = [(surface, index, row)]
        if alias and row == -1:
            locations += [("snap_x", index, None), ("ctx_cont", owner["ctx_cont_index"], None)]
        elif surface == "seq_x" and row == -1 and owner["surface"] != "signal":
            locations += [("snap_x", index, None)]
        elif surface == "snap_x":
            locations += [("seq_x", index, -1)]
        elif owner["surface"] == "signal":
            # A local enum is the existing shared seq+snap embedding owner.
            locations = [("seq_x", index, t) for t in range(-flat["seq_x"].shape[1], 0)] + [("snap_x", index, None)]
            result["route"] = "enum_local_history_and_snapshot"
        if index in spec["source_constants"]:
            return reject("source_declared_constant", "source_constant_is_not_a_free_coordinate")
        own_masks = [mask for value, mask in spec["masking_pairs"] if value == index]
        current = flat[surface][0, row, index] if rank == 3 else flat[surface][0, index]
        for mask in own_masks:
            observed = flat[surface][0, row, mask] if rank == 3 else flat[surface][0, mask]
            if observed.item() != 1:
                return reject("not_exercised_missing", "own_observation_mask_zero")
        dependents = [value for value, mask in spec["masking_pairs"] if mask == index]
        if dependents:
            if kind != "mask_drop":
                return reject("invalid_counterfactual", "observation_mask_requires_atomic_drop")
            if current.item() != 1:
                return reject("not_exercised_missing", "cannot_invent_observation")
            for target, target_index, target_row in locations:
                zero_closure(target, (target_index,), (target_row,))
        else:
            domain = spec["enum_domains"].get(index)
            binary = index in spec["binary_indices"]
            if domain is not None:
                if kind not in ("category", "value"):
                    return reject("invalid_counterfactual", "enum_kind")
                replacement = domain[(domain.index(current.item()) + 1) % len(domain)]
            elif binary:
                if kind != "value":
                    return reject("invalid_counterfactual", "binary_value_kind")
                replacement = 1.0 - current.item()
            else:
                if kind != "value":
                    return reject("invalid_counterfactual", "continuous_value_kind")
                center, scale = _normalization_surface(normalization, surface, layout)
                center, scale = float(center[index]), float(scale[index])
                replacement = center + scale if current.item() == center else center
                # This fraction has an explicit physical [0,1] domain, unlike generic asinh inputs.
                if spec["field_names"][index].removeprefix("ctx_cont.") == "close_distance_below_high_range_fraction" and not 0 <= replacement <= 1:
                    return reject("invalid_counterfactual", "range_fraction_replacement_out_of_domain")
            if (not math.isfinite(float(replacement))
                    or abs(float(replacement)) > torch.finfo(torch.float32).max):
                return reject("invalid_counterfactual", "replacement_not_finite_float32")
            for target, target_index, target_row in locations:
                put(target, target_index, replacement, target_row)
    nested = _nest_inputs(flat)
    require_dynamic_compact_inputs(nested, layout=layout)
    if not changed:
        return reject("invalid_counterfactual", "no_changed_coordinate")
    return {**result, "status": "valid", "inputs": nested,
            "meaning": "coupled_encoder_domain_intervention_not_market_realizability_or_admission"}


def pack_dynamic_compact_cases(cases, *, baseline_inputs, batch_rows=BATCH_ROWS):
    """Yield fixed B16 independent one-witness cases; padding never earns evidence."""
    import torch
    if type(batch_rows) is not int or batch_rows != BATCH_ROWS:
        _fail("PACK_BATCH_MUST_BE_16")
    baseline = _flat_inputs(baseline_inputs)
    if not 1 <= baseline["seq_x"].shape[0] <= BATCH_ROWS:
        _fail("PACK_BASELINE_BATCH")
    stream = iter(cases)
    count = 0
    while True:
        chunk = tuple(islice(stream, batch_rows))
        if not chunk:
            return
        count += len(chunk)
        if count > MAX_CASES:
            _fail("CASE_BUDGET")
        flats = []
        for case in chunk:
            if case.get("status") != "valid":
                _fail("PACK_INVALID_CASE")
            item = _flat_inputs(case["inputs"])
            for name, value in item.items():
                if value.shape != (1, *baseline[name].shape[1:]) or value.dtype != baseline[name].dtype or value.device != baseline[name].device:
                    _fail("PACK_SHAPE_DTYPE_DEVICE:" + name)
            flats.append(item)
        padded = flats + [flats[-1]] * (batch_rows - len(flats))
        yield {"inputs": _nest_inputs({name: torch.cat([item[name] for item in padded], dim=0) for name in baseline}),
               "case_tokens": tuple(case["token"] for case in chunk) + (None,) * (batch_rows - len(chunk)),
               "witness_rows": tuple(case["witness_row"] for case in chunk) + (None,) * (batch_rows - len(chunk)),
               "score_mask": torch.arange(batch_rows, device=baseline["seq_x"].device) < len(chunk)}
        del chunk, flats, padded


def require_dynamic_compact_output_parity(expected, actual, *, inverse_permutation=None):
    """Exact fixed-shape control, including component, candidate, tie and query identity."""
    import torch
    batch = expected[_CANDIDATE].shape[0]
    _require_outputs(expected, batch)
    _require_outputs(actual, batch)
    if inverse_permutation is not None:
        inverse = torch.as_tensor(inverse_permutation, device=expected[_CANDIDATE].device)
        if inverse.dtype != torch.int64 or inverse.shape != (batch,) or sorted(inverse.tolist()) != list(range(batch)):
            _fail("PARITY_PERMUTATION")
    else:
        inverse = None
    keys = (*_COMPONENTS, _CANDIDATE, _CURVE, _TIE, _HORIZON)
    for name in keys:
        value = actual[name]
        if inverse is not None and name != _HORIZON:
            value = value.index_select(0, inverse)
        if not _same_bits(expected[name], value):
            _fail("OUTPUT_PARITY:" + name)
    return {"bit_exact": True, "rows": batch, "output_keys": keys}


def dynamic_compact_output_response(expected, actual, *, witness_rows, score_mask):
    """Score valid packed rows only; component-only change never becomes candidate proof."""
    import torch
    batch = actual[_CANDIDATE].shape[0]
    _require_outputs(expected, expected[_CANDIDATE].shape[0])
    _require_outputs(actual, batch)
    if (not isinstance(score_mask, torch.Tensor) or score_mask.dtype != torch.bool
            or score_mask.shape != (batch,) or len(witness_rows) != batch):
        _fail("RESPONSE_ROW_MAP")
    positions = score_mask.nonzero(as_tuple=False).flatten()
    rows = []
    for position, witness in enumerate(witness_rows):
        if bool(score_mask[position]):
            if type(witness) is not int or not 0 <= witness < expected[_CANDIDATE].shape[0]:
                _fail("RESPONSE_WITNESS")
            rows.append(witness)
        elif witness is not None:
            _fail("PADDING_MUST_NOT_BE_SCORED")
    source_rows = torch.tensor(rows, dtype=torch.int64, device=expected[_CANDIDATE].device)
    reference = expected[_CANDIDATE].index_select(0, source_rows)
    candidate = actual[_CANDIDATE].index_select(0, positions)
    margins_reference = torch.stack((reference[:, 0] - reference[:, 1], reference[:, 0] - reference[:, 2],
                                     reference[:, 1] - reference[:, 2]), dim=1)
    margins_candidate = torch.stack((candidate[:, 0] - candidate[:, 1], candidate[:, 0] - candidate[:, 2],
                                     candidate[:, 1] - candidate[:, 2]), dim=1)
    return {"packed_rows": positions, "witness_rows": tuple(rows),
            "candidate_margin_delta_bps": (margins_candidate - margins_reference).abs().amax(dim=1),
            "component_delta_bps": {
                name: (actual[name].index_select(0, positions) - expected[name].index_select(0, source_rows)).abs().amax(dim=(1, 2))
                for name in _COMPONENTS},
            "interpretation": "finite_intervention_sensitivity_only_candidate_and_components_separate"}


def dynamic_compact_influence_budget(predeclared_cases, executed_cases):
    """Check the already bound schedule; this function authorizes no execution."""
    if (type(predeclared_cases) is not int or type(executed_cases) is not int or
            not 0 <= executed_cases <= predeclared_cases <= MAX_CASES):
        _fail("CASE_BUDGET")
    counterfactual = (executed_cases + BATCH_ROWS - 1) // BATCH_ROWS
    total = 2 + counterfactual
    if total > MAX_FORWARD_CALLS:
        _fail("FORWARD_BUDGET")
    return {"predeclared_cases": predeclared_cases, "executed_cases": executed_cases,
            "gradient_forward_calls": 1, "parity_forward_calls": 1,
            "counterfactual_forward_calls": counterfactual, "forward_calls": total,
            "reverse_calls": REVERSE_CALLS, "execution_authorized": False}


def _dynamic_compact_hash_tensor(digest, name, value):
    import json
    import torch
    if not isinstance(value, torch.Tensor) or value.dtype not in (torch.float32, torch.int64):
        _fail("BIT_DIGEST_DTYPE:" + name)
    if not bool(torch.isfinite(value).all()):
        _fail("BIT_DIGEST_NONFINITE:" + name)
    array = value.detach().cpu().contiguous().numpy()
    dtype = "<f4" if value.dtype == torch.float32 else "<i8"
    raw = array.astype(dtype, copy=False).tobytes(order="C")
    header = (json.dumps({"name": name, "dtype": dtype, "shape": list(value.shape)},
                         sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(len(raw).to_bytes(8, "little"))
    digest.update(raw)


def dynamic_compact_input_sha256(inputs):
    """Frame fixed physical names, dtype, shape and exact little-endian input bits."""
    import hashlib
    import torch
    flat = _flat_inputs(inputs)
    digest = hashlib.sha256(b"gx1_dynamic_compact_input_bits_v1\n")
    batch = None
    for name, value in flat.items():
        rank = 3 if name == "seq_x" or name in _MTF else 2
        dtype = torch.int64 if name == "ctx_cat" else torch.float32
        if (not isinstance(value, torch.Tensor) or value.ndim != rank or value.dtype != dtype
                or min(value.shape) < 1 or batch is not None and value.shape[0] != batch):
            _fail("BIT_DIGEST_SHAPE:" + name)
        batch = value.shape[0]
        _dynamic_compact_hash_tensor(digest, name, value)
    return digest.hexdigest()


def dynamic_compact_normalization_f32_view(normalization, *, layout):
    """Clone the seven source-stat surfaces exactly as model float32 buffers."""
    import torch
    surfaces = normalization.get("surfaces") if isinstance(normalization, Mapping) else None
    if not isinstance(surfaces, Mapping):
        _fail("NORMALIZATION_SURFACES")
    result = {"surfaces": {}}
    for surface in ("seq_x", "ctx_cont", *_MTF):
        key = "signal" if surface == "seq_x" else "mtf_" + surface[4:] if surface in _MTF else surface
        spec = surfaces.get(key)
        if (not isinstance(spec, Mapping)
                or tuple(spec.get("field_names", ())) != tuple(layout["domains"][surface]["field_names"])):
            _fail("NORMALIZATION_FIELD_ORDER:" + key)
        center = torch.as_tensor(spec.get("center"), dtype=torch.float32).detach().cpu().clone()
        scale = torch.as_tensor(spec.get("scale"), dtype=torch.float32).detach().cpu().clone()
        if (center.ndim != 1 or scale.shape != center.shape or center.numel() != len(spec["field_names"])
                or not bool(torch.isfinite(center).all() & torch.isfinite(scale).all() & (scale > 0).all())):
            _fail("NORMALIZATION_STATS:" + key)
        result["surfaces"][key] = {"field_names": tuple(spec["field_names"]), "center": center, "scale": scale}
    return result


def _dynamic_compact_normalization_sha256(view):
    import hashlib
    import json
    digest = hashlib.sha256(b"gx1_dynamic_compact_normalization_f32_v1\n")
    for key in ("signal", "ctx_cont", *("mtf_" + lane[4:] for lane in _MTF)):
        spec = view["surfaces"][key]
        fields = (json.dumps(list(spec["field_names"]), separators=(",", ":")) + "\n").encode("utf-8")
        digest.update(len(fields).to_bytes(8, "little"))
        digest.update(fields)
        for suffix in ("center", "scale"):
            _dynamic_compact_hash_tensor(digest, key + "." + suffix, spec[suffix])
    return digest.hexdigest()


def _dynamic_compact_case_source(owner, group, layout):
    if group == "families":
        return None, None, None, (), ()
    alias = group == "aliases"
    surface = "seq_x" if alias or owner["surface"] == "signal" else owner["surface"]
    index = owner["signal_index"] if alias else owner["source_index"]
    spec = layout["domains"][surface]
    own = tuple(mask for value, mask in spec["masking_pairs"] if value == index)
    dependents = tuple(value for value, mask in spec["masking_pairs"] if mask == index)
    domain = ("source_constant" if index in spec["source_constants"] else
              "observation_mask" if dependents else "source_enum" if index in spec["enum_domains"] else
              "source_binary" if index in spec["binary_indices"] else "finite_asinh_input")
    return surface, index, domain, own, dependents


def _dynamic_compact_case_eligible(flat, layout, owner, group, row):
    import torch
    def coordinate(surface, index, *, all_consumed=False):
        spec = layout["domains"][surface]
        array = flat[surface]
        values = (array[row] if all_consumed else array[row, -1:]) if array.ndim == 3 else array[row:row + 1]
        own = [mask for source, mask in spec["masking_pairs"] if source == index]
        eligible = values.new_ones(values.shape[0], dtype=torch.bool)
        for mask in own:
            eligible &= values[:, mask] == 1
        # A mask-drop may not turn absence into a synthetic observation.
        if any(mask == index for _, mask in spec["masking_pairs"]):
            eligible &= values[:, index] == 1
        return bool(eligible.any())
    if group == "families":
        if owner["surface"] == "local":
            routes = (("seq_x", owner["signal_indices"]), ("ctx_cont", owner["ctx_cont_indices"]),
                      ("ctx_cat", owner["ctx_cat_indices"]))
        else:
            routes = ((owner["surface"], owner["source_indices"]),)
        return any(coordinate(surface, index, all_consumed=True) for surface, indices in routes for index in indices)
    surface, index, domain, _, _ = _dynamic_compact_case_source(owner, group, layout)
    return domain == "source_constant" or coordinate(surface, index)


def build_dynamic_compact_case_manifest(inputs, *, layout, normalization, parent_rows, decision_times_ns, genuine_rows=3):
    """Freeze current legal cases using input masks only; construct no model or fit.

    The compiled production layout supplies 3446 physical owner labels. Counts
    are also checked against the passed layout so small source-shaped hand
    fixtures can exercise the same algorithm without a full native constructor.
    Caller admission still requires the exact source-compiled production layout.
    """
    import hashlib
    import json
    flat = require_dynamic_compact_inputs(inputs, layout=layout)
    if (type(genuine_rows) is not int or not 1 <= genuine_rows <= flat["seq_x"].shape[0]
            or not isinstance(parent_rows, (tuple, list)) or not isinstance(decision_times_ns, (tuple, list))
            or len(parent_rows) != genuine_rows or len(decision_times_ns) != genuine_rows
            or any(type(value) is not int or value < 0 for value in parent_rows)
            or any(type(value) is not int for value in decision_times_ns)
            or len(set(parent_rows)) != genuine_rows):
        _fail("CASE_WITNESS_IDENTITY")
    groups = ("numeric", "aliases", "categorical", "families")
    counts = {key: len(layout[key]) for key in groups}
    tokens = [owner["token"] for key in groups for owner in layout[key]]
    if (counts != dict(layout["counts"]) or len(tokens) != len(set(tokens))
            or any(not isinstance(token, str) for token in tokens) or not 0 < len(tokens) <= MAX_CASES):
        _fail("CASE_OWNER_INVENTORY")
    view = dynamic_compact_normalization_f32_view(normalization, layout=layout)
    norm_sha = _dynamic_compact_normalization_sha256(view)
    encoded_layout = (json.dumps(layout, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    order = sorted(range(genuine_rows), key=lambda row: (decision_times_ns[row], parent_rows[row]))
    cases = []
    statuses = {name: 0 for name in ("valid", "not_exercised_missing", "source_declared_constant", "invalid_counterfactual")}
    for group in groups:
        for owner in layout[group]:
            surface, index, domain, own, dependent = _dynamic_compact_case_source(owner, group, layout)
            kind = "family" if group == "families" else "mask_drop" if dependent else "category" if domain == "source_enum" else "value"
            recipe = ("zero_compiled_family_with_transitive_masks_and_constants" if kind == "family" else
                      "source_observed_mask_drop_1_to_0_and_dependent_storage_zero" if kind == "mask_drop" else
                      "next_source_declared_enum_member" if domain == "source_enum" else
                      "source_binary_1_minus_current" if domain == "source_binary" else
                      "source_declared_constant_no_intervention" if domain == "source_constant" else
                      "frozen_F32_center_else_Python_sum_center_plus_scale_then_F32")
            row = next((row for row in order if _dynamic_compact_case_eligible(flat, layout, owner, group, row)), None)
            item = {"ordinal": len(cases), "owner_group": group, "token": owner["token"], "kind": kind,
                    "status": "not_exercised_missing", "reason": "no_genuine_current_eligible_witness",
                    "witness_row": row, "physical_parent_row": None if row is None else parent_rows[row],
                    "decision_time_ns": None if row is None else decision_times_ns[row],
                    "route": "current", "time_index": None, "effective_time_index": None,
                    "source_surface": surface, "source_index": index,
                    "source_domain_kind": "family_ablation" if group == "families" else domain,
                    "own_observation_index": own[0] if own else None, "own_observation_indices": list(own),
                    "changed_coordinates": [], "input_sha256": None,
                    "replacement_recipe": {"name": recipe, "normalization_f32_sha256": norm_sha,
                        "source_enum_domain": [] if surface is None else list(layout["domains"][surface]["enum_domains"].get(index, ())),
                        "source_constant": None if surface is None else layout["domains"][surface]["source_constants"].get(index),
                        "selection_uses_only_source_masks_and_fixed_time_parent_order": True,
                        "eligibility_scope": "all_consumed_family_coordinates" if group == "families" else "current_coordinate",
                        "retry_allowed": False}}
            if row is not None:
                # Exactly one attempt after mask-only selection. Invalid does
                # not trigger a different witness, recipe or repaired domain.
                made = make_dynamic_compact_counterfactual(inputs, layout=layout, token=owner["token"],
                    witness_row=row, kind=kind, normalization=view, time_index=None)
                if made["status"] not in statuses:
                    _fail("CASE_STATUS")
                item.update(status=made["status"], reason=made.get("reason"), route=made["route"],
                            effective_time_index=made["time_index"],
                            changed_coordinates=made["changed_coordinates"])
                if made["status"] == "valid":
                    item["input_sha256"] = dynamic_compact_input_sha256(made["inputs"])
                del made
            statuses[item["status"]] += 1
            # Normalize tuple descriptors into plain JSON values, retaining no tensors.
            cases.append(json.loads(json.dumps(item, sort_keys=True, allow_nan=False)))
    valid = statuses["valid"]
    result = {"schema_version": "gx1_dynamic_compact_case_manifest_v1",
        "input_sha256": dynamic_compact_input_sha256(inputs), "normalization_f32_sha256": norm_sha,
        "layout_sha256": hashlib.sha256(encoded_layout).hexdigest(), "genuine_rows": genuine_rows,
        "parent_rows": list(parent_rows), "decision_times_ns": list(decision_times_ns), "witness_order": order,
        "owner_counts": counts, "inventory_label_count": len(cases), "valid_case_count": valid,
        "status_counts": statuses, "cases": cases,
        "budget": dynamic_compact_influence_budget(valid, valid),
        "max_inventory_budget": dynamic_compact_influence_budget(len(cases), len(cases)),
        "skip_rule": "execute_every_valid_predeclared_case",
        "history_scope": "Current numeric/alias finite cases only. Local enum/family cases explicitly change their declared histories; no separate historical value-attribution proof.",
        "physical_route_labels_are_not_independent_current_manifolds": True,
        "selection_uses_predictions_targets_or_gradients": False, "execution_authorized": False}
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result


def _direction_witness_moments(values):
    """Reduce a real derivative only; absent connectivity has no zero placeholder."""
    import torch
    values = values.detach().cpu().double().reshape(-1)
    if not values.numel() or not bool(torch.isfinite(values).all()):
        _fail("DIRECTION_DERIVATIVE_INVALID")
    return {"count": values.numel(), "mean": float(values.mean()),
            "mean_absolute": float(values.abs().mean()),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "l2_norm": float(torch.linalg.vector_norm(values)),
            "nonzero_count": int(torch.count_nonzero(values))}


def _direction_witness_gradient_groups(named, hidden, forecast_input):
    """The actual native parameter names; column groups overlap the decoder group."""
    definitions = {
        "decoder_projection": ("dynamic_m5_query_projection.",),
        "entry_joint_projection": ("entry_q_joint_norm.", "entry_q_joint_in."),
        "shared_mtf_gates": ("family_tf_context_gate.", "family_tf_token_gate."),
        "terminal_head": ("head_dynamic_terminal_net_markout.",),
        "mfe_head": ("head_dynamic_mfe.",),
        "mae_head": ("head_dynamic_pre_mfe_mae_upper.",),
        "forecast_head": ("head_forecast.",),
    }
    groups = {group: [(index, name, "all") for index, (name, _) in enumerate(named)
                      if name.startswith(prefixes)] for group, prefixes in definitions.items()}
    weight = next((index for index, (name, _) in enumerate(named)
                   if name == "dynamic_m5_query_projection.0.weight"), None)
    if weight is None:
        _fail("DIRECTION_DECODER_WEIGHT_MISSING")
    for group, selection in (("decoder_hidden_columns", "hidden"),
                             ("decoder_query_column", "query"),
                             ("decoder_endpoint_column", "endpoint")):
        groups[group] = [(weight, named[weight][0], selection)]
    groups["entry_joint_hidden"] = [(len(named), "outputs.entry_q_joint_hidden", "all")]
    groups["shared_forecast_input"] = [(len(named) + 1, "head_forecast.input.z", "all")]
    return groups, (*[parameter for _, parameter in named], hidden, forecast_input)


def _direction_witness_group_vectors(gradients, targets, groups):
    import torch

    def selected(value, selection):
        return (value[:, :-2] if selection == "hidden" else value[:, -2] if selection == "query"
                else value[:, -1] if selection == "endpoint" else value)

    vectors, summaries = {}, {}
    for group, members in groups.items():
        parts, connected, zero, unused = [], [], [], []
        for index, name, selection in members:
            gradient = gradients[index]
            if gradient is None:
                unused.append(name)
                part = torch.zeros(selected(targets[index].detach(), selection).numel(), dtype=torch.float64)
            else:
                part = selected(gradient.detach(), selection).cpu().double().reshape(-1)
                if not bool(torch.isfinite(part).all()):
                    _fail("DIRECTION_NONFINITE_GRADIENT:" + group)
                connected.append(name)
                if not bool(torch.count_nonzero(part)):
                    zero.append(name)
            parts.append(part)
        vector = torch.cat(parts) if parts else torch.empty(0, dtype=torch.float64)
        norm = float(torch.linalg.vector_norm(vector)) if connected else None
        vectors[group] = vector
        summaries[group] = {
            "status": "no_parameters" if not members else "unused" if not connected
                      else "connected_zero" if norm == 0.0 else "nonzero",
            "l2_norm": norm, "scalar_coordinates": vector.numel(),
            "connected_parameter_names": connected, "connected_zero_parameter_names": zero,
            "unused_parameter_names": unused,
        }
    return vectors, summaries


def _direction_witness_pairs(vectors, summaries, pairs):
    import torch
    result = {}
    for first, second in pairs:
        row = {}
        for group in vectors[first]:
            a, b = summaries[first][group]["l2_norm"], summaries[second][group]["l2_norm"]
            dot = (float(torch.dot(vectors[first][group], vectors[second][group]))
                   if a is not None and b is not None else None)
            row[group] = {"dot": dot, "cosine": dot / (a * b) if a and b else None}
        result[first + "__" + second] = row
    return result


def _direction_witness_terminal_group_losses(*, prediction, target, valid, rms):
    """Canonical170 data contributions only; no task precision or +s here."""
    from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import (
        DYNAMIC_M5_TERMINAL_HORIZON_GROUPS, dynamic_m5_scaled_square_sum,
    )
    pieces = []
    for name, start, end in DYNAMIC_M5_TERMINAL_HORIZON_GROUPS:
        squared, count = dynamic_m5_scaled_square_sum(prediction, target, valid,
            surface="dynamic_terminal_net_markout_bps", rms=rms,
            terminal_query_range=(start, end))
        pieces.append((name, start, end, squared, count))
    active = sum(count > 0 for _, _, _, _, count in pieces)
    losses, statistics = {}, {}
    for name, start, end, squared, count in pieces:
        # Empty sums retain the real connected-zero graph, not invented evidence.
        contribution = squared / count / active if count else squared
        losses[name] = contribution
        mask = valid[:, start - 1:end]
        statistics[name] = {
            "query_minutes": [start, end], "group_observed_cells_including_flat": count,
            "observed_side_cells": int(mask[..., :2].sum()),
            "group_flat_cells": int(mask[..., 2].sum()), "nonempty_groups": active,
            "unit_group_mean": float((squared / count).detach()) if count else None,
            "optimizer_group_coefficient": 1 / active if count else 0,
            "unit_loss_contribution": float(contribution.detach()),
            "role": "observed_optimizer_group" if count else "unobserved_group_not_evidence",
        }
    return losses, statistics


def dynamic_m5_direction_witness_gradients(*, model, outputs, batch, query_inputs,
                                         forecast_input, headline_horizons,
                                         terminal_objective="pooled_v1"):
    """Observe one native eval graph: historic22 or explicit native17016 reverses.

    The caller owns admission, the sole forward, native FP32 dispatch, frozen
    TRAIN/state identities and mode/RNG preservation. This function neither calls
    a model nor mutates its state. Gradients are local eval derivatives, not AdamW
    updates, historical training attribution, learned value or Entry admission.
    """
    import json
    import torch
    from gx1.contracts.entry_causal_m1_outcomes_v1 import (
        DYNAMIC_M1_PREFIX_HORIZON_MINUTES, DYNAMIC_M1_PREFIX_ENDPOINT_KINDS,
        DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT, DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT,
    )
    from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import (
        DYNAMIC_M5_LOSS_SCALE_BATCH_KEY, dynamic_m5_scaled_square_sum,
    )
    from gx1.contracts.entry_observed_market_v1 import (
        DYNAMIC_M5_MARKOUT_TARGET_KEY, DYNAMIC_M5_MARKOUT_VALID_KEY,
        DYNAMIC_M5_PATH_VALID_KEY, DYNAMIC_M5_CLOSED_PATH_VALID_KEY,
    )
    from gx1.models.entry_v10.entry_v10_ctx_train_v3 import (
        _dynamic_m5_task_loss_surfaces, _dynamic_m5_optimizer_task_losses, _joint_task_loss,
    )
    from gx1.models.entry_v10.entry_v10_ctx_hybrid_transformer import (
        DYNAMIC_M5_QUERY_CENTER_MINUTES, DYNAMIC_M5_QUERY_SCALE_MINUTES,
        normalize_dynamic_m5_query_minutes,
    )

    if terminal_objective not in ("pooled_v1", "native_170_groupmean_v1"):
        _fail("DIRECTION_TERMINAL_OBJECTIVE_MODE")
    native170 = terminal_objective == "native_170_groupmean_v1"
    expected_reverses = 16 if native170 else 22
    if not torch.is_grad_enabled() or model.training:
        _fail("DIRECTION_REQUIRES_EVAL_GRAD_GRAPH")
    size = int(batch["seq_x"].shape[0])
    _require_outputs(outputs, size)
    horizons = tuple(DYNAMIC_M1_PREFIX_HORIZON_MINUTES)
    if (not isinstance(headline_horizons, (tuple, list)) or len(headline_horizons) != 7
            or any(type(h) is not int or h not in horizons for h in headline_horizons)
            or tuple(sorted(set(headline_horizons))) != tuple(headline_horizons)):
        _fail("DIRECTION_HEADLINE_HORIZONS")
    hidden = outputs.get("entry_q_joint_hidden")
    if (not isinstance(hidden, torch.Tensor) or hidden.ndim != 2 or hidden.shape[0] != size
            or hidden.dtype != torch.float32 or not hidden.requires_grad
            or not isinstance(forecast_input, torch.Tensor) or forecast_input.shape != hidden.shape
            or forecast_input.dtype != hidden.dtype or forecast_input.device != hidden.device
            or not forecast_input.requires_grad or not bool(torch.isfinite(forecast_input).all())
            or not bool(torch.isfinite(hidden).all())):
        _fail("DIRECTION_REPRESENTATION_GRAPH")
    if not isinstance(query_inputs, (tuple, list)) or len(query_inputs) != len(DYNAMIC_M1_PREFIX_ENDPOINT_KINDS):
        _fail("DIRECTION_QUERY_INPUTS")
    query = normalize_dynamic_m5_query_minutes(torch.tensor(horizons, dtype=hidden.dtype, device=hidden.device))
    for endpoint, value in enumerate(query_inputs):
        if (not isinstance(value, torch.Tensor) or value.shape != (size, len(horizons), hidden.shape[1] + 2)
                or value.dtype != hidden.dtype or value.device != hidden.device or not value.requires_grad
                or not _same_bits(value[..., :-2], hidden[:, None].expand(-1, len(horizons), -1))
                or not _same_bits(value[..., -2], query[None].expand(size, -1))
                or bool((value[..., -1] != endpoint).any())):
            _fail("DIRECTION_QUERY_ORDER_OR_IDENTITY")
    if any(parameter.grad is not None for parameter in model.parameters()):
        _fail("DIRECTION_MODEL_GRAD_PREEXISTS")
    raw, cells = _dynamic_m5_task_loss_surfaces(outputs, batch, hidden.device)
    unit, unit_cells = _dynamic_m5_task_loss_surfaces(outputs, batch, hidden.device, loss_units="train_rms")
    optimizer, group_cells = ( _dynamic_m5_optimizer_task_losses(
        outputs, batch, hidden.device, pooled_task_losses=unit) if native170 else (unit, None))
    task_names = tuple(model.task_log_variances)
    # The fixed physical witness requires all four tasks; do not manufacture a
    # missing task or its +s just to fill the declared reverse-call schedule.
    if (len(task_names) != 4 or set(unit) != set(task_names)
            or set(optimizer) != set(task_names) or cells != unit_cells):
        _fail("DIRECTION_TASK_SUPERVISION_OR_DENOMINATOR")
    joint_loss, joint_stats = _joint_task_loss(model, optimizer, loss_units="train_rms")
    prefixes = ("dynamic_m5_query_projection.", "entry_q_joint_norm.", "entry_q_joint_in.",
                "family_tf_context_gate.", "family_tf_token_gate.",
                "head_dynamic_terminal_net_markout.", "head_dynamic_mfe.",
                "head_dynamic_pre_mfe_mae_upper.", "head_forecast.")
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if name.startswith(prefixes) and parameter.requires_grad]
    groups, targets = _direction_witness_gradient_groups(named, hidden, forecast_input)
    precision = {name: joint_stats["joint_task_effective_precision_" + name] for name in task_names}
    # The native scalarizer also owns each projection; +s has no derivative on
    # these parameter/representation targets, which exclude task log variances.
    projections = {name: _joint_task_loss(model, {name: optimizer[name]}, loss_units="train_rms")[0]
                   for name in task_names}
    terminal_name = _COMPONENTS[0]
    target = batch[DYNAMIC_M5_MARKOUT_TARGET_KEY].detach().to(hidden.device)
    valid = batch[DYNAMIC_M5_MARKOUT_VALID_KEY].detach().to(hidden.device)
    rms = batch[DYNAMIC_M5_LOSS_SCALE_BATCH_KEY]["rms"][terminal_name]
    partitions, partition_losses = {}, []
    if native170:
        group_losses, group_statistics = _direction_witness_terminal_group_losses(
            prediction=outputs[terminal_name], target=target, valid=valid, rms=rms)
        if group_cells != {name: item["group_observed_cells_including_flat"]
                           for name, item in group_statistics.items()}:
            _fail("DIRECTION_TERMINAL_GROUP_DENOMINATOR")
        for name, loss in group_losses.items():
            key = "terminal." + name
            # Use the sole native scalarizer; cancel its +s for these data-only
            # contributions. Log-variance parameters are not gradient targets.
            projections[key] = (_joint_task_loss(model, {terminal_name: loss},
                loss_units="train_rms")[0] - model.task_log_variances[terminal_name])
            partition_losses.append(loss)
            partitions[key] = group_statistics[name]
    else:
        selected = set(headline_horizons)
        partitions, partition_losses = {}, []
        for name, minutes in [*(('h' + str(h), (h,)) for h in headline_horizons),
                              ("complement", tuple(h for h in horizons if h not in selected))]:
            horizon_mask = torch.tensor([h in minutes for h in horizons], dtype=torch.bool, device=hidden.device)
            partmask = valid & horizon_mask[None, :, None]
            squared, partition_count = dynamic_m5_scaled_square_sum(
                outputs[terminal_name], torch.where(partmask, target, torch.zeros_like(target)),
                partmask, surface=terminal_name, rms=rms)
            loss = squared / cells[terminal_name]
            key = "terminal." + name
            projections[key] = _joint_task_loss(model, {terminal_name: loss}, loss_units="train_rms")[0]
            partition_losses.append(loss)
            partitions[key] = {"query_minutes": list(minutes), "observed_side_cells": int(partmask[..., :2].sum()),
                "partition_observed_cells_including_flat": partition_count,
                "partition_flat_cells": int(partmask[..., 2].sum()),
                "analytic_flat_cells_in_native_denominator": int(valid[..., 2].sum()),
                "native_task_denominator": cells[terminal_name], "unit_loss_contribution": float(loss.detach()),
                "role": "observed_loss_partition" if bool(partmask.any()) else "unobserved_partition_not_evidence"}
    vectors, summaries, reverse_calls = {}, {}, 0
    for name, scalar in projections.items():
        gradients = torch.autograd.grad(scalar, targets, retain_graph=True, create_graph=False, allow_unused=True)
        reverse_calls += 1
        vectors[name], summaries[name] = _direction_witness_group_vectors(gradients, targets, groups)
        del gradients
    task_pairs = [(a, b) for index, a in enumerate(task_names) for b in task_names[index + 1:]]
    terminal_pairs = [(name, terminal_name) for name in partitions]
    partition_pairs = ([( "terminal.early_1_15", "terminal.later_16_480")] if native170 else
                       [(name, "terminal.complement") for name in partitions if name != "terminal.complement"])
    gradient_pairs = {
        "tasks": _direction_witness_pairs(vectors, summaries, task_pairs),
        "terminal_partitions_to_native_terminal": _direction_witness_pairs(vectors, summaries, terminal_pairs),
        ("terminal_early_to_later" if native170 else "terminal_headlines_to_complement"):
            _direction_witness_pairs(vectors, summaries, partition_pairs),
    }
    group_sum_residual = {}
    if native170:
        for group, native_vector in vectors[terminal_name].items():
            connected = summaries[terminal_name][group]["l2_norm"] is not None
            residual = sum((vectors[name][group] for name in partitions),
                           torch.zeros_like(native_vector)) - native_vector
            group_sum_residual[group] = {
                "status": "observed_gradient_decomposition" if connected else "unused_not_evidence",
                "l2_norm": float(torch.linalg.vector_norm(residual)) if connected else None,
                "max_absolute": float(residual.abs().max()) if connected and residual.numel() else None,
                "native_l2_norm": summaries[terminal_name][group]["l2_norm"],
            }
    query_shares = {}
    for name in projections:
        total = summaries[name]["decoder_projection"]["l2_norm"]
        coordinate = summaries[name]["decoder_query_column"]["l2_norm"]
        query_shares[name] = (coordinate ** 2 / total ** 2
                              if coordinate is not None and total else None)
    sensitivity = {}
    for component in _COMPONENTS:
        endpoint_kind = (DYNAMIC_M1_PREFIX_CLOSED_ENDPOINT if component in _COMPONENTS[3:]
                         else DYNAMIC_M1_PREFIX_TERMINAL_ENDPOINT)
        endpoint = DYNAMIC_M1_PREFIX_ENDPOINT_KINDS.index(endpoint_kind)
        for side, label in enumerate(("long", "short")):
            scalar = outputs[component][..., side].sum()
            if not scalar.requires_grad:
                _fail("DIRECTION_QUERY_OUTPUT_DISCONNECTED")
            last = component == _COMPONENTS[-1] and side == 1
            gradients = torch.autograd.grad(scalar, tuple(query_inputs), retain_graph=not last,
                                            create_graph=False, allow_unused=True)
            reverse_calls += 1
            row = {"endpoint_kind": DYNAMIC_M1_PREFIX_ENDPOINT_KINDS[endpoint],
                   "query_coordinate_units": "standardized_fixed_query_grid_v1",
                   "query_coordinate_center_minutes": DYNAMIC_M5_QUERY_CENTER_MINUTES,
                   "query_coordinate_scale_minutes": DYNAMIC_M5_QUERY_SCALE_MINUTES,
                   "derivative_units": "bps_per_minute",
                   "endpoint_input_connectivity": ["unused" if g is None else
                       "connected_zero" if not bool(torch.count_nonzero(g)) else "nonzero" for g in gradients]}
            gradient = gradients[endpoint]
            other = gradients[1 - endpoint]
            if other is not None and bool(torch.count_nonzero(other)):
                _fail("DIRECTION_QUERY_ENDPOINT_CROSS_CONNECTION")
            if gradient is None:
                row["status"] = "unused"
            else:
                if not bool(torch.isfinite(gradient).all()):
                    _fail("DIRECTION_NONFINITE_QUERY_GRADIENT")
                derivative = gradient[..., -2] / DYNAMIC_M5_QUERY_SCALE_MINUTES
                row.update(status="nonzero" if bool(torch.count_nonzero(derivative)) else "connected_zero",
                    all_queries=_direction_witness_moments(derivative),
                    headline_queries={str(h): _direction_witness_moments(derivative[:, horizons.index(h)])
                                      for h in headline_horizons})
            sensitivity[component + "." + label] = row
            del gradients
    if reverse_calls != expected_reverses or any(parameter.grad is not None for parameter in model.parameters()):
        _fail("DIRECTION_REVERSE_BUDGET_OR_GRAD_MUTATION")
    result = {
        "schema_version": ("gx1_dynamic_m5_direction_witness_gradients_v2" if native170 else
                           "gx1_dynamic_m5_direction_witness_gradients_v1"),
        "scope": "local_final_state_eval_derivatives_on_TRAIN_not_epoch_causality_or_AdamW_updates",
        "forward_calls": 0, "reverse_calls": reverse_calls, "optimizer_updates": 0,
        "model_training": bool(model.training), "joint_loss": float(joint_loss.detach()),
        "native_joint_statistics": joint_stats,
        "task_statistics": {name: {"observed_task_cells": cells[name], "raw_bps_mse": float(raw[name].detach()),
            "train_rms_mse": float(unit[name].detach()),
            **({"pooled_train_rms_mse": float(unit[name].detach()),
                "optimizer_objective": float(optimizer[name].detach())} if native170 else {}),
            "log_variance": float(model.task_log_variances[name].detach()),
            "effective_precision": precision[name]} for name in task_names},
        "surface_observed_cells": {"terminal_including_flat": int(valid.sum()),
            "terminal_sides": int(valid[..., :2].sum()), "terminal_flat": int(valid[..., 2].sum()),
            "inclusive_path": int(batch[DYNAMIC_M5_PATH_VALID_KEY].sum()),
            "closed_path": int(batch[DYNAMIC_M5_CLOSED_PATH_VALID_KEY].sum())},
        "headline_horizons": list(headline_horizons), "terminal_partitions": partitions,
        ("terminal_group_sum_minus_native_optimizer_objective" if native170 else
         "terminal_partition_sum_minus_native_unit_loss"):
            float((torch.stack(partition_losses).sum() - optimizer[terminal_name]).detach()),
        "parameter_groups": {name: [{"name": label, "selection": selection} for _, label, selection in members]
                             for name, members in groups.items()},
        "gradient_summaries": summaries, "gradient_pairs": gradient_pairs,
        "decoder_query_column_squared_l2_share": query_shares,
        "query_sensitivity": sensitivity,
        "interpretation": "Same-coordinate gradient dots are local first-order alignment only. Column groups overlap decoder_projection. No epsilon, fitted threshold, invented task weight, optimizer or model selection.",
    }
    if native170:
        from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import dynamic_m5_terminal_horizon_objective_contract
        result.update(terminal_objective_mode=terminal_objective,
            terminal_optimizer_objective_contract=dynamic_m5_terminal_horizon_objective_contract(),
            terminal_group_counts=group_cells, active_native_tasks=list(optimizer),
            terminal_group_vector_sum_residual=group_sum_residual,
            terminal_group_projection_role="weighted_data_gradient_only_no_repeated_log_variance_penalty")
    json.dumps(result, sort_keys=True, allow_nan=False)
    return result



def _direction_input_coordinates(layout, normalization):
    """One source-owned current coordinate; physical aliases are participants."""
    from gx1.features.entry_specialist_feature_groups_v1 import MODEL_NATIVE_TRAINING_SPECIALISTS
    from gx1.features.htf_features import MULTI_TF_STRUCTURAL_BINARY_FEATURES_V4
    profile, coordinates = layout["profile"], []
    aliases = {}
    for item in layout["physical_aliases"]:
        if item["signal_index"] in profile["signal_indices"] and item["ctx_cont_index"] in profile["ctx_cont_indices"]:
            aliases.setdefault(item["signal_index"], []).append(item["ctx_cont_index"])
    signal_owner = {i: f for f, indices in profile["local_signal_indices_by_specialist"].items() for i in indices}
    ctx_owner = {i: f for f, indices in profile["ctx_cont_indices_by_specialist"].items() for i in indices}
    mtf_owner = {i: f for f, indices in profile["mtf_indices_by_specialist"].items() for i in indices}

    def add(lane, surface, index, family, parts, history=None, macro=False):
        key = "signal" if surface in ("seq_x", "snap_x") else "mtf_" + surface[4:] if surface in _MTF else surface
        spec, domain = normalization["surfaces"][key], layout["domains"][surface]
        field = domain["field_names"][index]
        identity = bool(spec["binary_mask"][index] or spec["categorical_mask"][index])
        fallback = spec["scale_source"][index] == "constant_train_unit_scale"
        masks = {m for _, m in domain["masking_pairs"]}
        # signal/htf_features preserve these local {-1,0,1} domains even though
        # their frozen055 statistics use asinh rather than categorical identity.
        local_ema_stack_enum = (surface in ("seq_x", "snap_x", "ctx_cont") and
            field.removeprefix("ctx_cont.") in {
                tf + "_ema_stack_aligned_v2" for tf in ("m5", "m15", "h1", "h4", "d1")})
        kind = ("integer_category" if surface == "ctx_cat" else "observation_mask" if index in masks else
                "source_enum" if local_ema_stack_enum or index in domain["enum_domains"] or spec["categorical_mask"][index] else
                "source_binary" if field.removeprefix("ctx_cont.") in MULTI_TF_STRUCTURAL_BINARY_FEATURES_V4 or
                index in domain["binary_indices"] or spec["binary_mask"][index] else
                "constant_train_fallback" if fallback else "continuous")
        coordinates.append({"token": lane + "." + field, "lane": lane, "field": field, "family": family,
            "surface": surface, "source_index": index, "current_participants": parts,
            "history_source": history, "domain_kind": kind, "asinh_applicable": not identity,
            "dispersion_supported": kind == "continuous" and spec["scale_source"][index] in
                ("raw_iqr", "median_positive_abs_deviation"),
            "source_constant": index in domain["source_constants"], "macro_overlap": macro})

    folded = set()
    for index in profile["signal_indices"]:
        ctx = aliases.get(index, [])
        folded.update(ctx)
        macro = any(layout["domains"]["ctx_cont"]["field_names"][c] in profile["macro_context_fields"] for c in ctx)
        add("local", "seq_x", index, signal_owner[index],
            [("seq_x", index), ("snap_x", index), *(("ctx_cont", c) for c in ctx)],
            history=("seq_x", index), macro=macro)
    for index in profile["ctx_cont_indices"]:
        if index not in folded:
            add("local", "ctx_cont", index, ctx_owner[index], [("ctx_cont", index)],
                macro=layout["domains"]["ctx_cont"]["field_names"][index] in profile["macro_context_fields"])
    for surface in _MTF:
        for index in profile["mtf_indices"]:
            add(surface, surface, index, mtf_owner[index], [(surface, index)], history=(surface, index))
    # Integer categories are fully accounted, but never enter a floating VJP.
    categorical = []
    for index, field in enumerate(profile["physical_field_names"]["ctx_cat"]):
        owners = [f["token"].removeprefix("local.") for f in layout["families"]
                  if f["surface"] == "local" and index in f.get("ctx_cat_indices", ())]
        categorical.append({"token": "local.ctx_cat." + field, "lane": "local", "field": field,
            "domain_kind": "integer_category", "source_index": index, "families": owners,
            "status": "integer_input_not_differentiated"})
    if len({c["token"] for c in coordinates}) != len(coordinates):
        _fail("INPUT_VJP_CANONICAL_COORDINATE_DUPLICATE")
    return coordinates, categorical, tuple(MODEL_NATIVE_TRAINING_SPECIALISTS)


def _direction_input_cell_stats(gradient, values, observed, factor, *, connected, current):
    """F64 post-gradient summaries; missing cells and unused surfaces stay explicit."""
    import torch
    count, total = int(observed.sum()), observed.numel()
    selected = gradient[observed] if connected else None
    converted = gradient * factor if connected and factor is not None else None
    result = {"observed_row_or_history_cell_count": count, "missing_cell_count": total - count,
        "observed_zero_input_count": int(((values == 0) & observed).sum()),
        "observed_nonzero_input_count": int(((values != 0) & observed).sum()),
        "status": "not_exercised" if not count else "surface_unused" if not connected else
                  "observed_zero_on_connected_surface" if not bool(torch.count_nonzero(selected)) else "nonzero",
        "observed_zero_count": int((selected == 0).sum()) if selected is not None else None,
        "observed_nonzero_count": int(torch.count_nonzero(selected)) if selected is not None else None,
        "raw_rms": float(torch.sqrt(selected.square().mean())) if count and connected else None,
        "raw_max_absolute": float(selected.abs().max()) if count and connected else None,
        "asinh_squared_sum": float(converted[observed].square().sum()) if converted is not None else None,
        "asinh_rms": float(torch.sqrt(converted[observed].square().mean()))
                     if count and converted is not None else None,
        "asinh_max_absolute": float(converted[observed].abs().max())
                              if count and converted is not None else None}
    if current:
        result["raw_signed_by_row"] = [float(g) if o else None for g, o in zip(gradient, observed)] if connected else None
        result["asinh_signed_by_row"] = [float(g) if o else None for g, o in zip(converted, observed)] if converted is not None else None
    return result, converted


def _direction_input_family_stats(coordinates, fields, indices, phase, *, categorical_count=0):
    import math
    cells = [fields[coordinates[i]["token"]][phase] for i in indices]
    supported = [(coordinates[i], fields[coordinates[i]["token"]][phase]) for i in indices
                 if coordinates[i]["dispersion_supported"]]
    observed = sum(c["observed_row_or_history_cell_count"] for _, c in supported)
    measured = sum(c["observed_row_or_history_cell_count"] for _, c in supported
                   if c["asinh_squared_sum"] is not None)
    square = sum(c["asinh_squared_sum"] or 0. for _, c in supported)
    return {"source_declared_active_field_count": len(indices) + categorical_count,
        "observed_distinct_field_count": sum(c["observed_row_or_history_cell_count"] > 0 for c in cells),
        "observed_row_or_history_cell_count": sum(c["observed_row_or_history_cell_count"] for c in cells),
        "missing_cell_count": sum(c["missing_cell_count"] for c in cells),
        "observed_zero_count": sum(c["observed_zero_count"] or 0 for c in cells),
        "observed_nonzero_count": sum(c["observed_nonzero_count"] or 0 for c in cells),
        "dispersion_supported_observed_cells": observed,
        "connected_dispersion_supported_observed_cells": measured,
        "unused_dispersion_supported_observed_cells": observed - measured,
        "dispersion_supported_asinh_rms": math.sqrt(square / observed) if observed and measured else None,
        "rms_status": "source_inactive" if not indices and not categorical_count else
                      "no_dispersion_supported_fields" if not supported else "not_exercised" if not observed else
                      "surface_unused_not_evidence" if not measured else "measured_local_derivative",
        "surface_autograd_unused_field_count": sum(c["status"] == "surface_unused" for c in cells),
        "binary_enum_mask_constant_fallback_counts": {
            kind: sum(coordinates[i]["domain_kind"] == kind for i in indices) +
                  (categorical_count if kind == "integer_category" else 0)
            for kind in ("source_binary", "source_enum", "observation_mask", "constant_train_fallback", "integer_category")},
        "unused_zero_arithmetic_note": "None is retained at surface level; absent partials enter sums as zero only. All-unused RMS is null."}


def dynamic_m5_direction_input_witness_vjps(*, outputs, flat_inputs, layout, normalization):
    """Five input-only VJPs on the caller's one native graph; no forward or fitting."""
    import json
    import torch
    flat = require_dynamic_compact_inputs(_nest_inputs(flat_inputs), layout=layout)
    if not torch.is_grad_enabled() or any(not flat[s].requires_grad or flat[s].grad is not None for s in _FLOAT_INPUTS):
        _fail("INPUT_VJP_REQUIRES_UNMODIFIED_INPUT_GRAD_GRAPH")
    _require_outputs(outputs, flat["seq_x"].shape[0])
    coordinates, categorical, family_order = _direction_input_coordinates(layout, normalization)
    for item in categorical:
        item["provided_integer_input_rows"] = int(flat["ctx_cat"].shape[0])
    values = {s: flat[s].detach().cpu().double() for s in _FLOAT_INPUTS}
    observed = {s: _observed(values[s], layout["domains"][s]) for s in _FLOAT_INPUTS}
    factors, centers, scales = {}, {}, {}
    for surface in _FLOAT_INPUTS:
        center, scale = _normalization_surface(normalization, surface, layout)
        centers[surface], scales[surface] = center.float().cpu(), scale.float().cpu()
        factors[surface] = torch.hypot(scales[surface].double(),
                                      values[surface] - centers[surface].double())
    # Shared aliases must really use the same raw value, observation and frozen F32 stats.
    for coordinate in coordinates:
        surface, index = coordinate["surface"], coordinate["source_index"]
        if surface == "seq_x":
            for other, column in coordinate["current_participants"][1:]:
                if (not _same_bits(centers[surface][[index]], centers[other][[column]]) or
                        not _same_bits(scales[surface][[index]], scales[other][[column]]) or
                        not torch.equal(observed[surface][:, -1, index], observed[other][:, column])):
                    _fail("INPUT_VJP_ALIAS_COORDINATE_OR_SUPPORT")
    terminal, q = outputs[_COMPONENTS[0]], outputs[_CANDIDATE]
    projections = {"terminal_direction_h" + str(h): (.5 * (terminal[:, h-1, 0] - terminal[:, h-1, 1])).sum()
                   for h in (1, 5, 15)}
    projections.update(candidate_long_minus_flat=(q[:, 0] - q[:, 2]).sum(),
                       candidate_short_minus_flat=(q[:, 1] - q[:, 2]).sum())
    result, current_vectors, current_support = {}, {}, {}
    ordered = tuple(flat[s] for s in _FLOAT_INPUTS)
    for ordinal, (name, scalar) in enumerate(projections.items()):
        if not scalar.requires_grad:
            _fail("INPUT_VJP_SCALAR_DISCONNECTED_NO_ZERO_ANCHOR")
        raw = torch.autograd.grad(scalar, ordered, allow_unused=True, create_graph=False,
                                  retain_graph=ordinal < 4)
        gradients = {s: None if g is None else g.detach().cpu().double() for s, g in zip(_FLOAT_INPUTS, raw)}
        connectivity = {s: {"grad_is_none": g is None, "status": "unused" if g is None else
                           "connected_zero" if not bool(torch.count_nonzero(g)) else "nonzero"}
                        for s, g in gradients.items()}
        if any(g is not None and not bool(torch.isfinite(g).all()) for g in gradients.values()):
            _fail("INPUT_VJP_NONFINITE_GRADIENT")
        fields, vector, support = {}, [], []
        for coordinate in coordinates:
            surface, index = coordinate["surface"], coordinate["source_index"]
            current_values = values[surface][:, -1, index] if values[surface].ndim == 3 else values[surface][:, index]
            current_observed = observed[surface][:, -1, index] if values[surface].ndim == 3 else observed[surface][:, index]
            factor = factors[surface][:, -1, index] if values[surface].ndim == 3 else factors[surface][:, index]
            parts = []
            for owner, column in coordinate["current_participants"]:
                grad = gradients[owner]
                if grad is not None:
                    parts.append(grad[:, -1, column] if grad.ndim == 3 else grad[:, column])
            connected = bool(parts)
            joint = sum(parts, torch.zeros_like(current_values))
            current, converted = _direction_input_cell_stats(joint, current_values, current_observed,
                factor if coordinate["asinh_applicable"] else None, connected=connected, current=True)
            history_owner = coordinate["history_source"]
            if history_owner is not None:
                owner, column = history_owner
                grad = gradients[owner]
                history, _ = _direction_input_cell_stats(
                    grad[:, :-1, column] if grad is not None else torch.zeros_like(values[owner][:, :-1, column]),
                    values[owner][:, :-1, column], observed[owner][:, :-1, column],
                    factors[owner][:, :-1, column] if coordinate["asinh_applicable"] else None,
                    connected=grad is not None, current=False)
            else:
                history, _ = _direction_input_cell_stats(torch.empty(0), torch.empty(0),
                    torch.empty(0, dtype=torch.bool), None, connected=False, current=False)
            fields[coordinate["token"]] = {"current": current, "history": history}
            if coordinate["dispersion_supported"]:
                vector.append(converted if converted is not None else torch.zeros_like(joint))
                support.append(current_observed & connected)
        families = {}
        for lane in ("local", *_MTF):
            for family in family_order:
                indices = [i for i, c in enumerate(coordinates) if c["lane"] == lane and c["family"] == family]
                cats = sum(family in c["families"] for c in categorical) if lane == "local" else 0
                families[lane + "." + family] = {phase: _direction_input_family_stats(
                    coordinates, fields, indices, phase, categorical_count=cats if phase == "current" else 0)
                    for phase in ("current", "history")}
        macro_indices = [i for i, c in enumerate(coordinates) if c["macro_overlap"]]
        result[name] = {"scalar_value_bps": float(scalar.detach()), "surface_connectivity": connectivity,
            "fields": fields, "family_lanes": families,
            "macro_context_overlap": {phase: _direction_input_family_stats(coordinates, fields, macro_indices, phase)
                                      for phase in ("current", "history")}}
        current_vectors[name] = torch.stack(vector, dim=1).reshape(-1) if vector else torch.empty(0, dtype=torch.float64)
        current_support[name] = torch.stack(support, dim=1).reshape(-1) if support else torch.empty(0, dtype=torch.bool)
        del raw, gradients, grad, fields, parts, vector, support, _
    geometry = {}
    names = tuple(projections)
    for i, first in enumerate(names):
        for second in names[i+1:]:
            common = current_support[first] & current_support[second]
            a, b = current_vectors[first][common], current_vectors[second][common]
            count = int(common.sum())
            na, nb = float(torch.linalg.vector_norm(a)), float(torch.linalg.vector_norm(b))
            dot = float(torch.dot(a, b)) if count else None
            geometry[first + "__" + second] = {"common_observed_connected_current_cells": count,
                "dot": dot, "first_l2_norm": na if count else None, "second_l2_norm": nb if count else None,
                "cosine": dot / (na * nb) if count and na and nb else None}
    if any(flat[s].grad is not None for s in _FLOAT_INPUTS):
        _fail("INPUT_VJP_INPUT_GRAD_MUTATION")
    profile = layout["profile"]
    answer = {"schema_version": "gx1_dynamic_m5_direction_input_witness_v1",
        "forward_calls": 0, "reverse_calls": 5, "optimizer_updates": 0,
        "projections": list(projections), "input_surface_order": list(_FLOAT_INPUTS),
        "canonical_current_coordinates": coordinates, "integer_input_accounting": categorical,
        "declared_eight_families": list(family_order), "declared_family_lane_count": 6 * len(family_order),
        "source_active_family_lane_count": len(layout["families"]),
        "macro_context_tag_fields": list(profile["macro_context_fields"]),
        "macro_context_is_overlapping_not_additive": True,
        "actual_sequence_lengths": {s: flat[s].shape[1] for s in ("seq_x", *_MTF)},
        "physical_active_field_references": {"seq_x": len(profile["signal_indices"]),
            "snap_x": len(profile["generic_snap_signal_indices"]), "ctx_cont": len(profile["ctx_cont_indices"]),
            "ctx_cat": len(profile["physical_field_names"]["ctx_cat"]),
            **{s: len(profile["mtf_indices"]) for s in _MTF}},
        "frozen_normalization_constant_fallback_references": sum(
            x == "constant_train_unit_scale" for spec in normalization["surfaces"].values() for x in spec["scale_source"]),
        "units": {"raw": "bps_per_original_feature_unit", "asinh": "bps_per_frozen_TRAIN_asinh_coordinate",
            "conversion": "F32rawgradient_times_F64hypot(F32scale,F32rawinput-F32center)",
            "current_alias_sum": "signed_F64_sum_of_native_F32_physical_partials_before_conversion",
            "not_standard_deviations_or_finite_interventions": True, "task_precision_or_111_loss_weighting": False},
        "candidate_exact_tied_rows_by_side": [int((outputs[_TIE][..., side].sum(dim=1) > 1).sum()) for side in (0, 1)],
        "candidate_derivative": "native max(dim) selected branch; exact ties are nonsmooth",
        "results": result, "current_projection_geometry": geometry,
        "limits": "Local fixed-TRAIN derivatives only. No per-field disconnection from zero tensor entries, feature importance, optimality, OOF value, epoch cause, generalization or economics."}
    json.dumps(answer, allow_nan=False)
    return answer
