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
