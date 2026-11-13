"""Synthetic NumPy-only checks; no market archives, model or fitted data reads."""
import copy
import hashlib
import json

import numpy as np
import pytest

from gx1.audit import ta_current_context_intervention_v1 as owner


def seal(value):
    value = copy.deepcopy(value)
    value.pop("contract_sha256", None)
    value["contract_sha256"] = hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return value


def normalization():
    # Synthetic surfaces at the exact declared layout. Only D1 age reproduces
    # the relevant current128 versus historical127 support discrepancy.
    surfaces = {}
    for surface, width in (("signal", 317), ("ctx_cont", 132)):
        surfaces[surface] = dict(field_names=[surface + "_" + str(i) for i in range(width)],
            center=[0.] * width, scale=[1.] * width, binary_mask=[0] * width,
            categorical_mask=[0] * width, train_transformed_min=[0.] * width,
            train_transformed_max=[0.] * width, continuous_transform=owner.TRANSFORM)
    aliases = []
    for name in owner.FIELDS:
        ci, si = owner.INDICES[name]
        if name == "d1_trend_state_age_bars_v2":
            center, scale, bounds = 182., 54., (128., 236.)
        elif name == "d1_ema_stack_aligned_v2":
            center, scale, bounds = 1., 1., (1., 1.)
        else:
            center, scale, bounds = 2., .5, (1., 3.)
        for surface, index, field_name in (("ctx_cont", ci, name), ("signal", si, "ctx_cont." + name)):
            s = surfaces[surface]
            raw_bounds = (127., 236.) if surface == "signal" and name == "d1_trend_state_age_bars_v2" else bounds
            z = np.arcsinh((np.array(raw_bounds, dtype=np.float64) - center) / scale)
            s["field_names"][index] = field_name
            for key, value in (("center", center), ("scale", scale),
                               ("train_transformed_min", float(z[0])), ("train_transformed_max", float(z[1]))):
                s[key][index] = value
        aliases.append(dict(ctx_cont_field=name, ctx_cont_index=ci, signal_field="ctx_cont." + name,
            signal_index=si, specialist="trend_ema_encoder" if name in owner.D1_FIELDS else "session_regime_encoder"))
    return seal(dict(surfaces=surfaces, temporal_aliases=aliases))


def inputs(spec):
    raw = dict(seq_x=np.zeros((2, 3, 317), np.float32), snap_x=np.zeros((2, 317), np.float32),
               ctx_cont=np.zeros((2, 132), np.float32), ctx_cat=np.array([[0], [3]], np.int64))
    for i, key in enumerate(owner.MTF_KEYS):
        raw[key] = np.full((2, i + 1, 255), i + .25, np.float32)
    for f in spec["fields"]:
        raw["ctx_cont"][:, f["ctx_cont_index"]] = f["center"]
        raw["seq_x"][..., f["signal_index"]] = f["center"]
    raw["seq_x"][..., 0] = np.float32(-0.)
    raw["snap_x"][:] = raw["seq_x"][:, -1]
    return raw


def set_alias(raw, name, values):
    ci, si = owner.INDICES[name]
    raw["seq_x"][..., si] = np.asarray(values, dtype=np.float32)
    raw["snap_x"][:, si] = raw["seq_x"][:, -1, si]
    raw["ctx_cont"][:, ci] = raw["snap_x"][:, si]


def assert_unchanged(raw, before):
    assert {k: v.tobytes() for k, v in raw.items()} == before


def test_spec_uses_current_bounds_without_erasing_different_sequence_support():
    norm = normalization(); before = copy.deepcopy(norm)
    spec = owner.build_spec(norm)
    age = next(f for f in spec["fields"] if f["name"] == "d1_trend_state_age_bars_v2")
    assert age["raw_bounds"] == [128., 236.]
    assert age["signal_raw_bounds"] == [127., 236.]
    assert all(f["exact_fp32_raw_bound_roundtrip"] for f in spec["fields"])
    assert spec["normalization_sha256"] == norm["contract_sha256"] and norm == before
    assert json.loads(json.dumps(spec, allow_nan=False)) == spec


def test_shared_current_interval_clamps_all_alias_times_and_preserves_other_bytes():
    spec = owner.build_spec(normalization()); raw = inputs(spec)
    set_alias(raw, "d1_trend_state_age_bars_v2", [[127., 128., 300.], [100., 236., 127.]])
    set_alias(raw, "d1_ema_stack_aligned_v2", [[-1., 0., 1.], [1., 1., -1.]])
    before = {k: v.tobytes() for k, v in raw.items()}
    for value in raw.values(): value.setflags(write=False)
    out, counts = owner.intervene(raw, spec, "D1")
    age = owner.INDICES["d1_trend_state_age_bars_v2"][1]
    np.testing.assert_array_equal(out["seq_x"][..., age], [[128., 128., 236.], [128., 236., 128.]])
    assert counts["per_field"]["d1_trend_state_age_bars_v2"]["seq_x"] == dict(below=3, above=1, changed=4)
    assert set(counts["changed_arrays"]) == {"seq_x", "snap_x", "ctx_cont"}
    assert out["seq_x"][:, -1].tobytes() == out["snap_x"].tobytes()
    for name in owner.D1_FIELDS:
        ci, si = owner.INDICES[name]
        assert out["snap_x"][:, si].tobytes() == out["ctx_cont"][:, ci].tobytes()
    for key in ("seq_x", "snap_x", "ctx_cont"):
        selected = {owner.INDICES[name][0 if key == "ctx_cont" else 1] for name in owner.D1_FIELDS}
        other = [i for i in range(raw[key].shape[-1]) if i not in selected]
        assert out[key][..., other].tobytes() == raw[key][..., other].tobytes()
        assert not np.shares_memory(out[key], raw[key])
    for key in (*owner.MTF_KEYS, "ctx_cat"):
        assert out[key] is raw[key] and counts["changed_cells"][key] == 0
    assert_unchanged(raw, before)
    assert not any(value.flags.writeable for value in raw.values())
    json.dumps(counts, allow_nan=False)


def test_joint_is_exact_composition_of_disjoint_arms_in_either_order():
    spec = owner.build_spec(normalization()); raw = inputs(spec)
    for name in owner.FIELDS:
        set_alias(raw, name, [[-1e10, 1e10, -1e10], [1e10, -1e10, 1e10]])
    before = {k: v.tobytes() for k, v in raw.items()}
    joint, j = owner.intervene(raw, spec, "JOINT")
    d1, d = owner.intervene(raw, spec, "D1"); dm, m = owner.intervene(d1, spec, "MACRO")
    macro, _ = owner.intervene(raw, spec, "MACRO"); md, _ = owner.intervene(macro, spec, "D1")
    assert {k: v.tobytes() for k, v in joint.items()} == {k: v.tobytes() for k, v in dm.items()} == {k: v.tobytes() for k, v in md.items()}
    assert all(j["changed_cells"][k] == d["changed_cells"][k] + m["changed_cells"][k] for k in raw)
    assert_unchanged(raw, before)


def test_baseline_and_in_range_values_are_borrowed_unchanged_and_extremes_clip():
    spec = owner.build_spec(normalization()); raw = inputs(spec)
    for value in raw.values(): value.setflags(write=False)
    before = {k: v.tobytes() for k, v in raw.items()}
    for arm in owner.ARMS:
        out, count = owner.intervene(raw, spec, arm)
        assert all(out[k] is raw[k] for k in raw)
        assert count["total_changed_cells"] == 0 and count["changed_arrays"] == []
    assert_unchanged(raw, before)
    extremes = inputs(spec); name = owner.MACRO_FIELDS[0]
    fmax = np.finfo(np.float32).max
    set_alias(extremes, name, [[-fmax, 1., fmax], [3., 2., 1.]])
    out, count = owner.intervene(extremes, spec, "MACRO")
    si = owner.INDICES[name][1]
    np.testing.assert_array_equal(out["seq_x"][..., si], [[1., 1., 3.], [3., 2., 1.]])
    assert count["per_field"][name]["seq_x"]["changed"] == 2


@pytest.mark.parametrize("fault,reason", [
    ("missing_name", "EXACT_FIELD_ROUTE"), ("duplicate_name", "FIELD_NAMES"),
    ("missing_alias", "EXACT_ALIAS"), ("wrong_alias", "EXACT_ALIAS"),
    ("binary", "CONTINUOUS_FIELD"), ("categorical", "CONTINUOUS_FIELD"),
    ("zero_scale", "FINITE_FIT_BOUNDS"), ("different_alias_scale", "ALIAS_TRANSFORM"),
    ("rounded_bound", "EXACT_RAW_BOUND_ROUNDTRIP"), ("hash", "NORMALIZATION_HASH"),
])
def test_malformed_normalization_and_identity_fields_fail_closed(fault, reason):
    norm = normalization(); ci, si = owner.INDICES[owner.D1_FIELDS[1]]
    s = norm["surfaces"]["ctx_cont"]
    if fault == "missing_name": s["field_names"][ci] = "not_the_bound_name"
    elif fault == "duplicate_name": s["field_names"][ci] = s["field_names"][0]
    elif fault == "missing_alias": norm["temporal_aliases"].pop(1)
    elif fault == "wrong_alias": norm["temporal_aliases"][1]["signal_index"] += 1
    elif fault == "binary": s["binary_mask"][ci] = 1
    elif fault == "categorical": s["categorical_mask"][ci] = 1
    elif fault == "zero_scale": s["scale"][ci] = 0.
    elif fault == "different_alias_scale":
        t = norm["surfaces"]["signal"]; t["scale"][si] = 27.
        t["train_transformed_min"][si] = float(np.arcsinh((127.-182.)/27.))
        t["train_transformed_max"][si] = float(np.arcsinh((236.-182.)/27.))
    elif fault == "rounded_bound": s["train_transformed_min"][ci] = float(np.nextafter(s["train_transformed_min"][ci], np.inf))
    if fault != "hash": norm = seal(norm)
    else: norm["contract_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match=reason): owner.build_spec(norm)


@pytest.mark.parametrize("fault,reason", [("arm", "ARM"), ("spec", "SPEC_BINDING"),
    ("snapshot", "CURRENT_SNAPSHOT_ALIAS"), ("context", "CURRENT_CONTEXT_ALIAS"),
    ("nonfinite", "ARRAY_LAYOUT"), ("dtype", "ARRAY_LAYOUT")])
def test_invalid_call_rejects_without_mutating_source(fault, reason):
    spec = owner.build_spec(normalization()); raw = inputs(spec); arm = "D1"
    if fault == "arm": arm = "OTHER"
    elif fault == "spec": spec["fields"][0]["raw_bounds"] = [0., 1.]
    elif fault == "snapshot": raw["snap_x"][0, 0] = 2.
    elif fault == "context": raw["ctx_cont"][0, owner.INDICES[owner.D1_FIELDS[0]][0]] = 0.
    elif fault == "nonfinite": raw["seq_h4"][0, 0, 0] = np.nan
    elif fault == "dtype": raw["seq_m15"] = raw["seq_m15"].astype(np.float64)
    before = {k: v.tobytes() for k, v in raw.items()}
    with pytest.raises(RuntimeError, match=reason): owner.intervene(raw, spec, arm)
    assert_unchanged(raw, before)
