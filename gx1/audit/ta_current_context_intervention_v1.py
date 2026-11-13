"""Fixed private raw-context clamps; no model, file IO, fit or run authority.

Unchanged arrays are borrowed. Only arrays with clipped cells are copied. The
current-context interval also applies to every historical signal alias: this
is an input-path intervention, not a physically reconstructed market history.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

import numpy as np

ARMS = ("BASELINE", "D1", "MACRO", "JOINT")
D1_FIELDS = ("d1_ema_stack_aligned_v2", "d1_trend_state_age_bars_v2")
MACRO_FIELDS = ("EUR_USD_rv_20D", "XPT_USD_rv_20D", "XCU_USD_rv_20D", "XAG_USD_rv_20D")
FIELDS = D1_FIELDS + MACRO_FIELDS
# Exact current TA317/context132 profile; never infer a different field route.
INDICES = {
    D1_FIELDS[0]: (68, 253), D1_FIELDS[1]: (63, 248),
    MACRO_FIELDS[0]: (82, 267), MACRO_FIELDS[1]: (118, 303),
    MACRO_FIELDS[2]: (110, 295), MACRO_FIELDS[3]: (102, 287),
}
MTF_KEYS = tuple("seq_" + tf for tf in ("m5", "m15", "h1", "h4", "d1"))
FEATURE_KEYS = frozenset(("seq_x", "snap_x", "ctx_cont", "ctx_cat", *MTF_KEYS))
SCHEMA = "gx1_ta_current_context_intervention_v1"
TRANSFORM = "asinh_affine_invertible_non_saturating"


def _need(ok, reason):
    if not bool(ok):
        raise RuntimeError("TA_CURRENT_CONTEXT_" + reason)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _same(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def _bounds(surface, index):
    _need(surface["continuous_transform"] == TRANSFORM
          and "zero_support_profile_sha256" not in surface, "TRANSFORM")
    _need(surface["binary_mask"][index] == surface["categorical_mask"][index] == 0,
          "CONTINUOUS_FIELD_REQUIRED")
    center, scale = float(surface["center"][index]), float(surface["scale"][index])
    z = np.asarray([surface["train_transformed_min"][index],
                    surface["train_transformed_max"][index]], dtype=np.float64)
    _need(np.isfinite([center, scale]).all() and scale > 0 and np.isfinite(z).all()
          and z[0] <= z[1], "FINITE_FIT_BOUNDS")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        _need(float(np.float32(center)) == center and float(np.float32(scale)) == scale,
              "FP32_FIT_STATS")
        # Keep the saved FP64 extrema until inversion. The generic inverse
        # accepts FP32 normalized inputs and would first round these extrema.
        raw64 = center + scale * np.sinh(z)
        raw32 = raw64.astype(np.float32)
        replay = np.arcsinh((raw32.astype(np.float64) - center) / scale)
    _need(np.isfinite(raw32).all() and np.array_equal(replay, z), "EXACT_RAW_BOUND_ROUNDTRIP")
    return center, scale, z.tolist(), raw32.astype(np.float64).tolist()


def build_spec(normalization):
    """Bind the six named aliases to frozen CURRENT bounds, without fitting.

    The caller binds the actual normalization artifact. Here its content hash,
    named columns, alias ownership and recoverable FP32 raw limits are checked.
    Binary/categorical identity fields are deliberately outside this scope.
    """
    try:
        _need(isinstance(normalization, Mapping)
              and normalization["contract_sha256"] == _sha({
                  k: v for k, v in normalization.items() if k != "contract_sha256"}),
              "NORMALIZATION_HASH")
        surfaces = normalization["surfaces"]
        for surface, width in (("signal", 317), ("ctx_cont", 132)):
            s = surfaces[surface]
            names = s["field_names"]
            _need(len(names) == len(set(names)) == width, "FIELD_NAMES:" + surface)
            _need(all(len(s[k]) == width for k in ("center", "scale", "binary_mask",
                  "categorical_mask", "train_transformed_min", "train_transformed_max")),
                  "STATS_WIDTH:" + surface)
        fields = []
        for name in FIELDS:
            ci, si = INDICES[name]
            group = "D1" if name in D1_FIELDS else "MACRO"
            owner = "trend_ema_encoder" if group == "D1" else "session_regime_encoder"
            _need(surfaces["ctx_cont"]["field_names"][ci] == name
                  and surfaces["signal"]["field_names"][si] == "ctx_cont." + name,
                  "EXACT_FIELD_ROUTE:" + name)
            aliases = [a for a in normalization["temporal_aliases"]
                       if a["ctx_cont_field"] == name or a["signal_field"] == "ctx_cont." + name
                       or a["ctx_cont_index"] == ci or a["signal_index"] == si]
            _need(aliases == [dict(ctx_cont_field=name, ctx_cont_index=ci,
                  signal_field="ctx_cont." + name, signal_index=si, specialist=owner)],
                  "EXACT_ALIAS:" + name)
            center, scale, current_z, current_raw = _bounds(surfaces["ctx_cont"], ci)
            signal_center, signal_scale, signal_z, signal_raw = _bounds(surfaces["signal"], si)
            _need(center == signal_center and scale == signal_scale, "ALIAS_TRANSFORM:" + name)
            fields.append(dict(name=name, group=group, specialist=owner,
                ctx_cont_index=ci, signal_index=si, center=center, scale=scale,
                raw_bounds=current_raw, signal_raw_bounds=signal_raw,
                current_transformed_bounds=current_z, signal_transformed_bounds=signal_z,
                exact_fp32_raw_bound_roundtrip=True))
        spec = dict(schema_version=SCHEMA, normalization_sha256=normalization["contract_sha256"],
            arms=list(ARMS), interval_owner="ctx_cont", fields=fields,
            application="all_seq_x_positions_plus_snap_x_and_ctx_cont",
            mtf_history_changed=False, physical_market_counterfactual=False)
        return dict(spec, spec_sha256=_sha(spec))
    except (KeyError, TypeError, ValueError, IndexError, FloatingPointError, OverflowError) as exc:
        raise RuntimeError("TA_CURRENT_CONTEXT_MALFORMED_NORMALIZATION") from exc


def _require_raw(raw, fields):
    _need(isinstance(raw, Mapping) and set(raw) == FEATURE_KEYS, "FEATURE_KEYS")
    x = raw["seq_x"]
    _need(isinstance(x, np.ndarray) and x.ndim == 3 and 0 < len(x) <= 16
          and x.shape[1] > 0 and x.shape[2] == 317, "SIGNAL_SHAPE")
    b = len(x)
    for name in FEATURE_KEYS:
        a = raw[name]
        _need(isinstance(a, np.ndarray), "ARRAY:" + name)
        if name in MTF_KEYS:
            shape_ok = a.ndim == 3 and a.shape[0] == b and a.shape[1] > 0 and a.shape[2] == 255
        else:
            shape_ok = a.shape == {"seq_x": x.shape, "snap_x": (b, 317),
                                  "ctx_cont": (b, 132), "ctx_cat": (b, 1)}[name]
        dtype = np.int64 if name == "ctx_cat" else np.float32
        _need(shape_ok and a.dtype == dtype and np.isfinite(a).all(), "ARRAY_LAYOUT:" + name)
    _need(np.isin(raw["ctx_cat"], (0, 1, 2, 3)).all(), "SESSION_DOMAIN")
    _need(_same(x[:, -1], raw["snap_x"]), "CURRENT_SNAPSHOT_ALIAS")
    for field in fields:
        _need(_same(raw["snap_x"][:, field["signal_index"]],
                    raw["ctx_cont"][:, field["ctx_cont_index"]]), "CURRENT_CONTEXT_ALIAS")


def intervene(raw, spec, arm):
    """Return (features, JSON counts), retaining original bytes outside clamps.

    Both signal surfaces use the CURRENT context interval, including historical
    sequence positions. MTF histories and session are borrowed unchanged.
    """
    _need(isinstance(spec, Mapping) and spec.get("schema_version") == SCHEMA
          and spec.get("arms") == list(ARMS) and spec.get("spec_sha256") == _sha({
              k: v for k, v in spec.items() if k != "spec_sha256"}), "SPEC_BINDING")
    _need(type(arm) is str and arm in ARMS, "ARM")
    fields = spec["fields"]
    _need([f["name"] for f in fields] == list(FIELDS)
          and all((f["ctx_cont_index"], f["signal_index"]) == INDICES[f["name"]]
                  for f in fields), "SPEC_ROUTES")
    _require_raw(raw, fields)
    selected = [f for f in fields if arm == "JOINT" or arm == f["group"]]
    out = dict(raw)
    changes = dict(arm=arm, selected_fields=[f["name"] for f in selected],
                   changed_arrays=[], changed_cells={k: 0 for k in sorted(FEATURE_KEYS)}, per_field={})
    for field in selected:
        lo, hi = (np.float32(v) for v in field["raw_bounds"])
        _need(np.isfinite([lo, hi]).all() and lo <= hi, "SPEC_BOUNDS")
        counts = {}
        for key, index in (("seq_x", field["signal_index"]), ("snap_x", field["signal_index"]),
                           ("ctx_cont", field["ctx_cont_index"])):
            before = raw[key][..., index]
            below, above = before < lo, before > hi
            n_below, n_above = int(np.count_nonzero(below)), int(np.count_nonzero(above))
            if n_below + n_above:
                if out[key] is raw[key]:
                    out[key] = raw[key].copy()
                    changes["changed_arrays"].append(key)
                column = out[key][..., index]
                column[below], column[above] = lo, hi
            counts[key] = dict(below=n_below, above=n_above, changed=n_below + n_above)
            changes["changed_cells"][key] += n_below + n_above
        changes["per_field"][field["name"]] = counts
    _require_raw(out, fields)
    changes["total_changed_cells"] = sum(changes["changed_cells"].values())
    return out, changes
