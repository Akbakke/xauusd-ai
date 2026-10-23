"""Fixed193 current-snapshot direction reference; no loader, native model or actions.

The caller owns physical parent/clock/target/055 binding, policy/caps and numerical
qualification. These pure projection/math functions grant no data or fit authority.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from gx1.contracts.entry_dynamic_compact_input_influence_v1 import (
    _MTF, _direction_input_coordinates, compile_dynamic_compact_influence_layout,
)
from gx1.contracts.entry_model_native_input_normalization_v1 import (
    apply_surface_normalization, require_input_normalization_contract,
    SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS, MTF_SEMANTIC_CATEGORICAL_DOMAINS,
)
from gx1.contracts.entry_model_native_training_objective_v1 import ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE
from gx1.contracts.gc_sequence_research_v1 import encode_inputs
from gx1.contracts.local_random_access_campaign_v2 import canonical_sha256

SCHEMA_VERSION = "gx1_direction_snapshot_reference_v1"
HORIZONS_MINUTES = (1, 5, 15)
RIDGE_ALPHA = 1.0  # Accepted193 existing sum-SSE convention, never an optimum.
PROFILE = {"name": "compact_mbp_smc_ema_atr_clock_macro_entry_v1",
           "contract_sha256": "2ac4fb4ddfac84f588f65363ac8959544233ace286d9648bfebf60b97153f125"}
_SESSION = "local.ctx_cat.session_id"
_SIGNED_LOCAL_EMA = frozenset(tf + "_ema_stack_aligned_v2" for tf in ("m5", "m15", "h1", "h4", "d1"))


def _require(condition, reason):
    if not condition:
        raise RuntimeError("DIRECTION_SNAPSHOT_REFERENCE_" + reason)


def _bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(
        np.ascontiguousarray(a).view(np.uint32), np.ascontiguousarray(b).view(np.uint32))


def compile_direction_snapshot_layout(normalization_contract):
    """Reuse the compact profile owner through its existing influence-layout adapter."""
    native = compile_dynamic_compact_influence_layout()
    profile = native["profile"]
    _require({k: profile[k] for k in PROFILE} == PROFILE, "PROFILE")
    fields = profile["physical_field_names"]
    norm = require_input_normalization_contract(normalization_contract,
        expected_field_names={"signal": fields["signal"], "ctx_cont": fields["ctx_cont"],
            **{"mtf_" + tf: fields["mtf"] for tf in ("m5", "m15", "h1", "h4", "d1")}},
        expected_ctx_cat_names=fields["ctx_cat"],
        expected_entry_training_phase=ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    coordinates, categorical, families = _direction_input_coordinates(native, norm)
    _require(len(coordinates) == 3317 and sum(c["lane"] == "local" for c in coordinates) == 2622
        and all(sum(c["lane"] == lane for c in coordinates) == 139 for lane in _MTF), "CANONICAL_WIDTH")
    _require(len(categorical) == 1 and categorical[0]["field"] == "session_id"
        and tuple(native["domains"]["ctx_cat"]["enum_domains"][categorical[0]["source_index"]]) == (0, 1, 2, 3), "SESSION_DOMAIN")
    nominal = {}
    for c in coordinates:
        declared = SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS if c["surface"] == "seq_x" else MTF_SEMANTIC_CATEGORICAL_DOMAINS if c["surface"] in _MTF else {}
        if c["field"] in declared:
            _require(tuple(declared[c["field"]]) == (0, 1, 2, 3, 4), "SMC_DOMAIN")
            nominal[c["token"]] = tuple(declared[c["field"]])
    _require(len(nominal) == 6 and {c["lane"] for c in coordinates if c["token"] in nominal} == {"local", *_MTF}, "SIX_NOMINAL_SMC_STATES")
    nominal[_SESSION] = (0, 1, 2, 3)
    names = [c["token"] for c in coordinates] + [_SESSION]
    columns = [name + "=" + str(v) if name in nominal else name
               for name in names for v in (nominal[name] if name in nominal else (None,))]
    _require(len(columns) == len(set(columns)) == 3345, "ENCODED_WIDTH")
    value = {"schema_version": SCHEMA_VERSION, "profile": dict(PROFILE),
        "normalization_contract_sha256": norm["contract_sha256"],
        "canonical_float_coordinates": 3317, "encoded_columns": columns,
        "nominal_domains": nominal, "families": list(families),
        "snapshot_only_no_historical_lags": True, "native_action_authority": False}
    contract = {**value, "contract_sha256": canonical_sha256(value)}
    return {"contract": contract, "native": native, "normalization": norm,
            "coordinates": coordinates, "categorical": categorical,
            "names": names, "nominal_domains": nominal, "columns": columns}


def _require_current_surface(value, surface, *, layout, rows):
    spec = layout["native"]["domains"][surface]
    _require(isinstance(value, np.ndarray) and value.ndim == 2
        and value.shape == (rows, len(spec["field_names"]))
        and value.dtype == (np.int64 if surface == "ctx_cat" else np.float32)
        and np.isfinite(value).all(), "CURRENT_SHAPE_DTYPE_FINITE:" + surface)
    for index in spec["binary_indices"]:
        _require(np.isin(value[:, index], (0, 1)).all(), "BINARY_DOMAIN:" + surface)
    for index, domain in spec["enum_domains"].items():
        _require(np.isin(value[:, index], domain).all(), "ENUM_DOMAIN:" + surface)
    for index, mask in spec["masking_pairs"]:
        observed = value[:, mask]
        _require(np.isin(observed, (0, 1)).all() and (value[observed == 0, index] == 0).all(), "MASKED_STORAGE:" + surface)
    for index, constant in spec["source_constants"].items():
        _require((value[:, index] == constant).all(), "SOURCE_CONSTANT:" + surface)


def project_direction_snapshot(raw_current, *, layout):
    """Project full raw current physical surfaces; optional seq_x is a last-row view."""
    required = {"snap_x", "ctx_cont", "ctx_cat", *_MTF}
    _require(isinstance(raw_current, Mapping) and required <= set(raw_current)
        and set(raw_current) <= required | {"seq_x"}, "CURRENT_KEYS")
    snap = raw_current["snap_x"]
    _require(isinstance(snap, np.ndarray) and snap.ndim == 2 and snap.shape[0] > 0, "CURRENT_ROWS")
    rows = snap.shape[0]
    for surface in required:
        _require_current_surface(raw_current[surface], surface, layout=layout, rows=rows)
    if "seq_x" in raw_current:
        seq = raw_current["seq_x"]
        _require(isinstance(seq, np.ndarray) and seq.dtype == np.float32 and seq.ndim in (2, 3), "SEQ_CURRENT_SHAPE")
        if seq.ndim == 3:
            _require(seq.shape[1] > 0, "EMPTY_HISTORY")
            seq = seq[:, -1, :]
        _require(_bits_equal(seq, snap), "SEQ_LAST_SNAP_NOT_EXACT")
    aliases = layout["native"]["physical_aliases"]
    si, ci = [a["signal_index"] for a in aliases], [a["ctx_cont_index"] for a in aliases]
    _require(_bits_equal(snap[:, si], raw_current["ctx_cont"][:, ci]), "CURRENT_ALIAS_NOT_EXACT")
    normalized = {}
    for surface in ("snap_x", "ctx_cont", *_MTF):
        key = "signal" if surface == "snap_x" else "mtf_" + surface[4:] if surface in _MTF else surface
        normalized[surface] = apply_surface_normalization(raw_current[surface], layout["normalization"]["surfaces"][key])
    columns = []
    for c in layout["coordinates"]:
        surface = "snap_x" if c["surface"] == "seq_x" else c["surface"]
        index = c["source_index"]
        raw = raw_current[surface][:, index]
        #180 source-domain corrections remain strict even where055 uses asinh.
        if c["domain_kind"] == "source_binary":
            _require(np.isin(raw, (0, 1)).all(), "ACTIVE_BINARY_DOMAIN:" + c["token"])
        elif c["domain_kind"] == "source_enum" and c["field"].removeprefix("ctx_cont.") in _SIGNED_LOCAL_EMA:
            _require(np.isin(raw, (-1, 0, 1)).all(), "SIGNED_EMA_DOMAIN:" + c["token"])
        current = normalized[surface][:, index]
        if c["token"] in layout["nominal_domains"]:
            _require(_bits_equal(current, raw), "NOMINAL_NORMALIZATION_IDENTITY:" + c["token"])
        columns.append(current)
    columns.append(raw_current["ctx_cat"][:, layout["categorical"][0]["source_index"]])
    matrix, names = encode_inputs(np.column_stack(columns), layout["names"], layout["nominal_domains"])
    _require(names == layout["columns"] and matrix.shape == (rows, 3345)
        and np.isfinite(matrix).all(), "PROJECTED_COLUMNS_OR_FINITE")
    return matrix.astype(np.float32)


def _ridge_owners():
    # Import only reviewed numerical primitives, never legacy run/selector/wrapper.
    from gx1.scripts.research_entry_direction_walkforward_v1 import (
        GRAM_CHUNK_ROWS, _chunked_gram, _chunked_xty, _chunked_matvec,
    )
    return GRAM_CHUNK_ROWS, _chunked_gram, _chunked_xty, _chunked_matvec


def _require_matrix(matrix):
    _require(isinstance(matrix, np.ndarray) and matrix.ndim == 2 and all(matrix.shape)
        and matrix.dtype == np.float32, "RIDGE_MATRIX")
    chunk, _, _, _ = _ridge_owners()
    for start in range(0, len(matrix), chunk):
        _require(np.isfinite(matrix[start:start+chunk]).all(), "RIDGE_NONFINITE_MATRIX")


def fit_direction_snapshot_ridge(matrix, targets, observed, *, split):
    """Fixed alpha1 sum-SSE, unpenalized intercept; own observed TRAIN rows per h."""
    _require(split == "train", "TRAIN_ONLY_FIT")
    _require_matrix(matrix)
    _require(isinstance(targets, np.ndarray) and targets.shape == (len(matrix), 3)
        and targets.dtype == np.float64 and isinstance(observed, np.ndarray)
        and observed.shape == targets.shape and observed.dtype == np.bool_, "TARGET_MASK_SHAPE_DTYPE")
    chunk, gram_owner, xty_owner, _ = _ridge_owners()
    fits = {}
    for column, horizon in enumerate(HORIZONS_MINUTES):
        rows = np.flatnonzero(observed[:, column])
        _require(len(rows) > 0 and np.isfinite(targets[rows, column]).all(), "OBSERVED_TRAIN_TARGET:" + str(horizon))
        mean = np.zeros(matrix.shape[1], dtype=np.float64)
        for start in range(0, len(rows), chunk):
            mean += matrix[rows[start:start+chunk]].sum(axis=0, dtype=np.float64)
        mean /= len(rows)
        target_mean = float(np.mean(targets[rows, column], dtype=np.float64))
        scale = np.ones(matrix.shape[1], dtype=np.float64)  #055 already owns units.
        gram = gram_owner(matrix, rows, mean=mean, scale=scale)
        gram.flat[::len(mean)+1] += RIDGE_ALPHA
        centered_y = targets[:, column].astype(np.float64) - target_mean
        rhs = xty_owner(matrix, centered_y, rows, mean=mean, scale=scale)
        _require(np.isfinite(gram).all() and np.isfinite(rhs).all(), "RIDGE_NONFINITE_SUFFICIENT_STATISTICS")
        beta = np.linalg.solve(gram, rhs)
        residual = gram @ beta - rhs
        rhs_norm = float(np.linalg.norm(rhs))
        _require(np.isfinite(beta).all() and np.isfinite(residual).all(), "RIDGE_NONFINITE_SOLVE")
        fits[horizon] = {"beta": beta, "feature_mean": mean, "target_mean_bps": target_mean,
            "intercept_bps": target_mean-float(mean @ beta), "observed_train_rows": int(len(rows)),
            "solver_residual_l2": float(np.linalg.norm(residual)), "rhs_l2": rhs_norm,
            "solver_relative_residual": float(np.linalg.norm(residual))/rhs_norm if rhs_norm else None}
        del gram, rhs, centered_y
    return {"schema_version": SCHEMA_VERSION + "_ridge", "horizons_minutes": HORIZONS_MINUTES,
        "alpha": RIDGE_ALPHA, "objective": "sum_SSE_plus_alpha_beta_squared_free_intercept",
        "feature_width": int(matrix.shape[1]), "feature_scale": "ones_after_frozen055",
        "fit_partition": "train", "fits": fits}


def predict_direction_snapshot_ridge(matrix, state):
    """Frozen TRAIN centering/intercept; no evaluation target or extra standardizer."""
    _require_matrix(matrix)
    _require(isinstance(state, Mapping) and state.get("schema_version") == SCHEMA_VERSION + "_ridge"
        and state.get("alpha") == RIDGE_ALPHA and state.get("fit_partition") == "train"
        and tuple(state.get("horizons_minutes", ())) == HORIZONS_MINUTES
        and state.get("feature_width") == matrix.shape[1], "RIDGE_STATE")
    _, _, _, matvec_owner = _ridge_owners()
    predictions = []
    for horizon in HORIZONS_MINUTES:
        fit = state["fits"][horizon]
        beta, mean = fit["beta"], fit["feature_mean"]
        _require(beta.shape == mean.shape == (matrix.shape[1],) and np.isfinite(beta).all()
            and np.isfinite(mean).all() and np.isfinite(fit["target_mean_bps"]), "RIDGE_FIT_STATE")
        predictions.append(matvec_owner(matrix, beta, mean=mean, scale=np.ones_like(mean)) + fit["target_mean_bps"])
    result = np.column_stack(predictions)
    _require(np.isfinite(result).all(), "RIDGE_NONFINITE_PREDICTIONS")
    return result
