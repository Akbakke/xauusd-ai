"""Synthetic193 projection/ridge qualification; no market payload or native model."""
import numpy as np
import pytest

from gx1.contracts import entry_direction_snapshot_reference_v1 as reference
from gx1.contracts.entry_model_native_input_normalization_v1 import CONTINUOUS_TRANSFORM


@pytest.fixture
def snapshot_case(monkeypatch):
    # Genuine runtime source layout, synthetic stats only. The strict055 lineage
    # validator is isolated here; no normalization fit or genuine artifact claimed.
    native = reference.compile_dynamic_compact_influence_layout()
    surfaces = {}
    for surface in ("snap_x", "ctx_cont", *reference._MTF):
        key = "signal" if surface == "snap_x" else "mtf_" + surface[4:] if surface in reference._MTF else surface
        domain = native["domains"][surface]
        names = domain["field_names"]
        categorical = reference.SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS if surface == "snap_x" else reference.MTF_SEMANTIC_CATEGORICAL_DOMAINS if surface in reference._MTF else {}
        surfaces[key] = {"field_names": list(names), "center": [0.]*len(names), "scale": [2.]*len(names),
            "binary_mask": [int(i in domain["binary_indices"]) for i in range(len(names))],
            "categorical_mask": [int(name in categorical) for name in names],
            "categorical_domains": categorical, "continuous_transform": CONTINUOUS_TRANSFORM,
            "scale_source": ["raw_iqr"]*len(names)}
    norm = {"contract_sha256": "a"*64, "surfaces": surfaces}
    monkeypatch.setattr(reference, "require_input_normalization_contract", lambda value, **kwargs: value)
    layout = reference.compile_direction_snapshot_layout(norm)
    raw = {surface: np.zeros((3, len(native["domains"][surface]["field_names"])), dtype=np.float32)
           for surface in ("snap_x", "ctx_cont", *reference._MTF)}
    raw["ctx_cat"] = np.array([[0], [1], [3]], dtype=np.int64)
    for c in layout["coordinates"]:
        if c["token"] in layout["nominal_domains"]:
            surface = "snap_x" if c["surface"] == "seq_x" else c["surface"]
            raw[surface][:, c["source_index"]] = (0, 2, 4)
    return layout, raw


def test_canonical_layout_and_all_six_nominal_domains_expand_without_alias_duplicates(snapshot_case):
    layout, raw = snapshot_case
    out = reference.project_direction_snapshot(raw, layout=layout)
    assert out.dtype == np.float32 and out.shape == (3, 3345)
    assert len(layout["coordinates"]) == 3317 and len(set(layout["columns"])) == 3345
    assert len(layout["nominal_domains"]) == 7
    for token, domain in layout["nominal_domains"].items():
        indices = [layout["columns"].index(token + "=" + str(value)) for value in domain]
        expected = np.eye(len(domain))[[0, 2, 4] if len(domain) == 5 else [0, 1, 3]]
        np.testing.assert_array_equal(out[:, indices], expected)
        assert token not in layout["columns"]
    assert sum(c["lane"] == "local" for c in layout["coordinates"]) == 2622
    assert all(sum(c["lane"] == tf for c in layout["coordinates"]) == 139 for tf in reference._MTF)


def test_signed_enum_uses_frozen_asinh_and_optional_history_is_only_current_alias(snapshot_case):
    layout, raw = snapshot_case
    c = next(c for c in layout["coordinates"] if c["field"] == "ctx_cont.h1_ema_stack_aligned_v2")
    for owner, index in c["current_participants"]:
        owner = "snap_x" if owner == "seq_x" else owner
        raw[owner][:, index] = (-1, 0, 1)
    history = np.stack((np.full_like(raw["snap_x"], np.nan), raw["snap_x"]), axis=1)
    raw["seq_x"] = history  # Earlier history is neither copied, flattened nor validated.
    out = reference.project_direction_snapshot(raw, layout=layout)
    np.testing.assert_allclose(out[:, layout["columns"].index(c["token"])], np.arcsinh(np.array([-1, 0, 1])/2), rtol=1e-7)
    assert c["token"] + "=-1" not in layout["columns"]
    raw["seq_x"][0, -1, c["source_index"]] = 0
    with pytest.raises(RuntimeError, match="SEQ_LAST_SNAP_NOT_EXACT"):
        reference.project_direction_snapshot(raw, layout=layout)


@pytest.mark.parametrize("fault", ["nominal", "session", "alias", "alias_signed_zero", "mask", "signed_enum", "binary"])
def test_projection_rejects_invalid_source_domains_masks_or_aliases(snapshot_case, fault):
    layout, raw = snapshot_case
    if fault == "nominal":
        c = next(c for c in layout["coordinates"] if c["token"] in layout["nominal_domains"])
        raw["snap_x"][0, c["source_index"]] = 5
    elif fault == "session": raw["ctx_cat"][0, 0] = 4
    elif fault in ("alias", "alias_signed_zero"):
        a = layout["native"]["physical_aliases"][0]
        raw["ctx_cont"][0, a["ctx_cont_index"]] = 1. if fault == "alias" else -0.
    elif fault == "mask":
        value, mask = layout["native"]["domains"]["snap_x"]["masking_pairs"][0]
        raw["snap_x"][0, value] = 1.  # Source mask is still genuinely0.
    elif fault == "signed_enum":
        c = next(c for c in layout["coordinates"] if c["field"] == "ctx_cont.h1_ema_stack_aligned_v2")
        for owner, index in c["current_participants"]:
            raw["snap_x" if owner == "seq_x" else owner][0, index] = 2.
    else:
        c = next(c for c in layout["coordinates"] if c["field"] == "smc_choch")
        raw["snap_x"][0, c["source_index"]] = .5
    with pytest.raises(RuntimeError, match="DIRECTION_SNAPSHOT_REFERENCE_|ENTRY_INPUT_NORMALIZATION_"):
        reference.project_direction_snapshot(raw, layout=layout)


def test_missing_storage_stays_zero_and_existing_observation_masks_remain(snapshot_case):
    layout, raw = snapshot_case
    domain = layout["native"]["domains"]["snap_x"]
    active = set(layout["native"]["profile"]["signal_indices"])
    value, mask = next((v, m) for v, m in domain["masking_pairs"] if v in active and m in active)
    raw["snap_x"][1, mask] = 1.
    for alias in layout["native"]["physical_aliases"]:
        raw["ctx_cont"][:, alias["ctx_cont_index"]] = raw["snap_x"][:, alias["signal_index"]]
    out = reference.project_direction_snapshot(raw, layout=layout)
    value_token = next(c["token"] for c in layout["coordinates"] if c["surface"] == "seq_x" and c["source_index"] == value)
    mask_token = next(c["token"] for c in layout["coordinates"] if c["surface"] == "seq_x" and c["source_index"] == mask)
    np.testing.assert_array_equal(out[:, layout["columns"].index(value_token)], [0, 0, 0])
    np.testing.assert_array_equal(out[:, layout["columns"].index(mask_token)], [0, 1, 0])


@pytest.fixture
def small_chunks(monkeypatch):
    from gx1.scripts import research_entry_direction_walkforward_v1 as owners
    monkeypatch.setattr(owners, "GRAM_CHUNK_ROWS", 2)


def test_chunked_fixed_ridge_equals_existing_dense_each_horizon_subset_and_free_intercept(small_chunks):
    from gx1.scripts.research_m5_intraday_recheck_v1 import ridge_fit
    x = np.array([[0, 0, 1], [1, 0, 1], [2, 1, 1], [3, 0, 1], [5, 1, 1]], dtype=np.float32)
    y = np.array([[2, 10, 7], [4, np.nan, 6], [8, 7, np.nan], [10, 5, 5], [16, 3, 4]], dtype=np.float64)
    observed = np.isfinite(y)
    before = x.copy(), y.copy(), observed.copy()
    state = reference.fit_direction_snapshot_ridge(x, y, observed, split="train")
    predictions = reference.predict_direction_snapshot_ridge(x, state)
    assert predictions.shape == (5, 3)
    for column, h in enumerate(reference.HORIZONS_MINUTES):
        rows = np.flatnonzero(observed[:, column])
        dense_x = x[rows].astype(np.float64)
        beta, intercept = ridge_fit(dense_x, y[rows, column], 1.)
        fit = state["fits"][h]
        np.testing.assert_allclose(fit["beta"], beta, rtol=1e-12, atol=1e-12)
        assert fit["intercept_bps"] == pytest.approx(intercept, rel=1e-12, abs=1e-12)
        np.testing.assert_allclose(predictions[:, column], x @ beta + intercept, rtol=1e-12, atol=1e-12)
        assert fit["observed_train_rows"] == len(rows)
        assert fit["target_mean_bps"] == pytest.approx(y[rows, column].mean())
        np.testing.assert_allclose(fit["feature_mean"], dense_x.mean(axis=0))
        assert fit["solver_residual_l2"] < 1e-11  # Declared tiny fixture equivalence, no empirical gate.
    for a, b in zip((x, y, observed), before): np.testing.assert_array_equal(a, b)
    assert state["alpha"] == 1. and state["feature_scale"] == "ones_after_frozen055"


def test_free_intercept_and_TRAIN_constant_do_not_fit_evaluation_targets(small_chunks):
    train = np.ones((4, 2), dtype=np.float32)
    targets = np.array([[7, 3, -2]]*4, dtype=np.float64)
    mask = np.ones_like(targets, dtype=bool)
    state = reference.fit_direction_snapshot_ridge(train, targets, mask, split="train")
    control = np.array([[99, -99], [0, 1000]], dtype=np.float32)
    prediction = reference.predict_direction_snapshot_ridge(control, state)
    np.testing.assert_array_equal(prediction, np.array([[7, 3, -2]]*2))
    for h in reference.HORIZONS_MINUTES:
        np.testing.assert_array_equal(state["fits"][h]["beta"], [0, 0])
        assert state["fits"][h]["solver_relative_residual"] is None
    with pytest.raises(RuntimeError, match="TRAIN_ONLY_FIT"):
        reference.fit_direction_snapshot_ridge(control, np.ones((2, 3)), np.ones((2, 3), bool), split="control")


@pytest.mark.parametrize("fault", ["no_h1", "observed_nan", "mask_dtype", "target_f32", "feature_nan"])
def test_ridge_fails_closed_without_union_mask_or_imputation(small_chunks, fault):
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    y, valid = np.ones((4, 3)), np.ones((4, 3), dtype=bool)
    if fault == "no_h1": valid[:, 0] = False
    elif fault == "observed_nan": y[0, 0] = np.nan
    elif fault == "mask_dtype": valid = valid.astype(np.int64)
    elif fault == "target_f32": y = y.astype(np.float32)
    else: x[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="DIRECTION_SNAPSHOT_REFERENCE_"):
        reference.fit_direction_snapshot_ridge(x, y, valid, split="train")
