"""Eight fabricated rows exercise explicit base/GC units delegation, never a model.

The patched expected profile is a test seam only. A separate check proves that
these small base populations cannot qualify under the actual 29,849-row profile.
"""
from copy import deepcopy
import hashlib

import numpy as np
import pytest

from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.contracts import entry_native_direction_prefix_units_v1 as units
from gx1.features import gc_native_context_v1 as gc
from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile
from gx1.features.htf_features import MULTI_TF_SHIFT
from gx1.models.entry_v10 import entry_v10_input_normalization as native

ROWS = 8
FIRST = "2024-09-30T23:00:00+00:00"
LAST = "2024-09-30T23:35:00+00:00"


@pytest.fixture
def synthetic_profile(monkeypatch):
    profile = units.october_prefix_units_profile()
    starts = np.datetime64("2024-09-30T23:00", "ns").astype(np.int64) + np.arange(ROWS) * 300_000_000_000
    profile["fit_prefix"]["rows"] = ROWS
    for name, axis in (("parent_rows", np.arange(ROWS)), ("bar_start_ns", starts),
                       ("decision_ns", starts + 300_000_000_000)):
        profile["fit_prefix"][name + "_sha256"] = hashlib.sha256(np.asarray(axis, dtype="<i8").tobytes()).hexdigest()
    profile["contract_sha256"] = units._sha({k: v for k, v in profile.items() if k != "contract_sha256"})
    monkeypatch.setattr(units, "october_prefix_units_profile", lambda: deepcopy(profile))
    return profile


def _base_kwargs(profile):
    physical = compile_compact_entry_execution_profile()["physical_field_names"]
    surfaces, raw_by_surface = {}, {}
    for name in norm.EXPECTED_SURFACES:
        fields = list(physical[name] if name in ("signal", "ctx_cont") else physical["mtf"])
        raw = np.ones((ROWS, len(fields)), np.float32)
        columns = units.prefix_units_surface_columns(profile, surface=name, field_names=fields)
        for index, entry in columns.items():
            raw[:, [index, entry["mask_index"]]] = 0.
        raw_by_surface[name] = raw
        domains = (norm.SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS if name == "signal" else
                   norm.CTX_CONT_SEMANTIC_CATEGORICAL_DOMAINS if name == "ctx_cont" else
                   norm.MTF_SEMANTIC_CATEGORICAL_DOMAINS)
        surfaces[name] = norm.fit_surface_normalization(raw, surface=name, field_names=fields,
            semantic_categorical_domains=domains, allow_constant_train_fields=True,
            unobserved_units_profile=profile)
    aliases = native._derive_temporal_aliases(physical["signal"])
    surfaces["ctx_cont"] = norm.share_temporal_alias_stats_from_signal(
        surfaces["ctx_cont"], surfaces["signal"], temporal_aliases=aliases,
        ctx_cont_values=raw_by_surface["ctx_cont"], unobserved_units_profile=profile)
    windows = {tf: {"left_index_inclusive": 0, "right_index_exclusive": ROWS,
                   "selected_unique_row_count": ROWS, "selected_row_indices_sha256": "6" * 64,
                   "selected_row_values_sha256": "7" * 64, "time_min_utc": FIRST, "time_max_utc": LAST}
               for tf in norm.EXPECTED_TFS}
    lineage = {"dataset_run_id": "SYNTHETIC_EIGHT_ROW_UNITS",
        "train_parquet_path": "/synthetic/train.parquet", "train_parquet_sha256": "1" * 64,
        "train_manifest_path": "/synthetic/MANIFEST.json", "train_manifest_sha256": "2" * 64,
        "m5_prebuilt_path": "/synthetic/m5.parquet", "m5_prebuilt_sha256": "3" * 64,
        "mtf_cache_manifest_path": "/synthetic/mtf.json", "mtf_cache_manifest_sha256": "4" * 64,
        "mtf_builder_version": "synthetic", "mtf_feature_names_sha256": "5" * 64,
        "train_row_count": ROWS, "entry_train_decision_row_count": ROWS, "exit_train_decision_row_count": 0,
        "local_fit_row_count": ROWS, "context_fit_row_count": ROWS, "val_fit_row_count": 0, "test_fit_row_count": 0,
        "train_time_min_utc": FIRST, "train_time_max_utc": LAST,
        "per_tf_seq_lens": {tf: 1 for tf in norm.EXPECTED_TFS},
        "per_tf_shift_seconds": {tf: int(MULTI_TF_SHIFT[tf].total_seconds()) for tf in norm.EXPECTED_TFS},
        "per_tf_fit_windows": windows}
    cats = np.column_stack([np.arange(ROWS) % len(domain) for domain in norm.CTX_CAT_DOMAINS.values()])
    return dict(fit_start_utc=FIRST, fit_end_utc=LAST, surfaces=surfaces,
                ctx_cat=norm.fit_ctx_cat_contract(cats, field_names=list(norm.CTX_CAT_DOMAINS)),
                lineage=lineage, temporal_aliases=aliases,
                entry_training_phase=norm.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)


def _require_base(contract, profile=None):
    return norm.require_input_normalization_contract(contract,
        expected_field_names={name: item["field_names"] for name, item in contract["surfaces"].items()},
        expected_ctx_cat_names=list(norm.CTX_CAT_DOMAINS), unobserved_units_profile=profile)


def test_base_build_and_require_need_explicit_units_opt_in(synthetic_profile):
    kwargs = _base_kwargs(synthetic_profile)
    original = deepcopy(kwargs)
    with pytest.raises(RuntimeError, match="OPT_IN_REQUIRED"):
        norm.build_input_normalization_contract(**kwargs)
    contract = norm.build_input_normalization_contract(**kwargs, unobserved_units_profile=synthetic_profile)
    assert _require_base(contract, synthetic_profile) == contract
    with pytest.raises(RuntimeError, match="OPT_IN_REQUIRED"):
        _require_base(contract)
    assert contract["surfaces"] == kwargs["surfaces"]
    assert kwargs == original
    assert contract["contract_sha256"] == norm._canonical_sha256(
        {k: v for k, v in contract.items() if k != "contract_sha256"})


def test_actual_locked_profile_rejects_small_base_population():
    profile = units.october_prefix_units_profile()
    assert profile["fit_prefix"]["rows"] == 29849
    with pytest.raises(RuntimeError, match="PREFIX_LINEAGE"):
        norm.build_input_normalization_contract(**_base_kwargs(profile), unobserved_units_profile=profile)


@pytest.mark.parametrize("fault", ["parents", "exit", "joint_phase", "decision_at_cutoff"])
def test_base_units_lineage_is_checked_on_build_and_revalidation(synthetic_profile, fault):
    kwargs = _base_kwargs(synthetic_profile)
    good = norm.build_input_normalization_contract(**kwargs, unobserved_units_profile=synthetic_profile)
    if fault == "parents":
        for key in ("train_row_count", "entry_train_decision_row_count", "context_fit_row_count"):
            kwargs["lineage"][key] -= 1
    elif fault == "exit":
        kwargs["lineage"]["entry_train_decision_row_count"] -= 1
        kwargs["lineage"]["exit_train_decision_row_count"] = 1
    elif fault == "joint_phase":
        kwargs["lineage"]["entry_train_decision_row_count"] -= 1
        kwargs["lineage"]["exit_train_decision_row_count"] = 1
        kwargs["entry_training_phase"] = norm.JOINT_FITTED_Q_TRAINING_PHASE
    else:
        kwargs["fit_end_utc"] = kwargs["lineage"]["train_time_max_utc"] = "2024-09-30T23:55:00+00:00"
    with pytest.raises(RuntimeError):
        norm.build_input_normalization_contract(**kwargs, unobserved_units_profile=synthetic_profile)
    bad = deepcopy(good)
    bad["lineage"] = kwargs["lineage"]
    bad["fit_end_utc"] = kwargs["fit_end_utc"]
    bad["fit_population"] = norm.normalization_fit_population_for_phase(kwargs["entry_training_phase"])
    bad["contract_sha256"] = norm._canonical_sha256({k: v for k, v in bad.items() if k != "contract_sha256"})
    with pytest.raises(RuntimeError):
        _require_base(bad, synthetic_profile)


def _gc_case(profile, *, missing=True):
    layout = history.gc_mtf_history_layout()
    cutoff = profile["fit_prefix"]["cutoff_ns"]
    surfaces, selected = {}, {}
    for tf in history.TIMEFRAMES:
        names = layout["field_names"][tf]
        values = np.broadcast_to(np.arange(1, ROWS + 1, dtype=np.float32)[:, None], (ROWS, len(names))).copy()
        for name in gc.mtf_history_binary_fields(names):
            values[:, names.index(name)] = np.arange(ROWS) % 2
        for _index, mask in gc.masking_pairs(names):
            values[:, mask] = 1.
        if missing and tf == "D1":
            columns = units.prefix_units_surface_columns(profile, surface="gc_mtf_d1", field_names=names)
            for index, entry in columns.items():
                values[:, [index, entry["mask_index"]]] = 0.
        selected[tf] = np.arange(ROWS, dtype=np.int64)
        step = int(MULTI_TF_SHIFT[tf].value)
        surfaces[tf] = {"native_row_indices": selected[tf].copy(),
            "available_ns": cutoff - (ROWS + 3) * step + selected[tf] * step,
            "features": values, "field_names": list(names)}
    return dict(surfaces=surfaces, train_row_indices_by_tf=selected, layout=layout,
                train_start_ns=cutoff - 3_600_000_000_000, train_end_ns=cutoff)


def test_gc_fit_and_all_wrapper_delegates_require_explicit_units(synthetic_profile):
    kwargs = _gc_case(synthetic_profile)
    original = {tf: item["features"].copy() for tf, item in kwargs["surfaces"].items()}
    with pytest.raises(RuntimeError, match="NO_PHYSICAL_TRAIN_OBSERVATIONS"):
        history.fit_gc_mtf_history_normalization(**kwargs)
    fitted = history.fit_gc_mtf_history_normalization(**kwargs, unobserved_units_profile=synthetic_profile)
    wrapper = {"layout": kwargs["layout"], "normalization": fitted}
    calls = [lambda **kw: history.require_gc_mtf_history_normalization(fitted, layout=kwargs["layout"], **kw),
             lambda **kw: history.require_gc_mtf_history(wrapper, **kw),
             lambda **kw: history.gc_mtf_model_layout(wrapper, **kw),
             lambda **kw: history.gc_mtf_normalization_surfaces(wrapper, **kw)]
    for call in calls:
        with pytest.raises(RuntimeError, match="OPT_IN_REQUIRED"):
            call()
        call(unobserved_units_profile=synthetic_profile)
    assert history.gc_mtf_model_layout(wrapper, unobserved_units_profile=synthetic_profile) == kwargs["layout"]
    assert history.gc_mtf_normalization_surfaces(wrapper, unobserved_units_profile=synthetic_profile) == fitted["surfaces"]
    for tf, item in kwargs["surfaces"].items():
        np.testing.assert_array_equal(item["features"], original[tf])
        if tf != "D1":
            legacy = norm.fit_surface_normalization(item["features"], surface="gc_mtf_" + tf.lower(),
                field_names=item["field_names"], row_count=ROWS,
                semantic_categorical_domains=kwargs["layout"]["categorical_domains"][tf])
            assert legacy == fitted["surfaces"]["gc_mtf_" + tf.lower()]


@pytest.mark.parametrize("fault", ["cutoff", "relative", "december"])
def test_gc_units_fit_rejects_other_scope_before_statistics(synthetic_profile, monkeypatch, fault):
    kwargs = _gc_case(synthetic_profile)
    if fault == "cutoff":
        kwargs["train_end_ns"] += 60_000_000_000
    elif fault == "relative":
        kwargs["relative_price_coordinates"] = {}
    else:
        kwargs["zero_support_profile"] = {}
    def unexpected_fit(*args, **kwargs):
        pytest.fail("conflicting scope reached a statistics fitter")
    monkeypatch.setattr(norm, "fit_surface_normalization", unexpected_fit)
    with pytest.raises(RuntimeError, match="UNOBSERVED_UNITS_"):
        history.fit_gc_mtf_history_normalization(**kwargs, unobserved_units_profile=synthetic_profile)


def test_gc_units_revalidation_rejects_resealed_wrong_cutoff(synthetic_profile):
    kwargs = _gc_case(synthetic_profile)
    fitted = history.fit_gc_mtf_history_normalization(**kwargs, unobserved_units_profile=synthetic_profile)
    fitted["provenance"]["train_end_ns"] += 60_000_000_000
    fitted["contract_sha256"] = history._sha({k: v for k, v in fitted.items() if k != "contract_sha256"})
    with pytest.raises(RuntimeError, match="UNOBSERVED_UNITS_FIT_CUTOFF"):
        history.require_gc_mtf_history_normalization(fitted, layout=kwargs["layout"],
                                                    unobserved_units_profile=synthetic_profile)


def test_gc_default_and_explicit_none_keep_identical_contracts():
    kwargs = _gc_case(units.october_prefix_units_profile(), missing=False)
    implicit = history.fit_gc_mtf_history_normalization(**kwargs)
    explicit = history.fit_gc_mtf_history_normalization(**kwargs, unobserved_units_profile=None)
    assert implicit == explicit
    wrapper = {"layout": kwargs["layout"], "normalization": implicit}
    assert history.require_gc_mtf_history(wrapper) == history.require_gc_mtf_history(wrapper, unobserved_units_profile=None)
