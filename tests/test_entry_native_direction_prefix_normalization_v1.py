"""Synthetic full native dimensions only; no files, models or market inputs."""
from copy import deepcopy
import hashlib

import numpy as np
import pandas as pd
import pytest

from gx1.contracts import entry_native_direction_prefix_normalization_v1 as owner
from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.contracts import entry_model_native_input_normalization_v1 as norm
from gx1.features import gc_native_context_v1 as gc
from gx1.features import htf_features as htf
from gx1.features.basic_v1 import session_native_binary_event_fields
from gx1.features.entry_specialist_feature_groups_v1 import compile_compact_entry_execution_profile
from gx1.features.micro_structure_v1 import range_observation_binary_fields, range_observation_masking_pairs


def _hash(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<i8").tobytes()).hexdigest()


def _bind_axes(case):
    case["expected_axes"] = {"source": {"path": "/synthetic/245_targets.npz", "sha256": "1" * 64},
        "rows": len(case["parent_rows"]), **{name + "_sha256": _hash(case[name])
            for name in ("parent_rows", "bar_start_ns", "decision_ns")}}


def _matrix(rows, names, surface):
    row = np.arange(rows, dtype=np.float32)
    values = row[:, None] * np.float32(.1) + np.arange(len(names), dtype=np.float32)[None, :] * np.float32(.001)
    binary = (set(norm.MTF_SEMANTIC_BINARY_FIELDS).intersection(names)
        if surface.startswith("mtf_") else gc.binary_fields(names))
    binary |= gc.mtf_history_binary_fields(names) | range_observation_binary_fields(names) | session_native_binary_event_fields(names)
    for name in binary:
        values[:, names.index(name)] = row % 2
    categories = (norm.SIGNAL_SEMANTIC_CATEGORICAL_DOMAINS if surface == "signal" else
        norm.MTF_SEMANTIC_CATEGORICAL_DOMAINS if surface.startswith("mtf_") else {})
    for name, domain in categories.items():
        if name in names:
            values[:, names.index(name)] = np.asarray(domain, np.float32)[np.arange(rows) % len(domain)]
    if "ema_stack_aligned_v2" in names:
        values[:, names.index("ema_stack_aligned_v2")] = row % 3 - 1
    if surface.startswith("mtf_"):
        for name in htf.MULTI_TF_SOURCE_CONSTANT_BINARY_EVENTS_V4:
            values[:, names.index(name)] = 0.
    for vi, mi in (*gc.masking_pairs(names), *range_observation_masking_pairs(names)):
        values[values[:, mi] == 0., vi] = 0.
    return values.astype(np.float32)


def _sync_current(case):
    src = case["entry_sequence_source"]
    case["snap_x"] = src["signal"][src["sample_positions"]].copy()
    fields = compile_compact_entry_execution_profile()["physical_field_names"]
    ctx = np.empty((len(case["parent_rows"]), len(fields["ctx_cont"])), np.float32)
    aliases = owner.native._derive_temporal_aliases(case["ordered_signal_names"])
    assert len(aliases) == len(fields["ctx_cont"])
    for alias in aliases:
        ctx[:, alias["ctx_cont_index"]] = case["snap_x"][:, alias["signal_index"]]
    case["ctx_cont"] = ctx


def prefix_case():
    """One small full-width fixture with causal warmup and later source rows."""
    fields = compile_compact_entry_execution_profile()["physical_field_names"]
    names = list(fields["signal"])
    origin = pd.Timestamp("2025-06-02T00:00:00Z").value
    source_times = origin + np.arange(120, dtype=np.int64) * owner.BAR_NS
    positions = np.array([100, 101, 103, 104, 106, 107], dtype=np.int64)
    starts = source_times[positions]
    decisions = starts + owner.BAR_NS
    case = {"parent_rows": np.array([2, 5, 7, 9, 12, 15], dtype=np.int64),
        "bar_start_ns": starts.copy(), "decision_ns": decisions.copy(),
        "cutoff_ns": int(decisions[4]), "expected_prefix_parent_rows": np.array([2, 5, 7, 9], dtype=np.int64),
        "entry_sequence_source": {"signal": _matrix(120, names, "signal"), "times_ns": source_times,
            "sample_positions": positions, "audit": {"synthetic_only": True}},
        "ordered_signal_names": names, "ctx_cat": (np.arange(6, dtype=np.int64) % 4)[:, None],
        "gc_layout": history.gc_mtf_history_layout(), "multi_tf_sources": {}, "gc_surfaces": {},
        "source_bindings": {"synthetic_only": True, "feature_constants": "fixed synthetic metadata, no fit"},
        "base_lineage": {"dataset_run_id": "SYNTHETIC_PREFIX", "train_parquet_path": "/synthetic/train.parquet",
            "train_parquet_sha256": "2" * 64, "train_manifest_path": "/synthetic/train.manifest.json",
            "train_manifest_sha256": "3" * 64, "m5_prebuilt_path": "/synthetic/m5.parquet",
            "m5_prebuilt_sha256": "4" * 64, "mtf_cache_manifest_path": "/synthetic/cache/manifest.json",
            "mtf_cache_manifest_sha256": "5" * 64, "mtf_builder_version": "synthetic_native_cache",
            "mtf_feature_names_sha256": norm._field_names_sha256(fields["mtf"])}}
    _bind_axes(case)
    _sync_current(case)
    for tf, length in owner.WINDOWS.items():
        step = htf.MULTI_TF_SHIFT[tf]
        end = htf.multi_tf_bar_label(pd.Timestamp(int(decisions[-1]), tz="UTC"), tf) + 2 * step
        times = pd.date_range(end=end, periods=length + 8, freq=step).as_unit("ns").asi8
        values = _matrix(len(times), list(fields["mtf"]), "mtf_" + tf.lower())
        frame = pd.DataFrame(values, index=pd.DatetimeIndex(times.astype("datetime64[ns]"), tz="UTC"), columns=fields["mtf"], copy=False)
        frame.attrs.update(ts_int64=times.copy(), feats_np=values, causal_warmup_rows=0, htf_feature_contract=htf.HTF_V4_MATRIX_CONTRACT)
        case["multi_tf_sources"][tf] = frame
        gc_names = case["gc_layout"]["field_names"][tf]
        raw = _matrix(len(times), gc_names, "gc_mtf_" + tf.lower())
        # A genuine paired continuous constant is accepted by the unchanged GC fitter.
        binary = gc.mtf_history_binary_fields(gc_names)
        vi, mi = next((vi, mi) for vi, mi in case["gc_layout"]["value_mask_pairs"][tf]
            if gc_names[vi] not in binary and vi != len(gc_names) - 1)
        raw[:, vi] = 7.
        raw[:, mi] = 1.
        raw[:, -1] = np.arange(len(times)) % (int(step.value) // 60_000_000_000)
        for value, mask in case["gc_layout"]["value_mask_pairs"][tf]:
            raw[raw[:, mask] == 0., value] = 0.
        case["gc_surfaces"][tf] = {"native_row_indices": np.arange(len(times), dtype=np.int64),
            "available_ns": times + int(step.value), "features": raw, "field_names": list(gc_names)}
    return case


def _qualify(case):
    return owner.qualify_native_direction_prefix_population(**case)


def _continuous_gc(case, tf):
    names = case["gc_layout"]["field_names"][tf]
    binary = gc.mtf_history_binary_fields(names)
    return next(i for i, name in enumerate(names) if name not in binary and i != len(names) - 1)


def test_full_native_population_and_unpatched_fit_require_path():
    case = prefix_case()
    q = _qualify(case)
    report, pop = q["report"], q["population"]
    assert report["fit_allowed"] and pop is not None and report["statistics_fits"] == 0
    assert len(report["support"]) == 12
    assert report["proof"]["prefix"]["rows"] == 4
    assert report["proof"]["local"]["selected_unique_row_count"] == 100
    assert report["support"]["ctx_cont"]["population_rows"] == 4
    for tf, length in owner.WINDOWS.items():
        proof = report["proof"]["tf"][tf]
        assert proof["routes"]["entry"]["enabled"] is True
        assert proof["routes"]["exit"]["enabled"] is False
        assert proof["selected_unique_row_count"] >= length
        assert report["proof"]["gc"][tf]["rows"] == proof["selected_unique_row_count"]
        assert len(report["support"]["gc_mtf_" + tf.lower()]["field_names"]) == case["gc_layout"]["widths"][tf]
    fitted = owner.fit_native_direction_prefix_normalization(q)
    base = fitted["base_normalization"]
    norm.require_input_normalization_contract(base, expected_field_names={k: v["field_names"] for k, v in base["surfaces"].items()},
        expected_ctx_cat_names=["session_id"], expected_entry_training_phase=owner.ENTRY_ONLY_DYNAMIC_M5_PATH_TRAINING_PHASE)
    history.require_gc_mtf_history(fitted["gc_mtf_history"])
    assert base["lineage"]["exit_train_decision_row_count"] == 0
    assert base["lineage"]["train_row_count"] == 4
    assert base["lineage"]["local_fit_row_count"] == 100
    assert any("constant_train_unit_scale" in s["scale_source"] for s in fitted["gc_mtf_history"]["normalization"]["surfaces"].values())
    assert fitted["old055_or223_normalization_identity_reused"] is False


def test_qualifier_has_no_statistics_fit_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("qualification cannot fit")
    monkeypatch.setattr(norm, "fit_surface_normalization", forbidden)
    monkeypatch.setattr(norm, "fit_ctx_cat_contract", forbidden)
    monkeypatch.setattr(history, "fit_gc_mtf_history_normalization", forbidden)
    assert _qualify(prefix_case())["report"]["fit_allowed"] is True


def test_original_barstart_and_strict_cutoff_select_exact_native_rows():
    case = prefix_case()
    q = _qualify(case)
    pop = q["population"]
    m5 = case["multi_tf_sources"]["M5"].attrs["ts_int64"]
    assert m5[pop["tf_ids"]["M5"][-1]] + owner.BAR_NS == case["decision_ns"][3]
    assert q["report"]["proof"]["prefix"]["last_decision_ns"] < case["cutoff_ns"]
    for tf in ("H4", "D1"):
        times = case["multi_tf_sources"][tf].attrs["ts_int64"]
        label = htf.multi_tf_last_closed_label(pd.Timestamp(int(case["bar_start_ns"][3]), tz="UTC"), tf)
        assert times[pop["tf_ids"][tf][-1]] == label.value


@pytest.mark.parametrize("fault", ["parent_hash", "clock_hash", "row_count", "double_shift", "offgrid", "prefix_omission", "cutoff_bool", "short_local", "missing_tf", "field_removal"])
def test_full_axis_geometry_and_prefix_fail_closed(fault):
    case = prefix_case()
    if fault == "parent_hash": case["expected_axes"]["parent_rows_sha256"] = "0" * 64
    if fault == "clock_hash": case["expected_axes"]["decision_ns_sha256"] = "0" * 64
    if fault == "row_count": case["expected_axes"]["rows"] += 1
    if fault == "double_shift": case["decision_ns"] += owner.BAR_NS
    if fault == "offgrid":
        case["bar_start_ns"] += 1
        case["decision_ns"] += 1
        _bind_axes(case)
    if fault == "prefix_omission": case["expected_prefix_parent_rows"] = case["parent_rows"][[0, 2, 3]]
    if fault == "cutoff_bool": case["cutoff_ns"] = True
    if fault == "short_local": case["entry_sequence_source"]["sample_positions"][0] = 94
    if fault == "missing_tf": del case["multi_tf_sources"]["D1"]
    if fault == "field_removal": case["ordered_signal_names"] = case["ordered_signal_names"][:-1]
    with pytest.raises(RuntimeError):
        _qualify(case)


def test_native_warmup_is_not_filled_or_skipped():
    case = prefix_case()
    q = _qualify(case)
    case["multi_tf_sources"]["M5"].attrs["causal_warmup_rows"] = int(q["population"]["tf_ids"]["M5"][0]) + 1
    with pytest.raises(RuntimeError, match="WARMUP_INCOMPLETE"):
        _qualify(case)


@pytest.mark.parametrize("surface", ["signal", "gc"])
def test_zero_observation_report_preserves_counts_without_population_or_fit(monkeypatch, surface):
    case = prefix_case()
    if surface == "signal":
        vi, mi = gc.masking_pairs(case["ordered_signal_names"])[0]
        case["entry_sequence_source"]["signal"][:, [vi, mi]] = 0.
        _sync_current(case)
    else:
        vi, mi = case["gc_layout"]["value_mask_pairs"]["M5"][0]
        case["gc_surfaces"]["M5"]["features"][:, [vi, mi]] = 0.
    q = _qualify(case)
    assert q["population"] is None and q["report"]["fit_allowed"] is False
    assert len(q["report"]["support"]) == 12
    assert any(item["reason"] == "no_observed_prefix_values" for item in q["report"]["blockers"])
    monkeypatch.setattr(norm, "fit_surface_normalization", lambda *a, **k: pytest.fail("fit before support"))
    with pytest.raises(RuntimeError, match="PREFIX_SUPPORT_BLOCKED"):
        owner.fit_native_direction_prefix_normalization(q)


def test_delayed_mask_distinguishes_observed_zero_from_unknown():
    case = prefix_case()
    vi, mi = gc.masking_pairs(case["ordered_signal_names"])[0]
    matrix = case["entry_sequence_source"]["signal"]
    matrix[:, mi] = 0.
    matrix[:, vi] = 0.
    matrix[101:, mi] = 1.
    _sync_current(case)
    q = _qualify(case)
    assert q["report"]["support"]["signal"]["observed_counts"][vi] == 4
    assert q["report"]["support"]["signal"]["nonzero_observed_counts"][vi] == 0
    ci = next(alias["ctx_cont_index"] for alias in owner.native._derive_temporal_aliases(case["ordered_signal_names"])
        if alias["signal_index"] == vi)
    assert q["report"]["support"]["ctx_cont"]["observed_counts"][ci] == 3
    assert q["report"]["fit_allowed"] is True


@pytest.mark.parametrize("fault", ["nan", "falsemask_nonzero", "alias_bits"])
def test_invalid_storage_and_alias_parity_are_rejected(fault):
    case = prefix_case()
    vi, mi = gc.masking_pairs(case["ordered_signal_names"])[0]
    if fault == "nan": case["entry_sequence_source"]["signal"][20, vi] = np.nan
    if fault == "falsemask_nonzero":
        case["entry_sequence_source"]["signal"][20, mi] = 0.
        case["entry_sequence_source"]["signal"][20, vi] = 2.
    if fault == "alias_bits": case["ctx_cont"][0, 0] += 1.
    with pytest.raises(RuntimeError):
        _qualify(case)


def test_inherited_current_alias_support_collapse_is_reported_before_fit():
    case = prefix_case()
    vi = case["ordered_signal_names"].index("ctx_cont.atr_bps")
    case["entry_sequence_source"]["signal"][case["entry_sequence_source"]["sample_positions"][:4], vi] = 3.
    _sync_current(case)
    q = _qualify(case)
    assert q["population"] is None
    assert any(item["reason"] == "inherited_continuous_alias_support_collapse" and item["field"] == "atr_bps"
        for item in q["report"]["blockers"])


def test_partial_current_alias_constant_support_is_retained_and_blocks_fit():
    case = prefix_case()
    names = case["ordered_signal_names"]
    aliases = {item["signal_index"]: item for item in owner.native._derive_temporal_aliases(names)}
    pairs = gc.masking_pairs(names)
    vi, mi = next((vi, mi) for vi, mi in pairs if vi in aliases and names[vi] not in gc.binary_fields(names))
    matrix = case["entry_sequence_source"]["signal"]
    matrix[:, mi] = 1.
    matrix[:, vi] = np.arange(len(matrix), dtype=np.float32) * np.float32(.1)
    positions = case["entry_sequence_source"]["sample_positions"][:4]
    matrix[positions, vi] = 3.
    matrix[positions[[0, 2]], mi] = 0.
    for dependent, mask in pairs:
        if mask == mi:
            matrix[positions[[0, 2]], dependent] = 0.
    _sync_current(case)
    q = _qualify(case)
    report = q["report"]
    ci = aliases[vi]["ctx_cont_index"]
    assert report["support"]["ctx_cont"]["observed_counts"][ci] == 2
    assert report["support"]["ctx_cont"]["observed_constant"][ci] is True
    assert report["support"]["signal"]["observed_constant"][vi] is False
    assert len(report["support"]) == 12 and report["statistics_fits"] == 0
    assert q["population"] is None and report["fit_allowed"] is False
    assert any(item["field"] == aliases[vi]["ctx_cont_field"] and
        item["reason"] == "partial_current_alias_constant_support_requires_statistics_validation"
        for item in report["blockers"])


@pytest.mark.parametrize("fault", ["proof", "local_value", "gc_value"])
def test_mutated_qualification_is_rejected_before_statistics(monkeypatch, fault):
    case = prefix_case()
    q = _qualify(case)
    if fault == "proof": q["report"]["proof"]["prefix"]["rows"] += 1
    if fault == "local_value": case["entry_sequence_source"]["signal"][20, case["ordered_signal_names"].index("ret_1")] += .25
    if fault == "gc_value":
        ids = q["population"]["tf_ids"]["M5"]
        vi = _continuous_gc(case, "M5")
        case["gc_surfaces"]["M5"]["features"][ids[0], vi] += .25
    monkeypatch.setattr(norm, "fit_surface_normalization", lambda *a, **k: pytest.fail("mutated inputs reached fit"))
    with pytest.raises(RuntimeError):
        owner.fit_native_direction_prefix_normalization(q)


@pytest.mark.parametrize("fault", ["availability", "missing_id", "width"])
def test_gc_uses_same_exact_native_ids_and_clocks(fault):
    case = prefix_case()
    q = _qualify(case)
    surface = case["gc_surfaces"]["M5"]
    if fault == "availability": surface["available_ns"] += 60_000_000_000
    if fault == "missing_id":
        missing = int(q["population"]["tf_ids"]["M5"][0])
        keep = surface["native_row_indices"] != missing
        for key in ("native_row_indices", "available_ns", "features"):
            surface[key] = surface[key][keep]
    if fault == "width": surface["features"] = surface["features"][:, :-1]
    with pytest.raises(RuntimeError):
        _qualify(case)


def test_future_value_perturbations_cannot_change_prefix_stats():
    case = prefix_case()
    q = _qualify(case)
    first = owner.fit_native_direction_prefix_normalization(q)
    altered = deepcopy(case)
    ret = altered["ordered_signal_names"].index("ret_1")
    altered["entry_sequence_source"]["signal"][105:, ret] += 1000.
    altered["snap_x"][4:, ret] += 1000.
    for tf in owner.WINDOWS:
        ids = q["population"]["tf_ids"][tf]
        frame = altered["multi_tf_sources"][tf]
        frame.attrs["feats_np"][ids[-1] + 1:, list(frame.columns).index("atr_bps_14")] += 1000.
        raw = altered["gc_surfaces"][tf]["features"]
        vi = _continuous_gc(altered, tf)
        mi = dict(altered["gc_layout"]["value_mask_pairs"][tf])[vi]
        future = np.arange(len(raw)) > ids[-1]
        raw[future & (raw[:, mi] == 1.), vi] += 1000.
    second = owner.fit_native_direction_prefix_normalization(_qualify(altered))
    assert first == second


def test_observed_constant_keeps_units_and_accepts_later_novel_values():
    case = prefix_case()
    ret = case["ordered_signal_names"].index("ret_1")
    case["entry_sequence_source"]["signal"][:, ret] = 7.
    _sync_current(case)
    fitted = owner.fit_native_direction_prefix_normalization(_qualify(case))
    stats = fitted["base_normalization"]["surfaces"]["signal"]
    assert stats["scale_source"][ret] == "constant_train_unit_scale"
    assert stats["center"][ret] == 7. and stats["scale"][ret] == 1.
    later = case["snap_x"][:1].copy()
    later[0, ret] = 9.
    transformed = norm.apply_surface_normalization(later, stats)
    assert transformed[0, ret] == np.float32(np.arcsinh(2.))
