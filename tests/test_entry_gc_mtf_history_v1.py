"""Source/clock/mask algebra on synthetic GC tables; no market or edge claim."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from gx1.contracts import entry_gc_mtf_history_v1 as history
from gx1.features import gc_native_context_v1 as gc
from gx1.features.htf_features import (
    MULTI_TF_SHIFT, multi_tf_bar_label, slice_multi_tf_v4_window,
)
from tests.test_entry_v10_input_normalization_fit import _mtf_frame, _mtf_values

MINUTE = 60_000_000_000
# The SMC builder writes these event columns as float(False/True), retaining
# unknown rows behind their own observation masks. Other GC values keep units.
FIRST_RETOUCH_COLUMNS = frozenset({
    "ob_bull_first_retouch", "ob_bear_first_retouch",
    "observed_proxy_ob_bull_first_retouch", "observed_proxy_ob_bear_first_retouch",
})
# Physical source-table identities, not tokens in a field's display name.
TABLE_TF = {"M5_GC_FEATURES.parquet": "M5"}
TABLE_TF.update({f"TRADE_FEATURES_{tf}.parquet": tf for tf in MULTI_TF_SHIFT})
TABLE_TF.update({f"BASIS_{tf}.parquet": tf for tf in ("M5", "M15", "H1", "H4")})


def _source_tf(key):
    return TABLE_TF[Path(gc.REGISTRY["tables"][key]["path"]).name]


def _source_tables():
    """Tiny in-memory Arrow tables, with distinguishable physical columns."""
    tables = {}
    anchor = pd.Timestamp("2025-01-02T00:00:00Z")
    paths = list(dict.fromkeys(row["path"] for row in gc.REGISTRY["tables"].values()))
    for ordinal, path in enumerate(paths):
        keys = [key for key, row in gc.REGISTRY["tables"].items() if row["path"] == path]
        tf = _source_tf(keys[0])
        step = int(MULTI_TF_SHIFT[tf].value)
        epoch = multi_tf_bar_label(anchor, tf).value + step
        clock = epoch + np.arange(3, dtype=np.int64) * step
        fields = [row for row in gc.REGISTRY["fields"] if row["table"] in keys]
        columns = {}
        for ci, name in enumerate(dict.fromkeys(row["column"] for row in fields)):
            columns[name] = ordinal * 1000.0 + ci * 10.0 + np.arange(3) + 0.25
        for row in fields:
            if row["name"] in gc.SOURCE_BINARY_VALUE_FIELDS:
                columns[row["column"]] = np.array([0.0, 1.0, 0.0])
            if row.get("role") == "structural_observation_mask":
                columns[row["column"]] = np.ones(3)
            if row.get("validity_column"):
                columns[row["validity_column"]] = np.ones(3)
            if row.get("source_clock_column"):
                columns[row["source_clock_column"]] = clock.copy()
        columns.update(
            available_ns=clock.copy(), bar_start_ns=clock - step,
            observed_book_valid=np.ones(3),
            observed_book_first_minute_ns=(clock - 2 * MINUTE).astype(np.float64),
            observed_book_last_minute_ns=(clock - 2 * MINUTE).astype(np.float64),
            observed_book_accepted_minutes=np.ones(3),
            observed_book_coverage_fraction=np.full(3, MINUTE / step),
            observed_book_response_span_minutes=np.ones(3),
            observed_book_response_contiguous=np.ones(3),
        )
        table = pa.table(columns)
        for key in keys:
            tables[key] = (clock, table)
    return tables


def _set_column(tables, key, column, values):
    clock, table = tables[key]
    values = pa.array(values)
    changed = table.set_column(table.schema.get_field_index(column), column, values)
    path = gc.REGISTRY["tables"][key]["path"]
    for alias, row in gc.REGISTRY["tables"].items():
        if row["path"] == path:
            tables[alias] = (clock, changed)


def _gap_source_key(tf):
    return f"TRADE_FEATURES_{tf}.parquet"


def _gap_name(tf):
    return f"gc_{tf.lower()}_observed_book_trailing_support_gap_minutes"


def _project(tf, times, tables, layout):
    return history.project_gc_mtf_source_rows(
        timeframe=tf, historical_close_ns=np.asarray(times, dtype=np.int64),
        source_tables={key: tables[key] for key in {row["table"] for row in layout["source_fields"][tf]}}, layout=layout,
    )


def gc_history_fixture():
    """Full source-owned widths; finite synthetic rows and unique TRAIN unions.

    Also reused by model tests. This does not read 024/049 payloads or 055.
    """
    from gx1.models.entry_v10.entry_v10_input_normalization import (
        select_shared_causal_mtf_fit_population,
    )
    layout = history.gc_mtf_history_layout()
    decision = pd.Timestamp("2025-01-02T12:00:00Z")
    entry_starts = np.array([decision.value - 10 * MINUTE, decision.value - 5 * MINUTE], dtype=np.int64)
    surfaces, selected = {}, {}
    for tf in layout["timeframes"]:
        length = layout["window_rows"][tf]
        step = MULTI_TF_SHIFT[tf]
        end = multi_tf_bar_label(decision, tf) + 2 * step
        starts = pd.date_range(end=end, periods=length + 5, freq=step).as_unit("ns").asi8
        frame = _mtf_frame(starts, _mtf_values(len(starts)))
        part, _window, _proof = select_shared_causal_mtf_fit_population(
            tf=tf, source=frame, entry_train_times_ns=entry_starts,
            exit_train_times_ns=np.array([], dtype=np.int64), seq_len=length,
            require_exit_route=False,
        )
        selected[tf] = part.row_indices.copy()
        names = layout["field_names"][tf]
        row = np.arange(len(starts), dtype=np.float32)
        values = (row[:, None] + np.arange(len(names), dtype=np.float32)[None, :] * 0.01).astype(np.float32)
        for name in layout["binary_fields"][tf]:
            values[:, names.index(name)] = 1.0
        for name, domain in layout["categorical_domains"][tf].items():
            values[:, names.index(name)] = np.asarray(domain, dtype=np.float32)[np.arange(len(row)) % len(domain)]
        values[:, names.index(_gap_name(tf))] = row % (int(step.value) // MINUTE)
        for vi, mi in layout["value_mask_pairs"][tf]:
            values[values[:, mi] == 0.0, vi] = 0.0
        surfaces[tf] = {
            "native_row_indices": np.arange(len(starts), dtype=np.int64),
            "available_ns": starts + int(step.value), "features": values,
            "field_names": list(names),
        }
    start_ns = int(entry_starts[0])
    end_ns = int(decision.value + MINUTE)
    normalization = history.fit_gc_mtf_history_normalization(
        surfaces, selected, train_start_ns=start_ns, train_end_ns=end_ns, layout=layout,
    )
    return {"layout": layout, "normalization": normalization}, surfaces, selected, start_ns, end_ns


@pytest.fixture(scope="module")
def gc_case():
    return gc_history_fixture()


def test_layout_keeps_each_registry_field_mask_and_owner_on_its_physical_tf():
    layout = history.gc_mtf_history_layout()
    history.require_gc_mtf_history_layout(deepcopy(layout))
    all_values = []
    for tf in layout["timeframes"]:
        expected = [row for row in gc.REGISTRY["fields"] if _source_tf(row["table"]) == tf]
        names = layout["field_names"][tf]
        assert names == [n for row in expected for n in (row["name"], row["availability_field"])] + [_gap_name(tf)]
        owners = layout["family_indices"][tf]
        flat = [i for indices in owners.values() for i in indices]
        assert sorted(flat) == list(range(len(names))) and len(flat) == len(set(flat))
        pairs = set(map(tuple, layout["value_mask_pairs"][tf]))
        for row in expected:
            value, mask = names.index(row["name"]), names.index(row["availability_field"])
            assert (value, mask) in pairs
            assert value in owners[row["owner"]] and mask in owners[row["owner"]]
            all_values.append(row["name"])
        own_bbo = names.index(f"gc_{tf.lower()}_observed_book_valid")
        assert (names.index(_gap_name(tf)), own_bbo) in pairs
    assert all_values == [row["name"] for tf in layout["timeframes"] for row in gc.REGISTRY["fields"] if _source_tf(row["table"]) == tf]
    assert set(all_values) == set(gc.VALUE_FIELDS)
    # A rolling15 indicator is physically on M5, despite the misleading token.
    assert "gc_book_rolling15_vwap_usd" in layout["field_names"]["M5"]
    assert "gc_book_rolling15_vwap_usd" not in layout["field_names"]["M15"]


def test_all_old_values_and_own_masks_are_projected_from_declared_source_columns():
    layout, tables = history.gc_mtf_history_layout(), _source_tables()
    for tf in layout["timeframes"]:
        stamps = tables[_gap_source_key(tf)][0]
        actual = _project(tf, stamps, tables, layout)
        assert actual.dtype == np.float32
        names = layout["field_names"][tf]
        for item in gc.REGISTRY["fields"]:
            if _source_tf(item["table"]) != tf:
                continue
            _clock, table = tables[item["table"]]
            expected = table[item["column"]].to_numpy().astype(np.float32)
            if item.get("transform") == "decision_age_minutes" or item["column"] in gc.DECISION_AGE_COLUMNS:
                source = table[item.get("source_clock_column") or "available_ns"].to_numpy()
                expected = ((stamps - source) / MINUTE).astype(np.float32)
            np.testing.assert_array_equal(actual[:, names.index(item["name"])], expected)
            np.testing.assert_array_equal(actual[:, names.index(item["availability_field"])], np.ones(3, np.float32))
        np.testing.assert_array_equal(actual[:, names.index(_gap_name(tf))], np.ones(3, np.float32))


def test_latest_own_bbo_mask_distinguishes_unknown_from_observed_zero_and_never_backsearches():
    layout, tables = history.gc_mtf_history_layout(), _source_tables()
    key = _gap_source_key("M5")
    clock = tables[key][0]
    _set_column(tables, key, "observed_book_valid", [1.0, 0.0, 1.0])
    last = clock - 2 * MINUTE
    last[-1] = clock[-1] - MINUTE  # Closed support endpoint exactly equals availability.
    # Arrow stays int64 despite a genuinely unknown endpoint in the invalid
    # middle row. A whole-column float conversion would hide ns precision.
    _set_column(tables, key, "observed_book_last_minute_ns", pa.array([int(last[0]), None, int(last[-1])], type=pa.int64()))
    actual = _project("M5", [clock[0] - 5 * MINUTE, *clock], tables, layout)
    names = layout["field_names"]["M5"]
    np.testing.assert_array_equal(actual[:, names.index(_gap_name("M5"))], [0, 1, 0, 0])
    np.testing.assert_array_equal(actual[:, names.index("gc_m5_observed_book_valid")], [0, 1, 0, 1])
    np.testing.assert_array_equal(actual[:, names.index("gc_m5_observed_book_valid_observed")], [0, 1, 1, 1])


@pytest.mark.parametrize("fault", ["off_grid", "future_endpoint", "before_bucket", "nonfinite", "mask_domain"])
def test_observed_gap_rejects_malformed_support(fault):
    layout, tables = history.gc_mtf_history_layout(), _source_tables()
    key = _gap_source_key("M5")
    clock, table = tables[key]
    last = table["observed_book_last_minute_ns"].to_numpy().copy()
    if fault == "mask_domain":
        _set_column(tables, key, "observed_book_valid", [1.0, 0.5, 1.0])
    else:
        if fault == "off_grid": last = last.astype(np.int64); last[1] += 1
        elif fault == "future_endpoint": last[1] = clock[1]
        elif fault == "before_bucket": last[1] = clock[1] - 6 * MINUTE
        elif fault == "nonfinite": last = last.astype(np.float64); last[1] = np.nan
        _set_column(tables, key, "observed_book_last_minute_ns", last)
    with pytest.raises(RuntimeError):
        _project("M5", clock, tables, layout)


def test_each_historical_close_joins_its_own_state_and_ignores_future_source_rows():
    layout, tables = history.gc_mtf_history_layout(), _source_tables()
    for tf in layout["timeframes"]:
        clock = tables[_gap_source_key(tf)][0]
        original = _project(tf, clock[:2], tables, layout)
        later_changed = deepcopy(tables)
        for key, (_c, table) in tables.items():
            if _source_tf(key) != tf:
                continue
            # Poison even future structural values; selected past rows must
            # remain invariant, including their masks and derived age values.
            protected = {"available_ns", "bar_start_ns", "observed_book_valid"}
            protected.update(row.get("source_clock_column") for row in gc.REGISTRY["fields"] if row["table"] == key)
            for column in table.column_names:
                if column in protected or column.endswith("_ns"):
                    continue
                values = table[column].to_numpy().copy()
                values[-1] += 10000.0
                _set_column(later_changed, key, column, values)
        np.testing.assert_array_equal(_project(tf, clock[:2], later_changed, layout), original)
        item = next(row for row in gc.REGISTRY["fields"]
                    if _source_tf(row["table"]) == tf
                    and row["column"] not in gc.DECISION_AGE_COLUMNS
                    and tables[row["table"]][1][row["column"]][0].as_py()
                        != tables[row["table"]][1][row["column"]][1].as_py())
        index = layout["field_names"][tf].index(item["name"])
        assert original[0, index] != original[1, index]


def test_missing_own_tf_bucket_stays_unknown_instead_of_carrying_previous_state():
    layout, tables = history.gc_mtf_history_layout(), _source_tables()
    for tf in layout["timeframes"]:
        clock = tables[_gap_source_key(tf)][0]
        missing = dict(tables)
        for key, (available, table) in tables.items():
            if _source_tf(key) == tf:
                missing[key] = (available[[0, 2]], table.take(pa.array([0, 2])))
        actual = _project(tf, clock, missing, layout)
        assert bool((actual[0] != 0).any()) and bool((actual[2] != 0).any())
        np.testing.assert_array_equal(actual[1], np.zeros(layout["widths"][tf], np.float32))


def test_native_row_index_option_returns_the_existing_exact_closed_window():
    start = pd.Timestamp("2025-01-02T00:00:00Z")
    stamps = pd.date_range(start, periods=24, freq="5min").as_unit("ns").asi8
    values = _mtf_values(len(stamps))
    frame = _mtf_frame(stamps, values)
    target = start + pd.Timedelta(minutes=100)
    before = slice_multi_tf_v4_window(frame, target, 16, MULTI_TF_SHIFT["M5"])
    window, indices = slice_multi_tf_v4_window(frame, target, 16, MULTI_TF_SHIFT["M5"], return_row_indices=True)
    np.testing.assert_array_equal(window, before)
    np.testing.assert_array_equal(indices, np.arange(4, 20, dtype=np.int64))
    np.testing.assert_array_equal(window, values[indices])


def test_gather_uses_exact_native_ids_and_clocks_for_each_variable_width(gc_case):
    bound, surfaces, _selected, _start, _end = gc_case
    layout = bound["layout"]
    validated = history.require_gc_mtf_history_surfaces(surfaces, layout=layout)
    assert len(set(layout["widths"].values())) > 1
    for tf, surface in validated.items():
        length = layout["window_rows"][tf]
        ids = surface["native_row_indices"][1:1 + length]
        clocks = surface["available_ns"][1:1 + length]
        actual = history.gather_gc_mtf_history_rows(surface, tf=tf, native_row_indices=ids, available_ns=clocks, layout=layout)
        np.testing.assert_array_equal(actual, surface["features"][1:1 + length])
        assert actual.shape == (length, len(layout["field_names"][tf]))
        assert not surface["features"].flags.writeable
        bad = clocks.copy(); bad[-1] += 1
        with pytest.raises(RuntimeError):
            history.gather_gc_mtf_history_rows(surface, tf=tf, native_row_indices=ids, available_ns=bad, layout=layout)
        missing = ids.copy(); missing[-1] += 1000
        with pytest.raises(RuntimeError):
            history.gather_gc_mtf_history_rows(surface, tf=tf, native_row_indices=missing, available_ns=clocks, layout=layout)


@pytest.mark.parametrize("fault", ["duplicate", "reordered", "missing", "end_boundary"])
def test_normalization_rejects_invalid_unique_train_row_selection(fault, gc_case):
    bound, surfaces, selected, start, end = gc_case
    chosen = deepcopy(selected)
    ids = chosen["M5"].copy()
    if fault == "duplicate": ids[1] = ids[0]
    elif fault == "reordered": ids[[0, 1]] = ids[[1, 0]]
    elif fault == "missing": ids[-1] += 1000
    else: end = int(surfaces["M5"]["available_ns"][ids[-1]])
    chosen["M5"] = ids
    with pytest.raises(RuntimeError):
        history.fit_gc_mtf_history_normalization(surfaces, chosen, train_start_ns=start, train_end_ns=end, layout=bound["layout"])


def test_normalization_fits_each_consumed_train_row_once_keeps_warmup_and_ignores_later_rows(gc_case):
    from gx1.contracts.entry_model_native_input_normalization_v1 import fit_surface_normalization
    bound, surfaces, selected, start, end = gc_case
    layout = bound["layout"]
    poisoned = deepcopy(surfaces)
    for tf, surface in poisoned.items():
        after = np.flatnonzero(surface["available_ns"] >= end)
        assert after.size > 0
        surface["features"][after] = np.nan  # Not part of TRAIN; must never enter fit.
        assert np.min(surfaces[tf]["available_ns"][selected[tf]]) < start
        expected = fit_surface_normalization(
            surfaces[tf]["features"][selected[tf]], surface="gc_mtf_" + tf.lower(),
            field_names=layout["field_names"][tf], semantic_categorical_domains=layout["categorical_domains"][tf],
        )
        assert bound["normalization"]["surfaces"]["gc_mtf_" + tf.lower()] == expected
        assert bound["normalization"]["provenance"]["row_union"][tf]["rows"] == len(np.unique(selected[tf]))
    changed = history.fit_gc_mtf_history_normalization(poisoned, selected, train_start_ns=start, train_end_ns=end, layout=layout)
    assert changed == bound["normalization"]
    history.require_gc_mtf_history(bound)



def test_first_retouch_history_binary_fields_follow_shared_source_domains():
    events = {f"gc_{tf.lower()}_observed_proxy_ob_{side}_first_retouch"
              for tf in history.TIMEFRAMES for side in ("bull", "bear")}
    events |= {f"gc_{tf}_ob_{side}_first_retouch"
               for tf in ("m5", "m15") for side in ("bull", "bear")}
    registry_events = {item["name"] for item in gc.REGISTRY["fields"]
                       if item["column"] in FIRST_RETOUCH_COLUMNS}
    assert registry_events == events and len(events) == 14
    old_binary = set(gc.OBSERVED_FIELDS) | set(gc.STRUCTURAL_VALUE_FIELDS)
    assert len(gc.SOURCE_BINARY_VALUE_FIELDS) == 76
    expected_binary = old_binary | gc.SOURCE_BINARY_VALUE_FIELDS
    unknown = {"gc_m30_observed_proxy_ob_bull_first_retouch",
               "gc_d1_observed_proxy_ob_bull_first_retouch_unproven"}
    for prefix in ("", "ctx_cont."):
        names = [prefix + name for name in (*gc.CONTEXT_FIELDS, *sorted(unknown))]
        expected = {prefix + name for name in expected_binary}
        assert gc.binary_fields(names) == expected
        assert gc.mtf_history_binary_fields(names) == expected
        assert {prefix + name for name in events}.issubset(expected)
    layout = history.gc_mtf_history_layout()
    for tf in history.TIMEFRAMES:
        assert set(layout["binary_fields"][tf]) == expected_binary.intersection(layout["field_names"][tf])


def test_one_positive_first_retouch_train_event_fits_identity_and_fractional_value_fails(gc_case):
    from gx1.contracts.entry_model_native_input_normalization_v1 import fit_surface_normalization
    bound, surfaces, selected, start, end = gc_case
    layout = bound["layout"]
    sparse = deepcopy(surfaces)
    for tf, surface in sparse.items():
        names = layout["field_names"][tf]
        for item in layout["source_fields"][tf]:
            if item["column"] not in FIRST_RETOUCH_COLUMNS:
                continue
            vi = names.index(item["name"])
            surface["features"][:, vi] = 0.0
            surface["features"][selected[tf][-1], vi] = 1.0
            assert np.count_nonzero(surface["features"][selected[tf], vi]) == 1
    normalization = history.fit_gc_mtf_history_normalization(
        sparse, selected, train_start_ns=start, train_end_ns=end, layout=layout,
    )
    history.require_gc_mtf_history_normalization(normalization, layout=layout)
    assert normalization["provenance"] == bound["normalization"]["provenance"]
    for tf in history.TIMEFRAMES:
        stats = normalization["surfaces"]["gc_mtf_" + tf.lower()]
        for item in layout["source_fields"][tf]:
            if item["column"] not in FIRST_RETOUCH_COLUMNS:
                continue
            vi = stats["field_names"].index(item["name"])
            assert (stats["center"][vi], stats["scale"][vi], stats["binary_mask"][vi]) == (0.0, 1.0, 1)
            assert stats["scale_source"][vi] == "semantic_binary_identity"
            assert (stats["train_transformed_min"][vi], stats["train_transformed_max"][vi]) == (0.0, 1.0)
    names = layout["field_names"]["D1"]
    bad = sparse["D1"]["features"][selected["D1"]].copy()
    bad[-1, names.index("gc_d1_observed_proxy_ob_bull_first_retouch")] = 0.5
    with pytest.raises(RuntimeError, match="BINARY_DOMAIN_INVALID"):
        fit_surface_normalization(bad, surface="gc_mtf_d1", field_names=names)


def test_first_retouch_identity_rejects_stale_055_continuous_metadata():
    from gx1.contracts.entry_model_native_input_normalization_v1 import (
        _stats_sha256, fit_surface_normalization, require_surface_normalization,
    )
    value = "gc_d1_observed_proxy_ob_bull_first_retouch"
    names = [value, value + "_observed"]
    raw = np.array([[0, 1], [0, 1], [1, 1], [1, 1]], dtype=np.float32)
    for surface in ("signal", "ctx_cont", "gc_mtf_m30"):
        stats = fit_surface_normalization(raw, surface=surface, field_names=names)
        assert stats["center"] == [0.0, 0.0] and stats["scale"] == [1.0, 1.0]
        assert stats["binary_mask"] == [1, 1]
        assert stats["scale_source"] == ["semantic_binary_identity", "semantic_binary_identity"]
        assert require_surface_normalization(stats, surface=surface, field_names=names) == stats
        # Former055 continuous encoding remains stale even with a valid stats hash.
        stale = deepcopy(stats)
        stale["center"][0] = 0.5
        stale["binary_mask"][0] = 0
        stale["binary_field_count"] = 1
        stale["scale_source"][0] = "raw_iqr"
        stale["train_transformed_min"][0] = float(np.arcsinh(-0.5))
        stale["train_transformed_max"][0] = float(np.arcsinh(0.5))
        stale["stats_sha256"] = _stats_sha256(**{key: stale[key] for key in (
            "field_names", "center", "scale", "binary_mask", "categorical_mask",
            "categorical_domains", "train_transformed_min", "train_transformed_max",
        )})
        with pytest.raises(RuntimeError, match="BINARY_CONTRACT_INVALID"):
            require_surface_normalization(stale, surface=surface, field_names=names)
    sparse = np.array([[0, 1], [0, 1], [0, 1], [0, 1], [1, 1]], dtype=np.float32)
    sparse_stats = fit_surface_normalization(sparse, surface="ctx_cont", field_names=names)
    assert sparse_stats["binary_mask"] == [1, 1]
    assert sparse_stats["scale_source"][0] == "semantic_binary_identity"
