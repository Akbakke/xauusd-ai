"""Bounded193 snapshot/target adapters; no policy, cache loader, model or fit.

The future capped runner must verify the191 recipe/scope/055 and051 file refs,
the complete immutable MTF cache inventory/bytes/liveness metadata, and output
publication. These functions do not grant admission. Target and input phases
are separate: no returned target object retains a quote index or fill surface.
"""
from __future__ import annotations

from collections.abc import Mapping
import gc
import hashlib
import numpy as np

HORIZONS = (1, 5, 15)
TIMEFRAMES = ("M5", "M15", "H1", "H4", "D1")
MAX_BLOCK_ROWS = 1024
SCHEMA_VERSION = "gx1_direction_snapshot_data_v1"


def _require(condition, reason):
    if not condition:
        raise RuntimeError("DIRECTION_SNAPSHOT_DATA_" + reason)


def _clock(value, *, length=None):
    _require(isinstance(value, np.ndarray) and value.dtype == np.int64
        and value.ndim == 1 and len(value) > 0
        and (length is None or len(value) == length)
        and np.all(np.diff(value) > 0), "CLOCK")
    return value


def _digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _block_rows(value):
    _require(type(value) is int and 0 < value <= MAX_BLOCK_ROWS, "BLOCK_ROWS")


def require_direction_sequence_snapshot_evidence(audit, *, parquet_binding,
        manifest_binding, rows):
    """Join already byte-verified051 all-row evidence; never decode seq/050."""
    checks = (
        "all_values_finite_float32",
        "every_emitted_timestamp_maps_exactly_once_to_feature_surface",
        "every_sequence_equals_exact_source_surface_history_bit_identical",
        "every_snapshot_equals_exact_source_surface_current_row_bit_identical",
        "source_feature_surface_hash_matches_split_manifest",
        "source_feature_surface_manifest_hash_matches_split_manifest",
        "source_feature_surface_schema_and_dimensions_exact",
        "source_timestamps_strictly_increasing",
    )
    _require(isinstance(audit, Mapping) and type(rows) is int and rows > 0
        and audit.get("schema_version") == "entry_model_native_sequence_source_reconstruction_audit_v1"
        and audit.get("decision") == "PASS" and audit.get("rows") == rows
        and audit.get("sequence_shape") == [rows, 96, 2725]
        and audit.get("snapshot_shape") == [rows, 2725]
        and all(audit.get("checks", {}).get(k) is True for k in checks)
        and {k: audit.get("parquet_" + k) for k in ("path", "sha256")} == parquet_binding
        and {k: audit.get("manifest_" + k) for k in ("path", "sha256")} == manifest_binding,
        "SEQUENCE_SNAPSHOT_EVIDENCE")
    return {"rows": rows, "seq_last_equals_snap": True,
        "proof": "same_byte_bound051_sequences_and_snapshots_equal_same_source_history_current_row",
        "feature_surface_sha256": audit["feature_surface_sha256"],
        "sequence_source_chain_sha256": audit["sequence_source_chain_sha256"]}


def prepare_direction_mtf_surface(*, timeframe, timestamps_ns, features,
        warmup_rows, sequence_bars):
    """Validate one already byte-verified cache surface once; no liveness refit."""
    from gx1.features.htf_features import (
        MULTI_TF_PER_BAR_FEATURES_V4, MULTI_TF_SHIFT, require_multi_tf_timestamp_grid,
    )
    from gx1.contracts.entry_exit_production_architecture_v1 import PRODUCTION_MTF_PER_TF_WINDOW_BARS
    _require(timeframe in TIMEFRAMES and type(sequence_bars) is int
        and sequence_bars == dict(PRODUCTION_MTF_PER_TF_WINDOW_BARS)[timeframe],
        "MTF_NATIVE_GEOMETRY")
    times = _clock(timestamps_ns)
    _require(isinstance(features, np.ndarray) and features.dtype == np.float32
        and features.shape == (len(times), len(MULTI_TF_PER_BAR_FEATURES_V4))
        and type(warmup_rows) is int and 0 <= warmup_rows < len(times), "MTF_SURFACE")
    require_multi_tf_timestamp_grid(times, timeframe=timeframe, context="DIRECTION_SNAPSHOT_DATA")
    times, values = times.view(), features.view()
    times.setflags(write=False)
    values.setflags(write=False)
    return {"timeframe": timeframe, "times": times, "features": values,
        "warmup_rows": warmup_rows, "sequence_bars": sequence_bars,
        "duration_ns": int(MULTI_TF_SHIFT[timeframe].value)}


def gather_direction_mtf_current(surface, bar_start_ns):
    """Same closed/as-of and full-window warmup geometry as native slice owner."""
    clock = _clock(bar_start_ns)
    _require(np.all(clock % (300 * 10**9) == 0), "M5_CLOCK_GRID")
    times, features = surface["times"], surface["features"]
    _require(not times.flags.writeable and not features.flags.writeable, "MTF_MUTABLE_VIEW")
    availability = clock + 300 * 10**9
    right = np.searchsorted(times, availability - surface["duration_ns"], side="right")
    _require(np.all(right >= surface["sequence_bars"])
        and np.all(right - surface["sequence_bars"] >= surface["warmup_rows"]),
        "MTF_HISTORY_OR_WARMUP")
    selected = right - 1
    current = np.asarray(features[selected], dtype=np.float32)
    _require(np.isfinite(current).all()
        and np.all(times[selected] + surface["duration_ns"] <= availability), "MTF_CURRENT")
    return current, selected.astype(np.int64), np.asarray(times[selected], dtype=np.int64)


def gather_direction_last_closed_m1_ofi(m1, *, parent_rows, decision_times_ns,
        current_m5_contract_index):
    """Project raw OFI at exactly [D-1min,D), retaining every supplied parent.

    The caller owns immutable file/NPZ, TRAIN scope and complete source-schema
    qualification. This function neither loads data nor fits statistics. Pass
    authoritative current-M5 contract IDs, or explicitly None when no such
    comparison is declared. Missing/invalid exact rows remain unknown; an older
    observed row is never substituted. Zero storage is paired with observed=False.
    Returned arrays are isolated readonly copies, not mutable source views.
    """
    minute_ns = 60 * 10**9
    required = {"bar_start_ns", "available_ns", "contract_index", "assigned",
        "ofi_contracts", "ofi_valid", "receive_clock_excluded",
        "book_event_state_excluded"}
    _require(isinstance(m1, Mapping) and required <= set(m1), "LAST_M1_KEYS")
    times = _clock(m1["bar_start_ns"])
    decisions = _clock(decision_times_ns)
    parents = parent_rows
    _require(isinstance(parents, np.ndarray) and parents.dtype == np.int64
        and parents.shape == decisions.shape and np.all(parents >= 0)
        and np.all(np.diff(parents) > 0), "LAST_M1_PARENTS")
    _require(np.all(times >= 0) and np.all(times % minute_ns == 0)
        and np.all(times <= np.iinfo(np.int64).max - minute_ns), "LAST_M1_GRID")
    _require(np.all(decisions >= minute_ns)
        and np.all(decisions % (5 * minute_ns) == 0), "LAST_M1_DECISION_GRID")
    for name in required:
        value = m1[name]
        dtype = (np.dtype("int64") if name in ("bar_start_ns", "available_ns")
            else np.dtype("int16") if name == "contract_index"
            else np.dtype("float64") if name == "ofi_contracts" else np.dtype("bool"))
        _require(isinstance(value, np.ndarray) and value.dtype == dtype
            and value.shape == times.shape and not value.flags.writeable,
            "LAST_M1_IMMUTABLE_SOURCE_HEADER:" + name)
    _require(not parents.flags.writeable and not decisions.flags.writeable,
        "LAST_M1_IMMUTABLE_PARENTS")
    if current_m5_contract_index is not None:
        current = current_m5_contract_index
        _require(isinstance(current, np.ndarray) and current.dtype.kind == "i"
            and current.shape == decisions.shape and not current.flags.writeable
            and np.all(current >= -1), "LAST_M1_M5_CONTRACT_HEADER")
    wanted = decisions - minute_ns
    positions = np.searchsorted(times, wanted, side="left")
    found = positions < len(times)
    found[found] &= times[positions[found]] == wanted[found]
    source_rows = np.full(len(parents), -1, dtype=np.int64)
    source_rows[found] = positions[found]
    selected = positions[found]
    _require(np.array_equal(m1["available_ns"][selected], decisions[found]),
        "LAST_M1_EXACT_AVAILABILITY")
    observed = np.zeros(len(parents), dtype=np.bool_)
    eligible = (m1["assigned"][selected] & m1["ofi_valid"][selected]
        & ~m1["receive_clock_excluded"][selected]
        & ~m1["book_event_state_excluded"][selected]
        & (m1["contract_index"][selected] >= 0))
    if current_m5_contract_index is not None:
        eligible &= ((current_m5_contract_index[found] >= 0)
            & (m1["contract_index"][selected] == current_m5_contract_index[found]))
    observed[found] = eligible
    values = np.zeros(len(parents), dtype=np.float64)
    raw = m1["ofi_contracts"][selected[eligible]]
    _require(np.isfinite(raw).all(), "LAST_M1_OBSERVED_OFI_NONFINITE")
    values[observed] = raw
    result = {"parent_rows": parents.copy(), "decision_ns": decisions.copy(),
        "source_row_indices": source_rows, "last_closed_m1_ofi_contracts": values,
        "last_closed_m1_ofi_observed": observed}
    for value in result.values():
        value.setflags(write=False)
    return result


def iter_direction_raw_snapshot_batches(parquet_binding, *, expected_times_ns,
        block_rows=MAX_BLOCK_ROWS):
    """One chunked nested-column pass; verified scope/metadata remain caller-owned."""
    import pandas as pd
    import pyarrow.parquet as pq
    from gx1.contracts.local_random_access_campaign_v2 import require_binding
    _block_rows(block_rows)
    expected = _clock(expected_times_ns)
    binding = require_binding(parquet_binding, label="direction physical snapshot", verify_file=True)
    pf = pq.ParquetFile(binding["path"])
    columns = ["time", "snap", "ctx_cont", "ctx_cat"]
    _require(pf.metadata.num_rows == len(expected)
        and set(columns) <= set(pf.schema_arrow.names), "PARQUET_SCOPE")
    offset = 0
    for batch in pf.iter_batches(batch_size=block_rows, columns=columns, use_threads=False):
        count = batch.num_rows
        times = pd.DatetimeIndex(pd.to_datetime(
            batch.column("time").to_pandas(), utc=True)).as_unit("ns").asi8
        _require(np.array_equal(times, expected[offset:offset + count]), "PHYSICAL_PARENT_CLOCK")
        raw = {}
        for name, output, width, dtype in (
                ("snap", "snap_x", 2725, np.float32),
                ("ctx_cont", "ctx_cont", 2542, np.float32),
                ("ctx_cat", "ctx_cat", 1, np.int64)):
            column = batch.column(name)
            _require(column.null_count == 0 and len(column.flatten()) == count * width
                and column.flatten().null_count == 0
                and np.all(np.diff(column.offsets.to_numpy()) == width), "NESTED_SHAPE:" + name)
            values = column.flatten().to_numpy(zero_copy_only=False).reshape(count, width)
            if name == "ctx_cat":
                _require(np.isfinite(values).all() and np.equal(values, np.floor(values)).all()
                    and np.isin(values, (0, 1, 2, 3)).all(), "CTX_CAT_DOMAIN")
            raw[output] = values.astype(dtype, copy=False)
        yield {"parents": np.arange(offset, offset + count, dtype=np.int64),
            "bar_start_ns": np.array(times, dtype=np.int64, copy=True), "raw": raw}
        offset += count
    _require(offset == len(expected), "PARQUET_ROW_COUNT")


def write_direction_snapshot_phase(batches, *, parent_times_ns, layout,
        mtf_surfaces, destination):
    """Write caller-owned F32 memmap incrementally; no dense centered copy."""
    from gx1.contracts.entry_direction_snapshot_reference_v1 import project_direction_snapshot
    expected = _clock(parent_times_ns)
    _require(set(mtf_surfaces) == set(TIMEFRAMES)
        and isinstance(destination, np.memmap) and destination.dtype == np.float32
        and destination.shape == (len(expected), 3345)
        and len(layout["columns"]) == 3345, "OUTPUT_OR_LAYOUT")
    rows, matrix_hash = 0, hashlib.sha256()
    tf_row_hash = {tf: hashlib.sha256() for tf in TIMEFRAMES}
    tf_clock_hash = {tf: hashlib.sha256() for tf in TIMEFRAMES}
    for batch in batches:
        parents, clock = batch["parents"], batch["bar_start_ns"]
        count = len(parents)
        _require(0 < count <= MAX_BLOCK_ROWS and parents.dtype == np.int64
            and np.array_equal(parents, np.arange(rows, rows + count, dtype=np.int64))
            and np.array_equal(clock, expected[rows:rows + count]), "BATCH_PARENT_ORDER")
        raw = dict(batch["raw"])
        for tf in TIMEFRAMES:
            current, selected, selected_clock = gather_direction_mtf_current(mtf_surfaces[tf], clock)
            raw["seq_" + tf.lower()] = current
            tf_row_hash[tf].update(selected.astype("<i8", copy=False).tobytes())
            tf_clock_hash[tf].update(selected_clock.astype("<i8", copy=False).tobytes())
        projected = project_direction_snapshot(raw, layout=layout)
        destination[rows:rows + count] = projected
        matrix_hash.update(np.ascontiguousarray(projected, dtype="<f4").tobytes())
        rows += count
    _require(rows == len(expected), "OUTPUT_ROW_COUNT")
    destination.flush()
    return {"schema_version": SCHEMA_VERSION, "rows": rows, "columns": 3345,
        "parent_order_sha256": _digest(np.arange(rows, dtype="<i8")),
        "bar_start_clock_sha256": _digest(expected.astype("<i8", copy=False)),
        "decision_clock_sha256": _digest((expected + 300 * 10**9).astype("<i8")),
        "projected_float32_sha256": matrix_hash.hexdigest(),
        "layout_contract_sha256": layout["contract"]["contract_sha256"],
        "normalization_contract_sha256": layout["contract"]["normalization_contract_sha256"],
        "tf_selected_row_sha256": {tf: h.hexdigest() for tf, h in tf_row_hash.items()},
        "tf_selected_clock_sha256": {tf: h.hexdigest() for tf, h in tf_clock_hash.items()},
        "model_forwards": 0, "reference_fits": 0}


def direction_targets_from_observations(observations, *, parent_rows,
        decision_times_ns, economics, split_end_exclusive_ns):
    """Native F64 gross/cost -> F32 sides ->192 F64 contrast; own terminal mask."""
    import pandas as pd
    import torch
    from gx1.contracts.entry_observed_market_v1 import (
        build_entry_observed_market_targets, require_entry_observed_market_economics,
    )
    parents, decisions = parent_rows, _clock(decision_times_ns)
    _require(isinstance(parents, np.ndarray) and parents.dtype == np.int64
        and parents.shape == decisions.shape and np.all(parents >= 0)
        and np.all(np.diff(parents) > 0), "TARGET_PARENTS")
    size, q = len(parents), len(parents) * len(HORIZONS)
    repeated, horizons = np.repeat(parents, 3), np.tile(HORIZONS, size)
    terminal, gross = observations["terminal_gross_observed"], observations["terminal_gross_return_bps"]
    end, elapsed = observations["terminal_time_ns"], observations["elapsed_wall_seconds"]
    _require(observations.get("endpoint_kind") == "terminal_open_inclusive"
        and observations.get("side_order") == ("long", "short")
        and np.array_equal(observations["query_entry_rows"], repeated)
        and np.array_equal(observations["horizon_minutes"], horizons)
        and isinstance(terminal, np.ndarray) and terminal.dtype == np.bool_
        and terminal.shape == (q, 2) and np.array_equal(terminal[:, 0], terminal[:, 1])
        and isinstance(gross, np.ndarray) and gross.dtype == np.float64 and gross.shape == (q, 2)
        and np.isfinite(gross[terminal]).all() and np.isnan(gross[~terminal]).all()
        and isinstance(end, np.ndarray) and end.dtype == np.int64 and end.shape == (q,)
        and isinstance(elapsed, np.ndarray) and elapsed.dtype == np.int64
        and np.array_equal(elapsed, horizons * 60)
        and np.array_equal(end, np.repeat(decisions, 3) + elapsed * 10**9)
        and type(split_end_exclusive_ns) is int, "TARGET_QUERY")
    observed = terminal[:, 0]
    cost = require_entry_observed_market_economics(economics)
    _require(np.all(end[observed] < split_end_exclusive_ns)
        and np.all((end - elapsed * 10**9)[observed] >= pd.Timestamp(cost["coverage_start_utc"]).value)
        and np.all(end[observed] < pd.Timestamp(cost["coverage_end_utc_exclusive"]).value),
        "TARGET_CUTOFF_OR_COST_COVERAGE")
    side_targets = np.zeros((q, 3), dtype=np.float32)
    if observed.any():
        side_targets[observed] = build_entry_observed_market_targets(
            gross_return_bps=torch.tensor(gross[observed], dtype=torch.float64),
            elapsed_wall_clock_seconds=torch.from_numpy(elapsed[observed].copy()),
            economics=economics)["target_bps"].numpy()
    _require(np.isfinite(side_targets).all(), "TARGET_F32_OVERFLOW")
    valid = np.column_stack((terminal, terminal[:, 0])).reshape(size, 3, 3)
    side_targets = side_targets.reshape(size, 3, 3)
    contrast = (side_targets[:, :, 0].astype(np.float64)
        - side_targets[:, :, 1].astype(np.float64)) / 2.0
    return {"terminal_net_bps": side_targets, "terminal_observed": valid,
        "contrast_bps": contrast, "contrast_observed": valid[:, :, 0] & valid[:, :, 1],
        "terminal_time_ns": end.reshape(size, 3).copy(),
        "elapsed_wall_seconds": elapsed.reshape(size, 3).copy()}


def materialize_direction_target_phase(scope, *, split, block_rows=MAX_BLOCK_ROWS):
    """Canonical scalar/index reader once per split; return only small arrays."""
    import pandas as pd
    from gx1.contracts.entry_causal_m1_outcomes_v1 import query_dynamic_m1_prefix
    from gx1.contracts.entry_observed_market_v1 import read_dynamic_m5_target_source
    _require(split in ("train", "val"), "TARGET_SPLIT")
    _block_rows(block_rows)
    source = read_dynamic_m5_target_source(scope, split=split)
    index, fill, frame = source["index"], source["fill"], source["frame"]
    bar_clock = pd.DatetimeIndex(pd.to_datetime(frame["time"], utc=True)).as_unit("ns").asi8.copy()
    _clock(bar_clock)
    rows = len(bar_clock)
    cutoff = pd.Timestamp(scope["train_control_cutoff" if split == "train"
        else "development_control_entry_end_exclusive"]).value
    output = {
        "parent_rows": np.arange(rows, dtype=np.int64), "bar_start_ns": bar_clock,
        "dynamic_entry_m1_row": frame["dynamic_entry_m1_row"].to_numpy(np.int64, copy=True),
        "terminal_net_bps": np.zeros((rows, 3, 3), np.float32),
        "terminal_observed": np.zeros((rows, 3, 3), bool),
        "contrast_bps": np.zeros((rows, 3), np.float64),
        "contrast_observed": np.zeros((rows, 3), bool),
        "terminal_time_ns": np.zeros((rows, 3), np.int64),
        "elapsed_wall_seconds": np.zeros((rows, 3), np.int64),
    }
    for start in range(0, rows, block_rows):
        stop = min(rows, start + block_rows)
        parents = output["parent_rows"][start:stop]
        queried = query_dynamic_m1_prefix(index, fill_surface=fill,
            query_entry_rows=np.repeat(parents, 3), horizon_minutes=np.tile(HORIZONS, len(parents)),
            endpoint_kind="terminal_open_inclusive")
        block = direction_targets_from_observations(queried, parent_rows=parents,
            decision_times_ns=bar_clock[start:stop] + 300 * 10**9,
            economics=scope["economics"], split_end_exclusive_ns=int(cutoff))
        for key, value in block.items():
            output[key][start:stop] = value
    output["metadata"] = {"schema_version": SCHEMA_VERSION + "_targets", "split": split,
        "physical_source": scope["physical_sources"][split], "rows": rows,
        "horizons_minutes": list(HORIZONS), "cutoff_ns_exclusive": int(cutoff),
        "parent_order_sha256": _digest(output["parent_rows"]),
        "bar_start_clock_sha256": _digest(bar_clock),
        "entry_m1_pointer_sha256": _digest(output["dynamic_entry_m1_row"]),
        "terminal_observed_counts": output["contrast_observed"].sum(axis=0).tolist(),
        "target_array_sha256": {k: _digest(v) for k, v in output.items() if isinstance(v, np.ndarray)},
        "source_quote_binding": source["metadata"]["source_quote_binding"],
        "index_binding": index.binding(), "economics_sha256": scope["economics"]["economics_sha256"],
        "original_observation_reads": 0, "model_forwards": 0, "reference_fits": 0}
    del source, index, fill, frame
    gc.collect()
    return output
