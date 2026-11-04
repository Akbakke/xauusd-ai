"""Publish the registered TRAIN prefix of TARGET_AUDIT_005 without decoding later labels."""
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from gx1.contracts.entry_dynamic_mtf_state_report_v1 import dense_train_prior_from_counts, dense_train_query_counts
from gx1.contracts.entry_gc_mtf_history_v1 import _read_gc_mtf_history_artifact_json as read_bound_json
from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.contracts import unified_exit_market_closure_authority_v1 as closures
from gx1.contracts.immutable_event_authority_v1 import _fsync_directory, _publish_bytes, _publish_file_noreplace
from gx1.features.htf_features import _load_verified_cache_npy


def _require(condition, reason):
    if not condition:
        raise ValueError("DYNAMIC_TRAIN_PREFIX_" + reason)


def _digest(array, dtype):
    return hashlib.sha256(np.asarray(array, dtype=dtype).tobytes()).hexdigest()


def _copy_fit_arrays(manifest, directory, *, prefix_rows, source_rows, cutoff_ns):
    """Verify whole-file SHA as opaque bytes, then copy only bounded NPY values."""
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    parents = {"parent_rows", "issue_ns", "issue_source_index"}
    try:
        def copy(name):
            item = manifest["arrays"][name]
            filename = name + ".npy"
            def identity():
                st = os.stat(filename, dir_fd=fd, follow_symlinks=False)
                return st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns
            initial = identity()
            values = _load_verified_cache_npy(fd, filename, expected_sha256=item["sha256"],
                expected_size_bytes=item["size_bytes"], label="TRAIN_PREFIX/" + name)
            _require(values.dtype.name == item["dtype"] and list(values.shape) == item["shape"], "SOURCE_HEADER:" + name)
            end = prefix_rows if name in parents else source_rows
            _require(0 < end < len(values), "NOT_STRICT_PREFIX:" + name)
            if name in ("validity_ns", "issue_ns"):
                _require(values[end - 1] < cutoff_ns <= values[end], "CLOCK_BOUNDARY:" + name)
            part = np.array(values[:end], copy=True)
            if name in ("validity_ns", "issue_ns"):
                _require(np.all(np.diff(part) > 0) and np.all(part < cutoff_ns), "NONMONOTONE_CLOCK:" + name)
            _require(identity() == initial, "SOURCE_CHANGED:" + name)
            return part

        # Establish both time boundaries before any label, threshold or mask is decoded.
        arrays = {key: copy(key) for key in ("validity_ns", "issue_ns", "maturity_ns")}
        _require(np.array_equal(arrays["maturity_ns"], np.broadcast_to(
            arrays["validity_ns"][:, None], (source_rows, len(target.TIMEFRAMES)))), "PREFIX_MATURITY")
        for name in target.SERIES_ARRAY_DTYPES:
            if name not in arrays:
                arrays[name] = copy(name)
    finally:
        os.close(fd)
    return arrays


def _prefix_atr_surfaces(arrays):
    """Recover the exact 005 compressed ATR surface selected by the prefix minutes."""
    surfaces = {}
    for col, tf in enumerate(target.TIMEFRAMES):
        timestamps = arrays["atr_available_ns"][:, col]
        threshold = arrays["threshold"][:, col]
        observed = timestamps > 0
        times, first = np.unique(timestamps[observed], return_index=True)
        _require(len(times) > 0, "NO_ATR:" + tf)
        atr = threshold[observed][first] / target.MINIMUM_MOVE_ATR
        indices = np.searchsorted(times, arrays["validity_ns"], side="right") - 1
        reconstructed = np.where(indices >= 0, target.MINIMUM_MOVE_ATR * atr[np.maximum(0, indices)], 0.0)
        _require(np.array_equal(np.where(indices >= 0, times[np.maximum(0, indices)], 0), timestamps)
                 and np.array_equal(reconstructed.view(np.uint64), threshold.view(np.uint64)), "ATR_ROUNDTRIP:" + tf)
        surfaces[tf] = {"rows": len(times), "available_ns_sha256": _digest(times, "<i8"), "atr_sha256": _digest(atr, "<f8")}
    return surfaces


def _calendar_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _prefix_closures(source, metadata, starts, directory):
    """Build an independently valid closure authority, preserving every prefix classification."""
    original = source["calendar_binding"]
    authority = closures.require_market_closure_authority(read_bound_json(original["closure_authority"]),
        expected_m1_source_sha256=metadata["source_refs"]["m1"]["sha256"],
        expected_m1_clock_sha256=metadata["supplied_clock_sha256"])
    schedule = closures.require_exact_market_schedule(read_bound_json(original["exact_schedule"]))
    evidence = read_bound_json(original["source_reference"])
    _require(metadata["closure_binding"] == {"authority_sha256": authority["artifact_sha256"],
             "schedule_sha256": authority["schedule_sha256"]} and authority["m1_row_count"] == metadata["rows"]
             and schedule["schedule_sha256"] == authority["schedule_sha256"]
             and authority["schedule_file_sha256"] == original["exact_schedule"]["sha256"]
             and schedule["source_reference_sha256"] == original["source_reference"]["sha256"]
             and evidence["m1_source"] == metadata["source_refs"]["m1"]
             and evidence["m1_clock_sha256"] == metadata["supplied_clock_sha256"]
             and evidence["calendar"] == original["published_calendar"] and evidence["test_data_used"] is False,
             "SOURCE_CLOSURE_BINDING")
    clock = pd.to_datetime(starts, utc=True)
    retained = [r for r in authority["intervals"] if r["gap_after_m1_row"] < len(clock) - 1]
    certified = [r for r in evidence["certified_gaps"] if r["gap_after_slice_row"] < len(clock) - 1]
    known = [r for r in retained if r["successor_across_gap_allowed"]]
    _require([(r["gap_after_slice_row"], r["source_event_id"]) for r in certified] ==
             [(r["gap_after_m1_row"], r["source_event_id"]) for r in known], "CERTIFIED_GAPS")
    payloads = {}
    def bind(name, value):
        data = _calendar_bytes(value)
        payloads[name] = data
        return {"path": str(directory / name), "sha256": hashlib.sha256(data).hexdigest()}

    prefix_source = {**evidence, "source_series": source["series"], "m1_clock_sha256": _digest(starts, "<i8"),
                     "certified_gaps": certified, "uncovered_gap_count": len(retained) - len(known),
                     "upstream_source_reference": original["source_reference"]}
    source_ref = bind("SOURCE_REFERENCE.json", prefix_source)
    narrow = {k: v for k, v in schedule.items() if k != "schedule_sha256"}
    narrow.update(coverage_start_utc=clock[0].isoformat(),
                  coverage_end_utc_exclusive=(clock[-1] + pd.Timedelta(minutes=1)).isoformat(),
                  source_reference_sha256=source_ref["sha256"],
                  intervals=[r for r in schedule["intervals"] if pd.Timestamp(r["end_utc_exclusive"]) <= clock[-1]])
    narrow = closures.seal_exact_market_schedule(narrow)
    schedule_ref = bind("EXACT_SCHEDULE.json", narrow)
    scoped = closures.build_market_closure_authority(m1_times=clock,
        m1_source_path=Path(authority["m1_source_path"]), m1_source_sha256=authority["m1_source_sha256"],
        m1_source_manifest_path=Path(authority["m1_source_manifest_path"]),
        m1_source_manifest_sha256=authority["m1_source_manifest_sha256"], exact_schedule=narrow,
        exact_schedule_path=Path(schedule_ref["path"]), exact_schedule_file_sha256=schedule_ref["sha256"])
    # Only hashes that incorporate the old schedule are expected to differ.
    def classification(record):
        return {k: v for k, v in record.items() if k not in ("interval_sha256", "schedule_sha256")}
    _require([classification(r) for r in scoped["intervals"]] == [classification(r) for r in retained],
             "PREFIX_GAP_CLASSIFICATION")
    authority_ref = bind("CLOSURE_AUTHORITY.json", scoped)
    calendar = {"published_calendar": original["published_calendar"], "source_reference": source_ref,
                "exact_schedule": schedule_ref, "closure_authority": authority_ref}
    binding = {"authority_sha256": scoped["artifact_sha256"], "schedule_sha256": narrow["schedule_sha256"]}
    return binding, calendar, payloads


def _publish_calendar(directory, payloads):
    stage = Path(tempfile.mkdtemp(prefix="." + directory.name + ".staging.", dir=directory.parent))
    for name, data in payloads.items():
        path = stage / name
        _publish_bytes(path, data)
        path.chmod(0o444)
        _require(path.read_bytes() == data, "CALENDAR_WRITE:" + name)
    _fsync_directory(stage)
    _publish_file_noreplace(stage, directory)
    _fsync_directory(directory.parent)


def _clock_support(series):
    """Exact endpoint, adjacency and D1 marginal support without a dense lead grid."""
    axis, issues = series.arrays["validity_ns"], series.arrays["issue_ns"]
    lead = target.LOCAL_WINDOW_M1_BARS["D1"]
    first, last = issues + target.MINUTE_NS, issues + lead * target.MINUTE_NS
    mature = last < series.metadata["cutoff_ns"]
    first_row = np.searchsorted(axis, first)
    last_row = np.searchsorted(axis, last)
    endpoint = (last_row < len(axis)) & (axis[np.minimum(last_row, len(axis) - 1)] == last) & mature
    start = (first_row < len(axis)) & (axis[np.minimum(first_row, len(axis) - 1)] == first)
    consecutive = start & endpoint & (last_row - first_row + 1 == lead)
    return {"parents": len(issues), "lead_minutes": lead, "mature_calendar_endpoints": int(mature.sum()),
            "exact_m1_endpoints": int(endpoint.sum()), "complete_wall_minute_source_prefixes": int(consecutive.sum()),
            "observed_D1_endpoints": int(series.arrays["observed"][last_row[endpoint], target.TIMEFRAMES.index("D1")].sum())}


def materialize_train_prefix_targets(*, target_result_ref, prefix_rows, cutoff_ns, output_dir):
    """Return a physical SERIES binding, its exact TRAIN counts, prior and clock support."""
    require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    repo = Path(__file__).resolve().parents[2]
    policy = json.loads((repo / "NEXT_RUN_POLICY.json").read_text())
    gate = policy["current_work"]["dynamic_mtf_train_gates_20261104"]
    review_ref = gate["target_review"]
    _require(review_ref["path"] == "docs/DYNAMIC_MTF_TRAIN_TARGET_RESULT_20261104.json", "REVIEW_PATH")
    review = read_bound_json({**review_ref, "path": str(repo / review_ref["path"])})
    _require(policy["training_enabled"] is False and gate["training_authorized"] is False
             and review["source_receipt"] == target_result_ref == policy["dynamic_mtf_state_target_audit_20261102_005"]["result"]
             and type(prefix_rows) is int and prefix_rows == review["parents"] == gate["fit_parents"]
             and type(cutoff_ns) is int and cutoff_ns == pd.Timestamp(review["train_fit_cutoff_exclusive"]).value
             == pd.Timestamp(gate["fit_cutoff_utc_exclusive"]).value, "FIT_BINDING")
    source = read_bound_json(target_result_ref)
    manifest_ref = source["series"]
    _require(manifest_ref == gate["oof_geometry"]["cross_checked_series"]
             and manifest_ref["sha256"] == review["series_sha256"]
             and cutoff_ns < source["fold"]["train_cutoff_ns"] < source["fold"]["series_cutoff_ns"], "SOURCE_SERIES")
    manifest = target.read_dynamic_mtf_state_series_metadata(manifest_ref,
        expected_source_refs=source["target_sources"], expected_cutoff_ns=source["fold"]["series_cutoff_ns"])
    origin = manifest["metadata"]
    n = review["m1_prefix_rows_including_context"]
    arrays = _copy_fit_arrays(manifest, Path(manifest_ref["path"]).parent, prefix_rows=prefix_rows,
                              source_rows=n, cutoff_ns=cutoff_ns)
    fit = gate["oof_geometry"]["blocks"][0]
    _require(fit["role"] == "fit" and fit["parent_range"] == [0, prefix_rows]
             and np.array_equal(arrays["parent_rows"], np.arange(prefix_rows))
             and _digest(arrays["parent_rows"], "<i8") == fit["parent_rows_sha256_le_i64"]
             and _digest(arrays["issue_ns"], "<i8") == fit["issue_ns_sha256_le_i64"]
             and arrays["source_row"][0] == origin["source_row_offset"], "PHYSICAL_AXES")
    starts = arrays["validity_ns"] - target.MINUTE_NS
    output_dir = Path(output_dir)
    calendar_dir = output_dir.parent / "FIT_CALENDAR"
    _require(output_dir.is_absolute() and output_dir.resolve() == output_dir and output_dir.parent.is_dir()
             and output_dir != calendar_dir and not output_dir.exists() and not os.path.lexists(calendar_dir),
             "OUTPUT_DIRECTORY")
    closure_binding, calendar, calendar_files = _prefix_closures(source, origin, starts, calendar_dir)
    metadata = {k: origin[k] for k in ("schema_version", "definition", "source_refs", "source_row_offset",
                                        "gap_count_scope", "target_rule_locked", "test_data_used")}
    metadata.update(cutoff_ns=cutoff_ns, rows=n, supplied_source_rows=n, parents=prefix_rows,
                    supplied_clock_sha256=_digest(starts, "<i8"), atr_surfaces=_prefix_atr_surfaces(arrays),
                    closure_binding=closure_binding,
                    prefix_lineage={"source_series": manifest_ref, "source_target_result": target_result_ref,
                                    "calendar_binding": calendar, "fit_review": review_ref, "source_row_slice": [0, n], "parent_slice": [0, prefix_rows]})
    series = target._series(arrays, metadata)
    leads = target.default_trial_lead_minutes()
    counts = dense_train_query_counts(series, prefix_rows=prefix_rows, cutoff_ns=cutoff_ns, lead_minutes=leads)
    prior = dense_train_prior_from_counts(counts, lead_minutes=leads)
    support = _clock_support(series)
    _require(support["mature_calendar_endpoints"] == fit["parents_with_full_1440min_before_cutoff_calendar_only"]
             and require_capped_lock_ancestry() == lock, "CLOCK_OR_LOCK")
    _publish_calendar(calendar_dir, calendar_files)
    binding = target.save_dynamic_mtf_state_series(series, output_dir)
    return {"series": binding, "counts": counts, "prior": prior, "clock_support": support, "calendar_binding": calendar}
