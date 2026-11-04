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
            if name == "validity_ns" and end is None:
                end = int(values.searchsorted(cutoff_ns, side="left"))
            _require(0 < end < len(values), "NOT_STRICT_PREFIX:" + name)
            if name in ("validity_ns", "issue_ns"):
                _require(values[end - 1] < cutoff_ns <= values[end], "CLOCK_BOUNDARY:" + name)
            part = np.array(values[:end], copy=True)
            if name in ("validity_ns", "issue_ns"):
                _require(np.all(np.diff(part) > 0) and np.all(part < cutoff_ns), "NONMONOTONE_CLOCK:" + name)
            _require(identity() == initial, "SOURCE_CHANGED:" + name)
            return part

        # Establish both time boundaries before any label, threshold or mask is decoded.
        arrays = {"validity_ns": copy("validity_ns")}
        source_rows = len(arrays["validity_ns"])
        arrays.update({key: copy(key) for key in ("issue_ns", "maturity_ns")})
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


def _prefix_closures(source, metadata, starts, directory, *, first_source_row=0):
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
    def local_records(records, key):
        return [{**r, key: r[key] - first_source_row} for r in records
                if first_source_row <= r[key] < first_source_row + len(clock) - 1]
    retained = local_records(authority["intervals"], "gap_after_m1_row")
    certified = local_records(evidence["certified_gaps"], "gap_after_slice_row")
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
                  intervals=[r for r in schedule["intervals"] if (first_source_row == 0 or pd.Timestamp(r["start_utc"]) >= clock[0])
                             and pd.Timestamp(r["end_utc_exclusive"]) <= clock[-1]])
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


def _d1_interval_event_support(series):
    """Count observed at-risk close intervals before censoring, inclusive of each first event."""
    a = series.arrays
    axis = a["validity_ns"]
    col = target.TIMEFRAMES.index("D1")
    horizon = target.LOCAL_WINDOW_M1_BARS["D1"]
    known = a["observed"][:, col] & a["path_observed"][:, col]
    state = a["state"][:, col]
    positions = a["issue_source_index"]
    starts = positions[(positions >= 0) & known[np.maximum(positions, 0)]]
    n = len(axis)
    # Last consecutive known close reachable from each start, before missing time or masks.
    cut = np.r_[~((np.diff(axis) == target.MINUTE_NS) & known[1:]), True]
    run_end = np.minimum.accumulate(np.where(cut, np.arange(n), n)[::-1])[::-1]
    ends = np.minimum(run_end[starts], starts + horizon)
    exits = np.full(starts.shape, n, dtype=np.int64)
    returns = np.full(starts.shape, n, dtype=np.int64)
    for label in (target.DOWN, target.FLAT, target.UP):
        same = state[starts] == label
        other_rows = np.r_[np.flatnonzero(known & (state != label)), n]
        same_rows = np.r_[np.flatnonzero(known & (state == label)), n]
        exits[same] = other_rows[np.searchsorted(other_rows, starts[same], side="right")]
        # A missing exit remains at n, which also represents missing return.
        returns[same] = same_rows[np.searchsorted(same_rows, np.minimum(exits[same] + 1, n))]

    def counts(event_rows):
        occurred = event_rows <= ends
        risk_end = np.minimum(event_rows, ends)
        risk = risk_end - starts
        event_count = int(occurred.sum())
        minutes = axis[event_rows[occurred]]
        nonevent = risk - occurred.astype(np.int64)
        # Union of observed non-event minutes. Several overlapping issue forecasts
        # can cover one physical minute and may pose different event questions.
        delimiters = (np.bincount(starts + 1, minlength=n + 1)
                      - np.bincount(starts + nonevent + 1, minlength=n + 1))
        exposure = np.cumsum(delimiters)[:-1]
        _require(int(exposure.sum()) == int(nonevent.sum()), "D1_EXPOSURE")
        return {"at_risk_issue_minutes": int(risk.sum()), "event_parent_issues": event_count,
                "observed_nonevent_issue_minutes": int(nonevent.sum()),
                "distinct_exposed_nonevent_minutes": int(np.count_nonzero(exposure)),
                "distinct_event_minutes": int(np.unique(minutes).size),
                "distinct_event_utc_days": int(np.unique(minutes.astype("datetime64[ns]").astype("datetime64[D]")).size),
                "censored_before_horizon_without_event": int(np.count_nonzero(~occurred & (risk < horizon))),
                "confirmed_full_horizon_without_event": int(np.count_nonzero(~occurred & (risk == horizon)))}

    return {"timeframe": "D1", "lead_minutes": horizon, "parents": len(positions),
            "known_state_and_path_at_issue": len(starts), "unknown_start": len(positions) - len(starts),
            "known_start_no_observed_next_interval": int(np.count_nonzero(starts == ends)),
            "first_exit": counts(exits), "first_exit_then_return": counts(returns)}


def materialize_train_prefix_targets(*, target_result_ref, prefix_rows, cutoff_ns, output_dir,
                                     admission_stage="dynamic_mtf_train_prefix_data_20261104_001"):
    """Return a physical SERIES binding, its exact TRAIN counts, prior and clock support."""
    require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    repo = Path(__file__).resolve().parents[2]
    policy = json.loads((repo / "NEXT_RUN_POLICY.json").read_text())
    december = admission_stage == "dynamic_mtf_train_prefix_dec01_20261104_001"
    _require(december or admission_stage == "dynamic_mtf_train_prefix_data_20261104_001", "UNKNOWN_STAGE")
    stage = policy[admission_stage]
    _require(policy["training_enabled"] is False and stage["training_enabled"] is False
             and stage["execution_authorized"] is True and stage["fit_target_view_authorized"] is True
             and stage["claimed_invocations"] == stage["completed_invocations"] == 0
             and stage["max_invocations"] == 1 and stage["relaunch_allowed"] is False
             and Path(output_dir) == Path(stage["run_root"]) / "FIT_TARGETS"
             and type(prefix_rows) is int and prefix_rows == stage["prefix_rows"]
             and type(cutoff_ns) is int and cutoff_ns == stage["fit_cutoff_ns"]
             and target_result_ref == stage["target_result"] == policy["dynamic_mtf_state_target_audit_20261102_005"]["result"],
             "FIT_ADMISSION")
    source = read_bound_json(target_result_ref)
    manifest_ref = source["series"]
    _require(manifest_ref == source["target_manifest"] and
             cutoff_ns < source["fold"]["train_cutoff_ns"] < source["fold"]["series_cutoff_ns"], "SOURCE_SERIES")
    if december:
        fit = stage["geometry"]["fit"]
        held = stage["geometry"]["held_out"]
        _require(cutoff_ns == pd.Timestamp("2024-12-01T00:00:00Z").value and prefix_rows == 41866
                 and stage["held_out_rows"] == held["parents"] == 5736 and held["role"] == "held_out"
                 and held["parent_range"] == [prefix_rows, source["fold"]["train_parents"]]
                 and source["fold"]["train_parents"] == prefix_rows + held["parents"], "DECEMBER_GEOMETRY")
        n = None  # Measured from the bound validity clock before any label is accessed.
        lineage = {"admission_stage": admission_stage, "specification": stage["specification"]}
    else:
        gate = policy["current_work"]["dynamic_mtf_train_gates_20261104"]
        review_ref = gate["target_review"]
        _require(review_ref["path"] == "docs/DYNAMIC_MTF_TRAIN_TARGET_RESULT_20261104.json", "REVIEW_PATH")
        review = read_bound_json({**review_ref, "path": str(repo / review_ref["path"])})
        _require(gate["training_authorized"] is False and review["source_receipt"] == target_result_ref
                 and prefix_rows == review["parents"] == gate["fit_parents"]
                 and cutoff_ns == pd.Timestamp(review["train_fit_cutoff_exclusive"]).value
                 == pd.Timestamp(gate["fit_cutoff_utc_exclusive"]).value
                 and manifest_ref == gate["oof_geometry"]["cross_checked_series"]
                 and manifest_ref["sha256"] == review["series_sha256"], "FIT_BINDING")
        n = review["m1_prefix_rows_including_context"]
        fit = gate["oof_geometry"]["blocks"][0]
        lineage = {"fit_review": review_ref}
    manifest = target.read_dynamic_mtf_state_series_metadata(manifest_ref,
        expected_source_refs=source["target_sources"], expected_cutoff_ns=source["fold"]["series_cutoff_ns"])
    origin = manifest["metadata"]
    arrays = _copy_fit_arrays(manifest, Path(manifest_ref["path"]).parent, prefix_rows=prefix_rows,
                              source_rows=n, cutoff_ns=cutoff_ns)
    n = len(arrays["validity_ns"])
    _require(fit["role"] == "fit" and fit["parent_range"] == [0, prefix_rows] and fit["parents"] == prefix_rows
             and np.array_equal(arrays["parent_rows"], np.arange(prefix_rows))
             and _digest(arrays["parent_rows"], "<i8") == fit["parent_rows_sha256_le_i64"]
             and _digest(arrays["issue_ns"], "<i8") == fit["issue_ns_sha256_le_i64"]
             and _digest(arrays["issue_ns"] - target.LOCAL_WINDOW_M1_BARS["M5"] * target.MINUTE_NS, "<i8")
                 == fit["bar_start_ns_sha256_le_i64"]
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
                                    "calendar_binding": calendar, **lineage,
                                    "source_row_slice": [0, n], "parent_slice": [0, prefix_rows]})
    series = target._series(arrays, metadata)
    leads = target.default_trial_lead_minutes()
    counts = dense_train_query_counts(series, prefix_rows=prefix_rows, cutoff_ns=cutoff_ns, lead_minutes=leads)
    prior = dense_train_prior_from_counts(counts, lead_minutes=leads)
    support = _clock_support(series)
    _require(support["mature_calendar_endpoints"] == fit["parents_with_full_1440min_before_cutoff_calendar_only"]
             and require_capped_lock_ancestry() == lock, "CLOCK_OR_LOCK")
    result = {"counts": counts, "prior": prior, "clock_support": support, "calendar_binding": calendar}
    if december:
        result["d1_interval_event_support"] = _d1_interval_event_support(series)
    _publish_calendar(calendar_dir, calendar_files)
    result["series"] = target.save_dynamic_mtf_state_series(series, output_dir)
    return result
