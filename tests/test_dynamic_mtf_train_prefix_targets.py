"""Physical fit scope and compressed ATR identity on a tiny clock with a gap."""
from pathlib import Path

import numpy as np
import pytest

from gx1.audit import dynamic_mtf_train_prefix_targets as view
from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
from tests.test_entry_dynamic_mtf_state_targets import START, atr_fixture, build, frame_from_closes


def test_copy_fences_labels_and_excludes_later_atr_before_publication(tmp_path, monkeypatch):
    minute = target.MINUTE_NS
    frame = frame_from_closes([100.0] * 19, minutes=[*range(5), *range(6, 20)])
    cutoff = int(START + 15 * minute)
    origin_cutoff = int(START + 25 * minute)
    times = np.array([START, cutoff], dtype=np.int64)
    atr = atr_fixture(origin_cutoff, values=[2.0, 7.0], available=times)
    source = build(frame, cutoff=origin_cutoff, atr=atr)
    source = target.bind_dynamic_mtf_state_parents(source, parent_rows=np.arange(4, dtype=np.int64),
        issue_ns=START + np.array([5, 10, 15, 20], np.int64) * minute)
    binding = target.save_dynamic_mtf_state_series(source, tmp_path / "ORIGINAL")
    manifest = view.read_bound_json(binding)
    end = int(np.searchsorted(source.arrays["validity_ns"], cutoff))
    original_loader = view._load_verified_cache_npy
    fetched = []

    class Bounded:
        def __init__(self, array, name):
            self.array, self.name = array, name
            self.shape, self.dtype = array.shape, array.dtype

        def __len__(self):
            return len(self.array)

        def __getitem__(self, key):
            limit = 2 if self.name in ("parent_rows", "issue_ns", "issue_source_index") else end
            if isinstance(key, slice):
                assert key.stop is not None and key.stop <= limit, "label read beyond fit"
            else:
                assert self.name in ("validity_ns", "issue_ns") and key <= limit, "later value read"
            return self.array[key]

    def checked_load(fd, filename, **kwargs):
        name = Path(filename).stem
        fetched.append(name)
        return Bounded(original_loader(fd, filename, **kwargs), name)

    monkeypatch.setattr(view, "_load_verified_cache_npy", checked_load)
    with pytest.raises(ValueError, match="CLOCK_BOUNDARY:validity_ns"):
        view._copy_fit_arrays(manifest, tmp_path / "ORIGINAL", prefix_rows=2, source_rows=end + 1, cutoff_ns=cutoff)
    assert fetched == ["validity_ns"]
    arrays = view._copy_fit_arrays(manifest, tmp_path / "ORIGINAL", prefix_rows=2, source_rows=end, cutoff_ns=cutoff)
    assert set(arrays) == set(target.SERIES_ARRAY_DTYPES)
    assert all(not isinstance(a, np.memmap) for a in arrays.values())
    assert all(len(arrays[k]) == (2 if k in ("parent_rows", "issue_ns", "issue_source_index") else end) for k in arrays)
    surface = view._prefix_atr_surfaces(arrays)
    prefix_atr = atr_fixture(cutoff, values=[2.0], available=times[:1])
    direct = build(frame.loc[frame.time.astype("int64") + minute < cutoff], cutoff=cutoff, atr=prefix_atr)
    assert surface == direct.metadata["atr_surfaces"] != source.metadata["atr_surfaces"]
    for k in arrays:
        if k not in ("parent_rows", "issue_ns", "issue_source_index"):
            np.testing.assert_array_equal(arrays[k], direct.arrays[k])


def test_mature_day_endpoint_and_unbroken_day_clock_are_separate():
    minute = target.MINUTE_NS
    axis = START + np.delete(np.arange(1, 1451), 5) * minute
    issues = START + np.array([0, 5, 10, 15]) * minute
    series = target.DynamicMtfStateSeries({"validity_ns": axis, "issue_ns": issues,
        "observed": np.ones((len(axis), len(target.TIMEFRAMES)), dtype=bool)}, {"cutoff_ns": int(axis[-1] + minute)})
    got = view._clock_support(series)
    assert got["mature_calendar_endpoints"] == got["exact_m1_endpoints"] == got["observed_D1_endpoints"] == 3
    assert got["complete_wall_minute_source_prefixes"] == 1


def test_prefix_calendar_opens_with_canonical_hashes_and_drops_later_gaps(tmp_path):
    import hashlib
    import pandas as pd
    from gx1.contracts import unified_exit_market_closure_authority_v1 as closure

    minute = target.MINUTE_NS
    starts = START + np.array([0, 1, 5, 6, 11, 12], np.int64) * minute
    times = pd.to_datetime(starts, utc=True)
    raw = {"path": "/read-only/source.parquet", "sha256": "1" * 64}
    published = {"path": "/read-only/published.json", "sha256": "2" * 64}
    def put(name, contents):
        p = tmp_path / name
        payload = view._calendar_bytes(contents)
        p.write_bytes(payload)
        return {"path": str(p), "sha256": hashlib.sha256(payload).hexdigest()}
    source = put("source.json", {"m1_source": raw, "m1_clock_sha256": closure.m1_clock_sha256(times),
        "calendar": published, "test_data_used": False, "certified_gaps": [
            {"gap_after_slice_row": i, "source_event_id": str(i)} for i in (1, 3)]})
    schedule = closure.seal_exact_market_schedule({
        "schema_version": closure.MARKET_CLOSURE_SCHEDULE_SCHEMA_VERSION, "decision": "PASS", "instrument": "XAU_USD",
        "timeframe": "M1", "coverage_start_utc": times[0].isoformat(),
        "coverage_end_utc_exclusive": (times[-1] + pd.Timedelta(minutes=1)).isoformat(),
        "interval_semantics": "left_closed_right_open_utc", "test_data_used": False,
        "source_method": "externally_sourced_exact_xau_utc_closure_intervals_v1", "source_reference_sha256": source["sha256"],
        "intervals": [{"kind": "holiday", "source_event_id": str(i),
            "start_utc": (times[i] + pd.Timedelta(minutes=1)).isoformat(),
            "end_utc_exclusive": times[i+1].isoformat()} for i in (1, 3)]})
    sched_ref = put("schedule.json", schedule)
    authority = closure.build_market_closure_authority(m1_times=times, m1_source_path=Path(raw["path"]),
        m1_source_sha256=raw["sha256"], m1_source_manifest_path=Path("/read-only/m1-manifest.json"),
        m1_source_manifest_sha256="3"*64, exact_schedule=schedule, exact_schedule_path=Path(sched_ref["path"]),
        exact_schedule_file_sha256=sched_ref["sha256"])
    original = {"source_reference": source, "published_calendar": published, "exact_schedule": sched_ref,
                "closure_authority": put("authority.json", authority)}
    upstream = {"authority_sha256": authority["artifact_sha256"], "schedule_sha256": schedule["schedule_sha256"]}
    metadata = {"source_refs": {"m1": raw}, "supplied_clock_sha256": authority["m1_clock_sha256"],
                "closure_binding": upstream, "rows": len(starts)}
    directory = tmp_path / "FIT_CALENDAR"
    bound, references, payloads = view._prefix_closures({"calendar_binding": original, "series": sched_ref},
                                                        metadata, starts[:4], directory)
    view._publish_calendar(directory, payloads)
    calendar = closure.require_exact_market_schedule(view.read_bound_json(references["exact_schedule"]))
    proof = closure.require_market_closure_authority(view.read_bound_json(references["closure_authority"]),
        expected_m1_source_sha256=raw["sha256"], expected_m1_clock_sha256=closure.m1_clock_sha256(times[:4]))
    assert bound == {"authority_sha256": proof["artifact_sha256"], "schedule_sha256": calendar["schedule_sha256"]}
    assert set(bound) == set(upstream) and all(bound[k] != upstream[k] for k in bound)
    assert [r["gap_after_m1_row"] for r in proof["intervals"]] == [1]
    assert [r["source_event_id"] for r in view.read_bound_json(references["source_reference"])["certified_gaps"]] == ["1"]
    assert proof["schedule_file_sha256"] == references["exact_schedule"]["sha256"]
