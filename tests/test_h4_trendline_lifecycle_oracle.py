"""Focused H4 clock, comparison, and one-use claim guards."""
import json

import numpy as np
import pytest

from gx1.audit.verify_h4_trendline_lifecycle_oracle_v1 import (
    H4_NS, NAMES, PREFIX_ROWS, REPO, STAGE, TRAIN_END_NS, TRAIN_FIRST,
    TRAIN_LAST, TRAIN_ROWS, TRAIN_START_NS, WITNESSES,
    align_train_rows, align_witnesses, compare_train_rows, read_claim,
)


def axes():
    cache = np.empty(PREFIX_ROWS + 1, dtype=np.int64)
    cache[:TRAIN_FIRST] = np.arange(
        TRAIN_START_NS - H4_NS - TRAIN_FIRST,
        TRAIN_START_NS - H4_NS, dtype=np.int64)
    points = [(row, timestamp) for row, _, timestamp in WITNESSES]
    points.append((TRAIN_LAST, TRAIN_END_NS - H4_NS - 2 * 3600 * 10**9))
    cache[TRAIN_FIRST] = points[0][1]
    for (left, start), (right, end) in zip(points, points[1:]):
        cache[left:right + 1] = np.linspace(start, end, right - left + 1, dtype=np.int64)
    cache[TRAIN_LAST + 1:] = np.arange(
        TRAIN_END_NS - H4_NS,
        TRAIN_END_NS - H4_NS + len(cache) - TRAIN_LAST - 1, dtype=np.int64)
    native = np.concatenate((
        np.arange(cache[0] - 1610, cache[0], dtype=np.int64), cache))
    records = [{"cache_row": row, "native_bar_index": pos, "timestamp_ns": timestamp}
               for row, pos, timestamp in WITNESSES]
    return cache, native, records


def test_all_1044_closed_h4_rows_and_eight_witnesses_align():
    cache, native, records = axes()
    rows, positions, timestamps = align_train_rows(cache, native)
    assert len(rows) == 1044
    assert np.array_equal(rows, TRAIN_ROWS)
    assert positions.tolist() == [row + 1610 for row in TRAIN_ROWS]
    assert timestamps[0] + H4_NS >= TRAIN_START_NS
    assert timestamps[-1] + H4_NS < TRAIN_END_NS
    assert align_witnesses(cache, native, records) == [x[1] for x in WITNESSES]
    cache[WITNESSES[1][0]] += 1
    with pytest.raises(RuntimeError, match="TRAIN_NATIVE_JOIN"):
        align_train_rows(cache, native)


def arrays():
    rows = TRAIN_ROWS.copy()
    times = TRAIN_START_NS + np.arange(len(rows), dtype=np.int64) * H4_NS
    oracle = np.zeros((len(rows), len(NAMES)), np.float64)
    cache = oracle.astype(np.float32)
    return rows, times, oracle, cache


def test_all_31_fields_and_19_lifecycle_fields_require_comparison():
    rows, times, oracle, cache = arrays()
    signed = NAMES.index("geomline_bars_since_break_signed")
    oracle[:, signed] = np.nan
    cache = oracle.astype(np.float32)
    report = compare_train_rows(oracle, cache, rows, times)
    assert (report["compared_cells"], report["lifecycle_cells"],
            report["mismatched_cells"]) == (32364, 19836, 0)
    assert report["all_train_window_cells_match"] is True
    assert "geomline_break_up" in report["unexercised_lifecycle_fields"]
    cache[0, signed] = 0
    cache[8, NAMES.index("geomline_break_up")] = 1
    report = compare_train_rows(oracle, cache, rows, times)
    assert report["mismatched_cells"] == report["lifecycle_mismatched_cells"] == 2
    assert report["anchor_mismatched_cells"] == 1


def test_discrete_exact_and_float_tolerance():
    rows, times, oracle, cache = arrays()
    slope = NAMES.index("geomline_above_slope_atr_per_bar")
    count = NAMES.index("geomline_above_touch_count")
    oracle[0, slope] = 0.5
    cache[0, slope] = np.float32(0.5 + 1e-6)
    assert compare_train_rows(oracle, cache, rows, times)["mismatched_cells"] == 0
    cache[0, slope] = np.float32(0.5 + 1e-4)
    cache[0, count] = np.float32(1e-7)
    report = compare_train_rows(oracle, cache, rows, times)
    assert report["mismatched_cells"] == 2
    assert report["lifecycle_mismatched_cells"] == 1
    with pytest.raises(RuntimeError, match="COMPARE_SHAPE"):
        compare_train_rows(oracle[:, :-1], cache[:, :-1], rows, times)


def test_claim_only_root_binds_plan_policy_worker_and_head(tmp_path):
    output = tmp_path / "one_use"
    output.mkdir()
    plan = {"path": "/plan.json", "sha256": "a" * 64}
    worker = {"path": "/worker.py", "sha256": "b" * 64}
    claim = {
        "schema_version": "gx1_h4_trendline_lifecycle_physical_claim_v1",
        "stage": STAGE, "plan": plan,
        "policy": {"path": str(REPO / "NEXT_RUN_POLICY.json"), "sha256": "c" * 64},
        "source_head": "d" * 40, "worker": worker,
        "source_only_oracle": True, "test_data_used": False,
    }
    (output / "CLAIM.json").write_text(json.dumps(claim))
    observed, ref, _ = read_claim(output, plan, worker, "d" * 40)
    assert observed == claim and ref["sha256"]
    with pytest.raises(RuntimeError, match="CLAIM_BINDING"):
        read_claim(output, plan, worker, "e" * 40)
    (output / "RESULT.json").write_text("{}")
    with pytest.raises(RuntimeError, match="CLAIM_ONLY_ROOT"):
        read_claim(output, plan, worker, "d" * 40)


def synthetic_preflight(monkeypatch):
    """Bind small synthetic JSON receipts; fail if a claim or raw payload is read."""
    from pathlib import Path
    from gx1.audit import verify_h4_trendline_lifecycle_oracle_v1 as h4

    plan_ref = {"path": "/synthetic/H4_PLAN.json", "sha256": "1" * 64}
    source_refs = {
        name: {"path": str(h4.REPO / relative), "sha256": "2" * 64}
        for name, relative in h4.SOURCE_PATHS.items()}
    cache_manifest = {"path": "/synthetic/cache/manifest.json", "sha256": "3" * 64}
    native_manifest = {"path": "/synthetic/native/MANIFEST.json", "sha256": "4" * 64}
    feats = {"path": "/synthetic/cache/H4_feats.npy", "sha256": "5" * 64}
    ts = {"path": "/synthetic/cache/H4_ts.npy", "sha256": "6" * 64}
    replay_plan = {"path": "/synthetic/replay/PLAN.json", "sha256": "7" * 64}
    replay_result = {"path": "/synthetic/replay/RESULT.json", "sha256": "8" * 64}
    replay_terminal = {"path": "/synthetic/replay/TERMINAL.json", "sha256": "9" * 64}
    fit_result = {"path": "/synthetic/fit/RESULT.json", "sha256": "a" * 64}
    enriched = {"path": "/synthetic/enriched/manifest.json", "sha256": "b" * 64}
    parts = [
        {"path": f"/synthetic/native/year={year}/part-000.parquet",
         "sha256": f"{year:064x}"}
        for year in range(2009, 2025)]
    old_inputs = {
        "cache_manifest": cache_manifest, "fit_result": fit_result,
        "enriched_manifest": enriched, "native_manifest": native_manifest,
        "native_parts": parts,
        "per_tf": {"H4": {"feats": feats, "ts": ts, "replay_plan": replay_plan,
                          "replay_result": replay_result, "replay_terminal": replay_terminal}},
    }
    plan = {
        "schema_version": "gx1_h4_trendline_lifecycle_physical_plan_v1",
        "stage": h4.STAGE, "status": "PREPARED_NOT_ADMITTED",
        "inputs": h4.OLD_REFS, "sources": source_refs, "scope": h4.SCOPE,
        "limits": h4.LIMITS,
        "output_dir": "/home/andre2/GX1_RUNS/" + h4.STAGE.upper(),
        "test_data_used": False,
    }
    records = [
        {"cache_row": row, "native_bar_index": pos, "timestamp_ns": stamp,
         "producer_cache_float32_bit_parity": True, "source_atr14": 1.0}
        for row, pos, stamp in h4.WITNESSES]
    prior = {
        "plan": h4.OLD_REFS["plan"], "status": "COMPLETE",
        "test_data_used": False, "remaining_lifecycle_fields": 19,
        "semantic_line_selection_qualified": False,
        "full_31_field_independent_numeric_qualification": False,
        "physical_train_witnesses": {
            "M5": [{}] * 11, "M15": [{}] * 11, "H1": [{}] * 10,
            "H4": records, "D1": [{}] * 3},
        "cache_prefix_representation": {"H4": {
            "scanned_rows": h4.PREFIX_ROWS, "valid_rows": h4.PREFIX_ROWS,
            "all_12_train_witness_classes_present": True,
            "all_31_representation_invariants_pass": True,
            "selected_witness_cache_rows": {
                "above": 21464, "below": 21464, "parallel_channel": 21464,
                "touch_above": 21466, "touch_below": 21465,
                "break_up": 21469, "break_down": 21473,
                "retest_hold_up": 21471, "retest_hold_down": 21464,
                "retest_fail_up": 21628, "retest_fail_down": 21474,
                "converging_channel": 21474}}},
    }
    manifest = {
        "v29_registry_constants": {"per_tf_seq_lens": {"H4": h4.SEQ_LEN},
                                   "trendline_band_atr": {"H4": h4.BAND_ATR}},
        "feature_names": [f"unused_{i}" for i in range(159)] + list(h4.NAMES)
                         + [f"tail_{i}" for i in range(63)],
        "tfs": {"H4": {
            "n_bars": 24816, "feature_count": 253,
            "feats_npy": "H4_feats.npy", "feats_npy_sha256": feats["sha256"],
            "ts_npy": "H4_ts.npy", "ts_npy_sha256": ts["sha256"]}},
    }
    replay = {
        "status": "COMPLETE", "input_price_precision": "native_float64_full_history",
        "source_clock_rows": h4.PREFIX_ROWS, "replay_parity": True,
        "all_causal_history_vs_cache": {"mismatched_cells": 0},
        "model_train_period_vs_cache": {
            "mismatched_cells": 0,
            "per_field": [{"both_finite_rows": len(h4.TRAIN_ROWS)}] * 31},
        "window_configuration": {
            "timeframe": "H4", "seq_len": h4.SEQ_LEN, "band_atr": h4.BAND_ATR},
        "test_data_used": False,
    }
    docs = {
        plan_ref["path"]: plan,
        h4.OLD_REFS["plan"]["path"]: {
            "scope": {"witness_end_exclusive_ns": h4.TRAIN_END_NS},
            "test_data_used": False, "inputs": old_inputs},
        h4.OLD_REFS["result"]["path"]: prior,
        h4.OLD_REFS["terminal"]["path"]: {
            "status": "COMPLETE", "exit_code": 0,
            "result": h4.OLD_REFS["result"], "test_data_used": False},
        cache_manifest["path"]: manifest,
        fit_result["path"]: {"fold": {"train_cutoff_ns": h4.TRAIN_END_NS},
                             "test_data_used": False},
        enriched["path"]: {"native_m5_source": {
            "manifest_path": native_manifest["path"],
            "manifest_sha256": native_manifest["sha256"]}},
        native_manifest["path"]: {
            "year_sha256": {f"year={year}": parts[year - 2009]["sha256"]
                            for year in range(2009, 2025)}},
        replay_plan["path"]: {"inputs": {
            "tf_feats": feats, "tf_ts": ts, "cache_manifest": cache_manifest}},
        replay_result["path"]: replay,
        replay_terminal["path"]: {"status": "COMPLETE", "exit_code": 0},
    }
    monkeypatch.setattr(h4, "bound_json", lambda ref, **_kwargs: docs[ref["path"]])
    monkeypatch.setattr(h4, "checked", lambda ref: (Path(ref["path"]), (1, 2, 3, 4)))
    monkeypatch.setattr(h4, "git_state", lambda: ("c" * 40, "work/gx1-current", ""))
    monkeypatch.setattr(h4, "read_claim", lambda *_args: pytest.fail("claim read by preflight"))
    return h4, plan_ref, docs


def test_preflight_reports_ready_without_claim_or_raw_payload(monkeypatch):
    h4, plan_ref, _docs = synthetic_preflight(monkeypatch)
    report = h4.preflight_read_only(plan_ref)
    assert report["status"] == "READY_NO_CLAIM"
    assert report["source_owner_refs_checked"] == len(h4.SOURCE_PATHS)
    assert report["native_parts_manifest_bound"] == 16
    assert report["train_cache_rows"] == 1044
    assert report["anchor_witnesses"] == 8
    assert report["claim_created"] is False
    assert report["raw_native_or_cache_payload_read"] is False


def test_preflight_rejects_wrong_h4_parameters_and_replay(monkeypatch):
    h4, plan_ref, docs = synthetic_preflight(monkeypatch)
    cache_path = "/synthetic/cache/manifest.json"
    docs[cache_path]["v29_registry_constants"]["trendline_band_atr"]["H4"] = 0.5
    with pytest.raises(RuntimeError, match="H4_FEATURE_CONTRACT"):
        h4.preflight_read_only(plan_ref)
    docs[cache_path]["v29_registry_constants"]["trendline_band_atr"]["H4"] = h4.BAND_ATR
    docs["/synthetic/replay/RESULT.json"]["all_causal_history_vs_cache"]["mismatched_cells"] = 1
    with pytest.raises(RuntimeError, match="H4_REPLAY_BINDING"):
        h4.preflight_read_only(plan_ref)
