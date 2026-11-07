"""One-use, TRAIN-only H4 trendline audit against the independent D1 oracle math.

The old conditional receipt supplies witness coordinates, never producer line
identities or state. No model, target selection, TEST, or trade is involved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

from gx1.audit import verify_d1_trendline_lifecycle_physical_v1 as physical_base
from gx1.audit.verify_d1_trendline_lifecycle_oracle_v1 import (
    LIFECYCLE_NAMES, NAMES, oracle_rows,
)

REPO = Path("/home/andre2/src/GX1_CURRENT")
STAGE = "h4_trendline_lifecycle_physical_oracle_20261107_001"
TRAIN_START_NS = 1714521600000000000
TRAIN_END_NS = 1735689600000000000
H4_NS = 4 * 3600 * 10**9
PREFIX_ROWS = 23270
TRAIN_FIRST = 21464
TRAIN_LAST = 22507
TRAIN_ROWS = np.arange(TRAIN_FIRST, TRAIN_LAST + 1, dtype=np.int64)
SEQ_LEN = 96
BAND_ATR = 0.9971390001300923
WITNESSES = (
    (21464, 23074, 1714514400000000000),
    (21465, 23075, 1714528800000000000),
    (21466, 23076, 1714543200000000000),
    (21469, 23079, 1714586400000000000),
    (21471, 23081, 1714615200000000000),
    (21473, 23083, 1714644000000000000),
    (21474, 23084, 1714658400000000000),
    (21628, 23238, 1717740000000000000),
)
OLD_ROOT = Path("/home/andre2/GX1_RUNS/TRENDLINE_CONDITIONAL_PREFIX_20261106_001")
OLD_REFS = {
    "plan": {"path": str(REPO / "configs/research/TRENDLINE_CONDITIONAL_PREFIX_20261106_001.json"),
             "sha256": "bc4469117ef476250e50ed308941b790da67ed30b134b5bf350e100240f17491"},
    "result": {"path": str(OLD_ROOT / "RESULT.json"),
               "sha256": "09be782ab60eaf522ee121254354ef8af33077d6d71a9ee173c779f0b77add3f"},
    "terminal": {"path": str(OLD_ROOT / "TERMINAL.json"),
                 "sha256": "02a100486d68e5afb59df94bbb7c5d8372d901f40cb1d5a3de6c5945c65acd2e"},
}
SOURCE_PATHS = {
    "worker": "gx1/audit/verify_h4_trendline_lifecycle_oracle_v1.py",
    "oracle": "gx1/audit/verify_d1_trendline_lifecycle_oracle_v1.py",
    "physical_base": "gx1/audit/verify_d1_trendline_lifecycle_physical_v1.py",
    "replay_loader": "gx1/scripts/audit_direction_d1_trendline_replay_v1.py",
    "resample": "gx1/features/htf_features.py",
    "atr": "gx1/features/technical_indicators_v1.py",
    "capped_contract": "gx1/contracts/gx1_capped_execution_v1.py",
    "capped_runner": "scripts/gx1_capped_run.sh",
    "publication": "gx1/contracts/immutable_event_authority_v1.py",
}
LIMITS = {"memory_max": "4G", "swap_max": "512M", "wall_seconds": 1800,
          "execution_class": "audit", "model_forwards": 0, "optimizer_steps": 0}
SCOPE = {"timeframe": "H4", "train_start_ns": TRAIN_START_NS,
         "train_end_exclusive_ns": TRAIN_END_NS, "train_cache_row_first": TRAIN_FIRST,
         "train_cache_row_last_inclusive": TRAIN_LAST, "train_cache_row_count": len(TRAIN_ROWS),
         "anchor_cache_rows": [row for row, _, _ in WITNESSES], "test_data_used": False}

regular = physical_base.regular
checked = physical_base.checked
bound_json = physical_base.bound_json
git_state = physical_base.git_state
DISCRETE = physical_base.DISCRETE
FLOAT_ABS_TOL = physical_base.FLOAT_ABS_TOL
FLOAT_REL_TOL = physical_base.FLOAT_REL_TOL


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError("H4_TRENDLINE_PHYSICAL_" + reason)


def read_claim(output: Path, plan_ref: dict, worker_ref: dict,
               source_head: str) -> tuple[dict, dict, tuple[int, int, int, int]]:
    need(output.is_dir() and not output.is_symlink()
         and {entry.name for entry in output.iterdir()} == {"CLAIM.json"},
         "CLAIM_ONLY_ROOT")
    path = output / "CLAIM.json"
    before = regular(path)
    need(before[2] <= 1024 * 1024, "CLAIM_SIZE")
    raw = path.read_bytes()
    need(regular(path) == before, "CLAIM_BOOKEND")
    claim = json.loads(raw)
    need(isinstance(claim, dict) and set(claim) == {
        "schema_version", "stage", "plan", "policy", "source_head",
        "worker", "source_only_oracle", "test_data_used",
    } and claim["schema_version"] == "gx1_h4_trendline_lifecycle_physical_claim_v1"
         and claim["stage"] == STAGE and claim["plan"] == plan_ref
         and claim["worker"] == worker_ref and claim["source_head"] == source_head
         and claim["source_only_oracle"] is True and claim["test_data_used"] is False
         and isinstance(claim["policy"], dict)
         and claim["policy"].get("path") == str(REPO / "NEXT_RUN_POLICY.json"),
         "CLAIM_BINDING")
    return claim, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}, before


def witness_receipt(result: dict) -> list[dict]:
    need(result.get("status") == "COMPLETE" and result.get("test_data_used") is False
         and result.get("remaining_lifecycle_fields") == 19
         and result.get("semantic_line_selection_qualified") is False
         and result.get("full_31_field_independent_numeric_qualification") is False,
         "OLD_RESULT_SCOPE")
    witnesses = result.get("physical_train_witnesses")
    need(isinstance(witnesses, dict) and set(witnesses) == {"M5", "M15", "H1", "H4", "D1"}
         and sum(map(len, witnesses.values())) == 43, "OLD_WITNESS_COUNT")
    records = witnesses["H4"]
    need(len(records) == len(WITNESSES), "H4_WITNESS_COUNT")
    for record, (row, pos, timestamp) in zip(records, WITNESSES):
        need((record["cache_row"], record["native_bar_index"], record["timestamp_ns"])
             == (row, pos, timestamp)
             and record["producer_cache_float32_bit_parity"] is True
             and math.isfinite(record["source_atr14"]) and record["source_atr14"] > 0,
             "H4_WITNESS_IDENTITY")
    scan = result["cache_prefix_representation"]["H4"]
    selected = scan["selected_witness_cache_rows"]
    need(scan["scanned_rows"] == scan["valid_rows"] == PREFIX_ROWS
         and scan["all_12_train_witness_classes_present"] is True
         and scan["all_31_representation_invariants_pass"] is True
         and len(selected) == 12
         and set(selected.values()) == {row for row, _, _ in WITNESSES},
         "H4_PREFIX_RECEIPT")
    return records


def align_train_rows(cache_ts: np.ndarray, native_ns: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map every May-Dec 2024 H4 close to an independently resampled H4 bar."""
    need(cache_ts.dtype == native_ns.dtype == np.int64
         and cache_ts.ndim == native_ns.ndim == 1
         and len(cache_ts) > PREFIX_ROWS and len(native_ns) > WITNESSES[-1][1]
         and np.all(np.diff(cache_ts) > 0) and np.all(np.diff(native_ns) > 0),
         "TRAIN_AXIS_SHAPE")
    labels = np.asarray(cache_ts[:PREFIX_ROWS], dtype=np.int64)
    rows = np.flatnonzero((labels + H4_NS >= TRAIN_START_NS)
                          & (labels + H4_NS < TRAIN_END_NS))
    need(np.array_equal(rows, TRAIN_ROWS)
         and int(cache_ts[TRAIN_LAST + 1]) + H4_NS >= TRAIN_END_NS,
         "TRAIN_ROW_RANGE")
    timestamps = labels[rows]
    positions = np.searchsorted(native_ns, timestamps)
    need(int(positions[-1]) < len(native_ns)
         and np.array_equal(native_ns[positions], timestamps)
         and np.all(np.diff(positions) > 0), "TRAIN_NATIVE_JOIN")
    return rows, positions, timestamps


def align_witnesses(cache_ts: np.ndarray, native_ns: np.ndarray,
                    records: list[dict]) -> list[int]:
    positions = []
    for record, (row, pos, timestamp) in zip(records, WITNESSES):
        need(int(cache_ts[row]) == timestamp and int(native_ns[pos]) == timestamp
             and int(np.searchsorted(native_ns, timestamp)) == pos
             and TRAIN_START_NS <= timestamp + H4_NS < TRAIN_END_NS,
             "WITNESS_CLOCK_" + str(row))
        positions.append(pos)
    return positions


def compare_train_rows(oracle: np.ndarray, cache: np.ndarray,
                       cache_rows: np.ndarray, timestamps: np.ndarray) -> dict:
    """Compare 1044 x 31 cells without importing producer-selected lines."""
    a = np.asarray(oracle, dtype=np.float64)
    b = np.asarray(cache, dtype=np.float32).astype(np.float64)
    rows = np.asarray(cache_rows, dtype=np.int64)
    times = np.asarray(timestamps, dtype=np.int64)
    need(a.shape == b.shape == (len(TRAIN_ROWS), len(NAMES))
         and np.array_equal(rows, TRAIN_ROWS)
         and times.shape == (len(TRAIN_ROWS),) and np.all(np.diff(times) > 0)
         and len(NAMES) == 31 and len(LIFECYCLE_NAMES) == 19
         and not np.isinf(a).any() and not np.isinf(b).any(), "COMPARE_SHAPE")
    lifecycle = set(LIFECYCLE_NAMES)
    anchor_rows = {row for row, _, _ in WITNESSES}
    per_field = []
    examples = []
    mismatches = lifecycle_mismatches = anchor_mismatches = 0
    for j, name in enumerate(NAMES):
        expected, observed = a[:, j], b[:, j]
        finite = np.isfinite(expected) & np.isfinite(observed)
        equal_missing = np.isnan(expected) & np.isnan(observed)
        error = np.abs(expected[finite] - observed[finite])
        tolerance = (np.zeros(len(error)) if j in DISCRETE
                     else np.maximum(FLOAT_ABS_TOL,
                                     FLOAT_REL_TOL * np.abs(expected[finite])))
        match = equal_missing.copy()
        match[finite] = error <= tolerance
        bad = np.flatnonzero(~match)
        mismatches += len(bad)
        lifecycle_mismatches += len(bad) if name in lifecycle else 0
        anchor_mismatches += sum(int(rows[i] in anchor_rows) for i in bad)
        for i in bad[:max(0, 128 - len(examples))]:
            examples.append({"cache_row": int(rows[i]), "timestamp_ns": int(times[i]),
                             "field": name, "oracle": float(expected[i]) if np.isfinite(expected[i]) else None,
                             "cache": float(observed[i]) if np.isfinite(observed[i]) else None})
        oracle_finite = np.isfinite(expected)
        cache_finite = np.isfinite(observed)
        oracle_nonzero = int(np.count_nonzero(oracle_finite & (expected != 0)))
        cache_nonzero = int(np.count_nonzero(cache_finite & (observed != 0)))
        per_field.append({
            "field": name, "lifecycle": name in lifecycle,
            "oracle_finite_rows": int(oracle_finite.sum()),
            "cache_finite_rows": int(cache_finite.sum()),
            "oracle_nonzero_or_event_rows": oracle_nonzero,
            "cache_nonzero_or_event_rows": cache_nonzero,
            "exercised": bool(oracle_nonzero and cache_nonzero),
            "mismatched_rows": len(bad),
            "max_finite_absolute_error": float(error.max()) if len(error) else None,
        })
    exercised = [x["field"] for x in per_field
                 if x["lifecycle"] and x["exercised"] and not x["mismatched_rows"]]
    return {
        "rows": len(TRAIN_ROWS), "fields": len(NAMES),
        "compared_cells": len(TRAIN_ROWS) * len(NAMES),
        "lifecycle_fields": len(LIFECYCLE_NAMES),
        "lifecycle_cells": len(TRAIN_ROWS) * len(LIFECYCLE_NAMES),
        "anchor_rows": len(anchor_rows), "anchor_cells": len(anchor_rows) * len(NAMES),
        "mismatched_cells": mismatches,
        "lifecycle_mismatched_cells": lifecycle_mismatches,
        "anchor_mismatched_cells": anchor_mismatches,
        "all_train_window_cells_match": mismatches == 0,
        "matched_and_exercised_lifecycle_fields": exercised,
        "unexercised_lifecycle_fields": [x["field"] for x in per_field
                                         if x["lifecycle"] and not x["exercised"]],
        "mismatched_lifecycle_fields": [x["field"] for x in per_field
                                        if x["lifecycle"] and x["mismatched_rows"]],
        "float_abs_tolerance": FLOAT_ABS_TOL,
        "float_rel_tolerance": FLOAT_REL_TOL,
        "per_field": per_field, "mismatch_examples": examples,
        "mismatch_examples_truncated": mismatches > len(examples),
    }


def inspect_plan(plan_ref: dict
                 ) -> tuple[dict, dict, dict, list[dict], dict, tuple[str, str, str]]:
    """Validate only small immutable receipts and source code, before any claim."""
    plan = bound_json(plan_ref)
    need(set(plan) == {"schema_version", "stage", "status", "inputs", "sources",
                       "scope", "limits", "output_dir", "test_data_used"}
         and plan["schema_version"] == "gx1_h4_trendline_lifecycle_physical_plan_v1"
         and plan["stage"] == STAGE and plan["status"] == "PREPARED_NOT_ADMITTED"
         and plan["inputs"] == OLD_REFS and plan["scope"] == SCOPE
         and plan["limits"] == LIMITS and plan["test_data_used"] is False,
         "PLAN_CONTRACT")
    state = git_state()
    need(state[1:] == ("work/gx1-current", ""), "SOURCE_TREE")
    need(set(plan["sources"]) == set(SOURCE_PATHS), "SOURCE_KEYS")
    source_stats = {}
    for name, relative in SOURCE_PATHS.items():
        ref = plan["sources"][name]
        need(ref["path"] == str(REPO / relative), "SOURCE_PATH_" + name)
        _, source_stats[name] = checked(ref)
    output = Path(plan["output_dir"])
    need(output.is_absolute() and output.parent == Path("/home/andre2/GX1_RUNS")
         and output.name == STAGE.upper() and output.parent.is_dir(), "OUTPUT_SCOPE")
    old = bound_json(OLD_REFS["plan"])
    prior = bound_json(OLD_REFS["result"])
    terminal = bound_json(OLD_REFS["terminal"])
    need(prior["plan"] == OLD_REFS["plan"] and terminal["status"] == "COMPLETE"
         and terminal["exit_code"] == 0 and terminal["result"] == OLD_REFS["result"]
         and terminal["test_data_used"] is False, "OLD_TERMINAL")
    records = witness_receipt(prior)
    inputs = old["inputs"]
    need(old["scope"]["witness_end_exclusive_ns"] == TRAIN_END_NS
         and old["test_data_used"] is False and len(inputs["native_parts"]) == 16
         and [int(Path(x["path"]).parent.name.split("=")[1]) for x in inputs["native_parts"]]
             == list(range(2009, 2025)), "OLD_INPUT_SCOPE")
    manifest = bound_json(inputs["cache_manifest"], max_bytes=64 * 1024 * 1024)
    fit = bound_json(inputs["fit_result"])
    enriched = bound_json(inputs["enriched_manifest"])
    native = bound_json(inputs["native_manifest"])
    need(fit["fold"]["train_cutoff_ns"] == TRAIN_END_NS
         and fit["test_data_used"] is False
         and enriched["native_m5_source"]["manifest_path"] == inputs["native_manifest"]["path"]
         and enriched["native_m5_source"]["manifest_sha256"] == inputs["native_manifest"]["sha256"]
         and all(native["year_sha256"][Path(ref["path"]).parent.name] == ref["sha256"]
                 for ref in inputs["native_parts"]), "INPUT_LINEAGE")
    constants = manifest["v29_registry_constants"]
    need(constants["per_tf_seq_lens"]["H4"] == SEQ_LEN
         and constants["trendline_band_atr"]["H4"] == BAND_ATR
         and [manifest["feature_names"].index(name) for name in NAMES]
             == list(range(159, 190)), "H4_FEATURE_CONTRACT")
    refs = inputs["per_tf"]["H4"]
    need(set(refs) == {"feats", "ts", "replay_plan", "replay_result", "replay_terminal"},
         "H4_REFS")
    cache_root = Path(inputs["cache_manifest"]["path"]).parent
    h4 = manifest["tfs"]["H4"]
    need(h4["n_bars"] == 24816 and h4["feature_count"] == 253
         and refs["feats"] == {"path": str(cache_root / h4["feats_npy"]),
                              "sha256": h4["feats_npy_sha256"]}
         and refs["ts"] == {"path": str(cache_root / h4["ts_npy"]),
                           "sha256": h4["ts_npy_sha256"]}, "H4_CACHE_MANIFEST_BINDING")
    replay_plan = bound_json(refs["replay_plan"])
    replay_result = bound_json(refs["replay_result"])
    replay_terminal = bound_json(refs["replay_terminal"])
    need(replay_plan["inputs"]["tf_feats"] == refs["feats"]
         and replay_plan["inputs"]["tf_ts"] == refs["ts"]
         and replay_plan["inputs"]["cache_manifest"] == inputs["cache_manifest"]
         and replay_result["status"] == "COMPLETE"
         and replay_result["input_price_precision"] == "native_float64_full_history"
         and replay_result["source_clock_rows"] == PREFIX_ROWS
         and replay_result["replay_parity"] is True
         and replay_result["all_causal_history_vs_cache"]["mismatched_cells"] == 0
         and replay_result["window_configuration"] == {
             "timeframe": "H4", "seq_len": SEQ_LEN, "band_atr": BAND_ATR}
         and replay_result["model_train_period_vs_cache"]["mismatched_cells"] == 0
         and len(replay_result["model_train_period_vs_cache"]["per_field"]) == len(NAMES)
         and all(field["both_finite_rows"] == len(TRAIN_ROWS)
                 for field in replay_result["model_train_period_vs_cache"]["per_field"])
         and replay_result["test_data_used"] is False
         and replay_terminal["status"] == "COMPLETE"
         and replay_terminal["exit_code"] == 0, "H4_REPLAY_BINDING")
    return plan, inputs, manifest, records, source_stats, state


def preflight_read_only(plan_ref: dict) -> dict:
    """Check exact plan and old receipts without CLAIM, raw parts, or cache arrays."""
    plan, inputs, _manifest, records, source_stats, state = inspect_plan(plan_ref)
    return {
        "schema_version": "gx1_h4_trendline_lifecycle_preflight_v1",
        "status": "READY_NO_CLAIM", "stage": STAGE,
        "plan": plan_ref, "source_head": state[0],
        "source_owner_refs_checked": len(source_stats),
        "prior_receipts_checked": len(OLD_REFS),
        "native_parts_manifest_bound": len(inputs["native_parts"]),
        "h4_cache_refs_manifest_bound": True,
        "train_cache_rows": len(TRAIN_ROWS),
        "anchor_witnesses": len(records),
        "claim_created": False, "raw_native_or_cache_payload_read": False,
        "model_forwards": 0, "optimizer_steps": 0, "test_data_used": False,
        "output_root": plan["output_dir"],
    }


def preflight(plan_ref: dict) -> tuple[dict, dict, dict, list[dict], dict, tuple[str, str, str]]:
    plan, inputs, manifest, records, source_stats, state = inspect_plan(plan_ref)
    output = Path(plan["output_dir"])
    claim, claim_ref, claim_stat = read_claim(
        output, plan_ref, plan["sources"]["worker"], state[0])
    policy = bound_json(claim["policy"])
    gate = policy.get(STAGE)
    need(isinstance(gate, dict) and gate.get("status") == "ADMITTED"
         and gate.get("plan") == plan_ref and gate.get("worker") == plan["sources"]["worker"]
         and gate.get("output_root") == plan["output_dir"]
         and gate.get("execution_authorized") is True
         and gate.get("data_reads_authorized") is True
         and gate.get("claimed_invocations") == gate.get("completed_invocations") == 0
         and gate.get("max_invocations") == 1 and gate.get("relaunch_allowed") is False
         and gate.get("memory_max") == "4G" and gate.get("swap_max") == "512M"
         and gate.get("wall_seconds") == 1800 and gate.get("test_data_used") is False
         and gate.get("model_forwards_authorized") is False
         and gate.get("optimizer_steps_authorized") is False
         and gate.get("checkpoint_reads_authorized") is False
         and gate.get("broker_access_allowed") is False
         and policy.get("training_enabled") is False
         and policy.get("native_training_enabled") is False
         and policy.get("native_launch_authorized") is False,
         "POLICY_GATE")
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry,
    )
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap["memory_max"] == cap["memory_high"] == 4294967296
         and cap["swap"] == 536870912 and cap["pids"] == 64
         and cap["numerical_threads"] == 1
         and lock["scope"] == cap["cgroup_path"][len("/sys/fs/cgroup"):],
         "CAP_LOCK")
    need(not any(x.split(".")[0] in ("torch", "sklearn", "joblib")
                 for x in sys.modules), "MODEL_IMPORT")
    return plan, inputs, manifest, records, {
        "execution": cap, "lock": lock, "source_stats": source_stats,
        "claim_ref": claim_ref, "claim_stat": claim_stat,
    }, state

def run(plan_path: Path, plan_sha256: str) -> dict:
    plan_ref = {"path": str(plan_path), "sha256": plan_sha256}
    plan, inputs, manifest, records, caps, original_state = preflight(plan_ref)
    output = Path(plan["output_dir"])
    old_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError("H4_TRENDLINE_PHYSICAL_WALL_LIMIT")))
    signal.alarm(LIMITS["wall_seconds"])
    started = time.monotonic()
    try:
        refs = inputs["per_tf"]["H4"]
        source_stats = {}
        for ref in (*inputs["native_parts"], refs["feats"], refs["ts"]):
            path, stat = checked(ref)
            source_stats[str(path)] = stat
        from gx1.scripts.audit_direction_d1_trendline_replay_v1 import load_native_parts_prefix
        from gx1.features.htf_features import (
            _resample_ohlcv, build_multi_tf_v4_closed_timestamp_indices,
        )
        from gx1.features.technical_indicators_v1 import ema50_200_spread_atr_block
        source = load_native_parts_prefix(inputs["native_parts"], TRAIN_END_NS)
        need(len(source) > 0 and source.index[-1].value < TRAIN_END_NS,
             "NATIVE_TRAIN_ONLY")
        index = build_multi_tf_v4_closed_timestamp_indices(source.index)["H4"]
        bars = _resample_ohlcv(source, "H4").dropna(
            subset=["open", "high", "low", "close"]).loc[index]
        need(bars.index.equals(index), "H4_RESAMPLE_AXIS")
        atr = ema50_200_spread_atr_block(
            bars["high"], bars["low"], bars["close"])["atr14"]
        cache_ts = np.load(refs["ts"]["path"], mmap_mode="r", allow_pickle=False)
        cache = np.load(refs["feats"]["path"], mmap_mode="r", allow_pickle=False)
        need(cache_ts.dtype == np.int64 and cache_ts.ndim == 1
             and cache.dtype == np.float32 and cache.ndim == 2
             and cache.shape == (len(cache_ts), manifest["feature_count"]),
             "H4_CACHE_SHAPE")
        rows, positions, timestamps = align_train_rows(cache_ts, index.asi8)
        anchors = align_witnesses(cache_ts, index.asi8, records)
        for record, pos in zip(records, anchors):
            need(int(positions[record["cache_row"] - TRAIN_FIRST]) == pos,
                 "ANCHOR_IN_TRAIN_RANGE")
            need(float(bars.iloc[pos]["close"]) == record["source_close"]
                 and math.isclose(float(atr.iloc[pos]), record["source_atr14"],
                                  rel_tol=0, abs_tol=1e-12),
                 "SOURCE_WITNESS_" + str(record["cache_row"]))
        causal_end = positions[-1] + 1
        oracle, event_counts = oracle_rows(
            bars["high"].to_numpy()[:causal_end],
            bars["low"].to_numpy()[:causal_end],
            bars["close"].to_numpy()[:causal_end],
            atr.to_numpy()[:causal_end],
            seq_len=SEQ_LEN, band_atr=BAND_ATR, witness_positions=positions)
        cached = np.asarray(cache[rows][:, 159:190], dtype=np.float32)
        comparison = compare_train_rows(oracle, cached, rows, timestamps)
        need(git_state() == original_state, "FINAL_SOURCE_TREE")
        for path, stat in source_stats.items():
            need(regular(Path(path)) == stat, "FINAL_SOURCE_STAT_" + Path(path).name)
        for name, stat in caps["source_stats"].items():
            need(regular(Path(plan["sources"][name]["path"])) == stat,
                 "FINAL_OWNER_STAT_" + name)
        need(regular(output / "CLAIM.json") == caps["claim_stat"]
             and {entry.name for entry in output.iterdir()} == {"CLAIM.json"},
             "FINAL_CLAIM_ONLY_ROOT")
        result = {
            "schema_version": "gx1_h4_trendline_lifecycle_physical_result_v1",
            "status": "COMPLETE",
            "decision": ("FAIL_H4_TRAIN_WINDOW" if comparison["mismatched_cells"] else
                         "PASS_H4_TRAIN_WINDOW_19_EXERCISED"
                         if not comparison["unexercised_lifecycle_fields"] else
                         "PASS_H4_TRAIN_WINDOW_WITH_SUPPORT_GAPS"),
            "plan": plan_ref, "claim": caps["claim_ref"],
            "source_head": original_state[0], "old_receipts": OLD_REFS,
            "source_refs": plan["sources"], "native_parts": inputs["native_parts"],
            "cache_refs": {"feats": refs["feats"], "ts": refs["ts"]},
            "train_end_exclusive_ns": TRAIN_END_NS,
            "timestamp_rule": "H4 cache label plus four UTC hours is the TRAIN close",
            "anchor_witnesses": records,
            "train_cache_rows": {"first": TRAIN_FIRST, "last_inclusive": TRAIN_LAST,
                                 "count": len(TRAIN_ROWS)},
            "comparison": comparison, "oracle_event_counts": event_counts,
            "remaining_lifecycle_fields_before_pilot": 19,
            "all_19_lifecycle_fields_globally_qualified": False,
            "all_five_timeframes_qualified": False,
            "model_forwards": 0, "optimizer_steps": 0,
            "test_data_used": False, "elapsed_seconds": time.monotonic() - started,
            "cap": {"execution": caps["execution"], "lock": caps["lock"]},
        }
        from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
        payload = (json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        _publish_bytes(output / "RESULT.json", payload)
        return result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    plan_ref = {"path": str(Path(args.plan)), "sha256": args.plan_sha256}
    if args.preflight:
        print(json.dumps(preflight_read_only(plan_ref), sort_keys=True, allow_nan=False))
    else:
        run(Path(args.plan), args.plan_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
