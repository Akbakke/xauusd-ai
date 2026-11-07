"""One-use TRAIN-only diagnosis of M5 market-state side provenance.

The saved 1.0-ATR candidate supplies the exact DOWN/UP endpoint cohort.
The existing target owner is replayed with the same pinned source and ATR;
a second replay starts cold at each endpoint's last five observed M1 bars.
This diagnoses whether a local side uses older history. It is not a new
target, model fit, prediction score, Entry rule, or economic result.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

from gx1.audit import build_full_train_market_state_endpoint_candidate_v1 as base

REPO = base.REPO
STAGE = "m5_state_side_locality_20261107_001"
PLAN_SCHEMA = "gx1_m5_state_side_locality_plan_v1"
CLAIM_SCHEMA = "gx1_m5_state_side_locality_claim_v1"
SOURCE_ROW_OFFSET = 5193205
EXACT_SIDES = 39670
WINDOW = 5
LIMITS = {"memory_max": "4G", "swap_max": "512M",
          "wall_seconds": 1800, "execution_class": "audit",
          "model_forwards": 0, "optimizer_steps": 0}
SCOPE = {"timeframe": "M5", "train_start_utc": base.TRAIN_START,
         "train_end_utc_exclusive": base.TRAIN_END,
         "issue_rows": 118706, "exact_side_endpoints": EXACT_SIDES,
         "local_observed_m1_bars": WINDOW, "test_data_used": False}
PREVIOUS_ROOT = Path("/home/andre2/GX1_RUNS/SINGLE_THRESHOLD_MARKET_STATE_CANDIDATE_20261107_001")
PREVIOUS_REFS = {
    "plan": {"path": str(REPO / "configs/research/SINGLE_THRESHOLD_MARKET_STATE_CANDIDATE_20261107_001.json"),
             "sha256": "9de70e39d0618ea579c4ccd95a8f7ba3e174511a8052befb3728d9f810d521a9"},
    "result": {"path": str(PREVIOUS_ROOT / "RESULT.json"),
               "sha256": "1bad8a8a88d61a67ee2487e387c246da06f4f4ccf319e7edd8ce35d39d13eba1"},
    "row_masks": {"path": str(PREVIOUS_ROOT / "ROW_MASKS_AND_RATIOS.npz"),
                  "sha256": "35257429efa4ec553293e1b10bec0993c3c494f04e7f5be72d55877112e590e5"},
    "terminal": {"path": str(PREVIOUS_ROOT / "TERMINAL.json"),
                 "sha256": "9ce11435f5be8174dd629c5f6c564f8a945cc432fb037afed14de47222798295"},
}
SOURCE_PATHS = {
    "worker": "gx1/audit/verify_m5_state_side_locality_v1.py",
    "previous_worker": "gx1/audit/verify_single_threshold_market_state_candidate_v1.py",
    "full_train_loader": "gx1/audit/build_full_train_market_state_endpoint_candidate_v1.py",
    "target": "gx1/contracts/entry_dynamic_mtf_state_targets_v1.py",
    "closure_owner": "gx1/contracts/unified_exit_market_closure_authority_v1.py",
    "capped_owner": "gx1/contracts/gx1_capped_execution_v1.py",
    "publication_owner": "gx1/contracts/immutable_event_authority_v1.py",
}


def need(ok: object, name: str) -> None:
    if not bool(ok):
        raise RuntimeError("M5_STATE_SIDE_LOCALITY_" + name)


def ref(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def git_state() -> tuple[str, str, str]:
    def command(*args: str) -> str:
        return subprocess.check_output(("git", *args), cwd=REPO, text=True).strip()
    return command("rev-parse", "HEAD"), command("branch", "--show-current"), command("status", "--porcelain")


def metadata_preflight() -> tuple[dict, dict, dict]:
    """Check reused receipt keys and saved masks without reading raw prices."""
    plan = base.read_json(PREVIOUS_REFS["plan"])
    result = base.read_json(PREVIOUS_REFS["result"])
    terminal = base.read_json(PREVIOUS_REFS["terminal"])
    need(plan["schema_version"] == "gx1_single_threshold_market_state_candidate_plan_v1"
         and plan["m1"] == base.FIXED_REFS["m1"]
         and plan["m5"] == base.FIXED_REFS["m5"]
         and plan["closure"] == base.FIXED_REFS["closure"]
         and plan["factor"] == 1.0
         and plan["train_start_utc"] == base.TRAIN_START
         and plan["train_end_utc_exclusive"] == base.TRAIN_END
         and plan["m1_row_groups"] == list(base.M1_GROUPS)
         and plan["m5_row_groups"] == list(base.M5_GROUPS)
         and plan["m1_expected_rows"] == base.M1_ROWS
         and plan["m5_expected_issues"] == SCOPE["issue_rows"]
         and plan["lead_minutes"] == list(base.LEADS)
         and plan["test_data_used"] is False, "PREVIOUS_PLAN")
    need(result["decision"] == "PASS_SINGLE_THRESHOLD_CANDIDATE_SOURCE_ONLY"
         and result["plan"] == PREVIOUS_REFS["plan"]
         and result["row_masks_and_ratios"] == PREVIOUS_REFS["row_masks"]
         and result["source_refs"]["m1"] == plan["m1"]
         and result["source_refs"]["m5"] == plan["m5"]
         and result["source_refs"]["closure"] == plan["closure"]
         and result["factor"] == 1.0
         and result["m1_rows"] == base.M1_ROWS
         and result["m5_issues"] == SCOPE["issue_rows"]
         and result["totals"]["M5"]["exact_singleton"] == 110539
         and sum(result["monthly"][m]["M5"]["down"] +
                 result["monthly"][m]["M5"]["up"] for m in result["monthly"]) == EXACT_SIDES
         and result["model_calls"] == result["optimizer_steps"] == 0
         and result["test_data_used"] is False and result["target_locked"] is False,
         "PREVIOUS_RESULT")
    need(terminal["status"] == "COMPLETE" and terminal["exit_code"] == 0
         and terminal["result"] == PREVIOUS_REFS["result"]
         and terminal["row_masks_and_ratios"] == PREVIOUS_REFS["row_masks"]
         and terminal["model_calls"] == terminal["optimizer_steps"] == 0
         and terminal["test_data_used"] is False and terminal["target_locked"] is False,
         "PREVIOUS_TERMINAL")
    raw, _ = base.checked(PREVIOUS_REFS["row_masks"], max_bytes=64 * 1024 * 1024)
    need(raw is not None, "PREVIOUS_MASK_BYTES")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        need(set(z.files) == {
            "issue_ns", "m5_source_row", "lead_minutes", "allowed_set",
            "m1_source_row", "complete_range_atr", "complete_range_atr_valid",
            "atr_to_quoted_spread", "atr_to_quoted_spread_valid"}, "PREVIOUS_NPZ_KEYS")
        saved = {name: np.array(z[name]) for name in
                 ("issue_ns", "m5_source_row", "lead_minutes", "allowed_set", "m1_source_row")}
    need(saved["issue_ns"].shape == (SCOPE["issue_rows"],)
         and saved["allowed_set"].shape == saved["m1_source_row"].shape ==
             (SCOPE["issue_rows"], len(base.LEADS))
         and np.array_equal(saved["lead_minutes"], np.asarray(base.LEADS))
         and np.all(np.diff(saved["issue_ns"]) > 0)
         and int(np.count_nonzero(np.isin(saved["allowed_set"][:, 0], (1, 4)))) == EXACT_SIDES,
         "PREVIOUS_NPZ_SCOPE")
    return plan, result, saved


def local_side(row: int, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
               close: np.ndarray, theta: np.ndarray, unknown_gap: np.ndarray,
               target) -> tuple[int, bool]:
    """Cold replay only the five observed bars ending at row."""
    start = row - WINDOW + 1
    need(start >= 0 and row < len(open_), "LOCAL_ROW")
    start_after_unknown = bool(unknown_gap[start])
    predecessor = float(open_[start] if start == 0 or start_after_unknown
                        else close[start - 1])
    direction = target.UNKNOWN
    anchor_low = anchor_high = predecessor
    for k in range(start, row + 1):
        if unknown_gap[k] or not (theta[k] > 0):
            direction = target.UNKNOWN
            anchor_low = anchor_high = float(open_[k])
        if not (theta[k] > 0):
            continue
        direction, anchor_low, anchor_high, _ = target._step_direction(
            direction, anchor_low, anchor_high,
            float(high[k]), float(low[k]), float(close[k]), float(theta[k]))
    return direction, start_after_unknown


def closure_window_flags(row: int, axis: np.ndarray,
                         unknown_gap: np.ndarray) -> tuple[bool, bool]:
    """Known closure among five bars, and immediately before their first bar."""
    start = row - WINDOW + 1
    need(start >= 0 and row < len(axis) and len(axis) == len(unknown_gap)
         and np.all(np.diff(axis[start:row + 1]) > 0), "CLOSURE_WINDOW")
    successors = np.flatnonzero(
        np.diff(axis[start:row + 1]) != base.MINUTE_NS) + start + 1
    need(not np.any(unknown_gap[successors]), "INCOMPLETE_LOCAL_WINDOW")
    before = bool(start > 0 and
                  axis[start] - axis[start - 1] != base.MINUTE_NS)
    return bool(len(successors)), bool(before and not unknown_gap[start])


def replay_side_entry(open_: np.ndarray, high: np.ndarray, low: np.ndarray,
                      close: np.ndarray, theta: np.ndarray,
                      unknown_gap: np.ndarray, flat: np.ndarray,
                      target) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track the last transition into a certified side and any later FLAT."""
    n = len(open_)
    need(all(len(x) == n for x in (high, low, close, theta, unknown_gap, flat)),
         "REPLAY_SHAPE")
    side = np.full(n, target.UNKNOWN, np.int8)
    last_entry = np.full(n, -1, np.int32)
    last_flat = np.full(n, -1, np.int32)
    direction = target.UNKNOWN
    anchor_low = anchor_high = float(open_[0])
    entry = flat_row = -1
    for k in range(n):
        if unknown_gap[k] or not (theta[k] > 0):
            direction = target.UNKNOWN
            anchor_low = anchor_high = float(open_[k])
            entry = -1
        if theta[k] > 0:
            prior = direction
            direction, anchor_low, anchor_high, _ = target._step_direction(
                direction, anchor_low, anchor_high,
                float(high[k]), float(low[k]), float(close[k]), float(theta[k]))
            if direction in (target.DOWN, target.UP) and prior != direction:
                entry = k
            elif direction == target.UNKNOWN:
                entry = -1
        if flat[k]:
            flat_row = k
        side[k], last_entry[k], last_flat[k] = direction, entry, flat_row
    return side, last_entry, last_flat


def summarize(mask: np.ndarray, local: np.ndarray, full: np.ndarray,
              age_bars: np.ndarray, age_minutes: np.ndarray,
              old_event: np.ndarray, before_issue: np.ndarray,
              flat_since: np.ndarray, closure_inside: np.ndarray,
              closure_before: np.ndarray, after_unknown: np.ndarray) -> dict:
    selected = np.flatnonzero(mask)
    same = local[selected] == full[selected]
    unknown = local[selected] == -1
    opposite = ~(same | unknown)
    need(np.all(np.isin(local[selected], (-1, 0, 2)))
         and np.all(np.isin(full[selected], (0, 2)))
         and np.all(same | unknown | opposite), "SUMMARY_CLASS")
    def quantiles(values):
        if len(values) == 0:
            return {"median": None, "p90": None}
        return {"median": float(np.median(values)), "p90": float(np.quantile(values, 0.9))}
    return {
        "endpoints": int(len(selected)), "local_same": int(same.sum()),
        "local_unknown": int(unknown.sum()), "local_opposite": int(opposite.sum()),
        "side_entry_older_than_local_window": int(old_event[selected].sum()),
        "side_entry_at_or_before_issue": int(before_issue[selected].sum()),
        "flat_since_side_entry": int(flat_since[selected].sum()),
        "known_closure_within_window": int(closure_inside[selected].sum()),
        "known_closure_before_first_window_bar": int(closure_before[selected].sum()),
        "known_closure_touching_window": int(
            (closure_inside[selected] | closure_before[selected]).sum()),
        "first_window_bar_after_unknown_gap": int(after_unknown[selected].sum()),
        "side_entry_age_observed_m1_bars": quantiles(age_bars[selected]),
        "side_entry_age_utc_minutes": quantiles(age_minutes[selected]),
    }


def preflight(plan_ref: dict) -> tuple[dict, dict, dict, dict, dict, tuple[str, str, str]]:
    plan = base.read_json(plan_ref)
    need(set(plan) == {"schema_version", "stage", "status", "inputs",
                       "source_owners", "scope", "limits", "output_dir", "test_data_used"}
         and plan["schema_version"] == PLAN_SCHEMA and plan["stage"] == STAGE
         and plan["status"] == "PREPARED_NOT_ADMITTED"
         and plan["inputs"] == PREVIOUS_REFS
         and plan["scope"] == SCOPE and plan["limits"] == LIMITS
         and plan["test_data_used"] is False, "PLAN")
    state = git_state()
    need(state[1:] == ("work/gx1-current", ""), "SOURCE_TREE")
    need(set(plan["source_owners"]) == set(SOURCE_PATHS), "SOURCE_KEYS")
    owner_stats = {}
    for name, relative in SOURCE_PATHS.items():
        owner = plan["source_owners"][name]
        need(owner["path"] == str(REPO / relative), "SOURCE_PATH_" + name)
        _, owner_stats[name] = base.checked(owner)
    output = Path(plan["output_dir"])
    need(output.is_absolute() and output.parent == Path("/home/andre2/GX1_RUNS")
         and output.name == STAGE.upper() and output.is_dir()
         and not output.is_symlink()
         and {entry.name for entry in output.iterdir()} == {"CLAIM.json"},
         "OUTPUT")
    claim_path = output / "CLAIM.json"
    claim_stat = base.regular(claim_path)
    raw_claim = claim_path.read_bytes()
    need(base.regular(claim_path) == claim_stat and len(raw_claim) <= 1024 * 1024,
         "CLAIM_BYTES")
    claim = json.loads(raw_claim)
    need(set(claim) == {"schema_version", "stage", "plan", "policy", "source_head",
                        "worker", "source_only_audit", "test_data_used"}
         and claim["schema_version"] == CLAIM_SCHEMA
         and claim["stage"] == STAGE and claim["plan"] == plan_ref
         and claim["source_head"] == state[0]
         and claim["worker"] == plan["source_owners"]["worker"]
         and claim["source_only_audit"] is True and claim["test_data_used"] is False,
         "CLAIM")
    policy_ref = claim["policy"]
    need(policy_ref["path"] == str(REPO / "NEXT_RUN_POLICY.json"), "POLICY_PATH")
    policy = base.read_json(policy_ref)
    gate = policy.get(STAGE)
    need(isinstance(gate, dict)
         and gate.get("status") == "ADMITTED"
         and gate.get("plan") == plan_ref
         and gate.get("worker") == plan["source_owners"]["worker"]
         and gate.get("output_root") == str(output)
         and gate.get("execution_authorized") is True
         and gate.get("data_reads_authorized") is True
         and gate.get("claimed_invocations") == gate.get("completed_invocations") == 0
         and gate.get("max_invocations") == 1
         and gate.get("relaunch_allowed") is False
         and gate.get("memory_max") == "4G"
         and gate.get("swap_max") == "512M"
         and gate.get("wall_seconds") == 1800
         and gate.get("model_forwards_authorized") is False
         and gate.get("optimizer_steps_authorized") is False
         and gate.get("checkpoint_reads_authorized") is False
         and gate.get("broker_access_allowed") is False
         and gate.get("test_data_used") is False
         and policy.get("training_enabled") is False
         and policy.get("native_training_enabled") is False
         and policy.get("native_launch_authorized") is False,
         "POLICY")
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry,
    )
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap["memory_max"] == cap["memory_high"] == 4294967296
         and cap["swap"] == 536870912 and cap["pids"] == 64
         and cap["numerical_threads"] == 1
         and lock["scope"] == cap["cgroup_path"][len("/sys/fs/cgroup"):],
         "CAP")
    old_plan, old_result, saved = metadata_preflight()
    previous_stats = {name: base.regular(Path(item["path"]))
                      for name, item in PREVIOUS_REFS.items()}
    return plan, old_plan, old_result, saved, {
        "cap": cap, "lock": lock, "claim": ref(claim_path),
        "claim_stat": claim_stat, "owner_stats": owner_stats,
        "previous_stats": previous_stats}, state


def run(plan_path: Path, plan_sha256: str) -> dict:
    from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

    plan_ref = {"path": str(plan_path), "sha256": plan_sha256}
    plan, old_plan, old_result, saved, context, original_state = preflight(plan_ref)
    output = Path(plan["output_dir"])
    old_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError("M5_STATE_SIDE_LOCALITY_WALL_LIMIT")))
    signal.alarm(LIMITS["wall_seconds"])
    started = time.monotonic()
    try:
        schedule, authority = base.read_closure(old_plan)
        m1, m5, issues, m5_source_rows, source_stats = base.load_sources(old_plan, authority)
        need(np.array_equal(issues, saved["issue_ns"])
             and np.array_equal(m5_source_rows, saved["m5_source_row"]), "ISSUE_PARITY")
        cutoff = base.pd_ns(base.TRAIN_END)
        need(target.MINIMUM_MOVE_ATR == 0.5
             and tuple(target.TIMEFRAMES) == ("M5", "M15", "H1", "H4", "D1"),
             "TARGET_INITIAL")
        atr = target.build_causal_tf_atr(m5, cutoff_ns=cutoff)
        target.MINIMUM_MOVE_ATR = 1.0
        try:
            atr["definition_sha256"] = target.dynamic_mtf_state_target_contract()["contract_sha256"]
            series = target.build_dynamic_mtf_state_series(
                m1, tf_atr=atr, source_ref=old_plan["m1"]["source"],
                atr_source_ref=old_plan["m5"]["source"], cutoff_ns=cutoff,
                source_row_offset=SOURCE_ROW_OFFSET, closure_authority=authority,
                exact_schedule=schedule)
        finally:
            target.MINIMUM_MOVE_ATR = 0.5
        tape = series.arrays
        axis = np.asarray(tape["validity_ns"], dtype=np.int64)
        need(len(axis) == base.M1_ROWS
             and np.array_equal(tape["source_row"],
                                np.arange(SOURCE_ROW_OFFSET, SOURCE_ROW_OFFSET + base.M1_ROWS)),
             "TAPE_AXIS")
        times = target._clock(m1, target.EXIT_DECISION_BAR_SECONDS)
        known, _ = target._known_closures(
            times, old_plan["m1"]["source"], authority, schedule)
        unknown_gap = np.zeros(len(axis), bool)
        for row in np.flatnonzero(np.diff(axis) != base.MINUTE_NS) + 1:
            unknown_gap[row] = int(row) not in known
        need(int(unknown_gap.sum()) == authority["unknown_source_gap_count"],
             "GAP_COUNT")
        selected_issue = np.flatnonzero(np.isin(saved["allowed_set"][:, 0], (1, 4)))
        rows = saved["m1_source_row"][selected_issue, 0] - SOURCE_ROW_OFFSET
        expected = np.where(saved["allowed_set"][selected_issue, 0] == 1,
                            target.DOWN, target.UP).astype(np.int8)
        need(len(rows) == EXACT_SIDES and np.all((WINDOW - 1 <= rows) & (rows < len(axis)))
             and np.array_equal(axis[rows],
                                issues[selected_issue] + base.M5_NS)
             and np.array_equal(tape["state"][rows, 0], expected)
             and np.all(tape["observed"][rows, 0])
             and np.all(tape["local_range"][rows, 0] >= tape["threshold"][rows, 0]),
             "ENDPOINTS")
        open_ = m1["open"].to_numpy(np.float64)
        high = m1["high"].to_numpy(np.float64)
        low = m1["low"].to_numpy(np.float64)
        close = m1["close"].to_numpy(np.float64)
        theta = np.asarray(tape["threshold"][:, 0])
        flat = (tape["state"][:, 0] == target.FLAT) & tape["observed"][:, 0]
        full_side, last_entry, last_flat = replay_side_entry(
            open_, high, low, close, theta, unknown_gap, flat, target)
        need(np.array_equal(full_side[rows], expected)
             and np.all(last_entry[rows] >= 0), "FULL_SIDE_REPLAY")
        local = np.full(len(issues), target.UNKNOWN, np.int8)
        full = np.full(len(issues), target.UNKNOWN, np.int8)
        age_bars = np.zeros(len(issues), np.int32)
        age_minutes = np.zeros(len(issues), np.float64)
        old_event = np.zeros(len(issues), bool)
        before_issue = np.zeros(len(issues), bool)
        flat_since = np.zeros(len(issues), bool)
        closure_inside = np.zeros(len(issues), bool)
        closure_before = np.zeros(len(issues), bool)
        after_unknown = np.zeros(len(issues), bool)
        for issue_index, row, side in zip(selected_issue, rows, expected):
            start = int(row) - WINDOW + 1
            local_side_, start_unknown = local_side(
                int(row), open_, high, low, close, theta, unknown_gap, target)
            inside, before = closure_window_flags(int(row), axis, unknown_gap)
            entry = int(last_entry[row])
            local[issue_index], full[issue_index] = local_side_, side
            age_bars[issue_index] = int(row) - entry
            age_minutes[issue_index] = (int(axis[row]) - int(axis[entry])) / base.MINUTE_NS
            old_event[issue_index] = entry < start
            before_issue[issue_index] = int(axis[entry]) <= int(issues[issue_index])
            flat_since[issue_index] = int(last_flat[row]) > entry
            closure_inside[issue_index] = inside
            closure_before[issue_index] = before
            after_unknown[issue_index] = start_unknown
        selected = np.zeros(len(issues), bool)
        selected[selected_issue] = True
        overall = summarize(selected, local, full, age_bars, age_minutes,
                            old_event, before_issue, flat_since,
                            closure_inside, closure_before, after_unknown)
        need(overall["endpoints"] == EXACT_SIDES
             and overall["local_same"] + overall["local_unknown"] +
                 overall["local_opposite"] == EXACT_SIDES,
             "GLOBAL_COUNTS")
        monthly = {}
        edges = base.month_edges()
        for start, stop in zip(edges[:-1], edges[1:]):
            mask = selected & (issues >= start) & (issues < stop)
            import datetime
            name = datetime.datetime.fromtimestamp(
                start / 1e9, datetime.timezone.utc).strftime("%Y-%m")
            monthly[name] = summarize(mask, local, full, age_bars, age_minutes,
                                      old_event, before_issue, flat_since,
                                      closure_inside, closure_before,
                                      after_unknown)
            need(monthly[name]["endpoints"] ==
                 old_result["monthly"][name]["M5"]["down"] +
                 old_result["monthly"][name]["M5"]["up"],
                 "MONTH_PARITY")
        by_side = {
            name: summarize(selected & (full == state), local, full,
                            age_bars, age_minutes, old_event, before_issue,
                            flat_since, closure_inside, closure_before,
                            after_unknown)
            for name, state in (("DOWN", target.DOWN), ("UP", target.UP))
        }
        touching_closure = closure_inside | closure_before
        by_closure = {
            "known_closure_touching_window": summarize(
                selected & touching_closure, local, full, age_bars, age_minutes,
                old_event, before_issue, flat_since, closure_inside,
                closure_before, after_unknown),
            "no_known_closure_touching_window": summarize(
                selected & ~touching_closure, local, full, age_bars, age_minutes,
                old_event, before_issue, flat_since, closure_inside,
                closure_before, after_unknown),
        }
        need(sum(x["endpoints"] for x in monthly.values()) == EXACT_SIDES
             and sum(x["endpoints"] for x in by_side.values()) == EXACT_SIDES
             and sum(x["endpoints"] for x in by_closure.values()) == EXACT_SIDES,
             "PARTITIONS")
        need(git_state() == original_state, "FINAL_SOURCE")
        for path, stat in source_stats.items():
            need(base.regular(Path(path)) == stat, "FINAL_RAW_STAT")
        for name, stat in context["owner_stats"].items():
            need(base.regular(Path(plan["source_owners"][name]["path"])) == stat,
                 "FINAL_OWNER_STAT_" + name)
        for name, stat in context["previous_stats"].items():
            need(base.regular(Path(PREVIOUS_REFS[name]["path"])) == stat,
                 "FINAL_PREVIOUS_STAT_" + name)
        need(base.regular(output / "CLAIM.json") == context["claim_stat"]
             and {entry.name for entry in output.iterdir()} == {"CLAIM.json"},
             "FINAL_CLAIM")
        result = {
            "schema_version": "gx1_m5_state_side_locality_result_v1",
            "status": "COMPLETE", "decision": "DIAGNOSTIC_ONLY",
            "plan": plan_ref, "claim": context["claim"],
            "source_head": original_state[0],
            "source_refs": {"m1": old_plan["m1"], "m5": old_plan["m5"],
                            "closure": old_plan["closure"],
                            "previous": PREVIOUS_REFS,
                            "owners": plan["source_owners"]},
            "source_read_scope": "TRAIN M1 OHLC and historical M5 OHLC; Jan2026 VAL M5 boundary group decoded then excluded before ATR/labels/summary. No TEST.",
            "m1_rows": len(axis), "m5_issues": len(issues),
            "exact_m5_side_endpoints": EXACT_SIDES,
            "factor": 1.0, "window_observed_m1_bars": WINDOW,
            "full_history_side_rule": "Existing _step_direction; side-entry age is transition into a side, not continuous confirmation.",
            "local_side_rule": "Cold _step_direction replay over last five observed M1 bars; preceding close initial anchor except first bar after unknown gap uses its open.",
            "overall": overall, "monthly": monthly,
            "side_partition": by_side, "closure_partition": by_closure,
            "model_forwards": 0, "optimizer_steps": 0,
            "test_data_used": False, "target_locked": False,
            "limit": "Overlapping endpoints; local replay is a provenance diagnostic, not a replacement label, independent episode duration, forecast skill, or economic edge.",
            "elapsed_seconds": time.monotonic() - started,
            "cap": {"execution": context["cap"], "lock": context["lock"]},
        }
        _publish_bytes(output / "RESULT.json", base.encode(result))
        return result
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_alarm)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    run(Path(args.plan), args.plan_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
