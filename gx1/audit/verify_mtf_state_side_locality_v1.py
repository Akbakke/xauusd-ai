"""One-use TRAIN diagnosis of M15/H1/H4/D1 singleton side locality.

Reuses the sealed source candidate and checks only its existing DOWN/UP cells.
A cold last-W-M1-bar replay is compared with the full-history side. This is
source provenance, not a new label, trained model, prediction, or trade.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

from gx1.audit import build_full_train_market_state_endpoint_candidate_v1 as base
from gx1.audit import verify_m5_state_side_locality_v1 as m5

REPO = base.REPO
STAGE = "mtf_state_side_locality_20261107_001"
PLAN_SCHEMA = "gx1_mtf_state_side_locality_plan_v1"
CLAIM_SCHEMA = "gx1_mtf_state_side_locality_claim_v1"
SOURCE_ROW_OFFSET = m5.SOURCE_ROW_OFFSET
TF_SPECS = (("M15", 1, 15, 45411), ("H1", 2, 60, 44635),
            ("H4", 3, 240, 46092), ("D1", 4, 1440, 36633))
LIMITS = {"memory_max": "4G", "swap_max": "512M", "wall_seconds": 1800,
          "execution_class": "audit", "model_forwards": 0, "optimizer_steps": 0}
SCOPE = {"timeframes": [x[0] for x in TF_SPECS],
         "train_start_utc": base.TRAIN_START,
         "train_end_utc_exclusive": base.TRAIN_END,
         "issue_rows": 118706,
         "singleton_side_endpoints": {tf: count for tf, _, _, count in TF_SPECS},
         "local_observed_m1_bars": {tf: window for tf, _, window, _ in TF_SPECS},
         "test_data_used": False}
FULL_ROOT = Path("/home/andre2/GX1_RUNS/FULL_TRAIN_MARKET_STATE_ENDPOINT_CANDIDATE_20261107_001")
FULL_REFS = {
    "plan": {"path": str(REPO / "configs/research/FULL_TRAIN_MARKET_STATE_ENDPOINT_CANDIDATE_20261107_001.json"),
             "sha256": "c3c789f4e7ed3edba2880bfb9e51decc9efad779324351bbd19ac86d2a607ce2"},
    "result": {"path": str(FULL_ROOT / "RESULT.json"),
               "sha256": "3a508ce45755c8db2ad8c078d56a6afef8b88a6af3e0a0b159ca388def38ebc5"},
    "row_masks": {"path": str(FULL_ROOT / "ROW_MASKS.npz"),
                  "sha256": "ad0e7d56c58b463bdd3d61fe6b2fcf2b206afa56e899dfb36a850e63c5ec3020"},
    "terminal": {"path": str(FULL_ROOT / "TERMINAL.json"),
                 "sha256": "528a96b83f5d5191c32d2ccd89df772284edb9e7815d974554dc5ff7e5839922"},
}
SELECTED_ROOT = Path("/home/andre2/GX1_RUNS/MARKET_STATE_SELECTED_ACTIVITY_MASKS_20261107_001")
SELECTED_REFS = {
    "plan": {"path": str(REPO / "configs/research/MARKET_STATE_SELECTED_ACTIVITY_MASKS_20261107_001.json"),
             "sha256": "9b43bd30b3d3a9bb414923df547191200ca5b9c1e6993fc906cfdd283cd0f406"},
    "result": {"path": str(SELECTED_ROOT / "RESULT.json"),
               "sha256": "a2f085b3d14b4fc6d934db22c47955eeff6f2f25732574f940dc8bba5281916e"},
    "row_masks": {"path": str(SELECTED_ROOT / "ROW_MASKS.npz"),
                  "sha256": "5a6e9b3377ded33760080d7f9145ecfcd8bf60faf24e53a34a42c420ad4a1a02"},
    "terminal": {"path": str(SELECTED_ROOT / "TERMINAL.json"),
                 "sha256": "2f994f52d594e403b9777063aefd0de70f335dc0e5d62a007b6180dd1b2b53fc"},
}
INPUTS = {"full": FULL_REFS, "single": m5.PREVIOUS_REFS, "selected": SELECTED_REFS}
SOURCE_PATHS = {
    "worker": "gx1/audit/verify_mtf_state_side_locality_v1.py",
    "m5_locality": "gx1/audit/verify_m5_state_side_locality_v1.py",
    "full_loader": "gx1/audit/build_full_train_market_state_endpoint_candidate_v1.py",
    "target": "gx1/contracts/entry_dynamic_mtf_state_targets_v1.py",
    "closure_owner": "gx1/contracts/unified_exit_market_closure_authority_v1.py",
    "capped_contract": "gx1/contracts/gx1_capped_execution_v1.py",
    "capped_runner": "scripts/gx1_capped_run.sh",
    "publication": "gx1/contracts/immutable_event_authority_v1.py",
}


def need(value: object, reason: str) -> None:
    if not bool(value):
        raise RuntimeError("MTF_SIDE_LOCALITY_" + reason)


def small_receipts() -> tuple[dict, dict, dict, dict]:
    """Inspect metadata only; neither NPZ payload nor raw M1/M5 is opened."""
    full_plan = base.read_json(FULL_REFS["plan"])
    full_result = base.read_json(FULL_REFS["result"])
    full_terminal = base.read_json(FULL_REFS["terminal"])
    single_plan = base.read_json(m5.PREVIOUS_REFS["plan"])
    single_result = base.read_json(m5.PREVIOUS_REFS["result"])
    single_terminal = base.read_json(m5.PREVIOUS_REFS["terminal"])
    selected_plan = base.read_json(SELECTED_REFS["plan"])
    selected_result = base.read_json(SELECTED_REFS["result"])
    selected_terminal = base.read_json(SELECTED_REFS["terminal"])
    need(full_plan["schema_version"] == base.PLAN_SCHEMA
         and full_plan["test_data_used"] is False
         and full_plan["train_start_utc"] == base.TRAIN_START
         and full_plan["train_end_utc_exclusive"] == base.TRAIN_END
         and full_plan["lead_minutes"] == list(base.LEADS)
         and full_plan["factors"] == [0.5, 1.0]
         and full_plan["m1_expected_rows"] == base.M1_ROWS
         and full_result["decision"] == "PASS_FULL_TRAIN_CANDIDATE_SOURCE_ONLY"
         and full_result["plan"] == FULL_REFS["plan"]
         and full_result["row_masks"] == FULL_REFS["row_masks"]
         and full_result["m1_rows"] == base.M1_ROWS
         and full_result["m5_issues"] == SCOPE["issue_rows"]
         and full_result["model_calls"] == full_result["optimizer_steps"] == 0
         and full_result["test_data_used"] is False
         and full_result["target_locked"] is False
         and full_terminal["status"] == "COMPLETE"
         and full_terminal["exit_code"] == 0
         and full_terminal["result"] == FULL_REFS["result"]
         and full_terminal["row_masks"] == FULL_REFS["row_masks"]
         and full_terminal["test_data_used"] is False,
         "FULL_RECEIPTS")
    need(single_plan["schema_version"] == "gx1_single_threshold_market_state_candidate_plan_v1"
         and single_plan["factor"] == 1.0
         and single_plan["test_data_used"] is False
         and all(single_plan[key] == full_plan[key] for key in ("m1", "m5", "closure"))
         and single_plan["lead_minutes"] == list(base.LEADS)
         and single_result["decision"] == "PASS_SINGLE_THRESHOLD_CANDIDATE_SOURCE_ONLY"
         and single_result["plan"] == m5.PREVIOUS_REFS["plan"]
         and single_result["row_masks_and_ratios"] == m5.PREVIOUS_REFS["row_masks"]
         and single_result["m1_rows"] == base.M1_ROWS
         and single_result["m5_issues"] == SCOPE["issue_rows"]
         and single_result["model_calls"] == single_result["optimizer_steps"] == 0
         and single_result["test_data_used"] is False
         and single_result["target_locked"] is False
         and single_terminal["status"] == "COMPLETE"
         and single_terminal["exit_code"] == 0
         and single_terminal["result"] == m5.PREVIOUS_REFS["result"]
         and single_terminal["row_masks_and_ratios"] == m5.PREVIOUS_REFS["row_masks"]
         and single_terminal["test_data_used"] is False,
         "SINGLE_RECEIPTS")
    need(selected_result["status"] == "COMPLETE"
         and selected_result["decision"] == "PASS_SELECTED_ACTIVITY_ROW_MASK_SOURCE_ONLY"
         and selected_result["plan"] == SELECTED_REFS["plan"]
         and selected_result["row_masks"] == SELECTED_REFS["row_masks"]
         and selected_result["activity_semantics_selected"] is True
         and selected_result["five_tf_target_locked"] is False
         and selected_result["model_forwards"] == selected_result["optimizer_steps"] == 0
         and selected_result["test_data_used"] is False
         and selected_terminal["status"] == "COMPLETE"
         and selected_terminal["decision"] == selected_result["decision"]
         and selected_terminal["exit_code"] == 0
         and selected_terminal["result"] == SELECTED_REFS["result"]
         and selected_terminal["row_masks"] == SELECTED_REFS["row_masks"]
         and selected_terminal["test_data_used"] is False
         and selected_plan["stage"] == "market_state_selected_activity_masks_20261107_001"
         and selected_plan["inputs"]["full_plan"] == FULL_REFS["plan"]
         and selected_plan["inputs"]["full_masks"] == FULL_REFS["row_masks"]
         and selected_plan["inputs"]["single_plan"] == m5.PREVIOUS_REFS["plan"]
         and selected_plan["inputs"]["single_masks"] == m5.PREVIOUS_REFS["row_masks"]
         and selected_plan["test_data_used"] is False, "SELECTED_RECEIPTS")
    for tf, _, _, count in TF_SPECS:
        need(sum(full_result["monthly"][month][tf]["down"] +
                 full_result["monthly"][month][tf]["up"]
                 for month in full_result["monthly"]) == count
             and sum(single_result["monthly"][month][tf]["down"] +
                     single_result["monthly"][month][tf]["up"]
                     for month in single_result["monthly"]) == count,
             "SIDE_COUNT_" + tf)
    return full_plan, full_result, single_plan, single_result


def inspect_plan(plan_ref: dict) -> tuple[dict, dict, dict, tuple[str, str, str]]:
    plan = base.read_json(plan_ref)
    need(set(plan) == {"schema_version", "stage", "status", "inputs",
                       "source_owners", "scope", "limits", "output_dir", "test_data_used"}
         and plan["schema_version"] == PLAN_SCHEMA
         and plan["stage"] == STAGE
         and plan["status"] == "PREPARED_NOT_ADMITTED"
         and plan["inputs"] == INPUTS
         and plan["scope"] == SCOPE
         and plan["limits"] == LIMITS
         and plan["test_data_used"] is False,
         "PLAN")
    state = m5.git_state()
    need(state[1:] == ("work/gx1-current", ""), "SOURCE_TREE")
    need(set(plan["source_owners"]) == set(SOURCE_PATHS), "SOURCE_KEYS")
    owner_stats = {}
    for name, rel in SOURCE_PATHS.items():
        item = plan["source_owners"][name]
        need(item["path"] == str(REPO / rel), "SOURCE_PATH_" + name)
        _, owner_stats[name] = base.checked(item)
    out = Path(plan["output_dir"])
    need(out.is_absolute() and out.parent == Path("/home/andre2/GX1_RUNS")
         and out.name == STAGE.upper() and out.parent.is_dir(), "OUTPUT_SCOPE")
    full_plan, full_result, single_plan, single_result = small_receipts()
    need(not any(x.split(".")[0] in ("torch", "sklearn", "joblib")
                 for x in sys.modules), "MODEL_IMPORT")
    return plan, {"full_plan": full_plan, "full_result": full_result,
                  "single_plan": single_plan, "single_result": single_result}, owner_stats, state


def preflight_read_only(plan_ref: dict) -> dict:
    plan, _old, owners, state = inspect_plan(plan_ref)
    need(not Path(plan["output_dir"]).exists(), "PREFLIGHT_OUTPUT_EXISTS")
    return {"schema_version": "gx1_mtf_state_side_locality_preflight_v1",
            "status": "READY_NO_CLAIM", "stage": STAGE,
            "plan": plan_ref, "source_head": state[0],
            "source_owner_refs_checked": len(owners),
            "prior_small_receipts_checked": 9,
            "native_parts_or_raw_payload_read": False,
            "claim_created": False, "model_forwards": 0,
            "optimizer_steps": 0, "test_data_used": False,
            "output_root": plan["output_dir"]}


def read_claim(plan: dict, plan_ref: dict, state: tuple[str, str, str]) -> tuple[dict, dict, tuple[int, ...]]:
    out = Path(plan["output_dir"])
    need(out.is_dir() and not out.is_symlink()
         and {p.name for p in out.iterdir()} == {"CLAIM.json"}, "CLAIM_ONLY")
    path = out / "CLAIM.json"
    before = base.regular(path)
    raw = path.read_bytes()
    need(len(raw) <= 1024 * 1024 and base.regular(path) == before, "CLAIM_BYTES")
    claim = json.loads(raw)
    need(set(claim) == {"schema_version", "stage", "plan", "policy",
                        "source_head", "worker", "source_only_audit", "test_data_used"}
         and claim["schema_version"] == CLAIM_SCHEMA
         and claim["stage"] == STAGE and claim["plan"] == plan_ref
         and claim["source_head"] == state[0]
         and claim["worker"] == plan["source_owners"]["worker"]
         and claim["source_only_audit"] is True
         and claim["test_data_used"] is False
         and claim["policy"]["path"] == str(REPO / "NEXT_RUN_POLICY.json"), "CLAIM")
    return claim, {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}, before


def cap_preflight(plan: dict, plan_ref: dict, state: tuple[str, str, str]) -> dict:
    claim, claim_ref, claim_stat = read_claim(plan, plan_ref, state)
    policy = base.read_json(claim["policy"])
    gate = policy.get(STAGE)
    need(isinstance(gate, dict)
         and gate.get("status") == "ADMITTED"
         and gate.get("plan") == plan_ref
         and gate.get("worker") == plan["source_owners"]["worker"]
         and gate.get("output_root") == plan["output_dir"]
         and gate.get("execution_authorized") is True
         and gate.get("data_reads_authorized") is True
         and gate.get("claimed_invocations") == gate.get("completed_invocations") == 0
         and gate.get("max_invocations") == 1 and gate.get("relaunch_allowed") is False
         and gate.get("memory_max") == "4G" and gate.get("swap_max") == "512M"
         and gate.get("wall_seconds") == 1800
         and gate.get("model_forwards_authorized") is False
         and gate.get("optimizer_steps_authorized") is False
         and gate.get("checkpoint_reads_authorized") is False
         and gate.get("broker_access_allowed") is False
         and gate.get("test_data_used") is False
         and gate.get("training_authorized") is False
         and policy.get("training_enabled") is False
         and policy.get("native_training_enabled") is False
         and policy.get("native_launch_authorized") is False,
         "POLICY")
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap["memory_max"] == cap["memory_high"] == 4294967296
         and cap["swap"] == 536870912 and cap["pids"] == 64
         and cap["numerical_threads"] == 1
         and lock["scope"] == cap["cgroup_path"][len("/sys/fs/cgroup"):], "CAP")
    return {"claim": claim_ref, "claim_stat": claim_stat,
            "cap": cap, "lock": lock}


def load_masks() -> dict:
    arrays = {}
    for name, refs in INPUTS.items():
        raw, _ = base.checked(refs["row_masks"], max_bytes=64 * 1024 * 1024)
        need(raw is not None, "MASK_BYTES")
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            expected_keys = (
                {"issue_ns", "m5_source_row", "lead_minutes", "allowed_set", "m1_source_row"}
                if name == "full" else
                {"issue_ns", "m5_source_row", "lead_minutes", "allowed_set",
                 "m1_source_row", "complete_range_atr", "complete_range_atr_valid",
                 "atr_to_quoted_spread", "atr_to_quoted_spread_valid"}
                if name == "single" else
                {"issue_ns", "m5_source_row", "lead_minutes", "allowed_set",
                 "m1_source_row", "activity_active", "activity_observed",
                 "side_observed", "side_up", "singleton_observed"})
            need(set(z.files) == expected_keys, "MASK_KEYS")
            arrays[name] = {key: np.array(z[key]) for key in
                            ("issue_ns", "m5_source_row", "lead_minutes",
                             "allowed_set", "m1_source_row")}
            if name == "selected":
                bits = arrays[name]["allowed_set"]
                need(np.array_equal(z["activity_observed"], bits != 7)
                     and np.array_equal(z["activity_active"], np.isin(bits, (1, 4, 5)))
                     and np.array_equal(z["side_observed"], np.isin(bits, (1, 4)))
                     and np.array_equal(z["side_up"], bits == 4)
                     and np.array_equal(z["singleton_observed"], np.isin(bits, (1, 2, 4))),
                     "SELECTED_EXPLICIT_MASKS")
    a, b, c = arrays["full"], arrays["single"], arrays["selected"]
    need(a["issue_ns"].shape == (SCOPE["issue_rows"],)
         and a["issue_ns"].dtype == np.int64
         and np.all(np.diff(a["issue_ns"]) > 0)
         and a["m5_source_row"].shape == a["issue_ns"].shape
         and a["allowed_set"].shape == a["m1_source_row"].shape ==
             (SCOPE["issue_rows"], len(base.LEADS))
         and a["allowed_set"].dtype == np.uint8
         and np.array_equal(a["lead_minutes"], np.asarray(base.LEADS))
         and all(np.array_equal(a[key], b[key]) and np.array_equal(a[key], c[key])
                 for key in ("issue_ns", "m5_source_row", "lead_minutes", "m1_source_row"))
         and np.all(np.isin(a["allowed_set"], (1, 2, 4, 5, 7))),
         "MASK_AXES")
    for tf, col, _, count in TF_SPECS:
        selected = np.isin(a["allowed_set"][:, col], (1, 4))
        need(int(selected.sum()) == count
             and np.array_equal(a["allowed_set"][selected, col],
                                b["allowed_set"][selected, col])
             and np.array_equal(a["allowed_set"][selected, col],
                                c["allowed_set"][selected, col])
             and not np.isin(b["allowed_set"][~selected, col], (1, 4)).any()
             and not np.isin(c["allowed_set"][~selected, col], (1, 4)).any(),
             "MASK_SIDES_" + tf)
    return c


def cold_side(row: int, window: int, open_: np.ndarray, high: np.ndarray,
              low: np.ndarray, close: np.ndarray, threshold: np.ndarray,
              unknown_gap: np.ndarray, target) -> tuple[int, bool]:
    """Reset at the start of the last W observed M1 bars; replay their true thresholds."""
    start = row - window + 1
    need(0 <= start <= row < len(open_)
         and all(len(x) == len(open_) for x in (high, low, close, threshold, unknown_gap)),
         "LOCAL_ROW")
    predecessor = float(open_[start] if start == 0 or unknown_gap[start]
                        else close[start - 1])
    direction = target.UNKNOWN
    lo = hi = predecessor
    for k in range(start, row + 1):
        if unknown_gap[k] or not (threshold[k] > 0):
            direction = target.UNKNOWN
            lo = hi = float(open_[k])
        if threshold[k] > 0:
            direction, lo, hi, _ = target._step_direction(
                direction, lo, hi, float(high[k]), float(low[k]),
                float(close[k]), float(threshold[k]))
    return direction, bool(unknown_gap[start])


def summarize(indices: np.ndarray, local: np.ndarray, expected: np.ndarray,
              age_bars: np.ndarray, before_issue: np.ndarray,
              flat_since: np.ndarray, closure_touch: np.ndarray) -> dict:
    idx = np.asarray(indices, dtype=np.int64)
    same = local[idx] == expected[idx]
    unknown = local[idx] == -1
    opposite = ~(same | unknown)
    need(np.all(np.isin(local[idx], (-1, 0, 2)))
         and np.all(np.isin(expected[idx], (0, 2))), "SUMMARY_SIDE")
    return {"endpoints": len(idx),
            "local_same": int(same.sum()),
            "local_unknown": int(unknown.sum()),
            "local_opposite": int(opposite.sum()),
            "side_entry_older_than_local_window": int((age_bars[idx] >= 0).sum()),
            "side_entry_at_or_before_issue": int(before_issue[idx].sum()),
            "flat_since_side_entry": int(flat_since[idx].sum()),
            "known_closure_touching_window": int(closure_touch[idx].sum())}


def run(plan_path: Path, plan_sha256: str) -> dict:
    from gx1.contracts import entry_dynamic_mtf_state_targets_v1 as target
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    plan_ref = {"path": str(plan_path), "sha256": plan_sha256}
    plan, old, owner_stats, source_state = inspect_plan(plan_ref)
    caps = cap_preflight(plan, plan_ref, source_state)
    out = Path(plan["output_dir"])
    old_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError("MTF_SIDE_LOCALITY_WALL_LIMIT")))
    signal.alarm(LIMITS["wall_seconds"])
    started = time.monotonic()
    try:
        saved = load_masks()
        input_stats = {ref["path"]: base.regular(Path(ref["path"]))
                       for refs in INPUTS.values() for ref in refs.values()}
        old_plan = old["full_plan"]
        schedule, authority = base.read_closure(old_plan)
        m1, m5_source, issues, m5_rows, source_stats = base.load_sources(old_plan, authority)
        need(np.array_equal(issues, saved["issue_ns"])
             and np.array_equal(m5_rows, saved["m5_source_row"]), "ISSUE_PARITY")
        cutoff = base.pd_ns(base.TRAIN_END)
        need(target.MINIMUM_MOVE_ATR == 0.5
             and tuple(target.TIMEFRAMES) == ("M5", "M15", "H1", "H4", "D1")
             and {tf: target.LOCAL_WINDOW_M1_BARS[tf] for tf, _, _, _ in TF_SPECS}
                 == SCOPE["local_observed_m1_bars"], "TARGET_INITIAL")
        atr = target.build_causal_tf_atr(m5_source, cutoff_ns=cutoff)
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
        need(int(unknown_gap.sum()) == authority["unknown_source_gap_count"], "GAP_COUNT")
        open_ = m1["open"].to_numpy(np.float64)
        high = m1["high"].to_numpy(np.float64)
        low = m1["low"].to_numpy(np.float64)
        close = m1["close"].to_numpy(np.float64)
        monthly = {}
        totals = {}
        mismatch_examples = {}
        edges = base.month_edges()
        for tf, col, window, count in TF_SPECS:
            selected = np.flatnonzero(np.isin(saved["allowed_set"][:, col], (1, 4)))
            rows = saved["m1_source_row"][selected, col] - SOURCE_ROW_OFFSET
            expected = np.where(saved["allowed_set"][selected, col] == 1,
                                target.DOWN, target.UP).astype(np.int8)
            need(len(rows) == count and np.all((window - 1 <= rows) & (rows < len(axis)))
                 and np.array_equal(axis[rows], issues[selected] +
                                    base.LEADS[col] * base.MINUTE_NS)
                 and np.array_equal(tape["state"][rows, col], expected)
                 and np.all(tape["observed"][rows, col])
                 and np.all(tape["local_range"][rows, col] >= tape["threshold"][rows, col]),
                 "ENDPOINTS_" + tf)
            theta = np.asarray(tape["threshold"][:, col])
            full, entry, flat_row = m5.replay_side_entry(
                open_, high, low, close, theta, unknown_gap,
                (tape["state"][:, col] == target.FLAT) & tape["observed"][:, col], target)
            need(np.array_equal(full[rows], expected)
                 and np.all(entry[rows] >= 0), "FULL_SIDE_" + tf)
            local = np.full(len(issues), target.UNKNOWN, np.int8)
            observed = np.full(len(issues), target.UNKNOWN, np.int8)
            age = np.full(len(issues), -1, np.int32)
            before_issue = np.zeros(len(issues), bool)
            flat_since = np.zeros(len(issues), bool)
            closure_touch = np.zeros(len(issues), bool)
            for i, row, side in zip(selected, rows, expected):
                row, i = int(row), int(i)
                start = row - window + 1
                need(not np.any(unknown_gap[start + 1:row + 1]), "COMPLETE_" + tf)
                got, _ = cold_side(row, window, open_, high, low, close,
                                   theta, unknown_gap, target)
                local[i], observed[i] = got, side
                age[i] = row - int(entry[row])
                before_issue[i] = axis[int(entry[row])] <= issues[i]
                flat_since[i] = int(flat_row[row]) > int(entry[row])
                successors = np.flatnonzero(
                    np.diff(axis[start:row + 1]) != base.MINUTE_NS) + start + 1
                need(not np.any(unknown_gap[successors]), "KNOWN_CLOSURE_" + tf)
                closure_touch[i] = bool(len(successors)) or bool(
                    start > 0 and axis[start] - axis[start - 1] != base.MINUTE_NS
                    and not unknown_gap[start])
            old_entry_age = np.where(age >= window, age, -1)
            def report_for(indices):
                report = summarize(indices, local, observed,
                                   old_entry_age,
                                   before_issue, flat_since, closure_touch)
                need(report["endpoints"] == report["local_same"] +
                     report["local_unknown"] + report["local_opposite"],
                     "PARTITION_" + tf)
                return report
            totals[tf] = report_for(selected)
            monthly[tf] = {}
            for lo, hi in zip(edges[:-1], edges[1:]):
                name = datetime.fromtimestamp(lo / 1e9, timezone.utc).strftime("%Y-%m")
                within = selected[(issues[selected] >= lo) & (issues[selected] < hi)]
                monthly[tf][name] = report_for(within)
                need(monthly[tf][name]["endpoints"] ==
                     old["full_result"]["monthly"][name][tf]["down"] +
                     old["full_result"]["monthly"][name][tf]["up"],
                     "MONTH_COUNT_" + tf)
            need(sum(x["endpoints"] for x in monthly[tf].values()) == count,
                 "MONTH_SUM_" + tf)
            bad = selected[local[selected] != observed[selected]]
            mismatch_examples[tf] = [
                {"issue_ns": int(issues[i]), "m1_source_row": int(saved["m1_source_row"][i, col]),
                 "old_side": int(observed[i]), "cold_side": int(local[i])}
                for i in bad[:24]]
        need(m5.git_state() == source_state, "FINAL_SOURCE")
        for path, stat in source_stats.items():
            need(base.regular(Path(path)) == stat, "FINAL_RAW_" + Path(path).name)
        for path, stat in input_stats.items():
            need(base.regular(Path(path)) == stat, "FINAL_INPUT_" + Path(path).name)
        for name, stat in owner_stats.items():
            need(base.regular(Path(plan["source_owners"][name]["path"])) == stat,
                 "FINAL_OWNER_" + name)
        need({p.name for p in out.iterdir()} == {"CLAIM.json"}
             and base.regular(out / "CLAIM.json") == caps["claim_stat"], "FINAL_CLAIM")
        result = {
            "schema_version": "gx1_mtf_state_side_locality_result_v1",
            "status": "COMPLETE", "decision": "DIAGNOSTIC_ONLY",
            "plan": plan_ref, "claim": caps["claim"], "source_head": source_state[0],
            "old_receipts": INPUTS, "source_owners": plan["source_owners"],
            "source_refs": {"m1": old_plan["m1"], "m5": old_plan["m5"],
                            "closure": old_plan["closure"]},
            "source_read_scope": "TRAIN M1 OHLC and historical M5 OHLC; Jan2026 VAL M5 boundary group decoded then excluded before ATR/labels/summary. No TEST.",
            "m1_rows": len(axis), "m5_issues": len(issues),
            "factor": 1.0, "observed_m1_windows": SCOPE["local_observed_m1_bars"],
            "totals": totals, "monthly": monthly,
            "mismatch_examples": mismatch_examples,
            "model_forwards": 0, "optimizer_steps": 0,
            "test_data_used": False, "target_locked": False,
            "limit": "Overlapping endpoints; cold replay diagnoses side provenance only, not a new side label, state duration, prediction or economic edge.",
            "elapsed_seconds": time.monotonic() - started,
            "cap": {"execution": caps["cap"], "lock": caps["lock"]},
        }
        _publish_bytes(out / "RESULT.json", base.encode(result))
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
