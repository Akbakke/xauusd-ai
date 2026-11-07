"""One-use TRAIN-only correction of seven unproven local side labels.

Only the seven exact singleton cells measured by the capped four-TF locality
receipt become ACTIVE with unknown side. Activity, all other labels, clocks,
and source rows remain unchanged. This is source-mask consistency, not a fit.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode = True
import numpy as np

from gx1.audit import verify_mtf_state_side_locality_v1 as mtf

base = mtf.base
REPO = base.REPO
STAGE = "market_state_local_side_exceptions_20261107_001"
PLAN_SCHEMA = "gx1_market_state_local_side_exceptions_plan_v1"
CLAIM_SCHEMA = "gx1_market_state_local_side_exceptions_claim_v1"
SIDE_ROOT = Path("/home/andre2/GX1_RUNS/MTF_STATE_SIDE_LOCALITY_20261107_001")
INPUTS = {
    "selected": mtf.SELECTED_REFS,
    "side": {
        "plan": {"path": str(REPO / "configs/research/MTF_STATE_SIDE_LOCALITY_20261107_001.json"),
                 "sha256": "5b5fe43b92bff84d4405d81b26e63224b47c1cfba995438a0ce24791045ac9aa"},
        "result": {"path": str(SIDE_ROOT / "RESULT.json"),
                   "sha256": "5d65c5aa5f416feaaf11b0e8570ece2f7455a6b65694bb87aa98e844ed12ac9e"},
        "terminal": {"path": str(SIDE_ROOT / "TERMINAL.json"),
                     "sha256": "eda79cacb5711041837b91ebc2b1a285ea5d0ab1d47d685921e5ebb7591eb3fd"},
    },
}
SOURCE_PATHS = {
    "worker": "gx1/audit/verify_market_state_local_side_exceptions_v1.py",
    "mtf_locality": "gx1/audit/verify_mtf_state_side_locality_v1.py",
    "m5_locality": "gx1/audit/verify_m5_state_side_locality_v1.py",
    "full_loader": "gx1/audit/build_full_train_market_state_endpoint_candidate_v1.py",
    "capped_contract": "gx1/contracts/gx1_capped_execution_v1.py",
    "capped_runner": "scripts/gx1_capped_run.sh",
    "publication": "gx1/contracts/immutable_event_authority_v1.py",
}
TFS = ("M5", "M15", "H1", "H4", "D1")
VALUES = (1, 2, 4, 5, 7)
KEYS = {"issue_ns", "m5_source_row", "lead_minutes", "m1_source_row",
        "allowed_set", "activity_observed", "activity_active",
        "side_observed", "side_up", "singleton_observed"}
EXCEPTIONS = (
    ("H4", 1724782500000000000, 5309577, 2),
    ("H4", 1727806800000000000, 5343824, 0),
    ("H4", 1740520800000000000, 5484926, 2),
    ("D1", 1740957600000000000, 5490270, 2),
    ("H4", 1746036300000000000, 5546658, 0),
    ("H4", 1746468300000000000, 5550786, 2),
    ("H4", 1756836300000000000, 5668426, 2),
)
EXPECTED_CORRECTED_COUNTS = {
    "H4": {"DOWN": 20774, "FLAT": 7875, "UP": 25312,
           "ACTIVE": 54455, "ALL": 10290},
    "D1": {"DOWN": 14910, "FLAT": 2409, "UP": 21722,
           "ACTIVE": 43546, "ALL": 36119},
}
EXPECTED_SUMMARY = {"activity_observed_cells": 539544,
                    "side_observed_cells": 212434,
                    "singleton_observed_cells": 255016}
SCOPE = {"train_start_utc": base.TRAIN_START,
         "train_end_utc_exclusive": base.TRAIN_END,
         "issue_rows": 118706, "timeframes": list(TFS),
         "exact_side_demotions": {"H4": 6, "D1": 1},
         "test_data_used": False}
LIMITS = {"memory_max": "4G", "swap_max": "512M", "wall_seconds": 1800,
          "execution_class": "audit", "model_forwards": 0, "optimizer_steps": 0}


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError("LOCAL_SIDE_EXCEPTIONS_" + reason)


def file_ref(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def month_name(issue_ns: int) -> str:
    return datetime.fromtimestamp(issue_ns / 1e9, timezone.utc).strftime("%Y-%m")


def check_small_receipts() -> tuple[dict, dict]:
    """Hashbound JSON lineage only. No NPZ, raw price, or CLAIM read."""
    mtf.small_receipts()
    selected = base.read_json(INPUTS["selected"]["result"])
    side_plan = base.read_json(INPUTS["side"]["plan"])
    side = base.read_json(INPUTS["side"]["result"])
    terminal = base.read_json(INPUTS["side"]["terminal"])
    need(side_plan["stage"] == mtf.STAGE
         and side_plan["inputs"] == mtf.INPUTS
         and side_plan["scope"] == mtf.SCOPE
         and side_plan["limits"] == mtf.LIMITS
         and side_plan["test_data_used"] is False
         and side["schema_version"] == "gx1_mtf_state_side_locality_result_v1"
         and side["status"] == "COMPLETE"
         and side["decision"] == "DIAGNOSTIC_ONLY"
         and side["plan"] == INPUTS["side"]["plan"]
         and side["old_receipts"] == mtf.INPUTS
         and side["m1_rows"] == base.M1_ROWS
         and side["m5_issues"] == SCOPE["issue_rows"]
         and side["model_forwards"] == side["optimizer_steps"] == 0
         and side["test_data_used"] is False and side["target_locked"] is False
         and terminal["status"] == "COMPLETE"
         and terminal["decision"] == "DIAGNOSTIC_ONLY"
         and terminal["exit_code"] == 0
         and terminal["result"] == INPUTS["side"]["result"]
         and terminal["test_data_used"] is False
         and selected["status"] == "COMPLETE"
         and selected["decision"] == "PASS_SELECTED_ACTIVITY_ROW_MASK_SOURCE_ONLY"
         and selected["row_masks"] == INPUTS["selected"]["row_masks"]
         and selected["activity_semantics_selected"] is True
         and selected["five_tf_target_locked"] is False,
         "RECEIPT_BINDING")
    examples = side["mismatch_examples"]
    need(set(examples) == {"M15", "H1", "H4", "D1"}
         and set(side["totals"]) == set(examples)
         and sum(x["endpoints"] for x in side["totals"].values()) == 172771
         and sum(x["local_unknown"] for x in side["totals"].values()) == 7
         and sum(x["local_opposite"] for x in side["totals"].values()) == 0
         and side["totals"]["H4"]["local_unknown"] == 6
         and side["totals"]["D1"]["local_unknown"] == 1
         and len(examples["M15"]) == len(examples["H1"]) == 0,
         "LOCALITY_TOTALS")
    actual = [(tf, x["issue_ns"], x["m1_source_row"], x["old_side"])
              for tf, items in examples.items() for x in items]
    need(len(actual) == len(set(actual)) == len(EXCEPTIONS) == 7
         and set(actual) == set(EXCEPTIONS)
         and all(x["cold_side"] == -1 for items in examples.values() for x in items),
         "EXACT_SEVEN")
    months = [month_name(int(edge)) for edge in base.month_edges()[:-1]]
    need(set(selected["monthly"]) == set(months)
         and set(side["monthly"]) == {"M15", "H1", "H4", "D1"},
         "MONTH_KEYS")
    for tf in side["monthly"]:
        need(set(side["monthly"][tf]) == set(months), "MONTH_TF_" + tf)
        for month in months:
            expected = sum(1 for x in EXCEPTIONS if x[0] == tf and month_name(x[1]) == month)
            need(side["monthly"][tf][month]["local_unknown"] == expected
                 and side["monthly"][tf][month]["local_opposite"] == 0,
                 "MONTH_EXCEPTIONS_" + tf + "_" + month)
    return selected, side


def inspect_plan(plan_ref: dict) -> tuple[dict, dict, dict, dict, tuple[str, str, str]]:
    plan = base.read_json(plan_ref)
    need(set(plan) == {"schema_version", "stage", "status", "inputs",
                       "source_owners", "scope", "limits", "output_dir",
                       "test_data_used"}
         and plan["schema_version"] == PLAN_SCHEMA
         and plan["stage"] == STAGE and plan["status"] == "PREPARED_NOT_ADMITTED"
         and plan["inputs"] == INPUTS and plan["scope"] == SCOPE
         and plan["limits"] == LIMITS and plan["test_data_used"] is False,
         "PLAN")
    state = mtf.m5.git_state()
    need(Path.cwd() == REPO and state[1:] == ("work/gx1-current", ""),
         "SOURCE_TREE")
    output = Path(plan["output_dir"])
    need(output == Path("/home/andre2/GX1_RUNS") / STAGE.upper()
         and output.parent.is_dir() and set(plan["source_owners"]) == set(SOURCE_PATHS),
         "OUTPUT_AND_SOURCES")
    owner_stats = {}
    for name, rel in SOURCE_PATHS.items():
        ref = plan["source_owners"][name]
        need(ref["path"] == str(REPO / rel), "SOURCE_PATH_" + name)
        _, owner_stats[name] = base.checked(ref)
    selected, side = check_small_receipts()
    need(not any(x.split(".")[0] in ("torch", "sklearn", "joblib")
                 for x in sys.modules), "MODEL_IMPORT")
    return plan, selected, side, owner_stats, state


def preflight_read_only(plan_ref: dict) -> dict:
    plan, _selected, _side, stats, state = inspect_plan(plan_ref)
    need(not Path(plan["output_dir"]).exists(), "PRECLAIM_OUTPUT_EXISTS")
    return {"schema_version": "gx1_market_state_local_side_exceptions_preflight_v1",
            "status": "READY_NO_CLAIM_NO_PAYLOAD", "stage": STAGE,
            "plan": plan_ref, "source_head": state[0],
            "source_owner_refs_checked": len(stats),
            "prior_small_receipts_checked": 13,
            "exception_cells": len(EXCEPTIONS),
            "claim_created": False, "npz_or_raw_payload_read": False,
            "model_forwards": 0, "optimizer_steps": 0, "test_data_used": False}


def check_claim_and_policy(plan: dict, plan_ref: dict,
                           state: tuple[str, str, str]) -> dict:
    out = Path(plan["output_dir"])
    need(out.is_dir() and not out.is_symlink()
         and {p.name for p in out.iterdir()} == {"CLAIM.json"}, "CLAIM_ONLY")
    path = out / "CLAIM.json"
    before = base.regular(path)
    need(before[3] <= 1024 * 1024, "CLAIM_SIZE")
    raw = path.read_bytes()
    need(base.regular(path) == before, "CLAIM_BOOKEND")
    claim = json.loads(raw)
    need(set(claim) == {"schema_version", "stage", "plan", "policy",
                        "source_head", "worker", "source_only_correction",
                        "test_data_used"}
         and claim["schema_version"] == CLAIM_SCHEMA
         and claim["stage"] == STAGE and claim["plan"] == plan_ref
         and claim["source_head"] == state[0]
         and claim["worker"] == plan["source_owners"]["worker"]
         and claim["source_only_correction"] is True
         and claim["test_data_used"] is False
         and claim["policy"]["path"] == str(REPO / "NEXT_RUN_POLICY.json"),
         "CLAIM")
    policy = base.read_json(claim["policy"])
    gate = policy.get(STAGE)
    need(isinstance(gate, dict) and gate.get("status") == "ADMITTED"
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
         and gate.get("training_authorized") is False
         and gate.get("test_data_used") is False
         and policy.get("market_state_activity_semantics", {}).get("status")
             == "OPERATOR_SELECTED_CANDIDATE_FULL_TARGET_PENDING"
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
         and lock["scope"] == cap["cgroup_path"][len("/sys/fs/cgroup"):],
         "CAP")
    return {"claim": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()},
            "claim_stat": before, "cap": cap, "lock": lock}


def load_selected(ref: dict, selected_result: dict) -> dict:
    raw, _ = base.checked(ref, max_bytes=64 * 1024 * 1024)
    need(raw is not None, "MASK_BYTES")
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        need(set(z.files) == KEYS, "MASK_KEYS")
        arrays = {key: np.array(z[key]) for key in KEYS}
    issue, allowed = arrays["issue_ns"], arrays["allowed_set"]
    need(issue.shape == (SCOPE["issue_rows"],) and issue.dtype == np.int64
         and np.all(np.diff(issue) > 0)
         and issue[0] >= base.month_edges()[0] and issue[-1] < base.month_edges()[-1]
         and allowed.shape == arrays["m1_source_row"].shape ==
             (len(issue), len(TFS)) and allowed.dtype == np.uint8
         and arrays["m5_source_row"].shape == issue.shape
         and np.array_equal(arrays["lead_minutes"], np.asarray(base.LEADS, np.int16))
         and np.all(np.isin(allowed, VALUES))
         and np.array_equal(arrays["activity_observed"], allowed != 7)
         and np.array_equal(arrays["activity_active"], np.isin(allowed, (1, 4, 5)))
         and np.array_equal(arrays["side_observed"], np.isin(allowed, (1, 4)))
         and np.array_equal(arrays["side_up"], allowed == 4)
         and np.array_equal(arrays["singleton_observed"], np.isin(allowed, (1, 2, 4))),
         "MASK_CONTRACT")
    need(monthly_counts(issue, allowed) == selected_result["monthly"],
         "SAVED_MONTHLY_COUNTS")
    return arrays


def monthly_counts(issue: np.ndarray, allowed: np.ndarray) -> dict:
    edges = np.asarray(base.month_edges(), np.int64)
    need(issue.ndim == 1 and allowed.shape == (len(issue), len(TFS))
         and np.all(np.diff(issue) > 0)
         and issue[0] >= edges[0] and issue[-1] < edges[-1],
         "MONTH_AXIS")
    month = np.searchsorted(edges[1:], issue, side="right")
    result = {}
    for i, edge in enumerate(edges[:-1]):
        name = month_name(int(edge))
        result[name] = {tf: {str(v): int(np.count_nonzero(allowed[month == i, j] == v))
                             for v in VALUES}
                        for j, tf in enumerate(TFS)}
    return result


def correct_seven(arrays: dict, exceptions: tuple = EXCEPTIONS) -> tuple[dict, list[dict]]:
    need(set(arrays) == KEYS, "ARRAY_KEYS")
    issue = arrays["issue_ns"]
    before = arrays["allowed_set"]
    need(issue.ndim == 1 and before.shape == (len(issue), len(TFS))
         and len(exceptions) == len(set(exceptions)) == 7,
         "EXCEPTION_SHAPE")
    corrected = {k: np.array(v, copy=True) for k, v in arrays.items()}
    changed = np.zeros(before.shape, bool)
    report = []
    for tf, stamp, source_row, old_side in exceptions:
        need(tf in ("H4", "D1") and old_side in (0, 2), "EXCEPTION_VALUE")
        col = TFS.index(tf)
        idx = int(np.searchsorted(issue, stamp))
        bit = 1 if old_side == 0 else 4
        need(idx < len(issue) and int(issue[idx]) == stamp
             and int(arrays["m1_source_row"][idx, col]) == source_row
             and int(before[idx, col]) == bit
             and bool(arrays["activity_observed"][idx, col])
             and bool(arrays["activity_active"][idx, col])
             and bool(arrays["side_observed"][idx, col])
             and bool(arrays["singleton_observed"][idx, col])
             and not changed[idx, col], "EXACT_ROW_" + tf + "_" + str(stamp))
        corrected["allowed_set"][idx, col] = 5
        corrected["side_observed"][idx, col] = False
        corrected["side_up"][idx, col] = False
        corrected["singleton_observed"][idx, col] = False
        changed[idx, col] = True
        report.append({"timeframe": tf, "issue_ns": stamp,
                       "m1_source_row": source_row, "old_allowed_set": bit,
                       "new_allowed_set": 5, "issue_month": month_name(stamp)})
    need(int(changed.sum()) == 7
         and np.array_equal(corrected["allowed_set"][~changed], before[~changed])
         and np.all(corrected["allowed_set"][changed] == 5)
         and all(np.array_equal(corrected[k], arrays[k]) for k in
                 ("issue_ns", "m5_source_row", "lead_minutes", "m1_source_row",
                  "activity_observed", "activity_active"))
         and np.array_equal(corrected["side_observed"],
                            np.isin(corrected["allowed_set"], (1, 4)))
         and np.array_equal(corrected["side_up"], corrected["allowed_set"] == 4)
         and np.array_equal(corrected["singleton_observed"],
                            np.isin(corrected["allowed_set"], (1, 2, 4))),
         "ONLY_SEVEN")
    return corrected, report


def verify_counts(before: dict, after: dict, selected_result: dict,
                  changes: list[dict]) -> dict:
    expected = deepcopy(selected_result["monthly"])
    for row in changes:
        month, tf = row["issue_month"], row["timeframe"]
        expected[month][tf][str(row["old_allowed_set"])] -= 1
        expected[month][tf]["5"] += 1
    actual = monthly_counts(after["issue_ns"], after["allowed_set"])
    need(actual == expected, "CORRECTED_MONTHLY_COUNTS")
    counts = {tf: dict(zip(("DOWN", "FLAT", "UP", "ACTIVE", "ALL"),
                           (int(np.count_nonzero(after["allowed_set"][:, j] == v))
                            for v in VALUES)))
              for j, tf in enumerate(TFS)}
    need(all(counts[tf] == EXPECTED_CORRECTED_COUNTS[tf]
             for tf in EXPECTED_CORRECTED_COUNTS)
         and all(counts[tf] == selected_result["summary"]["counts_by_tf"][tf]
                 for tf in ("M5", "M15", "H1"))
         and all(int(after[key].sum()) == total for key, total in
                 (("activity_observed", EXPECTED_SUMMARY["activity_observed_cells"]),
                  ("side_observed", EXPECTED_SUMMARY["side_observed_cells"]),
                  ("singleton_observed", EXPECTED_SUMMARY["singleton_observed_cells"])))
         and np.array_equal(before["activity_observed"], after["activity_observed"])
         and np.array_equal(before["activity_active"], after["activity_active"]),
         "CORRECTED_TOTALS")
    return {"counts_by_tf": counts,
            **EXPECTED_SUMMARY,
            "monthly": actual}


def run(plan_path: Path, plan_sha256: str) -> dict:
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
    plan_ref = {"path": str(plan_path), "sha256": plan_sha256}
    plan, selected_result, side_result, owner_stats, state = inspect_plan(plan_ref)
    cap = check_claim_and_policy(plan, plan_ref, state)
    old_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError("LOCAL_SIDE_EXCEPTIONS_WALL_LIMIT")))
    signal.alarm(LIMITS["wall_seconds"])
    started = time.monotonic()
    try:
        before = load_selected(INPUTS["selected"]["row_masks"], selected_result)
        after, changes = correct_seven(before)
        counts = verify_counts(before, after, selected_result, changes)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **{name: after[name] for name in sorted(after)})
        payload = buffer.getvalue()
        need(len(payload) < 64 * 1024 * 1024, "OUTPUT_SIZE")
        need(mtf.m5.git_state() == state, "FINAL_SOURCE_TREE")
        for name, stat in owner_stats.items():
            ref = plan["source_owners"][name]
            need(base.regular(Path(ref["path"])) == stat,
                 "FINAL_OWNER_STAT_" + name)
            base.checked(ref)
        for refs in INPUTS.values():
            for ref in refs.values():
                base.checked(ref, max_bytes=64 * 1024 * 1024)
        out = Path(plan["output_dir"])
        need({p.name for p in out.iterdir()} == {"CLAIM.json"}
             and base.regular(out / "CLAIM.json") == cap["claim_stat"]
             and file_ref(out / "CLAIM.json") == cap["claim"],
             "FINAL_CLAIM_ONLY")
        mask_path = out / "ROW_MASKS.npz"
        mask_ref = {"path": str(mask_path),
                    "sha256": hashlib.sha256(payload).hexdigest()}
        result = {
            "schema_version": "gx1_market_state_local_side_exceptions_result_v1",
            "status": "COMPLETE", "decision": "PASS_SEVEN_LOCAL_SIDE_DEMOTIONS_SOURCE_ONLY",
            "plan": plan_ref, "claim": cap["claim"], "source_head": state[0],
            "inputs": INPUTS, "source_owners": plan["source_owners"],
            "row_masks": mask_ref, "scope": SCOPE,
            "changes": changes, "changed_cells": len(changes),
            "summary": {k: v for k, v in counts.items() if k != "monthly"},
            "monthly": counts["monthly"],
            "activity_semantics_selected": True,
            "five_tf_target_locked": False,
            "model_forwards": 0, "optimizer_steps": 0,
            "test_data_used": False,
            "limit": "Seven old singleton sides lack cold local proof and become ACTIVE with unknown side. No raw replay, new side promotion, model learning, prediction or economy.",
            "elapsed_seconds": time.monotonic() - started,
            "cap": {"execution": cap["cap"], "lock": cap["lock"]},
        }
        raw_result = base.encode(result)
        need(len(raw_result) < 16 * 1024 * 1024, "RESULT_SIZE")
        _publish_bytes(mask_path, payload)
        _publish_bytes(out / "RESULT.json", raw_result)
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
    ref = {"path": str(Path(args.plan)), "sha256": args.plan_sha256}
    if args.preflight:
        print(json.dumps(preflight_read_only(ref), sort_keys=True, allow_nan=False))
    else:
        run(Path(args.plan), args.plan_sha256)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
