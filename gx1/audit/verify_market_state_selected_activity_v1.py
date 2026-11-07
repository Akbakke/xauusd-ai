"""One-use TRAIN-only source mask for selected 0.5 ATR activity semantics.

The old two-threshold target supplies conservative side and gap masks. The
independent single-threshold receipt supplies full-window range/ATR. Only
fully observed 0.5..1.0 ATR cells change from ALL to ACTIVE; no side is added.
This audit does not fit a model or qualify a full five-TF learning target.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

from gx1.audit import verify_market_state_gray_physical_v1 as gray

REPO = gray.REPO
base = gray.base
STAGE = "market_state_selected_activity_masks_20261107_001"
PLAN_SCHEMA = "gx1_market_state_selected_activity_plan_v1"
CLAIM_SCHEMA = "gx1_market_state_selected_activity_claim_v1"
FULL_ROOT = gray.FULL_ROOT
SINGLE_ROOT = gray.SINGLE_ROOT
GRAY_ROOT = Path("/home/andre2/GX1_RUNS/MARKET_STATE_GRAY_PHYSICAL_20261107_001")
INPUTS = {
    **gray.INPUTS,
    "gray_plan": {
        "path": str(REPO / "configs/research/MARKET_STATE_GRAY_PHYSICAL_20261107_001.json"),
        "sha256": "791ca45b5e6cda1b7785aeb2be3d6d41e24a9470460eb81396a27cca17954faf",
    },
    "gray_result": {
        "path": str(GRAY_ROOT / "RESULT.json"),
        "sha256": "0b8392ed9713113c64a251edd8e1daa70521c0738a2995e85b17ef9de639581e",
    },
    "gray_terminal": {
        "path": str(GRAY_ROOT / "TERMINAL.json"),
        "sha256": "8f83777d03c90794dbe5c534e1647d41ccb9137e3dcd0c9af1f2168105e871fb",
    },
}
SOURCE_PATHS = {
    "worker": "gx1/audit/verify_market_state_selected_activity_v1.py",
    "gray_worker": "gx1/audit/verify_market_state_gray_physical_v1.py",
    "full_loader": "gx1/audit/build_full_train_market_state_endpoint_candidate_v1.py",
    "single_worker": "gx1/audit/verify_single_threshold_market_state_candidate_v1.py",
    "capped_contract": "gx1/contracts/gx1_capped_execution_v1.py",
    "capped_runner": "scripts/gx1_capped_run.sh",
    "publication": "gx1/contracts/immutable_event_authority_v1.py",
}
TFS = ("M5", "M15", "H1", "H4", "D1")
VALUES = (1, 2, 4, 5, 7)
EXPECTED_GRAY = (62687, 57992, 56316, 54197, 42008)
EXPECTED_COUNTS = {
    "M5": (18921, 8182, 20749, 70346, 508),
    "M15": (21375, 12280, 24036, 59549, 1466),
    "H1": (20265, 11836, 24370, 56632, 5603),
    "H4": (20776, 7875, 25316, 54449, 10290),
    "D1": (14910, 2409, 21723, 43545, 36119),
}
SCOPE = {
    "train_start_utc": base.TRAIN_START,
    "train_end_utc_exclusive": base.TRAIN_END,
    "m5_issues": 118706,
    "lead_minutes": list(base.LEADS),
    "gray_cells": sum(EXPECTED_GRAY),
    "selected_activity_rule": "complete_R_below_0.5ATR_FLAT_at_or_above_ACTIVE_side_only_if_separately_certified",
    "test_data_used": False,
}
LIMITS = {"memory_max": "4G", "swap_max": "512M", "wall_seconds": 1800,
          "execution_class": "audit", "model_forwards": 0, "optimizer_steps": 0}
OUTPUT_KEYS = {"issue_ns", "m5_source_row", "lead_minutes", "m1_source_row",
               "allowed_set", "activity_observed", "activity_active",
               "side_observed", "side_up", "singleton_observed"}


def need(ok: object, reason: str) -> None:
    if not bool(ok):
        raise RuntimeError("SELECTED_ACTIVITY_" + reason)


def file_ref(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def inspect_plan(plan_ref: dict) -> tuple[dict, dict, dict, tuple[str, str, str]]:
    """Small receipt/source checks only; safe before CLAIM and without NPZ reads."""
    plan = base.read_json(plan_ref)
    need(set(plan) == {"schema_version", "stage", "status", "inputs",
                       "source_owners", "scope", "limits", "output_dir",
                       "test_data_used"}
         and plan["schema_version"] == PLAN_SCHEMA
         and plan["stage"] == STAGE and plan["status"] == "PREPARED_NOT_ADMITTED"
         and plan["inputs"] == INPUTS and plan["scope"] == SCOPE
         and plan["limits"] == LIMITS and plan["test_data_used"] is False,
         "PLAN")
    state = gray.git_state()
    need(Path.cwd() == REPO and state[1:] == ("work/gx1-current", ""),
         "SOURCE_TREE")
    output = Path(plan["output_dir"])
    need(output == Path("/home/andre2/GX1_RUNS") / STAGE.upper()
         and output.parent.is_dir() and len(plan["source_owners"]) == len(SOURCE_PATHS)
         and set(plan["source_owners"]) == set(SOURCE_PATHS),
         "OUTPUT_AND_OWNERS")
    owners = {}
    for name, rel in SOURCE_PATHS.items():
        ref = plan["source_owners"][name]
        need(ref["path"] == str(REPO / rel), "OWNER_PATH_" + name)
        _, owners[name] = base.checked(ref)
    old_plan = base.read_json(INPUTS["gray_plan"])
    need(old_plan["inputs"] == gray.INPUTS
         and old_plan["schema_version"] == gray.PLAN_SCHEMA
         and old_plan["stage"] == gray.STAGE, "GRAY_PLAN")
    _, single_result = gray.check_plan(old_plan, INPUTS["gray_plan"])
    gray_result = base.read_json(INPUTS["gray_result"])
    gray_terminal = base.read_json(INPUTS["gray_terminal"])
    need(gray_result["schema_version"] == "gx1_market_state_gray_physical_result_v1"
         and gray_result["status"] == "COMPLETE"
         and gray_result["decision"] == "DIAGNOSTIC_ONLY_TARGET_OPEN"
         and gray_result["plan"] == INPUTS["gray_plan"]
         and gray_result["source_refs"]["candidates"] == gray.INPUTS
         and gray_result["m5_issues"] == SCOPE["m5_issues"]
         and gray_result["model_forwards"] == gray_result["optimizer_steps"] == 0
         and gray_result["test_data_used"] is False
         and gray_result["target_locked"] is False
         and gray_terminal["status"] == "COMPLETE"
         and gray_terminal["decision"] == "DIAGNOSTIC_ONLY_TARGET_OPEN"
         and gray_terminal["exit_code"] == 0
         and gray_terminal["result"] == INPUTS["gray_result"]
         and gray_terminal["test_data_used"] is False,
         "GRAY_RECEIPT")
    need(tuple(gray_result["total"][tf]["gray_0_5_to_1_0"]["n"]
               for tf in TFS) == EXPECTED_GRAY, "GRAY_RECEIPT_COUNTS")
    return plan, single_result, owners, state


def preflight_read_only(plan_ref: dict) -> dict:
    plan, _, owners, state = inspect_plan(plan_ref)
    need(not Path(plan["output_dir"]).exists(), "PRECLAIM_OUTPUT_EXISTS")
    return {
        "schema_version": "gx1_market_state_selected_activity_preflight_v1",
        "status": "READY_NO_CLAIM_NO_PAYLOAD", "stage": STAGE,
        "plan": plan_ref, "source_head": state[0],
        "source_owner_refs_checked": len(owners),
        "small_prior_receipts_checked": len(INPUTS) - 2,
        "gray_cells_expected": SCOPE["gray_cells"],
        "claim_created": False, "npz_or_raw_payload_read": False,
        "model_forwards": 0, "optimizer_steps": 0, "test_data_used": False,
    }


def check_claim_and_policy(plan: dict, plan_ref: dict,
                           state: tuple[str, str, str]) -> dict:
    output = Path(plan["output_dir"])
    need(output.is_dir() and not output.is_symlink()
         and {p.name for p in output.iterdir()} == {"CLAIM.json"}, "CLAIM_ONLY_ROOT")
    path = output / "CLAIM.json"
    claim_stat = base.regular(path)
    need(claim_stat[3] <= 1024 * 1024, "CLAIM_SIZE")
    raw = path.read_bytes()
    need(base.regular(path) == claim_stat, "CLAIM_BOOKEND")
    claim = json.loads(raw)
    need(set(claim) == {"schema_version", "stage", "plan", "policy",
                        "source_head", "worker", "source_only_mask",
                        "test_data_used"}
         and claim["schema_version"] == CLAIM_SCHEMA
         and claim["stage"] == STAGE and claim["plan"] == plan_ref
         and claim["worker"] == plan["source_owners"]["worker"]
         and claim["source_head"] == state[0]
         and claim["source_only_mask"] is True
         and claim["test_data_used"] is False, "CLAIM_BINDING")
    policy_ref = claim["policy"]
    need(policy_ref["path"] == str(REPO / "NEXT_RUN_POLICY.json"),
         "POLICY_PATH")
    policy = base.read_json(policy_ref)
    gate = policy.get(STAGE)
    selected = policy.get("market_state_activity_semantics")
    need(isinstance(gate, dict) and gate.get("status") == "ADMITTED"
         and gate.get("plan") == plan_ref
         and gate.get("worker") == plan["source_owners"]["worker"]
         and gate.get("output_root") == str(output)
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
         and isinstance(selected, dict)
         and selected.get("status") == "OPERATOR_SELECTED_CANDIDATE_FULL_TARGET_PENDING"
         and selected.get("complete_local_range_below_atr14") == 0.5
         and selected.get("gray_band_allowed_set") == ["DOWN", "UP"]
         and selected.get("five_tf_rowwise_target_locked") is False
         and policy.get("training_enabled") is False
         and policy.get("native_training_enabled") is False
         and policy.get("native_launch_authorized") is False,
         "POLICY_GATE")
    from gx1.contracts.gx1_capped_execution_v1 import (
        require_capped_cpu_audit_execution, require_capped_lock_ancestry)
    cap = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    need(cap["memory_max"] == cap["memory_high"] == 4294967296
         and cap["swap"] == 536870912 and cap["pids"] == 64
         and cap["numerical_threads"] == 1
         and lock["scope"] == cap["cgroup_path"][len("/sys/fs/cgroup"):],
         "CAP_LOCK")
    need(not any(x.split(".")[0] in ("torch", "sklearn", "joblib")
                 for x in sys.modules), "MODEL_IMPORT")
    return {"claim": {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()},
            "claim_stat": claim_stat,
            "cap": cap, "lock": lock}


def select_masks(old: dict, single: dict, cohorts: dict) -> tuple[dict, dict]:
    """Change only complete gray ALL cells; expose separate true masks."""
    need(set(old) == gray.OLD_KEYS and set(single) == gray.NEW_KEYS
         and set(cohorts) == set(gray.COHORTS), "INPUT_KEYS")
    before = np.asarray(old["allowed_set"])
    ratio = np.asarray(single["complete_range_atr"])
    valid = np.asarray(single["complete_range_atr_valid"])
    gray_cells = np.asarray(cohorts["gray_0_5_to_1_0"])
    need(before.shape == ratio.shape == valid.shape == gray_cells.shape
         == (SCOPE["m5_issues"], len(TFS))
         and before.dtype == np.uint8 and ratio.dtype == np.float64
         and valid.dtype == gray_cells.dtype == np.bool_
         and np.array_equal(gray_cells, valid & (ratio >= 0.5) & (ratio < 1.0))
         and np.all(before[gray_cells] == 7)
         and np.all(single["allowed_set"][gray_cells] == 2)
         and np.array_equal(single["allowed_set"][~gray_cells], before[~gray_cells])
         and tuple(int(gray_cells[:, j].sum()) for j in range(len(TFS)))
             == EXPECTED_GRAY, "GRAY_IDENTITY")
    selected = before.copy()
    selected[gray_cells] = 5
    need(np.array_equal(selected[~gray_cells], before[~gray_cells])
         and np.all(selected[gray_cells] == 5)
         and np.count_nonzero(selected != before) == SCOPE["gray_cells"]
         and np.all(np.isin(selected, VALUES)), "ONLY_GRAY_CHANGED")
    counts = {tf: tuple(int(np.count_nonzero(selected[:, j] == value))
                        for value in VALUES) for j, tf in enumerate(TFS)}
    need(counts == EXPECTED_COUNTS, "EXPECTED_COUNTS")
    observed = selected != 7
    active = (selected == 1) | (selected == 4) | (selected == 5)
    side = (selected == 1) | (selected == 4)
    singleton = side | (selected == 2)
    output = {key: np.array(old[key], copy=True) for key in gray.OLD_KEYS - {"allowed_set"}}
    output.update({"allowed_set": selected,
                   "activity_observed": observed,
                   "activity_active": active,
                   "side_observed": side,
                   "side_up": selected == 4,
                   "singleton_observed": singleton})
    need(set(output) == OUTPUT_KEYS
         and np.all(side <= active) and np.all(singleton <= observed)
         and np.all(~observed[before == 7] | gray_cells[before == 7])
         and not np.any(side[gray_cells]), "OUTPUT_MASKS")
    report = {"gray_by_tf": dict(zip(TFS, EXPECTED_GRAY)),
              "counts_by_tf": {tf: dict(zip(("DOWN", "FLAT", "UP", "ACTIVE", "ALL"),
                                             counts[tf])) for tf in TFS},
              "changed_cells": int(np.count_nonzero(selected != before)),
              "unchanged_cells": int(selected.size - np.count_nonzero(selected != before)),
              "activity_observed_cells": int(observed.sum()),
              "side_observed_cells": int(side.sum()),
              "singleton_observed_cells": int(singleton.sum())}
    return output, report


def monthly_counts(issues: np.ndarray, mask: np.ndarray) -> dict:
    edges = np.asarray(base.month_edges(), dtype=np.int64)
    need(issues.shape == (SCOPE["m5_issues"],) and issues.dtype == np.int64
         and mask.shape == (SCOPE["m5_issues"], len(TFS))
         and mask.dtype == np.uint8 and np.all(np.isin(mask, VALUES))
         and np.all(np.diff(issues) > 0)
         and issues[0] >= edges[0] and issues[-1] < edges[-1], "MONTH_AXIS")
    month = np.searchsorted(edges[1:], issues, side="right")
    need(np.all((month >= 0) & (month < 20)), "MONTH_PARTITION")
    result = {}
    for i in range(20):
        name = time.strftime("%Y-%m", time.gmtime(edges[i] // 10**9))
        result[name] = {tf: {str(value): int(np.count_nonzero(mask[month == i, j] == value))
                             for value in VALUES}
                        for j, tf in enumerate(TFS)}
    need(all(sum(result[name][tf][str(v)] for name in result)
             == int(np.count_nonzero(mask[:, j] == v))
             for j, tf in enumerate(TFS) for v in VALUES), "MONTH_SUMS")
    return result


def verify_monthly_relabels(issues: np.ndarray, before: np.ndarray,
                            after: np.ndarray, single_result: dict) -> dict:
    """Cross-check each month/TF gray promotion against the saved source receipt."""
    edges = np.asarray(base.month_edges(), dtype=np.int64)
    need(issues.shape == (SCOPE["m5_issues"],)
         and before.shape == after.shape == (len(issues), len(TFS))
         and np.all(np.diff(issues) > 0)
         and issues[0] >= edges[0] and issues[-1] < edges[-1],
         "RELABEL_MONTH_AXIS")
    months = [time.strftime("%Y-%m", time.gmtime(edge // 10**9))
              for edge in edges[:-1]]
    need(set(single_result["monthly"]) == set(months), "RELABEL_MONTH_KEYS")
    assigned = np.searchsorted(edges[1:], issues, side="right")
    result = {}
    for i, name in enumerate(months):
        result[name] = {}
        for j, tf in enumerate(TFS):
            changed = (before[assigned == i, j] == 7) & (after[assigned == i, j] == 5)
            observed = int(np.count_nonzero(changed))
            expected = single_result["monthly"][name][tf]["comparison_to_old"]["old_gray_to_flat"]
            need(isinstance(expected, int) and observed == expected,
                 "RELABEL_MONTH_" + name + "_" + tf)
            result[name][tf] = observed
    need(sum(result[name][tf] for name in months for tf in TFS)
         == SCOPE["gray_cells"], "RELABEL_MONTH_TOTAL")
    return result


def run(plan_path: Path, plan_sha256: str) -> dict:
    from gx1.contracts.immutable_event_authority_v1 import _publish_bytes

    plan_ref = {"path": str(plan_path), "sha256": plan_sha256}
    plan, single_result, owners, state = inspect_plan(plan_ref)
    cap = check_claim_and_policy(plan, plan_ref, state)
    old_alarm = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(
        TimeoutError("SELECTED_ACTIVITY_WALL_LIMIT")))
    signal.alarm(LIMITS["wall_seconds"])
    started = time.monotonic()
    try:
        old, single, cohorts = gray.load_masks(single_result)
        arrays, summary = select_masks(old, single, cohorts)
        monthly = monthly_counts(arrays["issue_ns"], arrays["allowed_set"])
        monthly_relabels = verify_monthly_relabels(
            arrays["issue_ns"], old["allowed_set"], arrays["allowed_set"],
            single_result)
        need(np.array_equal(arrays["issue_ns"], single["issue_ns"])
             and np.array_equal(arrays["m5_source_row"], single["m5_source_row"])
             and np.array_equal(arrays["lead_minutes"], single["lead_minutes"])
             and np.array_equal(arrays["m1_source_row"], single["m1_source_row"]),
             "SOURCE_AXES")
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **{name: arrays[name] for name in sorted(arrays)})
        raw_masks = buffer.getvalue()
        need(len(raw_masks) < 64 * 1024 * 1024, "OUTPUT_SIZE")
        need(gray.git_state() == state, "FINAL_SOURCE_TREE")
        for name, ref in plan["source_owners"].items():
            need(base.regular(Path(ref["path"])) == owners[name],
                 "FINAL_OWNER_STAT_" + name)
            base.checked(ref)
        for ref in INPUTS.values():
            base.checked(ref, max_bytes=64 * 1024 * 1024)
        output = Path(plan["output_dir"])
        need({p.name for p in output.iterdir()} == {"CLAIM.json"}
             and base.regular(output / "CLAIM.json") == cap["claim_stat"]
             and file_ref(output / "CLAIM.json") == cap["claim"],
             "FINAL_CLAIM_ONLY_ROOT")
        mask_path = output / "ROW_MASKS.npz"
        mask_ref = {"path": str(mask_path),
                    "sha256": hashlib.sha256(raw_masks).hexdigest()}
        result = {
            "schema_version": "gx1_market_state_selected_activity_result_v1",
            "status": "COMPLETE", "decision": "PASS_SELECTED_ACTIVITY_ROW_MASK_SOURCE_ONLY",
            "plan": plan_ref, "claim": cap["claim"], "source_head": state[0],
            "source_refs": {"inputs": INPUTS, "owners": plan["source_owners"]},
            "row_masks": mask_ref, "scope": SCOPE,
            "selection_rule": "Only fully observed 0.5<=local_range/ATR14<1.0 old ALL=7 becomes ACTIVE=5; no side promotion. Other masks and all source axes are unchanged.",
            "mask_values": {"DOWN": 1, "FLAT": 2, "UP": 4, "ACTIVE": 5, "ALL": 7},
            "summary": summary, "monthly": monthly,
            "monthly_gray_relabels": monthly_relabels,
            "activity_semantics_selected": True,
            "five_tf_target_locked": False,
            "model_forwards": 0, "optimizer_steps": 0,
            "test_data_used": False,
            "elapsed_seconds": time.monotonic() - started,
            "cap": {"execution": cap["cap"], "lock": cap["lock"]},
        }
        raw_result = base.encode(result)
        need(len(raw_result) < 16 * 1024 * 1024, "RESULT_SIZE")
        _publish_bytes(mask_path, raw_masks)
        _publish_bytes(output / "RESULT.json", raw_result)
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
