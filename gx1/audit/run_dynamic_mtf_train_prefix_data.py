"""One capped, data-only publication for the 04 November GX1 fit prefix."""
import argparse
import json
import signal
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.contracts.local_random_access_campaign_v2 import file_sha256, read_bound_json


REPO = Path("/home/andre2/src/GX1_CURRENT")
ROOT = Path("/home/andre2/GX1_RUNS/DYNAMIC_MTF_TRAIN_GATES_20261104_001/DATA_PREFIX_001")
STAGE = "dynamic_mtf_train_prefix_data_20261104_001"
DECEMBER_STAGE = "dynamic_mtf_train_prefix_dec01_20261104_001"
BEAR_STAGE = "dynamic_mtf_gc_d1_bear_source_trace_20261104_001"
ROOTS = {STAGE: ROOT, DECEMBER_STAGE: ROOT.parent / "DATA_PREFIX_DEC01_001",
         BEAR_STAGE: ROOT.parent / "GC_D1_BEAR_SOURCE_TRACE_001"}


def require(condition, reason):
    if not condition:
        raise RuntimeError("GX1_TRAIN_PREFIX_DATA_" + reason)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def publish(path, value, limit):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    existing = sum(p.stat().st_size for p in path.parent.glob("*.json"))
    require(existing + len(data) <= limit, "JSON_BUDGET")
    _publish_bytes(path, data)
    return {"path": str(path), "sha256": file_sha256(path)}


def frozen(binding):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()
    require(Path.cwd() == REPO and git("branch", "--show-current") == "work/gx1-current"
            and git("rev-parse", "HEAD") == binding["source_head"] and not git("status", "--porcelain"),
            "SOURCE_IDENTITY")
    require(file_sha256(REPO / "NEXT_RUN_POLICY.json") == binding["policy_sha256"], "POLICY_CHANGED")


def _prefix_result(policy, stage_name, root, budget):
    stage = policy[stage_name]
    from gx1.audit.dynamic_mtf_train_prefix_targets import materialize_train_prefix_targets
    target_args = {"target_result_ref": stage["target_result"], "prefix_rows": stage["prefix_rows"],
                   "cutoff_ns": stage["fit_cutoff_ns"], "output_dir": root / "FIT_TARGETS"}
    extra = {}
    if stage_name == DECEMBER_STAGE:
        from gx1.audit.dynamic_mtf_train_prefix_inputs import verify_december_prefix_input_support
        support = verify_december_prefix_input_support(policy)
        target_args["admission_stage"] = DECEMBER_STAGE
    else:
        from gx1.audit.dynamic_mtf_train_prefix_inputs import review_train_gc_d1_donor, verify_train_prefix_input_support
        support = verify_train_prefix_input_support(policy)
        support["gc_d1_donor"] = review_train_gc_d1_donor(stage["gc_donor_terminal"], stage["fit_cutoff_ns"])
    allowed_status = ("BLOCK", "FULL_QUALIFICATION_REQUIRED") if stage_name == DECEMBER_STAGE else ("BLOCK",)
    require(support["status"] in allowed_status and support["fit_allowed"] is False
            and support["statistics_fits"] == 0 and support["normalization"] is None, "REQUIRED_INPUT_BLOCK")
    support_ref = publish(root / "PREFIX_INPUT_SUPPORT.json", support, budget)
    built = materialize_train_prefix_targets(**target_args)
    if stage_name == DECEMBER_STAGE:
        events = built["d1_interval_event_support"]
        both_outcomes = all(events[name]["event_parent_issues"] > 0
                            and events[name]["observed_nonevent_issue_minutes"] > 0
                            for name in ("first_exit", "first_exit_then_return"))
        require((support["status"] == "BLOCK") == bool(support["exact_missing_fields"]), "INPUT_STATUS")
        extra = {"admission_stage": stage_name, "geometry": stage["geometry"],
                 "fit_only_measurement": stage["fit_only_measurement"], "input_gate_status": support["status"],
                 "d1_interval_event_support": events, "d1_both_risk_sets_have_events_and_nonevents": both_outcomes,
                 "held_out_reserved_parents": stage["held_out_rows"], "held_out_predictions": 0,
                 "held_out_outcomes_used": False}
    counts_ref = publish(root / "FIT_COUNTS.json", built["counts"], budget)
    prior_ref = publish(root / "FIT_PRIOR.json", built["prior"], budget)
    require(built["clock_support"]["parents"] == stage["prefix_rows"], "CLOCK_POPULATION")
    result_status = ("COMPLETE_DATA_PORT_MODEL_BLOCKED" if support["status"] == "BLOCK"
                     else "COMPLETE_DATA_PORT_FULL_INPUT_QUALIFICATION_REQUIRED")
    return {"status": result_status,
        "fit_prefix": stage["prefix_rows"], "fit_cutoff_ns": stage["fit_cutoff_ns"],
        "input_support": support_ref, "series": built["series"], "counts": counts_ref, "prior": prior_ref,
        "clock_support": built["clock_support"], "calendar_binding": built["calendar_binding"],
        "normalization_fits": 0, "model_calls": 0,
        "optimizer_updates": 0, "oof_predictions": 0, "test_data_used": False, **extra}


def run(binding_sha256, stage_name=STAGE):
    require(stage_name in ROOTS, "STAGE")
    root = ROOTS[stage_name]
    require(root.is_dir() and root.resolve() == root, "OUTPUT_ROOT")
    binding = read_bound_json(root / "EXECUTION_BINDING.json", binding_sha256)
    require(set(binding) == {"stage", "source_head", "policy_sha256"} and binding["stage"] == stage_name, "BINDING")
    frozen(binding)
    policy = read_bound_json(REPO / "NEXT_RUN_POLICY.json", binding["policy_sha256"])
    stage = policy[stage_name]
    prereg = policy["current_work"]["dynamic_mtf_train_gates_20261104"]
    require(prereg["specification"] == stage["specification"]
            and file_sha256(REPO / stage["specification"]["path"]) == stage["specification"]["sha256"]
            and stage["run_root"] == str(root), "SCOPE")
    if stage_name != BEAR_STAGE:
        require(stage["target_result"] == policy["dynamic_mtf_state_target_audit_20261102_005"]["result"],
                "TARGET_AUTHORITY")
    if stage_name == BEAR_STAGE:
        previous = policy[DECEMBER_STAGE]
        require(stage["status"] == "ADMITTED_ONE_TRAIN_GC_D1_SOURCE_TRACE_ONLY"
                and stage["previous_terminal"] == previous["terminal"]
                and previous["claimed_invocations"] == previous["completed_invocations"] == 1
                and previous["execution_authorized"] is False and len(previous["measured"]["missing_fields"]) == 7
                and stage["source_dates"] == 214 and stage["fit_cutoff_ns"] == previous["fit_cutoff_ns"]
                and stage["fit_cutoff_ns"] == 1733011200000000000
                and stage["prefix_rows"] == previous["prefix_rows"] == 41866
                and stage["held_out_rows"] == previous["held_out_rows"] == 5736, "BEAR_TRACE_SCOPE")
    elif stage_name == DECEMBER_STAGE:
        revision = prereg["proposed_data_only_revision"]
        geometry = stage["geometry"]
        require(revision["admitted"] is True and revision["admission_stage"] == stage_name
                and stage["status"] == "ADMITTED_ONE_DECEMBER_DATA_CHECK_NO_MODEL"
                and stage["fit_cutoff_ns"] == 1733011200000000000 and stage["held_out_cutoff_ns"] == 1735689600000000000
                and geometry["fit"]["parents"] == stage["prefix_rows"] == 41866
                and geometry["held_out"]["parents"] == stage["held_out_rows"] == 5736
                and geometry["held_out"]["parent_range"] == [41866, 47602]
                and stage["previous_terminal"] == policy[STAGE]["terminal"]
                and policy[STAGE]["execution_authorized"] is False and policy[STAGE]["completed_invocations"] == 1,
                "DECEMBER_REVISION")
    else:
        require(prereg["fit_parents"] == stage["prefix_rows"] == 29849
                and stage["fit_cutoff_ns"] == 1727740800000000000
                and stage["status"] == "ADMITTED_ONE_DATA_PUBLICATION_NO_FIT", "OCTOBER_SCOPE")
    require(stage["max_invocations"] == 1 and stage["claimed_invocations"] == stage["completed_invocations"] == 0
            and stage["execution_authorized"] is True and stage["fit_target_view_authorized"] is (stage_name != BEAR_STAGE)
            and all(stage[k] is False for k in ("normalization_fit_authorized", "model_calls_authorized",
                    "optimizer_updates_authorized", "training_enabled", "relaunch_allowed", "test_authorized",
                    "broker_authorized", "trading_authorized"))
            and all(policy[k] is False for k in ("training_enabled", "native_training_enabled",
                    "full_epoch_training_allowed", "full_val_allowed")), "ADMISSION")
    require({p.name for p in root.iterdir()} == {"EXECUTION_BINDING.json"}, "CONSUMED")
    guard = require_capped_cpu_audit_execution()
    lock = require_capped_lock_ancestry()
    require(all(guard[k] == stage["limits"][k] for k in ("memory_max", "swap", "pids", "numerical_threads")), "CAPS")
    budget = stage["limits"]["max_json_bytes"]
    claim = publish(root / "CLAIM.json", {"binding_sha256": binding_sha256, "source_head": binding["source_head"],
                    "started_utc": stamp(), "guard": guard, "lock": lock}, budget)

    def deadline(*_args):
        raise TimeoutError("GX1_TRAIN_PREFIX_DATA_DEADLINE")
    signal.signal(signal.SIGALRM, deadline)
    signal.signal(signal.SIGTERM, deadline)
    signal.alarm(stage["limits"]["max_wall_seconds"])
    try:
        if stage_name == BEAR_STAGE:
            from gx1.audit.dynamic_mtf_train_prefix_inputs import review_december_bear_source
            payload, rows = review_december_bear_source(policy[DECEMBER_STAGE], stage)
            payload["row_trace"] = publish(root / "SOURCE_DATES.json", {"rows": rows}, budget)
        else:
            payload = _prefix_result(policy, stage_name, root, budget)
        frozen(binding)
        require(require_capped_lock_ancestry() == lock, "LOCK_CHANGED")
        result = publish(root / "RESULT.json", {**payload, "claim": claim, "specification": stage["specification"],
                         "source_head": binding["source_head"]}, budget)
    except Exception as exc:
        publish(root / "TERMINAL.json", {"status": "FAILED", "claim": claim, "finished_utc": stamp(),
                "exception": type(exc).__name__, "detail": str(exc)[:1200]}, budget)
        raise
    finally:
        signal.alarm(0)
    terminal = publish(root / "TERMINAL.json", {"status": "COMPLETE", "claim": claim, "result": result,
                       "finished_utc": stamp(), "exit_code": 0, "source_unchanged": True,
                       "test_data_used": False}, budget)
    print(json.dumps(terminal))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--stage", choices=tuple(ROOTS), default=STAGE)
    args = parser.parse_args()
    run(args.binding_sha256, args.stage)
