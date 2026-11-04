"""One capped, data-only publication for the 04 November GX1 fit prefix."""
import argparse
import json
import signal
import subprocess
from datetime import datetime, timezone
from time import monotonic
from pathlib import Path

from gx1.contracts.gx1_capped_execution_v1 import require_capped_cpu_audit_execution, require_capped_lock_ancestry
from gx1.contracts.immutable_event_authority_v1 import _publish_bytes
from gx1.contracts.local_random_access_campaign_v2 import (capture_verified_file_snapshot, file_sha256,
                                                         file_verification_scope, read_bound_json)


REPO = Path("/home/andre2/src/GX1_CURRENT")
ROOT = Path("/home/andre2/GX1_RUNS/DYNAMIC_MTF_TRAIN_GATES_20261104_001/DATA_PREFIX_001")
STAGE = "dynamic_mtf_train_prefix_data_20261104_001"
DECEMBER_STAGE = "dynamic_mtf_train_prefix_dec01_20261104_001"
BEAR_STAGE = "dynamic_mtf_gc_d1_bear_source_trace_20261104_001"
MASKED_STAGE = "dynamic_mtf_masked_december_input_normalization_20261104_001"
CANDLE_STAGE = "dynamic_mtf_december_candle_and_plan_20261104_001"
ROOTS = {STAGE: ROOT, DECEMBER_STAGE: ROOT.parent / "DATA_PREFIX_DEC01_001",
         BEAR_STAGE: ROOT.parent / "GC_D1_BEAR_SOURCE_TRACE_001",
         MASKED_STAGE: ROOT.parent / "MASKED_DECEMBER_INPUT_NORMALIZATION_001",
         CANDLE_STAGE: ROOT.parent / "DECEMBER_CANDLE_AND_CANDIDATE_PLAN_001"}


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


def _require_masked_december_stage(policy, stage):
    from gx1.contracts.entry_native_direction_prefix_normalization_v1 import december_zero_support_profile

    previous = policy[DECEMBER_STAGE]
    geometry = policy["current_work"]["dynamic_mtf_train_gates_20261104"]["december_candidate_geometry"]
    endpoint = policy["dynamic_mtf_native_endpoint_loss_step1_20261104_001"]
    raw = policy["dynamic_m5_direction_gc_boolean_physical_prefix_normalization_267"]
    profile = december_zero_support_profile()
    require(stage["status"] == "ADMITTED_ONE_MASKED_TRAIN_PREFIX_NORMALIZATION_NO_MODEL"
            and stage["profile"] == {k: profile[k] for k in ("name", "contract_sha256")}
            and stage["candidate_geometry"] == geometry["binding"]
            and stage["fit_target_view"] == previous["fit_target_view"]
            and stage["prior_input_support"] == previous["input_support"]
            and stage["prior_input_review"] == previous["verified_review"]
            and stage["previous_terminal"] == previous["terminal"]
            and stage["physical_provider_plan"] == raw["plan"]
            and stage["physical_provider_worker"] == raw["worker"]
            and stage["endpoint_loss_arithmetic"] == endpoint["result"]
            and stage["historical_recipe"] == policy["dynamic_mtf_state_onepass_fit_20261103_001"]["recipe"]
            and previous["completed_invocations"] == endpoint["completed_invocations"] == 1
            and previous["execution_authorized"] is False and endpoint["milestone_complete"] is True,
            "MASKED_PROFILE_BINDINGS")
    require(stage["fit_cutoff_ns"] == previous["fit_cutoff_ns"] == 1733011200000000000
            and stage["prefix_rows"] == previous["prefix_rows"] == 41866
            and stage["held_out_rows"] == previous["held_out_rows"] == 5736
            and geometry["fit_parent_range"] == [0, 41866] and geometry["reserved_parent_range"] == [41866, 47602]
            and geometry["run_admission"] is geometry["held_out_outcomes_read"] is False
            and stage["masked_fields"] == previous["measured"]["missing_fields"] and len(stage["masked_fields"]) == 7
            and stage["max_invocations"] == 1 and stage["data_reads_authorized"] is True
            and stage["normalization_fit_authorized"] is True
            and all(stage[k] is False for k in ("held_out_features_authorized", "held_out_outcomes_authorized",
                    "test_authorized", "native_model_authorized", "external_data_authorized")),
            "MASKED_PROFILE_SCOPE")
    for key in ("candidate_geometry", "fit_target_view", "prior_input_support", "prior_input_review", "previous_terminal",
                "endpoint_loss_arithmetic", "physical_provider_plan", "physical_provider_worker", "historical_recipe"):
        ref = stage[key]
        require(file_sha256(Path(ref["path"])) == ref["sha256"], "INPUT_BINDING:" + key)
    required_owners = {"gx1/audit/run_dynamic_mtf_train_prefix_data.py", "gx1/audit/dynamic_mtf_december_normalization.py",
                       "gx1/contracts/entry_native_direction_prefix_normalization_v1.py",
                       "gx1/contracts/entry_native_direction_feature_view_v1.py",
                       "gx1/contracts/entry_model_native_input_normalization_v1.py"}
    _require_tested_sources(stage, ROOTS[MASKED_STAGE], required_owners)


def _require_tested_sources(stage, root, required_owners):
    require(required_owners <= set(stage["source_owners"]), "STAGE_SOURCE_OWNERS")
    for name, ref in stage["source_owners"].items():
        require(Path(ref["path"]) == REPO / name and file_sha256(REPO / name) == ref["sha256"], "OWNER:" + name)
    tests = stage["focused_tests"]
    require(Path(tests["path"]) == root / "FOCUSED_TESTS.json", "TEST_RECEIPT_PATH")
    tested = read_bound_json(Path(tests["path"]), tests["sha256"])
    require(tested["exit_code"] == 0 and tested["source_owners"] == stage["source_owners"], "TESTED_SOURCE")


def _bind_masked_feature_view(policy, stage, fitted, normalization_ref, qualification_ref):
    """Join the new statistics, current input code and registered objective without a model."""
    from gx1.contracts.entry_native_direction_feature_view_v1 import bind_native_direction_feature_view

    def read(ref):
        return read_bound_json(Path(ref["path"]), ref["sha256"])

    recipe = read(stage["historical_recipe"])
    manifest = read(recipe["input_manifest"])
    constructor_ref = manifest["constructor_metadata"]
    sources = dict(manifest["view_source_bindings"])
    sources["input_qualification"] = qualification_ref
    sources["dynamic_state_target_source"] = recipe["source_owners"]["gx1/contracts/entry_dynamic_mtf_state_targets_v1.py"]
    for name, old in sources.items():
        path = Path(old["path"])
        if path.is_relative_to(REPO):
            sources[name] = stage["source_owners"][str(path.relative_to(REPO))]
        ref = sources[name]
        require(file_sha256(Path(ref["path"])) == ref["sha256"], "VIEW_SOURCE:" + name)
    numeric = recipe["numerical_recipe"]
    view = bind_native_direction_feature_view(prefix_normalization=fitted, prefix_normalization_ref=normalization_ref,
        constructor_metadata=read(constructor_ref), constructor_metadata_ref=constructor_ref, source_bindings=sources,
        max_batch_rows=numeric["batch_size"], full_feature_profile=True, dynamic_state_supervision=True,
        current_state_first_loss_weight=numeric["current_state_first_loss_weight"],
        first_flat_activity_weight=numeric["first_flat_activity_weight"], native_endpoint_loss=True,
        zero_support_profile=fitted["zero_support_profile"])
    endpoint = policy["dynamic_mtf_native_endpoint_loss_step1_20261104_001"]
    observer = view["reserved_observer_axes"]
    require(view["prefix"]["rows"] == observer["fit"]["rows"] == stage["prefix_rows"]
            and observer["held_out"]["rows"] == stage["held_out_rows"]
            and view["full_axes"]["rows"] == observer["rows"] == stage["prefix_rows"] + stage["held_out_rows"]
            and view["zero_support_profile"] == fitted["zero_support_profile"]
            and view["prefix_normalization"]["binding"] == normalization_ref
            and view["research_objective"]["contract_sha256"] == endpoint["new_objective_sha256"], "VIEW_IDENTITY")
    return view


def _process_io():
    return {k: int(v) for k, v in (line.split() for line in Path("/proc/self/io").read_text().splitlines())}


def _masked_prefix_result(policy, stage, root, budget):
    from gx1.audit.dynamic_mtf_december_normalization import fit_masked_december_prefix
    from gx1.contracts.entry_native_direction_prefix_normalization_v1 import WINDOWS, december_zero_support_profile

    start, before = monotonic(), _process_io()
    report, fitted, sources = fit_masked_december_prefix(policy, stage)
    after = _process_io()
    rchar = after["rchar:"] - before["rchar:"]
    require(rchar <= stage["limits"]["max_rchar_bytes"], "RCHAR_LIMIT")
    geometry = policy[DECEMBER_STAGE]["geometry"]["fit"]
    prefix = report["proof"]["prefix"]
    require(prefix["rows"] == geometry["parents"] and prefix["cutoff_ns"] == stage["fit_cutoff_ns"]
            and all(prefix[name + "_sha256"] == geometry[key + "_sha256_le_i64"]
                    for name, key in (("parent_rows", "parent_rows"), ("decision_ns", "issue_ns"),
                                      ("bar_start_ns", "bar_start_ns"))), "DECEMBER_PREFIX_IDENTITY")
    support = report["support"]
    require(set(support) == {"signal", "ctx_cont"} | {"mtf_" + tf.lower() for tf in WINDOWS}
            | {"gc_mtf_" + tf.lower() for tf in WINDOWS}, "ALL_INPUT_SURFACES")
    empty = {s: [field for field, n in zip(d["field_names"], d["observed_counts"]) if n == 0]
             for s, d in support.items()}
    fields = set(stage["masked_fields"])
    expected = {"gc_mtf_d1": fields, "ctx_cont": fields, "signal": {"ctx_cont." + field for field in fields}}
    for name, data in support.items():
        require(len(data["field_names"]) == len(data["observed_counts"]) == len(set(data["field_names"]))
                and data["population_rows"] > 0, "SUPPORT:" + name)
    if fitted is not None:
        require(report["fit_allowed"] is True and not report["blockers"]
                and all(set(names) == expected.get(name, set()) for name, names in empty.items())
                and fitted["zero_support_profile"] == december_zero_support_profile()
                and fitted["prefix_qualification"] == report and fitted["native_launch_authorized"] is False
                and fitted["genuine_model_fit_authorized"] is False, "MASKED_NORMALIZATION_ACCEPTANCE")
    else:
        require(report["fit_allowed"] is False and bool(report["blockers"]), "UNEXPLAINED_FIT_BLOCK")
    qualification = publish(root / "QUALIFICATION.json", report, budget)
    normalization = publish(root / "PREFIX_NORMALIZATION.json", fitted, budget) if fitted is not None else None
    view_ref = None
    if normalization is not None:
        view = _bind_masked_feature_view(policy, stage, fitted, normalization, qualification)
        view_ref = publish(root / "FEATURE_VIEW.json", view, budget)
    return {"status": ("QUALIFIED_MASKED_DECEMBER_TRAIN_PREFIX_NORMALIZATION" if normalization else
                       "BLOCKED_MASKED_DECEMBER_TRAIN_PREFIX_BEFORE_FIT"),
            "profile": stage["profile"], "fit_prefix": prefix, "reserved_december_parents": stage["held_out_rows"],
            "qualification": qualification, "normalization": normalization, "feature_view": view_ref,
            "normalization_fits": int(fitted is not None),
            "surface_rows": {s: d["population_rows"] for s, d in support.items()},
            "surface_widths": {s: len(d["field_names"]) for s, d in support.items()},
            "zero_support_fields_by_surface": {s: names for s, names in empty.items() if names},
            "blockers": report["blockers"], "source_summary": sources, "elapsed_seconds": monotonic() - start,
            "process_rchar_delta": rchar, "process_read_bytes_delta": after["read_bytes:"] - before["read_bytes:"],
            "model_calls": 0, "optimizer_updates": 0, "held_out_predictions": 0,
            "held_out_outcomes_used": False, "test_data_used": False}


def _require_candle_stage(policy, stage):
    prior = policy[MASKED_STAGE]
    data = policy[DECEMBER_STAGE]
    geometry = policy["current_work"]["dynamic_mtf_train_gates_20261104"]["december_candidate_geometry"]
    ref_map = {"candidate_geometry": geometry["binding"], "fit_target_view": data["fit_target_view"],
               "prefix_counts": geometry["prefix_counts"], "prefix_prior": geometry["prefix_prior_and_markov"],
               "verified_input_profile": prior["verified_review"], "normalization_result": prior["result"],
               "masked_profile_terminal": prior["terminal"], "prefix_normalization": prior["normalization"],
               "feature_view": prior["feature_view"], "historical_recipe": prior["historical_recipe"],
               "endpoint_result": geometry["qualified_endpoint_loss_arithmetic"]["result"]}
    require(stage["status"] == "ADMITTED_ONE_PREFIX_CANDLE_AND_METADATA_PLAN_ONLY"
            and stage["candle_reference_authorized"] is stage["plan_publication_authorized"] is True
            and stage["fixed_fit_cohort_support_authorized"] is stage["data_reads_authorized"] is True
            and prior["claimed_invocations"] == prior["completed_invocations"] == 1
            and prior["execution_authorized"] is prior["relaunch_allowed"] is False
            and stage["prefix_rows"] == data["prefix_rows"] == 41866 and stage["held_out_rows"] == data["held_out_rows"] == 5736
            and stage["fit_cutoff_ns"] == data["fit_cutoff_ns"] == 1733011200000000000
            and stage["held_out_cutoff_ns"] == data["held_out_cutoff_ns"] == 1735689600000000000
            and geometry["masked_research_profile"]["run_admission"] is False
            and geometry["held_out_outcomes_read"] is False and all(stage[k] is False for k in (
                "held_out_features_authorized", "held_out_outcomes_authorized", "external_data_authorized",
                "native_launch_authorized")), "CANDLE_SCOPE")
    for k, ref in ref_map.items():
        require(stage[k] == ref and file_sha256(Path(ref["path"])) == ref["sha256"], "CANDLE_BINDING:" + k)
    _require_tested_sources(stage, ROOTS[CANDLE_STAGE], {"gx1/audit/run_dynamic_mtf_train_prefix_data.py",
        "gx1/audit/dynamic_mtf_december_candle.py","gx1/contracts/entry_dynamic_mtf_december_candidate_plan_v1.py",
        "gx1/contracts/entry_dynamic_mtf_state_report_v1.py","gx1/contracts/entry_dynamic_mtf_state_targets_v1.py"})


def _candle_and_plan_result(policy, stage, root, budget):
    from gx1.audit.dynamic_mtf_december_candle import calculate_prefix_candle
    from gx1.contracts.entry_dynamic_mtf_december_candidate_plan_v1 import build_december_candidate_plan

    start, before = monotonic(), _process_io()
    candle = calculate_prefix_candle(policy, stage)
    endpoint = read_bound_json(Path(stage["endpoint_result"]["path"]), stage["endpoint_result"]["sha256"])
    require(candle["fit_target_view"] == stage["fit_target_view"] and candle["feature_view"] == stage["feature_view"]
            and candle["fit_parents"] == stage["prefix_rows"] and candle["fit_cutoff_ns"] == stage["fit_cutoff_ns"]
            and {tf: panel["observed_native"] for tf, panel in candle["panels"].items()} == endpoint["native_cells_by_tf"],
            "CANDLE_INDEPENDENT_POPULATION")
    candle_ref = publish(root / "CANDLE_REFERENCE.json", candle, budget)
    plan_ref = None
    if candle["status"] == "FROZEN_PREFIX_CANDLE_REFERENCE":
        plan = build_december_candidate_plan(policy, stage, candle_reference_ref=candle_ref, candle_reference=candle)
        plan_ref = publish(root / "CANDIDATE_PLAN.json", plan, budget)
    else:
        require(candle["status"] == "UNDETERMINED_PREFIX_CANDLE_REFERENCE", "CANDLE_STATUS")
    after = _process_io()
    rchar = after["rchar:"] - before["rchar:"]
    require(rchar <= stage["limits"]["max_rchar_bytes"], "CANDLE_RCHAR_LIMIT")
    return {"status": ("COMPLETE_PREFIX_CANDLE_AND_REGISTERED_PLAN_NO_MODEL" if plan_ref else
                       "UNDETERMINED_PREFIX_CANDLE_NO_CANDIDATE_PLAN"),
            "candle_reference": candle_ref, "candidate_plan": plan_ref, "fit_parents": stage["prefix_rows"],
            "process_rchar_delta": rchar, "process_read_bytes_delta": after["read_bytes:"] - before["read_bytes:"],
            "reserved_december_parents": stage["held_out_rows"], "elapsed_seconds": monotonic() - start,
            "held_out_predictions": 0, "held_out_outcomes_used": False, "model_calls": 0,
            "optimizer_updates": 0, "normalization_fits": 0, "native_launch_authorized": False, "test_data_used": False}


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
    if stage_name in (STAGE, DECEMBER_STAGE):
        require(stage["target_result"] == policy["dynamic_mtf_state_target_audit_20261102_005"]["result"],
                "TARGET_AUTHORITY")
    if stage_name == CANDLE_STAGE:
        _require_candle_stage(policy, stage)
    elif stage_name == MASKED_STAGE:
        _require_masked_december_stage(policy, stage)
    elif stage_name == BEAR_STAGE:
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
            and stage["execution_authorized"] is True and stage["fit_target_view_authorized"] is (stage_name in (STAGE, DECEMBER_STAGE, CANDLE_STAGE))
            and stage["normalization_fit_authorized"] is (stage_name == MASKED_STAGE)
            and all(stage[k] is False for k in ("model_calls_authorized",
                    "optimizer_updates_authorized", "training_enabled", "relaunch_allowed", "test_authorized",
                    "broker_authorized", "trading_authorized"))
            and all(policy[k] is False for k in ("training_enabled", "native_training_enabled",
                    "full_epoch_training_allowed", "full_val_allowed")), "ADMISSION")
    inventory = {"EXECUTION_BINDING.json", "FOCUSED_TESTS.json"} if stage_name in (MASKED_STAGE, CANDLE_STAGE) else {"EXECUTION_BINDING.json"}
    require({p.name for p in root.iterdir()} == inventory, "CONSUMED")
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
        if stage_name == CANDLE_STAGE:
            with file_verification_scope():
                payload = _candle_and_plan_result(policy, stage, root, budget)
                capture_verified_file_snapshot()
        elif stage_name == MASKED_STAGE:
            payload = _masked_prefix_result(policy, stage, root, budget)
        elif stage_name == BEAR_STAGE:
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
