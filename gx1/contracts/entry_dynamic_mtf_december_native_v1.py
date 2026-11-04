"""Exact one-use recipe bindings for the pre-registered December candidate. No GPU or Torch imports."""
import math
from pathlib import Path

from gx1.contracts import local_random_access_campaign_v2 as campaign

REPO = Path("/home/andre2/src/GX1_CURRENT")
BASE = Path("/home/andre2/GX1_RUNS/DYNAMIC_MTF_TRAIN_GATES_20261104_001")
PLAN_REF = {"path": str(BASE / "DECEMBER_CANDLE_AND_CANDIDATE_PLAN_001" / "CANDIDATE_PLAN.json"),
            "sha256": "a5bc7eb8fe55ee4cdfef79bd0cd3b42661ab73cc43a350a11baa87db3f7f63c7"}
PRIVATE_SCOPE = "private_december_masked_dynamic_mtf_state_qualification_v1"
FIT_SCOPE = "fit_december_masked_dynamic_mtf_state_onepass_v1"
SCORE_SCOPE = "score_december_masked_dynamic_mtf_state_once_v1"
SCOPES = {"private": PRIVATE_SCOPE, "fit": FIT_SCOPE, "score": SCORE_SCOPE}
STAGES = {p: "dynamic_mtf_state_december_native_" + p + "_20261104_001" for p in SCOPES}
ROOTS = {p: BASE / ("DECEMBER_NATIVE_" + p.upper() + "_001") / "RUNTIME" for p in SCOPES}
PRIVATE_RECIPE_PATH = ROOTS["private"].parent / "NATIVE_RECIPE.json"
FOLD = {"fold_id": "registered_december_2024", "prefix_rows": 41866, "oof_start_row": 41866, "oof_end_row": 47602,
        "train_cutoff_ns": 1733011200000000000, "oof_cutoff_ns": 1735689600000000000}
RECIPE_SCHEMAS = {"private": "gx1_native_direction_research_launch_recipe_v1",
                  "fit": "gx1_native_direction_prefix_fit_launch_recipe_v1",
                  "score": "gx1_native_direction_prefix_fit_launch_recipe_v1"}
# SHA of the checkpoint-visible identity derived from the registered feature_view and objective.
VIEW_IDENTITY_SHA = "94d0cb41eaf35497ec6fdb00b567b0e6458b8891544072d55ea77dfe45d6835b"

def require_checkpoint_view(identity):
    _require(campaign.canonical_sha256(identity) == VIEW_IDENTITY_SHA, "MASKED_CHECKPOINT_VIEW")
    return identity


NEW_OWNERS = {"gx1/contracts/entry_dynamic_mtf_december_native_v1.py", "gx1/audit/dynamic_mtf_december_native_inputs.py"}
SCORE_OWNER = "gx1/contracts/entry_dynamic_mtf_december_score_v1.py"
# These are the role adapter, route, and state serialization files changed for this candidate.
TRANSITION_OWNERS = {
    "gx1/audit/dynamic_mtf_december_normalization.py",
    "gx1/contracts/entry_native_direction_feature_view_v1.py", "gx1/contracts/entry_native_direction_prefix_fit_campaign_v1.py",
    "gx1/contracts/entry_native_direction_research_campaign_v1.py", "gx1/contracts/entry_native_direction_research_state_v1.py",
    "gx1/scripts/run_entry_native_direction_dense_qualification_v1.py", "gx1/scripts/run_entry_native_direction_prefix_fit_window_v1.py",
    "gx1/scripts/run_entry_native_direction_research_window_v1.py"}



def _require(condition, reason):
    if not condition:
        raise RuntimeError("DECEMBER_NATIVE_" + reason)


def _read(ref):
    return campaign.read_bound_json(Path(ref["path"]), ref["sha256"])


def _registered():
    plan = _read(PLAN_REF)
    _require(plan["schema_version"] == "gx1_closed_december_dynamic_mtf_candidate_plan_v1"
             and plan["contract_sha256"] == campaign.canonical_sha256({k:v for k,v in plan.items() if k != "contract_sha256"})
             and plan["authority"]["native_launch_authorized"] is False and plan["qualification"]["cohorts"]["private_sparse_support_blockers"] == []
             and plan["fit"]["prefix_rows"] == FOLD["prefix_rows"] and plan["reserved_score"]["parents"] == 5736,
             "REGISTERED_PLAN")
    return plan


def _owners(plan, phase, supplied):
    # Include installed optimizer/serialization owners, the original registered files, and the phase's new executable owners.
    keys = set(plan["source_freeze"]["owners"]) | NEW_OWNERS
    if phase == "score":
        keys.add(SCORE_OWNER)
    if supplied is not None:
        _require(isinstance(supplied, dict) and set(supplied) == keys, "SOURCE_INVENTORY")
    owners = {}
    for k in sorted(keys):
        path = REPO / k
        _require(path.resolve(strict=True) == path and path.is_file(), "SOURCE_PATH:" + k)
        if supplied is None:
            owners[k] = {"path": str(path), "sha256": campaign.file_sha256(path)}
        else:
            _require(supplied[k]["path"] == str(path), "SOURCE_PATH:" + k)
            owners[k] = campaign.require_binding(supplied[k], label="December native source " + k)
        if k in plan["source_freeze"]["owners"] and k not in TRANSITION_OWNERS:
            _require(owners[k] == plan["source_freeze"]["owners"][k], "UNREGISTERED_SOURCE_CHANGE:" + k)
    return owners


def require_endpoint_step(step, wanted):
    loss = step.get("native_endpoint_loss")
    _require(type(loss) is float and math.isfinite(loss) and loss >= 0
             and step.get("native_endpoint_eligible_per_tf") == wanted
             and step.get("native_endpoint_eligible_cells") == sum(wanted), "PRIVATE_ENDPOINT_SUPPORT")


def _stage_file(ref, root, name):
    _require(isinstance(ref, dict) and ref.get("path") == str(root / name), "ARTIFACT_PATH:" + name)
    return _read(ref)


def _bound_under(ref, root):
    checked = campaign.require_binding(ref, label="December predecessor artifact")
    _require(Path(checked["path"]).is_relative_to(root), "ARTIFACT_ROOT")
    return checked


def build_december_native_recipe(*, phase, runtime_root, admission_stage, source_owners=None,
                                 qualification_review=None, fit_terminal=None, scoring_binding=None):
    """Construct the exact recipe; the existing policy, source, boot, lock and one-use claim still gate execution."""
    _require(phase in SCOPES and Path(runtime_root) == ROOTS[phase] and admission_stage == STAGES[phase], "PHASE_COORDINATES")
    with campaign.file_verification_scope():
        plan = _registered()
        refs = plan["bindings"]
        original = _read(refs["historical_recipe"])
        fit_meta = _read(refs["fit_target_view"])["metadata"]
        view = _read(refs["feature_view"])
        review = _read(refs["verified_input_profile"])
        _require(view["contract_sha256"] == plan["feature_view_contract_sha256"] == review["feature_view_contract_sha256"]
                 and view["prefix_normalization"]["binding"] == refs["prefix_normalization"] == review["normalization"]
                 and view["reserved_observer_axes"] == plan["observer_axes"] and fit_meta["parents"] == FOLD["prefix_rows"]
                 and fit_meta["cutoff_ns"] == FOLD["train_cutoff_ns"] and fit_meta["test_data_used"] is False,
                 "MASKED_INPUT_BINDING")
        require_checkpoint_view({"contract_sha256": view["contract_sha256"], "prefix_normalization": view["prefix_normalization"],
                                 "research_objective_sha256": view["research_objective"]["contract_sha256"]})
        for key in ("prefix_normalization", "prefix_counts", "prefix_prior", "candidate_geometry", "frozen_candle_reference"):
            campaign.require_binding(refs[key], label="December native immutable input " + key)
        part = plan["qualification" if phase == "private" else "fit"]
        numerical = part["numerical_recipe"]
        _require(numerical["native_endpoint_loss"] is True and numerical["research_objective_sha256"] == plan["objective_sha256"]
                 and numerical["batch_size"] == 16 and numerical["max_updates"] == (2 if phase == "private" else 2617)
                 and numerical["seed"] == 20260911, "PLANNED_NUMERICS")
        result = {"schema_version": RECIPE_SCHEMAS[phase], "source_repo": str(REPO), "run_scope": SCOPES[phase],
            "admission_stage": admission_stage, "runtime_root": str(runtime_root),
            "execution_binding_path": str(Path(runtime_root)/"EXECUTION_BINDING.json"), "candidate_plan": dict(PLAN_REF),
            "input_role": "score" if phase == "score" else "fit", "fold": dict(FOLD), "full_feature_profile": True,
            "feature_view": refs["feature_view"], "prefix_normalization": refs["prefix_normalization"],
            "prefix_prior": refs["prefix_prior"], "label_audit": refs["prefix_counts"], "input_manifest": original["input_manifest"],
            "target_manifest": refs["fit_target_view"], "target_sources": fit_meta["source_refs"], "numerical_recipe": numerical,
            "source_owners": _owners(plan, phase, source_owners), "training_enabled": phase != "score",
            "full_epoch_training_allowed": phase == "fit", "qualification_weights_for_main_fit": False,
            "test_data_used": False, "predictive_value_claimed": False}
        if phase == "private":
            _require(qualification_review is None and fit_terminal is None and scoring_binding is None, "PRIVATE_NO_ORIGIN")
            result.update({key: part[key] for key in ("max_private_updates", "max_model_constructions","max_forwards",
                          "max_backwards","max_checkpoint_bytes","max_published_json_bytes")})
            cohort = part["cohorts"]
            result["qualification_parent_rows"] = cohort["private_parents_sorted"]
            result["initial_parent_rows"] = cohort["initial_parents_in_order"]
            result["endpoint_support_per_batch"] = [
                [row[tf]["native"]["observed"] for tf in ("M5", "M15", "H1", "H4", "D1")]
                for row in cohort["private_batch_sparse_support"]]
        else:
            result.update({key: part[key] for key in ("train_passes","initial_train_measurement_rows","final_train_measurement_rows",
                          "max_windows","max_checkpoint_bytes","max_published_json_bytes","checkpoint_every_updates",
                          "max_checkpoint_total_bytes","parent_order_algorithm")})
            result["max_windows"] = 3 if phase == "fit" else 1
            result["initial_measurement_rows"] = result.pop("initial_train_measurement_rows")
            result["eval_batch_size"] = numerical["batch_size"]
            result["native_dynamic_state_qualification"] = qualification_review
            _require_private_terminal(qualification_review, result)
            if phase == "fit":
                _require(fit_terminal is None and scoring_binding is None, "FIT_NO_SCORE")
            else:
                result.update(_score_gate(fit_terminal, scoring_binding, result, plan))
        result["recipe_sha256"] = campaign.canonical_sha256(result)
        return result


def require_december_native_recipe(value, *, source_repo, source_commit, phase):
    _require(Path(source_repo) == REPO and isinstance(source_commit,str) and len(source_commit) == 40
             and all(c in "0123456789abcdef" for c in source_commit) and phase in SCOPES, "SOURCE_REPO")
    _require(isinstance(value, dict) and value.get("run_scope") == SCOPES[phase], "PHASE")
    expected = build_december_native_recipe(phase=phase, runtime_root=value.get("runtime_root",""),
        admission_stage=value.get("admission_stage"), source_owners=value.get("source_owners"),
        qualification_review=value.get("native_dynamic_state_qualification"), fit_terminal=value.get("fit_terminal"),
        scoring_binding=value.get("scoring_binding"))
    _require(campaign.canonical_sha256(value) == campaign.canonical_sha256(expected), "EXACT_REGISTERED_RECIPE")
    return expected



def require_phase_gate(gate, recipe):
    phase = next(p for p, scope in SCOPES.items() if scope == recipe["run_scope"])
    score = phase == "score"
    _require(gate.get("candidate_plan") == PLAN_REF and gate.get("input_role") == recipe["input_role"]
             and gate.get("held_out_features_authorized") is score and gate.get("held_out_outcomes_authorized") is score,
             "SEPARATE_ROLE_ADMISSION")
    if score:
        _require(gate.get("checkpoint_reads_authorized") is True and gate.get("zero_update_reserved_scoring_authorized") is True,
                 "FRESH_FIT_CHECKPOINT_SCORING_GATE")


def _require_private_terminal(ref, recipe):
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix
    root = ROOTS["private"]
    review = _stage_file(ref, root.parent, "ACTUAL_ROOT_REVIEW.json")
    paths = {"plan":root.parent/"CAMPAIGN/CAMPAIGN_PLAN.json", "result":root/"private-research/RESULT.json",
             "receipt":root/"receipts/invocation-0001.json", "initial_report":root/"private-research/INITIAL_TRAIN.json",
             "power_after":root.parent/"POWER_AFTER.json"}
    _require(all(review[key]["path"] == str(path) for key, path in paths.items()), "PRIVATE_REVIEW_PATHS")
    plan = _read(review["plan"])
    _require(plan["recipe"]["path"] == str(PRIVATE_RECIPE_PATH) and plan["runtime_root"] == str(root),
             "PRIVATE_RECIPE_LOCATION")
    from gx1.contracts import entry_native_direction_research_campaign_v1 as private
    receipt = _read(review["receipt"])
    _require(not (root/"ACTIVE_INVOCATION.json").exists()
             and sorted(p.name for p in (root/"receipts").glob("invocation-*.json")) == ["invocation-0001.json"],
             "PRIVATE_CLOSED_CLAIM")
    campaign.require_receipt_chain(campaign.require_plan(plan), [receipt])
    cursor = private.require_research_cursor(_stage_file(receipt["checkpoint_pointer_after"],root,"research-cursor/RESUME_CURSOR.json"),
        expected_recipe=plan["recipe"], batch_size=16)
    _require(cursor["result"] == review["result"] and cursor["outcome"] == "COMPLETE"
             and cursor["parent_cursor"] == 32 and cursor["optimizer_updates"] == 4, "PRIVATE_TERMINAL_CURSOR")
    prefix.require_dynamic_state_qualification(recipe, source_repo=REPO, source_commit=plan["source_commit"])


def require_december_fit_origin(fit_terminal):
    """Recheck the consumed fit, actual receipts and durable full-state before admitting reserved target rows."""
    from gx1.contracts import entry_native_direction_prefix_fit_campaign_v1 as prefix

    root = ROOTS["fit"]
    review = _stage_file(fit_terminal, root.parent, "ACTUAL_ROOT_REVIEW.json")
    _require(review.get("schema_version") == "gx1_december_native_fit_terminal_review_v1"
             and review.get("status") == "COMPLETE_CONSUMED_DECEMBER_FIT" and review.get("candidate_plan") == PLAN_REF
             and review.get("test_data_used") is False, "FIT_TERMINAL_REVIEW")
    plan = _stage_file(review["plan"], root.parent, "CAMPAIGN/CAMPAIGN_PLAN.json")
    _require(plan["recipe"] == review["recipe"] and plan["recipe"]["path"] == str(root.parent/"RECIPE.json")
             and plan["runtime_root"] == str(root), "FIT_TERMINAL_CAMPAIGN")
    checked = campaign.require_plan(plan, verify_files=True)
    receipts = sorted((root/"receipts").glob("invocation-*.json"))
    _require(1 <= len(receipts) <= 3 and [p.name for p in receipts] == [f"invocation-{n:04d}.json" for n in range(1,len(receipts)+1)]
             and review["receipt"]["path"] == str(receipts[-1]) and not (root/"ACTIVE_INVOCATION.json").exists(),
             "FIT_RECEIPT_SET")
    rows = campaign.require_receipt_chain(checked, [campaign.read_bound_json(p, campaign.file_sha256(p)) for p in receipts])
    receipt = _read(review["receipt"])
    _require(receipt["receipt_sha256"] == rows[-1]["receipt_sha256"] and rows[-1]["outcome"] == "COMPLETE"
             and all(r["outcome"] == "RESUMABLE" for r in rows[:-1]) and rows[-1]["guard_decision"] == "PASS"
             and rows[-1]["checkpoint_pointer_after"] == review["cursor"], "FIT_TERMINAL_RECEIPTS")
    raw = _stage_file(review["cursor"], root, "prefix-cursor/RESUME_CURSOR.json")
    cursor = prefix.require_prefix_fit_cursor(raw, expected_recipe=plan["recipe"], expected_fold=FOLD)
    result = _stage_file(review["result"], root, f"prefix-fit/window-{len(rows):04d}/RESULT.json")
    _require(cursor["outcome"] == cursor["phase_state"] == result["outcome"] == "COMPLETE"
             and cursor["parent_cursor"] == result["parent_cursor"] == FOLD["prefix_rows"]
             and 0 < cursor["optimizer_updates"] == result["optimizer_updates"] <= 2617
             and cursor["oof_cursor"] == result["oof_cursor"] == 0 and result.get("oof_measurement") is None
             and result["error"] is None and result["source_unchanged"] is True and result["forbidden_open_paths"] == []
             and result["qualification_weights_reused"] is False and result["test_data_used"] is False
             and cursor["result"] == review["result"] and result["recipe"] == review["recipe"]
             and result.get("candidate_plan") == PLAN_REF and result.get("held_out_features_read") is False
             and result.get("held_out_outcomes_read") is False, "FIT_TERMINAL_OUTCOME")
    require_checkpoint_view(cursor["feature_view_identity"])
    checkpoint = result["durable_checkpoint"]
    _require(checkpoint["state"] == result["full_state"] == cursor["full_state"] and checkpoint["parent_cursor"] == cursor["parent_cursor"]
             and checkpoint["optimizer_updates"] == cursor["optimizer_updates"]
             and checkpoint["state"]["path"] == str(root/"prefix-fit/states"/
                 f"w{len(rows):04d}_TERMINAL_u{cursor['optimizer_updates']:06d}_p041866.pt"), "FIT_TERMINAL_CHECKPOINT")
    campaign._sha(checkpoint["contract_sha256"], "December fit full-state seal")
    admitted = _read(plan["recipe"])
    for role, key, parents in (("INITIAL_TRAIN","initial_measurement",256), ("FINAL_TRAIN","final_train_measurement",41866)):
        report = _read(_bound_under(cursor[key], root/"prefix-fit"))
        _require(report["role"] == role and report["fold"] == FOLD and report["target_manifest"] == admitted["target_manifest"]
                 and report["research_objective_sha256"] == admitted["numerical_recipe"]["research_objective_sha256"]
                 and report["feature_view_identity"] == cursor["feature_view_identity"] and report["test_data_used"] is False
                 and report["own_initial_state"] == cursor["own_initial_state"] and len(report["timeframes"]) == 5,
                 "FIT_TERMINAL_MEASUREMENT_" + role)
        artifact = _read(_bound_under(report["predictions"], root/"prefix-fit"))
        _require(artifact["arrays"]["probabilities"]["shape"] == [parents,5,1440,3], "FIT_TERMINAL_MEASUREMENT_ROWS_" + role)
    power = _stage_file(review["power_after"], root.parent, "POWER_AFTER.json")
    gpu = [s.strip() for s in power["gpu"][0].split(",")] if len(power.get("gpu", [])) == 1 else []
    _require(len(gpu) == 6 and gpu[0] == plan["gpu_uuid"] and float(gpu[1]) == power.get("configured_power_limit_w") == 250
             and power.get("task_state") == "Disabled" and power.get("test_data_used") is False
             and {r["TaskName"] for r in power["keeper"] if r.get("State") == 4} == {"GX1GpuPowerLimit","GX1HostTelemetryBridge"},
             "FIT_TERMINAL_POWER")
    return {"fit_terminal_review":fit_terminal, "fit_recipe":review["recipe"], "fit_cursor":review["cursor"],
            "fit_checkpoint":cursor["full_state"], "own_initial_state":cursor["own_initial_state"],
            "initial_measurement":cursor["initial_measurement"], "final_train_measurement":cursor["final_train_measurement"],
            "state_seal":checkpoint["contract_sha256"]}


def _score_gate(fit_terminal, scoring_binding, recipe, plan):
    from gx1.contracts.entry_dynamic_mtf_december_score_v1 import december_score_storage_plan

    origin = require_december_fit_origin(fit_terminal)
    score = _stage_file(scoring_binding, ROOTS["score"].parent, "SCORING_BINDING.json")
    target_root = BASE/"DECEMBER_NATIVE_SCORE_TARGETS_001"
    storage = december_score_storage_plan(max_published_json_bytes=recipe["max_published_json_bytes"],
                                         max_checkpoint_total_bytes=recipe["max_checkpoint_total_bytes"])
    _require(score.get("input_role") == "score" and score.get("candidate_plan") == PLAN_REF and score.get("source_terminal") == fit_terminal
             and score.get("fit_checkpoint") == origin["fit_checkpoint"] and score.get("scorer") == recipe["source_owners"][SCORE_OWNER]
             and score.get("storage_plan") == storage and score.get("scoring_contract_sha256") == plan["reserved_score"]["contract"]["contract_sha256"]
             and score.get("test_data_used") is False, "SCORE_BINDING")
    target_result = _stage_file(score["target_result"], target_root, "RESULT.json")
    target_terminal = _stage_file(score["target_terminal"], target_root, "TERMINAL.json")
    _require(target_terminal.get("status") == "COMPLETE" and target_terminal.get("exit_code") == 0
             and target_terminal.get("source_unchanged") is True and target_terminal.get("result") == score["target_result"]
             and target_terminal.get("test_data_used") is False and target_result.get("status") == "MATERIALIZED_RESERVED_ONLY_TARGETS_NO_SCORES"
             and target_result.get("candidate_plan") == PLAN_REF and target_result.get("fit_origin") == origin
             and target_result.get("fit_checkpoint") == origin["fit_checkpoint"] and target_result.get("fit_terminal_review") == fit_terminal
             and target_result.get("target_manifest") == score["target_manifest"] and target_result.get("calendar_authority") == score["calendar_authority"]
             and target_result.get("reserved_parents") == 5736 and target_result.get("fit_cutoff_ns") == FOLD["train_cutoff_ns"]
             and target_result.get("score_cutoff_ns") == FOLD["oof_cutoff_ns"] and target_result.get("model_calls") == target_result.get("optimizer_updates") == 0
             and target_result.get("score_computed") is False, "RESERVED_TARGET_TERMINAL")
    target = _read(_bound_under(score["target_manifest"], target_root))
    _require(target["metadata"]["parents"] == 5736 and target["metadata"]["cutoff_ns"] == FOLD["oof_cutoff_ns"]
             and target["metadata"]["test_data_used"] is False, "SCORE_TARGET_GEOMETRY")
    cal = _stage_file(score["calendar_authority"], target_root, "SCORE_CLOCK_CLASSIFICATION.json")
    _require(cal["role"] == "score" and cal["candidate_plan"] == PLAN_REF and cal["target_manifest"] == score["target_manifest"], "SCORE_CALENDAR")
    return {"fit_terminal":fit_terminal, "score_origin":origin, "scoring_binding":scoring_binding,
            "target_manifest":score["target_manifest"], "target_sources":target["metadata"]["source_refs"],
            "score_storage_plan":storage, "candle_reference":plan["bindings"]["frozen_candle_reference"]}
