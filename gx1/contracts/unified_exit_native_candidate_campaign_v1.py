"""Campaign resume bindings for the existing native candidate session.

The campaign cursor binds the canonical TRAIN pointer/state and any active
VAL cursor together. It is a transport receipt, not another training state.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.local_random_access_campaign_v2 import (
    canonical_sha256, file_sha256, read_bound_json, require_binding,
)
from gx1.contracts.unified_exit_random_access_sampler_v1 import canonical_sha256 as native_sha256
from gx1.contracts.entry_candidate_checkpoint_policy_v1 import native_checkpoint_monitor
from gx1.contracts.unified_exit_random_access_index_v1 import (
    LATEST_YEAR_ROOT_SCHEMA_VERSION, require_random_access_index_root,
)


CURSOR_SCHEMA = "gx1_native_candidate_campaign_cursor_v1"
NATIVE_KIND = "native_candidate_window"
NATIVE_PHASE = "native_candidate"
NATIVE_MODULE = "gx1.scripts.run_unified_exit_native_candidate_window_v1"
WINDOW_SCHEMA = "gx1_native_candidate_window_policy_v1"


def require_native_recipe_metadata(
    binding: Mapping[str, str], *, source_repo: Path, source_commit: str,
) -> tuple[dict[str, Any], int]:
    """Check campaign metadata without constructing datasets or model tensors.

    The guarded native entry point still owns full source/data/model validation.
    """
    checked = require_binding(binding, label="native recipe")
    recipe = read_bound_json(Path(checked["path"]), checked["sha256"])
    controls = recipe.get("trainer_cli", {})
    economics_binding = require_binding(recipe.get("files", {}).get("economics_readiness"), label="native economics readiness")
    economics = read_bound_json(Path(economics_binding["path"]), economics_binding["sha256"])
    monitor = native_checkpoint_monitor(economics["economics_objective_contract"])
    if (
        recipe.get("schema_version") != "gx1_unified_exit_random_access_full_train_recipe_v1"
        or recipe.get("profile") != "candidate" or recipe.get("test_data_used") is not False
        or recipe.get("source_repo") != str(source_repo)
        or recipe.get("source_commit") != source_commit
        or recipe.get("recipe_sha256") != native_sha256({k: v for k, v in recipe.items() if k != "recipe_sha256"})
        or any(recipe.get("val_limits", {}).get(k) != v for k, v in
               {"policy_batch_size": 256, "cpu_pipeline_workers": 8,
                "max_wall_seconds": 10800, "progress_interval_forwards": 64}.items())
        or controls.get("epochs") != 30 or controls.get("batch_size") != 16
        or controls.get("early_stopping_patience") != 5
        or controls.get("checkpoint_monitor") != monitor
    ):
        raise RuntimeError("NATIVE_CANDIDATE_RECIPE_METADATA_INVALID")
    root_binding = require_binding(recipe["files"]["random_access_root"], label="native index root")
    root = read_bound_json(Path(root_binding["path"]), root_binding["sha256"])
    if (
        root.get("decision") != "PASS" or root.get("allowed_splits") != ["train", "val"]
        or root.get("test_accessed") is not False
        or root.get("root_sha256") != native_sha256({k: v for k, v in root.items() if k != "root_sha256"})
    ):
        raise RuntimeError("NATIVE_CANDIDATE_INDEX_ROOT_METADATA_INVALID")
    latest_year = root.get("schema_version") == LATEST_YEAR_ROOT_SCHEMA_VERSION
    if latest_year:
        require_random_access_index_root(root)
    split = root["splits"]["train"]
    path = Path(split["manifest_path"])
    manifest = read_bound_json(path, file_sha256(path))
    count = manifest.get("entry_row_count")
    if (
        manifest.get("manifest_sha256") != split["manifest_sha256"]
        or manifest.get("manifest_sha256") != native_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        or type(count) is not int or count < 1
        or (not latest_year and manifest.get("parent_entry_source_rows") != count)
        or (latest_year and split.get("entry_row_count") != count)
    ):
        raise RuntimeError("NATIVE_CANDIDATE_FULL_POPULATION_METADATA_INVALID")
    if latest_year:
        count = root["latest_year_population"]["splits"]["train"]["selected_entry_row_count"]
    return recipe, count


def require_native_calibration_run(recipe: Mapping[str, Any]) -> dict[str, Any] | None:
    """The two finite native arms compare 32 uninterrupted versus 16+16 steps."""
    if "native_calibration" not in recipe:
        return None
    value = recipe["native_calibration"]
    if (not isinstance(value, Mapping) or set(value) != {
            "schema_version", "arm", "report_only_val"}
            or value["schema_version"] != "gx1_native_learning_calibration_run_v1"
            or type(value["arm"]) is not str or value["arm"] not in {"reference", "split"}
            or type(value["report_only_val"]) is not bool):
        raise RuntimeError("NATIVE_CALIBRATION_RUN_INVALID")
    return dict(value)


def require_native_run_scope(
    recipe: Mapping[str, Any], *, invocation_number: int | None = None,
    execution_budget: Mapping[str, Any] | None = None,
) -> int | None:
    """Enforce the operator's bound readiness policy before native model work.

    The sole pre-training exception is a finite, declared TRAIN calibration.
    It uses the normal native session, production profile and machine guards.
    """
    repo = Path(__file__).resolve().parents[2]
    origin = recipe.get("candidate_resume_origin")
    origin_fields = {"schema_version", "contract", "pointer"}
    optional_origin_fields = {"exit_value_initialization", "train_population_scope"}
    if (not isinstance(origin, Mapping) or not origin_fields <= set(origin)
            or set(origin) - origin_fields - optional_origin_fields
            or ("train_population_scope" in origin and (
                type(origin["train_population_scope"]) is not str
                or origin["train_population_scope"] != "latest_year_2025_2026_v1"))
            or origin.get("schema_version") != "gx1_candidate_economics_transition_origin_v1"
            or ("exit_value_initialization" in origin and (
                type(origin["exit_value_initialization"]) is not str
                or origin["exit_value_initialization"] != "close_now_baseline_v1"))):
        raise RuntimeError("NATIVE_ECONOMICS_TRANSITION_ORIGIN_REQUIRED")
    initialization = origin.get("exit_value_initialization")
    calibration = require_native_calibration_run(recipe)
    population = origin.get("train_population_scope")
    require_binding(origin["contract"], label="native origin contract")
    origin_pointer = require_binding(origin["pointer"], label="native origin pointer")
    binding = require_binding(recipe.get("next_run_policy"), label="next native run policy")
    if Path(binding["path"]) != repo / "NEXT_RUN_POLICY.json":
        raise RuntimeError("NATIVE_NEXT_RUN_POLICY_PATH_INVALID")
    policy = read_bound_json(Path(binding["path"]), binding["sha256"])
    if (("exit_value_initialization" in policy and (
            type(policy["exit_value_initialization"]) is not str
            or policy["exit_value_initialization"] != "close_now_baseline_v1"))
            or policy.get("exit_value_initialization") != initialization):
        raise RuntimeError("NATIVE_EXIT_VALUE_INITIALIZATION_POLICY_MISMATCH")
    if (("train_population_scope" in policy and (
            type(policy["train_population_scope"]) is not str
            or policy["train_population_scope"] != "latest_year_2025_2026_v1"))
            or policy.get("train_population_scope") != population):
        raise RuntimeError("NATIVE_TRAIN_POPULATION_POLICY_MISMATCH")
    if population is not None:
        root_binding = require_binding(recipe.get("files", {}).get("random_access_root"), label="native training population")
        root = require_random_access_index_root(read_bound_json(Path(root_binding["path"]), root_binding["sha256"]))
        if root["schema_version"] != LATEST_YEAR_ROOT_SCHEMA_VERSION:
            raise RuntimeError("NATIVE_LATEST_YEAR_POPULATION_REQUIRED")
    profile = {"policy_batch_size": 256, "cpu_pipeline_workers": 8,
               "max_wall_seconds": 10800, "progress_interval_forwards": 64}
    if (policy.get("schema_version") != "gx1_next_native_run_policy_v1"
            or policy.get("canonical_source_repo") != str(repo)
            or policy.get("canonical_branch") != "work/gx1-current"
            or policy.get("training_module") != NATIVE_MODULE
            or policy.get("required_val_profile") != profile
            or any(recipe.get("val_limits", {}).get(k) != v for k, v in profile.items())
            or policy.get("train_batch_size") != 16
            or recipe.get("trainer_cli", {}).get("batch_size") != 16
            or policy.get("precision") != "float32" or policy.get("tf32_allowed") is not False
            or policy.get("native_invocation_seconds") != 12000
            or policy.get("outer_guard_seconds") != 13800):
        raise RuntimeError("NATIVE_NEXT_RUN_PROFILE_INVALID")
    evidence = policy.get("required_evidence", {})
    risk_binding = require_binding(evidence.get("risk_objective"), label="native risk objective")
    risk = read_bound_json(Path(risk_binding["path"]), risk_binding["sha256"])
    econ_binding = require_binding(recipe.get("files", {}).get("economics_readiness"), label="native economics")
    economics = read_bound_json(Path(econ_binding["path"]), econ_binding["sha256"])
    objective = economics.get("economics_objective_contract", {})
    if (risk.get("decision") != "PASS" or risk.get("test_data_used") is not False
            or risk.get("maximum_holding_seconds") is not None
            or risk.get("absolute_loss_limit_bps") is not None
            or risk.get("reward_accounting") != "liquidation_advantage_v1"
            or risk.get("economics_objective_schema") != "gx1_unified_exit_economics_objective_v4"
            or objective.get("reward_accounting") != risk["reward_accounting"]
            or objective.get("schema_version") != risk["economics_objective_schema"]):
        raise RuntimeError("NATIVE_NEXT_RUN_RISK_OBJECTIVE_INVALID")
    if policy.get("training_enabled") is True:
        if calibration is not None:
            raise RuntimeError("NATIVE_CALIBRATION_CANNOT_ENABLE_FULL_TRAINING")
        for role in ("checkpoint_transition", "learning_calibration", "gpu_batch256_parity",
                     "end_to_end_throughput", "resume_equivalence"):
            proof_binding = require_binding(evidence.get(role), label=f"native {role}")
            proof = read_bound_json(Path(proof_binding["path"]), proof_binding["sha256"])
            if (proof.get("decision") != "PASS" or proof.get("test_data_used") is not False
                    or proof.get("evidence_role") != role
                    or proof.get("economics_objective_contract_sha256") != objective.get("contract_sha256")
                    or proof.get("source_bindings_sha256") != recipe.get("source_bindings_sha256")
                    or proof.get("training_origin_pointer_sha256") != origin_pointer["sha256"]
                    or ("exit_value_initialization" in proof and (
                        type(proof["exit_value_initialization"]) is not str
                        or proof["exit_value_initialization"] != "close_now_baseline_v1"))
                    or proof.get("exit_value_initialization") != initialization
                    or proof.get("training_population_root_sha256") != recipe.get("files", {}).get("random_access_root", {}).get("sha256")
                    or proof.get("native_val_profile") != profile):
                raise RuntimeError("NATIVE_NEXT_RUN_EVIDENCE_NOT_PASS")
        ceiling = None
    elif policy.get("training_enabled") is False:
        scope = policy.get("native_learning_calibration")
        old_scope = {
            "schema_version": "gx1_native_learning_calibration_scope_v1",
            "optimizer_step_ceilings": [16, 32],
            "full_epoch_training_allowed": False, "test_data_used": False,
        }
        measured_scope = {
            **old_scope, "schema_version": "gx1_native_learning_calibration_scope_v2",
            "reference_optimizer_step_ceiling": 32, "native_report_only_val": True,
        }
        if (not isinstance(scope, Mapping)
                or scope not in (old_scope, measured_scope)
                or any(type(x) is not int for x in scope["optimizer_step_ceilings"])
                or scope["full_epoch_training_allowed"] is not False
                or scope["test_data_used"] is not False
                or (calibration is None) != (scope == old_scope)
                or (scope == measured_scope and scope["native_report_only_val"] is not True)):
            raise RuntimeError("NATIVE_TRAINING_BLOCKED_CALIBRATION_SCOPE_REQUIRED")
        ceilings = ([32] if calibration is not None and calibration["arm"] == "reference"
                    else scope["optimizer_step_ceilings"])
        if invocation_number is not None:
            if type(invocation_number) is not int or not 1 <= invocation_number <= len(ceilings):
                raise RuntimeError("NATIVE_CALIBRATION_INVOCATION_INVALID")
            ceiling = ceilings[invocation_number - 1]
        elif execution_budget is not None:
            ceiling = execution_budget.get("stop_after_optimizer_steps")
            if type(ceiling) is not int or ceiling not in ceilings:
                raise RuntimeError("NATIVE_CALIBRATION_STEP_CEILING_INVALID")
        else:
            raise RuntimeError("NATIVE_CALIBRATION_BOUNDARY_REQUIRED")
    else:
        raise RuntimeError("NATIVE_NEXT_RUN_ENABLEMENT_INVALID")
    if execution_budget is not None and (
            execution_budget.get("stop_after_optimizer_steps") != ceiling
            or execution_budget.get("stop_after_completed_val_epochs") is not None
            or execution_budget.get("max_invocation_seconds") != 12000
            or "resume_probe_val_rows" in execution_budget):
        raise RuntimeError("NATIVE_NEXT_RUN_BUDGET_INVALID")
    return ceiling


def require_native_window_policy(value: Any, *, verify_files: bool = True) -> dict[str, Any]:
    """Bind a window to one recipe and private campaign output paths."""
    fields = {
        "schema_version", "recipe", "invocation_number", "max_invocation_seconds",
        "budget_path", "progress_path", "campaign_cursor_path",
        "training_session_directory", "test_data_used", "policy_sha256",
    }
    if (
        not isinstance(value, Mapping) or set(value) != fields
        or value["schema_version"] != WINDOW_SCHEMA or value["test_data_used"] is not False
        or type(value["invocation_number"]) is not int or value["invocation_number"] < 1
        or type(value["max_invocation_seconds"]) is not int
        or value["max_invocation_seconds"] != 12000
        or value["policy_sha256"] != canonical_sha256({k: v for k, v in value.items() if k != "policy_sha256"})
    ):
        raise RuntimeError("NATIVE_CANDIDATE_WINDOW_POLICY_INVALID")
    require_binding(value["recipe"], label="native window recipe", verify_file=verify_files)
    for field in ("budget_path", "progress_path", "campaign_cursor_path", "training_session_directory"):
        if not isinstance(value[field], str):
            raise RuntimeError("NATIVE_CANDIDATE_WINDOW_PATH_INVALID")
        path = Path(value[field])
        if not path.is_absolute() or path.resolve() != path or path.is_symlink():
            raise RuntimeError("NATIVE_CANDIDATE_WINDOW_PATH_INVALID")
    if len({value[key] for key in ("budget_path", "progress_path", "campaign_cursor_path")}) != 3:
        raise RuntimeError("NATIVE_CANDIDATE_WINDOW_PATH_COLLISION")
    return dict(value)



def require_complete_val_observation(result: Mapping[str, Any]) -> dict[str, Any]:
    """Require all June outcomes; natural split-end censoring is not a data failure."""
    from collections import Counter
    from gx1.contracts.unified_exit_entry_policy_evaluation_v1 import (
        require_entry_policy_decisions, coupled_entry_exit_policy_metrics,
        marked_entry_exit_policy_metrics,
    )

    from gx1.contracts.unified_exit_random_access_val_evaluator_v1 import (
        RESULT_SCHEMA_VERSION, MARKED_RESULT_SCHEMA_VERSION, LIQUIDATION_RELATIVE_RESULT_SCHEMA_VERSION,
    )
    if result.get("schema_version") not in {RESULT_SCHEMA_VERSION, MARKED_RESULT_SCHEMA_VERSION, LIQUIDATION_RELATIVE_RESULT_SCHEMA_VERSION}:
        raise RuntimeError("NATIVE_CANDIDATE_VAL_SCHEMA_INVALID")
    outcomes = result.get("trade_outcomes")
    if (not isinstance(outcomes, list) or len(outcomes) != 11016
            or any(not isinstance(row, Mapping) for row in outcomes)):
        raise RuntimeError("NATIVE_CANDIDATE_FULL_VAL_OUTCOMES_REQUIRED")
    counts = Counter(row.get("status") for row in outcomes)
    censored = counts.get("RIGHT_CENSORED_SPLIT_END", 0)
    if (set(counts) - {"EXITED", "RIGHT_CENSORED_SPLIT_END"}
            or result.get("decision") != ("COMPLETE_WITH_RIGHT_CENSORING" if censored else "PASS_COMPLETE")
            or result.get("test_data_used") is not False
            or result.get("rollout_execution_complete") is not True
            or result.get("entry_pair_cohort_size") != 5508
            or result.get("side_trade_count") != 11016
            or result.get("exited_side_trade_count") != counts.get("EXITED", 0)
            or result.get("right_censored_side_trade_count") != censored
            or result.get("compute_truncated_side_trade_count") != 0):
        raise RuntimeError("NATIVE_CANDIDATE_COMPLETE_OBSERVED_VAL_REQUIRED")
    policy = require_entry_policy_decisions(
        result.get("entry_policy_decisions"), entry_row_indices=list(range(5508)),
        checkpoint_binding_sha256=result["checkpoint_binding_sha256"],
    )
    metrics = coupled_entry_exit_policy_metrics(
        entry_policy=policy, trade_outcomes=outcomes, full_cohort_authoritative=True,
    )
    if (result.get("entry_exit_policy_metrics") != metrics
            or result.get("full_cohort_policy_metrics_authoritative")
            is not metrics["full_cohort_authoritative"]):
        raise RuntimeError("NATIVE_CANDIDATE_FULL_VAL_POLICY_METRICS_INVALID")
    relative = result["schema_version"] == LIQUIDATION_RELATIVE_RESULT_SCHEMA_VERSION
    if ((relative and result.get("exit_q_value_coordinates") != "advantage_over_executable_liquidation_bps")
            or (not relative and "exit_q_value_coordinates" in result)):
        raise RuntimeError("NATIVE_CANDIDATE_VAL_VALUE_COORDINATES_INVALID")
    if result["schema_version"] in {MARKED_RESULT_SCHEMA_VERSION, LIQUIDATION_RELATIVE_RESULT_SCHEMA_VERSION}:
        marked = marked_entry_exit_policy_metrics(
            entry_policy=policy, trade_outcomes=outcomes, full_cohort_authoritative=True,
        )
        if result.get("marked_policy_evaluation") != marked:
            raise RuntimeError("NATIVE_CANDIDATE_MARKED_VAL_METRICS_INVALID")
    elif "marked_policy_evaluation" in result:
        raise RuntimeError("NATIVE_CANDIDATE_MARKED_VAL_SCHEMA_MISMATCH")
    return dict(result)


def require_native_completed_smoke(
    *, plan: Mapping[str, Any], prior: Mapping[str, Any], recipe: Mapping[str, Any],
) -> None:
    from gx1.contracts.local_random_access_campaign_v2 import require_receipt_chain
    from gx1.scripts.local_random_access_campaign_v2 import _receipts

    receipts = require_receipt_chain(prior, _receipts(Path(prior["runtime_root"])), verify_files=True)
    if (
        prior["phase"] != "full_val" or not receipts or receipts[-1]["outcome"] != "COMPLETE"
        or recipe["seed_authority"] != prior["final_train_checkpoint_authority"]
        or plan["final_train_checkpoint_authority"] != recipe["seed_authority"]
        or plan["selection_receipt"] != prior["selection_receipt"]
    ):
        raise RuntimeError("NATIVE_CANDIDATE_COMPLETED_SMOKE_CAMPAIGN_REQUIRED")
    argv = prior["checked_invocations"][receipts[-1]["invocation_number"] - 1]["launcher_argv"]
    binding = require_binding(recipe["smoke_full_val"], label="complete smoke VAL result")
    if argv.count("--result-path") != 1 or binding["path"] != argv[argv.index("--result-path") + 1]:
        raise RuntimeError("NATIVE_CANDIDATE_SMOKE_RESULT_PATH_MISMATCH")
    result = read_bound_json(Path(binding["path"]), binding["sha256"])
    require_complete_val_observation(result)
    if result.get("semantic_result_sha256") != native_sha256(
        {k: v for k, v in result.items() if k != "semantic_result_sha256"}
    ):
        raise RuntimeError("NATIVE_CANDIDATE_COMPLETE_SMOKE_RESULT_BINDING_INVALID")


def require_native_cursor(
    value: Any, *, expected_recipe: Mapping[str, str], verify_files: bool = True,
) -> dict[str, Any]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema_version", "recipe", "resume_state", "outcome", "test_data_used", "cursor_sha256"}
        or value["schema_version"] != CURSOR_SCHEMA
        or value["recipe"] != expected_recipe or value["test_data_used"] is not False
        or value["outcome"] not in {"RESUMABLE", "COMPLETE"}
        or value["cursor_sha256"] != canonical_sha256({k: v for k, v in value.items() if k != "cursor_sha256"})
    ):
        raise RuntimeError("NATIVE_CANDIDATE_CAMPAIGN_CURSOR_INVALID")
    require_binding(value["recipe"], label="native candidate recipe", verify_file=verify_files)
    state = value["resume_state"]
    fields = {
        "training_pointer", "training_state", "active_val_cursor", "active_val_model_forwards",
        "epoch_schedule_sha256", "session_contract_sha256", "phase", "epoch_index",
        "next_batch_offset", "global_optimizer_steps", "complete",
    }
    if not isinstance(state, Mapping) or set(state) != fields:
        raise RuntimeError("NATIVE_CANDIDATE_RESUME_STATE_INVALID")
    if (
        state["phase"] not in {"train", "validation"}
        or type(state["complete"]) is not bool
        or state["complete"] != (value["outcome"] == "COMPLETE")
        or any(type(state[key]) is not int or state[key] < 0 for key in (
            "epoch_index", "next_batch_offset", "global_optimizer_steps", "active_val_model_forwards",
        ))
        or state["epoch_index"] > 30
        or (state["epoch_index"] == 30 and not state["complete"])
        or any(not isinstance(state[key], str) or len(state[key]) != 64
               or any(c not in "0123456789abcdef" for c in state[key])
               for key in ("epoch_schedule_sha256", "session_contract_sha256"))
    ):
        raise RuntimeError("NATIVE_CANDIDATE_RESUME_POSITION_INVALID")
    pointer = require_binding(state["training_pointer"], label="native TRAIN pointer", verify_file=verify_files)
    checkpoint = require_binding(state["training_state"], label="native TRAIN state", verify_file=verify_files)
    directory = Path(pointer["path"]).parent
    if (
        Path(pointer["path"]).name != "CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json"
        or Path(checkpoint["path"]).parent != directory
    ):
        raise RuntimeError("NATIVE_CANDIDATE_CHECKPOINT_LAYOUT_INVALID")
    cursor = state["active_val_cursor"]
    if cursor is not None:
        cursor = require_binding(cursor, label="native active VAL cursor", verify_file=verify_files)
        expected = directory / "native_val" / f"epoch_{state['epoch_index'] + 1:04d}" / "ROLLOUT_PROGRESS.json"
        if state["phase"] != "validation" or cursor["path"] != str(expected):
            raise RuntimeError("NATIVE_CANDIDATE_VAL_CURSOR_LAYOUT_INVALID")
        if verify_files:
            raw = read_bound_json(Path(cursor["path"]), cursor["sha256"])
            if raw["model_forward_count"] != state["active_val_model_forwards"]:
                raise RuntimeError("NATIVE_CANDIDATE_VAL_CURSOR_COUNTER_MISMATCH")
    elif state["active_val_model_forwards"] != 0:
        raise RuntimeError("NATIVE_CANDIDATE_VAL_CURSOR_MISSING")
    if verify_files:
        raw = read_bound_json(Path(pointer["path"]), pointer["sha256"])
        if (
            raw.get("schema_version") != "gx1_candidate_training_session_v1"
            or raw.get("state_sha256") != checkpoint["sha256"]
            or type(raw.get("slot")) is not int or raw["slot"] not in (0, 1)
            or Path(checkpoint["path"]).name != f"candidate_training_state_slot_{raw['slot']}.pt"
            or any(raw.get(key) != state[key] for key in (
                "session_contract_sha256", "phase", "epoch_index", "next_batch_offset",
                "global_optimizer_steps", "complete",
            ))
        ):
            raise RuntimeError("NATIVE_CANDIDATE_CANONICAL_POINTER_MISMATCH")
    return dict(value)


def build_native_cursor(
    *, recipe: Mapping[str, str], resume_state: Mapping[str, Any], outcome: str,
) -> dict[str, Any]:
    cursor = {
        "schema_version": CURSOR_SCHEMA, "recipe": dict(recipe),
        "resume_state": dict(resume_state), "outcome": outcome, "test_data_used": False,
    }
    cursor["cursor_sha256"] = canonical_sha256(cursor)
    return require_native_cursor(cursor, expected_recipe=recipe)
