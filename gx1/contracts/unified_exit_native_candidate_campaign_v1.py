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
    if (
        recipe.get("schema_version") != "gx1_unified_exit_random_access_full_train_recipe_v1"
        or recipe.get("profile") != "candidate" or recipe.get("test_data_used") is not False
        or recipe.get("source_repo") != str(source_repo)
        or recipe.get("source_commit") != source_commit
        or recipe.get("recipe_sha256") != native_sha256({k: v for k, v in recipe.items() if k != "recipe_sha256"})
        or controls.get("epochs") != 30 or controls.get("batch_size") != 16
        or controls.get("early_stopping_patience") != 5
        or controls.get("checkpoint_monitor") != "entry_exit_policy_metrics.mean_net_bps_per_entry"
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
    split = root["splits"]["train"]
    path = Path(split["manifest_path"])
    manifest = read_bound_json(path, file_sha256(path))
    count = manifest.get("entry_row_count")
    if (
        manifest.get("manifest_sha256") != split["manifest_sha256"]
        or manifest.get("manifest_sha256") != native_sha256({k: v for k, v in manifest.items() if k != "manifest_sha256"})
        or type(count) is not int or count < 1
        or manifest.get("parent_entry_source_rows") != count
    ):
        raise RuntimeError("NATIVE_CANDIDATE_FULL_POPULATION_METADATA_INVALID")
    return recipe, count


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
        or value["max_invocation_seconds"] != 5400
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
    )

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
