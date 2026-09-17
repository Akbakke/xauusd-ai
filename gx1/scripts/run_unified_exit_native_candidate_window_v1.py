"""Run one native candidate window inside the existing reboot campaign guard."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Mapping

from gx1.contracts.local_random_access_campaign_v2 import (
    ACTIVE_SCHEMA, PROGRESS_SCHEMA, canonical_sha256, file_sha256,
    read_bound_json, require_plan, require_progress,
)
from gx1.contracts.unified_exit_native_candidate_campaign_v1 import (
    NATIVE_KIND, NATIVE_PHASE, build_native_cursor, require_native_cursor,
    require_native_window_policy, require_native_run_scope, native_completed_val_ceiling,
)
from gx1.scripts import local_random_access_campaign_v2 as campaign
from gx1.scripts import run_unified_exit_random_access_full_train_v1 as native


def _context(
    policy_path: Path, policy_file_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    policy = require_native_window_policy(read_bound_json(policy_path, policy_file_sha256))
    required_env = (
        "GX1_CAMPAIGN_PLAN_PATH", "GX1_CAMPAIGN_PLAN_FILE_SHA256",
        "GX1_CAMPAIGN_PLAN_SHA256", "GX1_CAMPAIGN_INVOCATION_SHA256",
    )
    if any(not os.environ.get(key) for key in required_env):
        raise RuntimeError("NATIVE_CANDIDATE_CAMPAIGN_ENV_MISSING")
    plan = require_plan(read_bound_json(
        Path(os.environ["GX1_CAMPAIGN_PLAN_PATH"]),
        os.environ["GX1_CAMPAIGN_PLAN_FILE_SHA256"],
    ), verify_files=True)
    matches = [item for item in plan["checked_invocations"]
               if item["invocation_sha256"] == os.environ["GX1_CAMPAIGN_INVOCATION_SHA256"]]
    if (
        plan["plan_sha256"] != os.environ["GX1_CAMPAIGN_PLAN_SHA256"]
        or plan["phase"] != NATIVE_PHASE or len(matches) != 1
        or plan["native_recipe"] != policy["recipe"]
    ):
        raise RuntimeError("NATIVE_CANDIDATE_CAMPAIGN_IDENTITY_INVALID")
    invocation = matches[0]
    execution = read_bound_json(Path(invocation["execution_manifest"]["path"]),
                                invocation["execution_manifest"]["sha256"])
    runtime = Path(plan["runtime_root"])
    if (
        invocation["kind"] != NATIVE_KIND
        or invocation["invocation_number"] != policy["invocation_number"]
        or invocation["progress_path"] != policy["progress_path"]
        or invocation["checkpoint"]["pointer_path"] != policy["campaign_cursor_path"]
        or execution["prelaunch_manifest"] != policy["recipe"]
        or execution["train_session_manifest"] != {"path": str(policy_path), "sha256": policy_file_sha256}
        or execution["train_session_manifest_sha256"] != policy["policy_sha256"]
        or policy["budget_path"] != str(runtime / "budgets" / f"{invocation['invocation_id']}.json")
        or policy["campaign_cursor_path"] != str(runtime / "native-candidate-cursor" / "RESUME_CURSOR.json")
    ):
        raise RuntimeError("NATIVE_CANDIDATE_WINDOW_CONTEXT_INVALID")
    marker_path = runtime / "ACTIVE_INVOCATION.json"
    marker = read_bound_json(marker_path, file_sha256(marker_path))
    if (
        marker.get("schema_version") != ACTIVE_SCHEMA
        or marker.get("marker_sha256") != canonical_sha256({k: v for k, v in marker.items() if k != "marker_sha256"})
        or any(marker.get(key) != expected for key, expected in {
            "plan_sha256": plan["plan_sha256"],
            "invocation_sha256": invocation["invocation_sha256"],
            "invocation_number": invocation["invocation_number"],
            "invocation_id": invocation["invocation_id"], "kind": NATIVE_KIND,
            "launcher_argv_sha256": invocation["launcher_argv_sha256"],
        }.items())
    ):
        raise RuntimeError("NATIVE_CANDIDATE_ACTIVE_MARKER_INVALID")
    return policy, plan, invocation, marker


def _expected_training_pointer(
    *, policy: Mapping[str, Any], invocation: Mapping[str, Any], marker: Mapping[str, Any],
) -> str | None:
    recipe = read_bound_json(Path(policy["recipe"]["path"]), policy["recipe"]["sha256"])
    output = Path(recipe["out_bundle_dir"])
    directory = output.parent / (native.trainer._CANDIDATE_TRAINING_SESSION_DIR_PREFIX + output.name)
    if policy["training_session_directory"] != str(directory):
        raise RuntimeError("NATIVE_CANDIDATE_SESSION_DIRECTORY_MISMATCH")
    cursor_path = Path(policy["campaign_cursor_path"])
    if invocation["checkpoint"]["before_mode"] == "GENESIS":
        if (
            marker["pointer_before_sha256"] != "GENESIS"
            or cursor_path.parent.exists() or cursor_path.parent.is_symlink()
            or directory.exists() or directory.is_symlink()
        ):
            raise RuntimeError("NATIVE_CANDIDATE_GENESIS_NOT_FRESH")
        return None
    if invocation["checkpoint"]["before_mode"] != "PREVIOUS_RECEIPT_AFTER":
        raise RuntimeError("NATIVE_CANDIDATE_RESUME_MODE_INVALID")
    cursor = require_native_cursor(
        read_bound_json(cursor_path, marker["pointer_before_sha256"]),
        expected_recipe=policy["recipe"], verify_files=True,
    )
    state = cursor["resume_state"]
    if cursor["outcome"] != "RESUMABLE" or Path(state["training_pointer"]["path"]).parent != directory:
        raise RuntimeError("NATIVE_CANDIDATE_PREDECESSOR_NOT_RESUMABLE")
    return state["training_pointer"]["sha256"]


def run_window(*, policy_path: Path, policy_file_sha256: str, progress_path: Path) -> dict[str, Any]:
    native.trainer._require_cuda_trainer_guard_execution(execution_tier="canonical")
    policy, plan, invocation, marker = _context(policy_path, policy_file_sha256)
    if str(progress_path) != policy["progress_path"]:
        raise RuntimeError("NATIVE_CANDIDATE_PROGRESS_ARGUMENT_MISMATCH")
    expected_pointer = _expected_training_pointer(policy=policy, invocation=invocation, marker=marker)
    progress_path = Path(policy["progress_path"])
    if progress_path.exists() or progress_path.is_symlink():
        raise RuntimeError("NATIVE_CANDIDATE_WINDOW_PROGRESS_EXISTS")
    recipe = read_bound_json(Path(policy["recipe"]["path"]), policy["recipe"]["sha256"])
    step_ceiling = require_native_run_scope(recipe, invocation_number=policy["invocation_number"])
    budget_path = Path(policy["budget_path"])
    campaign._prepare_private_directory(budget_path.parent, label="native window budget")
    campaign._atomic_new(budget_path, {
        "schema_version": native.launch_owner.CANDIDATE_EXECUTION_BUDGET_SCHEMA,
        "recipe_json": policy["recipe"]["path"], "recipe_sha256": policy["recipe"]["sha256"],
        "expected_active_pointer_sha256": expected_pointer,
        "stop_after_optimizer_steps": step_ceiling,
        "stop_after_completed_val_epochs": native_completed_val_ceiling(recipe),
        "max_invocation_seconds": policy["max_invocation_seconds"],
    })
    result = native.run_guarded_native_candidate_invocation(
        recipe_path=Path(policy["recipe"]["path"]), recipe_file_sha256=policy["recipe"]["sha256"],
        execution_budget_path=budget_path, execution_budget_file_sha256=file_sha256(budget_path),
    )
    outcomes = {"PAUSED_RESUMABLE": "RESUMABLE", "COMPLETE": "COMPLETE"}
    if result.get("decision") not in outcomes:
        raise RuntimeError("NATIVE_CANDIDATE_WINDOW_OUTCOME_INVALID")
    outcome = outcomes[result["decision"]]
    cursor = build_native_cursor(recipe=policy["recipe"], resume_state=result["resume_state"], outcome=outcome)
    cursor_path = Path(policy["campaign_cursor_path"])
    campaign._prepare_private_directory(cursor_path.parent, label="native campaign cursor")
    native.trainer._candidate_training_session_atomic_write_json(cursor_path, cursor)
    state = cursor["resume_state"]
    batches = math.ceil(plan["entry_pairs_per_epoch"] / plan["selected_batch_size"])
    progress = {
        "schema_version": PROGRESS_SCHEMA, "plan_sha256": plan["plan_sha256"],
        "invocation_sha256": invocation["invocation_sha256"], "phase": NATIVE_KIND,
        "epoch_index": state["epoch_index"], "global_optimizer_steps": state["global_optimizer_steps"],
        "next_batch_offset": state["next_batch_offset"], "total_batches": batches,
        "completed_units": state["global_optimizer_steps"],
        "total_units": max(1, step_ceiling) if "chronological_prefix" in recipe else batches * 30,
        "epoch_schedule_sha256": state["epoch_schedule_sha256"],
        "selection_receipt_sha256": plan["selection_artifact_sha256"],
        "checkpoint_pointer": {"path": str(cursor_path), "sha256": file_sha256(cursor_path)},
        "terminal": True, "outcome": outcome, "observed_utc": datetime.now(timezone.utc).isoformat(),
    }
    progress["progress_sha256"] = canonical_sha256(progress)
    checked = require_progress(progress, plan_sha256=plan["plan_sha256"], invocation=invocation,
                               expected_selection_receipt_sha256=plan["selection_artifact_sha256"], verify_file=True)
    campaign._atomic_new(progress_path, checked)
    return {"decision": outcome, "epoch_index": state["epoch_index"],
            "global_optimizer_steps": state["global_optimizer_steps"],
            "active_val_model_forwards": state["active_val_model_forwards"],
            "progress": {"path": str(progress_path), "sha256": file_sha256(progress_path)},
            "test_data_used": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window-policy", type=Path, required=True)
    parser.add_argument("--window-policy-file-sha256", required=True)
    parser.add_argument("--progress-path", type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(json.dumps(run_window(policy_path=args.window_policy,
                                policy_file_sha256=args.window_policy_file_sha256,
                                progress_path=args.progress_path), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
