"""Publish one explicitly admitted, model-free whole-TRAIN target-RMS fit.

The consumed input recipe supplies target/data provenance only. This CLI never
calls a native launcher, constructs a model, or resumes a training session.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import stat
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

PLAN_SCHEMA = "gx1_dynamic_m5_loss_scale_fit_plan_v1"
BINDING_SCHEMA = "gx1_dynamic_m5_loss_scale_fit_execution_binding_v1"
REPO = Path("/home/andre2/src/GX1_CURRENT")
SCRIPT = REPO / "gx1/scripts/fit_dynamic_m5_loss_scales_v1.py"
RUNS_ROOT = Path("/home/andre2/GX1_RUNS")
_SHA = re.compile(r"[0-9a-f]{64}")
_HEAD = re.compile(r"[0-9a-f]{40}")


def _require(condition, error):
    if not condition:
        raise RuntimeError(error)


def _identity(path):
    _require(path.is_absolute() and path.resolve(strict=True) == path and not path.is_symlink(),
        "DYNAMIC_M5_LOSS_FIT_PATH_INVALID")
    value = path.stat()
    _require(stat.S_ISREG(value.st_mode), "DYNAMIC_M5_LOSS_FIT_FILE_TYPE_INVALID")
    return {key: getattr(value, key) for key in
        ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _ref(value):
    _require(isinstance(value, dict) and set(value) == {"path", "sha256"}
        and isinstance(value["path"], str) and Path(value["path"]).is_absolute()
        and isinstance(value["sha256"], str) and _SHA.fullmatch(value["sha256"]) is not None,
        "DYNAMIC_M5_LOSS_FIT_BINDING_INVALID")
    return value


def _read_json(ref, identities):
    _ref(ref)
    path = Path(ref["path"])
    before = _identity(path)
    data = path.read_bytes()
    _require(_digest(data) == ref["sha256"] and _identity(path) == before,
        "DYNAMIC_M5_LOSS_FIT_INPUT_SHA_OR_IDENTITY_INVALID")
    identities[str(path)] = before
    # Reuse the existing path/SHA-bound JSON owner for metadata.
    from gx1.contracts.local_random_access_campaign_v2 import read_bound_json
    value = read_bound_json(path, ref["sha256"])
    _require(_identity(path) == before, "DYNAMIC_M5_LOSS_FIT_INPUT_CHANGED")
    return value


def _require_plan(plan):
    keys = {"schema_version", "run_id", "output_dir", "prepared_source_head", "input_recipe",
        "population_design", "chunk_rows", "progress_interval_batches", "expected_train_rows"}
    _require(isinstance(plan, dict) and set(plan) == keys and plan["schema_version"] == PLAN_SCHEMA,
        "DYNAMIC_M5_LOSS_FIT_PLAN_INVALID")
    _require(isinstance(plan["run_id"], str) and bool(re.fullmatch(r"[A-Za-z0-9_-]+", plan["run_id"]))
        and isinstance(plan["prepared_source_head"], str) and _HEAD.fullmatch(plan["prepared_source_head"]) is not None,
        "DYNAMIC_M5_LOSS_FIT_PLAN_IDENTITY_INVALID")
    _require(isinstance(plan["output_dir"], str), "DYNAMIC_M5_LOSS_FIT_OUTPUT_INVALID")
    output = Path(plan["output_dir"])
    _require(output.is_absolute() and output.resolve(strict=False) == output
        and output.is_relative_to(RUNS_ROOT) and output != RUNS_ROOT,
        "DYNAMIC_M5_LOSS_FIT_OUTPUT_INVALID")
    for key in ("chunk_rows", "progress_interval_batches", "expected_train_rows"):
        _require(type(plan[key]) is int and plan[key] > 0, "DYNAMIC_M5_LOSS_FIT_PLAN_COUNT_INVALID")
    _require(plan["chunk_rows"] <= plan["expected_train_rows"], "DYNAMIC_M5_LOSS_FIT_CHUNK_INVALID")
    _ref(plan["input_recipe"])
    _ref(plan["population_design"])
    return plan


def _require_stage(policy, plan_ref, script_ref):
    _require(isinstance(policy, dict) and all(policy.get(k) is False for k in
        ("training_enabled", "full_epoch_training_allowed", "full_val_allowed")),
        "DYNAMIC_M5_LOSS_FIT_TRAINING_MUST_STAY_CLOSED")
    stage = policy.get("dynamic_m5_loss_scale_fit_run")
    _require(isinstance(stage, dict) and stage.get("execution_authorized") is True
        and stage.get("training_enabled") is False and stage.get("relaunch_allowed") is False
        and type(stage.get("planned_invocations")) is int and stage["planned_invocations"] == 1
        and type(stage.get("completed_invocations")) is int and stage["completed_invocations"] == 0
        and stage.get("plan") == plan_ref and stage.get("script") == script_ref,
        "DYNAMIC_M5_LOSS_FIT_NOT_EXACTLY_AUTHORIZED")


def _source_state():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    _require(git("branch", "--show-current") == "work/gx1-current"
        and not git("status", "--porcelain"), "DYNAMIC_M5_LOSS_FIT_SOURCE_NOT_CLEAN")
    return git("rev-parse", "HEAD")


def _unchanged(binding, identities):
    _require(_source_state() == binding["source_head"], "DYNAMIC_M5_LOSS_FIT_SOURCE_CHANGED")
    _require(_digest((REPO / "NEXT_RUN_POLICY.json").read_bytes()) == binding["policy_sha256"],
        "DYNAMIC_M5_LOSS_FIT_POLICY_CHANGED")
    for name, expected in identities.items():
        _require(_identity(Path(name)) == expected, "DYNAMIC_M5_LOSS_FIT_INPUT_CHANGED:" + name)


def _publish(output, name, value):
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path = output / name
    fd, temporary = tempfile.mkstemp(prefix="." + name + ".", dir=output)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    fd = os.open(output, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"path": str(path), "sha256": _digest(data)}


def _prepare_targets(plan, identities):
    import numpy as np
    from gx1.contracts.local_random_access_campaign_v2 import (
        fresh_file_verification_scope, capture_verified_file_snapshot,
    )
    from gx1.contracts import entry_observed_market_v1 as observed
    from gx1.contracts.entry_observed_market_horizons_v1 import (
        require_m5_single_epoch_population, build_masked_support_forecast_targets,
    )
    from gx1.contracts.entry_model_native_aux_targets_v3 import (
        MODEL_NATIVE_FORECAST_TARGET_COLUMNS, require_m5_forecast_observation_storage,
    )
    from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import dynamic_m5_loss_fit_lineage
    with fresh_file_verification_scope():
        recipe = _read_json(plan["input_recipe"], identities)
        design = _read_json(plan["population_design"], identities)
        _require(recipe.get("chronological_prefix", {}).get("design") == plan["population_design"],
            "DYNAMIC_M5_LOSS_FIT_RECIPE_DESIGN_MISMATCH")
        scope = observed.require_entry_observed_market_scope(recipe)
        population = require_m5_single_epoch_population(design)
        _require(isinstance(scope, dict) and isinstance(population, dict)
            and population["train_rows"] == plan["expected_train_rows"],
            "DYNAMIC_M5_LOSS_FIT_POPULATION_MISMATCH")
        lineage = dynamic_m5_loss_fit_lineage(scope, population)
        snapshot = capture_verified_file_snapshot()
    # The hash memo is closed before physical target/price/index construction.
    for ref in (population["binding"], population["witness"]["eligible_parent_rows"],
            population["witness"]["epoch0_parent_order"], lineage["train_physical_source"]["parquet"],
            lineage["train_physical_source"]["manifest"], lineage["source_quote_binding"]):
        identities[ref["path"]] = _identity(Path(ref["path"]))
    source = observed.read_dynamic_m5_target_source(scope, split="train")
    snapshot.require_unchanged()
    rows = population["eligible_parent_rows"]
    _require(rows.dtype == np.int64 and np.array_equal(rows, np.arange(plan["expected_train_rows"], dtype=np.int64))
        and source["first_valid"].all(), "DYNAMIC_M5_LOSS_FIT_WHOLE_TRAIN_REQUIRED")
    valid = require_m5_forecast_observation_storage(source["frame"], context="DYNAMIC_M5_LOSS_FIT")
    support, support_valid = build_masked_support_forecast_targets(
        frame=source["frame"], scope=scope, split="train", admitted_mask=source["first_valid"])
    forecast = np.concatenate((source["frame"][list(MODEL_NATIVE_FORECAST_TARGET_COLUMNS)].to_numpy(dtype=np.float32), support), axis=1)
    forecast_valid = np.concatenate((valid, support_valid), axis=1)
    provider = observed.DynamicM5TargetProvider(index=source["index"], fill_surface=source["fill"],
        admitted_rows=source["first_valid"], economics=scope["economics"])
    return lineage, rows, provider, forecast, forecast_valid, snapshot


def _target_batches(plan, rows, provider, forecast, forecast_valid, started):
    from gx1.contracts import entry_observed_market_v1 as observed
    specs = (
        ("dynamic_terminal_net_markout_bps", observed.DYNAMIC_M5_MARKOUT_TARGET_KEY, observed.DYNAMIC_M5_MARKOUT_VALID_KEY),
        ("dynamic_mfe_bps", observed.DYNAMIC_M5_MFE_TARGET_KEY, observed.DYNAMIC_M5_PATH_VALID_KEY),
        ("dynamic_pre_mfe_mae_upper_bps", observed.DYNAMIC_M5_MAE_TARGET_KEY, observed.DYNAMIC_M5_PATH_VALID_KEY),
        ("dynamic_closed_mfe_bps", observed.DYNAMIC_M5_CLOSED_MFE_TARGET_KEY, observed.DYNAMIC_M5_CLOSED_PATH_VALID_KEY),
        ("dynamic_closed_pre_mfe_mae_upper_bps", observed.DYNAMIC_M5_CLOSED_MAE_TARGET_KEY, observed.DYNAMIC_M5_CLOSED_PATH_VALID_KEY),
    )
    for batch_number, begin in enumerate(range(0, len(rows), plan["chunk_rows"]), 1):
        parents = rows[begin:begin + plan["chunk_rows"]]
        target = provider.targets_for_rows(parents)
        surfaces = {name: (target[key], target[mask]) for name, key, mask in specs}
        surfaces["forecast_pred"] = (forecast[parents], forecast_valid[parents])
        yield parents, surfaces
        if batch_number % plan["progress_interval_batches"] == 0 or begin + len(parents) == len(rows):
            print(json.dumps({"phase": "TRAIN_target_RMS_fit", "batches_completed": batch_number,
                "parents_completed": begin + len(parents), "train_rows": len(rows),
                "elapsed_seconds": time.monotonic() - started}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execution-binding-json", type=Path, required=True)
    parser.add_argument("--execution-binding-sha256", required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    claimed, output, binding, identities, snapshot = False, None, None, {}, None
    try:
        _require(os.environ.get("GX1_CAPPED_CLASS") == "audit"
            and int(os.environ["GX1_CAPPED_MEMORY_BYTES"]) == 4294967296
            and int(os.environ["GX1_CAPPED_SWAP_BYTES"]) == 536870912,
            "DYNAMIC_M5_LOSS_FIT_CAP_REQUIRED")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        sys.dont_write_bytecode = True
        plan_ref = {"path": str(args.plan_json), "sha256": args.plan_sha256}
        binding_ref = {"path": str(args.execution_binding_json), "sha256": args.execution_binding_sha256}
        plan = _require_plan(_read_json(plan_ref, identities))
        binding = _read_json(binding_ref, identities)
        keys = {"schema_version", "source_head", "policy_sha256", "plan_sha256", "script_sha256"}
        _require(isinstance(binding, dict) and set(binding) == keys and binding["schema_version"] == BINDING_SCHEMA
            and isinstance(binding["source_head"], str) and _HEAD.fullmatch(binding["source_head"]) is not None
            and binding["plan_sha256"] == args.plan_sha256,
            "DYNAMIC_M5_LOSS_FIT_EXECUTION_BINDING_INVALID")
        script_ref = {"path": str(SCRIPT), "sha256": binding["script_sha256"]}
        _ref(script_ref)
        identities[str(SCRIPT)] = _identity(SCRIPT)
        _require(Path(__file__).absolute() == SCRIPT and _digest(SCRIPT.read_bytes()) == binding["script_sha256"],
            "DYNAMIC_M5_LOSS_FIT_SCRIPT_BINDING_INVALID")
        policy_ref = {"path": str(REPO / "NEXT_RUN_POLICY.json"), "sha256": binding["policy_sha256"]}
        policy = _read_json(policy_ref, identities)
        _require_stage(policy, plan_ref, script_ref)
        _unchanged(binding, identities)
        output = Path(plan["output_dir"])
        output.mkdir(exist_ok=True)
        _require(output.is_dir() and not output.is_symlink() and output.resolve(strict=True) == output,
            "DYNAMIC_M5_LOSS_FIT_OUTPUT_INVALID")
        _require(not any((output / name).exists() for name in ("CLAIM.json", "LOSS_SCALES.json", "RESULT.json", "TERMINAL.json")),
            "DYNAMIC_M5_LOSS_FIT_ALREADY_CONSUMED")
        claim = _publish(output, "CLAIM.json", {"schema_version": "gx1_dynamic_m5_loss_scale_fit_claim_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(), "plan": plan_ref, "execution_binding": binding_ref,
            "run_id": plan["run_id"], "pid": os.getpid(), "relaunch_allowed": False})
        claimed = True
        setup_started = time.monotonic()
        lineage, rows, provider, forecast, forecast_valid, snapshot = _prepare_targets(plan, identities)
        setup_seconds = time.monotonic() - setup_started
        _unchanged(binding, identities)
        snapshot.require_unchanged()
        print(json.dumps({"phase": "target_source_ready", "train_rows": len(rows), "setup_seconds": setup_seconds}), flush=True)
        from gx1.contracts.entry_dynamic_m5_loss_scale_v1 import fit_dynamic_m5_loss_scales
        fit_started = time.monotonic()
        artifact = fit_dynamic_m5_loss_scales(batches=_target_batches(plan, rows, provider, forecast, forecast_valid, started),
            lineage=lineage, expected_parent_rows=rows, fit_split="train")
        fit_seconds = time.monotonic() - fit_started
        snapshot.require_unchanged()
        _unchanged(binding, identities)
        scales = _publish(output, "LOSS_SCALES.json", artifact)
        result = {"schema_version": "gx1_dynamic_m5_loss_scale_fit_result_v1",
            "decision": "WHOLE_TRAIN_TARGET_RMS_LOSS_UNITS_PUBLISHED_NO_NATIVE_OR_TRAINING_ADMISSION",
            "created_utc": datetime.now(timezone.utc).isoformat(), "run_id": plan["run_id"],
            "plan": plan_ref, "execution_binding": binding_ref, "execution_binding_values": binding,
            "claim": claim, "loss_scales": scales, "artifact_semantic_sha256": artifact["artifact_sha256"],
            "contract_sha256": artifact["contract"]["contract_sha256"], "fit_rows": artifact["fit_rows"],
            "input_recipe": plan["input_recipe"], "population_design": plan["population_design"],
            "input_recipe_usage": "target_scope_and_data_provenance_only_never_relaunch",
            "verified_metadata_input_file_count": snapshot.file_count, "input_stat_identities": identities,
            "source_unchanged": True, "policy_unchanged": True, "verified_inputs_unchanged": True,
            "setup_seconds": setup_seconds, "fit_seconds": fit_seconds, "elapsed_seconds": time.monotonic() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "model_constructions": 0, "model_forwards": 0, "optimizer_updates": 0,
            "gpu_operations": 0, "input_normalization_refits": 0, "test_data_used": False,
            "training_enabled": False, "native_training_admitted": False, "relaunch_allowed": False}
        result_ref = _publish(output, "RESULT.json", result)
        _publish(output, "TERMINAL.json", {"schema_version": "gx1_dynamic_m5_loss_scale_fit_terminal_v1",
            "status": "COMPLETE", "exit_code": 0, "result": result_ref,
            "observed_utc": datetime.now(timezone.utc).isoformat(), "source_unchanged": True,
            "completed_invocations": 1, "training_enabled": False, "test_data_used": False,
            "relaunch_allowed": False, "elapsed_seconds": time.monotonic() - started})
        print(json.dumps({"decision": result["decision"], "result": result_ref, "loss_scales": scales,
            "setup_seconds": setup_seconds, "fit_seconds": fit_seconds}), flush=True)
        return 0
    except BaseException as error:
        if claimed:
            unchanged = False
            try:
                _unchanged(binding, identities)
                if snapshot is not None:
                    snapshot.require_unchanged()
                unchanged = True
            except BaseException:
                pass
            _publish(output, "TERMINAL.json", {"schema_version": "gx1_dynamic_m5_loss_scale_fit_terminal_v1",
                "status": "FAILED_NO_ADMISSION", "exit_code": 1, "error_type": type(error).__name__,
                "error": str(error), "observed_utc": datetime.now(timezone.utc).isoformat(),
                "source_and_inputs_unchanged": unchanged,
                "source_unchanged": True if unchanged and snapshot is not None else None,
                "test_data_used": None, "test_access_observation": "unproved_after_failure",
                "completed_invocations": 1, "training_enabled": False,
                "relaunch_allowed": False, "elapsed_seconds": time.monotonic() - started})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
