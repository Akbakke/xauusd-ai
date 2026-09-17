#!/usr/bin/env python3
"""Read completed GX1 evidence and the sole next-run policy. Never run training."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path_text: str, *, label: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    if any(part.upper() == "TEST" for part in path.parts):
        raise ValueError(f"{label} must not access a TEST path")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{label} must be an existing non-symlink regular file")
    if path.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ValueError(f"{label} exceeds {MAX_EVIDENCE_BYTES} bytes; bind its manifest")
    return path.resolve(strict=True)


def _repo(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError("repo must be an absolute non-symlink directory")
    if any(part.upper() == "TEST" for part in path.parts):
        raise ValueError("repo must not be below TEST")
    return path.resolve(strict=True)


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=15,
    )
    return completed.stdout.rstrip("\n")


def next_run_readiness(
    repo: Path, *, native_window_policy: Path | None = None,
    native_window_policy_file_sha256: str | None = None,
) -> dict:
    """Report readiness; a calibration exception requires its exact native window."""
    if (native_window_policy is None) != (native_window_policy_file_sha256 is None):
        raise ValueError("NEXT_RUN_NATIVE_WINDOW_BINDING_PAIR_REQUIRED")
    path = _regular_file(str(repo / "NEXT_RUN_POLICY.json"), label="next_run_policy")
    policy = json.loads(path.read_text())
    expected = {"policy_batch_size": 256, "cpu_pipeline_workers": 8,
                "max_wall_seconds": 10800, "progress_interval_forwards": 64}
    if (policy.get("schema_version") != "gx1_next_native_run_policy_v1"
            or policy.get("canonical_source_repo") != str(repo.resolve())
            or policy.get("canonical_branch") != "work/gx1-current"
            or policy.get("required_val_profile") != expected
            or policy.get("native_invocation_seconds") != 12000
            or policy.get("outer_guard_seconds") != 13800
            or policy.get("train_batch_size") != 16
            or policy.get("precision") != "float32" or policy.get("tf32_allowed") is not False
            or policy.get("training_module") != "gx1.scripts.run_unified_exit_native_candidate_window_v1"):
        raise ValueError("NEXT_RUN_PERFORMANCE_PROFILE_INVALID")
    if _git(repo, "branch", "--show-current") != policy["canonical_branch"]:
        raise ValueError("NEXT_RUN_CANONICAL_BRANCH_REQUIRED")
    head = _git(repo, "rev-parse", "HEAD")
    reasons = []
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        reasons.append("source_not_clean_committed")
    clock = policy["gpu_clock_launcher"]
    if _sha256(_regular_file(clock["path"], label="clock_launcher")) != clock["sha256"]:
        raise ValueError("NEXT_RUN_CLOCK_LAUNCHER_MISMATCH")
    native_scope = None
    if native_window_policy is not None:
        # These contract owners use only the standard library at import time.
        # Also support direct execution of this script, outside Python -m.
        import sys
        source_root = str(Path(__file__).resolve().parents[1])
        if source_root not in sys.path:
            sys.path.insert(0, source_root)
        from gx1.contracts.local_random_access_campaign_v2 import read_bound_json
        from gx1.contracts.unified_exit_random_access_sampler_v1 import canonical_sha256 as native_sha256
        from gx1.contracts.unified_exit_native_candidate_campaign_v1 import (
            require_native_window_policy, require_native_run_scope,
        )
        window_path = _regular_file(str(native_window_policy), label="native_window_policy")
        window = require_native_window_policy(read_bound_json(window_path, native_window_policy_file_sha256))
        recipe_path = _regular_file(window["recipe"]["path"], label="native_recipe")
        recipe = read_bound_json(recipe_path, window["recipe"]["sha256"])
        if (recipe.get("schema_version") != "gx1_unified_exit_random_access_full_train_recipe_v1"
                or recipe.get("profile") != "candidate" or recipe.get("test_data_used") is not False
                or recipe.get("source_repo") != str(repo.resolve()) or recipe.get("source_commit") != head
                or recipe.get("recipe_sha256") != native_sha256({k: v for k, v in recipe.items() if k != "recipe_sha256"})):
            raise ValueError("NEXT_RUN_NATIVE_RECIPE_IDENTITY_INVALID")
        # The same owner enforces risk, bounded calibration and all full-run
        # evidence roles; do not duplicate its proof/profile interpretation.
        ceiling = require_native_run_scope(recipe, invocation_number=window["invocation_number"])
        native_scope = {"invocation_number": window["invocation_number"],
                        "optimizer_step_ceiling": ceiling,
                        "full_epoch_training_allowed": ceiling is None,
                        "window_policy_file_sha256": native_window_policy_file_sha256}
    else:
        # An unbound handover remains observation-only, including when the
        # policy has declared a bounded learning measurement permissible.
        reasons.append("native_window_binding_required")
        if policy.get("training_enabled") is not True:
            reasons.append("operator_stop_not_resolved")
        evidence = policy.get("required_evidence", {})
        for role in ("risk_objective", "checkpoint_transition", "learning_calibration",
                     "gpu_batch256_parity", "end_to_end_throughput", "resume_equivalence"):
            item = evidence.get(role)
            if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
                reasons.append(role + "_missing")
                continue
            proof_path = _regular_file(item["path"], label=role)
            if _sha256(proof_path) != item["sha256"]:
                raise ValueError("NEXT_RUN_EVIDENCE_HASH_MISMATCH: " + role)
            proof = json.loads(proof_path.read_text())
            if proof.get("decision") != "PASS" or proof.get("test_data_used") is not False:
                reasons.append(role + "_not_passed")
            if role != "risk_objective" and proof.get("native_val_profile") != expected:
                reasons.append(role + "_profile_mismatch")
    return {"decision": "READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES" if not reasons else "BLOCKED",
            "blocked_reasons": reasons, "canonical_source_repo": str(repo),
            "source_commit": head, "required_val_profile": expected,
            "native_invocation_seconds": 12000, "outer_guard_seconds": 13800,
            "policy_sha256": _sha256(path), "training_started": False,
            **({"native_window_scope": native_scope} if native_scope is not None else {})}


def _native_processes(source: Path) -> list[dict[str, str]]:
    result = subprocess.run(
        ["ps", "-eo", "pid,ppid,etime,pcpu,rss,args"], check=True,
        capture_output=True, text=True, timeout=15,
    )
    found = []
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        command = fields[5]
        if command.startswith(str(source) + "/.venv/bin/python ") and (
            "-m gx1.scripts.run_unified_exit_native_candidate_window_v1 " in command
        ):
            found.append(dict(zip(("pid", "ppid", "elapsed", "cpu_percent", "rss_kib"), fields[:5])))
    return found


def _current_work_status(repo: Path, *, source_only: bool) -> dict | None:
    """Keep mutable operator notes separate from current process/checkpoint evidence."""
    path = repo / "RUNNING_NATIVE_CALIBRATION.json"
    if not path.exists():
        return None
    record = json.loads(_regular_file(str(path), label="current_work").read_text())
    source = _repo(record["source_repo"])
    result = {
        "recorded_status": record["status"],
        "recorded_observed_utc": record.get("latest_observation_utc"),
        "source_repo": str(source),
        "working_head": _git(source, "rev-parse", "HEAD"),
        "working_tree_clean": not bool(_git(source, "status", "--porcelain=v1")),
        "training_source_commit": record.get("source_commit"),
        "runtime_root": record.get("runtime_root"),
        "latest_completed_native": record.get("completed_residual_normalized_fixed256") or record.get("completed_causal_entry_fixed256"),
        "recorded_next_diagnostic": record.get("next_diagnostic"),
        "last_diagnostic_failure": record.get("completed_entry_signal_failure"),
        "latest_completed_diagnostic": record.get("completed_residual_representation") or record.get("completed_entry_signal_inference_check") or record.get("completed_entry_forward_parity"),
        "next_action": record.get("next_action"),
        "latest_model_correction": record.get("completed_main_encoder_correction"),
        "handover_resume_point": record.get("handover_resume_point"),
        "learning_gate": record.get("learning_gate"),
        "full_epoch_training_allowed": record.get("full_epoch_training_allowed", False),
        "observation_is_run_authority": False,
    }
    if source_only:
        return result
    processes = _native_processes(source)
    result["native_processes"] = processes
    result["process_observation"] = "RUNNING" if processes else "NO_NATIVE_PROCESS_OBSERVED"
    directory = record.get("session_directory")
    # A running source tree is frozen. Its operator note can therefore still
    # point at the previous run; the explicit current policy owns the selection.
    policy_path = repo / "NEXT_RUN_POLICY.json"
    policy = json.loads(policy_path.read_text()) if policy_path.is_file() else {}
    diagnostic = policy.get("entry_gradient_diagnostic")
    initial = policy.get("chronological_initial_measurement")
    scope = policy.get("chronological_learning_run") or initial or diagnostic
    if scope is not None:
        output = Path(scope["out_bundle_dir"])
        if output.parent.name != scope["run_id"]:
            raise ValueError("Current native run directory does not match policy")
        result["declared_run_id"] = scope["run_id"]
        if diagnostic is not None:
            result["checkpoint_selection"] = "preserved_training_origin_not_diagnostic_progress"
            result["diagnostic_artifact_root"] = str(output.parent)
        else:
            directory = str(output.parent / (".gx1-candidate-training-session." + output.name))
            result["checkpoint_selection"] = "current_policy_session_not_previous_operator_note"
        preparation_path = output.parent / "PREPARATION_RESULT.json"
        if preparation_path.is_file():
            preparation = json.loads(preparation_path.read_text())
            recipe_binding = preparation["recipe"]
            recipe_path = _regular_file(recipe_binding["path"], label="current native recipe")
            if _sha256(recipe_path) != recipe_binding["sha256"]:
                raise ValueError("Current native recipe hash mismatch")
            recipe = json.loads(recipe_path.read_text())
            if (recipe["run_id"] != scope["run_id"] or recipe["out_bundle_dir"] != scope["out_bundle_dir"]
                    or recipe["source_repo"] != str(source)
                    or recipe["source_commit"] != preparation["source_commit"]):
                raise ValueError("Current native recipe scope mismatch")
            result["diagnostic_source_commit" if diagnostic is not None else "training_source_commit"] = preparation["source_commit"]
            result["runtime_root"] = preparation["runtime_root"]
            receipt_path = Path(preparation["runtime_root"]) / "receipts/invocation-0001.json"
            if receipt_path.is_file():
                result["latest_terminal_receipt"] = json.loads(receipt_path.read_text())
        if processes:
            result["next_action"] = "Observe the active bound run; do not relaunch or change its frozen source. " + ("Review the diagnostic result; no optimizer steps are allowed." if diagnostic is not None else "Verify the zero-step TRAIN-only initial measurement; no learning is measured." if initial is not None else "Review final ONLINE at the declared ceiling.")
        elif result.get("latest_terminal_receipt") is not None:
            result["next_action"] = "The bound invocation has a terminal receipt. " + ("Review the diagnostic result and close its used scope; completion is not learning evidence." if diagnostic is not None else "Audit the initial baseline and close its used scope; no learning is measured." if initial is not None else "Verify final measurement and review learning before any new run.")
    if directory:
        pointer_path = Path(directory) / "CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json"
        result["session_directory"] = directory
        result["checkpoint_pointer_path"] = str(pointer_path)
        if pointer_path.exists():
            result["checkpoint"] = json.loads(
                _regular_file(str(pointer_path), label="current_checkpoint_pointer").read_text()
            )
        else:
            result["checkpoint"] = None
            result["checkpoint_observation"] = "NOT_YET_WRITTEN_NO_HISTORICAL_FALLBACK"
    return result


def native_status(binding_path: Path, *, source_only: bool = False) -> dict[str, object]:
    """Observe the explicit native run without touching model/data payloads.

    This verifies small immutable bindings, not complete dataset/checkpoint bytes.
    Mutable progress is an observation; the existing campaign owns resume validation.
    """
    binding_path = _regular_file(str(binding_path), label="native_binding")
    binding = json.loads(binding_path.read_text())
    if binding.get("schema_version") != "gx1_native_handover_binding_v1":
        raise ValueError("invalid native handover binding schema")
    source = _repo(binding["source_repo"])
    commit = _git(source, "rev-parse", "HEAD")
    if commit != binding["source_commit"]:
        raise ValueError("native source commit mismatch")
    dirty = _git(source, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise ValueError("native source is dirty")
    verified = {}
    for role, artifact in binding["immutable_artifacts"].items():
        path = _regular_file(artifact["path"], label=role)
        digest = _sha256(path)
        if digest != artifact["sha256"]:
            raise ValueError(f"native artifact hash mismatch: {role}")
        verified[role] = {"path": str(path), "sha256": digest}
    out = {
        "schema_version": "gx1_native_handover_observation_v1",
        "decision": "OBSERVATION_ONLY_NOT_RUN_AUTHORITY",
        "observation_scope": "completed_run_history; current_work reports the current canonical workspace",
        "current_work": _current_work_status(binding_path.parent, source_only=source_only),
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(source), "source_commit": commit, "source_clean": True,
        "immutable_artifacts_verified": verified, "test_accessed": False,
        "state_payload_rehashed": False,
    }
    # Recorded operator intent is separate from live process observation.
    out["recorded_operator_stop"] = binding.get("observed_stop")
    out["next_run"] = next_run_readiness(binding_path.parent) if (binding_path.parent / "NEXT_RUN_POLICY.json").exists() else None
    for name, artifact in binding.get("analyses", {}).items():
        p = _regular_file(artifact["path"], label="analysis")
        if _sha256(p) != artifact["sha256"]:
            raise ValueError("analysis hash mismatch: " + name)
    out["analyses"] = binding.get("analyses", {})
    completed = binding.get("completed_val_result")
    if completed:
        p = _regular_file(completed["path"], label="completed_val_result")
        if _sha256(p) != completed["sha256"]:
            raise ValueError("completed VAL result hash mismatch")
        result = json.loads(p.read_text())
        out["completed_val"] = {key: result.get(key) for key in (
            "decision", "semantic_result_sha256", "materialized_state_view_count",
            "model_forward_count", "entry_exit_policy_metrics", "rollout_execution_complete")}
        out["completed_val"]["result_file"] = completed
    if source_only:
        return out

    def read_json(path: Path) -> dict | None:
        if not path.exists():
            return None
        return json.loads(_regular_file(str(path), label="runtime_metadata").read_text())

    runtime = _repo(binding["runtime_root"])
    session = _repo(binding["training_session"])
    pointer = read_json(session / "CANDIDATE_TRAINING_SESSION_RESUME_POINTER.json")
    if pointer is None:
        raise ValueError("native training checkpoint pointer is missing")
    if pointer.get("session_contract_sha256") != binding["session_contract_sha256"]:
        raise ValueError("native session contract mismatch")
    out["checkpoint"] = pointer
    processes = _native_processes(source)
    out["native_processes"] = processes
    out["process_observation"] = "RUNNING" if processes else "NO_NATIVE_PROCESS_OBSERVED"
    active = read_json(runtime / "ACTIVE_INVOCATION.json")
    out["active_invocation"] = ({k: active.get(k) for k in
        ("invocation_id", "invocation_number", "kind", "started_utc")} if active else None)
    # Numeric ordering is display-only. It never selects a checkpoint for execution.
    receipts = list((runtime / "receipts").glob("invocation-*.json"))
    if receipts:
        path = max(receipts, key=lambda p: int(p.stem.split("-")[-1]))
        receipt = read_json(path)
        out["latest_terminal_receipt"] = {k: receipt.get(k) for k in
            ("invocation_number", "outcome", "trainer_guard_exit_code",
             "progress_observer_exit_code", "guard_decision")}
    if active:
        invocation = active["invocation_id"]
        if not invocation.startswith("invocation-") or not invocation[11:].isdigit():
            raise ValueError("invalid native invocation id")
        guard = runtime / "guard" / (invocation + ".log")
        if guard.is_file():
            with guard.open("rb") as handle:
                handle.seek(max(0, guard.stat().st_size - 3000))
                out["guard_tail"] = handle.read().decode(errors="replace").splitlines()[-3:]
    if pointer["phase"] == "validation":
        path = session / "native_val" / f"epoch_{pointer['epoch_index'] + 1:04d}" / "ROLLOUT_PROGRESS.json"
        progress = read_json(path)
        if progress:
            out["native_val_progress"] = {k: value for k, value in progress.items()
                if not isinstance(value, (list, dict))}
            trades = [side for pair in progress.get("trade_accumulators", []) for side in pair]
            statuses = {}
            for trade in trades:
                key = trade["status"]
                statuses[key] = statuses.get(key, 0) + 1
            out["simulated_side_status_counts"] = statuses
            out["native_val_progress"]["updated_utc"] = datetime.fromtimestamp(
                path.stat().st_mtime, timezone.utc).isoformat()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-binding", type=Path)
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--require-next-run", action="store_true")
    parser.add_argument("--native-window-policy", type=Path)
    parser.add_argument("--native-window-policy-file-sha256")
    args = parser.parse_args()
    if (args.native_window_policy is None) != (args.native_window_policy_file_sha256 is None):
        parser.error("--native-window-policy and --native-window-policy-file-sha256 are required together")
    if args.native_window_policy is not None and not args.require_next_run:
        parser.error("a native window requires --require-next-run")
    repo = Path(__file__).resolve().parents[1]
    if args.require_next_run:
        result = next_run_readiness(
            repo, native_window_policy=args.native_window_policy,
            native_window_policy_file_sha256=args.native_window_policy_file_sha256,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["decision"] == "READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES" else 78
    if args.native_binding is None:
        parser.error("--native-binding is required; legacy bundle/launch paths are retired")
    print(json.dumps(native_status(args.native_binding, source_only=args.source_only), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
