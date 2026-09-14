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


def next_run_readiness(repo: Path) -> dict:
    """Report the one current route; this never launches or grants evidence."""
    path = _regular_file(str(repo / "NEXT_RUN_POLICY.json"), label="next_run_policy")
    policy = json.loads(path.read_text())
    expected = {"policy_batch_size": 256, "cpu_pipeline_workers": 8,
                "max_wall_seconds": 10800, "progress_interval_forwards": 64}
    if (policy.get("schema_version") != "gx1_next_native_run_policy_v1"
            or policy.get("canonical_source_repo") != str(repo.resolve())
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
    if policy.get("training_enabled") is not True:
        reasons.append("operator_stop_not_resolved")
    evidence = policy.get("required_evidence", {})
    for role in ("risk_objective", "gpu_batch256_parity", "end_to_end_throughput", "resume_equivalence"):
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
        if role != "risk_objective" and proof.get("required_val_profile") != expected:
            reasons.append(role + "_profile_mismatch")
    clock = policy["gpu_clock_launcher"]
    if _sha256(_regular_file(clock["path"], label="clock_launcher")) != clock["sha256"]:
        raise ValueError("NEXT_RUN_CLOCK_LAUNCHER_MISMATCH")
    return {"decision": "READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES" if not reasons else "BLOCKED",
            "blocked_reasons": reasons, "canonical_source_repo": str(repo),
            "source_commit": head, "required_val_profile": expected,
            "native_invocation_seconds": 12000, "outer_guard_seconds": 13800,
            "policy_sha256": _sha256(path), "training_started": False}


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
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.require_next_run:
        result = next_run_readiness(repo)
        print(json.dumps(result, indent=2))
        return 0 if result["decision"] == "READY_FOR_EXISTING_BOUND_CAMPAIGN_GATES" else 78
    if args.native_binding is None:
        parser.error("--native-binding is required; legacy bundle/launch paths are retired")
    print(json.dumps(native_status(args.native_binding, source_only=args.source_only), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
