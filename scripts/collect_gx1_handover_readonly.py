#!/usr/bin/env python3
"""Read CURRENT source and its sole current policy. Never run training."""
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
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError(f"{label} must use its canonical path; symlink parents are forbidden")
    return resolved


def _repo(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError("repo must be an absolute non-symlink directory")
    if any(part.upper() == "TEST" for part in path.parts):
        raise ValueError("repo must not be below TEST")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise ValueError("repo must use its canonical path; symlink parents are forbidden")
    return resolved


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


# Ported from the archived handover (archive/gx1-engine-audit-v9-20260926:
# scripts/gx1_handover.sh). Ignored bytes are invisible to the worktree
# fingerprint (GX1_RULES.md rule 24), so a heavy route may bind its source only
# when every ignored path is a reviewed local runtime exclusion or a
# regenerable cache. The launch state owns the exclusion list; this pins it.
LAUNCH_STATE_NAME = "PROJECT_STATE_xau_direction_launch.json"
REVIEWED_LOCAL_RUNTIME_EXCLUSIONS_SCHEMA = "gx1_reviewed_local_runtime_exclusions_v1"
REVIEWED_LOCAL_RUNTIME_EXCLUSION_PATHS = frozenset({".claude/worktrees/", ".env", ".venv/"})
REVIEWED_REGENERABLE_CACHE_PATHS = frozenset({".pytest_cache/", ".ruff_cache/"})
# The rebuild chain, the trainer wrapper and the edge control gate on these lines.
SOURCE_IDENTITY_KEYS = (
    "head_commit", "worktree_fingerprint", "changed_path_count", "ignored_path_count",
    "prunable_worktree_count", "reviewed_ignored_path_count",
    "unexpected_ignored_path_count", "source_identity_gate",
)


def _git_bytes(repo: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=15,
    ).stdout


def _reviewed_local_runtime_exclusions(repo: Path, worktree_porcelain: str) -> frozenset[str]:
    state = json.loads(_regular_file(str(repo / LAUNCH_STATE_NAME), label="launch_state").read_text())
    reviewed = state.get("reviewed_local_runtime_exclusions")
    if (
        not isinstance(reviewed, dict)
        or set(reviewed) != {"schema_version", "paths"}
        or reviewed["schema_version"] != REVIEWED_LOCAL_RUNTIME_EXCLUSIONS_SCHEMA
        or not isinstance(reviewed["paths"], list)
        or not all(isinstance(path, str) for path in reviewed["paths"])
        or len(reviewed["paths"]) != len(REVIEWED_LOCAL_RUNTIME_EXCLUSION_PATHS)
        or set(reviewed["paths"]) != REVIEWED_LOCAL_RUNTIME_EXCLUSION_PATHS
    ):
        raise ValueError("reviewed local runtime exclusions are invalid")
    environment_file = repo / ".env"
    if environment_file.is_symlink() or (
        environment_file.exists()
        and (
            not environment_file.is_file()
            or os.stat(environment_file, follow_symlinks=False).st_mode & 0o077
        )
    ):
        raise ValueError("reviewed local .env exclusion is unsafe")
    venv = repo / ".venv"
    if venv.is_symlink() or (
        venv.exists()
        and (
            not venv.is_dir()
            or (venv / "pyvenv.cfg").is_symlink()
            or not (venv / "pyvenv.cfg").is_file()
        )
    ):
        raise ValueError("reviewed local virtual environment exclusion is invalid")
    worktree_root = repo / ".claude" / "worktrees"
    registered = {
        Path(line.removeprefix("worktree ")).resolve()
        for line in worktree_porcelain.splitlines()
        if line.startswith("worktree ")
    }
    if worktree_root.is_symlink() or (
        worktree_root.exists()
        and (
            not worktree_root.is_dir()
            or not any(
                str(path).startswith(str(worktree_root.resolve()) + os.sep)
                for path in registered
            )
        )
    ):
        raise ValueError("reviewed local Claude worktree exclusion is invalid")
    return frozenset(reviewed["paths"])


def source_identity(repo: Path) -> dict[str, object]:
    """Bind HEAD, the tracked diff and untracked bytes; classify ignored content."""
    head = _git_bytes(repo, "rev-parse", "--verify", "HEAD")
    worktree = hashlib.sha256()
    for label, payload in (
        (b"head", head),
        (b"tracked-diff", _git_bytes(repo, "diff", "--binary", "--no-ext-diff", "HEAD", "--")),
    ):
        worktree.update(len(label).to_bytes(4, "big"))
        worktree.update(label)
        worktree.update(len(payload).to_bytes(8, "big"))
        worktree.update(payload)
    untracked = _git_bytes(repo, "ls-files", "--others", "--exclude-standard", "-z")
    for raw in filter(None, untracked.split(b"\0")):
        path = repo / os.fsdecode(raw)
        if path.is_symlink():
            kind, payload = b"symlink", os.readlink(path).encode("utf-8", errors="surrogateescape")
        elif path.is_file():
            kind, payload = b"file", path.read_bytes()
        else:
            raise ValueError(f"unsupported untracked entry: {path}")
        for value in (raw, kind, payload):
            worktree.update(len(value).to_bytes(8, "big"))
            worktree.update(value)
    status = _git_bytes(repo, "status", "--porcelain=v1", "-z")
    changed = len(tuple(filter(None, status.split(b"\0"))))
    ignored_status = _git_bytes(repo, "status", "--ignored", "--porcelain=v1", "-z")
    ignored_paths = tuple(
        os.fsdecode(entry[3:])
        for entry in filter(None, ignored_status.split(b"\0"))
        if entry.startswith(b"!! ")
    )
    worktree_porcelain = _git_bytes(repo, "worktree", "list", "--porcelain").decode("utf-8")
    prunable = sum(1 for line in worktree_porcelain.splitlines() if line.startswith("prunable"))
    exclusions = _reviewed_local_runtime_exclusions(repo, worktree_porcelain)

    def reviewed(path: str) -> bool:
        return (
            path in exclusions
            or path in REVIEWED_REGENERABLE_CACHE_PATHS
            or path.endswith("/__pycache__/")
        )

    unexpected = sorted(path for path in ignored_paths if not reviewed(path))
    if prunable:
        gate = "BLOCK_PRUNABLE_WORKTREE_REGISTRATION"
    elif changed:
        gate = "BLOCK_DIRTY_WORKTREE"
    elif unexpected:
        gate = "BLOCK_UNEXPECTED_IGNORED_CONTENT"
    else:
        gate = "READY_CLEAN_WORKTREE__REVIEWED_LOCAL_EXCLUSIONS"
    return {
        "head_commit": head.decode("utf-8").strip(),
        "worktree_fingerprint": worktree.hexdigest(),
        "changed_path_count": changed,
        "ignored_path_count": len(ignored_paths),
        "prunable_worktree_count": prunable,
        "reviewed_ignored_path_count": len(ignored_paths) - len(unexpected),
        "unexpected_ignored_path_count": len(unexpected),
        "source_identity_gate": gate,
        "unexpected_ignored_paths": unexpected,
    }


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


def _current_processes(source: Path) -> list[dict[str, str]]:
    """Observe CURRENT Python workloads, including external benchmark operators."""
    result = subprocess.run(
        ["ps", "-eo", "pid,ppid,etime,pcpu,rss,args"], check=True,
        capture_output=True, text=True, timeout=15,
    )
    found = []
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[0].isdigit() or int(fields[0]) == os.getpid():
            continue
        command = fields[5]
        program = command.split(maxsplit=1)[0]
        if program not in (str(source / ".venv/bin/python"),
                           ".venv/bin/python", "./.venv/bin/python"):
            continue
        # A process may exit between the ps snapshot and the cwd read.
        try:
            cwd = Path(os.readlink(f"/proc/{fields[0]}/cwd"))
        except FileNotFoundError:
            continue
        if cwd != source:
            continue
        found.append(dict(zip(("pid", "ppid", "elapsed", "cpu_percent", "rss_kib", "command"), fields)))
    return found


def current_status(repo: Path, *, source_only: bool = False) -> dict[str, object]:
    """Read the sole current policy; never choose a historical run/checkpoint."""
    repo = _repo(str(repo))
    policy_path = _regular_file(str(repo / "NEXT_RUN_POLICY.json"), label="current_policy")
    policy = json.loads(policy_path.read_text())
    if (policy.get("schema_version") != "gx1_next_native_run_policy_v1"
            or policy.get("canonical_source_repo") != str(repo)
            or policy.get("canonical_branch") != "work/gx1-current"
            or _git(repo, "branch", "--show-current") != policy["canonical_branch"]):
        raise ValueError("CURRENT_POLICY_IDENTITY_INVALID")
    work = policy.get("current_work")
    if not isinstance(work, dict):
        raise ValueError("CURRENT_WORK_REQUIRED_NO_HISTORICAL_FALLBACK")
    out = {
        "decision": "OBSERVATION_ONLY_NOT_RUN_AUTHORITY",
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "source_repo": str(repo), "source_commit": _git(repo, "rev-parse", "HEAD"),
        "recorded_status": policy["status"],
        "recorded_observed_utc": policy["observed_utc"],
        "next_action": policy["next_action"],
        "training_enabled": policy["training_enabled"],
        "full_epoch_training_allowed": policy["full_epoch_training_allowed"],
        "full_val_allowed": policy["full_val_allowed"],
        "learning_gate": policy["learning_gate"],
        "test_accessed": False, "state_payload_rehashed": False,
        "current_work": work,
    }
    if source_only:
        return out
    binding = work.get("latest_terminal")
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("CURRENT_TERMINAL_BINDING_REQUIRED")
    terminal_path = _regular_file(binding["path"], label="current_terminal")
    if _sha256(terminal_path) != binding["sha256"]:
        raise ValueError("CURRENT_TERMINAL_HASH_MISMATCH")
    terminal = json.loads(terminal_path.read_text())
    if (type(terminal.get("exit_code")) is not int
            or terminal["exit_code"] != work["terminal_exit_code"]
            or terminal.get("source_unchanged") is not work["source_unchanged_at_terminal"]
            or terminal.get("test_data_used") is not False):
        raise ValueError("CURRENT_TERMINAL_STATE_MISMATCH")
    out["latest_terminal_receipt"] = terminal
    out["current_processes"] = _current_processes(repo)
    out["process_observation"] = "RUNNING" if out["current_processes"] else "NO_CURRENT_PYTHON_WORKLOAD_OBSERVED"
    out["next_run"] = next_run_readiness(repo)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    status = current_status(repo, source_only=args.source_only)
    identity = source_identity(repo)
    print(json.dumps(status, indent=2))
    for key in SOURCE_IDENTITY_KEYS:
        print(f"{key}: {identity[key]}")
    if identity["unexpected_ignored_paths"]:
        print("unexpected_ignored_paths: " + json.dumps(identity["unexpected_ignored_paths"]))
    if args.source_only and str(identity["source_identity_gate"]).startswith("BLOCK"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
