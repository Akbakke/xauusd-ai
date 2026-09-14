#!/usr/bin/env python3
"""Create one immutable GX1 handover bundle from explicit read-only evidence.

The collector executes only a fixed set of read-only git commands. Host/task,
boot, probe, guard, process and GPU state must already exist as snapshot files.
The native-binding mode also reads process and small runtime metadata.
Neither mode can start, stop, resume, install, reboot or load a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


SCHEMA = "gx1_handover_bundle_v1"
REQUIRED_ROLES = (
    "data_authority",
    "model_authority",
    "checkpoint_authority",
    "campaign_plan",
    "task_status",
    "boot_status",
    "probe_status",
    "guard_status",
    "process_status",
    "gpu_safety_status",
)
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


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


def _parse_evidence(values: list[str]) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        role, separator, path = value.partition("=")
        if not separator or role not in REQUIRED_ROLES:
            raise ValueError(f"invalid evidence binding: {value}")
        if role in parsed:
            raise ValueError(f"duplicate evidence role: {role}")
        parsed[role] = _regular_file(path, label=role)
    missing = sorted(set(REQUIRED_ROLES) - set(parsed))
    if missing:
        raise ValueError(f"missing evidence roles: {', '.join(missing)}")
    return parsed


def _json_summary(path: Path) -> dict[str, object] | None:
    if path.suffix.lower() != ".json":
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence must contain one object: {path}")
    return {
        key: value[key]
        for key in ("schema_version", "decision", "status", "boot_id", "observed_utc")
        if key in value
    }


def collect(
    *,
    repo: Path,
    output: Path,
    evidence: dict[str, Path],
    expected_branch: str,
    expected_commit: str,
) -> Path:
    if not output.is_absolute():
        raise ValueError("output must be an absolute path")
    if any(part.upper() == "TEST" for part in output.parts):
        raise ValueError("output must not access a TEST path")
    if output.exists() or output.is_symlink():
        raise ValueError("output must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)

    top = Path(_git(repo, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top != repo:
        raise ValueError(f"repo is not its git top level: {repo}")
    actual_branch = _git(repo, "branch", "--show-current")
    actual_commit = _git(repo, "rev-parse", "HEAD")
    if not expected_branch or actual_branch != expected_branch:
        raise ValueError(
            f"branch mismatch: expected {expected_branch!r}, observed {actual_branch!r}"
        )
    if actual_commit != expected_commit:
        raise ValueError(
            f"commit mismatch: expected {expected_commit!r}, observed {actual_commit!r}"
        )
    git_state = {
        "repo": str(repo),
        "branch": actual_branch,
        "commit": actual_commit,
        "expected_branch": expected_branch,
        "expected_commit": expected_commit,
        "status_porcelain_v1": _git(repo, "status", "--porcelain=v1", "--untracked-files=all"),
    }
    git_state["clean"] = git_state["status_porcelain_v1"] == ""

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=str(output.parent))
    )
    try:
        evidence_dir = temporary / "evidence"
        evidence_dir.mkdir(mode=0o700)
        bindings: dict[str, dict[str, object]] = {}
        for role in REQUIRED_ROLES:
            source = evidence[role]
            suffix = source.suffix.lower() if source.suffix else ".bin"
            copied = evidence_dir / f"{role}{suffix}"
            shutil.copyfile(source, copied)
            os.chmod(copied, 0o600)
            source_sha = _sha256(source)
            if _sha256(copied) != source_sha:
                raise RuntimeError(f"copy hash mismatch for {role}")
            bindings[role] = {
                "source_path": str(source),
                "bundle_path": str(copied.relative_to(temporary)),
                "sha256": source_sha,
                "size_bytes": source.stat().st_size,
                "summary": _json_summary(copied),
            }

        payload: dict[str, object] = {
            "schema_version": SCHEMA,
            "decision": "CAPTURED_NOT_RUN_AUTHORITY",
            "observed_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "collection_mode": "READ_ONLY_EXPLICIT_SNAPSHOTS",
            "test_accessed": False,
            "commands_executed": [
                "git rev-parse --show-toplevel",
                "git branch --show-current",
                "git rev-parse HEAD",
                "git status --porcelain=v1 --untracked-files=all",
            ],
            "git": git_state,
            "evidence": bindings,
        }
        payload["bundle_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
        manifest = temporary / "MANIFEST.json"
        manifest.write_bytes(_canonical_bytes(payload) + b"\n")
        os.chmod(manifest, 0o600)

        for directory in (evidence_dir, temporary):
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.replace(temporary, output)
        return output / "MANIFEST.json"
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


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
    out["capacity_preparation"] = binding.get("capacity_preparation")
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
    if "--native-binding" in sys.argv[1:]:
        parser = argparse.ArgumentParser(description="Read-only native GX1 campaign observation")
        parser.add_argument("--native-binding", required=True)
        parser.add_argument("--source-only", action="store_true")
        args = parser.parse_args()
        print(json.dumps(native_status(Path(args.native_binding), source_only=args.source_only), indent=2))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--evidence",
        action="append",
        default=[],
        metavar="ROLE=/ABSOLUTE/PATH",
        help="repeat once for each required role: " + ", ".join(REQUIRED_ROLES),
    )
    args = parser.parse_args()
    manifest = collect(
        repo=_repo(args.repo),
        output=Path(args.output),
        evidence=_parse_evidence(args.evidence),
        expected_branch=args.expected_branch,
        expected_commit=args.expected_commit,
    )
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
