#!/usr/bin/env python3
"""Create one immutable GX1 handover bundle from explicit read-only evidence.

The collector executes only a fixed set of read-only git commands. Host/task,
boot, probe, guard, process and GPU state must already exist as snapshot files.
It cannot start, stop, resume, install, reboot or query training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
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


def main() -> int:
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
