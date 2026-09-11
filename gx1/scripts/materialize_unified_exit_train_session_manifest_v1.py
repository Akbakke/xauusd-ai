"""Validate or atomically publish a post-GPU-selection TRAIN session manifest."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from gx1.contracts.unified_exit_fixed_step_resume_equivalence_v1 import require_equivalence
from gx1.contracts.unified_exit_gpu_batch_selection_v1 import file_sha256, require_selection
from gx1.contracts.unified_exit_train_session_manifest_v1 import (
    build_train_session_manifest,
    require_train_session_manifest,
)


def _load(path: Path) -> dict:
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_TRAIN_SESSION_INPUT_INVALID")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("UNIFIED_EXIT_TRAIN_SESSION_INPUT_INVALID")
    return value


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("resume_proof", "epoch1"), required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--prelaunch", type=Path, required=True)
    parser.add_argument("--gpu-selection", type=Path, required=True)
    parser.add_argument("--resume-equivalence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    prelaunch = args.prelaunch.resolve()
    selection_path = args.gpu_selection.resolve()
    prelaunch_value = _load(prelaunch)
    selection = require_selection(_load(selection_path), verify_files=True)
    eq_path = args.resume_equivalence.resolve() if args.resume_equivalence else None
    eq = require_equivalence(_load(eq_path), verify_files=True) if eq_path else None
    value = require_train_session_manifest(
        build_train_session_manifest(
            phase=args.phase,
            source_commit=args.source_commit,
            prelaunch_binding=_binding(prelaunch),
            prelaunch_manifest_sha256=str(prelaunch_value.get("manifest_sha256", "")),
            gpu_selection_binding=_binding(selection_path),
            gpu_selection=selection,
            resume_equivalence_binding=_binding(eq_path) if eq_path else None,
            resume_equivalence=eq,
        ),
        expected_phase=args.phase,
        verify_files=True,
    )
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise RuntimeError("UNIFIED_EXIT_TRAIN_SESSION_OUTPUT_INVALID")
    if args.publish:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
        fd, tmp = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, output)
            directory_fd = os.open(output.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    print(json.dumps({
        "decision": "PASS", "published": args.publish, "output": str(output),
        "manifest_sha256": value["manifest_sha256"],
        "file_sha256": file_sha256(output) if args.publish else None,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
