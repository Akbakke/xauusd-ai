#!/usr/bin/env python3
"""Materialize an immutable capacity benchmark from one committed host smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from gx1.contracts.cloud_training_capacity_gate_v1 import (
    CloudTrainingCapacityGateError,
    build_hopper_benchmark_from_smoke_bundle,
)


def _canonical_output_path(value: str) -> Path:
    output = Path(value).expanduser()
    if not output.is_absolute() or output.suffix != ".json":
        raise CloudTrainingCapacityGateError("output must be an absolute JSON path")
    parent = output.parent
    if (
        not parent.is_dir()
        or parent.is_symlink()
        or parent.resolve(strict=True) != parent
        or any(ancestor.is_symlink() for ancestor in parent.parents)
    ):
        raise CloudTrainingCapacityGateError("output parent is not canonical")
    if os.path.lexists(output):
        raise CloudTrainingCapacityGateError("output already exists or is a symlink")
    return output


def _canonical_json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_immutable(output: Path, encoded: bytes) -> str:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, output, follow_symlinks=False)
        except FileExistsError as exc:
            raise CloudTrainingCapacityGateError(
                "output already exists or is a symlink"
            ) from exc
        directory_fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def run(args: argparse.Namespace) -> tuple[Path, str, dict[str, Any]]:
    payload = build_hopper_benchmark_from_smoke_bundle(Path(args.smoke_bundle))
    output = _canonical_output_path(str(args.output))
    digest = _write_immutable(output, _canonical_json_bytes(payload))
    return output, digest, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-bundle", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    output, digest, _ = run(build_parser().parse_args(argv))
    print(f"{digest}  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
