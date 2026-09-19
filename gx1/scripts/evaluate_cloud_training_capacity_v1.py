#!/usr/bin/env python3
"""Write one immutable report-only capacity gate from a measured benchmark."""

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
    evaluate_cloud_training_capacity,
    require_hopper_benchmark,
)


def _canonical_output_path(value: str) -> Path:
    output = Path(value).expanduser()
    if not output.is_absolute():
        raise CloudTrainingCapacityGateError("output path must be absolute")
    if output.suffix != ".json":
        raise CloudTrainingCapacityGateError("output path must end in .json")
    parent = output.parent
    if not parent.is_dir() or parent.is_symlink():
        raise CloudTrainingCapacityGateError("output parent is not a regular directory")
    if any(ancestor.is_symlink() for ancestor in parent.parents):
        raise CloudTrainingCapacityGateError("output parent path contains a symlink")
    if parent.resolve(strict=True) != parent:
        raise CloudTrainingCapacityGateError("output parent path is not canonical")
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


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_immutable(output: Path, encoded: bytes) -> str:
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
        _fsync_directory(output.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return hashlib.sha256(encoded).hexdigest()


def run(args: argparse.Namespace) -> tuple[Path, str, dict[str, Any]]:
    benchmark_path = Path(args.benchmark).expanduser()
    benchmark, binding = require_hopper_benchmark(
        benchmark_path, str(args.benchmark_sha256)
    )
    output = _canonical_output_path(str(args.output))
    if output == Path(binding["path"]):
        raise CloudTrainingCapacityGateError("output cannot replace the benchmark")
    payload = evaluate_cloud_training_capacity(benchmark, benchmark_binding=binding)
    digest = _atomic_write_immutable(output, _canonical_json_bytes(payload))
    return output, digest, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--benchmark-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    output, digest, _ = run(build_parser().parse_args(argv))
    print(f"{digest}  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
