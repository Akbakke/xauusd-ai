#!/usr/bin/env python3
"""Publish one exact immutable model-native seq513 train-recipe audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from gx1.contracts.entry_model_native_train_launch_v1 import (
    build_parser as build_launch_parser,
    build_recipe_audit_payload,
)
from gx1.contracts.immutable_event_authority_v1 import (
    next_immutable_event_created_utc,
    write_immutable_json_event,
)


EVENT_PREFIX = "ENTRY_MODEL_NATIVE_SEQ513_TRAIN_RECIPE_AUDIT"
_STAMP_RE = re.compile(r"(?:^|[^0-9])20[0-9]{6}T[0-9]{6}(?:[0-9]{6})?Z(?:[^0-9]|$)")
_MUTABLE_POINTER_RE = re.compile(
    r"(?:^|[/_.-])latest(?:[/_.-]|$)",
    re.IGNORECASE,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_worktree(repo: Path) -> None:
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    ).splitlines()
    if status:
        raise RuntimeError(
            "TRAIN_RECIPE_SOURCE_WORKTREE_DIRTY: " + "; ".join(status[:20])
        )


def _output_dir(raw: str, *, dataset_dir: Path, out_bundle_dir: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise RuntimeError("train recipe out_dir must be absolute")
    if _MUTABLE_POINTER_RE.search(str(path)):
        raise RuntimeError("train recipe out_dir uses a mutable pointer")
    if not _STAMP_RE.search(str(path)):
        raise RuntimeError("train recipe out_dir must contain an immutable UTC timestamp")
    resolved = path.resolve()
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise RuntimeError("train recipe out_dir must be an exact directory")
    for protected in (dataset_dir, out_bundle_dir):
        if resolved == protected or resolved in protected.parents or protected in resolved.parents:
            raise RuntimeError(
                f"train recipe out_dir overlaps protected path: {resolved} vs {protected}"
            )
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = build_launch_parser(require_recipe_audit=False)
    parser.description = __doc__
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser


def run(args: argparse.Namespace) -> tuple[Path, dict]:
    repo = Path(args.repo).expanduser().resolve(strict=True)
    dataset_dir = Path(args.dataset_dir).expanduser().resolve(strict=True)
    out_bundle_dir = Path(args.out_bundle_dir).expanduser().resolve()
    out_dir = _output_dir(
        args.out_dir,
        dataset_dir=dataset_dir,
        out_bundle_dir=out_bundle_dir,
    )
    _clean_worktree(repo)
    created = next_immutable_event_created_utc(out_dir, EVENT_PREFIX)
    payload = build_recipe_audit_payload(args, created_utc=created.isoformat())
    _clean_worktree(repo)
    path, event = write_immutable_json_event(out_dir, EVENT_PREFIX, payload)
    if not args.quiet:
        print(
            json.dumps(
                {
                    "decision": event["decision"],
                    "json_path": str(path),
                    "sha256": _sha256_file(path),
                    "trainer_env_count": len(event["trainer_env"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return path, event


def main(argv: Sequence[str] | None = None) -> int:
    try:
        run(build_parser().parse_args(argv))
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
        print(f"FATAL: model-native train recipe rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
