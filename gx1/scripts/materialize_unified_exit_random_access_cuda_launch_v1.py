"""Materialize the immutable launch-eligible random-access CUDA smoke manifest."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from gx1.contracts.unified_exit_selected_sampler_v1 import file_sha256
from gx1.scripts.run_unified_exit_random_access_fixed_step_v1 import (
    build_launch_manifest,
    require_launch_manifest,
)

RECIPE_SCHEMA = "gx1_unified_exit_random_access_cuda_launch_recipe_v1"


def _read(path: Path) -> dict[str, Any]:
    if (
        not path.is_absolute()
        or path.resolve() != path
        or not path.is_file()
        or path.is_symlink()
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_LAUNCH_RECIPE_PATH_INVALID")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("UNIFIED_EXIT_CUDA_LAUNCH_RECIPE_INVALID")
    return value


def materialize(
    *, recipe_path: Path, output_path: Path, publish: bool
) -> dict[str, Any]:
    recipe = _read(recipe_path)
    if (
        set(recipe)
        != {
            "schema_version",
            "source_repo",
            "source_commit",
            "files",
            "checkpoint_dir",
            "seed",
            "test_data_used",
        }
        or recipe.get("schema_version") != RECIPE_SCHEMA
        or recipe.get("test_data_used") is not False
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_LAUNCH_RECIPE_INVALID")
    source_repo = Path(str(recipe["source_repo"]))
    checkpoint_dir = Path(str(recipe["checkpoint_dir"]))
    if any(
        not path.is_absolute() or path.resolve() != path
        for path in (source_repo, checkpoint_dir, output_path)
    ):
        raise RuntimeError("UNIFIED_EXIT_CUDA_LAUNCH_OUTPUT_PATH_INVALID")
    head = subprocess.run(
        ["git", "-C", str(source_repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if head != recipe["source_commit"]:
        raise RuntimeError("UNIFIED_EXIT_CUDA_LAUNCH_SOURCE_COMMIT_INVALID")
    files = recipe.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("UNIFIED_EXIT_CUDA_LAUNCH_FILES_INVALID")
    for binding in files.values():
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
            raise RuntimeError("UNIFIED_EXIT_CUDA_LAUNCH_FILE_BINDING_INVALID")
        path = Path(str(binding["path"]))
        if (
            not path.is_file()
            or path.is_symlink()
            or file_sha256(path) != binding["sha256"]
        ):
            raise RuntimeError("UNIFIED_EXIT_CUDA_LAUNCH_FILE_BINDING_INVALID")
    value = build_launch_manifest(
        source_repo=source_repo,
        source_commit=head,
        files=files,
        launch_manifest_path=output_path,
        checkpoint_dir=checkpoint_dir,
        seed=int(recipe["seed"]),
    )
    require_launch_manifest(value)
    if publish:
        if output_path.exists():
            raise RuntimeError("UNIFIED_EXIT_CUDA_LAUNCH_OUTPUT_EXISTS")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{output_path.name}.", dir=output_path.parent
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, output_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    return {
        "decision": "PASS_GPU_SMOKE_MATRIX_ELIGIBLE",
        "published": publish,
        "output_path": str(output_path),
        "manifest_sha256": value["manifest_sha256"],
        "output_file_sha256": file_sha256(output_path) if publish else None,
        "test_data_used": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            materialize(
                recipe_path=args.recipe.resolve(),
                output_path=args.output.resolve(),
                publish=args.publish,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
